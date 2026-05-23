from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nordic_sentiment.config import ProjectConfigs
from nordic_sentiment.fundamentals.snapshots import build_fundamental_filings, build_monthly_fundamental_snapshots
from nordic_sentiment.identifiers.builders import (
    attach_company_security_ids,
    build_company_dimension,
    build_security_dimension,
)
from nordic_sentiment.io.compustat_sweden import (
    load_sweden_bond_yield_10y,
    load_sweden_daily_prices,
    load_sweden_economic_consumer_sentiment,
    load_sweden_interbank_rate_3m,
    load_sweden_ipo_offers,
    load_sweden_macro_control_cpi,
    load_sweden_macro_control_em,
    load_sweden_macro_control_ip,
    load_sweden_macro_control_ppi,
    load_sweden_monthly_prices,
    load_sweden_quarterly_fundamentals,
)
from nordic_sentiment.market.calendar import build_calendar_month
from nordic_sentiment.market.characteristics import (
    attach_ivol_to_characteristics,
    attach_liquidity_dividend_to_characteristics,
    build_sweden_characteristics_layers,
)
from nordic_sentiment.market.daily_factors import build_sweden_factor_daily, build_sweden_ivol_monthly
from nordic_sentiment.market.controls import build_sweden_factor_tables
from nordic_sentiment.market.liquidity import build_sweden_liquidity_dividend_monthly
from nordic_sentiment.market.monthly import build_monthly_market_panels
from nordic_sentiment.market.returns import build_daily_return_sanity_audit, clean_sweden_monthly_returns
from nordic_sentiment.market.universe import filter_primary_sweden_equity_universe
from nordic_sentiment.sentiment.sweden import (
    apply_ipo_return_availability_to_paper_spec,
    build_sweden_ipo_return_table,
    build_sweden_sentiment_proxy_mart,
)
from nordic_sentiment.staging.sweden import (
    stage_sweden_daily_prices,
    stage_sweden_ipo_offers,
    stage_sweden_monthly_prices,
    stage_sweden_quarterly_fundamentals,
)
from nordic_sentiment.staging.sweden_rates import stage_sweden_rates


@dataclass
class SwedenRawData:
    daily_prices: pd.DataFrame
    monthly_prices: pd.DataFrame
    quarterly_fundamentals: pd.DataFrame
    economic_consumer_sentiment: pd.DataFrame
    interbank_rate_3m: pd.DataFrame
    bond_yield_10y: pd.DataFrame
    macro_control_cpi: pd.DataFrame
    macro_control_ppi: pd.DataFrame
    macro_control_ip: pd.DataFrame
    macro_control_em: pd.DataFrame
    ipo_offers: pd.DataFrame | None = None


@dataclass
class SwedenStagedData:
    daily_prices: pd.DataFrame
    monthly_prices: pd.DataFrame
    quarterly_fundamentals: pd.DataFrame
    rates_monthly: pd.DataFrame
    ipo_offers: pd.DataFrame | None = None


@dataclass
class SwedenArtifacts:
    dim_company: pd.DataFrame
    dim_security: pd.DataFrame
    dim_calendar_month: pd.DataFrame
    fct_price_daily: pd.DataFrame
    fct_factor_daily: pd.DataFrame
    fct_market_monthly_source: pd.DataFrame
    fct_market_monthly: pd.DataFrame
    audit_equity_universe_monthly: pd.DataFrame
    audit_return_sanity_monthly: pd.DataFrame
    audit_daily_return_sanity_monthly: pd.DataFrame
    audit_fundamental_staleness_monthly: pd.DataFrame
    audit_value_sort_monthly: pd.DataFrame
    fct_fundamental_filing: pd.DataFrame
    mart_fundamental_snapshot_monthly: pd.DataFrame
    mart_characteristics_monthly: pd.DataFrame
    mart_characteristics_annual_comparison: pd.DataFrame
    fct_rate_monthly: pd.DataFrame
    fct_factor_monthly: pd.DataFrame
    fct_ivol_monthly: pd.DataFrame
    fct_liquidity_dividend_monthly: pd.DataFrame
    audit_factor_legs_monthly: pd.DataFrame
    fct_macro_controls_monthly: pd.DataFrame
    fct_ipo_return: pd.DataFrame
    audit_sweden_sentiment_paper_spec: pd.DataFrame
    fct_sentiment_proxy_monthly_source: pd.DataFrame
    mart_sentiment_proxy_monthly: pd.DataFrame
    audit_sweden_variable_availability: pd.DataFrame


def load_sweden_raw_data(configs: ProjectConfigs) -> SwedenRawData:
    return SwedenRawData(
        daily_prices=load_sweden_daily_prices(configs),
        monthly_prices=load_sweden_monthly_prices(configs),
        quarterly_fundamentals=load_sweden_quarterly_fundamentals(configs),
        economic_consumer_sentiment=load_sweden_economic_consumer_sentiment(configs),
        interbank_rate_3m=load_sweden_interbank_rate_3m(configs),
        bond_yield_10y=load_sweden_bond_yield_10y(configs),
        macro_control_cpi=load_sweden_macro_control_cpi(configs),
        macro_control_ppi=load_sweden_macro_control_ppi(configs),
        macro_control_ip=load_sweden_macro_control_ip(configs),
        macro_control_em=load_sweden_macro_control_em(configs),
        ipo_offers=load_sweden_ipo_offers(configs),
    )


def stage_sweden_raw_data(raw: SwedenRawData) -> SwedenStagedData:
    daily = attach_company_security_ids(stage_sweden_daily_prices(raw.daily_prices))
    monthly = attach_company_security_ids(stage_sweden_monthly_prices(raw.monthly_prices))
    fundamentals = attach_company_security_ids(stage_sweden_quarterly_fundamentals(raw.quarterly_fundamentals))
    rates_monthly = stage_sweden_rates(raw.interbank_rate_3m, raw.bond_yield_10y)
    ipo_offers = stage_sweden_ipo_offers(raw.ipo_offers if raw.ipo_offers is not None else pd.DataFrame())
    return SwedenStagedData(
        daily_prices=daily,
        monthly_prices=monthly,
        quarterly_fundamentals=fundamentals,
        rates_monthly=rates_monthly,
        ipo_offers=ipo_offers,
    )


def _calendar_bounds(staged: SwedenStagedData) -> tuple[pd.Timestamp, pd.Timestamp]:
    candidates: list[pd.Series] = []
    for frame, column in [
        (staged.daily_prices, "trade_date"),
        (staged.monthly_prices, "month_end_date"),
        (staged.quarterly_fundamentals, "fiscal_period_end"),
    ]:
        if frame is not None and not frame.empty and column in frame.columns:
            candidates.append(pd.to_datetime(frame[column], errors="coerce").dropna())
    if not candidates:
        return pd.Timestamp("2000-01-31"), pd.Timestamp("2025-12-31")
    combined = pd.concat(candidates, ignore_index=True)
    return combined.min(), combined.max()


def _filter_daily_to_monthly_universe(daily_prices: pd.DataFrame, market_monthly: pd.DataFrame) -> pd.DataFrame:
    if daily_prices is None or daily_prices.empty or market_monthly is None or market_monthly.empty:
        return pd.DataFrame(daily_prices).copy()
    if "security_id" not in daily_prices.columns or "security_id" not in market_monthly.columns:
        return pd.DataFrame(daily_prices).copy()

    daily = daily_prices.copy()
    daily["trade_date"] = pd.to_datetime(daily.get("trade_date"), errors="coerce")
    daily["month_end_date"] = daily["trade_date"].dt.to_period("M").dt.to_timestamp("M")
    selected = market_monthly[["security_id", "month_end_date"]].drop_duplicates().copy()
    selected["month_end_date"] = pd.to_datetime(selected["month_end_date"], errors="coerce")
    filtered = daily.merge(selected, on=["security_id", "month_end_date"], how="inner")
    return filtered.drop(columns=["month_end_date"]).reset_index(drop=True)


def _build_fundamental_staleness_audit(
    market_monthly: pd.DataFrame,
    fundamental_snapshots: pd.DataFrame,
    *,
    country_code: str = "SWE",
) -> pd.DataFrame:
    columns = [
        "country_code",
        "month_end_date",
        "year_num",
        "market_rows",
        "market_companies",
        "fresh_fundamental_rows",
        "stale_capped_rows",
        "missing_snapshot_rows",
        "fresh_fundamental_share",
        "stale_capped_share",
        "missing_snapshot_share",
        "be_me_available_after_cap_count",
        "be_me_removed_by_stale_cap_count",
        "be_me_removed_share_of_pre_cap_available",
        "mean_fundamental_staleness_days",
        "p95_fundamental_staleness_days",
        "max_fundamental_staleness_days",
    ]
    if market_monthly is None or market_monthly.empty:
        return pd.DataFrame(columns=columns)

    market = market_monthly.copy()
    market["month_end_date"] = pd.to_datetime(market["month_end_date"], errors="coerce")
    market["month_end_market_cap"] = pd.to_numeric(market.get("month_end_market_cap"), errors="coerce")
    if "country_code" in market.columns:
        market["country_code"] = market["country_code"].fillna(country_code)
    else:
        market["country_code"] = country_code

    snapshots = pd.DataFrame(fundamental_snapshots).copy()
    if not snapshots.empty:
        snapshots["month_end_date"] = pd.to_datetime(snapshots["month_end_date"], errors="coerce")
        be_source = snapshots.get(
            "BE",
            snapshots.get("book_equity", pd.Series(pd.NA, index=snapshots.index)),
        )
        snapshots["BE"] = pd.to_numeric(be_source, errors="coerce")
        snapshot_cols = [
            column
            for column in [
                "company_id",
                "month_end_date",
                "BE",
                "fundamental_staleness_days",
                "stale_fundamental_flag",
                "stale_fundamental_had_book_equity_flag",
                "fundamental_snapshot_status",
            ]
            if column in snapshots.columns
        ]
        snapshots = snapshots[snapshot_cols].drop_duplicates(["company_id", "month_end_date"], keep="last")

    panel = market.merge(snapshots, on=["company_id", "month_end_date"], how="left") if not snapshots.empty else market.copy()
    status = panel.get("fundamental_snapshot_status", pd.Series(pd.NA, index=panel.index)).astype("string")
    stale_flag = (
        panel.get("stale_fundamental_flag", pd.Series(False, index=panel.index))
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    stale_had_be = (
        panel.get("stale_fundamental_had_book_equity_flag", pd.Series(False, index=panel.index))
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    market_cap_positive = pd.to_numeric(panel.get("month_end_market_cap"), errors="coerce").gt(0)
    panel["_fresh_fundamental_row"] = status.eq("fresh").fillna(False).astype(bool)
    panel["_stale_capped_row"] = stale_flag
    panel["_missing_snapshot_row"] = status.isna()
    panel["_be_me_available_after_cap"] = pd.to_numeric(panel.get("BE"), errors="coerce").gt(0) & market_cap_positive
    panel["_be_me_removed_by_stale_cap"] = stale_had_be & market_cap_positive
    panel["fundamental_staleness_days"] = pd.to_numeric(panel.get("fundamental_staleness_days"), errors="coerce")

    rows: list[dict[str, object]] = []
    for (country, month_end_date), group in panel.groupby(["country_code", "month_end_date"], dropna=False, sort=True):
        market_rows = len(group)
        be_me_after_cap_count = int(group["_be_me_available_after_cap"].sum())
        be_me_removed_count = int(group["_be_me_removed_by_stale_cap"].sum())
        pre_cap_available = be_me_after_cap_count + be_me_removed_count
        staleness = pd.to_numeric(group["fundamental_staleness_days"], errors="coerce").dropna()
        rows.append(
            {
                "country_code": country if pd.notna(country) else country_code,
                "month_end_date": month_end_date,
                "year_num": pd.Timestamp(month_end_date).year if pd.notna(month_end_date) else pd.NA,
                "market_rows": market_rows,
                "market_companies": int(group["company_id"].nunique()),
                "fresh_fundamental_rows": int(group["_fresh_fundamental_row"].sum()),
                "stale_capped_rows": int(group["_stale_capped_row"].sum()),
                "missing_snapshot_rows": int(group["_missing_snapshot_row"].sum()),
                "fresh_fundamental_share": float(group["_fresh_fundamental_row"].mean()) if market_rows else float("nan"),
                "stale_capped_share": float(group["_stale_capped_row"].mean()) if market_rows else float("nan"),
                "missing_snapshot_share": float(group["_missing_snapshot_row"].mean()) if market_rows else float("nan"),
                "be_me_available_after_cap_count": be_me_after_cap_count,
                "be_me_removed_by_stale_cap_count": be_me_removed_count,
                "be_me_removed_share_of_pre_cap_available": (
                    be_me_removed_count / pre_cap_available if pre_cap_available else float("nan")
                ),
                "mean_fundamental_staleness_days": float(staleness.mean()) if not staleness.empty else float("nan"),
                "p95_fundamental_staleness_days": float(staleness.quantile(0.95)) if not staleness.empty else float("nan"),
                "max_fundamental_staleness_days": float(staleness.max()) if not staleness.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_value_sort_audit(characteristics: pd.DataFrame, *, country_code: str = "SWE") -> pd.DataFrame:
    columns = [
        "country_code",
        "month_end_date",
        "market_rows",
        "be_available_rows",
        "nonpositive_book_equity_rows",
        "ceq_zero_replaced_rows",
        "be_me_available_rows",
        "be_me_missing_with_book_equity_rows",
        "nonpositive_book_equity_share",
        "ceq_zero_replaced_share",
    ]
    if characteristics is None or characteristics.empty:
        return pd.DataFrame(columns=columns)

    frame = characteristics.copy()
    frame["month_end_date"] = pd.to_datetime(frame.get("month_end_date"), errors="coerce")
    if "country_code" in frame.columns:
        frame["country_code"] = frame["country_code"].fillna(country_code)
    else:
        frame["country_code"] = country_code
    be = pd.to_numeric(frame.get("BE"), errors="coerce")
    frame["_be_available"] = be.notna()
    frame["_nonpositive_be"] = be.notna() & be.le(0)
    frame["_be_me_available"] = pd.to_numeric(frame.get("BE_ME"), errors="coerce").notna()
    frame["_be_me_missing_with_be"] = frame["_be_available"] & ~frame["_be_me_available"]
    frame["_ceq_zero_replaced"] = (
        frame.get("book_equity_zero_ceq_replaced_flag", pd.Series(False, index=frame.index))
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    rows: list[dict[str, object]] = []
    for (country, month_end_date), group in frame.groupby(["country_code", "month_end_date"], dropna=False, sort=True):
        market_rows = len(group)
        rows.append(
            {
                "country_code": country if pd.notna(country) else country_code,
                "month_end_date": month_end_date,
                "market_rows": market_rows,
                "be_available_rows": int(group["_be_available"].sum()),
                "nonpositive_book_equity_rows": int(group["_nonpositive_be"].sum()),
                "ceq_zero_replaced_rows": int(group["_ceq_zero_replaced"].sum()),
                "be_me_available_rows": int(group["_be_me_available"].sum()),
                "be_me_missing_with_book_equity_rows": int(group["_be_me_missing_with_be"].sum()),
                "nonpositive_book_equity_share": float(group["_nonpositive_be"].mean()) if market_rows else float("nan"),
                "ceq_zero_replaced_share": float(group["_ceq_zero_replaced"].mean()) if market_rows else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_sweden_artifacts(
    staged: SwedenStagedData,
    *,
    sentiment_paper_spec: pd.DataFrame,
    sentiment_proxy_source: pd.DataFrame,
    macro_controls_monthly: pd.DataFrame,
) -> SwedenArtifacts:
    for name, frame in [
        ("sentiment_paper_spec", sentiment_paper_spec),
        ("sentiment_proxy_source", sentiment_proxy_source),
        ("macro_controls_monthly", macro_controls_monthly),
    ]:
        if frame is None or frame.empty:
            raise RuntimeError(f"{name} is required for the final Sweden thesis pipeline.")

    start_date, end_date = _calendar_bounds(staged)
    dim_calendar_month = build_calendar_month(start_date, end_date)
    dim_company = build_company_dimension(
        staged.daily_prices,
        staged.monthly_prices,
        staged.quarterly_fundamentals,
        country_code="SWE",
    )
    dim_security = build_security_dimension(staged.daily_prices, staged.monthly_prices, country_code="SWE")
    market_source, market_monthly_unfiltered = build_monthly_market_panels(staged.daily_prices, staged.monthly_prices)
    market_monthly, equity_universe_audit = filter_primary_sweden_equity_universe(
        market_monthly_unfiltered,
        country_code="SWE",
    )
    market_monthly, return_sanity_audit = clean_sweden_monthly_returns(market_monthly)
    fundamental_filings = build_fundamental_filings(staged.quarterly_fundamentals, dim_company)
    fundamental_snapshots = build_monthly_fundamental_snapshots(fundamental_filings, dim_calendar_month)
    fundamental_staleness_audit = _build_fundamental_staleness_audit(market_monthly, fundamental_snapshots)
    characteristics, annual_comparison, availability_audit = build_sweden_characteristics_layers(
        market_monthly,
        fundamental_snapshots,
    )
    factor_monthly, factor_legs = build_sweden_factor_tables(
        market_monthly,
        fundamental_snapshots,
        staged.rates_monthly,
        dim_security,
    )
    factor_daily_input = _filter_daily_to_monthly_universe(staged.daily_prices, market_monthly)
    factor_daily, daily_panel = build_sweden_factor_daily(
        factor_daily_input,
        staged.rates_monthly,
        market_monthly,
        fundamental_snapshots,
    )
    daily_return_sanity_audit = build_daily_return_sanity_audit(daily_panel)
    liquidity_dividend_monthly = build_sweden_liquidity_dividend_monthly(daily_panel)
    ivol_monthly = build_sweden_ivol_monthly(daily_panel, factor_daily)
    characteristics, annual_comparison = attach_liquidity_dividend_to_characteristics(
        characteristics,
        liquidity_dividend_monthly,
    )
    characteristics, annual_comparison = attach_ivol_to_characteristics(characteristics, ivol_monthly)
    value_sort_audit = _build_value_sort_audit(characteristics)

    ipo_returns = build_sweden_ipo_return_table(
        staged.ipo_offers if staged.ipo_offers is not None else pd.DataFrame(),
        staged.daily_prices,
    )
    paper_spec = sentiment_paper_spec.copy()
    paper_spec = apply_ipo_return_availability_to_paper_spec(paper_spec, ipo_returns)
    proxy_source = sentiment_proxy_source.copy()
    proxy_mart = build_sweden_sentiment_proxy_mart(proxy_source)
    if proxy_mart.empty:
        raise RuntimeError("Final Sweden sentiment proxy mart is empty after cleaning.")
    macro_controls = macro_controls_monthly.copy()

    return SwedenArtifacts(
        dim_company=dim_company,
        dim_security=dim_security,
        dim_calendar_month=dim_calendar_month,
        fct_price_daily=staged.daily_prices,
        fct_factor_daily=factor_daily,
        fct_market_monthly_source=market_source,
        fct_market_monthly=market_monthly,
        audit_equity_universe_monthly=equity_universe_audit,
        audit_return_sanity_monthly=return_sanity_audit,
        audit_daily_return_sanity_monthly=daily_return_sanity_audit,
        audit_fundamental_staleness_monthly=fundamental_staleness_audit,
        audit_value_sort_monthly=value_sort_audit,
        fct_fundamental_filing=fundamental_filings,
        mart_fundamental_snapshot_monthly=fundamental_snapshots,
        mart_characteristics_monthly=characteristics,
        mart_characteristics_annual_comparison=annual_comparison,
        fct_rate_monthly=staged.rates_monthly,
        fct_factor_monthly=factor_monthly,
        fct_ivol_monthly=ivol_monthly,
        fct_liquidity_dividend_monthly=liquidity_dividend_monthly,
        audit_factor_legs_monthly=factor_legs,
        fct_macro_controls_monthly=macro_controls,
        fct_ipo_return=ipo_returns,
        audit_sweden_sentiment_paper_spec=paper_spec,
        fct_sentiment_proxy_monthly_source=proxy_source,
        mart_sentiment_proxy_monthly=proxy_mart,
        audit_sweden_variable_availability=availability_audit,
    )
