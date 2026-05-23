from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_CAP_FULL_TO_MILLIONS = 1_000_000.0


RATE_COLUMNS = [
    "country_code",
    "rate_code",
    "month_end_date",
    "raw_annual_rate_pct",
    "monthly_rate_proxy",
    "source_name",
]

FACTOR_COLUMNS = ["country_code", "factor_code", "month_end_date", "factor_value"]

AUDIT_COLUMNS = [
    "country_code",
    "factor_code",
    "month_end_date",
    "leg_code",
    "breakpoint_low",
    "breakpoint_high",
    "n_constituents",
    "lagged_weight_sum",
    "ew_return",
    "vw_return",
    "thin_leg_flag",
]


def _empty_rates() -> pd.DataFrame:
    return pd.DataFrame(columns=RATE_COLUMNS)


def _empty_factors() -> pd.DataFrame:
    return pd.DataFrame(columns=FACTOR_COLUMNS)


def _empty_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=AUDIT_COLUMNS)


def _normalize_rate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(column).strip().lower() for column in out.columns]
    return out


def stage_sweden_rate_series(
    frame: pd.DataFrame,
    *,
    rate_code: str,
    value_column: str,
    source_name: str,
    country_code: str = "SWE",
    derive_monthly_proxy: bool,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_rates()

    out = _normalize_rate_columns(frame)
    if "observation_date" not in out.columns:
        return _empty_rates()

    value_key = value_column.lower()
    if value_key not in out.columns:
        candidate_columns = [column for column in out.columns if column != "observation_date"]
        if not candidate_columns:
            return _empty_rates()
        value_key = candidate_columns[0]

    out["month_end_date"] = pd.to_datetime(out["observation_date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    out["raw_annual_rate_pct"] = pd.to_numeric(out[value_key], errors="coerce")
    out["monthly_rate_proxy"] = (
        out["raw_annual_rate_pct"] / 100.0 / 12.0 if derive_monthly_proxy else np.nan
    )
    out["country_code"] = country_code
    out["rate_code"] = rate_code
    out["source_name"] = source_name
    out = out.dropna(subset=["month_end_date"]).sort_values("month_end_date")
    out = out.drop_duplicates(["country_code", "rate_code", "month_end_date"], keep="last")
    return out[RATE_COLUMNS].reset_index(drop=True)


def build_sweden_rates_monthly(
    interbank_rate_3m: pd.DataFrame,
    bond_yield_10y: pd.DataFrame,
    *,
    country_code: str = "SWE",
) -> pd.DataFrame:
    staged = [
        stage_sweden_rate_series(
            interbank_rate_3m,
            rate_code="RF_3M_PROXY",
            value_column="IR3TIB01SEM156N",
            source_name="3-month_interbank_rate_sweden.csv",
            country_code=country_code,
            derive_monthly_proxy=True,
        ),
        stage_sweden_rate_series(
            bond_yield_10y,
            rate_code="YIELD_10Y",
            value_column="IRLTLT01SEM156N",
            source_name="10-year_bond_yield_sweden.csv",
            country_code=country_code,
            derive_monthly_proxy=False,
        ),
    ]
    out = pd.concat([frame for frame in staged if not frame.empty], ignore_index=True) if any(
        not frame.empty for frame in staged
    ) else _empty_rates()
    if out.empty:
        return out
    return out.sort_values(["country_code", "rate_code", "month_end_date"]).reset_index(drop=True)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numer = pd.to_numeric(numerator, errors="coerce")
    denom = pd.to_numeric(denominator, errors="coerce")
    return pd.Series(np.where(denom.notna() & denom.ne(0), numer / denom, np.nan), index=numer.index, dtype="float64")


def _common_equity_mask(frame: pd.DataFrame) -> pd.Series:
    if "issue_type_code" not in frame.columns:
        return pd.Series(True, index=frame.index)
    issue_type = frame["issue_type_code"].astype("string").str.strip().str.upper()
    return issue_type.isna() | issue_type.eq("0") | issue_type.eq("EQ") | issue_type.eq("COM")


def _weighted_return(frame: pd.DataFrame, return_column: str, weight_column: str) -> float:
    returns = pd.to_numeric(frame.get(return_column), errors="coerce")
    weights = pd.to_numeric(frame.get(weight_column), errors="coerce")
    valid = returns.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(returns.loc[valid], weights=weights.loc[valid]))


def _equal_weight_return(frame: pd.DataFrame, return_column: str) -> float:
    returns = pd.to_numeric(frame.get(return_column), errors="coerce").dropna()
    if returns.empty:
        return float("nan")
    return float(returns.mean())


def _momentum_signal(series: pd.Series) -> pd.Series:
    shifted = pd.to_numeric(series, errors="coerce").shift(2)
    return (
        (1.0 + shifted)
        .rolling(window=11, min_periods=9)
        .apply(np.prod, raw=True)
        .sub(1.0)
    )


def _assign_three_way_bucket(series: pd.Series) -> tuple[pd.Series, float, float]:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    if valid.nunique() < 3:
        return out, float("nan"), float("nan")

    low_break = float(valid.quantile(0.3))
    high_break = float(valid.quantile(0.7))
    if not np.isfinite(low_break) or not np.isfinite(high_break) or low_break >= high_break:
        return out, low_break, high_break

    numeric = pd.to_numeric(series, errors="coerce")
    out.loc[numeric.le(low_break)] = "low"
    out.loc[numeric.gt(high_break)] = "high"
    out.loc[numeric.gt(low_break) & numeric.le(high_break)] = "mid"
    return out, low_break, high_break


def _factor_rows(
    month_end_date: pd.Timestamp,
    country_code: str,
    factor_code: str,
    factor_value: float,
) -> list[dict[str, object]]:
    return [
        {
            "country_code": country_code,
            "factor_code": factor_code,
            "month_end_date": month_end_date,
            "factor_value": factor_value,
        }
    ]


def build_sweden_factor_tables(
    market_monthly: pd.DataFrame,
    fundamental_snapshots: pd.DataFrame,
    rates_monthly: pd.DataFrame,
    security_dim: pd.DataFrame | None = None,
    *,
    country_code: str = "SWE",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if market_monthly is None or market_monthly.empty:
        return _empty_factors(), _empty_audit()

    market = market_monthly.copy()
    market["month_end_date"] = pd.to_datetime(market["month_end_date"], errors="coerce")
    market["monthly_total_return"] = pd.to_numeric(market["monthly_total_return"], errors="coerce")
    market["month_end_market_cap"] = pd.to_numeric(market["month_end_market_cap"], errors="coerce")
    market_cap_millions_source = market.get(
        "month_end_market_cap_millions",
        pd.Series(np.nan, index=market.index),
    )
    market["month_end_market_cap_millions"] = pd.to_numeric(market_cap_millions_source, errors="coerce")
    market["month_end_market_cap_millions"] = market["month_end_market_cap_millions"].where(
        market["month_end_market_cap_millions"].notna(),
        market["month_end_market_cap"] / MARKET_CAP_FULL_TO_MILLIONS,
    )
    market = market.sort_values(["security_id", "month_end_date"]).reset_index(drop=True)
    market["lagged_me"] = market.groupby("security_id")["month_end_market_cap"].shift(1)

    snapshots = pd.DataFrame(fundamental_snapshots).copy()
    if not snapshots.empty:
        snapshots["month_end_date"] = pd.to_datetime(snapshots["month_end_date"], errors="coerce")
        snapshots["effective_month_end"] = pd.to_datetime(snapshots.get("effective_month_end"), errors="coerce")
        snapshots["BE"] = pd.to_numeric(
            snapshots.get("BE", snapshots.get("book_equity")),
            errors="coerce",
        )
        snapshots = snapshots[["company_id", "month_end_date", "effective_month_end", "BE"]].drop_duplicates(
            ["company_id", "month_end_date"],
            keep="last",
        )

    panel = market.merge(snapshots, on=["company_id", "month_end_date"], how="left") if not snapshots.empty else market.copy()
    if security_dim is not None and not security_dim.empty:
        panel = panel.merge(
            security_dim[["security_id", "issue_type_code"]].drop_duplicates("security_id"),
            on="security_id",
            how="left",
        )
    panel["country_code"] = panel.get("country_code", country_code).fillna(country_code)
    panel["common_equity_flag"] = _common_equity_mask(panel)
    be = pd.to_numeric(panel.get("BE"), errors="coerce")
    me_millions = pd.to_numeric(panel.get("month_end_market_cap_millions"), errors="coerce")
    panel["current_be_me"] = np.where(
        be.notna() & be.gt(0) & me_millions.notna() & me_millions.gt(0),
        be / me_millions,
        np.nan,
    )
    panel["sort_me"] = panel["lagged_me"]
    panel["sort_be_me"] = panel.groupby("security_id")["current_be_me"].shift(1)
    panel["momentum_signal"] = panel.groupby("security_id", group_keys=False)["monthly_total_return"].apply(_momentum_signal)

    rates = pd.DataFrame(rates_monthly).copy()
    if not rates.empty:
        rates["month_end_date"] = pd.to_datetime(rates["month_end_date"], errors="coerce")
        rates["raw_annual_rate_pct"] = pd.to_numeric(rates["raw_annual_rate_pct"], errors="coerce")
        rates["monthly_rate_proxy"] = pd.to_numeric(rates["monthly_rate_proxy"], errors="coerce")
    rates_pivot = rates.pivot_table(
        index="month_end_date",
        columns="rate_code",
        values=["raw_annual_rate_pct", "monthly_rate_proxy"],
        aggfunc="last",
    ) if not rates.empty else pd.DataFrame()

    factor_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for month_end_date, monthly_frame in panel.groupby("month_end_date", sort=True):
        eligible = monthly_frame.loc[monthly_frame["common_equity_flag"]].copy()
        weighted_market = _weighted_return(eligible, "monthly_total_return", "lagged_me")
        factor_rows.extend(
            _factor_rows(month_end_date, country_code, "RM_LOCAL_VW", weighted_market * 100.0 if np.isfinite(weighted_market) else float("nan"))
        )
        audit_rows.append(
            {
                "country_code": country_code,
                "factor_code": "RM_LOCAL_VW",
                "month_end_date": month_end_date,
                "leg_code": "market",
                "breakpoint_low": np.nan,
                "breakpoint_high": np.nan,
                "n_constituents": int(
                    (eligible["monthly_total_return"].notna() & eligible["lagged_me"].notna() & eligible["lagged_me"].gt(0)).sum()
                ),
                "lagged_weight_sum": float(
                    pd.to_numeric(eligible["lagged_me"], errors="coerce")
                    .where(pd.to_numeric(eligible["lagged_me"], errors="coerce").gt(0))
                    .sum(min_count=1)
                ),
                "ew_return": _equal_weight_return(eligible, "monthly_total_return") * 100.0,
                "vw_return": weighted_market * 100.0 if np.isfinite(weighted_market) else float("nan"),
                "thin_leg_flag": int(eligible["security_id"].nunique()) < 3,
            }
        )

        rf_value = float("nan")
        y10_value = float("nan")
        if not rates_pivot.empty and month_end_date in rates_pivot.index:
            if ("monthly_rate_proxy", "RF_3M_PROXY") in rates_pivot.columns:
                rf_value = float(rates_pivot.loc[month_end_date, ("monthly_rate_proxy", "RF_3M_PROXY")])
            if ("raw_annual_rate_pct", "YIELD_10Y") in rates_pivot.columns:
                y10_value = float(rates_pivot.loc[month_end_date, ("raw_annual_rate_pct", "YIELD_10Y")])

        factor_rows.extend(_factor_rows(month_end_date, country_code, "RF_3M_PROXY", rf_value * 100.0 if np.isfinite(rf_value) else float("nan")))
        factor_rows.extend(_factor_rows(month_end_date, country_code, "YIELD_10Y", y10_value))
        factor_rows.extend(
            _factor_rows(
                month_end_date,
                country_code,
                "RMRF",
                ((weighted_market - rf_value) * 100.0) if np.isfinite(weighted_market) and np.isfinite(rf_value) else float("nan"),
            )
        )

        for factor_code, sort_column, labels in [
            ("SMB", "sort_me", {"low": "small", "mid": "middle", "high": "big"}),
            ("HML", "sort_be_me", {"low": "low_bm", "mid": "middle_bm", "high": "high_bm"}),
            ("UMD", "momentum_signal", {"low": "loser", "mid": "middle_mom", "high": "winner"}),
        ]:
            factor_input = eligible.loc[
                eligible["monthly_total_return"].notna()
                & eligible["lagged_me"].notna()
                & eligible["lagged_me"].gt(0)
                & pd.to_numeric(eligible[sort_column], errors="coerce").notna()
            ].copy()
            buckets, low_break, high_break = _assign_three_way_bucket(factor_input[sort_column])
            factor_input["bucket"] = buckets
            leg_values: dict[str, float] = {}
            for bucket_key, leg_code in labels.items():
                leg_frame = factor_input.loc[factor_input["bucket"] == bucket_key].copy()
                vw_leg = _weighted_return(leg_frame, "monthly_total_return", "lagged_me")
                ew_leg = _equal_weight_return(leg_frame, "monthly_total_return")
                if bucket_key != "mid":
                    leg_values[bucket_key] = vw_leg
                audit_rows.append(
                    {
                        "country_code": country_code,
                        "factor_code": factor_code,
                        "month_end_date": month_end_date,
                        "leg_code": leg_code,
                        "breakpoint_low": low_break,
                        "breakpoint_high": high_break,
                        "n_constituents": int(leg_frame["security_id"].nunique()),
                        "lagged_weight_sum": float(pd.to_numeric(leg_frame["lagged_me"], errors="coerce").sum(min_count=1)),
                        "ew_return": ew_leg * 100.0 if np.isfinite(ew_leg) else float("nan"),
                        "vw_return": vw_leg * 100.0 if np.isfinite(vw_leg) else float("nan"),
                        "thin_leg_flag": int(leg_frame["security_id"].nunique()) < 3,
                    }
                )

            if factor_code == "SMB":
                factor_value = (leg_values.get("low", float("nan")) - leg_values.get("high", float("nan"))) * 100.0
            else:
                factor_value = (leg_values.get("high", float("nan")) - leg_values.get("low", float("nan"))) * 100.0
            factor_rows.extend(_factor_rows(month_end_date, country_code, factor_code, factor_value))

    factors = pd.DataFrame(factor_rows, columns=FACTOR_COLUMNS)
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    if not factors.empty:
        factors = factors.sort_values(["country_code", "factor_code", "month_end_date"]).reset_index(drop=True)
    if not audit.empty:
        audit = audit.sort_values(["country_code", "factor_code", "month_end_date", "leg_code"]).reset_index(drop=True)
    return factors, audit
