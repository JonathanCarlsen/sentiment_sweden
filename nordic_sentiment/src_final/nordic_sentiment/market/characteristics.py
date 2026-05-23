from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_CAP_FULL_TO_MILLIONS = 1_000_000.0

CHARACTERISTIC_COLUMNS = [
    "security_id",
    "company_id",
    "country_code",
    "month_end_date",
    "fiscal_period_end",
    "report_available_date",
    "effective_month_end",
    "reporting_periodicity",
    "ret",
    "ME",
    "risk",
    "IVOL_FF3",
    "BE",
    "book_equity_base_source",
    "book_equity_nonpositive_flag",
    "book_equity_zero_ceq_replaced_flag",
    "BE_ME_raw",
    "BE_ME",
    "PPE_A_raw",
    "PPE_A",
    "E_plus_BE_raw",
    "E_plus_BE",
    "D_PAYER",
    "D_PAYER_daily",
    "D_PAYER_fundamental",
    "D_PAYER_source",
    "dividend_payer_year",
    "regular_dividend_payer_fundamental",
    "regular_dividend_payer_formation_flag",
    "consecutive_nondividend_years",
    "NON_D_PAYER",
    "E_POSITIVE",
    "UNPROFITABLE",
    "GS_raw",
    "GS",
    "ILLIQ_raw",
    "ILLIQ",
    "XTURN_raw",
    "XTURN",
    "liquidity_dividend_status",
    "be_me_valid_flag",
    "be_me_invalid_reason",
    "ratio_invalid_reason",
    "age",
    "ME10",
    "risk10",
    "IVOL_FF310",
    "BE_ME10",
    "GS10",
    "age10",
    "E_plus_BE10",
    "PPE_A10",
    "ILLIQ10",
    "XTURN10",
]


def winsorize_by_group(
    frame: pd.DataFrame,
    column: str,
    group_column: str,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")

    def _clip_group(group: pd.Series) -> pd.Series:
        non_null = group.dropna()
        if non_null.empty:
            return group
        lower_bound = non_null.quantile(lower)
        upper_bound = non_null.quantile(upper)
        return group.clip(lower=lower_bound, upper=upper_bound)

    return values.groupby(frame[group_column]).transform(_clip_group)


def assign_deciles_with_reference_subset(
    frame: pd.DataFrame,
    column: str,
    group_column: str,
    reference_mask: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    groups = frame[group_column]
    output = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    for group_value in groups.dropna().unique():
        group_mask = groups.eq(group_value)
        reference_values = values.loc[group_mask & reference_mask.fillna(False) & values.notna()]
        if reference_values.empty:
            continue
        quantiles = reference_values.quantile(np.linspace(0.0, 1.0, 11)).to_numpy(dtype="float64")
        bins = np.unique(quantiles)
        target_values = values.loc[group_mask]

        if bins.size <= 1:
            output.loc[group_mask & target_values.notna()] = 1
            continue

        labels = list(range(1, bins.size))
        cut = pd.cut(
            target_values,
            bins=bins,
            labels=labels,
            include_lowest=True,
            duplicates="drop",
        )
        output.loc[group_mask & target_values.notna()] = cut.astype("Int64")

    return output


def _empty_characteristics() -> pd.DataFrame:
    return pd.DataFrame(columns=CHARACTERISTIC_COLUMNS)


def _to_numeric(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def _safe_ratio(numerator: pd.Series | None, denominator: pd.Series | None, index: pd.Index, scale: float = 1.0) -> pd.Series:
    numer = _to_numeric(numerator, index=index)
    denom = _to_numeric(denominator, index=index)
    return pd.Series(
        np.where(denom.notna() & denom.ne(0), (numer / denom) * scale, np.nan),
        index=index,
        dtype="float64",
    )


def _positive_denominator_ratio(
    numerator: pd.Series | None,
    denominator: pd.Series | None,
    index: pd.Index,
    *,
    scale: float = 1.0,
    require_positive_numerator: bool = False,
) -> pd.Series:
    numer = _to_numeric(numerator, index=index)
    denom = _to_numeric(denominator, index=index)
    valid = denom.notna() & denom.gt(0) & numer.notna()
    if require_positive_numerator:
        valid &= numer.gt(0)
    return pd.Series(np.where(valid, (numer / denom) * scale, np.nan), index=index, dtype="float64")


def _join_reasons(reason_columns: pd.DataFrame) -> pd.Series:
    if reason_columns.empty:
        return pd.Series("ok", index=reason_columns.index, dtype="string")
    reasons: list[str] = []
    for _, row in reason_columns.iterrows():
        values = [str(value) for value in row.dropna().tolist() if str(value)]
        reasons.append(";".join(values) if values else "ok")
    return pd.Series(reasons, index=reason_columns.index, dtype="string")


def _market_cap_millions(frame: pd.DataFrame, index: pd.Index) -> pd.Series:
    market_cap_millions = _to_numeric(frame.get("month_end_market_cap_millions"), index)
    market_cap_raw = _to_numeric(frame.get("month_end_market_cap"), index)
    return market_cap_millions.where(market_cap_millions.notna(), market_cap_raw / MARKET_CAP_FULL_TO_MILLIONS)


def _availability_row(
    frame: pd.DataFrame,
    *,
    column_name: str,
    source_columns: list[str],
    translation_rule: str,
) -> dict[str, object]:
    available_sources = [column for column in source_columns if column in frame.columns]
    non_null_rows = int(frame[available_sources].notna().any(axis=1).sum()) if available_sources else 0
    return {
        "column_name": column_name,
        "source_columns": ",".join(source_columns),
        "available": bool(available_sources),
        "non_null_source_rows": non_null_rows,
        "translation_rule": translation_rule,
    }


def build_sweden_availability_audit(fundamental_snapshots: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(fundamental_snapshots).copy()
    rules = [
        (
            "BE",
            ["seqq", "at", "ltq", "pstkq", "book_equity", "BE"],
            "Latest filing snapshot book equity: seqq, with at-minus-ltq fallback and preferred-stock adjustment.",
        ),
        (
            "BE_ME",
            ["seqq", "at", "ltq", "pstkq", "book_equity", "BE"],
            "Positive book equity in millions divided by positive month-end market cap in millions.",
        ),
        (
            "PPE_A",
            ["ppegt", "ppentq", "at"],
            "Net property, plant, and equipment over assets; raw ppentq is mapped to ppegt during staging.",
        ),
        (
            "E_plus_BE",
            ["earning_ttm", "earning", "ib", "book_equity", "BE"],
            "Positive trailing earnings over positive book equity.",
        ),
        (
            "GS",
            ["revenue_ttm", "lag_revenue_ttm", "GS"],
            "Trailing annual revenue growth from revtq TTM versus the prior-year revtq TTM window.",
        ),
        (
            "D_PAYER",
            [
                "regular_dividend_payer_fundamental",
                "dividend_payer_year",
                "dvtq",
                "dvty",
                "dvy",
            ],
            "Backward-looking regular dividend-payer flag from fundamentals; daily cash-distribution history is retained only as fallback/diagnostic.",
        ),
        (
            "E_POSITIVE",
            ["earning_ttm", "earning", "ib"],
            "Indicator for positive trailing earnings from ib, with non-positive earnings classified as unprofitable.",
        ),
    ]
    return pd.DataFrame(
        [_availability_row(frame, column_name=name, source_columns=sources, translation_rule=rule) for name, sources, rule in rules]
    )


def build_sweden_characteristics_layers(
    market_monthly: pd.DataFrame,
    fundamental_snapshots: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if market_monthly is None or market_monthly.empty:
        empty = _empty_characteristics()
        return empty, empty.copy(), build_sweden_availability_audit(pd.DataFrame())

    market = market_monthly.copy()
    market["month_end_date"] = pd.to_datetime(market["month_end_date"], errors="coerce")
    market = market.sort_values(["security_id", "month_end_date"])

    if fundamental_snapshots is None or fundamental_snapshots.empty:
        merged = market.copy()
    else:
        snapshots = fundamental_snapshots.copy()
        snapshots["month_end_date"] = pd.to_datetime(snapshots["month_end_date"], errors="coerce")
        merged = market.merge(
            snapshots,
            on=["company_id", "month_end_date"],
            how="left",
            suffixes=("", "_fund"),
        )

    idx = merged.index
    merged["country_code"] = merged.get("country_code", "SWE")
    merged["ret"] = _to_numeric(merged.get("monthly_total_return"), idx) * 100.0
    merged["ME"] = _market_cap_millions(merged, idx)
    merged["risk"] = (
        merged.groupby("security_id")["ret"].rolling(window=12, min_periods=9).std().reset_index(level=0, drop=True)
    )

    first_month = merged.groupby("security_id")["month_end_date"].transform("min")
    months_since_listing = (
        (merged["month_end_date"].dt.year - first_month.dt.year) * 12
        + (merged["month_end_date"].dt.month - first_month.dt.month)
    )
    merged["age"] = months_since_listing / 12.0

    merged["BE"] = _to_numeric(merged.get("BE", merged.get("book_equity")), idx)
    me = _to_numeric(merged.get("ME"), idx)
    at = _to_numeric(merged.get("at"), idx)
    revenue_ttm = _to_numeric(merged.get("revenue_ttm"), idx)
    lag_revenue_ttm = _to_numeric(merged.get("lag_revenue_ttm"), idx)
    be_positive = merged["BE"].gt(0)
    me_positive = me.gt(0)
    at_positive = at.gt(0)
    valid_sales_growth_denominator = lag_revenue_ttm.gt(0) & revenue_ttm.ge(0)

    merged["BE_ME_raw"] = _safe_ratio(merged["BE"], me, idx)
    merged["BE_ME"] = _positive_denominator_ratio(merged["BE"], me, idx, require_positive_numerator=True)
    merged["be_me_valid_flag"] = merged["BE_ME"].notna()
    be_me_reasons = pd.DataFrame(
        {
            "missing_be": np.where(merged["BE"].isna(), "missing_be", pd.NA),
            "nonpositive_be": np.where(merged["BE"].notna() & ~be_positive, "nonpositive_be", pd.NA),
            "missing_me": np.where(me.isna(), "missing_me", pd.NA),
            "nonpositive_me": np.where(me.notna() & ~me_positive, "nonpositive_me", pd.NA),
        },
        index=idx,
    )
    merged["be_me_invalid_reason"] = _join_reasons(be_me_reasons).where(~merged["be_me_valid_flag"], "ok")

    merged["PPE_A_raw"] = _safe_ratio(merged.get("ppegt"), at, idx, scale=100.0)
    merged["PPE_A"] = merged["PPE_A_raw"].where(at_positive)

    earning_ttm = _to_numeric(merged.get("earning_ttm"), idx)
    earning_positive = earning_ttm.clip(lower=0.0)
    merged["E_plus_BE_raw"] = _safe_ratio(earning_positive, merged["BE"], idx, scale=100.0)
    merged["E_plus_BE"] = merged["E_plus_BE_raw"].where(be_positive)

    regular_dividend_payer = _to_numeric(merged.get("regular_dividend_payer_fundamental"), idx)
    merged["D_PAYER_fundamental"] = regular_dividend_payer
    merged["D_PAYER"] = merged["D_PAYER_fundamental"]
    merged["D_PAYER_source"] = np.where(merged["D_PAYER"].notna(), "fundamentals", pd.NA)

    merged["GS_raw"] = _to_numeric(merged.get("GS"), idx)
    merged["GS"] = merged["GS_raw"].where(valid_sales_growth_denominator)
    merged["E_POSITIVE"] = np.where(earning_ttm.notna(), earning_ttm.gt(0).astype(float), np.nan)
    merged["UNPROFITABLE"] = np.where(earning_ttm.notna(), earning_ttm.le(0).astype(float), np.nan)
    merged["NON_D_PAYER"] = np.where(merged["D_PAYER"].notna(), 1.0 - merged["D_PAYER"], np.nan)
    ratio_reasons = pd.DataFrame(
        {
            "be_me_invalid": np.where(merged["BE_ME_raw"].notna() & merged["BE_ME"].isna(), "be_me_invalid", pd.NA),
            "book_equity_denominator_invalid": np.where(
                merged["E_plus_BE_raw"].notna() & merged["E_plus_BE"].isna(),
                "book_equity_denominator_invalid",
                pd.NA,
            ),
            "asset_denominator_invalid": np.where(
                merged["PPE_A_raw"].notna() & merged["PPE_A"].isna(),
                "asset_denominator_invalid",
                pd.NA,
            ),
            "sales_growth_denominator_invalid": np.where(
                merged["GS_raw"].notna() & merged["GS"].isna(),
                "sales_growth_denominator_invalid",
                pd.NA,
            ),
        },
        index=idx,
    )
    merged["ratio_invalid_reason"] = _join_reasons(ratio_reasons)
    merged["fyear"] = merged["month_end_date"].dt.year

    for column in ["ME", "risk", "E_plus_BE", "PPE_A", "BE_ME", "GS"]:
        merged[column] = winsorize_by_group(merged, column, "month_end_date")
    merged["age"] = winsorize_by_group(merged, "age", "fyear")

    reference_mask = merged["security_id"].notna()
    for column in ["ME", "risk", "BE_ME", "GS"]:
        merged[f"{column}10"] = assign_deciles_with_reference_subset(merged, column, "month_end_date", reference_mask)
    merged["age10"] = assign_deciles_with_reference_subset(merged, "age", "fyear", reference_mask)
    for column in ["E_plus_BE", "PPE_A"]:
        merged[f"{column}10"] = assign_deciles_with_reference_subset(merged, column, "month_end_date", reference_mask)

    merged.loc[earning_ttm.le(0) | earning_ttm.isna(), "E_plus_BE10"] = 0
    merged.loc[_to_numeric(merged.get("ppegt"), idx).fillna(0.0).eq(0), "PPE_A10"] = 0

    merged["me"] = merged["ME"]
    merged["be_me"] = merged["BE_ME"]
    merged["gs"] = merged["GS"]
    merged["me10"] = merged["ME10"]
    merged["be_me10"] = merged["BE_ME10"]
    merged["gs10"] = merged["GS10"]

    ordered = [column for column in CHARACTERISTIC_COLUMNS if column in merged.columns]
    passthrough = [column for column in merged.columns if column not in ordered]
    monthly = (
        merged.sort_values(["security_id", "month_end_date", "fiscal_period_end"])
        .drop_duplicates(["security_id", "month_end_date"], keep="last")[ordered + passthrough]
        .reset_index(drop=True)
    )

    december = monthly.loc[monthly["month_end_date"].dt.month == 12].copy()
    if december.empty:
        december = monthly.sort_values("month_end_date").groupby(
            ["security_id", monthly["month_end_date"].dt.year], as_index=False
        ).tail(1)
    december["year_num"] = december["month_end_date"].dt.year

    return monthly, december.reset_index(drop=True), build_sweden_availability_audit(fundamental_snapshots)


def attach_liquidity_dividend_to_characteristics(
    characteristics_monthly: pd.DataFrame,
    liquidity_dividend_monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = pd.DataFrame(characteristics_monthly).copy()
    if monthly.empty:
        return monthly, monthly.copy()

    if "D_PAYER" not in monthly.columns:
        monthly["D_PAYER"] = np.nan
    if "D_PAYER_fundamental" not in monthly.columns:
        monthly["D_PAYER_fundamental"] = pd.to_numeric(monthly["D_PAYER"], errors="coerce")
    if "D_PAYER_source" not in monthly.columns:
        monthly["D_PAYER_source"] = np.where(monthly["D_PAYER"].notna(), "fundamentals", pd.NA)

    liquidity = pd.DataFrame(liquidity_dividend_monthly).copy()
    if not liquidity.empty:
        liquidity["month_end_date"] = pd.to_datetime(liquidity["month_end_date"], errors="coerce")
        merge_columns = [
            column
            for column in [
                "security_id",
                "month_end_date",
                "ILLIQ_raw",
                "ILLIQ",
                "XTURN_raw",
                "XTURN",
                "D_PAYER_daily",
                "daily_dividend_observed_flag",
                "liquidity_dividend_status",
                "n_daily_obs",
                "n_illiq_obs",
                "monthly_trading_volume",
                "month_end_shares_outstanding_liquidity",
                "monthly_traded_value_millions",
            ]
            if column in liquidity.columns
        ]
        monthly = monthly.merge(liquidity[merge_columns], on=["security_id", "month_end_date"], how="left")
    else:
        for column in [
            "ILLIQ_raw",
            "ILLIQ",
            "XTURN_raw",
            "XTURN",
            "D_PAYER_daily",
            "daily_dividend_observed_flag",
            "liquidity_dividend_status",
            "n_daily_obs",
            "n_illiq_obs",
        ]:
            if column not in monthly.columns:
                monthly[column] = np.nan

    daily_d_payer = pd.to_numeric(monthly.get("D_PAYER_daily"), errors="coerce")
    fundamental_d_payer = pd.to_numeric(monthly.get("D_PAYER_fundamental"), errors="coerce")
    monthly["D_PAYER"] = fundamental_d_payer.where(fundamental_d_payer.notna(), daily_d_payer)
    monthly["D_PAYER_source"] = np.select(
        [fundamental_d_payer.notna(), fundamental_d_payer.isna() & daily_d_payer.notna()],
        ["fundamentals", "daily"],
        default=pd.NA,
    )
    monthly["NON_D_PAYER"] = np.where(monthly["D_PAYER"].notna(), 1.0 - monthly["D_PAYER"], np.nan)
    for column in ["ILLIQ", "XTURN"]:
        if column in monthly.columns:
            monthly[column] = winsorize_by_group(monthly, column, "month_end_date")

    reference_mask = monthly["security_id"].notna()
    for column in ["ILLIQ", "XTURN"]:
        if column in monthly.columns:
            monthly[f"{column}10"] = assign_deciles_with_reference_subset(
                monthly,
                column,
                "month_end_date",
                reference_mask,
            )

    ordered = [column for column in CHARACTERISTIC_COLUMNS if column in monthly.columns]
    passthrough = [column for column in monthly.columns if column not in ordered]
    monthly = monthly[ordered + passthrough].sort_values(["security_id", "month_end_date"]).reset_index(drop=True)

    annual = monthly.loc[monthly["month_end_date"].dt.month == 12].copy()
    if annual.empty:
        annual = monthly.sort_values("month_end_date").groupby(
            ["security_id", monthly["month_end_date"].dt.year],
            as_index=False,
        ).tail(1)
    annual["year_num"] = annual["month_end_date"].dt.year
    return monthly, annual.reset_index(drop=True)


def attach_ivol_to_characteristics(
    characteristics_monthly: pd.DataFrame,
    ivol_monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = pd.DataFrame(characteristics_monthly).copy()
    if monthly.empty:
        return monthly, monthly.copy()

    ivol = pd.DataFrame(ivol_monthly).copy()
    if ivol.empty:
        if "year_num" in monthly.columns:
            annual = monthly.loc[monthly["month_end_date"].dt.month == 12].copy()
            return monthly, annual.reset_index(drop=True)
        annual = monthly.loc[monthly["month_end_date"].dt.month == 12].copy()
        return monthly, annual.reset_index(drop=True)

    ivol["month_end_date"] = pd.to_datetime(ivol["month_end_date"], errors="coerce")
    monthly = monthly.merge(
        ivol[["security_id", "month_end_date", "ivol_ff3"]],
        on=["security_id", "month_end_date"],
        how="left",
    )
    monthly = monthly.rename(columns={"ivol_ff3": "IVOL_FF3"})
    monthly["IVOL_FF3"] = winsorize_by_group(monthly, "IVOL_FF3", "month_end_date")

    reference_mask = monthly["security_id"].notna()
    monthly["IVOL_FF310"] = assign_deciles_with_reference_subset(monthly, "IVOL_FF3", "month_end_date", reference_mask)

    ordered = [column for column in CHARACTERISTIC_COLUMNS if column in monthly.columns]
    passthrough = [column for column in monthly.columns if column not in ordered]
    monthly = monthly[ordered + passthrough].sort_values(["security_id", "month_end_date"]).reset_index(drop=True)

    annual = monthly.loc[monthly["month_end_date"].dt.month == 12].copy()
    if annual.empty:
        annual = monthly.sort_values("month_end_date").groupby(
            ["security_id", monthly["month_end_date"].dt.year], as_index=False
        ).tail(1)
    annual["year_num"] = annual["month_end_date"].dt.year
    return monthly, annual.reset_index(drop=True)


def build_characteristics_monthly(
    market_monthly: pd.DataFrame,
    fundamental_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    monthly, _, _ = build_sweden_characteristics_layers(market_monthly, fundamental_snapshots)
    return monthly
