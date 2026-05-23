from __future__ import annotations

import numpy as np
import pandas as pd


HYBRID_WORKBOOK_START = pd.Timestamp("2007-01-31")
MARKET_CAP_FULL_TO_MILLIONS = 1_000_000.0

MONTHLY_SOURCE_COLUMNS = [
    "source_name",
    "security_id",
    "company_id",
    "country_code",
    "month_end_date",
    "last_trade_date_in_month",
    "gvkey",
    "iid",
    "isin",
    "exchange_code",
    "issue_type_code",
    "company_name",
    "month_end_price",
    "month_end_shares_outstanding",
    "month_end_trading_volume",
    "month_end_market_cap",
    "month_end_market_cap_millions",
    "month_end_total_return_factor",
    "monthly_total_return",
    "derived_from_daily_flag",
    "source_priority",
    "return_calc_status",
]


def _empty_monthly_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=MONTHLY_SOURCE_COLUMNS)


def _pick_column(frame: pd.DataFrame, *candidates: str) -> pd.Series:
    for candidate in candidates:
        if candidate in frame.columns:
            return frame[candidate]
    return pd.Series(np.nan, index=frame.index)


def _compute_returns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["security_id", "month_end_date"]).copy()
    adj_price = out["month_end_price"] * out["month_end_total_return_factor"]
    prev_adj = adj_price.groupby(out["security_id"]).shift(1)
    valid = adj_price.notna() & prev_adj.notna() & prev_adj.ne(0)
    out["monthly_total_return"] = np.where(valid, (adj_price / prev_adj) - 1.0, np.nan)
    out["return_calc_status"] = np.where(valid, "ok", "missing_leg")
    return out


def _is_valid_monthly_row(frame: pd.DataFrame) -> pd.Series:
    price = pd.to_numeric(frame.get("month_end_price"), errors="coerce")
    return price.notna() & price.gt(0)


def derive_monthly_from_daily(daily_prices: pd.DataFrame) -> pd.DataFrame:
    if daily_prices is None or daily_prices.empty:
        return _empty_monthly_frame()
    if not {"security_id", "trade_date"}.issubset(daily_prices.columns):
        return _empty_monthly_frame()

    frame = daily_prices.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["month_end_date"] = frame["trade_date"].dt.to_period("M").dt.to_timestamp("M")
    frame = frame.dropna(subset=["security_id", "trade_date", "month_end_date"])
    idx = frame.groupby(["security_id", "month_end_date"])["trade_date"].idxmax()
    month_end = frame.loc[idx].copy()
    month_end["last_trade_date_in_month"] = month_end["trade_date"]
    month_end["gvkey"] = _pick_column(month_end, "gvkey")
    month_end["iid"] = _pick_column(month_end, "iid")
    month_end["isin"] = _pick_column(month_end, "isin")
    month_end["exchange_code"] = _pick_column(month_end, "exchg", "exchange_code")
    month_end["issue_type_code"] = _pick_column(month_end, "tpci", "issue_type_code")
    month_end["company_name"] = _pick_column(month_end, "conm", "company_name")
    month_end["month_end_price"] = _pick_column(month_end, "prccd", "close_price")
    month_end["month_end_shares_outstanding"] = _pick_column(month_end, "cshoc", "shares_outstanding")
    if "cshtrd" in frame.columns:
        frame["cshtrd"] = pd.to_numeric(frame["cshtrd"], errors="coerce")
    if "cshtrd" in frame.columns:
        volume = frame.groupby(["security_id", "month_end_date"])["cshtrd"].sum(min_count=1)
        month_end = month_end.merge(
            volume.rename("month_end_trading_volume").reset_index(),
            on=["security_id", "month_end_date"],
            how="left",
        )
    else:
        month_end["month_end_trading_volume"] = np.nan
    month_end["month_end_market_cap"] = month_end["month_end_price"] * month_end["month_end_shares_outstanding"]
    month_end["month_end_market_cap_millions"] = month_end["month_end_market_cap"] / MARKET_CAP_FULL_TO_MILLIONS
    month_end["month_end_total_return_factor"] = _pick_column(month_end, "trfd", "total_return_factor")
    month_end["source_name"] = "daily_derived"
    month_end["derived_from_daily_flag"] = 1
    month_end["source_priority"] = 2
    out = month_end.reindex(columns=MONTHLY_SOURCE_COLUMNS)
    return _compute_returns(out)


def prepare_monthly_workbook_source(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    if monthly_prices is None or monthly_prices.empty:
        return _empty_monthly_frame()
    if not {"security_id", "month_end_date"}.issubset(monthly_prices.columns):
        return _empty_monthly_frame()

    frame = monthly_prices.copy()
    frame["month_end_date"] = pd.to_datetime(frame["month_end_date"], errors="coerce")
    if "trade_date" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    else:
        frame["trade_date"] = frame["month_end_date"]
    frame = frame.sort_values(["security_id", "month_end_date", "trade_date"]).drop_duplicates(
        ["security_id", "month_end_date"],
        keep="last",
    )
    frame["last_trade_date_in_month"] = frame["trade_date"]
    frame["gvkey"] = _pick_column(frame, "gvkey")
    frame["iid"] = _pick_column(frame, "iid")
    frame["isin"] = _pick_column(frame, "isin")
    frame["exchange_code"] = _pick_column(frame, "exchg", "exchange_code")
    frame["issue_type_code"] = _pick_column(frame, "tpci", "issue_type_code")
    frame["company_name"] = _pick_column(frame, "conm", "company_name")
    frame["month_end_price"] = _pick_column(frame, "prccd", "close_price")
    frame["month_end_shares_outstanding"] = _pick_column(frame, "cshoc", "shares_outstanding")
    frame["month_end_trading_volume"] = _pick_column(frame, "cshtrm", "monthly_trading_volume")
    frame["month_end_market_cap"] = frame["month_end_price"] * frame["month_end_shares_outstanding"]
    frame["month_end_market_cap_millions"] = frame["month_end_market_cap"] / MARKET_CAP_FULL_TO_MILLIONS
    frame["month_end_total_return_factor"] = _pick_column(frame, "trfd", "total_return_factor")
    frame["source_name"] = "monthly_workbook"
    frame["derived_from_daily_flag"] = 0
    frame["source_priority"] = 1
    out = frame.reindex(columns=MONTHLY_SOURCE_COLUMNS)
    return _compute_returns(out)


def build_monthly_market_panels(
    daily_prices: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    workbook_priority_start: pd.Timestamp = HYBRID_WORKBOOK_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_source = derive_monthly_from_daily(daily_prices)
    workbook_source = prepare_monthly_workbook_source(monthly_prices)
    if workbook_source.empty:
        source_table = daily_source.copy()
    elif daily_source.empty:
        source_table = workbook_source.copy()
    else:
        source_table = pd.concat([workbook_source, daily_source], ignore_index=True)

    if source_table.empty:
        resolved = _empty_monthly_frame().assign(
            workbook_overlap_flag=pd.Series(dtype="boolean"),
            daily_available_flag=pd.Series(dtype="boolean"),
            workbook_available_flag=pd.Series(dtype="boolean"),
            workbook_valid_flag=pd.Series(dtype="boolean"),
            post_workbook_priority_flag=pd.Series(dtype="boolean"),
            resolved_source_name=pd.Series(dtype="string"),
            resolution_reason=pd.Series(dtype="string"),
            price_abs_diff=pd.Series(dtype="float64"),
            market_cap_abs_diff=pd.Series(dtype="float64"),
            return_abs_diff=pd.Series(dtype="float64"),
        )
        return source_table, resolved

    if workbook_source.empty:
        resolved = daily_source.copy()
        for column, dtype in [
            ("workbook_overlap_flag", "boolean"),
            ("daily_available_flag", "boolean"),
            ("workbook_available_flag", "boolean"),
            ("workbook_valid_flag", "boolean"),
            ("post_workbook_priority_flag", "boolean"),
            ("resolved_source_name", "string"),
            ("resolution_reason", "string"),
            ("price_abs_diff", "float64"),
            ("market_cap_abs_diff", "float64"),
            ("return_abs_diff", "float64"),
        ]:
            resolved[column] = pd.Series(index=resolved.index, dtype=dtype)
        resolved["daily_available_flag"] = True
        resolved["workbook_available_flag"] = False
        resolved["workbook_valid_flag"] = False
        resolved["post_workbook_priority_flag"] = resolved["month_end_date"].ge(workbook_priority_start)
        resolved["resolved_source_name"] = resolved["source_name"]
        resolved["resolution_reason"] = np.where(
            resolved["month_end_date"].ge(workbook_priority_start),
            "post_2007_daily_fallback_missing_workbook",
            "pre_2007_daily_only",
        )
        return source_table.reset_index(drop=True), resolved.reset_index(drop=True)

    workbook = workbook_source.add_suffix("_workbook")
    daily = daily_source.add_suffix("_daily")
    overlap = workbook.merge(
        daily,
        left_on=["security_id_workbook", "month_end_date_workbook"],
        right_on=["security_id_daily", "month_end_date_daily"],
        how="outer",
    )
    overlap["security_id"] = overlap["security_id_workbook"].combine_first(overlap["security_id_daily"])
    overlap["month_end_date"] = overlap["month_end_date_workbook"].combine_first(overlap["month_end_date_daily"])
    overlap["workbook_overlap_flag"] = overlap["security_id_workbook"].notna() & overlap["security_id_daily"].notna()
    overlap["daily_available_flag"] = overlap["security_id_daily"].notna()
    overlap["workbook_available_flag"] = overlap["security_id_workbook"].notna()
    overlap["workbook_valid_flag"] = _is_valid_monthly_row(
        overlap.rename(columns={"month_end_price_workbook": "month_end_price"})
    )
    overlap["post_workbook_priority_flag"] = overlap["month_end_date"].ge(workbook_priority_start)
    overlap["price_abs_diff"] = (overlap["month_end_price_workbook"] - overlap["month_end_price_daily"]).abs()
    overlap["market_cap_abs_diff"] = (
        overlap["month_end_market_cap_workbook"] - overlap["month_end_market_cap_daily"]
    ).abs()
    overlap["return_abs_diff"] = (
        overlap["monthly_total_return_workbook"] - overlap["monthly_total_return_daily"]
    ).abs()
    overlap["resolved_source_name"] = np.where(
        overlap["post_workbook_priority_flag"] & overlap["workbook_valid_flag"],
        "monthly_workbook",
        np.where(overlap["daily_available_flag"], "daily_derived", pd.NA),
    )
    overlap["resolution_reason"] = np.select(
        [
            overlap["post_workbook_priority_flag"] & overlap["workbook_valid_flag"],
            overlap["post_workbook_priority_flag"]
            & overlap["workbook_available_flag"]
            & ~overlap["workbook_valid_flag"]
            & overlap["daily_available_flag"],
            overlap["post_workbook_priority_flag"] & ~overlap["workbook_available_flag"] & overlap["daily_available_flag"],
            ~overlap["post_workbook_priority_flag"] & overlap["daily_available_flag"],
        ],
        [
            "post_2007_workbook_primary",
            "post_2007_daily_fallback_invalid_workbook",
            "post_2007_daily_fallback_missing_workbook",
            "pre_2007_daily_only",
        ],
        default=pd.NA,
    )

    workbook_keyed = workbook_source.set_index(["security_id", "month_end_date"]).sort_index()
    daily_keyed = daily_source.set_index(["security_id", "month_end_date"]).sort_index()
    union_index = workbook_keyed.index.union(daily_keyed.index).sort_values()
    resolved = daily_keyed.reindex(union_index).copy()
    workbook_reindexed = workbook_keyed.reindex(union_index)
    month_index = pd.Series(union_index.get_level_values("month_end_date"), index=union_index)
    prefer_workbook = month_index.ge(workbook_priority_start) & _is_valid_monthly_row(workbook_reindexed)
    if prefer_workbook.any():
        for column in MONTHLY_SOURCE_COLUMNS:
            if column in workbook_reindexed.columns:
                selected = workbook_reindexed.loc[prefer_workbook, column]
                if selected.empty:
                    continue
                selected = selected.where(selected.notna(), resolved.loc[prefer_workbook, column])
                try:
                    selected = selected.astype(resolved[column].dtype)
                except (TypeError, ValueError):
                    pass
                resolved.loc[prefer_workbook, column] = selected.to_numpy()
    resolved = resolved.reset_index()
    resolved = resolved.loc[resolved["source_name"].notna()].copy()

    diagnostics = overlap[
        [
            "security_id",
            "month_end_date",
            "workbook_overlap_flag",
            "daily_available_flag",
            "workbook_available_flag",
            "workbook_valid_flag",
            "post_workbook_priority_flag",
            "resolved_source_name",
            "resolution_reason",
            "price_abs_diff",
            "market_cap_abs_diff",
            "return_abs_diff",
        ]
    ]
    resolved = resolved.merge(diagnostics, on=["security_id", "month_end_date"], how="left")
    return source_table.sort_values(["security_id", "month_end_date"]).reset_index(drop=True), resolved.sort_values(
        ["security_id", "month_end_date"]
    ).reset_index(drop=True)


def derive_monthly_market_from_daily(daily_prices: pd.DataFrame) -> pd.DataFrame:
    return derive_monthly_from_daily(daily_prices)
