from __future__ import annotations

import numpy as np
import pandas as pd


RETURN_SANITY_AUDIT_COLUMNS = [
    "country_code",
    "month_end_date",
    "source_name",
    "return_sanity_status",
    "return_sanity_reason",
    "rows",
    "raw_return_non_null",
    "clean_return_non_null",
    "adjusted_split_like_count",
    "missing_ambiguous_extreme_count",
    "raw_return_mean",
    "clean_return_mean",
    "raw_return_min",
    "raw_return_max",
    "clean_return_min",
    "clean_return_max",
]

DAILY_RETURN_SANITY_AUDIT_COLUMNS = [
    "country_code",
    "month_end_date",
    "daily_return_sanity_status",
    "daily_return_sanity_reason",
    "rows",
    "securities",
    "raw_daily_return_non_null",
    "clean_daily_return_non_null",
    "adjusted_split_like_count",
    "missing_ambiguous_extreme_count",
    "raw_daily_return_mean",
    "clean_daily_return_mean",
    "raw_daily_return_min",
    "raw_daily_return_max",
    "clean_daily_return_min",
    "clean_daily_return_max",
]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    valid = numerator.notna() & denominator.notna() & denominator.gt(0)
    return numerator.where(valid) / denominator.where(valid)


def _empty_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=RETURN_SANITY_AUDIT_COLUMNS)


def _empty_daily_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=DAILY_RETURN_SANITY_AUDIT_COLUMNS)


def build_return_sanity_audit(cleaned_monthly: pd.DataFrame) -> pd.DataFrame:
    """Summarize monthly return-cleaning outcomes for audit output."""

    if cleaned_monthly is None or cleaned_monthly.empty:
        return _empty_audit()

    frame = cleaned_monthly.copy()
    if "country_code" not in frame.columns:
        frame["country_code"] = pd.NA
    if "source_name" not in frame.columns:
        frame["source_name"] = pd.NA
    if "return_sanity_status" not in frame.columns:
        frame["return_sanity_status"] = pd.NA
    if "return_sanity_reason" not in frame.columns:
        frame["return_sanity_reason"] = pd.NA
    frame["month_end_date"] = pd.to_datetime(frame.get("month_end_date"), errors="coerce")

    rows: list[dict[str, object]] = []
    group_cols = ["country_code", "month_end_date", "source_name", "return_sanity_status", "return_sanity_reason"]
    for key, group in frame.groupby(group_cols, dropna=False):
        country_code, month_end_date, source_name, status, reason = key
        raw_return = _numeric(group, "monthly_total_return_raw")
        clean_return = _numeric(group, "monthly_total_return_clean")
        rows.append(
            {
                "country_code": country_code,
                "month_end_date": month_end_date,
                "source_name": source_name,
                "return_sanity_status": status,
                "return_sanity_reason": reason,
                "rows": len(group),
                "raw_return_non_null": int(raw_return.notna().sum()),
                "clean_return_non_null": int(clean_return.notna().sum()),
                "adjusted_split_like_count": int(pd.Series(status, index=group.index).eq("adjusted_split_like").sum()),
                "missing_ambiguous_extreme_count": int(
                    pd.Series(status, index=group.index).eq("missing_ambiguous_extreme").sum()
                ),
                "raw_return_mean": raw_return.mean(),
                "clean_return_mean": clean_return.mean(),
                "raw_return_min": raw_return.min(),
                "raw_return_max": raw_return.max(),
                "clean_return_min": clean_return.min(),
                "clean_return_max": clean_return.max(),
            }
        )
    return pd.DataFrame(rows).reindex(columns=RETURN_SANITY_AUDIT_COLUMNS)


def build_daily_return_sanity_audit(cleaned_daily: pd.DataFrame) -> pd.DataFrame:
    if cleaned_daily is None or cleaned_daily.empty:
        return _empty_daily_audit()

    frame = cleaned_daily.copy()
    if "country_code" not in frame.columns:
        frame["country_code"] = pd.NA
    if "daily_return_sanity_status" not in frame.columns:
        frame["daily_return_sanity_status"] = pd.NA
    if "daily_return_sanity_reason" not in frame.columns:
        frame["daily_return_sanity_reason"] = pd.NA
    if "month_end_date" not in frame.columns and "trade_date" in frame.columns:
        frame["month_end_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    frame["month_end_date"] = pd.to_datetime(frame.get("month_end_date"), errors="coerce")

    rows: list[dict[str, object]] = []
    group_cols = ["country_code", "month_end_date", "daily_return_sanity_status", "daily_return_sanity_reason"]
    for key, group in frame.groupby(group_cols, dropna=False):
        country_code, month_end_date, status, reason = key
        raw_return = _numeric(group, "daily_total_return_raw")
        clean_return = _numeric(group, "daily_total_return_clean")
        rows.append(
            {
                "country_code": country_code,
                "month_end_date": month_end_date,
                "daily_return_sanity_status": status,
                "daily_return_sanity_reason": reason,
                "rows": len(group),
                "securities": int(group["security_id"].nunique()) if "security_id" in group.columns else 0,
                "raw_daily_return_non_null": int(raw_return.notna().sum()),
                "clean_daily_return_non_null": int(clean_return.notna().sum()),
                "adjusted_split_like_count": int(pd.Series(status, index=group.index).eq("adjusted_split_like").sum()),
                "missing_ambiguous_extreme_count": int(
                    pd.Series(status, index=group.index).eq("missing_ambiguous_extreme").sum()
                ),
                "raw_daily_return_mean": raw_return.mean(),
                "clean_daily_return_mean": clean_return.mean(),
                "raw_daily_return_min": raw_return.min(),
                "raw_daily_return_max": raw_return.max(),
                "clean_daily_return_min": clean_return.min(),
                "clean_daily_return_max": clean_return.max(),
            }
        )
    return pd.DataFrame(rows).reindex(columns=DAILY_RETURN_SANITY_AUDIT_COLUMNS)


def clean_sweden_monthly_returns(
    market_monthly: pd.DataFrame,
    *,
    high_return_threshold: float = 5.0,
    low_return_threshold: float = -0.95,
    share_change_threshold: float = 0.50,
    log_price_share_tolerance: float = float(np.log(1.25)),
    total_return_factor_change_threshold: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean obvious unadjusted split-like monthly returns in the Sweden panel.

    The function preserves the original Compustat-derived return in
    `monthly_total_return_raw`, writes the production return to
    `monthly_total_return_clean`, and aliases `monthly_total_return` to the
    clean value for existing downstream transforms.
    """

    if market_monthly is None or market_monthly.empty:
        empty = pd.DataFrame(market_monthly).copy()
        return empty, _empty_audit()

    required = {"security_id", "month_end_date", "monthly_total_return"}
    if not required.issubset(market_monthly.columns):
        out = market_monthly.copy()
        out["monthly_total_return_raw"] = _numeric(out, "monthly_total_return")
        out["monthly_total_return_share_adjusted"] = np.nan
        out["monthly_total_return_clean"] = out["monthly_total_return_raw"]
        out["return_sanity_status"] = "ok"
        out["return_sanity_reason"] = "return_cleaning_skipped_missing_required_columns"
        return out, build_return_sanity_audit(out)

    out = market_monthly.copy()
    out["month_end_date"] = pd.to_datetime(out["month_end_date"], errors="coerce")
    out = out.sort_values(["security_id", "month_end_date"]).copy()

    price = _numeric(out, "month_end_price")
    shares = _numeric(out, "month_end_shares_outstanding")
    tr_factor = _numeric(out, "month_end_total_return_factor")
    market_cap = _numeric(out, "month_end_market_cap")
    raw_return = _numeric(out, "monthly_total_return")

    group = out["security_id"]
    lag_price = price.groupby(group).shift(1)
    lag_shares = shares.groupby(group).shift(1)
    lag_tr_factor = tr_factor.groupby(group).shift(1)
    lag_market_cap = market_cap.groupby(group).shift(1)

    price_ratio = _safe_ratio(price, lag_price)
    share_ratio = _safe_ratio(shares, lag_shares)
    tr_factor_ratio = _safe_ratio(tr_factor, lag_tr_factor)
    market_cap_ratio = _safe_ratio(market_cap, lag_market_cap)

    raw_price_return = price_ratio - 1.0
    shares_change = share_ratio - 1.0
    tr_factor_change = tr_factor_ratio - 1.0
    market_cap_return = market_cap_ratio - 1.0
    share_adjusted_return = (price_ratio * share_ratio) - 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        log_price_share_offset = np.log(price_ratio) + np.log(share_ratio)

    high_extreme = raw_return.ge(high_return_threshold)
    low_extreme = raw_return.le(low_return_threshold)
    missing_leg = raw_return.isna() | price_ratio.isna() | share_ratio.isna()
    candidate_extreme = ~missing_leg & (high_extreme | low_extreme)
    raw_price_same_direction = (high_extreme & raw_price_return.ge(high_return_threshold)) | (
        low_extreme & raw_price_return.le(low_return_threshold)
    )
    shares_opposite_direction = (high_extreme & shares_change.le(-share_change_threshold)) | (
        low_extreme & shares_change.ge(share_change_threshold)
    )
    close_price_share_offset = pd.Series(log_price_share_offset, index=out.index).abs().le(log_price_share_tolerance)
    tr_factor_insufficient = tr_factor_change.isna() | tr_factor_change.abs().le(total_return_factor_change_threshold)
    split_like = (
        candidate_extreme
        & raw_price_same_direction
        & shares_opposite_direction
        & close_price_share_offset
        & tr_factor_insufficient
        & share_adjusted_return.notna()
    )
    ambiguous_extreme = candidate_extreme & ~split_like

    clean_return = raw_return.copy()
    clean_return.loc[split_like] = share_adjusted_return.loc[split_like]
    clean_return.loc[ambiguous_extreme] = np.nan
    clean_return.loc[missing_leg] = np.nan

    status = pd.Series("ok", index=out.index, dtype="string")
    status.loc[missing_leg] = "missing_leg"
    status.loc[split_like] = "adjusted_split_like"
    status.loc[ambiguous_extreme] = "missing_ambiguous_extreme"

    reasons: list[str] = []
    for idx in out.index:
        row_reasons: list[str] = []
        if bool(missing_leg.loc[idx]):
            row_reasons.append("missing_return_leg")
        if bool(candidate_extreme.loc[idx]):
            row_reasons.append("strict_extreme_return")
            if bool(raw_price_same_direction.loc[idx]):
                row_reasons.append("raw_price_extreme_same_direction")
            if bool(shares_opposite_direction.loc[idx]):
                row_reasons.append("shares_move_opposite_extreme_price")
            if bool(close_price_share_offset.loc[idx]):
                row_reasons.append("price_share_log_offset_within_tolerance")
            if bool(tr_factor_insufficient.loc[idx]):
                row_reasons.append("total_return_factor_did_not_explain_event")
        if bool(split_like.loc[idx]):
            row_reasons.append("share_adjusted_return_used")
        if bool(ambiguous_extreme.loc[idx]):
            row_reasons.append("clean_return_set_missing")
        reasons.append(";".join(row_reasons) if row_reasons else "regular_return")

    out["monthly_total_return_raw"] = raw_return
    out["lag_month_end_price"] = lag_price
    out["lag_month_end_shares_outstanding"] = lag_shares
    out["lag_month_end_total_return_factor"] = lag_tr_factor
    out["lag_month_end_market_cap"] = lag_market_cap
    out["raw_price_return"] = raw_price_return
    out["shares_outstanding_change"] = shares_change
    out["total_return_factor_change"] = tr_factor_change
    out["market_cap_return"] = market_cap_return
    out["price_share_log_offset"] = log_price_share_offset
    out["monthly_total_return_share_adjusted"] = share_adjusted_return.where(split_like)
    out["monthly_total_return_clean"] = clean_return
    out["monthly_total_return"] = clean_return
    out["return_sanity_status"] = status
    out["return_sanity_reason"] = reasons

    return out.reset_index(drop=True), build_return_sanity_audit(out)


def clean_sweden_daily_returns(
    daily_prices: pd.DataFrame,
    *,
    high_return_threshold: float = 5.0,
    low_return_threshold: float = -0.95,
    share_change_threshold: float = 0.50,
    log_price_share_tolerance: float = float(np.log(1.25)),
    total_return_factor_change_threshold: float = 0.50,
) -> pd.DataFrame:
    """Clean obvious unadjusted split-like daily returns for factors and IVOL."""

    if daily_prices is None or daily_prices.empty:
        return pd.DataFrame(daily_prices).copy()

    required = {"security_id", "trade_date", "daily_total_return", "prccd", "cshoc"}
    if not required.issubset(daily_prices.columns):
        out = daily_prices.copy()
        out["daily_total_return_raw"] = _numeric(out, "daily_total_return")
        out["daily_total_return_share_adjusted"] = np.nan
        out["daily_total_return_clean"] = out["daily_total_return_raw"]
        out["daily_total_return"] = out["daily_total_return_clean"]
        out["daily_return_sanity_status"] = "ok"
        out["daily_return_sanity_reason"] = "daily_return_cleaning_skipped_missing_required_columns"
        return out

    out = daily_prices.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    if "month_end_date" not in out.columns:
        out["month_end_date"] = out["trade_date"].dt.to_period("M").dt.to_timestamp("M")
    out = out.sort_values(["security_id", "trade_date"]).copy()

    price = _numeric(out, "prccd")
    shares = _numeric(out, "cshoc")
    tr_factor = _numeric(out, "trfd")
    raw_return = _numeric(out, "daily_total_return")

    group = out["security_id"]
    lag_price = price.groupby(group).shift(1)
    lag_shares = shares.groupby(group).shift(1)
    lag_tr_factor = tr_factor.groupby(group).shift(1)

    price_ratio = _safe_ratio(price, lag_price)
    share_ratio = _safe_ratio(shares, lag_shares)
    tr_factor_ratio = _safe_ratio(tr_factor, lag_tr_factor)

    raw_price_return = price_ratio - 1.0
    shares_change = share_ratio - 1.0
    tr_factor_change = tr_factor_ratio - 1.0
    share_adjusted_return = (price_ratio * share_ratio) - 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        log_price_share_offset = np.log(price_ratio) + np.log(share_ratio)

    high_extreme = raw_return.ge(high_return_threshold)
    low_extreme = raw_return.le(low_return_threshold)
    missing_leg = raw_return.isna() | price_ratio.isna() | share_ratio.isna()
    candidate_extreme = ~missing_leg & (high_extreme | low_extreme)
    raw_price_same_direction = (high_extreme & raw_price_return.ge(high_return_threshold)) | (
        low_extreme & raw_price_return.le(low_return_threshold)
    )
    shares_opposite_direction = (high_extreme & shares_change.le(-share_change_threshold)) | (
        low_extreme & shares_change.ge(share_change_threshold)
    )
    close_price_share_offset = pd.Series(log_price_share_offset, index=out.index).abs().le(log_price_share_tolerance)
    tr_factor_insufficient = tr_factor_change.isna() | tr_factor_change.abs().le(total_return_factor_change_threshold)
    split_like = (
        candidate_extreme
        & raw_price_same_direction
        & shares_opposite_direction
        & close_price_share_offset
        & tr_factor_insufficient
        & share_adjusted_return.notna()
    )
    ambiguous_extreme = candidate_extreme & ~split_like

    clean_return = raw_return.copy()
    clean_return.loc[split_like] = share_adjusted_return.loc[split_like]
    clean_return.loc[ambiguous_extreme] = np.nan
    clean_return.loc[missing_leg] = np.nan

    status = pd.Series("ok", index=out.index, dtype="string")
    status.loc[missing_leg] = "missing_leg"
    status.loc[split_like] = "adjusted_split_like"
    status.loc[ambiguous_extreme] = "missing_ambiguous_extreme"

    reasons: list[str] = []
    for idx in out.index:
        row_reasons: list[str] = []
        if bool(missing_leg.loc[idx]):
            row_reasons.append("missing_return_leg")
        if bool(candidate_extreme.loc[idx]):
            row_reasons.append("strict_extreme_return")
            if bool(raw_price_same_direction.loc[idx]):
                row_reasons.append("raw_price_extreme_same_direction")
            if bool(shares_opposite_direction.loc[idx]):
                row_reasons.append("shares_move_opposite_extreme_price")
            if bool(close_price_share_offset.loc[idx]):
                row_reasons.append("price_share_log_offset_within_tolerance")
            if bool(tr_factor_insufficient.loc[idx]):
                row_reasons.append("total_return_factor_did_not_explain_event")
        if bool(split_like.loc[idx]):
            row_reasons.append("share_adjusted_return_used")
        if bool(ambiguous_extreme.loc[idx]):
            row_reasons.append("clean_return_set_missing")
        reasons.append(";".join(row_reasons) if row_reasons else "regular_return")

    out["daily_total_return_raw"] = raw_return
    out["lag_daily_price"] = lag_price
    out["lag_daily_shares_outstanding"] = lag_shares
    out["lag_daily_total_return_factor"] = lag_tr_factor
    out["daily_raw_price_return"] = raw_price_return
    out["daily_shares_outstanding_change"] = shares_change
    out["daily_total_return_factor_change"] = tr_factor_change
    out["daily_price_share_log_offset"] = log_price_share_offset
    out["daily_total_return_share_adjusted"] = share_adjusted_return.where(split_like)
    out["daily_total_return_clean"] = clean_return
    out["daily_total_return"] = clean_return
    out["daily_return_sanity_status"] = status
    out["daily_return_sanity_reason"] = reasons
    return out.reset_index(drop=True)
