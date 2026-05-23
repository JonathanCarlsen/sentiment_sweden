from __future__ import annotations

import numpy as np
import pandas as pd


MARKET_CAP_FULL_TO_MILLIONS = 1_000_000.0

LIQUIDITY_DIVIDEND_MONTHLY_COLUMNS = [
    "security_id",
    "company_id",
    "country_code",
    "month_end_date",
    "ILLIQ_raw",
    "ILLIQ",
    "XTURN_raw",
    "XTURN",
    "D_PAYER_daily",
    "D_PAYER",
    "n_daily_obs",
    "n_illiq_obs",
    "monthly_trading_volume",
    "month_end_shares_outstanding_liquidity",
    "monthly_traded_value_millions",
    "daily_dividend_observed_flag",
    "liquidity_dividend_status",
]

DAILY_DIVIDEND_COLUMNS = [
    "div",
    "divd",
    "divdgross",
    "divdnet",
    "divgross",
    "divnet",
    "divrc",
    "divrcgross",
    "divrcnet",
    "divsp",
    "divspgross",
    "divspnet",
]


def _empty_liquidity_dividend_monthly() -> pd.DataFrame:
    return pd.DataFrame(columns=LIQUIDITY_DIVIDEND_MONTHLY_COLUMNS)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    values = frame[column]
    if values.dtype == object or str(values.dtype).startswith("string"):
        values = values.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(values, errors="coerce")


def _last_non_null(series: pd.Series) -> object:
    non_null = series.dropna()
    return non_null.iloc[-1] if not non_null.empty else pd.NA


def _sum_min_count(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").sum(min_count=1)
    return float(value) if not pd.isna(value) else np.nan


def build_sweden_liquidity_dividend_monthly(
    cleaned_daily_prices: pd.DataFrame,
    *,
    min_illiq_obs: int = 5,
) -> pd.DataFrame:
    """Build monthly security-level liquidity and dividend-payer characteristics.

    `ILLIQ` follows a monthly Amihud-style construction using cleaned daily
    returns and daily traded value in millions. `D_PAYER` is a trailing
    12-month indicator based on observed daily cash-distribution fields.
    """

    if cleaned_daily_prices is None or cleaned_daily_prices.empty:
        return _empty_liquidity_dividend_monthly()
    if "security_id" not in cleaned_daily_prices.columns or "trade_date" not in cleaned_daily_prices.columns:
        return _empty_liquidity_dividend_monthly()

    frame = cleaned_daily_prices.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    if "month_end_date" not in frame.columns:
        frame["month_end_date"] = frame["trade_date"].dt.to_period("M").dt.to_timestamp("M")
    else:
        frame["month_end_date"] = pd.to_datetime(frame["month_end_date"], errors="coerce")
    frame = frame.dropna(subset=["security_id", "trade_date", "month_end_date"]).sort_values(
        ["security_id", "trade_date"]
    )
    if frame.empty:
        return _empty_liquidity_dividend_monthly()

    return_source = "daily_total_return_clean" if "daily_total_return_clean" in frame.columns else "daily_total_return"
    frame["_daily_return_for_illiq"] = _numeric(frame, return_source)
    frame["_price"] = _numeric(frame, "prccd").abs()
    frame["_shares_traded"] = _numeric(frame, "cshtrd")
    frame["_shares_outstanding"] = _numeric(frame, "cshoc")
    frame["_daily_traded_value_millions"] = frame["_price"] * frame["_shares_traded"] / MARKET_CAP_FULL_TO_MILLIONS
    valid_illiq = (
        frame["_daily_return_for_illiq"].notna()
        & frame["_daily_traded_value_millions"].notna()
        & frame["_daily_traded_value_millions"].gt(0)
    )
    frame["_illiq_component"] = np.where(
        valid_illiq,
        frame["_daily_return_for_illiq"].abs() / frame["_daily_traded_value_millions"],
        np.nan,
    )

    dividend_candidates = [column for column in DAILY_DIVIDEND_COLUMNS if column in frame.columns]
    if dividend_candidates:
        dividend_values = pd.concat([_numeric(frame, column) for column in dividend_candidates], axis=1)
        frame["_daily_dividend_positive"] = dividend_values.gt(0).any(axis=1)
        frame["_daily_dividend_observed"] = dividend_values.notna().any(axis=1)
    else:
        frame["_daily_dividend_positive"] = False
        frame["_daily_dividend_observed"] = False

    group_columns = ["security_id", "month_end_date"]
    if "company_id" not in frame.columns:
        frame["company_id"] = pd.NA
    if "country_code" not in frame.columns:
        frame["country_code"] = pd.NA

    monthly = (
        frame.groupby(group_columns, as_index=False)
        .agg(
            company_id=("company_id", _last_non_null),
            country_code=("country_code", _last_non_null),
            ILLIQ_raw=("_illiq_component", "mean"),
            n_daily_obs=("trade_date", "size"),
            n_illiq_obs=("_illiq_component", "count"),
            monthly_trading_volume=("_shares_traded", _sum_min_count),
            month_end_shares_outstanding_liquidity=("_shares_outstanding", _last_non_null),
            monthly_traded_value_millions=("_daily_traded_value_millions", _sum_min_count),
            monthly_dividend_positive=("_daily_dividend_positive", "max"),
            daily_dividend_observed_flag=("_daily_dividend_observed", "max"),
        )
        .sort_values(["security_id", "month_end_date"])
        .reset_index(drop=True)
    )
    monthly["ILLIQ"] = monthly["ILLIQ_raw"].where(monthly["n_illiq_obs"].ge(int(min_illiq_obs)))
    shares = pd.to_numeric(monthly["month_end_shares_outstanding_liquidity"], errors="coerce")
    volume = pd.to_numeric(monthly["monthly_trading_volume"], errors="coerce")
    monthly["XTURN_raw"] = np.where(shares.notna() & shares.gt(0) & volume.notna(), volume / shares, np.nan)
    monthly["XTURN"] = monthly["XTURN_raw"]

    dividend_monthly = monthly[
        ["security_id", "month_end_date", "monthly_dividend_positive", "daily_dividend_observed_flag"]
    ].copy()
    dividend_monthly["monthly_dividend_positive"] = dividend_monthly["monthly_dividend_positive"].fillna(False).astype(int)
    dividend_monthly["monthly_dividend_observed"] = (
        dividend_monthly["daily_dividend_observed_flag"].fillna(False).astype(int)
    )
    rolling_positive = (
        dividend_monthly.groupby("security_id")["monthly_dividend_positive"]
        .rolling(window=12, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .astype("float64")
    )
    rolling_observed = (
        dividend_monthly.groupby("security_id")["monthly_dividend_observed"]
        .rolling(window=12, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .astype("float64")
    )
    dividend_monthly["D_PAYER_daily"] = rolling_positive.where(rolling_observed.eq(1.0), np.nan)
    monthly = monthly.merge(
        dividend_monthly[["security_id", "month_end_date", "D_PAYER_daily"]],
        on=["security_id", "month_end_date"],
        how="left",
    )
    monthly["D_PAYER"] = monthly["D_PAYER_daily"]
    monthly["liquidity_dividend_status"] = np.select(
        [
            monthly["ILLIQ"].notna() & monthly["XTURN"].notna(),
            monthly["ILLIQ"].isna() & monthly["XTURN"].notna(),
            monthly["ILLIQ"].notna() & monthly["XTURN"].isna(),
        ],
        ["ok", "missing_illiq", "missing_xturn"],
        default="missing_illiq_and_xturn",
    )
    monthly["daily_dividend_observed_flag"] = monthly["daily_dividend_observed_flag"].fillna(False).astype(bool)
    return monthly.reindex(columns=LIQUIDITY_DIVIDEND_MONTHLY_COLUMNS)
