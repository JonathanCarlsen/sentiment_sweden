from __future__ import annotations

import numpy as np
import pandas as pd

from nordic_sentiment.market.returns import clean_sweden_daily_returns

MARKET_CAP_FULL_TO_MILLIONS = 1_000_000.0


FACTOR_DAILY_COLUMNS = [
    "country_code",
    "trade_date",
    "factor_code",
    "factor_value",
    "source_name",
    "estimation_status",
]

IVOL_MONTHLY_COLUMNS = [
    "security_id",
    "company_id",
    "country_code",
    "month_end_date",
    "ivol_ff3",
    "n_daily_obs",
    "ivol_ff3_status",
]


def _empty_factor_daily() -> pd.DataFrame:
    return pd.DataFrame(columns=FACTOR_DAILY_COLUMNS)


def _empty_ivol_monthly() -> pd.DataFrame:
    return pd.DataFrame(columns=IVOL_MONTHLY_COLUMNS)


def _common_equity_mask(frame: pd.DataFrame) -> pd.Series:
    if "issue_type_code" in frame.columns:
        codes = frame["issue_type_code"].astype("string").str.strip().str.upper()
    elif "tpci" in frame.columns:
        codes = frame["tpci"].astype("string").str.strip().str.upper()
    else:
        return pd.Series(True, index=frame.index)
    return codes.isna() | codes.eq("0") | codes.eq("EQ") | codes.eq("COM")


def _assign_three_way_bucket(series: pd.Series) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    if valid.nunique() < 3:
        return out
    low_break = float(valid.quantile(0.3))
    high_break = float(valid.quantile(0.7))
    if not np.isfinite(low_break) or not np.isfinite(high_break) or low_break >= high_break:
        return out
    numeric = pd.to_numeric(series, errors="coerce")
    out.loc[numeric.le(low_break)] = "low"
    out.loc[numeric.gt(high_break)] = "high"
    out.loc[numeric.gt(low_break) & numeric.le(high_break)] = "mid"
    return out


def _weighted_return(frame: pd.DataFrame, return_column: str, weight_column: str) -> float:
    returns = pd.to_numeric(frame.get(return_column), errors="coerce")
    weights = pd.to_numeric(frame.get(weight_column), errors="coerce")
    valid = returns.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(returns.loc[valid], weights=weights.loc[valid]))


def _standardize_daily_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    if daily_prices is None or daily_prices.empty:
        return pd.DataFrame()
    frame = daily_prices.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["month_end_date"] = frame["trade_date"].dt.to_period("M").dt.to_timestamp("M")
    frame["prccd"] = pd.to_numeric(frame.get("prccd"), errors="coerce")
    frame["cshoc"] = pd.to_numeric(frame.get("cshoc"), errors="coerce")
    frame["trfd"] = pd.to_numeric(frame.get("trfd"), errors="coerce")
    frame = frame.dropna(subset=["security_id", "trade_date", "month_end_date"])
    frame = frame.sort_values(["security_id", "trade_date"]).reset_index(drop=True)
    frame["adjusted_price"] = frame["prccd"] * frame["trfd"]
    frame["prev_adjusted_price"] = frame.groupby("security_id")["adjusted_price"].shift(1)
    valid = frame["adjusted_price"].notna() & frame["prev_adjusted_price"].notna() & frame["prev_adjusted_price"].ne(0)
    frame["daily_total_return"] = np.where(valid, (frame["adjusted_price"] / frame["prev_adjusted_price"]) - 1.0, np.nan)
    frame["market_cap"] = frame["prccd"] * frame["cshoc"]
    frame["lagged_daily_me"] = frame.groupby("security_id")["market_cap"].shift(1)
    frame["common_equity_flag"] = _common_equity_mask(frame)
    return frame


def _build_monthly_bucket_map(
    market_monthly: pd.DataFrame,
    fundamental_snapshots: pd.DataFrame,
    *,
    country_code: str = "SWE",
) -> pd.DataFrame:
    if market_monthly is None or market_monthly.empty:
        return pd.DataFrame(columns=["security_id", "month_end_date", "size_bucket", "value_bucket"])

    market = market_monthly.copy()
    market["month_end_date"] = pd.to_datetime(market["month_end_date"], errors="coerce")
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
        snapshots["BE"] = pd.to_numeric(
            snapshots.get("BE", snapshots.get("book_equity")),
            errors="coerce",
        )
        snapshots = snapshots[["company_id", "month_end_date", "BE"]].drop_duplicates(
            ["company_id", "month_end_date"],
            keep="last",
        )

    panel = market.merge(snapshots, on=["company_id", "month_end_date"], how="left") if not snapshots.empty else market.copy()
    panel["country_code"] = panel.get("country_code", country_code).fillna(country_code)
    be = pd.to_numeric(panel.get("BE"), errors="coerce")
    me_millions = pd.to_numeric(panel.get("month_end_market_cap_millions"), errors="coerce")
    panel["current_be_me"] = np.where(
        be.notna() & be.gt(0) & me_millions.notna() & me_millions.gt(0),
        be / me_millions,
        np.nan,
    )
    panel["sort_me"] = panel["lagged_me"]
    panel["sort_be_me"] = panel.groupby("security_id")["current_be_me"].shift(1)

    rows: list[pd.DataFrame] = []
    for month_end_date, monthly_frame in panel.groupby("month_end_date", sort=True):
        eligible = monthly_frame.loc[_common_equity_mask(monthly_frame)].copy()
        eligible["size_bucket"] = _assign_three_way_bucket(eligible["sort_me"])
        eligible["value_bucket"] = _assign_three_way_bucket(eligible["sort_be_me"])
        rows.append(eligible[["security_id", "month_end_date", "size_bucket", "value_bucket"]])
    if not rows:
        return pd.DataFrame(columns=["security_id", "month_end_date", "size_bucket", "value_bucket"])
    return pd.concat(rows, ignore_index=True)


def build_sweden_factor_daily(
    daily_prices: pd.DataFrame,
    rates_monthly: pd.DataFrame,
    market_monthly: pd.DataFrame,
    fundamental_snapshots: pd.DataFrame,
    *,
    country_code: str = "SWE",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = clean_sweden_daily_returns(_standardize_daily_prices(daily_prices))
    if frame.empty:
        return _empty_factor_daily(), frame

    rates = pd.DataFrame(rates_monthly).copy()
    rates["month_end_date"] = pd.to_datetime(rates.get("month_end_date"), errors="coerce")
    rates["monthly_rate_proxy"] = pd.to_numeric(rates.get("monthly_rate_proxy"), errors="coerce")
    rf_monthly = rates.loc[rates["rate_code"] == "RF_3M_PROXY", ["month_end_date", "monthly_rate_proxy"]].drop_duplicates(
        "month_end_date",
        keep="last",
    )
    trading_days = frame.groupby("month_end_date")["trade_date"].nunique().rename("n_trading_days").reset_index()
    rf_map = trading_days.merge(rf_monthly, on="month_end_date", how="left")
    rf_map["rf_daily"] = np.where(
        rf_map["monthly_rate_proxy"].notna() & rf_map["n_trading_days"].gt(0),
        np.power(1.0 + rf_map["monthly_rate_proxy"], 1.0 / rf_map["n_trading_days"]) - 1.0,
        np.nan,
    )
    frame = frame.merge(rf_map[["month_end_date", "rf_daily"]], on="month_end_date", how="left")

    bucket_map = _build_monthly_bucket_map(market_monthly, fundamental_snapshots, country_code=country_code)
    frame = frame.merge(bucket_map, on=["security_id", "month_end_date"], how="left")

    factor_rows: list[dict[str, object]] = []
    eligible = frame.loc[frame["common_equity_flag"]].copy()
    for trade_date, daily_frame in eligible.groupby("trade_date", sort=True):
        month_end_date = pd.Timestamp(trade_date).to_period("M").to_timestamp("M")
        rf_daily = pd.to_numeric(daily_frame["rf_daily"], errors="coerce").dropna()
        rf_value = float(rf_daily.iloc[0]) if not rf_daily.empty else float("nan")
        market_return = _weighted_return(daily_frame, "daily_total_return", "lagged_daily_me")
        mkt_rf = market_return - rf_value if np.isfinite(market_return) and np.isfinite(rf_value) else float("nan")

        factor_rows.extend(
            [
                {
                    "country_code": country_code,
                    "trade_date": trade_date,
                    "factor_code": "RF_D",
                    "factor_value": rf_value,
                    "source_name": "3-month_interbank_rate_sweden.csv",
                    "estimation_status": "ok" if np.isfinite(rf_value) else "missing_factor_inputs",
                },
                {
                    "country_code": country_code,
                    "trade_date": trade_date,
                    "factor_code": "MKT_RF_D",
                    "factor_value": mkt_rf,
                    "source_name": "daily_prices_plus_monthly_3m_proxy",
                    "estimation_status": "ok" if np.isfinite(mkt_rf) else "missing_factor_inputs",
                },
            ]
        )

        size_small = daily_frame.loc[daily_frame["size_bucket"] == "low"]
        size_big = daily_frame.loc[daily_frame["size_bucket"] == "high"]
        smb = _weighted_return(size_small, "daily_total_return", "lagged_daily_me") - _weighted_return(
            size_big, "daily_total_return", "lagged_daily_me"
        )
        value_high = daily_frame.loc[daily_frame["value_bucket"] == "high"]
        value_low = daily_frame.loc[daily_frame["value_bucket"] == "low"]
        hml = _weighted_return(value_high, "daily_total_return", "lagged_daily_me") - _weighted_return(
            value_low, "daily_total_return", "lagged_daily_me"
        )
        factor_rows.extend(
            [
                {
                    "country_code": country_code,
                    "trade_date": trade_date,
                    "factor_code": "SMB_D",
                    "factor_value": smb,
                    "source_name": "daily_prices_plus_monthly_size_buckets",
                    "estimation_status": "ok" if np.isfinite(smb) else "missing_factor_inputs",
                },
                {
                    "country_code": country_code,
                    "trade_date": trade_date,
                    "factor_code": "HML_D",
                    "factor_value": hml,
                    "source_name": "daily_prices_plus_monthly_value_buckets",
                    "estimation_status": "ok" if np.isfinite(hml) else "missing_factor_inputs",
                },
            ]
        )

    factors = pd.DataFrame(factor_rows, columns=FACTOR_DAILY_COLUMNS)
    if factors.empty:
        return _empty_factor_daily(), frame
    factors = factors.drop_duplicates(["country_code", "trade_date", "factor_code"], keep="last")
    return factors.sort_values(["country_code", "trade_date", "factor_code"]).reset_index(drop=True), frame


def _estimate_ivol(
    frame: pd.DataFrame,
    y_column: str,
    factor_columns: list[str],
    *,
    min_obs: int,
) -> tuple[float, int, str]:
    work = frame.copy()
    for column in factor_columns:
        if column not in work.columns:
            work[column] = np.nan
    y = pd.to_numeric(work[y_column], errors="coerce")
    if y.notna().sum() == 0:
        return float("nan"), 0, "missing_security_returns"
    x_frame = work[factor_columns].apply(pd.to_numeric, errors="coerce")
    if x_frame.notna().all(axis=1).sum() == 0:
        return float("nan"), int(y.notna().sum()), "missing_factor_inputs"
    valid = pd.concat([y.rename(y_column), x_frame], axis=1).dropna()
    n_obs = int(valid.shape[0])
    if n_obs < min_obs:
        return float("nan"), n_obs, "insufficient_daily_obs"

    y_values = valid[y_column].to_numpy(dtype="float64")
    x = valid[factor_columns].to_numpy(dtype="float64")
    x = np.column_stack([np.ones(len(valid)), x])
    beta, _, _, _ = np.linalg.lstsq(x, y_values, rcond=None)
    resid = y_values - x @ beta
    if resid.size <= 1:
        return float("nan"), n_obs, "insufficient_daily_obs"
    return float(np.std(resid, ddof=1)), n_obs, "ok"


def build_sweden_ivol_monthly(
    daily_prices: pd.DataFrame,
    factor_daily: pd.DataFrame,
    *,
    country_code: str = "SWE",
    min_daily_obs: int = 15,
) -> pd.DataFrame:
    frame = clean_sweden_daily_returns(_standardize_daily_prices(daily_prices))
    if frame.empty or factor_daily is None or factor_daily.empty:
        return _empty_ivol_monthly()

    factors = factor_daily.pivot_table(
        index="trade_date",
        columns="factor_code",
        values="factor_value",
        aggfunc="last",
    ).reset_index()
    merged = frame.merge(factors, on="trade_date", how="left")
    merged["excess_return"] = merged["daily_total_return"] - merged["RF_D"]

    rows: list[dict[str, object]] = []
    for (security_id, month_end_date), monthly_frame in merged.groupby(["security_id", "month_end_date"], sort=True):
        base = monthly_frame[["company_id", "country_code"]].dropna().head(1)
        company_id = base["company_id"].iloc[0] if not base.empty else pd.NA
        row_country = base["country_code"].iloc[0] if not base.empty else country_code
        ivol_ff3, n_obs_ff3, status_ff3 = _estimate_ivol(
            monthly_frame,
            "excess_return",
            ["MKT_RF_D", "SMB_D", "HML_D"],
            min_obs=min_daily_obs,
        )
        rows.append(
            {
                "security_id": security_id,
                "company_id": company_id,
                "country_code": row_country,
                "month_end_date": month_end_date,
                "ivol_ff3": ivol_ff3,
                "n_daily_obs": n_obs_ff3,
                "ivol_ff3_status": status_ff3,
            }
        )
    out = pd.DataFrame(rows, columns=IVOL_MONTHLY_COLUMNS)
    if out.empty:
        return _empty_ivol_monthly()
    return out.sort_values(["security_id", "month_end_date"]).reset_index(drop=True)
