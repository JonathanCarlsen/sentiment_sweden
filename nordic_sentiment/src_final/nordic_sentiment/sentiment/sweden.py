from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from nordic_sentiment.market.returns import clean_sweden_daily_returns
from nordic_sentiment.sentiment.index import annualize_sentiment_index


PAPER_SPEC_COLUMNS = [
    "paper_proxy_name",
    "paper_definition",
    "frequency",
    "lag_rule",
    "transform_rule",
    "orthogonalized",
    "repo_source_candidate",
    "build_status",
    "required_for_final_index",
]

PROXY_SOURCE_COLUMNS = [
    "country_code",
    "month_end_date",
    "proxy_code",
    "raw_value",
    "paper_proxy_name",
    "source_name",
    "build_status",
    "exact_replication_flag",
]

IPO_RETURN_COLUMNS = [
    "country_code",
    "bloomberg_ticker",
    "company_name_bloomberg",
    "isin",
    "ipo_date",
    "ipo_month_end_date",
    "ipo_offer_price",
    "currency",
    "primary_exchange_name",
    "bloomberg_offer_to_first_open_return_pct",
    "bloomberg_offer_to_first_open_return",
    "first_close_date",
    "first_close_price",
    "offer_to_first_close_return",
    "ripo_return",
    "ripo_return_source",
    "days_after_ipo",
    "match_status",
    "daily_gvkey",
    "daily_iid",
    "daily_company_name",
]

MACRO_CONTROL_COLUMNS = ["country_code", "month_end_date", "CPI", "PPI", "IP", "EM"]
SIBLEY_MACRO_CONTROL_COLUMNS = [
    "UNEMP",
    "dCPI",
    "dCONS",
    "dIND",
    "TBILL",
    "TERM",
    "VWRETD",
    "MKTVOL",
    "PCTZERO",
]
BASE_SWEDEN_SENTIMENT_PROXIES = ["ESI", "CCI", "TURN", "NIPO", "RIPO", "ED_RATIO"]
DIVIDEND_PREMIUM_PROXY = "DIV_PREMIUM"
TURN_PROXY_HISTORY_START = pd.Timestamp("2000-01-31")

SENTIMENT_ANALYSIS_START = pd.Timestamp("2001-02-28")
SENTIMENT_ANALYSIS_END = pd.Timestamp("2025-12-31")
ESENT_WEIGHTS = {
    "ESI": 0.2352,
    "NIPO": 0.1337,
    "ED_RATIO": 0.0088,
    "CCI_L1": 0.1404,
    "TURN_L1": 0.2646,
    "RIPO_L1": 0.3433,
}

DIVIDEND_PREMIUM_DIAGNOSTIC_COLUMNS = [
    "country_code",
    "month_end_date",
    "eligible_rows",
    "payer_count",
    "nonpayer_count",
    "payer_market_cap",
    "nonpayer_market_cap",
    "payer_mb_value_weight",
    "nonpayer_mb_value_weight",
    "DIV_PREMIUM_VW",
    "raw_value",
    "min_group_count",
    "dividend_premium_status",
]


def empty_sweden_paper_spec_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=PAPER_SPEC_COLUMNS)


def empty_sweden_proxy_source() -> pd.DataFrame:
    return pd.DataFrame(columns=PROXY_SOURCE_COLUMNS)


def empty_sweden_ipo_return_table() -> pd.DataFrame:
    return pd.DataFrame(columns=IPO_RETURN_COLUMNS)


def empty_sweden_macro_controls() -> pd.DataFrame:
    return pd.DataFrame(columns=MACRO_CONTROL_COLUMNS)


def empty_sweden_sibley_macro_controls() -> pd.DataFrame:
    return pd.DataFrame(columns=["country_code", "month_end_date", *SIBLEY_MACRO_CONTROL_COLUMNS])


def default_sweden_paper_spec_audit() -> pd.DataFrame:
    rows = [
        {
            "paper_proxy_name": "ESI",
            "paper_definition": "Economic Sentiment Index growth rate.",
            "frequency": "monthly",
            "lag_rule": "Use contemporaneous ESI_t in the final ESent model.",
            "transform_rule": "Monthly growth rate of Economic Sentiment Index.",
            "orthogonalized": False,
            "repo_source_candidate": "Economic and consumer sentiment.xlsx:EUESSE Index  (L1)",
            "build_status": "exact",
            "required_for_final_index": True,
        },
        {
            "paper_proxy_name": "CCI",
            "paper_definition": "Consumer Confidence Index growth rate.",
            "frequency": "monthly",
            "lag_rule": "Use one-month lag CCI_t-1 in the final ESent model.",
            "transform_rule": "Monthly growth rate of Consumer Confidence Index.",
            "orthogonalized": False,
            "repo_source_candidate": "Economic and consumer sentiment.xlsx:SWETCI Index  (R1)",
            "build_status": "exact",
            "required_for_final_index": True,
        },
        {
            "paper_proxy_name": "TURN",
            "paper_definition": "Logarithmic market turnover for the Swedish stock market.",
            "frequency": "monthly",
            "lag_rule": "Use one-month lag TURN_t-1 in the final ESent model.",
            "transform_rule": "Log of monthly total trading volume divided by monthly total outstanding shares.",
            "orthogonalized": False,
            "repo_source_candidate": "Daily prices jan 2000 - dec 2025 zip limited (preferred).csv:cshtrd,cshoc",
            "build_status": "exact",
            "required_for_final_index": True,
        },
        {
            "paper_proxy_name": "NIPO",
            "paper_definition": "Monthly IPO number including IPO-owned companies registered in Sweden.",
            "frequency": "monthly",
            "lag_rule": "Use contemporaneous NIPO_t in the final ESent model.",
            "transform_rule": "Monthly IPO count.",
            "orthogonalized": False,
            "repo_source_candidate": "ipo_dates_and_offering_price_sweden.csv:IPO Dt",
            "build_status": "missing",
            "required_for_final_index": True,
        },
        {
            "paper_proxy_name": "RIPO",
            "paper_definition": "Monthly IPO return measured as first-day issuance yield.",
            "frequency": "monthly",
            "lag_rule": "Use one-month lag RIPO_t-1 in the final ESent model.",
            "transform_rule": "Monthly first-day IPO return in percent.",
            "orthogonalized": False,
            "repo_source_candidate": "ipo_dates_and_offering_price_sweden.csv + Daily prices jan 2000 - dec 2025 zip limited (preferred).csv",
            "build_status": "missing",
            "required_for_final_index": True,
        },
        {
            "paper_proxy_name": "ED_RATIO",
            "paper_definition": "Inverted OMX debt-to-equity ratio.",
            "frequency": "monthly",
            "lag_rule": "Use contemporaneous E/D-Ratio_t in the final ESent model.",
            "transform_rule": "Inverted OMX debt-to-equity ratio.",
            "orthogonalized": False,
            "repo_source_candidate": "Missing OMX debt-to-equity ratio time series",
            "build_status": "missing",
            "required_for_final_index": True,
        },
        {
            "paper_proxy_name": "CPI",
            "paper_definition": "Swedish consumer price index used for macro-adjusted sentiment residualization.",
            "frequency": "monthly",
            "lag_rule": "Contemporaneous in macro-adjustment regressions.",
            "transform_rule": "Min-max normalize before regressions.",
            "orthogonalized": True,
            "repo_source_candidate": "Macrocontrols/Consumer Price Index (CPI), total 2020=100. Month 1980M01 - 2026M03.csv",
            "build_status": "approx_only",
            "required_for_final_index": False,
        },
        {
            "paper_proxy_name": "PPI",
            "paper_definition": "Swedish producer price index used for macro-adjusted sentiment residualization.",
            "frequency": "monthly",
            "lag_rule": "Contemporaneous in macro-adjustment regressions.",
            "transform_rule": "Min-max normalize before regressions.",
            "orthogonalized": True,
            "repo_source_candidate": "Macrocontrols/Producer Price Index by market and products SPIN 2015, 2020=100. Month 1990M01 - 2026M03.csv",
            "build_status": "approx_only",
            "required_for_final_index": False,
        },
        {
            "paper_proxy_name": "IP",
            "paper_definition": "Swedish industrial production index used for macro-adjusted sentiment residualization.",
            "frequency": "monthly",
            "lag_rule": "Contemporaneous in macro-adjustment regressions.",
            "transform_rule": "Min-max normalize before regressions.",
            "orthogonalized": True,
            "repo_source_candidate": "Macrocontrols/Industrial production index. Chain index, 2021=100, by industral classification NACE Rev.2. Monthly 2000M01 - 2026M02.csv",
            "build_status": "approx_only",
            "required_for_final_index": False,
        },
        {
            "paper_proxy_name": "EM",
            "paper_definition": "Swedish employment rate used for macro-adjusted sentiment residualization.",
            "frequency": "monthly",
            "lag_rule": "Contemporaneous in macro-adjustment regressions.",
            "transform_rule": "Min-max normalize before regressions.",
            "orthogonalized": True,
            "repo_source_candidate": "Macrocontrols/Population aged 15-74 (LFS) by labour status, type of data, sex and age. Month 2001M01 - 2026M03.csv",
            "build_status": "approx_only",
            "required_for_final_index": False,
        },
    ]
    return pd.DataFrame(rows, columns=PAPER_SPEC_COLUMNS)


def _paper_spec_status_map(paper_spec: pd.DataFrame | None) -> dict[str, str]:
    audit = default_sweden_paper_spec_audit() if paper_spec is None or paper_spec.empty else paper_spec.copy()
    return {
        str(row.paper_proxy_name): str(row.build_status)
        for row in audit.itertuples(index=False)
        if pd.notna(row.paper_proxy_name)
    }


def _sentiment_analysis_filter(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["month_end_date"] = pd.to_datetime(out["month_end_date"], errors="coerce")
    return out.loc[out["month_end_date"].between(SENTIMENT_ANALYSIS_START, SENTIMENT_ANALYSIS_END)].copy()


def _sentiment_analysis_month_calendar(country_code: str = "SWE") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country_code": country_code,
            "month_end_date": pd.date_range(SENTIMENT_ANALYSIS_START, SENTIMENT_ANALYSIS_END, freq="ME"),
        }
    )


def _monthly_growth(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.div(values.shift(1)).sub(1.0)


def _log_growth(series: pd.Series, periods: int = 1) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    positive = values.where(values.gt(0))
    return np.log(positive).diff(periods)


def _clean_column_name(column: object) -> str:
    return (
        str(column)
        .strip()
        .lower()
        .replace("%", "pct")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _ed_ratio_source_column(frame: pd.DataFrame) -> str | None:
    if frame is None or frame.empty:
        return None
    explicit_candidates = {
        "omx30_total_debt_to_equity",
        "omx30_debt_to_equity",
        "omx30_total_debt_to_total_equity",
        # The current workbook column carries this label, but the series is
        # user-supplied as OMX30 debt-to-equity for the sentiment proxy.
        "omx30_total_debt_to_total_assets",
    }
    normalized = {_clean_column_name(column): column for column in frame.columns}
    for candidate in explicit_candidates:
        if candidate in normalized:
            return normalized[candidate]
    for normalized_name, original_name in normalized.items():
        if "omx30" in normalized_name and "debt" in normalized_name and "equity" in normalized_name:
            return original_name
    return None


def _invert_debt_to_equity(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    positive = values.where(values.gt(0))
    median_abs = positive.abs().median(skipna=True)
    if pd.notna(median_abs) and median_abs > 10.0:
        return 100.0 / positive
    return 1.0 / positive


def _parse_scb_month(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return pd.to_datetime(values.str.replace("M", "-", regex=False) + "-01", errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def _series_from_frame(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _common_equity_mask(frame: pd.DataFrame) -> pd.Series:
    if "issue_type_code" in frame.columns:
        codes = frame["issue_type_code"].astype("string").str.strip().str.upper()
    elif "tpci" in frame.columns:
        codes = frame["tpci"].astype("string").str.strip().str.upper()
    else:
        return pd.Series(True, index=frame.index)
    return codes.isna() | codes.eq("0") | codes.eq("EQ") | codes.eq("COM")


def _proxy_rows(
    frame: pd.DataFrame,
    *,
    proxy_code: str,
    paper_proxy_name: str,
    source_name: str,
    build_status: str,
    country_code: str = "SWE",
    apply_analysis_filter: bool = True,
) -> pd.DataFrame:
    if frame.empty:
        return empty_sweden_proxy_source()
    out = frame.copy()
    out["country_code"] = country_code
    out["proxy_code"] = proxy_code
    out["paper_proxy_name"] = paper_proxy_name
    out["source_name"] = source_name
    out["build_status"] = build_status
    out["exact_replication_flag"] = build_status == "exact"
    out["month_end_date"] = pd.to_datetime(out["month_end_date"], errors="coerce")
    out["raw_value"] = pd.to_numeric(out["raw_value"], errors="coerce")
    out = out.dropna(subset=["month_end_date", "raw_value"])
    if apply_analysis_filter:
        out = _sentiment_analysis_filter(out)
    out = out.drop_duplicates(["country_code", "month_end_date", "proxy_code"], keep="last")
    return out[PROXY_SOURCE_COLUMNS].sort_values(["proxy_code", "month_end_date"]).reset_index(drop=True)


def build_sweden_dividend_premium_proxy_source(
    characteristics_monthly: pd.DataFrame,
    *,
    country_code: str = "SWE",
    min_group_count: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a Baker-Wurgler-style dividend premium proxy from monthly characteristics.

    The production proxy is the log difference between value-weighted average
    market-to-book ratios for regular dividend payers and non-payers.
    """

    if min_group_count <= 0:
        raise ValueError("min_group_count must be positive.")
    diagnostics_empty = pd.DataFrame(columns=DIVIDEND_PREMIUM_DIAGNOSTIC_COLUMNS)
    if characteristics_monthly is None or characteristics_monthly.empty:
        return empty_sweden_proxy_source(), diagnostics_empty

    required = {"month_end_date", "D_PAYER", "BE_ME", "ME"}
    missing = required.difference(characteristics_monthly.columns)
    if missing:
        diagnostics_empty["missing_required_columns"] = [",".join(sorted(missing))] * len(diagnostics_empty)
        return empty_sweden_proxy_source(), diagnostics_empty

    frame = characteristics_monthly.copy()
    frame["month_end_date"] = pd.to_datetime(frame["month_end_date"], errors="coerce")
    if "country_code" in frame.columns:
        frame["country_code"] = frame["country_code"].fillna(country_code)
    else:
        frame["country_code"] = country_code
    frame["D_PAYER"] = pd.to_numeric(frame["D_PAYER"], errors="coerce")
    frame["BE_ME"] = pd.to_numeric(frame["BE_ME"], errors="coerce")
    frame["ME"] = pd.to_numeric(frame["ME"], errors="coerce")
    valid = (
        frame["month_end_date"].notna()
        & frame["D_PAYER"].isin([0.0, 1.0])
        & frame["BE_ME"].gt(0)
        & frame["ME"].gt(0)
    )
    frame = frame.loc[valid, ["country_code", "month_end_date", "D_PAYER", "BE_ME", "ME"]].copy()
    frame["market_to_book"] = 1.0 / frame["BE_ME"]
    frame = frame.loc[frame["market_to_book"].gt(0) & np.isfinite(frame["market_to_book"])].copy()
    if frame.empty:
        return empty_sweden_proxy_source(), diagnostics_empty

    rows: list[dict[str, object]] = []
    for (country, month_end_date), group in frame.groupby(["country_code", "month_end_date"], dropna=False, sort=True):
        payer = group.loc[group["D_PAYER"].eq(1.0)].copy()
        nonpayer = group.loc[group["D_PAYER"].eq(0.0)].copy()

        def _value_weighted_mb(part: pd.DataFrame) -> float:
            weights = pd.to_numeric(part["ME"], errors="coerce")
            values = pd.to_numeric(part["market_to_book"], errors="coerce")
            valid_part = weights.gt(0) & values.gt(0)
            if not valid_part.any():
                return float("nan")
            return float(np.average(values.loc[valid_part], weights=weights.loc[valid_part]))

        payer_vw = _value_weighted_mb(payer)
        nonpayer_vw = _value_weighted_mb(nonpayer)
        enough_groups = len(payer) >= min_group_count and len(nonpayer) >= min_group_count
        positive_vw = pd.notna(payer_vw) and pd.notna(nonpayer_vw) and payer_vw > 0 and nonpayer_vw > 0
        div_premium_vw = float(np.log(payer_vw) - np.log(nonpayer_vw)) if enough_groups and positive_vw else np.nan
        if enough_groups and positive_vw:
            status = "ok"
        elif not enough_groups:
            status = "insufficient_payer_or_nonpayer_count"
        else:
            status = "nonpositive_or_missing_market_to_book"

        rows.append(
            {
                "country_code": country if pd.notna(country) else country_code,
                "month_end_date": month_end_date,
                "eligible_rows": int(len(group)),
                "payer_count": int(len(payer)),
                "nonpayer_count": int(len(nonpayer)),
                "payer_market_cap": float(payer["ME"].sum()) if not payer.empty else 0.0,
                "nonpayer_market_cap": float(nonpayer["ME"].sum()) if not nonpayer.empty else 0.0,
                "payer_mb_value_weight": payer_vw,
                "nonpayer_mb_value_weight": nonpayer_vw,
                "DIV_PREMIUM_VW": div_premium_vw,
                "raw_value": div_premium_vw,
                "min_group_count": int(min_group_count),
                "dividend_premium_status": status,
            }
        )

    diagnostics = pd.DataFrame(rows, columns=DIVIDEND_PREMIUM_DIAGNOSTIC_COLUMNS)
    source = _proxy_rows(
        diagnostics.loc[diagnostics["dividend_premium_status"].eq("ok"), ["month_end_date", "raw_value"]],
        proxy_code=DIVIDEND_PREMIUM_PROXY,
        paper_proxy_name=DIVIDEND_PREMIUM_PROXY,
        source_name="mart_characteristics_monthly: value-weighted dividend payer premium from D_PAYER and BE_ME",
        build_status="adapted",
        country_code=country_code,
    )
    return source, diagnostics


def _macro_rows(frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    if frame.empty:
        return empty_sweden_macro_controls()
    out = frame.copy()
    out["country_code"] = "SWE"
    out["month_end_date"] = pd.to_datetime(out["month_end_date"], errors="coerce")
    out[column_name] = pd.to_numeric(out[column_name], errors="coerce")
    out = out.dropna(subset=["month_end_date", column_name])
    return out[["country_code", "month_end_date", column_name]].drop_duplicates(["country_code", "month_end_date"], keep="last")


def _extract_cpi_series(cpi_frame: pd.DataFrame) -> pd.DataFrame:
    if cpi_frame is None or cpi_frame.empty:
        return empty_sweden_macro_controls()
    value_column = next((column for column in cpi_frame.columns if column not in {"month", "observations"}), None)
    if value_column is None:
        return empty_sweden_macro_controls()
    frame = cpi_frame.copy()
    frame["month_end_date"] = _parse_scb_month(frame["month"])
    mask = frame["observations"].astype("string").str.strip().eq("CPI, Shadow Index numbers")
    frame = frame.loc[mask, ["month_end_date", value_column]].rename(columns={value_column: "CPI"})
    return _macro_rows(frame, "CPI")


def _extract_ppi_series(ppi_frame: pd.DataFrame) -> pd.DataFrame:
    if ppi_frame is None or ppi_frame.empty:
        return empty_sweden_macro_controls()
    value_column = next((column for column in ppi_frame.columns if column not in {"products by SPIN 2015", "month", "observations"}), None)
    if value_column is None:
        return empty_sweden_macro_controls()
    frame = ppi_frame.copy()
    frame["month_end_date"] = _parse_scb_month(frame["month"])
    products = frame["products by SPIN 2015"].astype("string").str.strip()
    observations = frame["observations"].astype("string").str.strip()
    mask = products.eq("B-E Total") & observations.eq("Producer Price Index (PPI)")
    frame = frame.loc[mask, ["month_end_date", value_column]].rename(columns={value_column: "PPI"})
    return _macro_rows(frame, "PPI")


def _extract_ip_series(ip_frame: pd.DataFrame) -> pd.DataFrame:
    if ip_frame is None or ip_frame.empty:
        return empty_sweden_macro_controls()
    value_column = next(
        (column for column in ip_frame.columns if column not in {"industrial classification (NACE Rev. 2)", "month", "observations"}),
        None,
    )
    if value_column is None:
        return empty_sweden_macro_controls()
    frame = ip_frame.copy()
    frame["month_end_date"] = _parse_scb_month(frame["month"])
    classification = frame["industrial classification (NACE Rev. 2)"].astype("string").str.strip()
    observations = frame["observations"].astype("string").str.strip()
    mask = classification.eq("B+C mining, quarrying, manufacturing") & observations.eq("Calendar adjusted and seasonally adjusted")
    frame = frame.loc[mask, ["month_end_date", value_column]].rename(columns={value_column: "IP"})
    return _macro_rows(frame, "IP")


def _extract_em_series(em_frame: pd.DataFrame) -> pd.DataFrame:
    if em_frame is None or em_frame.empty:
        return empty_sweden_macro_controls()
    value_column = next((column for column in em_frame.columns if column not in {"labour status", "type of data", "sex", "age", "month"}), None)
    if value_column is None:
        return empty_sweden_macro_controls()
    frame = em_frame.copy()
    frame["month_end_date"] = _parse_scb_month(frame["month"])
    labour_status = frame["labour status"].astype("string").str.strip()
    type_of_data = frame["type of data"].astype("string").str.strip()
    sex = frame["sex"].astype("string").str.strip()
    age = frame["age"].astype("string").str.strip()
    mask = (
        labour_status.eq("employment rate, percent")
        & type_of_data.eq("seasonally adjusted and smoothed")
        & sex.eq("total")
        & age.eq("total 15-74 years")
    )
    frame = frame.loc[mask, ["month_end_date", value_column]].rename(columns={value_column: "EM"})
    return _macro_rows(frame, "EM")


def _extract_unemployment_series(labour_frame: pd.DataFrame) -> pd.DataFrame:
    if labour_frame is None or labour_frame.empty:
        return empty_sweden_sibley_macro_controls()
    value_column = next(
        (column for column in labour_frame.columns if column not in {"labour status", "type of data", "sex", "age", "month"}),
        None,
    )
    if value_column is None:
        return empty_sweden_sibley_macro_controls()
    frame = labour_frame.copy()
    frame["month_end_date"] = _parse_scb_month(frame["month"])
    labour_status = frame["labour status"].astype("string").str.strip()
    type_of_data = frame["type of data"].astype("string").str.strip()
    sex = frame["sex"].astype("string").str.strip()
    age = frame["age"].astype("string").str.strip()
    mask = (
        labour_status.eq("unemployment rate, percent")
        & type_of_data.eq("seasonally adjusted")
        & sex.eq("total")
        & age.eq("total 15-74 years")
    )
    frame = frame.loc[mask, ["month_end_date", value_column]].rename(columns={value_column: "UNEMP"})
    return _macro_rows(frame, "UNEMP")


def _extract_consumption_growth_series(consumption_frame: pd.DataFrame) -> pd.DataFrame:
    if consumption_frame is None or consumption_frame.empty:
        return empty_sweden_sibley_macro_controls()
    value_column = next(
        (column for column in consumption_frame.columns if column not in {"DATE", "TIME PERIOD", "date", "time period"}),
        None,
    )
    if value_column is None:
        return empty_sweden_sibley_macro_controls()
    frame = consumption_frame.copy()
    date_column = "DATE" if "DATE" in frame.columns else "date"
    frame["quarter_end_date"] = pd.to_datetime(frame[date_column], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    frame = frame.dropna(subset=["quarter_end_date"]).sort_values("quarter_end_date")
    frame["dCONS"] = _log_growth(frame[value_column])
    quarter_growth = frame[["quarter_end_date", "dCONS"]].dropna().drop_duplicates("quarter_end_date", keep="last")
    if quarter_growth.empty:
        return empty_sweden_sibley_macro_controls()
    month_calendar = pd.DataFrame(
        {
            "month_end_date": pd.date_range(
                SENTIMENT_ANALYSIS_START,
                SENTIMENT_ANALYSIS_END,
                freq="ME",
            )
        }
    )
    merged = pd.merge_asof(
        month_calendar.sort_values("month_end_date"),
        quarter_growth.rename(columns={"quarter_end_date": "month_end_date"}).sort_values("month_end_date"),
        on="month_end_date",
        direction="backward",
    )
    return _macro_rows(merged, "dCONS")


def _extract_rate_state_controls(rates_monthly: pd.DataFrame) -> pd.DataFrame:
    if rates_monthly is None or rates_monthly.empty:
        return empty_sweden_sibley_macro_controls()
    rates = rates_monthly.copy()
    rates["month_end_date"] = pd.to_datetime(rates.get("month_end_date"), errors="coerce")
    rates["raw_annual_rate_pct"] = pd.to_numeric(rates.get("raw_annual_rate_pct"), errors="coerce")
    pivot = rates.pivot_table(
        index="month_end_date",
        columns="rate_code",
        values="raw_annual_rate_pct",
        aggfunc="last",
    ).reset_index()
    if "RF_3M_PROXY" not in pivot.columns or "YIELD_10Y" not in pivot.columns:
        return empty_sweden_sibley_macro_controls()
    pivot["TBILL"] = pd.to_numeric(pivot["RF_3M_PROXY"], errors="coerce")
    pivot["TERM"] = pd.to_numeric(pivot["YIELD_10Y"], errors="coerce") - pivot["TBILL"]
    out = _macro_rows(pivot, "TBILL")
    return out.merge(_macro_rows(pivot, "TERM"), on=["country_code", "month_end_date"], how="outer")


def _standardize_sibley_daily_prices(daily_prices: pd.DataFrame, market_monthly: pd.DataFrame | None) -> pd.DataFrame:
    if daily_prices is None or daily_prices.empty:
        return pd.DataFrame()
    frame = daily_prices.copy()
    frame["trade_date"] = pd.to_datetime(frame.get("trade_date"), errors="coerce")
    frame["month_end_date"] = frame["trade_date"].dt.to_period("M").dt.to_timestamp("M")
    frame["prccd"] = pd.to_numeric(frame.get("prccd"), errors="coerce")
    frame["trfd"] = pd.to_numeric(frame.get("trfd"), errors="coerce")
    frame["cshoc"] = pd.to_numeric(frame.get("cshoc"), errors="coerce")
    frame = frame.dropna(subset=["security_id", "trade_date", "month_end_date"]).sort_values(["security_id", "trade_date"])
    if market_monthly is not None and not market_monthly.empty and {"security_id", "month_end_date"}.issubset(market_monthly.columns):
        selected = market_monthly[["security_id", "month_end_date"]].drop_duplicates().copy()
        selected["month_end_date"] = pd.to_datetime(selected["month_end_date"], errors="coerce")
        frame = frame.merge(selected, on=["security_id", "month_end_date"], how="inner")
    frame["adjusted_price"] = frame["prccd"] * frame["trfd"]
    frame["prev_adjusted_price"] = frame.groupby("security_id")["adjusted_price"].shift(1)
    valid = frame["adjusted_price"].notna() & frame["prev_adjusted_price"].notna() & frame["prev_adjusted_price"].ne(0)
    frame["daily_total_return"] = np.where(valid, frame["adjusted_price"] / frame["prev_adjusted_price"] - 1.0, np.nan)
    frame["market_cap"] = frame["prccd"] * frame["cshoc"]
    frame["lagged_daily_me"] = frame.groupby("security_id")["market_cap"].shift(1)
    frame["common_equity_flag"] = _common_equity_mask(frame)
    return clean_sweden_daily_returns(frame)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def _extract_market_state_controls(daily_prices: pd.DataFrame, market_monthly: pd.DataFrame | None) -> pd.DataFrame:
    frame = _standardize_sibley_daily_prices(daily_prices, market_monthly)
    if frame.empty:
        return empty_sweden_sibley_macro_controls()
    frame = frame.loc[frame["common_equity_flag"]].copy()
    returns = pd.to_numeric(frame.get("daily_total_return_clean", frame.get("daily_total_return")), errors="coerce")
    weights = pd.to_numeric(frame.get("lagged_daily_me"), errors="coerce")
    frame["daily_return_for_market"] = returns
    frame["zero_return_flag"] = np.where(returns.notna(), returns.abs().le(1e-12).astype(float), np.nan)

    daily_rows: list[dict[str, object]] = []
    for trade_date, daily_frame in frame.groupby("trade_date", sort=True):
        daily_rows.append(
            {
                "trade_date": trade_date,
                "month_end_date": pd.Timestamp(trade_date).to_period("M").to_timestamp("M"),
                "market_return": _weighted_average(
                    pd.to_numeric(daily_frame["daily_return_for_market"], errors="coerce"),
                    pd.to_numeric(daily_frame["lagged_daily_me"], errors="coerce"),
                ),
                "zero_return_share": pd.to_numeric(daily_frame["zero_return_flag"], errors="coerce").mean(),
            }
        )
    daily_market = pd.DataFrame(daily_rows)
    if daily_market.empty:
        return empty_sweden_sibley_macro_controls()

    rows: list[dict[str, object]] = []
    for month_end_date, group in daily_market.groupby("month_end_date", sort=True):
        market_returns = pd.to_numeric(group["market_return"], errors="coerce").dropna()
        if market_returns.empty:
            vwretd = float("nan")
            mktvol = float("nan")
        else:
            vwretd = float(np.prod(1.0 + market_returns.to_numpy(dtype=float)) - 1.0)
            mktvol = float(market_returns.std(ddof=1) * np.sqrt(252.0)) if market_returns.size >= 2 else float("nan")
        rows.append(
            {
                "country_code": "SWE",
                "month_end_date": month_end_date,
                "VWRETD": vwretd,
                "MKTVOL": mktvol,
                "PCTZERO": pd.to_numeric(group["zero_return_share"], errors="coerce").mean(),
            }
        )
    return pd.DataFrame(rows)


def build_sweden_sibley_macro_controls_monthly(
    *,
    cpi: pd.DataFrame,
    ip: pd.DataFrame,
    labour_market: pd.DataFrame,
    consumption: pd.DataFrame,
    rates_monthly: pd.DataFrame,
    market_monthly: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> pd.DataFrame:
    cpi_series = _extract_cpi_series(cpi)
    cpi_series["dCPI"] = _log_growth(cpi_series["CPI"])
    d_cpi = _macro_rows(cpi_series, "dCPI")

    ip_series = _extract_ip_series(ip)
    ip_series["dIND"] = _log_growth(ip_series["IP"])
    d_ind = _macro_rows(ip_series, "dIND")

    frames = [
        _extract_unemployment_series(labour_market),
        d_cpi,
        _extract_consumption_growth_series(consumption),
        d_ind,
        _extract_rate_state_controls(rates_monthly),
        _extract_market_state_controls(daily_prices, market_monthly),
    ]
    built = [frame for frame in frames if frame is not None and not frame.empty]
    if not built:
        return empty_sweden_sibley_macro_controls()
    out = built[0]
    for frame in built[1:]:
        out = out.merge(frame, on=["country_code", "month_end_date"], how="outer")
    out = out.sort_values("month_end_date").reset_index(drop=True)
    missing = [
        column
        for column in SIBLEY_MACRO_CONTROL_COLUMNS
        if column not in out.columns or not pd.to_numeric(out[column], errors="coerce").notna().any()
    ]
    if missing:
        raise RuntimeError(f"Could not construct required Sibley macro-control series: {', '.join(missing)}")
    return out[["country_code", "month_end_date", *SIBLEY_MACRO_CONTROL_COLUMNS]]


def build_sweden_macro_controls_monthly(
    *,
    cpi: pd.DataFrame | None = None,
    ppi: pd.DataFrame | None = None,
    ip: pd.DataFrame | None = None,
    em: pd.DataFrame | None = None,
) -> pd.DataFrame:
    missing_inputs = [
        name
        for name, frame in [("cpi", cpi), ("ppi", ppi), ("ip", ip), ("em", em)]
        if frame is None or frame.empty
    ]
    if missing_inputs:
        raise RuntimeError(f"Missing required macro-control source tables: {', '.join(missing_inputs)}")

    frames = [
        _extract_cpi_series(cpi),
        _extract_ppi_series(ppi),
        _extract_ip_series(ip),
        _extract_em_series(em),
    ]
    required_columns = ["CPI", "PPI", "IP", "EM"]
    missing_series = [
        column
        for column, frame in zip(required_columns, frames, strict=True)
        if frame.empty or column not in frame.columns or not frame[column].notna().any()
    ]
    if missing_series:
        raise RuntimeError(f"Could not construct required macro-control series: {', '.join(missing_series)}")

    built = [frame for frame in frames if not frame.empty]
    out = built[0]
    for frame in built[1:]:
        out = out.merge(frame, on=["country_code", "month_end_date"], how="outer")
    out = out.sort_values("month_end_date").reset_index(drop=True)
    return out[["country_code", "month_end_date", "CPI", "PPI", "IP", "EM"]]


def apply_macro_control_availability_to_paper_spec(
    paper_spec: pd.DataFrame,
    macro_controls: pd.DataFrame | None,
) -> pd.DataFrame:
    audit = paper_spec.copy()
    if macro_controls is None or macro_controls.empty:
        return audit
    available_columns = {column for column in ["CPI", "PPI", "IP", "EM"] if column in macro_controls.columns and macro_controls[column].notna().any()}
    if not available_columns:
        return audit
    mask = audit["paper_proxy_name"].isin(available_columns)
    audit.loc[mask, "build_status"] = "exact"
    return audit


def apply_ipo_return_availability_to_paper_spec(
    paper_spec: pd.DataFrame,
    ipo_returns: pd.DataFrame | None,
) -> pd.DataFrame:
    audit = paper_spec.copy()
    if ipo_returns is None or ipo_returns.empty:
        return audit
    if "ipo_month_end_date" in ipo_returns.columns and pd.to_datetime(
        ipo_returns["ipo_month_end_date"], errors="coerce"
    ).notna().any():
        audit.loc[audit["paper_proxy_name"].eq("NIPO"), "build_status"] = "exact"
        audit.loc[audit["paper_proxy_name"].eq("NIPO"), "repo_source_candidate"] = (
            "ipo_dates_and_offering_price_sweden.csv:IPO Dt"
        )
    source_column = "ripo_return" if "ripo_return" in ipo_returns.columns else "offer_to_first_close_return"
    if source_column not in ipo_returns.columns:
        return audit
    if not pd.to_numeric(ipo_returns[source_column], errors="coerce").notna().any():
        return audit
    audit.loc[audit["paper_proxy_name"].eq("RIPO"), "build_status"] = "exact"
    audit.loc[audit["paper_proxy_name"].eq("RIPO"), "repo_source_candidate"] = (
        "ipo_dates_and_offering_price_sweden.csv + Daily prices jan 2000 - dec 2025 zip limited (preferred).csv"
    )
    return audit


def apply_ed_ratio_availability_to_paper_spec(
    paper_spec: pd.DataFrame,
    economic_consumer_sentiment: pd.DataFrame | None,
) -> pd.DataFrame:
    audit = paper_spec.copy()
    if economic_consumer_sentiment is None or economic_consumer_sentiment.empty:
        return audit
    source_column = _ed_ratio_source_column(economic_consumer_sentiment)
    if source_column is None:
        return audit
    ed_ratio = _invert_debt_to_equity(economic_consumer_sentiment[source_column])
    if not ed_ratio.notna().any():
        return audit
    mask = audit["paper_proxy_name"].eq("ED_RATIO")
    audit.loc[mask, "build_status"] = "exact"
    audit.loc[mask, "repo_source_candidate"] = f"Economic and consumer sentiment.xlsx:{source_column}"
    return audit


def build_economic_consumer_sentiment_proxy_source(
    workbook: pd.DataFrame,
    *,
    country_code: str = "SWE",
    paper_spec: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if workbook is None or workbook.empty:
        return empty_sweden_proxy_source()

    frame = workbook.copy()
    status_map = _paper_spec_status_map(paper_spec)
    column_map = {
        "EUESSE Index  (L1)": ("ESI", "ESI"),
        "SWETCI Index  (R1)": ("CCI", "CCI"),
    }
    if "Date" not in frame.columns:
        return empty_sweden_proxy_source()

    out_frames: list[pd.DataFrame] = []
    dates = pd.to_datetime(frame["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    for source_column, (proxy_code, paper_proxy_name) in column_map.items():
        if source_column not in frame.columns:
            continue
        proxy_frame = pd.DataFrame(
            {
                "month_end_date": dates,
                "raw_value": _monthly_growth(frame[source_column]),
            }
        )
        out_frames.append(
            _proxy_rows(
                proxy_frame,
                proxy_code=proxy_code,
                paper_proxy_name=paper_proxy_name,
                source_name="Economic and consumer sentiment.xlsx",
                build_status=status_map.get(paper_proxy_name, "exact"),
                country_code=country_code,
            )
        )
    ed_ratio_column = _ed_ratio_source_column(frame)
    if ed_ratio_column is not None:
        ed_ratio_frame = pd.DataFrame(
            {
                "month_end_date": dates,
                "raw_value": _invert_debt_to_equity(frame[ed_ratio_column]),
            }
        )
        out_frames.append(
            _proxy_rows(
                ed_ratio_frame,
                proxy_code="ED_RATIO",
                paper_proxy_name="ED_RATIO",
                source_name=f"Economic and consumer sentiment.xlsx:{ed_ratio_column}",
                build_status=status_map.get("ED_RATIO", "exact"),
                country_code=country_code,
            )
        )
    return (
        pd.concat(out_frames, ignore_index=True).sort_values(["proxy_code", "month_end_date"]).reset_index(drop=True)
        if out_frames
        else empty_sweden_proxy_source()
    )


def build_turnover_proxy_source(
    daily_prices: pd.DataFrame,
    *,
    country_code: str = "SWE",
    paper_spec: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if daily_prices is None or daily_prices.empty:
        return empty_sweden_proxy_source()

    frame = daily_prices.copy()
    if "trade_date" not in frame.columns and "datadate" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["datadate"], errors="coerce")
    else:
        frame["trade_date"] = pd.to_datetime(frame.get("trade_date"), errors="coerce")
    if "cshtrd" not in frame.columns or "cshoc" not in frame.columns:
        return empty_sweden_proxy_source()

    frame["cshtrd"] = pd.to_numeric(frame["cshtrd"], errors="coerce")
    frame["cshoc"] = pd.to_numeric(frame["cshoc"], errors="coerce")
    frame = frame.loc[_common_equity_mask(frame)].copy()
    frame["month_end_date"] = frame["trade_date"].dt.to_period("M").dt.to_timestamp("M")
    frame = frame.dropna(subset=["security_id", "month_end_date"])
    if frame.empty:
        return empty_sweden_proxy_source()

    volume = (
        frame.groupby(["security_id", "month_end_date"], as_index=False)["cshtrd"]
        .sum(min_count=1)
        .rename(columns={"cshtrd": "monthly_trading_volume"})
    )
    shares = (
        frame.sort_values(["security_id", "month_end_date", "trade_date"])
        .drop_duplicates(["security_id", "month_end_date"], keep="last")[["security_id", "month_end_date", "cshoc"]]
        .rename(columns={"cshoc": "month_end_shares_outstanding"})
    )
    turnover = volume.merge(shares, on=["security_id", "month_end_date"], how="left")
    monthly = (
        turnover.groupby("month_end_date", as_index=False)
        .agg(
            total_trading_volume=("monthly_trading_volume", "sum"),
            total_outstanding_shares=("month_end_shares_outstanding", "sum"),
        )
    )
    monthly["raw_value"] = np.where(
        monthly["total_outstanding_shares"].notna() & monthly["total_outstanding_shares"].gt(0),
        np.log(monthly["total_trading_volume"] / monthly["total_outstanding_shares"]),
        np.nan,
    )
    monthly = monthly.loc[pd.to_datetime(monthly["month_end_date"], errors="coerce").ge(TURN_PROXY_HISTORY_START)].copy()
    status_map = _paper_spec_status_map(paper_spec)
    return _proxy_rows(
        monthly,
        proxy_code="TURN",
        paper_proxy_name="TURN",
        source_name="Daily prices jan 2000 - dec 2025 zip limited (preferred).csv",
        build_status=status_map.get("TURN", "exact"),
        country_code=country_code,
        apply_analysis_filter=False,
    )


def build_ipo_count_proxy_source(
    ipo_returns: pd.DataFrame,
    *,
    country_code: str = "SWE",
    paper_spec: pd.DataFrame | None = None,
) -> pd.DataFrame:
    calendar = _sentiment_analysis_month_calendar(country_code=country_code)
    status_map = _paper_spec_status_map(paper_spec)
    if ipo_returns is None or ipo_returns.empty or "ipo_month_end_date" not in ipo_returns.columns:
        return empty_sweden_proxy_source()

    frame = ipo_returns.copy()
    frame["month_end_date"] = pd.to_datetime(frame["ipo_month_end_date"], errors="coerce")
    id_columns = [column for column in ["isin", "ipo_date", "bloomberg_ticker"] if column in frame.columns]
    if id_columns:
        dedupe_columns = ["month_end_date", *id_columns]
    else:
        frame["_ipo_count_row_id"] = np.arange(len(frame))
        dedupe_columns = ["month_end_date", "_ipo_count_row_id"]
    frame = frame.dropna(subset=["month_end_date"])
    if frame.empty:
        return empty_sweden_proxy_source()

    counts = (
        frame[dedupe_columns]
        .drop_duplicates()
        .groupby("month_end_date", as_index=False)
        .size()
        .rename(columns={"size": "raw_value"})
    )
    monthly = calendar.merge(counts, on="month_end_date", how="left")
    monthly["raw_value"] = monthly["raw_value"].fillna(0.0)
    return _proxy_rows(
        monthly,
        proxy_code="NIPO",
        paper_proxy_name="NIPO",
        source_name="ipo_dates_and_offering_price_sweden.csv",
        build_status=status_map.get("NIPO", "exact"),
        country_code=country_code,
    )


def build_fundamentals_ipo_count_audit_source(
    quarterly_fundamentals: pd.DataFrame,
    *,
    country_code: str = "SWE",
) -> pd.DataFrame:
    calendar = _sentiment_analysis_month_calendar(country_code=country_code)
    if quarterly_fundamentals is None or quarterly_fundamentals.empty or "ipodate" not in quarterly_fundamentals.columns:
        return empty_sweden_proxy_source()

    frame = quarterly_fundamentals.copy()
    frame["ipodate"] = pd.to_datetime(frame["ipodate"], errors="coerce")
    frame["month_end_date"] = frame["ipodate"].dt.to_period("M").dt.to_timestamp("M")
    id_column = "company_id" if "company_id" in frame.columns else "gvkey"
    frame = frame.dropna(subset=[id_column, "month_end_date"])
    if frame.empty:
        return empty_sweden_proxy_source()

    counts = (
        frame[[id_column, "month_end_date"]]
        .drop_duplicates()
        .groupby("month_end_date", as_index=False)
        .size()
        .rename(columns={"size": "raw_value"})
    )
    monthly = calendar.merge(counts, on="month_end_date", how="left")
    monthly["raw_value"] = monthly["raw_value"].fillna(0.0)
    return _proxy_rows(
        monthly,
        proxy_code="NIPO_FUNDAMENTALS_AUDIT",
        paper_proxy_name="NIPO",
        source_name="Quarterly fundamentals sweden jan 2000 - dec 2025.xlsx:ipodate",
        build_status="audit_only",
        country_code=country_code,
    )


def build_sweden_ipo_return_table(
    ipo_offers: pd.DataFrame,
    daily_prices: pd.DataFrame,
    *,
    country_code: str = "SWE",
) -> pd.DataFrame:
    if ipo_offers is None or ipo_offers.empty:
        return empty_sweden_ipo_return_table()

    offers = ipo_offers.copy()
    offers["country_code"] = country_code
    offers["isin"] = offers.get("isin", pd.Series(pd.NA, index=offers.index)).astype("string").str.strip().str.upper()
    offers["ipo_date"] = pd.to_datetime(offers.get("ipo_date"), errors="coerce")
    offers["ipo_offer_price"] = pd.to_numeric(offers.get("ipo_offer_price"), errors="coerce")
    if "bloomberg_offer_to_first_open_return" in offers.columns:
        offers["bloomberg_offer_to_first_open_return"] = pd.to_numeric(
            offers["bloomberg_offer_to_first_open_return"],
            errors="coerce",
        )
    elif "bloomberg_offer_to_first_open_return_pct" in offers.columns:
        offers["bloomberg_offer_to_first_open_return"] = (
            pd.to_numeric(offers["bloomberg_offer_to_first_open_return_pct"], errors="coerce") / 100.0
        )
    else:
        offers["bloomberg_offer_to_first_open_return"] = np.nan
    offers["bloomberg_offer_to_first_open_return_pct"] = offers["bloomberg_offer_to_first_open_return"] * 100.0
    valid = (
        offers["isin"].str.len().eq(12).fillna(False)
        & offers["ipo_date"].notna()
        & offers["ipo_offer_price"].notna()
        & offers["ipo_offer_price"].gt(0)
    )
    offers = offers.loc[valid].copy()
    if offers.empty:
        return empty_sweden_ipo_return_table()

    for column in ["bloomberg_ticker", "company_name_bloomberg", "currency", "primary_exchange_name"]:
        if column not in offers.columns:
            offers[column] = pd.NA
    offers["ipo_month_end_date"] = offers["ipo_date"].dt.to_period("M").dt.to_timestamp("M")
    offers = offers.reset_index(drop=True).reset_index(names="ipo_row_id")

    if daily_prices is None or daily_prices.empty:
        out = offers.copy()
        out["first_close_date"] = pd.NaT
        out["first_close_price"] = np.nan
        out["offer_to_first_close_return"] = np.nan
        out["ripo_return"] = out["bloomberg_offer_to_first_open_return"]
        out["ripo_return_source"] = np.where(out["ripo_return"].notna(), "bloomberg_first_open_no_daily_prices", pd.NA)
        out["days_after_ipo"] = np.nan
        out["match_status"] = "no_daily_prices"
        out["daily_gvkey"] = pd.NA
        out["daily_iid"] = pd.NA
        out["daily_company_name"] = pd.NA
        return out.reindex(columns=IPO_RETURN_COLUMNS)

    daily = daily_prices.copy()
    if "isin" not in daily.columns or "trade_date" not in daily.columns:
        out = offers.copy()
        out["first_close_date"] = pd.NaT
        out["first_close_price"] = np.nan
        out["offer_to_first_close_return"] = np.nan
        out["ripo_return"] = out["bloomberg_offer_to_first_open_return"]
        out["ripo_return_source"] = np.where(out["ripo_return"].notna(), "bloomberg_first_open_missing_daily_match_columns", pd.NA)
        out["days_after_ipo"] = np.nan
        out["match_status"] = "missing_daily_match_columns"
        out["daily_gvkey"] = pd.NA
        out["daily_iid"] = pd.NA
        out["daily_company_name"] = pd.NA
        return out.reindex(columns=IPO_RETURN_COLUMNS)

    daily["isin"] = daily["isin"].astype("string").str.strip().str.upper()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce")
    daily["prccd"] = pd.to_numeric(daily.get("prccd"), errors="coerce")
    daily = daily.loc[
        daily["isin"].isin(offers["isin"].dropna().unique())
        & daily["trade_date"].notna()
        & daily["prccd"].notna()
    ].copy()
    if daily.empty:
        out = offers.copy()
        out["first_close_date"] = pd.NaT
        out["first_close_price"] = np.nan
        out["offer_to_first_close_return"] = np.nan
        out["ripo_return"] = out["bloomberg_offer_to_first_open_return"]
        out["ripo_return_source"] = np.where(out["ripo_return"].notna(), "bloomberg_first_open_no_isin_match", pd.NA)
        out["days_after_ipo"] = np.nan
        out["match_status"] = "no_isin_match"
        out["daily_gvkey"] = pd.NA
        out["daily_iid"] = pd.NA
        out["daily_company_name"] = pd.NA
        return out.reindex(columns=IPO_RETURN_COLUMNS)

    keep_daily_columns = [
        column for column in ["isin", "trade_date", "prccd", "gvkey", "iid", "conm"] if column in daily.columns
    ]
    candidates = offers[["ipo_row_id", "isin", "ipo_date"]].merge(daily[keep_daily_columns], on="isin", how="left")
    candidates = candidates.loc[candidates["trade_date"].ge(candidates["ipo_date"])].copy()
    candidates["days_after_ipo"] = (candidates["trade_date"] - candidates["ipo_date"]).dt.days
    first_close = (
        candidates.sort_values(["ipo_row_id", "days_after_ipo", "trade_date"])
        .drop_duplicates("ipo_row_id", keep="first")
        .rename(
            columns={
                "trade_date": "first_close_date",
                "prccd": "first_close_price",
                "gvkey": "daily_gvkey",
                "iid": "daily_iid",
                "conm": "daily_company_name",
            }
        )
    )
    out = offers.merge(
        first_close[
            [
                column
                for column in [
                    "ipo_row_id",
                    "first_close_date",
                    "first_close_price",
                    "days_after_ipo",
                    "daily_gvkey",
                    "daily_iid",
                    "daily_company_name",
                ]
                if column in first_close.columns
            ]
        ],
        on="ipo_row_id",
        how="left",
    )
    for column in ["daily_gvkey", "daily_iid", "daily_company_name"]:
        if column not in out.columns:
            out[column] = pd.NA
    out["offer_to_first_close_return"] = np.where(
        out["first_close_price"].notna() & out["ipo_offer_price"].gt(0),
        (out["first_close_price"] / out["ipo_offer_price"]) - 1.0,
        np.nan,
    )
    out["match_status"] = np.where(
        out["first_close_price"].notna(),
        np.where(out["days_after_ipo"].eq(0), "matched_exact_ipo_date", "matched_next_available_trading_day"),
        "no_close_on_or_after_ipo_date",
    )
    use_computed = out["offer_to_first_close_return"].notna() & out["days_after_ipo"].lt(4)
    use_bloomberg = ~use_computed & out["bloomberg_offer_to_first_open_return"].notna()
    out["ripo_return"] = np.select(
        [use_computed, use_bloomberg],
        [out["offer_to_first_close_return"], out["bloomberg_offer_to_first_open_return"]],
        default=np.nan,
    )
    out["ripo_return_source"] = np.select(
        [use_computed, use_bloomberg],
        ["computed_first_close_within_3_days", "bloomberg_first_open_after_3_days_or_missing_close"],
        default=pd.NA,
    )
    return out.reindex(columns=IPO_RETURN_COLUMNS).sort_values(["ipo_date", "isin"]).reset_index(drop=True)


def build_ipo_return_proxy_source(
    ipo_returns: pd.DataFrame,
    *,
    country_code: str = "SWE",
) -> pd.DataFrame:
    calendar = _sentiment_analysis_month_calendar(country_code=country_code)
    flag = calendar.assign(raw_value=0.0)
    if ipo_returns is None or ipo_returns.empty or (
        "ripo_return" not in ipo_returns.columns and "offer_to_first_close_return" not in ipo_returns.columns
    ):
        return _proxy_rows(
            flag,
            proxy_code="RIPO_OBSERVED_FLAG",
            paper_proxy_name="RIPO_OBSERVED_FLAG",
            source_name="ipo_dates_and_offering_price_sweden.csv + daily prccd",
            build_status="exact",
            country_code=country_code,
        )
    frame = ipo_returns.copy()
    frame["month_end_date"] = pd.to_datetime(frame.get("ipo_month_end_date"), errors="coerce")
    source_column = "ripo_return" if "ripo_return" in frame.columns else "offer_to_first_close_return"
    frame["raw_value"] = pd.to_numeric(frame[source_column], errors="coerce")
    frame = frame.dropna(subset=["month_end_date", "raw_value"])
    if frame.empty:
        return _proxy_rows(
            flag,
            proxy_code="RIPO_OBSERVED_FLAG",
            paper_proxy_name="RIPO_OBSERVED_FLAG",
            source_name="ipo_dates_and_offering_price_sweden.csv + daily prccd",
            build_status="exact",
            country_code=country_code,
        )
    monthly = frame.groupby("month_end_date", as_index=False)["raw_value"].mean()
    observed = monthly[["month_end_date"]].drop_duplicates().assign(observed_flag=1.0)
    flag = calendar.merge(observed, on="month_end_date", how="left")
    flag["raw_value"] = flag["observed_flag"].fillna(0.0)
    ripo = _proxy_rows(
        monthly,
        proxy_code="RIPO",
        paper_proxy_name="RIPO",
        source_name="ipo_dates_and_offering_price_sweden.csv + daily prccd",
        build_status="exact",
        country_code=country_code,
    )
    ripo_flag = _proxy_rows(
        flag,
        proxy_code="RIPO_OBSERVED_FLAG",
        paper_proxy_name="RIPO_OBSERVED_FLAG",
        source_name="ipo_dates_and_offering_price_sweden.csv + daily prccd",
        build_status="exact",
        country_code=country_code,
    )
    return pd.concat([ripo, ripo_flag], ignore_index=True).sort_values(["proxy_code", "month_end_date"]).reset_index(drop=True)


def build_sweden_sentiment_proxy_source(
    *,
    economic_consumer_sentiment: pd.DataFrame | None = None,
    daily_prices: pd.DataFrame | None = None,
    quarterly_fundamentals: pd.DataFrame | None = None,
    ipo_offers: pd.DataFrame | None = None,
    ipo_returns: pd.DataFrame | None = None,
    paper_spec: pd.DataFrame | None = None,
    country_code: str = "SWE",
) -> pd.DataFrame:
    if economic_consumer_sentiment is None or economic_consumer_sentiment.empty:
        raise RuntimeError("Economic and consumer sentiment workbook is required for final sentiment proxies.")
    if daily_prices is None or daily_prices.empty:
        raise RuntimeError("Daily prices are required for TURN and IPO-return sentiment proxies.")
    if ipo_returns is None and (ipo_offers is None or ipo_offers.empty):
        raise RuntimeError("IPO offers or precomputed IPO returns are required for NIPO and RIPO.")

    ripo_returns = (
        build_sweden_ipo_return_table(
            ipo_offers,
            daily_prices,
            country_code=country_code,
        )
        if ipo_returns is None
        else ipo_returns
    )
    frames = [
        build_economic_consumer_sentiment_proxy_source(
            economic_consumer_sentiment if economic_consumer_sentiment is not None else pd.DataFrame(),
            country_code=country_code,
            paper_spec=paper_spec,
        ),
        build_turnover_proxy_source(
            daily_prices if daily_prices is not None else pd.DataFrame(),
            country_code=country_code,
            paper_spec=paper_spec,
        ),
        build_ipo_count_proxy_source(
            ripo_returns,
            country_code=country_code,
            paper_spec=paper_spec,
        ),
        build_ipo_return_proxy_source(
            ripo_returns,
            country_code=country_code,
        ),
        build_fundamentals_ipo_count_audit_source(
            quarterly_fundamentals if quarterly_fundamentals is not None else pd.DataFrame(),
            country_code=country_code,
        ),
    ]
    built = [frame for frame in frames if not frame.empty]
    if not built:
        raise RuntimeError("No final Sweden sentiment proxy sources could be constructed.")
    out = pd.concat(built, ignore_index=True)
    required_proxies = set(BASE_SWEDEN_SENTIMENT_PROXIES)
    available_proxies = set(out["proxy_code"].dropna().astype(str))
    missing_proxies = sorted(required_proxies.difference(available_proxies))
    if missing_proxies:
        raise RuntimeError(f"Missing required final Sweden sentiment proxies: {missing_proxies}")
    return out.sort_values(["proxy_code", "month_end_date"]).reset_index(drop=True)


def build_sweden_sentiment_proxy_mart(proxy_source: pd.DataFrame) -> pd.DataFrame:
    if proxy_source is None or proxy_source.empty:
        return empty_sweden_proxy_source()
    out = proxy_source.copy()
    for column, default in [
        ("paper_proxy_name", pd.NA),
        ("source_name", pd.NA),
        ("build_status", pd.NA),
        ("exact_replication_flag", pd.NA),
    ]:
        if column not in out.columns:
            out[column] = default
    out["month_end_date"] = pd.to_datetime(out["month_end_date"], errors="coerce")
    out["raw_value"] = pd.to_numeric(out["raw_value"], errors="coerce")
    out = out.dropna(subset=["country_code", "month_end_date", "proxy_code", "raw_value"])
    out = out.sort_values(["country_code", "proxy_code", "month_end_date"])
    out = out.drop_duplicates(["country_code", "month_end_date", "proxy_code"], keep="last")
    return out.reset_index(drop=True)


def _rename_dividend_premium_sentiment_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(frame).copy()
    return frame.rename(
        columns={
            "ESENT_REEST": "ESENT_REEST_DIV_PREMIUM",
            "ESENT_PUBLISHED": "ESENT_PUBLISHED_BASE_WEIGHTS_DIV_PREMIUM",
            "RSENT_REEST": "RSENT_REEST_DIV_PREMIUM",
            "SENT": "SENT_DIV_PREMIUM",
            "SENT_ORTH": "SENT_ORTH_DIV_PREMIUM",
            "sentiment_raw": "sentiment_raw_div_premium",
            "sentiment_orth": "sentiment_orth_div_premium",
            "s0": "s0_div_premium",
            "s0_lag": "s0_div_premium_lag",
        }
    )


def build_sweden_sentiment_index_with_dividend_premium(
    paper_spec: pd.DataFrame,
    proxy_mart: pd.DataFrame,
    macro_controls: pd.DataFrame | None = None,
    *,
    macro_control_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build a Baker-Wurgler-style seven-proxy index with dividend premium added."""

    monthly, _, diagnostics = build_sweden_sentiment_index(
        paper_spec,
        proxy_mart,
        macro_controls=macro_controls,
        additional_base_variables=[DIVIDEND_PREMIUM_PROXY],
        macro_control_columns=macro_control_columns,
    )
    monthly = _rename_dividend_premium_sentiment_columns(monthly)
    annual = annualize_sentiment_index(monthly)
    diagnostics = {
        **diagnostics,
        "dividend_premium_proxy_role": pd.DataFrame(
            {
                "proxy_code": [DIVIDEND_PREMIUM_PROXY],
                "role": ["Baker-Wurgler dividend payer premium added as a seventh sentiment proxy"],
                "primary_construction": [
                    "log(value-weighted average market-to-book of regular dividend payers) "
                    "minus log(value-weighted average market-to-book of non-payers)"
                ],
            }
        ),
    }
    return monthly, annual, diagnostics


def assert_sweden_paper_spec_ready(paper_spec: pd.DataFrame) -> None:
    if paper_spec is None or paper_spec.empty:
        raise RuntimeError("Sweden paper-spec audit is empty. Lock the paper proxy set before building the final sentiment index.")

    audit = paper_spec.copy()
    if "required_for_final_index" not in audit.columns or "build_status" not in audit.columns:
        raise KeyError("Paper-spec audit must contain required_for_final_index and build_status columns.")

    required = audit.loc[audit["required_for_final_index"].fillna(False)].copy()
    missing = required.loc[required["build_status"].astype("string").ne("exact")]
    if not missing.empty:
        names = ", ".join(sorted(missing["paper_proxy_name"].astype(str).tolist()))
        raise RuntimeError(
            "Strict Sweden sentiment replication is not ready. The following required paper proxies are not exact: "
            f"{names}"
        )


def _zscore_with_sample(
    frame: pd.DataFrame,
    neutral_impute_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    neutral_columns = {column for column in (neutral_impute_columns or set()) if column in numeric.columns}
    if neutral_columns:
        mean = numeric.mean()
        std = numeric.std(ddof=0)
    else:
        sample = numeric.dropna()
        if sample.empty:
            raise RuntimeError("No complete observations are available for PCA estimation.")
        mean = sample.mean()
        std = sample.std(ddof=0)
    bad = std.isna() | std.eq(0)
    if bad.any():
        names = ", ".join(std.index[bad].tolist())
        raise RuntimeError(f"Cannot estimate PCA because these variables have zero or missing variance: {names}")
    z = (numeric - mean) / std
    if neutral_columns:
        z.loc[:, sorted(neutral_columns)] = z.loc[:, sorted(neutral_columns)].fillna(0.0)
        sample = z.dropna()
        if sample.empty:
            raise RuntimeError("No complete observations are available for PCA estimation.")
    return z, mean, std


def _pca_score_coefficients(
    frame: pd.DataFrame,
    label: str,
    min_cumulative: float = 0.85,
    neutral_impute_columns: set[str] | None = None,
) -> tuple[pd.Series, dict[str, pd.DataFrame], pd.DataFrame]:
    z, mean, std = _zscore_with_sample(frame, neutral_impute_columns=neutral_impute_columns)
    sample = z.dropna()
    pca = PCA(n_components=min(sample.shape[0], sample.shape[1]), random_state=0)
    pca.fit(sample)

    eigenvalues = pd.Series(pca.explained_variance_, name="eigenvalue")
    shares = eigenvalues / eigenvalues.sum()
    cumulative = shares.cumsum()
    retained = int(np.searchsorted(cumulative.to_numpy(), min_cumulative, side="left") + 1)

    loadings = pd.DataFrame(
        pca.components_[:retained].T,
        index=frame.columns,
        columns=[f"Comp{i}" for i in range(1, retained + 1)],
    )
    for column in loadings.columns:
        if loadings[column].sum() < 0:
            loadings[column] = -loadings[column]

    weighted = loadings.mul(shares.iloc[:retained].to_numpy(), axis=1)
    coefficients = weighted.sum(axis=1).rename("coefficient")
    diagnostics = {
        f"{label}_eigenvalues": pd.DataFrame(
            {
                "component": [f"Comp{i}" for i in range(1, len(eigenvalues) + 1)],
                "eigenvalue": eigenvalues.to_numpy(),
                "proportion": shares.to_numpy(),
                "cumulative": cumulative.to_numpy(),
                "retained_flag": [i < retained for i in range(len(eigenvalues))],
            }
        ),
        f"{label}_loadings": loadings.reset_index(names="variable"),
        f"{label}_weighted_coefficients": weighted.assign(sum=coefficients).reset_index(names="variable"),
        f"{label}_sample_stats": pd.DataFrame(
            {
                "variable": frame.columns,
                "sample_mean": mean.reindex(frame.columns).to_numpy(),
                "sample_std": std.reindex(frame.columns).to_numpy(),
                "neutral_impute_after_standardization": [
                    column in (neutral_impute_columns or set()) for column in frame.columns
                ],
            }
        ),
    }
    return coefficients, diagnostics, z


def _weighted_index(z_frame: pd.DataFrame, coefficients: pd.Series) -> pd.Series:
    aligned = z_frame.reindex(columns=coefficients.index)
    return aligned.mul(coefficients, axis=1).sum(axis=1, min_count=len(coefficients))


def _lead_lag_selection(
    frame: pd.DataFrame,
    sentiment: pd.Series,
    *,
    base_variables: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    base_variables = BASE_SWEDEN_SENTIMENT_PROXIES if base_variables is None else list(base_variables)
    pairs = [(variable, f"{variable}_L1") for variable in base_variables]
    rows: list[dict[str, Any]] = []
    selected: list[str] = []
    for current, lagged in pairs:
        corr_current = frame[current].corr(sentiment) if current in frame.columns else np.nan
        corr_lagged = frame[lagged].corr(sentiment) if lagged in frame.columns else np.nan
        if pd.isna(corr_current) and pd.isna(corr_lagged):
            chosen = current
        elif pd.isna(corr_lagged):
            chosen = current
        elif pd.isna(corr_current):
            chosen = lagged
        else:
            chosen = current if corr_current >= corr_lagged else lagged
        selected.append(chosen)
        rows.append(
            {
                "base_proxy": current,
                "current_proxy": current,
                "lagged_proxy": lagged,
                "corr_current": corr_current,
                "corr_lagged": corr_lagged,
                "selected_proxy": chosen,
            }
        )
    return pd.DataFrame(rows), selected


def _published_esent(frame: pd.DataFrame) -> pd.Series:
    return (
        ESENT_WEIGHTS["ESI"] * _series_from_frame(frame, "ESI")
        + ESENT_WEIGHTS["NIPO"] * _series_from_frame(frame, "NIPO")
        + ESENT_WEIGHTS["ED_RATIO"] * _series_from_frame(frame, "ED_RATIO")
        + ESENT_WEIGHTS["CCI_L1"] * _series_from_frame(frame, "CCI_L1")
        + ESENT_WEIGHTS["TURN_L1"] * _series_from_frame(frame, "TURN_L1")
        + ESENT_WEIGHTS["RIPO_L1"] * _series_from_frame(frame, "RIPO_L1")
    )


def _min_max_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.apply(pd.to_numeric, errors="coerce")
    minimum = out.min()
    maximum = out.max()
    span = maximum - minimum
    return out.sub(minimum, axis=1).div(span.replace(0, np.nan), axis=1)


def _ols_residual(series: pd.Series, regressors: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    x = regressors.apply(pd.to_numeric, errors="coerce").copy()
    x["intercept"] = 1.0
    valid = y.notna() & x.notna().all(axis=1)
    residual = pd.Series(np.nan, index=series.index, dtype="float64")
    if valid.sum() <= x.shape[1]:
        return residual
    beta, *_ = np.linalg.lstsq(x.loc[valid].to_numpy(dtype=float), y.loc[valid].to_numpy(dtype=float), rcond=None)
    fitted = x.loc[valid].to_numpy(dtype=float) @ beta
    residual.loc[valid] = y.loc[valid].to_numpy(dtype=float) - fitted
    return residual


def _macro_adjusted_index(
    frame: pd.DataFrame,
    selected_variables: list[str],
    macro_controls: pd.DataFrame | None,
    *,
    macro_control_columns: list[str] | None = None,
) -> tuple[pd.Series, dict[str, pd.DataFrame]]:
    if macro_controls is None or macro_controls.empty:
        return pd.Series(np.nan, index=frame.index, dtype="float64"), {}

    macro = macro_controls.copy()
    macro["month_end_date"] = pd.to_datetime(macro["month_end_date"], errors="coerce")
    macro = _sentiment_analysis_filter(macro)
    required_macro_columns = list(macro_control_columns or ["CPI", "PPI", "IP", "EM"])
    macro_columns = [column for column in required_macro_columns if column in macro.columns]
    if len(macro_columns) < len(required_macro_columns):
        return pd.Series(np.nan, index=frame.index, dtype="float64"), {}

    normalized = _min_max_normalize(macro[macro_columns])
    normalized["month_end_date"] = macro["month_end_date"].to_numpy()
    merged = frame[["country_code", "month_end_date", *selected_variables]].merge(normalized, on="month_end_date", how="left")
    residual_columns: list[str] = []
    diagnostics: dict[str, pd.DataFrame] = {}
    for variable in selected_variables:
        residual_name = f"r{variable}"
        merged[residual_name] = _ols_residual(merged[variable], merged[macro_columns])
        residual_columns.append(residual_name)
    residual_frame = merged[residual_columns]
    neutral_columns = {column for column in residual_columns if "RIPO" in column}
    coefficients, pca_diags, z_residual = _pca_score_coefficients(
        residual_frame,
        "stage3",
        neutral_impute_columns=neutral_columns,
    )
    diagnostics.update(pca_diags)
    diagnostics["stage3_selected_variables"] = pd.DataFrame({"selected_proxy": residual_columns})
    diagnostics["stage3_macro_controls"] = pd.DataFrame({"macro_control": macro_columns})
    return _weighted_index(z_residual, coefficients), diagnostics


def build_sweden_sentiment_index(
    paper_spec: pd.DataFrame,
    proxy_mart: pd.DataFrame,
    macro_controls: pd.DataFrame | None = None,
    *,
    selected_variables_override: list[str] | None = None,
    additional_base_variables: list[str] | None = None,
    macro_control_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    assert_sweden_paper_spec_ready(paper_spec)
    if proxy_mart is None or proxy_mart.empty:
        raise RuntimeError("Sentiment proxy mart is empty.")

    frame = build_sweden_sentiment_proxy_mart(proxy_mart)
    wide = (
        frame.pivot_table(
            index=["country_code", "month_end_date"],
            columns="proxy_code",
            values="raw_value",
            aggfunc="last",
        )
        .sort_index()
        .reset_index()
    )
    if wide.empty:
        raise RuntimeError("Sentiment proxy mart could not be pivoted into the paper proxy layout.")

    base_variables = list(BASE_SWEDEN_SENTIMENT_PROXIES)
    for variable in additional_base_variables or []:
        if variable not in base_variables:
            base_variables.append(variable)
    for column in base_variables:
        if column not in wide.columns:
            wide[column] = np.nan
        wide[f"{column}_L1"] = wide.groupby("country_code")[column].shift(1)

    stage1_variables = [column for variable in base_variables for column in [variable, f"{variable}_L1"]]
    ripo_neutral_columns = {column for column in stage1_variables if "RIPO" in column}
    stage1_coefficients, stage1_diags, z_stage1 = _pca_score_coefficients(
        wide[stage1_variables],
        "stage1",
        neutral_impute_columns=ripo_neutral_columns,
    )
    wide["sent_stage1"] = _weighted_index(z_stage1, stage1_coefficients)
    lead_lag_diagnostics, selected_variables = _lead_lag_selection(
        wide[stage1_variables],
        wide["sent_stage1"],
        base_variables=base_variables,
    )
    if selected_variables_override is not None:
        selected_variables = list(selected_variables_override)
        unknown = sorted(set(selected_variables).difference(stage1_variables))
        if unknown:
            raise KeyError(f"Selected sentiment lead-lag variables are not available: {unknown}")
        fixed_by_base = {
            variable.removesuffix("_L1"): variable
            for variable in selected_variables
        }
        lead_lag_diagnostics = lead_lag_diagnostics.assign(
            data_selected_proxy=lead_lag_diagnostics["selected_proxy"],
            selected_proxy=lead_lag_diagnostics["base_proxy"].map(fixed_by_base),
            selection_mode="fixed_selected_variables",
        )

    stage2_coefficients, stage2_diags, z_stage2 = _pca_score_coefficients(
        wide[selected_variables],
        "stage2",
        neutral_impute_columns={column for column in selected_variables if "RIPO" in column},
    )
    wide["ESENT_REEST"] = _weighted_index(z_stage2, stage2_coefficients)
    wide["ESENT_PUBLISHED"] = _published_esent(wide)
    rsent, stage3_diags = _macro_adjusted_index(
        wide,
        selected_variables,
        macro_controls,
        macro_control_columns=macro_control_columns,
    )
    wide["RSENT_REEST"] = rsent
    wide["SENT"] = wide["ESENT_REEST"]
    wide["SENT_ORTH"] = wide["RSENT_REEST"]
    wide["sentiment_raw"] = wide["ESENT_REEST"]
    wide["sentiment_orth"] = wide["RSENT_REEST"]
    wide["s0"] = np.where(wide["SENT"].notna(), np.where(wide["SENT"] >= 0, 1, 0), np.nan)
    wide["s0_lag"] = wide.groupby("country_code")["s0"].shift(1)
    monthly = _sentiment_analysis_filter(wide)
    annual = annualize_sentiment_index(monthly)
    diagnostics = {
        **stage1_diags,
        **stage2_diags,
        **stage3_diags,
        "lead_lag_selection": lead_lag_diagnostics,
        "stage2_selected_variables": pd.DataFrame({"selected_proxy": selected_variables}),
        "paper_published_weights": pd.DataFrame(
            {
                "selected_proxy": list(ESENT_WEIGHTS.keys()),
                "published_weight": list(ESENT_WEIGHTS.values()),
            }
        ),
        "stage2_reestimated_weights": stage2_coefficients.rename_axis("selected_proxy").reset_index(),
    }
    return monthly, annual, diagnostics
