from __future__ import annotations

import numpy as np
import pandas as pd

from nordic_sentiment.fundamentals.book_equity import compute_book_equity_components


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    normalized: list[str] = []
    for column in out.columns:
        text = str(column).strip()
        if text.startswith("(") and ")" in text:
            token = text[1 : text.index(")")].strip().lower()
            normalized.append(token)
        else:
            normalized.append(text.lower())
    out.columns = normalized
    return out


def _standardize_string_codes(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].astype("string").str.strip()
            out[column] = out[column].replace({"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out


def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    values = pd.Series(series, copy=False)
    if values.empty:
        return pd.to_datetime(values, errors="coerce")

    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    numeric = pd.to_numeric(values, errors="coerce")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        out.loc[numeric_mask] = pd.to_datetime(
            numeric.loc[numeric_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    remaining_mask = out.isna() & values.notna()
    if remaining_mask.any():
        text = values.loc[remaining_mask].astype("string").str.strip()
        text = text.replace({"": pd.NA, "<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})
        out.loc[remaining_mask] = pd.to_datetime(text, errors="coerce", format="mixed")

    return out


def _coerce_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            values = out[column]
            if values.dtype == object or str(values.dtype).startswith("string"):
                values = values.astype("string").str.strip().str.replace(",", ".", regex=False)
            out[column] = pd.to_numeric(values, errors="coerce")
    return out


def _compute_book_equity(frame: pd.DataFrame) -> pd.Series:
    return compute_book_equity_components(frame)["book_equity"]


def stage_sweden_daily_prices(frame: pd.DataFrame, country_code: str = "SWE") -> pd.DataFrame:
    out = _normalize_columns(frame)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "country_code",
                "gvkey",
                "iid",
                "trade_date",
                "prccd",
                "cshoc",
                "trfd",
                "isin",
                "conm",
                "fic",
                "exchg",
                "tpci",
                "monthend",
            ]
        )
    out = _standardize_string_codes(out, ["gvkey", "iid", "isin", "tpci", "fic", "exchg", "conm"])
    out["country_code"] = country_code
    if "datadate" in out.columns:
        out["trade_date"] = _parse_mixed_dates(out["datadate"])
    elif "trade_date" in out.columns:
        out["trade_date"] = _parse_mixed_dates(out["trade_date"])
    if "monthend" in out.columns:
        out["monthend"] = out["monthend"].astype("string")
    return out


def stage_sweden_monthly_prices(frame: pd.DataFrame, country_code: str = "SWE") -> pd.DataFrame:
    out = _normalize_columns(frame)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "country_code",
                "gvkey",
                "iid",
                "trade_date",
                "month_end_date",
                "prccd",
                "cshoc",
                "trfd",
                "isin",
            ]
        )
    out = _standardize_string_codes(out, ["gvkey", "iid", "isin"])
    out["country_code"] = country_code
    rename_map = {
        "prccm": "prccd",
    }
    out = out.rename(columns={key: value for key, value in rename_map.items() if key in out.columns})
    source_date = "datadate" if "datadate" in out.columns else "trade_date"
    out["trade_date"] = _parse_mixed_dates(out[source_date])
    out["month_end_date"] = out["trade_date"].dt.to_period("M").dt.to_timestamp("M")
    return out


def _infer_periodicity(frame: pd.DataFrame) -> pd.Series:
    if "reporting_periodicity" in frame.columns:
        return frame["reporting_periodicity"].astype("string").fillna("Q")
    if "gvkey" not in frame.columns or "fiscal_period_end" not in frame.columns:
        return pd.Series("Q", index=frame.index, dtype="string")

    sorted_frame = frame[["gvkey", "fiscal_period_end"]].copy()
    sorted_frame["gap_days"] = (
        sorted_frame.sort_values(["gvkey", "fiscal_period_end"])
        .groupby("gvkey")["fiscal_period_end"]
        .diff()
        .dt.days
    )
    periodicity = pd.Series("Q", index=frame.index, dtype="string")
    periodicity.loc[sorted_frame["gap_days"].gt(120).fillna(False)] = "SA"
    return periodicity


def stage_sweden_quarterly_fundamentals(frame: pd.DataFrame, country_code: str = "SWE") -> pd.DataFrame:
    out = _normalize_columns(frame)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "country_code",
                "gvkey",
                "fiscal_period_end",
                "revenue",
                "at",
                "ceq",
                "pdateq",
                "fdateq",
                "reporting_periodicity",
                "report_available_date",
            ]
        )
    out = _standardize_string_codes(out, ["gvkey", "indfmt", "consol", "popsrc", "datafmt"])
    out["country_code"] = country_code
    rename_map = {
        "atq": "at",
        "ceqq": "ceq",
        "ibq": "ib",
        "ltq": "ltq",
        "seqq": "seqq",
        "pstkq": "pstkq",
    }
    out = out.rename(columns={key: value for key, value in rename_map.items() if key in out.columns})
    if "revtq" in out.columns:
        out["revenue"] = out["revtq"]
    if "ppentq" in out.columns:
        if "ppegt" in out.columns:
            out["ppegt"] = out["ppegt"].where(out["ppegt"].notna(), out["ppentq"])
        else:
            out = out.rename(columns={"ppentq": "ppegt"})
    out = _coerce_numeric_columns(
        out,
        [
            "revtq",
            "revenue",
            "at",
            "ceq",
            "seqq",
            "ltq",
            "pstkq",
            "ib",
            "ppegt",
            "dvtq",
            "dvty",
            "dvy",
        ],
    )
    source_date = "datadate" if "datadate" in out.columns else "fiscal_period_end"
    out["fiscal_period_end"] = _parse_mixed_dates(out[source_date])
    for column in ["pdateq", "fdateq", "ipodate"]:
        if column in out.columns:
            out[column] = _parse_mixed_dates(out[column])
    out["reporting_periodicity"] = _infer_periodicity(out)
    fallback_days = out["reporting_periodicity"].map({"Q": 90, "SA": 180}).fillna(90)
    final_date = out["fdateq"] if "fdateq" in out.columns else pd.Series(pd.NaT, index=out.index)
    preliminary_date = out["pdateq"] if "pdateq" in out.columns else pd.Series(pd.NaT, index=out.index)
    out["report_available_date"] = final_date.fillna(preliminary_date).fillna(
        out["fiscal_period_end"] + pd.to_timedelta(fallback_days, unit="D")
    )
    book_equity_components = compute_book_equity_components(out)
    for column in book_equity_components.columns:
        out[column] = book_equity_components[column]
    return out


def stage_sweden_ipo_offers(frame: pd.DataFrame, country_code: str = "SWE") -> pd.DataFrame:
    out = _normalize_columns(frame)
    columns = [
        "country_code",
        "bloomberg_ticker",
        "company_name_bloomberg",
        "ipo_date",
        "ipo_offer_price",
        "primary_exchange_name",
        "isin",
        "currency",
        "bloomberg_offer_to_first_open_return_pct",
        "bloomberg_offer_to_first_open_return",
    ]
    if out.empty:
        return pd.DataFrame(columns=columns)

    rename_map = {
        "ticker": "bloomberg_ticker",
        "name": "company_name_bloomberg",
        "ipo dt": "ipo_date",
        "ipo sh px": "ipo_offer_price",
        "prim exch nm": "primary_exchange_name",
        "curncy": "currency",
        "ipo offer px 1st opn px % chg": "bloomberg_offer_to_first_open_return_pct",
    }
    out = out.rename(columns={key: value for key, value in rename_map.items() if key in out.columns})
    out = _standardize_string_codes(
        out,
        ["bloomberg_ticker", "company_name_bloomberg", "primary_exchange_name", "isin", "currency"],
    )
    out["country_code"] = country_code
    if "ipo_date" in out.columns:
        out["ipo_date"] = _parse_mixed_dates(out["ipo_date"])
    else:
        out["ipo_date"] = pd.NaT
    out = _coerce_numeric_columns(out, ["ipo_offer_price", "bloomberg_offer_to_first_open_return_pct"])
    # Bloomberg's "% Chg" export is stored as a decimal return; multiply by 100
    # only for the percentage-points companion column.
    out["bloomberg_offer_to_first_open_return"] = out["bloomberg_offer_to_first_open_return_pct"]
    out["bloomberg_offer_to_first_open_return_pct"] = out["bloomberg_offer_to_first_open_return"] * 100.0
    if "isin" in out.columns:
        out["isin"] = out["isin"].astype("string").str.strip().str.upper()
    else:
        out["isin"] = pd.Series(pd.NA, index=out.index, dtype="string")

    valid = (
        out["isin"].str.len().eq(12).fillna(False)
        & out["ipo_date"].notna()
        & out["ipo_offer_price"].notna()
        & out["ipo_offer_price"].gt(0)
    )
    out = out.loc[valid].copy()
    return out.reindex(columns=columns).sort_values(["ipo_date", "isin"]).reset_index(drop=True)
