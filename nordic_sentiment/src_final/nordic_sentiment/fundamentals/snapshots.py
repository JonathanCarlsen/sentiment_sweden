from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from nordic_sentiment.fundamentals.book_equity import compute_book_equity_components

DIVIDEND_PAYER_FIELD_CANDIDATES = ["dvtq", "dvty", "dvy"]
DIVIDEND_TOTAL_FIELD_CANDIDATES: list[str] = []


def compute_effective_month_end(report_available_date: pd.Series) -> pd.Series:
    report_available_date = pd.to_datetime(report_available_date, errors="coerce")
    current_month_end = report_available_date + MonthEnd(0)
    return current_month_end.where(current_month_end > report_available_date, current_month_end + MonthEnd(1))


def _empty_filings() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "company_id",
            "fiscal_period_end",
            "report_available_date",
            "effective_month_end",
            "reporting_periodicity",
            "fallback_lag_days",
            "report_available_source",
        ]
    )


def _rolling_annual_sum(group: pd.DataFrame, source: str) -> pd.Series:
    periods_per_year = int(group["reporting_periodicity"].map({"Q": 4, "SA": 2}).iloc[0])
    series = pd.to_numeric(group[source], errors="coerce")
    return series.rolling(periods_per_year, min_periods=periods_per_year).sum()


def _periods_per_year(periodicity: object) -> int:
    return 2 if periodicity == "SA" else 4


def _compute_book_equity(frame: pd.DataFrame) -> pd.Series:
    return compute_book_equity_components(frame)["book_equity"]


_ACCOUNTING_COLUMNS_TO_STALE_NULL = {
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
    "book_equity_base",
    "book_equity_base_source",
    "book_equity_preferred_stock_adjustment",
    "book_equity_nonpositive_flag",
    "book_equity_zero_ceq_replaced_flag",
    "BE",
    "book_equity",
    "earning",
    "dividend_total",
    "dividend_positive_filing",
    "dividend_payer_year",
    "regular_dividend_payer_fundamental",
    "regular_dividend_payer_formation_flag",
    "consecutive_nondividend_years",
    "revenue_ttm",
    "earning_ttm",
    "lag_revenue_ttm",
    "GS",
}


def _stale_null_columns(frame: pd.DataFrame) -> list[str]:
    columns: set[str] = set()
    for column in frame.columns:
        column_text = str(column)
        column_lower = column_text.lower()
        if column_text in _ACCOUNTING_COLUMNS_TO_STALE_NULL:
            columns.add(column_text)
        elif column_lower.endswith("_ttm"):
            columns.add(column_text)
        elif (
            column_lower.startswith("lag_year_")
            or column_lower.startswith("lag_revenue_")
        ):
            columns.add(column_text)
    return sorted(columns)


def _add_fundamental_staleness_cap(
    snapshots: pd.DataFrame,
    *,
    max_staleness_days: int | None,
) -> pd.DataFrame:
    out = snapshots.copy()
    out["month_end_date"] = pd.to_datetime(out["month_end_date"], errors="coerce")
    out["effective_month_end"] = pd.to_datetime(out["effective_month_end"], errors="coerce")
    out["fundamental_staleness_days"] = (out["month_end_date"] - out["effective_month_end"]).dt.days
    out["max_staleness_days"] = pd.NA if max_staleness_days is None else int(max_staleness_days)

    book_equity_source = out.get(
        "BE",
        out.get("book_equity", pd.Series(np.nan, index=out.index)),
    )
    book_equity_before_cap = pd.to_numeric(book_equity_source, errors="coerce")
    out["stale_fundamental_had_book_equity_flag"] = False

    if max_staleness_days is None:
        out["stale_fundamental_flag"] = False
        out["fundamental_snapshot_status"] = "uncapped"
        return out

    stale = pd.to_numeric(out["fundamental_staleness_days"], errors="coerce").gt(int(max_staleness_days))
    out["stale_fundamental_flag"] = stale
    out["stale_fundamental_had_book_equity_flag"] = stale & book_equity_before_cap.notna()
    out["fundamental_snapshot_status"] = np.where(stale, "stale_capped", "fresh")
    stale_null_columns = _stale_null_columns(out)
    if stale_null_columns and stale.any():
        out.loc[stale, stale_null_columns] = np.nan
    return out


def _add_regular_dividend_payer_status(filings: pd.DataFrame) -> pd.DataFrame:
    """Classify regular dividend payers using only observable historical filings.

    A firm becomes a regular dividend payer after at least two dividend-paying
    years within the latest three observable fiscal years. Once classified, it
    keeps the flag through one missed dividend year and loses it after two
    consecutive non-dividend years.
    """

    out = filings.copy()
    for column in DIVIDEND_PAYER_FIELD_CANDIDATES:
        if column not in out.columns:
            out[column] = np.nan
    dividend_values = out[DIVIDEND_PAYER_FIELD_CANDIDATES].apply(pd.to_numeric, errors="coerce")
    out["dividend_positive_filing"] = dividend_values.gt(0).any(axis=1)
    out["dividend_field_observed_flag"] = dividend_values.notna().any(axis=1)
    out["fiscal_year"] = pd.to_datetime(out["fiscal_period_end"], errors="coerce").dt.year

    status_columns = [
        "dividend_payer_year",
        "regular_dividend_payer_fundamental",
        "regular_dividend_payer_formation_flag",
        "consecutive_nondividend_years",
    ]
    if "company_id" not in out.columns or out[["company_id", "fiscal_year"]].dropna().empty:
        for column in status_columns:
            out[column] = np.nan
        return out

    firm_year = (
        out.dropna(subset=["company_id", "fiscal_year"])
        .groupby(["company_id", "fiscal_year"], as_index=False)
        .agg(
            dividend_payer_year=("dividend_positive_filing", "max"),
            dividend_field_observed_flag=("dividend_field_observed_flag", "max"),
        )
        .sort_values(["company_id", "fiscal_year"])
    )

    records: list[dict[str, object]] = []
    for company_id, group in firm_year.groupby("company_id", sort=False):
        dividend_history: list[bool] = []
        regular = False
        consecutive_misses = 0
        for row in group.itertuples(index=False):
            paid = bool(row.dividend_payer_year)
            dividend_history.append(paid)
            formed_this_year = False

            if paid:
                consecutive_misses = 0
                if not regular and sum(dividend_history[-3:]) >= 2:
                    regular = True
                    formed_this_year = True
            elif regular:
                consecutive_misses += 1
                if consecutive_misses >= 2:
                    regular = False

            records.append(
                {
                    "company_id": company_id,
                    "fiscal_year": row.fiscal_year,
                    "dividend_payer_year": float(paid),
                    "regular_dividend_payer_fundamental": float(regular),
                    "regular_dividend_payer_formation_flag": bool(formed_this_year),
                    "consecutive_nondividend_years": int(consecutive_misses) if regular or consecutive_misses else 0,
                }
            )

    status = pd.DataFrame(records)
    out = out.merge(status, on=["company_id", "fiscal_year"], how="left", suffixes=("", "_status"))
    for column in status_columns:
        status_column = f"{column}_status"
        if status_column in out.columns:
            out[column] = out[status_column].where(out[status_column].notna(), out.get(column))
            out = out.drop(columns=status_column)
    return out


def build_fundamental_filings(fundamentals: pd.DataFrame, company_dim: pd.DataFrame) -> pd.DataFrame:
    if fundamentals is None or fundamentals.empty:
        return _empty_filings()

    filings = fundamentals.copy()
    if "company_id" not in filings.columns and not company_dim.empty and "gvkey" in filings.columns:
        company_lookup = company_dim[["company_id", "gvkey"]].drop_duplicates()
        filings = filings.merge(company_lookup, on="gvkey", how="left")
    if "fiscal_period_end" in filings.columns:
        filings["fiscal_period_end"] = pd.to_datetime(filings["fiscal_period_end"], errors="coerce")
    for column in ["pdateq", "fdateq", "final_date", "preliminary_date", "report_available_date"]:
        if column in filings.columns:
            filings[column] = pd.to_datetime(filings[column], errors="coerce")
    if "reporting_periodicity" not in filings.columns:
        filings["reporting_periodicity"] = "Q"
    filings["reporting_periodicity"] = filings["reporting_periodicity"].astype("string").fillna("Q")

    fallback_lag_days = filings["reporting_periodicity"].map({"Q": 90, "SA": 180}).fillna(90).astype(int)
    final_date = filings.get("fdateq", filings.get("final_date", pd.Series(pd.NaT, index=filings.index)))
    preliminary_date = filings.get("pdateq", filings.get("preliminary_date", pd.Series(pd.NaT, index=filings.index)))
    report_available = filings.get("report_available_date", pd.Series(pd.NaT, index=filings.index))
    report_available = report_available.fillna(final_date).fillna(preliminary_date)
    report_available = report_available.fillna(
        filings["fiscal_period_end"] + pd.to_timedelta(fallback_lag_days, unit="D")
    )
    filings["fallback_lag_days"] = fallback_lag_days
    filings["report_available_date"] = pd.to_datetime(report_available, errors="coerce")
    filings["report_available_source"] = np.select(
        [final_date.notna(), preliminary_date.notna()],
        ["final_date", "preliminary_date"],
        default="fallback_lag",
    )
    filings["effective_month_end"] = compute_effective_month_end(filings["report_available_date"])

    numeric_candidates = [
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
    ]
    for column in numeric_candidates:
        if column not in filings.columns:
            filings[column] = np.nan
        filings[column] = pd.to_numeric(filings[column], errors="coerce")

    book_equity_components = compute_book_equity_components(filings)
    for column in book_equity_components.columns:
        filings[column] = book_equity_components[column]
    filings["earning"] = filings.get("ib", pd.Series(np.nan, index=filings.index))
    filings["dividend_total"] = pd.Series(np.nan, index=filings.index, dtype="float64")
    for column in DIVIDEND_TOTAL_FIELD_CANDIDATES:
        if column in filings.columns:
            filings["dividend_total"] = filings["dividend_total"].fillna(filings[column])
    filings = _add_regular_dividend_payer_status(filings)

    filings = filings.sort_values(["company_id", "reporting_periodicity", "fiscal_period_end"]).reset_index(drop=True)
    group_id = (
        filings["company_id"].astype("string").fillna("")
        + "|"
        + filings["reporting_periodicity"].astype("string").fillna("Q")
    )
    for source, target in {
        "revenue": "revenue_ttm",
        "earning": "earning_ttm",
    }.items():
        if source in filings.columns:
            filings[target] = filings.groupby(group_id, group_keys=False)[source].apply(
                lambda series: series.rolling(
                    _periods_per_year(filings.loc[series.index[0], "reporting_periodicity"]),
                    min_periods=_periods_per_year(filings.loc[series.index[0], "reporting_periodicity"]),
                ).sum()
            )
        else:
            filings[target] = np.nan

    filings["lag_revenue_ttm"] = np.nan
    annual_periods = filings["reporting_periodicity"].map({"Q": 4, "SA": 2}).fillna(4).astype(int)
    for lag_period in sorted(annual_periods.unique()):
        mask = annual_periods.eq(int(lag_period))
        filings.loc[mask, "lag_revenue_ttm"] = (
            filings.loc[mask].groupby("company_id")["revenue_ttm"].shift(int(lag_period))
        )

    filings["GS"] = np.where(
        filings["lag_revenue_ttm"].gt(0) & filings["revenue_ttm"].ge(0),
        filings["revenue_ttm"] / filings["lag_revenue_ttm"] - 1.0,
        np.nan,
    )
    return filings


def build_monthly_fundamental_snapshots(
    quarterly_fundamentals: pd.DataFrame,
    calendar_month: pd.DataFrame,
    *,
    max_staleness_days: int | None = 365,
) -> pd.DataFrame:
    if quarterly_fundamentals is None or quarterly_fundamentals.empty or calendar_month is None or calendar_month.empty:
        return pd.DataFrame(
            columns=[
                "company_id",
                "month_end_date",
                "fiscal_period_end",
                "report_available_date",
                "effective_month_end",
                "book_equity",
                "fundamental_staleness_days",
                "max_staleness_days",
                "stale_fundamental_flag",
                "stale_fundamental_had_book_equity_flag",
                "fundamental_snapshot_status",
            ]
        )

    filings = quarterly_fundamentals.copy()
    if "report_available_date" not in filings.columns:
        filings = build_fundamental_filings(filings, pd.DataFrame())
    filings["report_available_date"] = pd.to_datetime(filings["report_available_date"], errors="coerce")
    if "effective_month_end" not in filings.columns:
        filings["effective_month_end"] = compute_effective_month_end(filings["report_available_date"])
    filings = filings.sort_values(["effective_month_end", "company_id", "fiscal_period_end"])

    months = calendar_month[["month_end_date"]].copy()
    months["month_end_date"] = pd.to_datetime(months["month_end_date"], errors="coerce")
    companies = filings[["company_id"]].drop_duplicates()
    grid = companies.assign(_tmp=1).merge(months.assign(_tmp=1), on="_tmp").drop(columns="_tmp")
    grid = grid.sort_values(["month_end_date", "company_id"])

    snapshots = pd.merge_asof(
        grid,
        filings,
        left_on="month_end_date",
        right_on="effective_month_end",
        by="company_id",
        direction="backward",
        allow_exact_matches=True,
    )
    snapshots = snapshots.loc[
        snapshots["effective_month_end"].notna() & (snapshots["effective_month_end"] <= snapshots["month_end_date"])
    ]
    snapshots = _add_fundamental_staleness_cap(
        snapshots.reset_index(drop=True),
        max_staleness_days=max_staleness_days,
    )
    return snapshots.reset_index(drop=True)
