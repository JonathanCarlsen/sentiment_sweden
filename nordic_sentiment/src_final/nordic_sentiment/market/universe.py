from __future__ import annotations

import numpy as np
import pandas as pd


EQUITY_UNIVERSE_AUDIT_COLUMNS = [
    "country_code",
    "company_id",
    "security_id",
    "month_end_date",
    "isin",
    "isin_clean",
    "iid",
    "iid_clean",
    "exchange_code",
    "exchange_clean",
    "issue_type_code",
    "issue_type_clean",
    "valid_isin_flag",
    "swedish_isin_flag",
    "known_common_equity_flag",
    "positive_price_flag",
    "candidate_primary_line_flag",
    "selected_primary_line_flag",
    "primary_selection_rank",
    "company_month_security_count",
    "company_month_candidate_count",
    "selection_reason",
]


def _clean_code(series: pd.Series, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series(pd.NA, index=index, dtype="string")
    values = series.astype("string").str.strip().str.upper()
    values = values.str.replace(r"\.0$", "", regex=True)
    return values.replace({"": pd.NA, "NAN": pd.NA, "<NA>": pd.NA})


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(pd.NA, index=frame.index)


def _valid_isin(series: pd.Series) -> pd.Series:
    values = _clean_code(series, index=series.index)
    return values.str.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", na=False)


def _primary_iid_rank(series: pd.Series) -> pd.Series:
    values = _clean_code(series, index=series.index)
    return pd.Series(
        np.select(
            [
                values.eq("01W"),
                values.str.match(r"^01", na=False),
                values.str.match(r"^0?1", na=False),
            ],
            [0, 1, 2],
            default=9,
        ),
        index=series.index,
        dtype="int64",
    )


def _empty_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=EQUITY_UNIVERSE_AUDIT_COLUMNS)


def filter_primary_sweden_equity_universe(
    market_monthly: pd.DataFrame,
    *,
    country_code: str = "SWE",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one valid common-equity security line per company-month.

    The production filter is deliberately conservative:

    1. valid ISIN;
    2. known common-equity issue code (`0`, `EQ`, `COM`);
    3. positive month-end price;
    4. one selected line per `company_id, month_end_date`.

    If the supplied frame has no usable metadata at all, the function returns
    the input unchanged and records that the filter was not applied. This keeps
    small synthetic tests and partially populated development inputs from being
    silently emptied.
    """

    if market_monthly is None or market_monthly.empty:
        empty = pd.DataFrame(market_monthly).copy()
        return empty, _empty_audit()

    frame = market_monthly.copy()
    frame["country_code"] = _series(frame, "country_code").fillna(country_code)
    frame["month_end_date"] = pd.to_datetime(frame.get("month_end_date"), errors="coerce")
    frame["company_id"] = _series(frame, "company_id").astype("string")
    frame["security_id"] = _series(frame, "security_id").astype("string")
    frame["isin_clean"] = _clean_code(_series(frame, "isin"), index=frame.index)
    frame["iid_clean"] = _clean_code(_series(frame, "iid"), index=frame.index)
    frame["exchange_clean"] = _clean_code(_series(frame, "exchange_code"), index=frame.index)
    frame["issue_type_clean"] = _clean_code(_series(frame, "issue_type_code"), index=frame.index)

    frame["valid_isin_flag"] = _valid_isin(frame["isin_clean"])
    frame["swedish_isin_flag"] = frame["isin_clean"].str.startswith("SE", na=False)
    frame["known_common_equity_flag"] = frame["issue_type_clean"].isin(["0", "EQ", "COM"])
    frame["positive_price_flag"] = pd.to_numeric(frame.get("month_end_price"), errors="coerce").gt(0)
    frame["candidate_primary_line_flag"] = (
        frame["valid_isin_flag"] & frame["known_common_equity_flag"] & frame["positive_price_flag"]
    )
    frame["company_month_security_count"] = frame.groupby(["company_id", "month_end_date"])["security_id"].transform(
        "nunique"
    )
    frame["company_month_candidate_count"] = frame.groupby(["company_id", "month_end_date"])[
        "candidate_primary_line_flag"
    ].transform("sum")

    if not frame["candidate_primary_line_flag"].any():
        audit = frame.copy()
        audit["selected_primary_line_flag"] = True
        audit["primary_selection_rank"] = 1
        audit["selection_reason"] = "metadata_unavailable_unfiltered"
        return market_monthly.copy(), audit.reindex(columns=EQUITY_UNIVERSE_AUDIT_COLUMNS)

    candidates = frame.loc[frame["candidate_primary_line_flag"]].copy()
    active_months = candidates.groupby("security_id")["month_end_date"].nunique().rename("security_active_months")
    candidates = candidates.merge(active_months, left_on="security_id", right_index=True, how="left")
    candidates["_iid_rank"] = _primary_iid_rank(candidates["iid_clean"])
    candidates["_swedish_isin_rank"] = candidates["swedish_isin_flag"].astype(int)
    candidates["_market_cap_rank"] = pd.to_numeric(
        _series(candidates, "month_end_market_cap"),
        errors="coerce",
    ).fillna(-np.inf)
    candidates["_volume_rank"] = pd.to_numeric(
        _series(candidates, "month_end_trading_volume"),
        errors="coerce",
    ).fillna(-np.inf)
    candidates["_history_rank"] = pd.to_numeric(candidates.get("security_active_months"), errors="coerce").fillna(
        -np.inf
    )
    candidates["_source_rank"] = np.where(
        _series(candidates, "source_name").astype("string").eq("monthly_workbook").fillna(False),
        0,
        1,
    )

    ordered = candidates.sort_values(
        [
            "company_id",
            "month_end_date",
            "_iid_rank",
            "_swedish_isin_rank",
            "_market_cap_rank",
            "_volume_rank",
            "_history_rank",
            "_source_rank",
            "security_id",
        ],
        ascending=[True, True, True, False, False, False, False, True, True],
    ).copy()
    ordered["primary_selection_rank"] = ordered.groupby(["company_id", "month_end_date"]).cumcount() + 1
    selected_keys = ordered.loc[
        ordered["primary_selection_rank"].eq(1),
        ["company_id", "month_end_date", "security_id", "primary_selection_rank"],
    ]

    frame = frame.merge(
        selected_keys.assign(selected_primary_line_flag=True),
        on=["company_id", "month_end_date", "security_id"],
        how="left",
    )
    frame["selected_primary_line_flag"] = frame["selected_primary_line_flag"].eq(True)
    frame["primary_selection_rank"] = frame["primary_selection_rank"].where(frame["candidate_primary_line_flag"])
    frame["selection_reason"] = np.select(
        [
            frame["selected_primary_line_flag"],
            ~frame["valid_isin_flag"],
            ~frame["known_common_equity_flag"],
            ~frame["positive_price_flag"],
            frame["candidate_primary_line_flag"] & ~frame["selected_primary_line_flag"],
        ],
        [
            "selected_primary_line",
            "excluded_invalid_or_missing_isin",
            "excluded_non_common_equity_issue_type",
            "excluded_non_positive_price",
            "excluded_secondary_company_month_line",
        ],
        default="excluded_unclassified",
    )

    filtered = frame.loc[frame["selected_primary_line_flag"]].copy()
    helper_columns = [
        "isin_clean",
        "iid_clean",
        "exchange_clean",
        "issue_type_clean",
        "valid_isin_flag",
        "swedish_isin_flag",
        "known_common_equity_flag",
        "positive_price_flag",
        "candidate_primary_line_flag",
        "selected_primary_line_flag",
        "primary_selection_rank",
        "company_month_security_count",
        "company_month_candidate_count",
        "selection_reason",
    ]
    filtered = filtered.drop(columns=[column for column in helper_columns if column in filtered.columns])
    audit = frame.reindex(columns=EQUITY_UNIVERSE_AUDIT_COLUMNS).sort_values(
        ["company_id", "month_end_date", "selected_primary_line_flag", "security_id"],
        ascending=[True, True, False, True],
    )
    return filtered.reset_index(drop=True), audit.reset_index(drop=True)
