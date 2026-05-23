from __future__ import annotations

import pandas as pd


def annualize_sentiment_index(monthly_index: pd.DataFrame) -> pd.DataFrame:
    if monthly_index is None or monthly_index.empty:
        raise RuntimeError("Monthly sentiment index is required for annualization.")

    frame = monthly_index.copy()
    frame["month_end_date"] = pd.to_datetime(frame["month_end_date"], errors="coerce")
    frame["year_num"] = frame["month_end_date"].dt.year
    december = frame.loc[frame["month_end_date"].dt.month == 12]
    if december.empty:
        december = frame.sort_values("month_end_date").groupby(["country_code", "year_num"], as_index=False).tail(1)
    return december.reset_index(drop=True)
