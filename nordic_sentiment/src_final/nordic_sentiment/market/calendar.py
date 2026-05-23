from __future__ import annotations

import pandas as pd


def build_calendar_day(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
    frame = pd.DataFrame({"calendar_date": dates})
    frame["year_num"] = frame["calendar_date"].dt.year
    frame["month_num"] = frame["calendar_date"].dt.month
    frame["quarter_num"] = frame["calendar_date"].dt.quarter
    frame["month_end_flag"] = frame["calendar_date"].dt.is_month_end
    frame["quarter_end_flag"] = frame["calendar_date"].dt.is_quarter_end
    frame["trading_day_flag"] = frame["calendar_date"].dt.dayofweek < 5
    return frame


def build_calendar_month(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="ME")
    frame = pd.DataFrame({"month_end_date": dates})
    frame["year_num"] = frame["month_end_date"].dt.year
    frame["month_num"] = frame["month_end_date"].dt.month
    frame["quarter_num"] = frame["month_end_date"].dt.quarter
    return frame


def build_month_calendar(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> pd.DataFrame:
    return build_calendar_month(start_date, end_date)
