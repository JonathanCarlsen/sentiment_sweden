from __future__ import annotations

from typing import Sequence

import pandas as pd


def dataset_audit(frame: pd.DataFrame, date_column: str | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            [{"rows": 0, "columns": 0, "min_date": None, "max_date": None, "null_cells": 0}]
        )

    min_date = None
    max_date = None
    if date_column and date_column in frame.columns:
        series = pd.to_datetime(frame[date_column], errors="coerce")
        min_date = series.min()
        max_date = series.max()

    return pd.DataFrame(
        [
            {
                "rows": int(len(frame)),
                "columns": int(frame.shape[1]),
                "min_date": min_date,
                "max_date": max_date,
                "null_cells": int(frame.isna().sum().sum()),
            }
        ]
    )


def duplicate_key_count(frame: pd.DataFrame, keys: Sequence[str]) -> int:
    if frame is None or frame.empty:
        return 0
    usable_keys = [key for key in keys if key in frame.columns]
    if len(usable_keys) != len(keys):
        return 0
    return int(frame.duplicated(usable_keys).sum())
