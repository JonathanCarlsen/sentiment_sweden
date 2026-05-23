from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_tabular_file(path: str | Path, file_format: str, **kwargs: Any) -> pd.DataFrame:
    """Load a supported file into a DataFrame.

    Missing files resolve to empty frames so the notebooks can still import and
    execute before local raw inputs are populated.
    """

    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()

    fmt = file_format.lower()
    if fmt == "csv":
        return pd.read_csv(file_path, **kwargs)
    if fmt == "parquet":
        return pd.read_parquet(file_path, **kwargs)
    if fmt == "dta":
        return pd.read_stata(file_path, **kwargs)
    if fmt == "xlsx":
        return pd.read_excel(file_path, **kwargs)
    raise ValueError(f"Unsupported file format: {file_format}")
