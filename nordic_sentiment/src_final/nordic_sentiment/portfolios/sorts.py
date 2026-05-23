from __future__ import annotations

import numpy as np
import pandas as pd


def _infer_date_column(frame: pd.DataFrame) -> str:
    for candidate in ["month_end_date", "ym", "date"]:
        if candidate in frame.columns:
            return candidate
    raise KeyError("No supported date column found.")


def _bucketize(series: pd.Series, n_portfolios: int) -> pd.Series:
    valid = series.dropna()
    if valid.nunique() < 2:
        return pd.Series(pd.NA, index=series.index, dtype="Int64")
    ranks = valid.rank(method="first")
    buckets = pd.qcut(ranks, n_portfolios, labels=False, duplicates="drop") + 1
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    out.loc[valid.index] = buckets.astype("Int64")
    return out


def add_lagged_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    entity_column: str = "security_id",
    date_column: str | None = None,
    suffix: str = "_lag",
) -> pd.DataFrame:
    """Attach one-period lagged columns within each entity."""
    if frame is None or frame.empty:
        return pd.DataFrame(frame).copy()

    out = frame.copy()
    date_col = date_column or _infer_date_column(out)
    required = {entity_column, date_col, *columns}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise KeyError(f"Missing required columns for lagging: {missing}")

    out["_original_order"] = range(len(out))
    ordered = out.sort_values([entity_column, date_col, "_original_order"]).copy()
    for column in columns:
        ordered[f"{column}{suffix}"] = ordered.groupby(entity_column, dropna=False)[column].shift(1)
    ordered = ordered.sort_values("_original_order").drop(columns=["_original_order"])
    return ordered.reset_index(drop=True)


def add_lagged_sort_signals(
    frame: pd.DataFrame,
    sort_columns: list[str],
    *,
    entity_column: str = "security_id",
    date_column: str | None = None,
    suffix: str = "_lag",
) -> pd.DataFrame:
    """Attach one-period lagged sort signals within each security."""
    return add_lagged_columns(
        frame,
        sort_columns,
        entity_column=entity_column,
        date_column=date_column,
        suffix=suffix,
    )


def build_lagged_sort_signal_coverage(
    frame: pd.DataFrame,
    sort_specs: list[tuple[str, str]],
    *,
    value_column: str = "ret",
    date_column: str | None = None,
) -> pd.DataFrame:
    """Summarize monthly eligibility for lagged-signal portfolio sorts."""
    columns = [
        "sort_variable",
        "month_end_date",
        "rows",
        "lagged_signal_non_null",
        "return_non_null",
        "eligible_rows",
        "formed_portfolios",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)

    out = frame.copy()
    date_col = date_column or _infer_date_column(out)
    rows: list[dict[str, object]] = []
    for sort_variable, lagged_column in sort_specs:
        if lagged_column not in out.columns or value_column not in out.columns:
            continue
        for month_end_date, group in out.groupby(date_col, sort=True):
            signal = pd.to_numeric(group[lagged_column], errors="coerce")
            returns = pd.to_numeric(group[value_column], errors="coerce")
            eligible = signal.notna() & returns.notna()
            rows.append(
                {
                    "sort_variable": sort_variable,
                    "month_end_date": month_end_date,
                    "rows": int(len(group)),
                    "lagged_signal_non_null": int(signal.notna().sum()),
                    "return_non_null": int(returns.notna().sum()),
                    "eligible_rows": int(eligible.sum()),
                    "formed_portfolios": int(_bucketize(signal.where(eligible), 10).nunique(dropna=True)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_forward_compounded_return_windows(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    date_column: str,
    return_column: str,
    horizons: tuple[int, ...] = (1, 3, 6, 12),
    output_column: str = "portfolio_return_window",
    input_return_unit: str = "decimal",
    output_return_unit: str | None = None,
) -> pd.DataFrame:
    """Compute overlapping forward compounded returns within each group."""
    valid_units = {"decimal", "percent"}
    if input_return_unit not in valid_units:
        raise ValueError("input_return_unit must be either 'decimal' or 'percent'.")
    output_return_unit = input_return_unit if output_return_unit is None else output_return_unit
    if output_return_unit not in valid_units:
        raise ValueError("output_return_unit must be either 'decimal' or 'percent'.")

    columns = [*group_columns, date_column, "return_horizon_months", output_column]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("All return horizons must be positive.")

    required = {date_column, return_column, *group_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns for forward return windows: {missing}")

    out = frame.copy()
    out[date_column] = pd.to_datetime(out[date_column], errors="coerce")
    out[return_column] = pd.to_numeric(out[return_column], errors="coerce")
    out = out.dropna(subset=[date_column]).sort_values([*group_columns, date_column] if group_columns else [date_column])

    grouped = out.groupby(group_columns, dropna=False, sort=True) if group_columns else [((), out)]
    rows: list[dict[str, object]] = []
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group = group.sort_values(date_column).reset_index(drop=True)
        month_ordinals = group[date_column].dt.year.mul(12).add(group[date_column].dt.month).to_numpy()
        returns = group[return_column].to_numpy(dtype=float)
        group_values = dict(zip(group_columns, group_key))
        for row_pos, month_end_date in enumerate(group[date_column]):
            for horizon in horizons:
                value = np.nan
                end_pos = row_pos + horizon
                if end_pos <= len(group):
                    window_ordinals = month_ordinals[row_pos:end_pos]
                    expected_ordinals = np.arange(month_ordinals[row_pos], month_ordinals[row_pos] + horizon)
                    window_returns = returns[row_pos:end_pos]
                    if np.array_equal(window_ordinals, expected_ordinals) and np.isfinite(window_returns).all():
                        decimal_returns = window_returns / 100.0 if input_return_unit == "percent" else window_returns
                        compounded_decimal = float(np.prod(1.0 + decimal_returns) - 1.0)
                        value = compounded_decimal * 100.0 if output_return_unit == "percent" else compounded_decimal
                rows.append(
                    {
                        **group_values,
                        date_column: month_end_date,
                        "return_horizon_months": int(horizon),
                        output_column: value,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def build_equal_weight_portfolios(
    frame: pd.DataFrame,
    sort_column: str,
    value_column: str,
    date_column: str | None = None,
    n_portfolios: int = 10,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["month_end_date", "portfolio_code", "ew_return"])

    out = frame.copy()
    date_col = date_column or _infer_date_column(out)
    out["portfolio_num"] = out.groupby(date_col)[sort_column].transform(lambda s: _bucketize(s, n_portfolios))
    out = out.dropna(subset=["portfolio_num", value_column])
    grouped = out.groupby([date_col, "portfolio_num"], as_index=False)[value_column].mean()
    grouped["portfolio_code"] = grouped["portfolio_num"].astype(int).map(lambda value: f"P{value:02d}")
    grouped = grouped.rename(columns={date_col: "month_end_date", value_column: "ew_return"})
    return grouped[["month_end_date", "portfolio_code", "ew_return"]]


def build_binary_flag_portfolios(
    frame: pd.DataFrame,
    sort_column: str,
    value_column: str,
    date_column: str | None = None,
    *,
    safe_code: str = "SAFE",
    prone_code: str = "PRONE",
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["month_end_date", "portfolio_code", "ew_return"])

    out = frame.copy()
    date_col = date_column or _infer_date_column(out)
    out["_binary_sort_value"] = pd.to_numeric(out[sort_column], errors="coerce")
    out = out.loc[out["_binary_sort_value"].isin([0.0, 1.0])].copy()
    out["portfolio_code"] = np.where(out["_binary_sort_value"].eq(1.0), prone_code, safe_code)
    out = out.dropna(subset=["portfolio_code", value_column])
    grouped = out.groupby([date_col, "portfolio_code"], as_index=False)[value_column].mean()
    grouped = grouped.rename(columns={date_col: "month_end_date", value_column: "ew_return"})
    return grouped[["month_end_date", "portfolio_code", "ew_return"]]


def build_long_short_spread(
    portfolios: pd.DataFrame,
    low_code: str = "P01",
    high_code: str = "P10",
) -> pd.DataFrame:
    if portfolios is None or portfolios.empty:
        return pd.DataFrame(columns=["month_end_date", "spread_code", "spread_return"])
    low = portfolios.loc[portfolios["portfolio_code"] == low_code, ["month_end_date", "ew_return"]].rename(
        columns={"ew_return": "low_return"}
    )
    high = portfolios.loc[portfolios["portfolio_code"] == high_code, ["month_end_date", "ew_return"]].rename(
        columns={"ew_return": "high_return"}
    )
    spread = high.merge(low, on="month_end_date", how="inner")
    spread["spread_code"] = f"{high_code}_{low_code}"
    spread["spread_return"] = spread["high_return"] - spread["low_return"]
    return spread[["month_end_date", "spread_code", "spread_return"]]


def build_spread_between_portfolios(
    portfolios: pd.DataFrame,
    *,
    long_code: str,
    short_code: str,
    spread_code: str = "LONG_SHORT",
) -> pd.DataFrame:
    if portfolios is None or portfolios.empty:
        return pd.DataFrame(columns=["month_end_date", "spread_code", "spread_return", "long_code", "short_code"])
    long_leg = portfolios.loc[portfolios["portfolio_code"] == long_code, ["month_end_date", "ew_return"]].rename(
        columns={"ew_return": "long_return"}
    )
    short_leg = portfolios.loc[portfolios["portfolio_code"] == short_code, ["month_end_date", "ew_return"]].rename(
        columns={"ew_return": "short_return"}
    )
    spread = long_leg.merge(short_leg, on="month_end_date", how="inner")
    spread["spread_code"] = spread_code
    spread["spread_return"] = spread["long_return"] - spread["short_return"]
    spread["long_code"] = long_code
    spread["short_code"] = short_code
    return spread[["month_end_date", "spread_code", "spread_return", "long_code", "short_code"]]
