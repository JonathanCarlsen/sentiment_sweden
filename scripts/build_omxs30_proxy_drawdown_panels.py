#!/usr/bin/env python3
"""Plot OMXS30 drawdowns with selected standardized sentiment proxies."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "final_sentiment_sweden"
OMXS_SOURCE = OUTPUT_DIR / "final_omxs30_sent_orth_drawdown_panel.csv"
DRAWDOWN_SOURCE = OUTPUT_DIR / "final_omxs30_drawdown_periods.csv"
SENTIMENT_SOURCE = OUTPUT_DIR / "final_sentiment_monthly.csv"
CSV_OUT = OUTPUT_DIR / "final_omxs30_selected_standardized_proxy_drawdown_panel.csv"
PNG_OUT = OUTPUT_DIR / "final_omxs30_selected_standardized_proxy_drawdown_panels.png"
INVENTORY = OUTPUT_DIR / "final_output_inventory.csv"


PROXIES = [
    ("TURN", "TURN", "#2ca02c"),
    ("DIV_PREMIUM", "DIVP", "#8c564b"),
    ("ED_RATIO", "ED_RATIO", "#17becf"),
    ("NIPO", "NIPO", "#9467bd"),
]


def build_panel() -> pd.DataFrame:
    omxs = pd.read_csv(OMXS_SOURCE, parse_dates=["month_end_date"])
    sentiment = pd.read_csv(SENTIMENT_SOURCE, parse_dates=["month_end_date"])
    standardized = sentiment[["month_end_date"]].copy()
    for column, _label, _color in PROXIES:
        series = pd.to_numeric(sentiment[column], errors="coerce")
        standardized[column] = (series - series.mean(skipna=True)) / series.std(
            skipna=True, ddof=1
        )
    panel = omxs[["month_end_date", "adj_close"]].merge(
        standardized, on="month_end_date", how="left"
    )
    return panel


def add_drawdown_spans(ax: plt.Axes, drawdowns: pd.DataFrame, *, label: bool) -> None:
    ordered = drawdowns.sort_values("peak_month").reset_index(drop=True)
    for idx, row in ordered.iterrows():
        ax.axvspan(
            row["peak_month"],
            row["trough_month"],
            color="#bdbdbd",
            alpha=0.26,
            label="OMXS30 peak-to-trough drawdown" if label and idx == 0 else None,
        )


def plot_panel(panel: pd.DataFrame) -> None:
    drawdowns = pd.read_csv(
        DRAWDOWN_SOURCE,
        parse_dates=["peak_month", "trough_month", "recovery_month"],
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(11.5, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1, 1, 1, 1], "hspace": 0.08},
    )

    top = axes[0]
    add_drawdown_spans(top, drawdowns, label=True)
    top.plot(
        panel["month_end_date"],
        panel["adj_close"],
        label="OMXS30",
        color="#1f77b4",
        linewidth=1.8,
    )
    top.set_ylabel("OMXS30")
    top.legend(loc="upper left", frameon=False, fontsize=9)

    for ax, (column, label, color) in zip(axes[1:], PROXIES):
        add_drawdown_spans(ax, drawdowns, label=False)
        ax.plot(
            panel["month_end_date"],
            panel[column],
            label=label,
            color=color,
            linewidth=1.55,
        )
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.8)
        ax.set_ylabel(label)
        ax.legend(loc="upper left", frameon=False, fontsize=9)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].tick_params(axis="x", rotation=45)
    axes[-1].set_xlabel("Month")

    for ax in axes:
        ax.margins(x=0.01)

    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def update_inventory() -> None:
    files = sorted(p for p in OUTPUT_DIR.iterdir() if p.is_file())
    with INVENTORY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "bytes"])
        for path in files:
            writer.writerow([path.name, path.stat().st_size])


def main() -> None:
    panel = build_panel()
    panel.to_csv(CSV_OUT, index=False)
    plot_panel(panel)
    update_inventory()
    print(f"Wrote {CSV_OUT.relative_to(ROOT)}")
    print(f"Wrote {PNG_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
