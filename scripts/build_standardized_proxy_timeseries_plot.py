#!/usr/bin/env python3
"""Plot standardized monthly sentiment proxy inputs."""

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
SOURCE = OUTPUT_DIR / "final_sentiment_monthly.csv"
CSV_OUT = OUTPUT_DIR / "final_standardized_sentiment_proxies_over_time.csv"
PNG_OUT = OUTPUT_DIR / "final_standardized_sentiment_proxies_over_time.png"
INVENTORY = OUTPUT_DIR / "final_output_inventory.csv"


PROXIES = [
    ("ESI", "ESI", "#1f77b4"),
    ("CCI", "CCI", "#d62728"),
    ("TURN", "TURN", "#2ca02c"),
    ("NIPO", "NIPO", "#9467bd"),
    ("RIPO", "RIPO", "#ff7f0e"),
    ("ED_RATIO", "ED ratio", "#17becf"),
    ("DIV_PREMIUM", "DIVP", "#8c564b"),
]


def build_standardized_panel() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, parse_dates=["month_end_date"])
    out = df[["month_end_date"]].copy()
    for column, _label, _color in PROXIES:
        series = pd.to_numeric(df[column], errors="coerce")
        out[column] = (series - series.mean(skipna=True)) / series.std(skipna=True, ddof=1)
    return out


def plot_panel(panel: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    for column, label, color in PROXIES:
        ax.plot(
            panel["month_end_date"],
            panel[column],
            label=label,
            color=color,
            linewidth=1.35,
            alpha=0.95,
        )

    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.85)
    ax.set_ylabel("Standard deviations from proxy mean")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False,
    )
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
    panel = build_standardized_panel()
    panel.to_csv(CSV_OUT, index=False)
    plot_panel(panel)
    update_inventory()
    print(f"Wrote {CSV_OUT.relative_to(ROOT)}")
    print(f"Wrote {PNG_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
