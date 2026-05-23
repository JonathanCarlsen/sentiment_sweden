#!/usr/bin/env python3
"""Build descriptive statistics for raw sentiment proxy inputs."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "final_sentiment_sweden"
SOURCE = OUTPUT_DIR / "final_sentiment_monthly.csv"
CSV_OUT = OUTPUT_DIR / "final_raw_sentiment_proxy_descriptive_statistics.csv"
TEX_OUT = OUTPUT_DIR / "final_raw_sentiment_proxy_descriptive_statistics_table.tex"
INVENTORY = OUTPUT_DIR / "final_output_inventory.csv"


VARIABLES = [
    ("ESI", "Economic sentiment index", "\\var{ESI_t}"),
    ("CCI", "Consumer confidence index", "\\var{CCI_t}"),
    ("TURN", "Turnover", "\\var{TURN_t}"),
    ("NIPO", "IPO number", "\\var{NIPO_t}"),
    ("RIPO", "IPO initial return", "\\var{RIPO_t}"),
    ("ED_RATIO", "Equity-to-debt ratio", "\\var{ED\\_RATIO_t}"),
    ("DIV_PREMIUM", "Dividend premium", "\\var{DIVP_t}"),
]


def fmt(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def build_table() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    rows = []
    for column, label, tex_label in VARIABLES:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        rows.append(
            {
                "variable": column,
                "label": label,
                "tex_label": tex_label,
                "mean": series.mean(),
                "std": series.std(ddof=1),
                "min": series.min(),
                "max": series.max(),
                "median": series.median(),
                "obs": int(series.count()),
            }
        )
    return pd.DataFrame(rows)


def make_latex(rows: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Descriptive statistics for raw sentiment proxy inputs}",
        "\\label{tab:appendix_raw_sentiment_proxy_descriptives}",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Variable & Mean & Std. dev. & Min. & Max. & Median & Obs. \\\\",
        "\\midrule",
    ]
    for row in rows.itertuples(index=False):
        label = f"{row.label} ({row.tex_label})"
        lines.append(
            " & ".join(
                [
                    label,
                    fmt(row.mean),
                    fmt(row.std),
                    fmt(row.min),
                    fmt(row.max),
                    fmt(row.median),
                    str(row.obs),
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\tablenote{Statistics are computed from the unstandardized monthly proxy series before PCA and before macro residualization. \\var{RIPO_t} is observed only in months with observed IPO initial-return information. \\var{NIPO_t} equals zero in months without IPO activity. \\var{TURN_t} and \\var{DIV\\_PREMIUM_t} are logarithmic proxy constructions.}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def update_inventory() -> None:
    files = sorted(p for p in OUTPUT_DIR.iterdir() if p.is_file())
    with INVENTORY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "bytes"])
        for path in files:
            writer.writerow([path.name, path.stat().st_size])


def main() -> None:
    rows = build_table()
    rows.to_csv(CSV_OUT, index=False)
    TEX_OUT.write_text(make_latex(rows), encoding="utf-8")
    update_inventory()
    print(f"Wrote {CSV_OUT.relative_to(ROOT)}")
    print(f"Wrote {TEX_OUT.relative_to(ROOT)}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
