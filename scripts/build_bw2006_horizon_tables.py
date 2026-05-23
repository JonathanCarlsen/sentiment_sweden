#!/usr/bin/env python3
"""Build BW2006-style horizon tables for the Swedish sentiment thesis."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "final_sentiment_sweden"
SOURCE = OUTPUT_DIR / "final_predictive_results_directional_spreads.csv"
BOOTSTRAP_SOURCE = OUTPUT_DIR / "final_predictive_results_directional_spreads_bootstrap.csv"
CSV_OUT = OUTPUT_DIR / "final_bw2006_horizon_coefficients.csv"
TEX_OUT = OUTPUT_DIR / "final_bw2006_horizon_tables.tex"
INVENTORY = OUTPUT_DIR / "final_output_inventory.csv"


GROUPS = [
    (
        "Panel A. Size, Age, and Risk",
        [
            ("ME", "\\var{ME}", "Small minus large"),
            ("age", "Age", "Young minus old"),
            ("risk", "\\var{\\sigma}", "High minus low"),
            ("IVOL_FF3", "\\var{IVOL^{FF3}}", "High minus low"),
        ],
    ),
    (
        "Panel B. Profitability and Dividend Policy",
        [
            ("E_plus_BE", "\\var{E+BE}", "Low minus high"),
            ("UNPROFITABLE", "Unprofitable", "Unprofitable minus profitable"),
            ("NON_D_PAYER", "Dividend", "Non-payers minus payers"),
        ],
    ),
    (
        "Panel C. Tangibility",
        [
            ("PPE_A", "\\var{PPE/A}", "Low minus high"),
        ],
    ),
    (
        "Panel D. Trading Frictions",
        [
            ("ILLIQ", "Illiquidity", "High minus low"),
            ("XTURN", "Turnover", "Low minus high"),
        ],
    ),
]


def normal_two_sided_pvalue(t_stat: float) -> float:
    return math.erfc(abs(t_stat) / math.sqrt(2.0))


def stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def fmt_num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    rounded = round(float(value), digits)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{digits}f}"


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return ""
    value = max(0.0, min(1.0, float(value)))
    if value < 0.001:
        return "[0.000]"
    return f"[{value:.3f}]"


def interpretation(coef: float, hac_p: float, boot_p: float) -> str:
    if coef < 0 and hac_p < 0.05 and boot_p < 0.05:
        return "Supports BW2006 direction"
    if coef < 0 and (hac_p < 0.10 or boot_p < 0.10):
        return "Expected sign, weaker support"
    if coef < 0:
        return "Expected sign, insignificant"
    return "Conflicts with expected sign"


def latex_escape(text: str) -> str:
    if "\\" in text:
        return text
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def build_rows() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    rows = df[
        (df["sentiment_index_label"].isin(["SENT_DIV_PREMIUM", "SENT_ORTH_DIV_PREMIUM"]))
        & (df["sentiment_term_flag"] == True)
    ].copy()
    rows["hac_p_value"] = rows["t_stat"].map(normal_two_sided_pvalue)
    boot = pd.read_csv(BOOTSTRAP_SOURCE)
    boot = boot[
        (boot["sentiment_index_label"] == "SENT_ORTH_DIV_PREMIUM")
        & (boot["sentiment_term_flag"] == True)
    ].copy()

    ordered = []
    for horizon in [1, 3, 6, 12]:
        hdf = rows[rows["return_horizon_months"] == horizon].set_index(
            ["sort_variable", "sentiment_index_label"]
        )
        bdf = boot[boot["return_horizon_months"] == horizon].set_index("sort_variable")
        for group_name, variables in GROUPS:
            for sort_variable, variable_label, spread in variables:
                raw = hdf.loc[(sort_variable, "SENT_DIV_PREMIUM")]
                orth = hdf.loc[(sort_variable, "SENT_ORTH_DIV_PREMIUM")]
                orth_boot = bdf.loc[sort_variable]
                ordered.append(
                    {
                        "return_horizon_months": horizon,
                        "group": group_name,
                        "sort_variable": sort_variable,
                        "variable_label": variable_label,
                        "spread": spread,
                        "raw_coef": float(raw["coef"]),
                        "raw_hac_t": float(raw["t_stat"]),
                        "raw_hac_p_value": float(raw["hac_p_value"]),
                        "orth_coef": float(orth["coef"]),
                        "orth_hac_t": float(orth["t_stat"]),
                        "orth_hac_p_value": float(orth["hac_p_value"]),
                        "orth_bootstrap_p_value": float(orth_boot["bootstrap_p_value"]),
                    }
                )
    return pd.DataFrame(ordered)


def table_for_horizon(rows: pd.DataFrame, horizon: int) -> str:
    label = f"tab:bw2006_horizon_{horizon}m"
    caption = f"Baker-Wurgler-style directional-spread regressions, {horizon}-month horizon"
    lines = [
        "\\begin{table}[H]",
        "    \\centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        "    \\scriptsize",
        "    \\begin{tabular}{llrrrrr}",
        "        \\toprule",
        "        & & \\multicolumn{2}{c}{\\var{SENT\\_DIV\\_PREMIUM_{t-1}}} & \\multicolumn{3}{c}{\\var{SENT\\_ORTH\\_DIV\\_PREMIUM_{t-1}}} \\\\",
        "        \\cmidrule(lr){3-4} \\cmidrule(lr){5-7}",
        "        Variable & Spread & \\var{d} & HAC \\var{p(d)} & \\var{d} & HAC \\var{p(d)} & Boot. \\var{p(d)} \\\\",
        "        \\midrule",
    ]
    subset = rows[rows["return_horizon_months"] == horizon]
    for group_index, (group_name, variables) in enumerate(GROUPS):
        lines.append(
            f"        \\multicolumn{{7}}{{c}}{{\\textit{{{latex_escape(group_name)}}}}} \\\\"
        )
        lines.append("        \\midrule")
        for sort_variable, _variable_label, _spread in variables:
            row = subset[subset["sort_variable"] == sort_variable].iloc[0]
            lines.append(
                "        "
                + " & ".join(
                    [
                        latex_escape(str(row["variable_label"])),
                        latex_escape(str(row["spread"])),
                        fmt_num(float(row["raw_coef"])),
                        fmt_p(float(row["raw_hac_p_value"])),
                        fmt_num(float(row["orth_coef"])),
                        fmt_p(float(row["orth_hac_p_value"])),
                        fmt_p(float(row["orth_bootstrap_p_value"])),
                    ]
                )
                + " \\\\"
            )
        if group_index != len(GROUPS) - 1:
            lines.append("        \\midrule")
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    \\tablesource{\\repo{outputs/final\\_sentiment\\_sweden/final\\_bw2006\\_horizon\\_coefficients.csv}.}",
            "    \\tablenote{Coefficients are percentage-point effects on future sentiment-prone-minus-insensitive returns. HAC p-values use Newey-West standard errors. Bootstrap p-values are shown for the orthogonalized index from the moving-block bootstrap output. Negative coefficients are expected for all directional spreads.}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(rows: pd.DataFrame) -> None:
    rows.to_csv(CSV_OUT, index=False)
    TEX_OUT.write_text(
        "\n".join(table_for_horizon(rows, h) for h in [1, 3, 6, 12]),
        encoding="utf-8",
    )


def update_inventory() -> None:
    files = sorted(p for p in OUTPUT_DIR.iterdir() if p.is_file())
    with INVENTORY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "bytes"])
        for path in files:
            writer.writerow([path.name, path.stat().st_size])


def main() -> None:
    rows = build_rows()
    if len(rows) != 40:
        raise RuntimeError(f"Expected 40 table rows, got {len(rows)}")
    write_outputs(rows)
    update_inventory()
    print(f"Wrote {CSV_OUT.relative_to(ROOT)}")
    print(f"Wrote {TEX_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
