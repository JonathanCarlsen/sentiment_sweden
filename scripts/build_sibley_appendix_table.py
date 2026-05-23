#!/usr/bin/env python3
"""Build appendix table comparing SENT^perp and Sibley-expanded SENT^S tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "final_sentiment_sweden"
SOURCE = OUTPUT_DIR / "final_predictive_results_directional_spreads_bootstrap.csv"
CSV_OUT = OUTPUT_DIR / "final_sibley_characteristic_horizon_appendix.csv"
TEX_OUT = OUTPUT_DIR / "final_sibley_characteristic_horizon_appendix_table.tex"
INVENTORY = OUTPUT_DIR / "final_output_inventory.csv"


ORDER = [
    ("ME", "Small firms"),
    ("age", "Young firms"),
    ("risk", "Total risk"),
    ("IVOL_FF3", "\\var{IVOL^{FF3}}"),
    ("E_plus_BE", "Low profitability"),
    ("UNPROFITABLE", "Unprofitable"),
    ("NON_D_PAYER", "Non-dividend payer"),
    ("PPE_A", "Low tangibility"),
    ("GS", "Sales growth"),
    ("ILLIQ", "Illiquidity"),
    ("XTURN", "Low turnover"),
]

INDEXES = {
    "SENT_ORTH_DIV_PREMIUM": "orth",
    "SENT_ORTH_SIBLEY_DIV_PREMIUM": "sibley",
}


def fmt(value: float) -> str:
    if pd.isna(value):
        return ""
    rounded = round(float(value), 2)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.2f}"


def latex_escape(text: str) -> str:
    if "\\" in text:
        return text
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def bold_if(text: str, condition: bool) -> str:
    return f"\\textbf{{{text}}}" if condition else text


def build_rows() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    rows = df[
        df["sentiment_term_flag"].astype(bool)
        & df["sentiment_index_label"].isin(INDEXES)
    ].copy()

    records = []
    for horizon in [1, 3, 6, 12]:
        hdf = rows[rows["return_horizon_months"] == horizon].set_index(
            ["sort_variable", "sentiment_index_label"]
        )
        for sort_variable, label in ORDER:
            orth = hdf.loc[(sort_variable, "SENT_ORTH_DIV_PREMIUM")]
            sibley = hdf.loc[(sort_variable, "SENT_ORTH_SIBLEY_DIV_PREMIUM")]
            orth_sig = bool(orth["theory_consistent_hac_5pct"])
            sibley_sig = bool(sibley["theory_consistent_hac_5pct"])
            if orth_sig and sibley_sig:
                status = "Stays 5%"
            elif orth_sig and not sibley_sig:
                status = "Loses 5%"
            elif not orth_sig and sibley_sig:
                status = "Gains 5%"
            else:
                status = "Not 5% before"
            records.append(
                {
                    "return_horizon_months": horizon,
                    "sort_variable": sort_variable,
                    "characteristic": label,
                    "sent_orth_coef": float(orth["coef"]),
                    "sent_orth_hac_t": float(orth["t_stat"]),
                    "sent_orth_expected_signed_hac_t": float(
                        orth["expected_signed_t_stat"]
                    ),
                    "sent_orth_theory_consistent_hac_5pct": orth_sig,
                    "sent_s_coef": float(sibley["coef"]),
                    "sent_s_hac_t": float(sibley["t_stat"]),
                    "sent_s_expected_signed_hac_t": float(
                        sibley["expected_signed_t_stat"]
                    ),
                    "sent_s_theory_consistent_hac_5pct": sibley_sig,
                    "loses_hac_5pct_under_sent_s": orth_sig and not sibley_sig,
                    "sibley_status": status,
                }
            )
    return pd.DataFrame(records)


def make_latex(rows: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Sibley-expanded sentiment robustness by characteristic and horizon}",
        "\\label{tab:appendix_sibley_characteristic_horizon}",
        "\\scriptsize",
        "\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}Xrrrr>{\\raggedright\\arraybackslash}p{0.19\\textwidth}}",
        "\\toprule",
        "Characteristic & \\multicolumn{2}{c}{\\var{SENT^{\\perp}}} & \\multicolumn{2}{c}{\\var{SENT^{S}}} & Status \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        " & Coef. & HAC \\var{t} & Coef. & HAC \\var{t} & \\\\",
        "\\midrule",
    ]

    for horizon in [1, 3, 6, 12]:
        subset = rows[rows["return_horizon_months"] == horizon]
        lines.append(
            f"\\multicolumn{{6}}{{l}}{{\\textit{{{horizon}-month horizon}}}}\\\\"
        )
        for row in subset.itertuples(index=False):
            lost = bool(row.loses_hac_5pct_under_sent_s)
            status = str(row.sibley_status).replace("%", "\\%")
            cells = [
                latex_escape(row.characteristic),
                fmt(row.sent_orth_coef),
                fmt(row.sent_orth_hac_t),
                fmt(row.sent_s_coef),
                fmt(row.sent_s_hac_t),
                status,
            ]
            cells = [bold_if(cell, lost) for cell in cells]
            lines.append(" & ".join(cells) + " \\\\")
        if horizon != 12:
            lines.append("\\midrule")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabularx}",
            "\\tablenote{Coefficients are percentage-point effects on future directional-spread returns. HAC t-statistics use the Newey-West standard errors from the predictive regressions. Status compares theory-consistent 5 percent HAC significance under \\var{SENT^{\\perp}} and \\var{SENT^{S}}. Bold rows identify cases where significance is lost under the Sibley-expanded index.}",
            "\\end{table}",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def update_inventory() -> None:
    files = sorted(p for p in OUTPUT_DIR.iterdir() if p.is_file())
    with INVENTORY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "bytes"])
        for path in files:
            writer.writerow([path.name, path.stat().st_size])


def main() -> None:
    rows = build_rows()
    rows.to_csv(CSV_OUT, index=False)
    TEX_OUT.write_text(make_latex(rows), encoding="utf-8")
    update_inventory()
    print(f"Wrote {CSV_OUT.relative_to(ROOT)}")
    print(f"Wrote {TEX_OUT.relative_to(ROOT)}")
    print(f"Rows: {len(rows)}")
    print(f"Loss flags: {int(rows['loses_hac_5pct_under_sent_s'].sum())}")


if __name__ == "__main__":
    main()
