from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nordic_sentiment" / "src"))

from nordic_sentiment.sentiment.sweden import build_sweden_macro_controls_monthly

OUT = ROOT / "outputs" / "final_sentiment_sweden"
MONTHLY = OUT / "final_sentiment_monthly.csv"
MACRO_DIR = ROOT / "Macrocontrols"

BASE_PROXIES = ["ESI", "CCI", "TURN", "NIPO", "RIPO", "ED_RATIO", "DIV_PREMIUM"]
MACRO_RESIDUAL_PROXIES = [
    "rESI_L1",
    "rCCI_L1",
    "rTURN_L1",
    "rNIPO",
    "rRIPO",
    "rED_RATIO_L1",
    "rDIV_PREMIUM",
]


def zscore_with_sample(frame: pd.DataFrame, neutral_impute_columns: set[str] | None = None):
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    neutral_columns = {column for column in (neutral_impute_columns or set()) if column in numeric.columns}
    if neutral_columns:
        mean = numeric.mean()
        std = numeric.std(ddof=0)
    else:
        sample = numeric.dropna()
        if sample.empty:
            raise RuntimeError("No complete observations are available for PCA estimation.")
        mean = sample.mean()
        std = sample.std(ddof=0)
    bad = std.isna() | std.eq(0)
    if bad.any():
        names = ", ".join(std.index[bad].tolist())
        raise RuntimeError(f"Cannot estimate PCA because these variables have zero or missing variance: {names}")
    z = (numeric - mean) / std
    if neutral_columns:
        z.loc[:, sorted(neutral_columns)] = z.loc[:, sorted(neutral_columns)].fillna(0.0)
    sample = z.dropna()
    if sample.empty:
        raise RuntimeError("No complete observations are available for PCA estimation.")
    return z, sample, mean, std


def pca_tables(frame: pd.DataFrame, neutral_impute_columns: set[str] | None = None, min_cumulative: float = 0.85):
    z, sample, mean, std = zscore_with_sample(frame, neutral_impute_columns=neutral_impute_columns)
    pca = PCA(n_components=min(sample.shape[0], sample.shape[1]), random_state=0)
    pca.fit(sample)
    eigenvalues = pd.Series(pca.explained_variance_, name="eigenvalue")
    shares = eigenvalues / eigenvalues.sum()
    cumulative = shares.cumsum()
    retained = int(np.searchsorted(cumulative.to_numpy(), min_cumulative, side="left") + 1)

    loadings = pd.DataFrame(
        pca.components_[:retained].T,
        index=frame.columns,
        columns=[f"Comp{i}" for i in range(1, retained + 1)],
    )
    for column in loadings.columns:
        if loadings[column].sum() < 0:
            loadings[column] = -loadings[column]
    weighted = loadings.mul(shares.iloc[:retained].to_numpy(), axis=1)
    coefficients = weighted.sum(axis=1).rename("coefficient")
    index = z.reindex(columns=coefficients.index).mul(coefficients, axis=1).sum(axis=1, min_count=len(coefficients))

    eigen = pd.DataFrame(
        {
            "component": [f"Comp{i}" for i in range(1, len(eigenvalues) + 1)],
            "eigenvalue": eigenvalues.to_numpy(),
            "proportion": shares.to_numpy(),
            "cumulative": cumulative.to_numpy(),
            "retained_flag": [i < retained for i in range(len(eigenvalues))],
        }
    )
    stats = pd.DataFrame(
        {
            "variable": frame.columns,
            "sample_mean": mean.reindex(frame.columns).to_numpy(),
            "sample_std": std.reindex(frame.columns).to_numpy(),
            "neutral_impute_after_standardization": [
                column in (neutral_impute_columns or set()) for column in frame.columns
            ],
        }
    )
    return {
        "z": z,
        "index": index,
        "eigenvalues": eigen,
        "weighted_coefficients": weighted.assign(sum=coefficients).reset_index(names="variable"),
        "sample_stats": stats,
    }


def format_num(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    return f"{float(value):.{digits}f}"


def display_proxy(name: str) -> str:
    replacements = {
        "DIV_PREMIUM": "DIVP",
        "DIV_PREMIUM_L1": "DIVP_L1",
        "rDIV_PREMIUM": "rDIVP",
    }
    return replacements.get(name, name).replace("_", "\\_")


def latex_table(caption: str, label: str, columns: list[str], rows: list[list[object]], note: str, source: str) -> str:
    aligns = "l" + "r" * (len(columns) - 1)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        f"\\begin{{tabular}}{{{aligns}}}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(item) for item in row) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\tablesource{{\\repo{{{source}}}.}}",
            f"\\tablenote{{{note}}}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def retained_eigen_rows(eigen: pd.DataFrame):
    retained = eigen.loc[eigen["retained_flag"]].copy()
    return [
        [row.component, format_num(row.eigenvalue), format_num(row.proportion), format_num(row.cumulative)]
        for row in retained.itertuples(index=False)
    ]


def coefficient_rows(coefficients: pd.DataFrame):
    rows = []
    for row in coefficients.itertuples(index=False):
        rows.append([getattr(row, "variable"), format_num(getattr(row, "sum"))])
    return rows


def correlation_rows(index: pd.Series, z: pd.DataFrame):
    rows = []
    for column in z.columns:
        rows.append([column, format_num(z[column].corr(index))])
    return rows


def min_max_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.apply(pd.to_numeric, errors="coerce")
    minimum = out.min()
    maximum = out.max()
    span = maximum - minimum
    return out.sub(minimum, axis=1).div(span.replace(0, np.nan), axis=1)


def ols_residual(series: pd.Series, regressors: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    x = regressors.apply(pd.to_numeric, errors="coerce").copy()
    x["intercept"] = 1.0
    valid = y.notna() & x.notna().all(axis=1)
    residual = pd.Series(np.nan, index=series.index, dtype="float64")
    if valid.sum() <= x.shape[1]:
        return residual
    beta, *_ = np.linalg.lstsq(x.loc[valid].to_numpy(dtype=float), y.loc[valid].to_numpy(dtype=float), rcond=None)
    fitted = x.loc[valid].to_numpy(dtype=float) @ beta
    residual.loc[valid] = y.loc[valid].to_numpy(dtype=float) - fitted
    return residual


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1")


def read_macro_controls() -> pd.DataFrame:
    return build_sweden_macro_controls_monthly(
        cpi=read_csv_with_fallback(MACRO_DIR / "Consumer Price Index (CPI), total 2020=100. Month 1980M01 - 2026M03.csv"),
        ppi=read_csv_with_fallback(MACRO_DIR / "Producer Price Index by market and products SPIN 2015, 2020=100. Month 1990M01 - 2026M03.csv"),
        ip=read_csv_with_fallback(MACRO_DIR / "Industrial production index. Chain index, 2021=100, by industral classification NACE Rev.2. Monthly 2000M01 - 2026M02.csv"),
        em=read_csv_with_fallback(MACRO_DIR / "Population aged 15-74 (LFS) by labour status, type of data, sex and age. Month 2001M01 - 2026M03.csv"),
    )


def residualized_selected_proxies(monthly: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    macro = read_macro_controls()
    macro["month_end_date"] = pd.to_datetime(macro["month_end_date"], errors="coerce")
    macro_columns = ["CPI", "PPI", "IP", "EM"]
    normalized = min_max_normalize(macro[macro_columns])
    normalized["month_end_date"] = macro["month_end_date"].to_numpy()
    merged = monthly[["country_code", "month_end_date", *selected]].merge(normalized, on="month_end_date", how="left")
    residuals = pd.DataFrame(index=merged.index)
    for variable in selected:
        residuals[f"r{variable}"] = ols_residual(merged[variable], merged[macro_columns])
    return residuals


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_csv(MONTHLY, parse_dates=["month_end_date"])

    stage1_variables = [column for proxy in BASE_PROXIES for column in [proxy, f"{proxy}_L1"]]
    stage1 = pca_tables(
        monthly[stage1_variables],
        neutral_impute_columns={column for column in stage1_variables if "RIPO" in column},
    )
    stage1["eigenvalues"].to_csv(OUT / "final_sentiment_pca_stage1_eigenvalues.csv", index=False)
    stage1["weighted_coefficients"].to_csv(
        OUT / "final_sentiment_pca_stage1_weighted_coefficients.csv",
        index=False,
    )

    lead_lag_rows = []
    selected = []
    for proxy in BASE_PROXIES:
        current = proxy
        lagged = f"{proxy}_L1"
        corr_current = monthly[current].corr(stage1["index"])
        corr_lagged = monthly[lagged].corr(stage1["index"])
        chosen = current if corr_current >= corr_lagged else lagged
        selected.append(chosen)
        lead_lag_rows.append(
            {
                "base_proxy": proxy,
                "corr_current": corr_current,
                "corr_lagged": corr_lagged,
                "selected_proxy": chosen,
            }
        )
    lead_lag = pd.DataFrame(lead_lag_rows)
    lead_lag.to_csv(OUT / "final_sentiment_pca_lead_lag_selection.csv", index=False)

    stage2 = pca_tables(
        monthly[selected],
        neutral_impute_columns={column for column in selected if "RIPO" in column},
    )
    stage2["eigenvalues"].to_csv(OUT / "final_sentiment_pca_stage2_eigenvalues.csv", index=False)
    stage2["weighted_coefficients"].to_csv(
        OUT / "final_sentiment_pca_stage2_weighted_coefficients.csv",
        index=False,
    )
    stage2_corr = pd.DataFrame(
        [{"variable": column, "correlation_with_SENT_DIV_PREMIUM": stage2["z"][column].corr(stage2["index"])} for column in selected]
    )
    stage2_corr.to_csv(OUT / "final_sentiment_pca_stage2_correlations.csv", index=False)

    stage3_eigen = pd.read_csv(OUT / "final_sentiment_diagnostic_stage3_eigenvalues.csv")
    stage3_coeff = pd.read_csv(OUT / "final_sentiment_diagnostic_stage3_weighted_coefficients.csv")
    stage3_residuals = residualized_selected_proxies(monthly, selected)
    stage3 = pca_tables(stage3_residuals, neutral_impute_columns={"rRIPO"})
    # Use the canonical saved diagnostics for values that enter the thesis.
    stage3_eigen.to_csv(OUT / "final_sentiment_pca_stage3_eigenvalues.csv", index=False)
    stage3_coeff.to_csv(OUT / "final_sentiment_pca_stage3_weighted_coefficients.csv", index=False)
    final_index = monthly["SENT_ORTH_DIV_PREMIUM"]
    stage3_corr = pd.DataFrame(
        [
            {"variable": column, "correlation_with_SENT_ORTH_DIV_PREMIUM": stage3["z"][column].corr(final_index)}
            for column in stage3["z"].columns
        ]
    )
    stage3_corr.to_csv(OUT / "final_sentiment_pca_stage3_correlations.csv", index=False)

    fragment = []
    fragment.append(
        latex_table(
            "Preliminary sentiment PCA retained components",
            "tab:sentiment_pca_stage1_eigen",
            ["Component", "Eigenvalue", "Variance", "Cumulative"],
            retained_eigen_rows(stage1["eigenvalues"]),
            "The preliminary PCA uses current and lagged values of the seven sentiment proxies. Components are retained until cumulative explained variance exceeds 85\\%.",
            "outputs/final\\_sentiment\\_sweden/final\\_sentiment\\_pca\\_stage1\\_eigenvalues.csv",
        )
    )
    fragment.append(
        latex_table(
            "Lead-lag selection for sentiment proxies",
            "tab:sentiment_pca_lead_lag",
            ["Proxy", "Current corr.", "Lag corr.", "Selected"],
            [
                [
                    display_proxy(row.base_proxy),
                    format_num(row.corr_current),
                    format_num(row.corr_lagged),
                    display_proxy(row.selected_proxy),
                ]
                for row in lead_lag.itertuples(index=False)
            ],
            "The selected proxy is the current or lagged version with the higher correlation with the preliminary PCA index.",
            "outputs/final\\_sentiment\\_sweden/final\\_sentiment\\_pca\\_lead\\_lag\\_selection.csv",
        )
    )
    fragment.append(
        latex_table(
            "Raw selected-proxy PCA retained components",
            "tab:sentiment_pca_stage2_eigen",
            ["Component", "Eigenvalue", "Variance", "Cumulative"],
            retained_eigen_rows(stage2["eigenvalues"]),
            "The raw index is estimated from the selected current or lagged proxies before macro residualization.",
            "outputs/final\\_sentiment\\_sweden/final\\_sentiment\\_pca\\_stage2\\_eigenvalues.csv",
        )
    )
    fragment.append(
        latex_table(
            "Raw selected-proxy PCA score coefficients",
            "tab:sentiment_pca_stage2_coeff",
            ["Proxy", "Score coef."],
            [[display_proxy(row[0]), row[1]] for row in coefficient_rows(stage2["weighted_coefficients"])],
            "Score coefficients are the summed weighted component coefficients used to construct \\var{SENT_t}.",
            "outputs/final\\_sentiment\\_sweden/final\\_sentiment\\_pca\\_stage2\\_weighted\\_coefficients.csv",
        )
    )
    fragment.append(
        latex_table(
            "Raw index correlations with selected proxies",
            "tab:sentiment_pca_stage2_corr",
            ["Proxy", "Correlation"],
            [[display_proxy(row[0]), row[1]] for row in correlation_rows(stage2["index"], stage2["z"])],
            "The table reports correlations between standardized selected proxies and the raw PCA index.",
            "outputs/final\\_sentiment\\_sweden/final\\_sentiment\\_pca\\_stage2\\_correlations.csv",
        )
    )
    fragment.append(
        latex_table(
            "Macro-residualized PCA retained components",
            "tab:sentiment_pca_stage3_eigen",
            ["Component", "Eigenvalue", "Variance", "Cumulative"],
            retained_eigen_rows(stage3_eigen),
            "The final index is estimated from selected proxies after residualizing each proxy against Swedish macro controls.",
            "outputs/final\\_sentiment\\_sweden/final\\_sentiment\\_pca\\_stage3\\_eigenvalues.csv",
        )
    )
    fragment.append(
        latex_table(
            "Macro-residualized PCA score coefficients",
            "tab:sentiment_pca_stage3_coeff",
            ["Proxy", "Score coef."],
            [[display_proxy(row.variable), format_num(row.sum)] for row in stage3_coeff.itertuples(index=False)],
            "Score coefficients are the summed weighted component coefficients used to construct \\var{SENT^{\\perp}_t}.",
            "outputs/final\\_sentiment\\_sweden/final\\_sentiment\\_pca\\_stage3\\_weighted\\_coefficients.csv",
        )
    )
    fragment.append(
        latex_table(
            "Final index correlations with residualized proxies",
            "tab:sentiment_pca_stage3_corr",
            ["Proxy", "Correlation"],
            [[display_proxy(row.variable), format_num(row.correlation_with_SENT_ORTH_DIV_PREMIUM)] for row in stage3_corr.itertuples(index=False)],
            "The table reports correlations between standardized residualized proxies and the final macro-orthogonalized PCA index.",
            "outputs/final\\_sentiment\\_sweden/final\\_sentiment\\_pca\\_stage3\\_correlations.csv",
        )
    )

    (OUT / "final_sentiment_pca_construction_tables.tex").write_text("\n\n".join(fragment) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
