from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "nordic_sentiment" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nordic_sentiment import load_project_configs  # noqa: E402
from nordic_sentiment.io.compustat_sweden import (  # noqa: E402
    load_sweden_macro_control_cpi,
    load_sweden_macro_control_em,
    load_sweden_macro_control_ip,
    load_sweden_macro_control_ppi,
)
from nordic_sentiment.regressions import run_ols_with_hac_and_bootstrap  # noqa: E402
from nordic_sentiment.sentiment.sweden import (  # noqa: E402
    STV_SWEDEN_SENTIMENT_PROXIES,
    build_sweden_macro_controls_monthly,
    build_sweden_sentiment_index,
    build_sweden_sentiment_index_with_paper_lead_lag,
    build_sweden_sentiment_proxy_mart,
    build_sweden_stv_sentiment_index,
    default_sweden_paper_spec_audit,
)


FINAL_OUTPUT_DIR = REPO_ROOT / "outputs" / "final_sentiment_sweden"
OUTPUT_DIR = REPO_ROOT / "outputs" / "alternative_sentiment_index_robustness"
BOOTSTRAP_REPETITIONS = 1_000
BOOTSTRAP_SEED = 20260510
CONTROL_COLUMNS = ["RMRF", "SMB", "HML", "UMD"]

INDEX_SPECS = [
    {
        "sentiment_index_label": "SENT_ORTH_DIV_PREMIUM",
        "sentiment_regressor": "SENT_ORTH_DIV_PREMIUM_lag",
        "sentiment_index_family": "dividend_premium_macro_adjusted_fixed_pca",
        "macro_orthogonalized_flag": True,
        "main_specification_flag": True,
    },
    {
        "sentiment_index_label": "SENT_ORTH",
        "sentiment_regressor": "SENT_ORTH_lag",
        "sentiment_index_family": "six_proxy_macro_adjusted_fixed_pca",
        "macro_orthogonalized_flag": True,
        "main_specification_flag": False,
    },
    {
        "sentiment_index_label": "SENT_ORTH_PAPER_LEAD_LAG",
        "sentiment_regressor": "SENT_ORTH_PAPER_LEAD_LAG_lag",
        "sentiment_index_family": "paper_lead_lag_macro_adjusted_fixed_pca",
        "macro_orthogonalized_flag": True,
        "main_specification_flag": False,
    },
    {
        "sentiment_index_label": "STV_SWE",
        "sentiment_regressor": "STV_SWE_lag",
        "sentiment_index_family": "rolling_36_month_macro_adjusted_pc1",
        "macro_orthogonalized_flag": True,
        "main_specification_flag": False,
    },
]

LABELS = {
    "SENT_ORTH_DIV_PREMIUM": r"\var{SENT^{\perp}}",
    "SENT_ORTH": r"\var{SENT^{\perp,noDIV}}",
    "SENT_ORTH_PAPER_LEAD_LAG": r"\var{SENT^{\perp,PLL}}",
    "STV_SWE": r"\var{STV^{SWE}}",
}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def save_table(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    return frame


def final_sentiment_proxy_mart() -> pd.DataFrame:
    final = pd.read_csv(require_file(FINAL_OUTPUT_DIR / "final_sentiment_monthly.csv"))
    final["month_end_date"] = pd.to_datetime(final["month_end_date"], errors="coerce")
    proxy_columns = [column for column in STV_SWEDEN_SENTIMENT_PROXIES if column in final.columns]
    rows = []
    for proxy in proxy_columns:
        part = final[["country_code", "month_end_date", proxy]].rename(columns={proxy: "raw_value"}).copy()
        part["proxy_code"] = proxy
        part["paper_proxy_name"] = proxy
        part["source_name"] = "final_sentiment_sweden_proxy_snapshot"
        part["build_status"] = np.where(proxy == "DIV_PREMIUM", "adapted", "exact")
        part["exact_replication_flag"] = proxy != "DIV_PREMIUM"
        rows.append(part)
    return build_sweden_sentiment_proxy_mart(pd.concat(rows, ignore_index=True))


def load_standard_macro_controls() -> pd.DataFrame:
    configs = load_project_configs()
    return build_sweden_macro_controls_monthly(
        cpi=load_sweden_macro_control_cpi(configs),
        ppi=load_sweden_macro_control_ppi(configs),
        ip=load_sweden_macro_control_ip(configs),
        em=load_sweden_macro_control_em(configs),
    )


def build_alternative_indices() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    proxy_mart = final_sentiment_proxy_mart()
    macro_controls = load_standard_macro_controls()
    paper_spec = default_sweden_paper_spec_audit()
    paper_spec.loc[paper_spec["required_for_final_index"].fillna(False), "build_status"] = "exact"

    no_div_monthly, _, no_div_diagnostics = build_sweden_sentiment_index(
        paper_spec,
        proxy_mart,
        macro_controls=macro_controls,
    )
    paper_monthly, _, paper_diagnostics = build_sweden_sentiment_index_with_paper_lead_lag(
        paper_spec,
        proxy_mart,
        macro_controls=macro_controls,
    )
    stv_monthly, _, stv_diagnostics = build_sweden_stv_sentiment_index(
        paper_spec,
        proxy_mart,
        macro_controls=macro_controls,
        window_months=36,
    )
    baseline = pd.read_csv(require_file(FINAL_OUTPUT_DIR / "final_sentiment_monthly.csv"))
    baseline["month_end_date"] = pd.to_datetime(baseline["month_end_date"], errors="coerce")

    monthly = baseline[["country_code", "month_end_date", "SENT_ORTH_DIV_PREMIUM"]].copy()
    monthly = monthly.merge(
        no_div_monthly[["country_code", "month_end_date", "SENT_ORTH"]],
        on=["country_code", "month_end_date"],
        how="left",
    )
    monthly = monthly.merge(
        paper_monthly[["country_code", "month_end_date", "SENT_ORTH_PAPER_LEAD_LAG"]],
        on=["country_code", "month_end_date"],
        how="left",
    )
    monthly = monthly.merge(
        stv_monthly[["country_code", "month_end_date", "STV_SWE"]],
        on=["country_code", "month_end_date"],
        how="left",
    )
    for column in ["SENT_ORTH_DIV_PREMIUM", "SENT_ORTH", "SENT_ORTH_PAPER_LEAD_LAG", "STV_SWE"]:
        monthly[f"{column}_lag"] = monthly.groupby("country_code")[column].shift(1)

    diagnostics = {
        "no_div_stage3_eigenvalues": no_div_diagnostics.get("stage3_eigenvalues", pd.DataFrame()),
        "paper_lead_lag_selected_variables": paper_diagnostics.get("paper_lead_lag_selected_variables", pd.DataFrame()),
        "paper_lead_lag_stage3_eigenvalues": paper_diagnostics.get("stage3_eigenvalues", pd.DataFrame()),
        "stv_swe_rolling_eigenvalues": stv_diagnostics.get("stv_swe_rolling_eigenvalues", pd.DataFrame()),
        "stv_swe_rolling_loadings": stv_diagnostics.get("stv_swe_rolling_loadings", pd.DataFrame()),
    }
    return monthly, diagnostics


def add_interpretation_columns(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    out = results.copy()
    sentiment_term = out["term"].eq(out["sentiment_regressor"])
    out["sentiment_term_flag"] = sentiment_term
    out["expected_coef_sign"] = np.nan
    out.loc[sentiment_term, "expected_coef_sign"] = -1.0
    out["expected_signed_t_stat"] = out["expected_coef_sign"] * pd.to_numeric(out["t_stat"], errors="coerce")
    out["expected_signed_bootstrap_t_stat"] = out["expected_coef_sign"] * pd.to_numeric(
        out.get("bootstrap_t_stat"), errors="coerce"
    )
    out["theory_consistent_sign"] = out["expected_signed_t_stat"].gt(0)
    out["theory_consistent_hac_5pct"] = out["expected_signed_t_stat"].ge(1.96)
    out["theory_consistent_bootstrap_5pct"] = out["expected_signed_bootstrap_t_stat"].ge(1.96)
    return out


def run_regressions(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "return_horizon_months",
        "return_leg",
        "sort_variable",
        "sentiment_prone_direction",
        "sentiment_prone_interpretation",
    ]
    for spec in INDEX_SPECS:
        sentiment_regressor = str(spec["sentiment_regressor"])
        if sentiment_regressor not in panel.columns:
            continue
        x_cols = [sentiment_regressor, *[column for column in CONTROL_COLUMNS if column in panel.columns]]
        for group_key, group in panel.groupby(group_columns, dropna=False, sort=True):
            metadata = dict(zip(group_columns, group_key, strict=True))
            horizon = int(metadata["return_horizon_months"])
            result = run_ols_with_hac_and_bootstrap(
                group,
                "portfolio_return_window",
                x_cols,
                min_hac_lags=max(horizon - 1, 0),
                return_horizon_months=horizon,
                bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
                bootstrap_seed=BOOTSTRAP_SEED,
            )
            for column, value in reversed(list(metadata.items())):
                result.insert(0, column, value)
            result.insert(0, "result_set", "alternative_sentiment_index_directional_spread")
            result.insert(1, "sentiment_index_family", spec["sentiment_index_family"])
            result.insert(2, "sentiment_index_label", spec["sentiment_index_label"])
            result.insert(3, "sentiment_regressor", sentiment_regressor)
            result.insert(4, "macro_orthogonalized_flag", bool(spec["macro_orthogonalized_flag"]))
            result.insert(5, "main_specification_flag", bool(spec["main_specification_flag"]))
            result.insert(6, "dependent_variable", "portfolio_return_window")
            result.insert(7, "model", f"portfolio_return_window_on_{sentiment_regressor}_and_factors")
            rows.append(result)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return add_interpretation_columns(combined)


def summarize_sentiment_terms(results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    terms = results.loc[results["term"].eq(results["sentiment_regressor"])].copy()
    if terms.empty:
        return terms
    return (
        terms.groupby(group_columns, dropna=False)
        .agg(
            model_count=("term", "size"),
            mean_coef=("coef", "mean"),
            mean_hac_t=("t_stat", "mean"),
            mean_expected_signed_t=("expected_signed_t_stat", "mean"),
            mean_bootstrap_t=("bootstrap_t_stat", "mean"),
            mean_expected_signed_bootstrap_t=("expected_signed_bootstrap_t_stat", "mean"),
            hac_theory_consistent_5pct_share=("theory_consistent_hac_5pct", "mean"),
            bootstrap_theory_consistent_5pct_share=("theory_consistent_bootstrap_5pct", "mean"),
            mean_r2=("r2", "mean"),
            min_n_obs=("n_obs", "min"),
            median_n_obs=("n_obs", "median"),
        )
        .reset_index()
    )


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def fmt(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{100.0 * float(value):.0f}\\%"


def build_latex_summary_table(summary_by_horizon: pd.DataFrame, correlation: pd.DataFrame, stv_eigen: pd.DataFrame) -> str:
    columns = ["SENT_ORTH_DIV_PREMIUM", "SENT_ORTH", "SENT_ORTH_PAPER_LEAD_LAG", "STV_SWE"]
    rows: list[tuple[str, dict[str, str]]] = []

    corr_lookup = correlation.set_index("variable") if not correlation.empty and "variable" in correlation.columns else pd.DataFrame()
    rows.append(
        (
            r"Correlation with \var{SENT^{\perp}}",
            {
                column: fmt(corr_lookup.loc[column, "SENT_ORTH_DIV_PREMIUM"], 3)
                if column in corr_lookup.index and "SENT_ORTH_DIV_PREMIUM" in corr_lookup.columns
                else "--"
                for column in columns
            },
        )
    )
    for horizon in [1, 3, 6, 12]:
        h = summary_by_horizon.loc[summary_by_horizon["return_horizon_months"].eq(horizon)]
        lookup = h.set_index("sentiment_index_label")
        rows.append(
            (
                f"Mean expected-signed HAC $t$, {horizon}-month horizon",
                {
                    column: fmt(lookup.loc[column, "mean_expected_signed_t"], 2) if column in lookup.index else "--"
                    for column in columns
                },
            )
        )
    h12 = summary_by_horizon.loc[summary_by_horizon["return_horizon_months"].eq(12)].set_index("sentiment_index_label")
    rows.append(
        (
            r"Theory-consistent 5\% HAC share, 12-month horizon",
            {
                column: fmt_pct(h12.loc[column, "hac_theory_consistent_5pct_share"]) if column in h12.index else "--"
                for column in columns
            },
        )
    )
    stv_pc1 = stv_eigen["explained_variance_ratio"].mean() if not stv_eigen.empty else np.nan
    rows.append(
        (
            "Average rolling PC1 variance share",
            {
                "SENT_ORTH_DIV_PREMIUM": "--",
                "SENT_ORTH": "--",
                "SENT_ORTH_PAPER_LEAD_LAG": "--",
                "STV_SWE": fmt_pct(stv_pc1),
            },
        )
    )

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Alternative sentiment-index robustness comparison}",
        r"\label{tab:appendix_alternative_sentiment_indices}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xrrrr}",
        r"\toprule",
        "Measure & " + " & ".join(LABELS[column] for column in columns) + r" \\",
        r"\midrule",
    ]
    for measure, values in rows:
        lines.append(measure + " & " + " & ".join(values[column] for column in columns) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\tablenote{All regressions use the current 11-characteristic directional-spread panel and the same factor controls as the main specification. Expected-signed \(t\)-statistics multiply the sentiment coefficient \(t\)-statistic by the predicted negative sign of the sentiment-prone spread. \var{STV^{SWE}} is a 36-month rolling, macro-residualized first-component PCA index adapted from the enhanced sentiment-index methodology.}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly, diagnostics = build_alternative_indices()
    save_table(monthly, "alternative_sentiment_monthly")
    index_columns = ["SENT_ORTH_DIV_PREMIUM", "SENT_ORTH", "SENT_ORTH_PAPER_LEAD_LAG", "STV_SWE"]
    correlation = monthly[index_columns].corr().reset_index().rename(columns={"index": "variable"})
    save_table(correlation, "alternative_sentiment_correlation_matrix")
    for name, frame in diagnostics.items():
        save_table(frame, f"alternative_{name}")

    panel = pd.read_csv(require_file(FINAL_OUTPUT_DIR / "final_predictive_regression_panel_directional_spreads.csv"))
    panel["month_end_date"] = pd.to_datetime(panel["month_end_date"], errors="coerce")
    merge_columns = ["country_code", "month_end_date", *index_columns, *[f"{column}_lag" for column in index_columns]]
    panel = panel.drop(columns=[column for column in merge_columns if column not in {"country_code", "month_end_date"}], errors="ignore")
    panel = panel.merge(monthly[merge_columns], on=["country_code", "month_end_date"], how="left")

    sorts = set(panel["sort_variable"].dropna().unique())
    if "GS" not in sorts or "IVOL_MKT" in sorts:
        raise RuntimeError("Alternative comparison must use the current 11-characteristic panel with GS and without IVOL_MKT.")

    results = run_regressions(panel)
    save_table(results.drop(columns=[c for c in results.columns if c.startswith("_")], errors="ignore"), "alternative_predictive_results_directional_spreads_bootstrap")
    bootstrap_columns = [
        "bootstrap_std_error",
        "bootstrap_t_stat",
        "bootstrap_p_value",
        "bootstrap_repetitions",
        "bootstrap_block_length",
        "bootstrap_seed",
        "bootstrap_valid_repetitions",
    ]
    save_table(results.drop(columns=bootstrap_columns, errors="ignore"), "alternative_predictive_results_directional_spreads")

    summary_by_horizon = summarize_sentiment_terms(
        results,
        ["return_horizon_months", "sentiment_index_label", "sentiment_regressor", "main_specification_flag"],
    )
    summary_by_characteristic = summarize_sentiment_terms(
        results,
        ["sort_variable", "sentiment_index_label", "sentiment_regressor", "main_specification_flag"],
    )
    save_table(summary_by_horizon, "alternative_summary_by_horizon")
    save_table(summary_by_characteristic, "alternative_summary_by_characteristic")

    table = build_latex_summary_table(
        summary_by_horizon,
        correlation,
        diagnostics.get("stv_swe_rolling_eigenvalues", pd.DataFrame()),
    )
    (OUTPUT_DIR / "alternative_sentiment_appendix_table.tex").write_text(table)

    print(f"Wrote alternative sentiment-index appendix outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
