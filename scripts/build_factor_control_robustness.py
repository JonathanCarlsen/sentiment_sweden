from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


OUTPUT_DIR = Path("outputs/final_sentiment_sweden")
PANEL_PATH = OUTPUT_DIR / "final_predictive_regression_panel_directional_spreads.csv"
FULL_OUT = OUTPUT_DIR / "final_factor_control_robustness.csv"
SUMMARY_OUT = OUTPUT_DIR / "final_factor_control_robustness_summary_by_horizon.csv"
SELECTED_OUT = OUTPUT_DIR / "final_factor_control_robustness_selected_12m.csv"

SENTIMENT_REGRESSOR = "SENT_ORTH_DIV_PREMIUM_lag"
DEPENDENT_VARIABLE = "portfolio_return_window"
FACTOR_TERMS = ["RMRF", "SMB", "HML", "UMD"]

SELECTED_SORTS = [
    "risk",
    "IVOL_MKT",
    "IVOL_FF3",
    "PPE_A",
    "NON_D_PAYER",
    "ME",
    "UNPROFITABLE",
]

SORT_LABELS = {
    "risk": "Total risk",
    "IVOL_MKT": "IVOL (market)",
    "IVOL_FF3": "IVOL (FF3)",
    "PPE_A": "Low tangibility",
    "NON_D_PAYER": "Non-dividend payer",
    "ME": "Small firms",
    "UNPROFITABLE": "Unprofitable",
    "age": "Young firms",
    "ILLIQ": "Illiquidity",
    "E_plus_BE": "Low profitability",
    "XTURN": "Low turnover",
}


def hac_lags(horizon: int) -> int:
    return max(4, int(horizon) - 1)


def fit_model(frame: pd.DataFrame, terms: list[str], horizon: int) -> dict[str, float]:
    x = sm.add_constant(frame[terms], has_constant="add")
    y = frame[DEPENDENT_VARIABLE]
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags(horizon)})
    return {
        "coef": float(model.params[SENTIMENT_REGRESSOR]),
        "std_error": float(model.bse[SENTIMENT_REGRESSOR]),
        "t_stat": float(model.tvalues[SENTIMENT_REGRESSOR]),
        "r2": float(model.rsquared),
    }


def main() -> None:
    panel = pd.read_csv(PANEL_PATH)
    rows: list[dict[str, object]] = []

    group_cols = [
        "sort_variable",
        "sentiment_prone_direction",
        "sentiment_prone_interpretation",
        "return_horizon_months",
    ]

    required_cols = [DEPENDENT_VARIABLE, SENTIMENT_REGRESSOR, *FACTOR_TERMS]
    for keys, group in panel.groupby(group_cols, dropna=False, sort=True):
        sort_variable, prone_direction, prone_interpretation, horizon = keys
        sample = group[required_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sample) < 30:
            continue

        sentiment_only = fit_model(sample, [SENTIMENT_REGRESSOR], int(horizon))
        factor_adjusted = fit_model(sample, [SENTIMENT_REGRESSOR, *FACTOR_TERMS], int(horizon))
        coef_change = factor_adjusted["coef"] - sentiment_only["coef"]

        rows.append(
            {
                "sort_variable": sort_variable,
                "characteristic": SORT_LABELS.get(str(sort_variable), str(sort_variable)),
                "sentiment_prone_direction": prone_direction,
                "sentiment_prone_interpretation": prone_interpretation,
                "return_horizon_months": int(horizon),
                "n_obs": int(len(sample)),
                "sentiment_only_coef": sentiment_only["coef"],
                "factor_adjusted_coef": factor_adjusted["coef"],
                "coef_change": coef_change,
                "sentiment_only_t": sentiment_only["t_stat"],
                "factor_adjusted_t": factor_adjusted["t_stat"],
                "sentiment_only_r2": sentiment_only["r2"],
                "factor_adjusted_r2": factor_adjusted["r2"],
                "delta_r2": factor_adjusted["r2"] - sentiment_only["r2"],
                "sentiment_sign_change": bool(
                    np.sign(sentiment_only["coef"]) != np.sign(factor_adjusted["coef"])
                ),
                "hac_lags": hac_lags(int(horizon)),
            }
        )

    full = pd.DataFrame(rows).sort_values(["return_horizon_months", "sort_variable"])
    full.to_csv(FULL_OUT, index=False)

    summary = (
        full.groupby("return_horizon_months", as_index=False)
        .agg(
            n_characteristics=("sort_variable", "count"),
            mean_sentiment_only_coef=("sentiment_only_coef", "mean"),
            mean_factor_adjusted_coef=("factor_adjusted_coef", "mean"),
            mean_coef_change=("coef_change", "mean"),
            mean_sentiment_only_r2=("sentiment_only_r2", "mean"),
            mean_factor_adjusted_r2=("factor_adjusted_r2", "mean"),
            mean_delta_r2=("delta_r2", "mean"),
            sign_change_share=("sentiment_sign_change", "mean"),
        )
        .sort_values("return_horizon_months")
    )
    summary.to_csv(SUMMARY_OUT, index=False)

    selected = full[
        full["return_horizon_months"].eq(12) & full["sort_variable"].isin(SELECTED_SORTS)
    ].copy()
    selected["selection_order"] = selected["sort_variable"].map(
        {key: i for i, key in enumerate(SELECTED_SORTS)}
    )
    selected = selected.sort_values("selection_order").drop(columns="selection_order")
    selected.to_csv(SELECTED_OUT, index=False)

    print(f"Wrote {FULL_OUT} with {len(full)} rows")
    print(f"Wrote {SUMMARY_OUT} with {len(summary)} rows")
    print(f"Wrote {SELECTED_OUT} with {len(selected)} rows")


if __name__ == "__main__":
    main()
