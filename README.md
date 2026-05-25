# Sentiment Sweden

This repository contains the empirical code, source data pointers, final outputs, and submitted thesis PDF for a bachelor thesis on investor sentiment and cross-sectional return predictability in the Swedish equity market.

The project asks whether beginning-of-period investor sentiment predicts relative future returns across Swedish firm-characteristic portfolios, and whether the relation is strongest among firms that are harder to value or harder to arbitrage. The empirical pipeline constructs a Swedish sentiment index, builds firm-characteristic portfolios, estimates predictive regressions, and produces thesis tables and figures.

(This .README was created with AI)
## What Is in This Repository

The repository is intentionally centered on the final Sweden analysis rather than the full exploratory project history.

```text
.
├── S170414.pdf
├── nordic_sentiment/
├── outputs/final_sentiment_sweden/
├── scripts/
├── Daily prices jan 2000 - dec 2025 zip limited (preferred).csv
├── monthly_prices_jan 2005 - dec 2025 (sweden filter.csv
├── quarterly_fundamentals_sweden jan 2000 - dec 2025.csv
└── omxs30_monthly_yfinance.csv
```

Key components:

- `S170414.pdf` is the submitted thesis PDF.
- `nordic_sentiment/notebooks/01_sweden/FINAL_SENTIMENT_SWEDEN.ipynb` is the canonical notebook that reproduces the final empirical analysis.
- `nordic_sentiment/src_final/nordic_sentiment/` is the reduced Python package used by the final notebook.
- `nordic_sentiment/config/` contains source-file paths, mappings, country metadata, variable metadata, and sentiment-proxy configuration.
- `outputs/final_sentiment_sweden/` contains the final generated tables, figures, regression panels, diagnostics, and robustness outputs used in the thesis.
- `scripts/` contains helper scripts used to generate selected thesis tables and figures from the final outputs.
- The large raw market and fundamentals CSV files are tracked with Git LFS.

## Main Reproducibility Path

The final analysis is reproduced through:

```text
nordic_sentiment/notebooks/01_sweden/FINAL_SENTIMENT_SWEDEN.ipynb
```

The notebook imports from:

```text
nordic_sentiment/src_final/
```

and writes its main outputs to:

```text
outputs/final_sentiment_sweden/
```

The notebook is the authoritative execution path for the final thesis results. Older exploratory notebooks and the broader development package are not required for the final replication path.

## Data Inputs

The raw data files used by the final pipeline are configured in:

```text
nordic_sentiment/config/sources.yml
```

The largest source files are tracked with Git LFS:

```text
Daily prices jan 2000 - dec 2025 zip limited (preferred).csv
monthly_prices_jan 2005 - dec 2025 (sweden filter.csv
quarterly_fundamentals_sweden jan 2000 - dec 2025.csv
```

The final notebook also uses:

```text
omxs30_monthly_yfinance.csv
```

Users cloning this repository should install Git LFS before cloning or pulling the raw data:

```bash
git lfs install
git lfs pull
```

## Python Environment

The project was developed in Python using common empirical-finance and data-analysis libraries, including:

- `pandas`
- `numpy`
- `matplotlib`
- `statsmodels`
- `scipy`
- `jupyter`
- `openpyxl`

If a clean environment is needed, install the required packages before running the notebook:

```bash
pip install pandas numpy matplotlib statsmodels scipy jupyter openpyxl
```

## Running the Final Notebook

From the repository root, start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
nordic_sentiment/notebooks/01_sweden/FINAL_SENTIMENT_SWEDEN.ipynb
```

Run the notebook from top to bottom. The notebook should read the configured source files, construct the analytical panels, estimate the regressions, and write outputs under:

```text
outputs/final_sentiment_sweden/
```

## Final Output Folder

`outputs/final_sentiment_sweden/` contains the final audit trail for the thesis. Important files include:

- `final_sentiment_monthly.csv`
- `final_sentiment_regression_panel_monthly.csv`
- `final_predictive_regression_panel_directional_spreads.csv`
- `final_predictive_regression_panel_by_leg.csv`
- `final_predictive_results_directional_spreads.csv`
- `final_predictive_results_by_leg.csv`
- `final_nonoverlap_horizon_results.csv`
- `final_sentiment_innovation_results.csv`
- `final_sentiment_orth_div_premium_over_time.png`
- `final_omxs30_sent_orth_drawdowns_over_time.png`

The output folder is included so the thesis results can be inspected without rerunning the full notebook.

## Helper Scripts

The `scripts/` folder contains helper scripts for final thesis tables and figures. These scripts generally read from `outputs/final_sentiment_sweden/` and write table or figure files back into the same output area.

Examples:

- `scripts/build_bw2006_horizon_tables.py`
- `scripts/build_factor_control_robustness.py`
- `scripts/build_sentiment_pca_construction_tables.py`
- `scripts/build_standardized_proxy_timeseries_plot.py`
- `scripts/build_omxs30_proxy_drawdown_panels.py`
- `scripts/build_sibley_appendix_table.py`
- `scripts/build_spread_correlation_heatmap.py`

## Repository Scope

This repository is not a general-purpose asset-pricing library. It is a thesis replication repository for one Sweden-focused empirical project.

The final production path is:

```text
raw source files
    -> nordic_sentiment/src_final/
    -> FINAL_SENTIMENT_SWEDEN.ipynb
    -> outputs/final_sentiment_sweden/
    -> S170414.pdf
```

## Notes on Git LFS

The large raw source files are stored with Git LFS. If the CSV files appear as small text pointer files after cloning, run:

```bash
git lfs pull
```

If Git LFS is not installed, install it before attempting to reproduce the raw-data pipeline.
