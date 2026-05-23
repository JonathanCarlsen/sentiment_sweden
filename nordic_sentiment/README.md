# Nordic Sentiment: Sweden Pipeline Guide

The appendix-facing production path is intentionally compact:

- `notebooks/01_sweden/FINAL_SENTIMENT_SWEDEN.ipynb` reproduces the final empirical results.
- `notebooks/01_sweden/00b_describe_sweden_equity_market.ipynb` produces the retained descriptive market-composition outputs.
- `src_final/nordic_sentiment/` is the reduced appendix-facing Python package used by the final notebook.

The original `src/nordic_sentiment/` package and older staged notebooks, broad model-comparison scripts, large exploratory outputs, and unused SQL infrastructure are retained for audit history. They are not part of the thesis production path.

This repository contains the active analytical pipeline for the Sweden-only part of a bachelor thesis in behavioral finance. The codebase is built to ingest raw Sweden market and fundamentals data, resolve a canonical monthly market panel, construct accounting snapshots and firm characteristics, build factor controls, stage sentiment-related proxy inputs, and support later portfolio and regression analysis.

## What This Repository Does

At a high level, the repository does five things:

1. Loads raw Sweden market, fundamentals, rates, and macro-control files from disk.
2. Standardizes those files into reusable staged tables with stable identifiers.
3. Builds canonical analytical marts:
   - daily prices
   - resolved monthly market panel
   - filing-level fundamentals
   - monthly as-of fundamental snapshots
   - firm characteristics and deciles
   - monthly factor controls
   - daily factor and monthly IVOL tables
   - sentiment proxy marts and placeholder sentiment outputs
4. Exposes the final thesis run through a compact Jupyter notebook, with broad diagnostics archived separately.
5. Provides tests for the core transformation logic so methodological rules remain explicit and reproducible.

The appendix-facing orchestration entrypoint is [src_final/nordic_sentiment/sweden_pipeline.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src_final/nordic_sentiment/sweden_pipeline.py).

## Start Here

If you have not seen the project before, read the written docs in this order:

1. [NORDIC_SENTIMENT.md](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/NORDIC_SENTIMENT.md)
   - Short project note.
   - Use this to confirm scope: Sweden-first, Sweden-only runtime.

2. [docs/architecture.md](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/docs/architecture.md)
   - High-level pipeline shape.
   - Explains how raw files become staged tables, marts, notebooks, and outputs.

3. [docs/schema.md](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/docs/schema.md)
   - Lists the intent of the main dimensions, facts, and marts.
   - Use this when you need to know what a table is supposed to represent.

4. [docs/sweden_data_audit.md](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/docs/sweden_data_audit.md)
   - Describes the raw Sweden input files, row counts, coverage, and known data issues.
   - Read this before debugging source-data behavior.

5. This `README.md`
   - This file explains where the implementation lives and how the pieces connect.

If you need the thesis-oriented navigation layer rather than the code/runtime layer, use:

- [AGENTS/REPO_MAP.md](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/AGENTS/REPO_MAP.md)
- [AGENTS/repo_map.yml](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/AGENTS/repo_map.yml)

Those files are meant to help with efficient repository retrieval and thesis writing, not day-to-day pipeline execution.

## Repository Layout

```text
nordic_sentiment/
├── config/          Source locations, mappings, countries, variable metadata
├── docs/            Human-readable pipeline, schema, and data notes
├── notebooks/       Sweden notebook workflow
├── src/             Python package with the active analytical logic
├── tests/           Unit and notebook-smoke tests
├── NORDIC_SENTIMENT.md
└── README.md
```

## Data Inputs and Configuration

The raw source configuration is defined in:

- [config/sources.yml](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/config/sources.yml)

This is the file that tells the pipeline where the Sweden source files live. It includes:
- daily prices
- monthly prices
- quarterly fundamentals
- 3M interbank rate
- 10Y bond yield
- economic and consumer sentiment workbook
- macro controls such as CPI, PPI, IP, and EM

Other configuration files:

- [config/countries.yml](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/config/countries.yml)
  - country metadata; now effectively Sweden-only in runtime use
- [config/mappings.yml](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/config/mappings.yml)
  - field and code mapping support
- [config/variables.yml](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/config/variables.yml)
  - variable-oriented metadata
- [config/sentiment_proxies.yml](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/config/sentiment_proxies.yml)
  - sentiment proxy configuration surface

## Runtime Entry Points

### Main pipeline entrypoint

- [src/nordic_sentiment/sweden_pipeline.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/sweden_pipeline.py)

This file defines:
- `SwedenRawData`
- `SwedenStagedData`
- `SwedenArtifacts`
- `load_sweden_raw_data()`
- `stage_sweden_raw_data()`
- `build_sweden_artifacts()`

If you want to understand what the pipeline produces end-to-end, this is the first code file to read.

### Paths and project config

- [src/nordic_sentiment/config.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/config.py)
- [src/nordic_sentiment/paths.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/paths.py)

These files are the runtime entry for reading project configuration and locating key directories such as notebooks.

## Where Each Part of the Pipeline Is Implemented

### 1. Raw file loading

Folder:
- `src/nordic_sentiment/io/`

Main files:
- [io/compustat_sweden.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/io/compustat_sweden.py)
  - loads the Sweden market, fundamentals, rate, sentiment, and macro files from the configured source paths
- [io/storage.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/io/storage.py)
  - storage helper surface

Current raw-file conventions:
- the large Sweden market and fundamentals source tables are standard CSV files
- the smaller workbook sources, such as the economic and consumer sentiment file, are read through normal `pandas`/`openpyxl` parsing
- there is no custom XLSX fallback parser in the active pipeline

### 2. Staging and normalization

Folder:
- `src/nordic_sentiment/staging/`

Main files:
- [staging/sweden.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/staging/sweden.py)
  - normalizes daily prices, monthly prices, and quarterly fundamentals
  - parses dates, renames source columns, standardizes strings, and computes staged `book_equity`
- [staging/sweden_rates.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/staging/sweden_rates.py)
  - stages the monthly 3M and 10Y rate series into a standard monthly rate table

This layer is where raw workbook/CSV field names stop mattering and canonical internal columns begin.

### 3. Identity and dimensions

Folder:
- `src/nordic_sentiment/identifiers/`

Main file:
- [identifiers/builders.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/identifiers/builders.py)

This file is responsible for:
- constructing `company_id = country_code + gvkey`
- constructing `security_id = country_code + gvkey + iid`
- building `dim_company`
- building `dim_security`

This is the canonical identity layer. Most downstream tables key off these IDs, not raw `isin` or source-specific identifiers.

### 4. Market panel construction

Folder:
- `src/nordic_sentiment/market/`

Main file:
- [market/monthly.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/market/monthly.py)

This file implements the Sweden monthly market strategy:
- derive monthly rows from daily prices for `2000-01` to `2006-12`
- use the monthly workbook as primary from `2007-01` onward
- fall back to daily-derived monthly rows when the workbook is missing or invalid

It produces:
- `fct_market_monthly_source`
- `fct_market_monthly`

If you want to understand where the canonical monthly market panel comes from, read this file.

### 5. Fundamental filing and as-of snapshot logic

Folder:
- `src/nordic_sentiment/fundamentals/`

Main file:
- [fundamentals/snapshots.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/fundamentals/snapshots.py)

This file implements:
- filing-level fundamental tables
- `report_available_date`
- `effective_month_end`
- the monthly as-of snapshot mart

This is the core anti-lookahead layer for accounting data. Any characteristic or factor using fundamentals should depend on these monthly snapshots rather than on raw filings directly.

### 6. Characteristics layer

Main file:
- [market/characteristics.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/market/characteristics.py)

This file builds:
- `mart_characteristics_monthly`
- `mart_characteristics_annual_comparison`
- the Sweden variable availability audit

Implemented characteristics include:
- `ME`
- `risk`
- `BE`
- `BE_ME`
- `PPE_A`
- `RD_A`
- `E_plus_BE`
- `D_BE`
- `GS`
- `EF_A`
- `age`

It also implements:
- within-year winsorization for selected continuous characteristics
- monthly or annual decile assignment, depending on characteristic
- special handling for zero-bin style characteristics

This is the main stock-level empirical feature layer used later in sorts and regressions.

### 7. Monthly control factors

Main file:
- [market/controls.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/market/controls.py)

This file stages and builds:
- `fct_rate_monthly`
- `fct_factor_monthly`
- `audit_factor_legs_monthly`

Implemented monthly factors:
- `RM_LOCAL_VW`
- `RMRF`
- `SMB`
- `HML`
- `UMD`
- `RF_3M_PROXY`
- `YIELD_10Y`

This is the Sweden monthly controls layer used for descriptive and later regression work.

### 8. Daily factors and IVOL

Main file:
- [market/daily_factors.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/market/daily_factors.py)

This file implements the IVOL extension. It builds:
- `fct_factor_daily`
- `fct_ivol_monthly`

Daily factor logic:
- derive daily stock returns from adjusted prices `prccd * trfd`
- derive daily RF from the monthly 3M rate proxy using within-month trading-day compounding
- build daily `MKT_RF_D`
- build daily `SMB_D`
- build daily `HML_D`

Monthly IVOL logic:
- estimate `ivol_mkt` from daily market-model residuals
- estimate `ivol_ff3` from daily FF3-style residuals
- store stock-month observation counts and status codes

The resulting IVOL measures are merged back into the characteristics mart as:
- `IVOL_MKT`
- `IVOL_FF3`
- `IVOL_MKT10`
- `IVOL_FF310`

### 9. Sentiment proxy and sentiment-index scaffolding

Folder:
- `src/nordic_sentiment/sentiment/`

Main files:
- [sentiment/sweden.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/sentiment/sweden.py)
  - Sweden-specific sentiment-proxy source builders
  - paper-spec audit scaffolding
  - macro-control parsing for sentiment work
- [sentiment/index.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/sentiment/index.py)
  - generic monthly and annual sentiment-index helpers

Current status:
- the sentiment workflow is scaffolded and partially implemented
- the final thesis sentiment path is `SENT_ORTH_DIV_PREMIUM`, with `SENT_DIV_PREMIUM` retained as the raw non-orthogonalized comparison.

### 10. Portfolio logic

Folder:
- `src/nordic_sentiment/portfolios/`

Main file:
- [portfolios/sorts.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/src/nordic_sentiment/portfolios/sorts.py)

This folder is where portfolio sort logic lives. It is intended to sit downstream of the characteristics and factor layers.

### 11. Regression layer

Folder:
- `src/nordic_sentiment/regressions/`

This package contains the HAC/Newey-West and moving-block bootstrap helpers used by the final thesis notebook.

### 12. Archived database layer

The earlier SQL persistence helpers have been moved to `../archive/src/nordic_sentiment/db/`.
The final thesis workflow is notebook and DataFrame based and does not require a database connection.

## Sweden Notebook Workflow

The visible notebooks under `notebooks/01_sweden/` are now the appendix-facing thesis workflow.

Recommended run order:

1. `FINAL_SENTIMENT_SWEDEN.ipynb`
   - reproduces the final empirical analysis and writes to `outputs/final_sentiment_sweden`.
2. `00b_describe_sweden_equity_market.ipynb`
   - reproduces the descriptive market-composition outputs and writes to `outputs/eda_sweden_equity_market`.

The older staged notebooks and the large master notebook are archived under `../archive/notebooks/`.

## Tests

Folder:
- `tests/`

Important test files:
- [tests/test_monthly_market.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/tests/test_monthly_market.py)
  - monthly market derivation and resolution logic
- [tests/test_fundamental_snapshots.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/tests/test_fundamental_snapshots.py)
  - filing snapshot timing rules
- [tests/test_sweden_controls.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/tests/test_sweden_controls.py)
  - rate staging and monthly factor construction
- [tests/test_sweden_ivol.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/tests/test_sweden_ivol.py)
  - daily RF derivation, daily factor table, and IVOL estimation
- [tests/test_sweden_pipeline.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/tests/test_sweden_pipeline.py)
  - end-to-end Sweden artifact build on small fixtures
- [tests/test_sweden_notebooks.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/tests/test_sweden_notebooks.py)
  - notebook existence and smoke-level content validation
- [tests/test_sweden_sentiment.py](/Users/jonathancarlsen/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/3.%20%C3%A5r/Bachelor/vs_inference/nordic_sentiment/tests/test_sweden_sentiment.py)
  - Sweden sentiment proxy and index scaffolding

If you need to validate the active Sweden analytical path, start with these tests rather than the full repo.

## Key Methodological Rules Embedded in Code

These rules are important because they determine how the thesis results are actually produced:

- Monthly market data before `2007-01` is derived from daily prices.
- Monthly market data from `2007-01` onward uses the workbook as primary, with daily fallback.
- Fundamentals are used through monthly as-of snapshots using `effective_month_end`, not raw filing dates directly.
- Characteristics are built at the monthly stock level and then summarized annually via December or year-end last observation.
- Monthly factor controls use the Sweden 3M interbank series as the risk-free proxy and the Sweden 10Y bond yield as a separate long-rate series.
- Daily IVOL uses a daily risk-free rate derived from the monthly 3M proxy because a continuous daily Sweden risk-free series is not available over the full sample.
- Sentiment scaffolding exists, but the paper-faithful final sentiment index is not complete until the missing Bloomberg-dependent proxies are sourced.

## How to Understand an Output Table

If you encounter a table name and need to know where it comes from, use this guide:

- `dim_*`
  - identity or calendar dimensions
- `fct_price_daily`
  - staged daily market data
- `fct_market_monthly_source`
  - competing raw monthly variants before resolution
- `fct_market_monthly`
  - resolved monthly market panel
- `fct_fundamental_filing`
  - staged filing-level accounting data
- `mart_fundamental_snapshot_monthly`
  - latest available fundamentals as of each month-end
- `mart_characteristics_monthly`
  - stock-month empirical characteristic layer
- `fct_rate_monthly`
  - monthly Sweden rates
- `fct_factor_monthly`
  - monthly Sweden factor controls
- `fct_factor_daily`
  - daily factor table for IVOL work
- `fct_ivol_monthly`
  - stock-month IVOL outputs
- `fct_macro_controls_monthly`
  - macro-control inputs used in sentiment work
- `fct_sentiment_proxy_monthly_source`
  - raw standardized sentiment proxy inputs
- `mart_sentiment_proxy_monthly`
  - cleaned monthly proxy mart
- `mart_sentiment_index_monthly`
  - current generic sentiment index output
- `mart_sentiment_index_annual`
  - annualized sentiment output

## What Is Still Partial or Incomplete

This repository is usable, but not every analytical branch is equally mature.

Current partial areas:
- the paper-faithful Sweden sentiment index is still incomplete because some required proxies depend on Bloomberg or equivalent sources
- the later portfolio and regression layers are present as workflow destinations, but the core code emphasis is still on data engineering and analytical marts
- the database layer exists, but the notebook/DataFrame path is the active practical runtime

Anyone reading or extending the project should distinguish between:
- implemented and tested core marts
- staged but incomplete sentiment work
- later empirical outputs that depend on completing the missing inputs

## Minimal Execution Path

To work with the project from scratch:

1. Ensure the raw files exist at the paths expected by `config/sources.yml`.
2. Install the package in editable mode:

```bash
pip install -e .
```

3. If you are using the DB layer, configure the MySQL environment variables described in the codebase.
4. Open the Sweden notebooks under `notebooks/01_sweden/`.
5. Run the notebooks in the documented order.

For code-first validation, use:

```bash
python3 -m pytest tests/test_monthly_market.py tests/test_fundamental_snapshots.py tests/test_sweden_controls.py tests/test_sweden_ivol.py tests/test_sweden_pipeline.py
```

## If You Need to Modify the Project

Use this rule of thumb:

- Change `config/` if the source files or mappings changed.
- Change `io/` or `staging/` if raw file formats changed.
- Change `identifiers/` if key construction changed.
- Change `market/monthly.py` if the canonical monthly market panel logic changed.
- Change `fundamentals/snapshots.py` if accounting timing changed.
- Change `market/characteristics.py` if a characteristic definition changed.
- Change `market/controls.py` if monthly factor controls changed.
- Change `market/daily_factors.py` if IVOL or daily factor logic changed.
- Change `sentiment/sweden.py` if sentiment proxy definitions or macro-control parsing changed.
- Change notebooks only when you want audit or presentation-layer behavior to change.

That separation is deliberate. Core methodological transformations should live in Python modules, not as duplicated notebook code.
