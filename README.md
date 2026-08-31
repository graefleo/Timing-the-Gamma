# Predicting Fama–MacBeth Risk Premia

The pipeline runs monthly Fama–MacBeth regressions on six characteristics (value, momentum,
size, operating profitability, asset growth, market beta), turns the estimated premia into
six gamma time series, and predicts them out-of-sample with models ranging from a historical
mean up to LASSO, random forest, XGBoost, MLP and LSTM. Economic value is evaluated
via direction-timing long–short portfolios.

---

## Repository policy: code + one data file

The R/Python code, this README, and **one**
data object, `panel_clean.rds`, are committed.
`panel_clean.rds` is the cleaned CRSP + Chen–Zimmermann stock panel; it is built from the
server-side Tidy Finance SQLite database and therefore **cannot be regenerated without
server access**, so it ships with the repo.

---

## What you must obtain yourself

The data-acquisition scripts (`00`, `01`, `01c`, `01d`) need external resources that are
**not** in this repo:

| Needed by | Resource | How to get it |
|---|---|---|
| `01c_macro_predictors.R` | **FRED API key** | Free at <https://fred.stlouisfed.org/docs/api/api_key.html>; set `FRED_API_KEY` as an environment variable |
| `01_data_pipeline.R` | **Tidy Finance SQLite DB** (CRSP returns, market cap, FF5) | Build via <https://www.tidy-finance.org/> (WRDS/CRSP subscription); path hard-coded to `/home/shared/data/tidy_finance.sqlite` |
| `00_trim_cz.R` | **Chen–Zimmermann signed predictors** (`signed_predictors_dl_wide.csv`, ~7.8 GB) | <https://www.openassetpricing.com/data/> |
| `01d_goyal_welch.R` | **Shiller `ie_data.xls`** | Auto-downloaded from Yale if missing; or place `shiller_ie_data.xls` in the working dir |
| `01e_momentum_factor.R` | **Fama–French momentum factor (UMD)** | Auto-downloaded from the Ken French data library if missing; cached as `F-F_Momentum_Factor.csv` |

---

## Requirements

- **R** (tidyverse, RSQLite, lubridate, slider, data.table, fredr, sandwich, lmtest, vars,
  glmnet, cpm, xgboost, PerformanceAnalytics, readxl)
- **Python / GPU (optional)** — the MLP and LSTM run on Google Colab via the CSV bridge in
  `colab/`; see `colab/` for the notebooks and import/export scripts. Pure-R paths exist in
  `04b`/`04c` as well.

Set the working directory to the repo root — all scripts read/write by bare filename.

---

## Run order

| Script | Needs external data? | Purpose |
|---|---|---|
| `00_trim_cz.R` | ✅ 7.8 GB CZ CSV | Subset 5 signals → `cz_signals_subset.rds` |
| `01_data_pipeline.R` | ✅ server SQLite | CRSP + CZ cleaning → `panel_clean.rds`*, `ff5_factors.rds`, `char_spreads.rds` |
| `01c_macro_predictors.R` | ✅ FRED key | FRED macro variables → `macro_predictors.rds` |
| `01d_goyal_welch.R` | ✅ Shiller xls | Goyal–Welch predictors → `goyal_welch_predictors.rds` |
| `01e_momentum_factor.R` | ✅ auto-download | UMD momentum factor → `umd_factor.rds` (needed by `05` for the six-factor alphas) |
| `02_fama_macbeth.R` | — | Monthly FM cross-sectional regressions → `gammas_ts.rds` |
| `00_diagnostics.R`, `03_gamma_analysis.R`, `03b_visual_exploration.R` | — | Optional: sample diagnostics, gamma descriptives, exploratory plots |
| `03c_structural_breaks.R` | — | Online CPM/GLR break detection → Table 7 |
| `04_predict_gammas.R` | — | OOS prediction loop (M0–M9 + combos) → `gamma_predictions.rds` (+ `_h3`, `_smooth`) |
| `04b`–`04i` | — | MLP, LSTM, SHAP, plots, multiple testing (`04f`: BH/BY FDR, model confidence sets, HAC-bandwidth sensitivity), family ablation (`04g`), tail robustness (`04h`), gamma-vs-factor horserace (`04i`) |
| `05_portfolio_construction.R` | — | Portfolio construction & evaluation (incl. RAFE) → `portfolio_results.rds` |
| `05b_vw_robustness.R` | — | Value-weighted quintile-leg robustness of direction timing → `vw_robustness.rds` |
| `06_tables_export.R` / `07_appendix_export.R` | — | LaTeX tables (main text / appendix) |

\* only `panel_clean.rds` is shipped in the repo.

**Starting from a fresh clone:** `panel_clean.rds` is present, but the other pipeline
inputs are not. Either (a) run `00`/`01`/`01c`/`01d`/`01e` with your own data access to
rebuild them, or (b) if you only have `panel_clean.rds`, you can run `02` onward after also
regenerating `ff5_factors.rds`, `char_spreads.rds`, `macro_predictors.rds`,
`goyal_welch_predictors.rds` and `umd_factor.rds` from their source scripts. `05` fails
immediately without `umd_factor.rds`, and `06`/`07` then fail on the missing
`portfolio_results.rds`, so run `01e` before `05`.


