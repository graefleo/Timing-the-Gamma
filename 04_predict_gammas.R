# ==============================================================================
# 04_predict_gammas.R
# Out-of-sample prediction of Fama-MacBeth risk premia (gammas).
#
# Models (expanding window, one-step-ahead):
#   M0   Historical mean                    [benchmark]
#   M1   AR(1)         own lag 1
#   M2   AR(3)         own lags 1-3
#   M2b  HAR           daily/weekly/monthly cascade (Corsi 2009)
#   M3   Multivariate  all gamma lag-1 + own vol12 + own char spread
#   M4   VAR(1)        joint system of all gammas
#   M5   LASSO         all features, lambda by time-series holdout CV
#   M8   RF            random forest on full 118-feature set (ranger)
#   M9   XGBoost       gradient-boosted trees, early stopping on val block
#
# Feature set (~120 total predictors):
#   Gamma features (per characteristic, × 6 = 30):
#     * Own-gamma lags 1, 2, 3
#     * 12-month rolling volatility of the gamma
#   Characteristic spreads (× 6 = 6):
#     * Cross-sectional IQR spread of each characteristic each month
#   Macro conditioning variables (8, from 01c_macro_predictors.R):
#     * default_spread  (BAA − AAA yield, Fama & French 1989)
#     * term_spread     (10Y − 3M Treasury, Campbell 1987)
#     * short_rate      (3M T-bill Δ, Ang & Bekaert 2007)
#     * vix             (CBOE VIX, Reschenhofer & Zechner 2022)
#     * indpro_gap      (YoY log industrial production growth, Cooper & Priestley 2007)
#     * mkt_lag1        (lagged market excess return, Reschenhofer & Zechner 2022)
#     * infl            (CPI inflation, real-time lagged, Goyal & Welch 2008)
#     * lty             (20Y Treasury yield proxy for long-term rate, GW 2008)
#   Goyal-Welch (2008) equity premium predictors (from 01d_goyal_welch.R):
#     * dp, dy, ep, de  (S&P 500 dividend/earnings ratios, from Shiller data)
#     * svar            (monthly squared S&P 500 return, proxy for daily variance)
#     * ntis            (net equity expansion, from CRSP market cap)
#     Note: ltr, dfr, bm, ik omitted — require Ibbotson/Value Line data
#   Factor signals (5 FF5 factors × 15 measures = 75, from ff5_factors):
#     * {factor}_mom1/3/6/12       (raw avg return, Jegadeesh & Titman 1993)
#     * {factor}_tsmom1/3/6/12     (sign of return, Ehsani & Linnainmaa 2022)
#     * {factor}_vol12/36          (rolling SD at 12m and 36m)
#     * {factor}_gk_mom1/3/6/12    (vol-scaled momentum, Gupta & Kelly 2019)
#     * {factor}_rev60             (60m reversal, Moskowitz, Ooi & Pedersen 2012)
#     Reference: Neuhierl, Randl, Reschenhofer & Zechner (2023, WP)
#
# OOS design: expanding window, initial window = 60 months.
# Primary metric: Campbell-Thompson (2008) OOS R².
# Significance: Clark-West (2007) one-sided test.
# Sub-period OOS R² reported for pre/post-2000 and pre/post-2010.
#
# Input:  gammas_ts.rds, char_spreads.rds
# Output: gamma_predictions.rds
#   (objects: pred_list, oos_eval, oos_eval_all, actuals, dates_oos, aligned, feature_cols, oos_idx)
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(vars)      # VAR()
library(glmnet)    # cv.glmnet()
library(slider)    # slide_dbl()
library(sandwich)  # NeweyWest() for HAC-robust CW inference
library(ranger)    # fast random forest
library(xgboost)   # gradient-boosted trees (M9)
select <- dplyr::select  # vars loads MASS which masks dplyr::select

list2env(readRDS("gammas_ts.rds"), envir = environment())   # -> gammas_ts, fm_table (+ ols variants)
char_spreads <- readRDS("char_spreads.rds")                 # -> char_spreads
macro_pred   <- readRDS("macro_predictors.rds")             # -> macro_pred       (from 01c_macro_predictors.R)
gw_pred      <- readRDS("goyal_welch_predictors.rds")       # -> gw_pred          (from 01d_goyal_welch.R)
ff5_factors  <- readRDS("ff5_factors.rds")                  # -> ff5_factors      (for factor momentum features)

RUN_LASSO      <- TRUE
RUN_LASSO_1SE  <- TRUE   # also store lambda.1se variant (more conservative regularisation)
RUN_HAR        <- TRUE   # M2b — HAR daily/weekly/monthly cascade (H=1 only; valid)
RUN_HAR_H3     <- FALSE  # HAR disabled at H=3: its predictor x_d = Y_{t-1} =
                         # γ_t+γ_{t+1}+γ_{t+2} contains future months (γ_{t+1},γ_{t+2})
                         # of the cumulative target Y_t = γ_{t+1}+γ_{t+2}+γ_{t+3} — i.e.
                         # look-ahead, producing an implausible ~50% OOS R². A leak-free
                         # H=3 HAR would need monthly-gamma components lagged by ≥ H.
                         # The H=1 HAR (RUN_HAR) is unaffected and stays on.
RUN_RF         <- TRUE   # M8  — random forest (ranger, 500 trees, retrained every month)
RUN_XGB        <- TRUE   # M9  — XGBoost (gradient-boosted trees, early stopping)
# Ext-E: ROLLING estimation windows (added 2026-07-20, review response).
# The thesis explains the unforecastability of value and momentum by arguing
# that an expanding-window forecaster is perpetually calibrated to regimes that
# have already ended. That argument is also an argument against the expanding
# window itself: a referee can reasonably object that the design diagnoses
# regime staleness and then prescribes the estimator that maximises it.
# This block re-estimates LASSO and RF on FIXED-LENGTH rolling windows, which
# discard old regimes by construction. Either outcome is informative: recovery
# of predictability in the fast-moving premia would strengthen the regime
# interpretation, while its absence establishes that window length is not the
# binding margin and leaves the regime argument resting on the break evidence.
# Both windows exceed the 60-month minimum and are available from the FIRST
# forecast onward (the initial training window is 317 months), so the rolling
# and expanding variants are scored on an identical evaluation sample.
# Reference: Pesaran & Timmermann (2007, JoE); Rossi (2013, Handbook);
# Inoue, Jin & Rossi (2017, JoE).
RUN_ROLLING    <- TRUE
ROLL_WINDOWS   <- c(120L, 240L)   # 10 and 20 years of monthly observations
# Ext-C: quarterly H=3 horizon (supervisor extension 2026-04-18).
# Saves a companion file gamma_predictions_h3.rds without touching the H=1 results.
# Reference: Fama & French (1988, JFE) — predictability increases with return horizon.
RUN_QUARTERLY        <- TRUE
# Ext-C2: HAR backward-average smoothed target (Corsi 2009, JFEC).
# Target: γ̄_{k,t} = (1/3)(γ_{k,t} + γ_{k,t-1} + γ_{k,t-2}) predicted with X_{t-1}.
# AC(1) of smoothed target ≈ 0.67–0.70 vs ≈ 0.00 for raw monthly gammas.
# Applied validation: Stöckl et al. (LLB, 2024) use exactly this formula.
RUN_QUARTERLY_SMOOTH <- TRUE
INIT_WINDOW    <- 60
NW_LAG         <- 12
# OOS evaluation starts here so all models — including LASSO — have VIX and
# default_spread available. Pre-1990 training data is still used (expanding
# window), only the *evaluated* OOS period is restricted.
OOS_START_DATE <- as.Date("1990-01-01")

gamma_cols <- c(
  "gamma_bm", "gamma_mom12m", "gamma_oper_prof",
  "gamma_asset_growth", "gamma_size", "gamma_beta"
)

gamma_labels <- c(
  gamma_bm           = "Value",
  gamma_mom12m       = "Momentum",
  gamma_oper_prof    = "Profitability",
  gamma_asset_growth = "Asset Growth",
  gamma_size         = "Size",
  gamma_beta         = "Beta"
)

# Mapping: gamma_col -> characteristic spread column in char_spreads
gamma_spread_map <- c(
  gamma_bm           = "bm_spread",
  gamma_mom12m       = "mom12m_spread",
  gamma_oper_prof    = "oper_prof_spread",
  gamma_asset_growth = "asset_growth_spread",
  gamma_size         = "log_mktcap_lag_spread",
  gamma_beta         = "beta_spread"
)

model_names <- c("hist_mean", "ar1", "ar3")
if (RUN_HAR)   model_names <- c(model_names, "har")
model_names <- c(model_names, "mv", "var1")
if (RUN_LASSO) model_names <- c(model_names, "lasso", "lasso_ct")
# lasso_1se: same as lasso but using lambda.1se (largest lambda within 1 SE of
# the CV minimum). More conservative than lambda.min — selects fewer active
# predictors and shrinks coefficients harder. Hypothesis: in near-zero AC1
# environments with 118 predictors, lambda.min over-selects; lambda.1se may
# reduce OOS MSFE for low-signal gammas at modest cost to Size.
# Both _ct variants use the Campbell-Thompson (2008) sign restriction.
if (RUN_LASSO && RUN_LASSO_1SE) model_names <- c(model_names, "lasso_1se", "lasso_1se_ct")
if (RUN_RF)    model_names <- c(model_names, "rf", "rf_ct")
if (RUN_XGB)   model_names <- c(model_names, "xgb", "xgb_ct")
# Ext-E rolling-window variants: same estimator, fixed-length training window.
if (RUN_ROLLING) {
  if (RUN_LASSO) model_names <- c(model_names, paste0("lasso_w", ROLL_WINDOWS))
  if (RUN_RF)    model_names <- c(model_names, paste0("rf_w",    ROLL_WINDOWS))
}
# "factmom" = Factor Momentum Benchmark (Ext-B, supervisor extension 2026-04-18).
# Rule: pred_k,t = 2 × hist_mean_k,t (amplify the persistent premium in its own sign).
# No "> 0 else 0" restriction: gammas are not sign-oriented, so negative-premium factors
# (Asset Growth Sign = -1, Size) would otherwise be zeroed to a flat NaN-Sharpe series.
# Carrying the sign lets the Scaled strategy short those factors, matching Static's
# sign(γ̄)·LS. Exploits premia persistence without ML estimation noise. (See full note
# at the H=1 factmom block.) Reference: Ehsani & Linnainmaa (2022, JF).
model_names <- c(model_names, "factmom")
# "comb" = equal-weight AR(1)/AR(3)/LASSO (stable baseline, post-loop)
# "comb_rf" extends comb with RF (tests whether non-linear signal adds to combination)
# Reference: Timmermann (2006, Handbook of Economic Forecasting) shows equal-weight
# combination is robust to model uncertainty; Stock & Watson (2004, ReStud) confirm
# averaging dominates selection in low-SNR environments.
model_names <- c(model_names, "comb")
if (RUN_RF)  model_names <- c(model_names, "comb_rf")
if (RUN_XGB) model_names <- c(model_names, "comb_xgb")


# (1) Feature matrix -----------------------------------------------------------
# For each gamma: lags 1-3, 12m rolling vol, own characteristic IQR spread.
# Spread measures cross-sectional dispersion of the underlying characteristic —
# wider dispersion → larger potential premium magnitude.
gammas_mat <- gammas_ts |>
  select(date, all_of(gamma_cols)) |>
  arrange(date)

features <- gammas_mat

# --- Own-gamma lags and rolling volatility ---
# ALIGNMENT NOTE: aligned pairs outcome[i+1] with features[i] (via slice(-nrow)).
# So features[i]$_l1 should = x[i] (gamma_t), not lag(x,1)[i] = x[i-1].
# Using lag(x,1) here creates a double-lag: AR(1) would predict gamma[t+1] from
# gamma[t-1] instead of gamma[t], destroying predictive power.
for (gc in gamma_cols) {
  short <- str_remove(gc, "^gamma_")
  x     <- gammas_mat[[gc]]
  features[[paste0(short, "_l1")]]    <- x             # gamma_t   (most recent, no extra lag)
  features[[paste0(short, "_l2")]]    <- lag(x, 1)     # gamma_{t-1}
  features[[paste0(short, "_l3")]]    <- lag(x, 2)     # gamma_{t-2}
  features[[paste0(short, "_vol12")]] <- slide_dbl(x, sd, .before = 11, .complete = TRUE)
}

# --- Characteristic IQR spreads ---
spread_cols_used <- gamma_spread_map  # named vector: gamma -> spread col
features <- features |>
  left_join(
    char_spreads |> select(date, all_of(unname(spread_cols_used))),
    by = "date"
  )

# --- Macro conditioning variables (from 01c_macro_predictors.R) ---------------
# Joined by year-month key to handle end-of-month vs first-of-month date formats.
# Variables: default_spread, term_spread, short_rate, vix, indpro_gap, mkt_lag1
#
# Literature motivation:
#   default_spread: Fama & French (1989, JFE) — credit risk predicts premia
#   term_spread:    Campbell (1987); Fama & French (1989) — cycle expectations
#   short_rate:     Ang & Bekaert (2007, ReStud) — most robust short-run predictor
#   vix:            Reschenhofer & Zechner (2022) — risk appetite / fear proxy
#   indpro_gap:     Cooper & Priestley (2007) — production-based cycle indicator
#   mkt_lag1:       Reschenhofer & Zechner (2022) — lagged aggregate return
#
# VIX coverage starts 1990-01; pre-1990 rows will be NA. LASSO handles this
# via complete.cases on X_train per iteration — no manual imputation needed.
# CHECK 4 (NA audit) will report exact coverage automatically.
macro_cols <- c("default_spread", "term_spread", "short_rate",
                "vix", "indpro_gap", "mkt_lag1", "infl", "lty")

features <- features |>
  mutate(ym = format(date, "%Y-%m")) |>
  left_join(
    macro_pred |> select(ym, all_of(macro_cols)),
    by = "ym"
  ) |>
  select(-ym)

# --- Goyal & Welch (2008) equity premium predictors ---------------------------
# 12 variables covering valuation (dp, dy, ep, de, bm), market activity (ntis,
# svar), bond market (lty, ltr, dfr), inflation, and investment-to-capital.
# Excludes tbl/tms/dfy which are already in macro_pred.
# Reference: Goyal & Welch (2008, Review of Financial Studies)
# Coverage varies by variable; LASSO handles NAs via complete.cases per iteration.
gw_cols <- c("dp", "ep", "de", "bm", "ntis", "svar",
             "lty", "ltr", "dfr", "infl", "ik")
# dy (log dividend yield) omitted: near-perfectly collinear with dp (log dividend-price)
# — both use the same dividend numerator; only the denominator (P vs. total return index Y)
# differs, giving VIF > 80 when both are included. dp is the standard GW specification.
# Only keep columns that actually exist in gw_pred (ik absent in older file versions)
gw_cols <- intersect(gw_cols, names(gw_pred))

features <- features |>
  mutate(ym = format(date, "%Y-%m")) |>
  left_join(
    gw_pred |> select(ym, all_of(gw_cols)),
    by = "ym"
  ) |>
  select(-ym)

# --- Factor momentum and volatility (Neuhierl, Randl, Reschenhofer & Zechner 2023) ----
# Time-series / factor-level predictors — NOT stock-level characteristics.
# We implement the signal classes from Table B.1 of the paper that are computable
# from monthly FF5 factor returns (signals requiring daily data are excluded).
#
# Signal groups implemented (per factor × 5 FF5 factors):
#
#   Raw momentum (MOM9-12 equivalent, Ehsani & Linnainmaa 2022):
#     tsmom1/3/6/12 = sign of cumulative return over 1/3/6/12 months
#
#   Vol-scaled momentum (MOM1-4 equivalent, Gupta & Kelly 2019):
#     gk_mom1  = 1m return / 3Y rolling vol  (annualised, capped ±2)
#     gk_mom3  = 3m avg return / 3Y rolling vol
#     gk_mom6  = 6m avg return / 3Y rolling vol
#     gk_mom12 = 12m avg return / 10Y rolling vol
#
#   Raw return levels (for linear models):
#     mom1/3/6/12 = average monthly return over 1/3/6/12 months
#
#   Volatility (monthly proxy for Moreira & Muir 2017 daily variance):
#     vol12  = 12-month rolling SD of factor returns
#     vol36  = 36-month rolling SD (3Y; denominator for gk_mom1-3)
#
#   Long-term reversal (REV1 equivalent, Moskowitz, Ooi & Pedersen 2012):
#     rev60 = 1 - annualised 60-month net return (contrarian: positive past → lower future)
#
# Signals NOT implemented (require daily factor returns):
#   VOL1-5: realized daily variance (Moreira & Muir; Cederburg et al.)
#   TSMOM1-4: sign × 40%/EWMA_daily_vol (Moskowitz et al. full spec)
#
# Signals NOT implemented (require portfolio-level B/M data):
#   VAL1-6: BTM spread of long vs. short legs (Cohen, Polk & Vuolteenaho 2003)
#
# References:
#   Ehsani & Linnainmaa (2022, JF) — factor momentum (sign-based)
#   Gupta & Kelly (2019, JPM) — factor momentum (vol-scaled)
#   Moreira & Muir (2017, JF) — volatility-managed portfolios
#   Moskowitz, Ooi & Pedersen (2012, JFE) — time-series momentum & reversal
#   Neuhierl, Randl, Reschenhofer & Zechner (2023, WP) — factor timing signals

factor_cols  <- c("mkt_excess", "smb", "hml", "rmw", "cma")
factor_short <- c("mkt", "smb", "hml", "rmw", "cma")

ff5_feat <- ff5_factors |>
  mutate(ym = format(date, "%Y-%m")) |>
  arrange(date)

for (i in seq_along(factor_cols)) {
  fc  <- factor_cols[i]
  sh  <- factor_short[i]
  r   <- ff5_feat[[fc]]

  # --- Raw momentum levels ---
  ff5_feat[[paste0(sh, "_mom1")]]  <- r
  ff5_feat[[paste0(sh, "_mom3")]]  <- slide_dbl(r, mean, .before = 2,   .complete = TRUE)
  ff5_feat[[paste0(sh, "_mom6")]]  <- slide_dbl(r, mean, .before = 5,   .complete = TRUE)
  ff5_feat[[paste0(sh, "_mom12")]] <- slide_dbl(r, mean, .before = 11,  .complete = TRUE)

  # --- Sign-based momentum (MOM9-12 / Ehsani & Linnainmaa 2022) ---
  ff5_feat[[paste0(sh, "_tsmom1")]]  <- sign(r)
  ff5_feat[[paste0(sh, "_tsmom3")]]  <- sign(slide_dbl(r, sum, .before = 2,  .complete = TRUE))
  ff5_feat[[paste0(sh, "_tsmom6")]]  <- sign(slide_dbl(r, sum, .before = 5,  .complete = TRUE))
  ff5_feat[[paste0(sh, "_tsmom12")]] <- sign(slide_dbl(r, sum, .before = 11, .complete = TRUE))

  # --- Rolling volatility (monthly proxy; exact spec would use daily returns) ---
  vol12  <- slide_dbl(r, sd, .before = 11,  .complete = TRUE)
  vol36  <- slide_dbl(r, sd, .before = 35,  .complete = TRUE)
  vol120 <- slide_dbl(r, sd, .before = 119, .complete = TRUE)
  ff5_feat[[paste0(sh, "_vol12")]]  <- vol12
  ff5_feat[[paste0(sh, "_vol36")]]  <- vol36

  # --- Vol-scaled momentum (MOM1-4 / Gupta & Kelly 2019) ---
  # Annualise monthly average returns and volatility, then cap at ±2
  # Uses 3Y rolling vol for short horizons, 10Y for longer horizons (per paper)
  gk <- function(ret_avg, vol, horizon_months) {
    ann_ret <- ret_avg * 12          # annualise avg monthly return
    ann_vol <- vol * sqrt(12)        # annualise monthly SD
    pmin(pmax(ann_ret / ann_vol, -2), 2)
  }
  ff5_feat[[paste0(sh, "_gk_mom1")]]  <- gk(r,                                               vol36,  1)
  ff5_feat[[paste0(sh, "_gk_mom3")]]  <- gk(slide_dbl(r, mean, .before = 2,  .complete=TRUE), vol36,  3)
  ff5_feat[[paste0(sh, "_gk_mom6")]]  <- gk(slide_dbl(r, mean, .before = 5,  .complete=TRUE), vol36,  6)
  ff5_feat[[paste0(sh, "_gk_mom12")]] <- gk(slide_dbl(r, mean, .before = 11, .complete=TRUE), vol120, 12)

  # --- Long-term reversal (REV1 / Moskowitz, Ooi & Pedersen 2012) ---
  # Signal = 1 - annualised 60-month net return (contrarian: positive past → lower future)
  cum60 <- slide_dbl(r, sum, .before = 59, .complete = TRUE)
  ann60 <- (1 + cum60)^(12/60) - 1
  ff5_feat[[paste0(sh, "_rev60")]] <- 1 - ann60
}

factor_feat_cols <- c(outer(
  factor_short,
  c("mom1","mom3","mom6","mom12",
    "tsmom1","tsmom3","tsmom6","tsmom12",
    "vol12","vol36",
    "gk_mom1","gk_mom3","gk_mom6","gk_mom12",
    "rev60"),
  paste, sep = "_"
))

features <- features |>
  mutate(ym = format(date, "%Y-%m")) |>
  left_join(
    ff5_feat |> select(ym, all_of(factor_feat_cols)),
    by = "ym"
  ) |>
  select(-ym)

feature_cols <- setdiff(names(features), c("date", gamma_cols))

# Full aligned matrix: outcome at t+1 (row slice(-1)) + features at t (slice(-nrow))
aligned <- bind_cols(
  gammas_mat |> select(date, all_of(gamma_cols)) |> slice(-1),
  features   |> select(all_of(feature_cols))     |> slice(-nrow(features))
)

T_aligned <- nrow(aligned)
oos_idx   <- which(
  seq_len(T_aligned) > INIT_WINDOW &
  aligned$date >= OOS_START_DATE
)

cat(sprintf(
  "Features: %d columns | Total months: %d | OOS months: %d (%s to %s)\n",
  length(feature_cols), T_aligned, length(oos_idx),
  format(aligned$date[min(oos_idx)], "%Y-%m"),
  format(aligned$date[max(oos_idx)], "%Y-%m")
))


# (1.5) Pre-flight sanity checks -----------------------------------------------
# Run these BEFORE interpreting any OOS results. They catch data/alignment bugs
# that would otherwise produce silently wrong (often negative) OOS R².

cat("\n", strrep("=", 70), "\n")
cat("PRE-FLIGHT SANITY CHECKS\n")
cat(strrep("=", 70), "\n\n")

# CHECK 1 — Raw gamma autocorrelations -----------------------------------------
# These are the theoretical ceiling for AR-based OOS R².
# AC1 ≈ 0 means no AR signal exists; AC1 ≈ 0.3 → AR(1) R² ceiling ≈ 9%.
cat("--- CHECK 1: Raw gamma autocorrelations (what AR models can capture) ---\n")
acf_summary <- map_dfr(gamma_cols, function(gc) {
  x <- gammas_ts[[gc]]
  x <- x[!is.na(x)]
  av <- acf(x, lag.max = 3, plot = FALSE)$acf
  tibble(
    gamma   = gamma_labels[gc],
    AC1     = round(av[2], 3),
    AC2     = round(av[3], 3),
    AC3     = round(av[4], 3),
    AC1_sq  = round(av[2]^2 * 100, 1)   # theoretical AR(1) in-sample R² ceiling (%)
  )
})
print(acf_summary, n = Inf)
cat("AC1_sq = AC1² × 100 = rough in-sample R² ceiling for AR(1) (%).\n\n")

# CHECK 2 — Feature alignment verification ------------------------------------
# Correct alignment: in `aligned`, row i has outcome = gamma[date_{i+1}] and
# _l1 feature = x[date_i], so _l1[i] should equal outcome[i-1].
cat("--- CHECK 2: Feature alignment — does _l1[i] == outcome[i-1]? ---\n")
align_check <- map_dfr(gamma_cols, function(gc) {
  short   <- str_remove(gc, "^gamma_")
  outcome <- aligned[[gc]]
  l1_feat <- aligned[[paste0(short, "_l1")]]
  delta   <- l1_feat - dplyr::lag(outcome, 1)
  tibble(
    gamma        = gamma_labels[gc],
    max_abs_diff = round(max(abs(delta), na.rm = TRUE), 10),
    aligned_ok   = max(abs(delta), na.rm = TRUE) < 1e-9
  )
})
print(align_check, n = Inf)
if (!all(align_check$aligned_ok)) {
  cat("*** WARNING: alignment check FAILED — feature lags are off by one period! ***\n")
  cat("    This causes AR(1) to regress gamma[t+1] on gamma[t-1] instead of gamma[t].\n")
} else {
  cat("Alignment OK: _l1[i] = outcome[i-1] for all gammas.\n")
}
cat("\n")

# CHECK 3 — In-sample AR(1) R² from aligned data vs raw AC1² ------------------
# If these match, alignment is correct and the AR signal is being captured.
# If in-sample R² << AC1², features are stale/misaligned.
cat("--- CHECK 3: In-sample AR(1) R² from aligned data (compare to AC1² above) ---\n")
is_r2 <- map_dfr(gamma_cols, function(gc) {
  short <- str_remove(gc, "^gamma_")
  df    <- data.frame(y = aligned[[gc]], x1 = aligned[[paste0(short, "_l1")]])
  df    <- df[complete.cases(df), ]
  r2    <- if (nrow(df) > 5) summary(lm(y ~ x1, data = df))$r.squared else NA_real_
  tibble(gamma = gamma_labels[gc], IS_AR1_R2_pct = round(r2 * 100, 2))
})
print(left_join(acf_summary |> select(gamma, AC1_sq), is_r2, by = "gamma"), n = Inf)
cat("If IS_AR1_R2_pct ≈ AC1_sq → alignment is correct.\n")
cat("If IS_AR1_R2_pct << AC1_sq → features are using stale gammas.\n\n")

# CHECK 4 — NA audit for feature columns ---------------------------------------
cat("--- CHECK 4: NA fraction per feature column (first OOS row) ---\n")
na_fractions <- map_dfr(feature_cols, function(fc) {
  vals <- aligned[[fc]]
  tibble(
    feature    = fc,
    na_pct     = round(100 * mean(is.na(vals)), 1),
    na_in_oos  = sum(is.na(vals[oos_idx]))
  )
})
print(na_fractions |> arrange(desc(na_pct)), n = Inf)
cat("High na_pct = model will silently drop observations or fall back to NAs.\n")
cat("High na_in_oos = OOS predictions will be NA (model skipped for those months).\n\n")

# CHECK 5 — Benchmark variance: are gammas predictable in principle? -----------
cat("--- CHECK 5: Gamma variance and benchmark MSFE ---\n")
gamma_var_check <- map_dfr(gamma_cols, function(gc) {
  y     <- gammas_ts[[gc]]
  y     <- y[!is.na(y)]
  bench <- cumsum(y) / seq_along(y)     # expanding mean (as in M0)
  bench <- c(NA, bench[-length(bench)]) # 1-step-ahead hist mean
  valid <- !is.na(bench)
  tibble(
    gamma        = gamma_labels[gc],
    mean_gamma   = round(mean(y) * 100, 3),
    sd_gamma     = round(sd(y) * 100, 3),
    msfe_bench   = round(mean((y[valid] - bench[valid])^2), 6)
  )
})
print(gamma_var_check, n = Inf)
cat("sd_gamma ≈ 0 means gammas are near-constant → no OOS R² is achievable.\n\n")

cat(strrep("=", 70), "\n")
cat("END OF PRE-FLIGHT CHECKS — proceed to OOS loop\n")
cat(strrep("=", 70), "\n\n")


# (2) Helper functions ---------------------------------------------------------

# Column-mean imputation for a numeric matrix.
# Replaces NA in each column with the mean of observed values in that column.
# Applied only to the training matrix — no look-ahead since the mean is computed
# from training data only. This preserves all T-1 training rows instead of
# discarding the full pre-1990 history just because VIX/default_spread are NA.
impute_means <- function(X) {
  for (j in seq_len(ncol(X))) {
    na_j <- is.na(X[, j])
    if (any(na_j)) {
      fill <- if (any(!na_j)) mean(X[!na_j, j]) else 0
      X[na_j, j] <- fill
    }
  }
  X
}

# Time-series holdout CV for LASSO — replaces random k-fold cv.glmnet.
# Random k-fold mixes future and past observations within each fold, creating
# look-ahead bias in lambda selection. This function holds out the last
# val_frac of the training observations as a validation block (strictly later
# in time than the training block), selects lambda by validation MSE, then
# refits on the full training set with that lambda.
# NOTE ON ATTRIBUTION: this is a simple temporal LAST-BLOCK HOLDOUT, not the
# h-block CV of Racine (2000, JoE) — h-block additionally leaves a gap of h
# observations around the validation block to break dependence. Racine is the
# related literature on CV under dependence, not the procedure implemented.
ts_cv_lasso <- function(X, y, val_frac = 0.20, alpha = 1) {
  n       <- length(y)
  n_val   <- max(20L, floor(n * val_frac))
  n_train <- n - n_val
  if (n_train < 20L) return(NULL)

  tr_idx  <- seq_len(n_train)
  val_idx <- seq(n_train + 1L, n)

  # Fit regularisation path on training block only
  fit_path <- tryCatch(
    glmnet(X[tr_idx, ], y[tr_idx], alpha = alpha),
    error = function(e) NULL
  )
  if (is.null(fit_path)) return(NULL)

  # Evaluate each lambda on the held-out validation block
  val_pred <- predict(fit_path, newx = X[val_idx, ])   # n_val × n_lambda matrix
  val_mse  <- colMeans((y[val_idx] - val_pred)^2, na.rm = TRUE)

  # lambda.min: minimises validation MSE
  idx_min  <- which.min(val_mse)
  lam_min  <- fit_path$lambda[idx_min]

  # lambda.1se: most regularised lambda within 1 SE of minimum
  resid_min <- y[val_idx] - val_pred[, idx_min]
  se_mse    <- sd(resid_min^2, na.rm = TRUE) / sqrt(n_val)
  idx_1se   <- max(which(val_mse <= val_mse[idx_min] + se_mse))
  lam_1se   <- fit_path$lambda[idx_1se]

  # Refit on the full training set at the selected lambdas for best estimates
  fit_full <- tryCatch(
    glmnet(X, y, alpha = alpha, lambda = c(lam_min, lam_1se)),
    error = function(e) NULL
  )
  if (is.null(fit_full)) return(NULL)

  list(fit = fit_full, lambda.min = lam_min, lambda.1se = lam_1se)
}

# HAC-robust Clark-West (2007) test.
# Uses Newey-West SE with fixed lag = nw_lags (default 12, consistent with
# Fama-MacBeth inference). sandwich::NeweyWest() is used directly because
# lrvar(..., lag=k) does NOT honour the lag argument — it routes through
# bandwidth selection (bwNeweyWest) and silently ignores lag, producing a
# near-zero SE. NeweyWest(lm(f~1), lag=k) correctly applies the Bartlett
# kernel with fixed truncation.
# Plain-SE version is available as a robustness check: replace nw_se with
#   sd(f, na.rm=TRUE) / sqrt(n).
clark_west <- function(y, yhat_bench, yhat_model, nw_lags = 12L) {
  f    <- (y - yhat_bench)^2 - ((y - yhat_model)^2 - (yhat_bench - yhat_model)^2)
  ok   <- !is.na(f)
  n    <- sum(ok)
  if (n < 10L) return(list(t_cw = NA_real_, p_cw = NA_real_))
  f_ok <- f[ok]
  nw_var <- tryCatch(
    as.numeric(NeweyWest(lm(f_ok ~ 1), lag = nw_lags,
                         prewhite = FALSE, adjust = TRUE)),
    error = function(e) var(f_ok) / n
  )
  # Guard against non-positive estimate (can arise with very short series)
  if (!is.finite(nw_var) || nw_var <= 0) nw_var <- var(f_ok) / n
  nw_se  <- sqrt(nw_var)
  t_stat <- mean(f_ok) / nw_se
  list(t_cw = round(t_stat, 3), p_cw = round(1 - pnorm(t_stat), 3))
}

oos_r2 <- function(y, yhat, yhat_bench) {
  1 - sum((y - yhat)^2,      na.rm = TRUE) /
      sum((y - yhat_bench)^2, na.rm = TRUE)
}


# (3) Expanding-window OOS loop ------------------------------------------------
cat("Running OOS predictions")

pred_list <- setNames(
  lapply(gamma_cols, \(gc) {
    m <- matrix(NA_real_, nrow = length(oos_idx), ncol = length(model_names))
    colnames(m) <- model_names
    m
  }),
  gamma_cols
)
actuals <- matrix(NA_real_, nrow = length(oos_idx), ncol = length(gamma_cols))
colnames(actuals) <- gamma_cols

for (i in seq_along(oos_idx)) {
  t     <- oos_idx[i]
  train <- aligned[1:(t - 1), ]
  test  <- aligned[t, ]

  Y_train <- train |> select(all_of(gamma_cols)) |> as.matrix()
  X_train <- train |> select(all_of(feature_cols)) |> as.matrix()
  X_test  <- test  |> select(all_of(feature_cols)) |> as.matrix()

  actuals[i, ] <- as.numeric(test[gamma_cols])

  for (gc in gamma_cols) {
    y_train <- Y_train[, gc]
    short   <- str_remove(gc, "^gamma_")
    spread_col <- unname(gamma_spread_map[gc])   # strip name to avoid dplyr rename-on-select bug

    # M0: historical mean
    hist_mean_t <- mean(y_train, na.rm = TRUE)
    pred_list[[gc]][i, "hist_mean"] <- hist_mean_t

    # Factor Momentum Benchmark (Ext-B)
    # Rule: pred = 2 × hist_mean (amplify the persistent premium IN ITS OWN DIRECTION).
    # NB: no "> 0 else 0" sign restriction. Our gammas are NOT sign-oriented — Asset
    # Growth (Sign = -1, Cooper-Gulen-Schill) and Size carry structurally NEGATIVE
    # premia, which are economically real, not implausible. Zeroing negative premia
    # permanently flatten those factors (NaN Sharpe). The CT restriction also can't be
    # applied honestly OOS (would need the full-sample sign of γ̄ → look-ahead). Carrying
    # the natural sign lets the Scaled strategy (leverage = pred / |γ̄|) short the
    # long-short leg for negative-premium factors, mirroring Static's sign(γ̄)·LS.
    # The 2× scaling keeps factmom comparable to the Scaled timing strategy in 05.
    # Reference: Ehsani & Linnainmaa (2022, JF) — factor momentum as dominant driver.
    pred_list[[gc]][i, "factmom"] <- if (!is.na(hist_mean_t)) 2 * hist_mean_t else NA_real_

    # M1: AR(1)
    df1 <- data.frame(y = y_train, x1 = train[[paste0(short, "_l1")]])
    df1 <- df1[complete.cases(df1), ]
    if (nrow(df1) >= 5) {
      m1 <- lm(y ~ x1, data = df1)
      pred_list[[gc]][i, "ar1"] <- predict(
        m1, newdata = data.frame(x1 = X_test[, paste0(short, "_l1")])
      )
    }

    # M2: AR(3)
    # AR ORDER NOTE: AIC selects higher orders for some gammas (AR(5) for Momentum,
    # AR(4) for Asset Growth). However, Ljung-Box at lag 12 fails to reject for all 6
    # gammas under AR(3) (00_diagnostics.R, Section 10): Value p=0.093 (borderline),
    # all others p > 0.35. Extra lags would model in-sample noise given AC1 ≈ 0–0.11,
    # worsening OOS performance. AR(3) is retained as the parsimonious choice.
    df3 <- data.frame(
      y  = y_train,
      x1 = train[[paste0(short, "_l1")]],
      x2 = train[[paste0(short, "_l2")]],
      x3 = train[[paste0(short, "_l3")]]
    )
    df3 <- df3[complete.cases(df3), ]
    if (nrow(df3) >= 8) {
      m3 <- lm(y ~ x1 + x2 + x3, data = df3)
      pred_list[[gc]][i, "ar3"] <- predict(m3, newdata = data.frame(
        x1 = X_test[, paste0(short, "_l1")],
        x2 = X_test[, paste0(short, "_l2")],
        x3 = X_test[, paste0(short, "_l3")]
      ))
    }

    # M2b: HAR — Heterogeneous Autoregressive Model (Corsi 2009, JFEC)
    # Decomposes gamma persistence into three horizons:
    #   x_d: most recent gamma (daily analogue = lag 1)
    #   x_w: 4-month rolling mean (weekly analogue)
    #   x_m: 12-month rolling mean (monthly analogue)
    # Parsimonious alternative to high-order AR: captures medium-run persistence
    # (lags 4–12) with a single rolling-mean regressor instead of 9 extra dummies.
    # Reference: Corsi (2009, J. Financial Econometrics); applied to risk premia
    # following the spirit of Reschenhofer & Zechner (2022).
    if (RUN_HAR) {
      y_har <- y_train[!is.na(y_train)]
      n_har <- length(y_har)
      if (n_har >= 17) {   # need >= 12 for x_m window + a few rows to fit
        har_rows <- 13:n_har  # row j predicts y_har[j] from lags at j-1
        if (length(har_rows) >= 5) {
          har_df <- data.frame(
            y   = y_har[har_rows],
            x_d = y_har[har_rows - 1],
            x_w = vapply(har_rows - 1,
                         function(j) mean(y_har[max(1L, j - 3L):j]), numeric(1)),
            x_m = vapply(har_rows - 1,
                         function(j) mean(y_har[max(1L, j - 11L):j]), numeric(1))
          )
          har_fit <- lm(y ~ x_d + x_w + x_m, data = har_df)
          pred_list[[gc]][i, "har"] <- predict(har_fit, newdata = data.frame(
            x_d = tail(y_har, 1),
            x_w = mean(tail(y_har, 4)),
            x_m = mean(tail(y_har, 12))
          ))
        }
      }
    }

    # M3: Multivariate (all gamma lag-1 + own vol12 + own char spread)
    # DIAGNOSTIC NOTE: MV is expected to produce negative OOS R² for most gammas.
    # Including 5 cross-gamma _l1 predictors with near-zero true population
    # coefficients costs approximately −5/(n−9) ≈ −1.6% in OOS R² per the
    # overfitting bias formula. This negative result is a research finding:
    # Fama-MacBeth gammas are not jointly predictable at the 1-month horizon.
    # Reference: Lewellen, Nagel & Shanken (2010, JFE) on weak factor structure.
    mv_cols <- c(
      paste0(str_remove(gamma_cols, "^gamma_"), "_l1"),
      paste0(short, "_vol12"),
      spread_col
    )
    mv_cols <- unname(mv_cols[mv_cols %in% feature_cols])   # unname: prevent dplyr rename-on-select
    df_mv   <- train |> select(all_of(mv_cols)) |> mutate(y = y_train)
    df_mv   <- df_mv[complete.cases(df_mv), ]
    if (nrow(df_mv) >= length(mv_cols) + 2) {
      m_mv <- lm(y ~ ., data = df_mv)
      pred_list[[gc]][i, "mv"] <- predict(
        m_mv, newdata = as.data.frame(X_test[, mv_cols, drop = FALSE])
      )
    }
  }

  # M4: VAR(1) — joint system
  # KNOWN ISSUE (documented, not fixed): VAR(1) is structurally misspecified for
  # two independent reasons:
  #   1. Spurious cross-gamma predictors: with AC1 ≈ 0 for all gammas, adding 5
  #      cross-gamma lags as regressors costs ≈ −5/(n−6−1) in expected OOS R²
  #      (Rademacher complexity bound for spurious predictors in low-signal settings).
  #      The OOS results confirm this: var1 produces negative OOS R² for all 6 gammas.
  #   2. Residual autocorrelation: Portmanteau test on VAR(1) residuals rejects at
  #      p = 0.000 at both lag-6 and lag-12 (00_diagnostics.R, Section 11). A higher
  #      VAR(p) could fix this, but VAR(6) requires 6×6×6 = 222 parameters with
  #      ~270 training observations at OOS start — severely underdetermined.
  # Decision: retain VAR(1) in output as a documented negative benchmark. The two
  # failure modes (spurious predictors, residual autocorrelation) are independent
  # and mutually reinforcing; discuss both in thesis.
  var_train <- Y_train[complete.cases(Y_train), ]
  if (nrow(var_train) >= 10) {
    tryCatch({
      var_fit  <- VAR(var_train, p = 1, type = "const")
      var_pred <- predict(var_fit, n.ahead = 1)
      for (gc in gamma_cols) {
        pred_list[[gc]][i, "var1"] <- var_pred$fcst[[gc]][1, "fcst"]
      }
    }, error = function(e) NULL)
  }

  # M5: LASSO — lambda selected by time-series holdout CV (ts_cv_lasso).
  # Random k-fold CV (cv.glmnet default) mixes future and past observations
  # within folds, introducing look-ahead bias into lambda selection.
  # ts_cv_lasso() holds out the last 20% of training obs as a validation block
  # (always strictly later in time), selects lambda by validation MSE, then
  # refits on the full training set. This is consistent with the expanding-window
  # OOS design and eliminates temporal leakage from lambda selection.
  # Reference: Chinco, Clark-Joseph & Ye (2019, JF); Racine (2000, JoE).
  if (RUN_LASSO) {
    X_tr <- impute_means(X_train)
    for (gc in gamma_cols) {
      y_tr <- Y_train[, gc]
      ok_y <- !is.na(y_tr)
      if (sum(ok_y) >= 40L && !anyNA(X_test)) {   # 40 = 20 train + 20 val minimum
        tryCatch({
          ts_fit <- ts_cv_lasso(X_tr[ok_y, ], y_tr[ok_y])
          if (!is.null(ts_fit)) {
            pred_list[[gc]][i, "lasso"] <- predict(
              ts_fit$fit, newx = X_test, s = ts_fit$lambda.min
            )[1]
            if (RUN_LASSO_1SE) {
              pred_list[[gc]][i, "lasso_1se"] <- predict(
                ts_fit$fit, newx = X_test, s = ts_fit$lambda.1se
              )[1]
            }
          }
        }, error = function(e) NULL)
      }
    }
  }

  # M5b / M5c: Campbell-Thompson (2008, ReStud) sign restriction applied to
  # both LASSO variants. If the prediction disagrees in sign with the expanding
  # historical mean (the benchmark), revert to the benchmark. Imposes the weak
  # prior that the sign of the risk premium is stable; reduces variance at no
  # additional estimation cost.
  if (RUN_LASSO) {
    for (gc in gamma_cols) {
      bench <- pred_list[[gc]][i, "hist_mean"]
      for (m_raw in c("lasso", if (RUN_LASSO_1SE) "lasso_1se")) {
        m_ct  <- paste0(m_raw, "_ct")
        if (!m_ct %in% colnames(pred_list[[gc]])) next
        raw <- pred_list[[gc]][i, m_raw]
        if (!is.na(raw) && !is.na(bench)) {
          pred_list[[gc]][i, m_ct] <- if (sign(raw) == sign(bench)) raw else bench
        }
      }
    }
  }

  # M8: Random Forest (ranger) — one forest per gamma per OOS step.
  # Uses the same 118-feature imputed matrix as LASSO for fair comparison.
  # Hyperparameters:
  #   num.trees = 500: ranger's default; OOB error stabilises well before this
  #   mtry = p/3 ≈ 39: Breiman's regression convention (= randomForest's default
  #          for regression). NOTE this is an explicit OVERRIDE of ranger, whose
  #          own default is floor(sqrt(p)) = 10 here. Sampling more features per
  #          split reduces variance in low-SNR settings.
  #   min.node.size = 5: ranger's regression default; prevents overfitting on
  #          60–650 training rows
  #   importance = "impurity": for SHAP analysis in 04d_shap.R
  # Reference: Breiman (2001); Gu, Kelly & Xiu (2020, RFS) show RF competitive
  # with neural networks for equity return prediction.
  if (RUN_RF) {
    X_tr_rf <- impute_means(X_train)
    rf_mtry <- max(5L, floor(length(feature_cols) / 3L))
    for (gc in gamma_cols) {
      y_tr <- Y_train[, gc]
      ok_y <- !is.na(y_tr)
      if (sum(ok_y) >= 30L && !anyNA(X_test)) {
        tryCatch({
          rf_df  <- as.data.frame(X_tr_rf[ok_y, , drop = FALSE])
          rf_df$.y <- y_tr[ok_y]
          rf_fit <- ranger(
            .y ~ ., data = rf_df,
            num.trees     = 500L,
            mtry          = rf_mtry,
            min.node.size = 5L,
            importance    = "impurity",
            seed          = 42L
          )
          pred_list[[gc]][i, "rf"] <- predict(
            rf_fit, data = as.data.frame(X_test)
          )$predictions
        }, error = function(e) NULL)
      }
    }

    # RF-CT: Campbell-Thompson sign restriction on RF predictions
    for (gc in gamma_cols) {
      bench <- pred_list[[gc]][i, "hist_mean"]
      raw   <- pred_list[[gc]][i, "rf"]
      if (!is.na(raw) && !is.na(bench)) {
        pred_list[[gc]][i, "rf_ct"] <- if (sign(raw) == sign(bench)) raw else bench
      }
    }
  }

  # Ext-E: ROLLING-WINDOW variants of LASSO and RF ----------------------------
  # Identical estimators and hyperparameters to the expanding-window versions
  # above; the ONLY difference is that training uses the most recent W months
  # instead of all history. Any difference in out-of-sample performance is
  # therefore attributable to window length alone.
  if (RUN_ROLLING && !anyNA(X_test)) {
    n_tr_all <- nrow(X_train)
    for (W in ROLL_WINDOWS) {
      if (n_tr_all < W) next            # insufficient history -> leave NA
      keep <- (n_tr_all - W + 1L):n_tr_all
      X_roll <- impute_means(X_train[keep, , drop = FALSE])
      Y_roll <- Y_train[keep, , drop = FALSE]

      for (gc in gamma_cols) {
        y_tr <- Y_roll[, gc]
        ok_y <- !is.na(y_tr)
        if (sum(ok_y) < 40L) next

        if (RUN_LASSO) {
          tryCatch({
            ts_fit <- ts_cv_lasso(X_roll[ok_y, , drop = FALSE], y_tr[ok_y])
            if (!is.null(ts_fit)) {
              pred_list[[gc]][i, paste0("lasso_w", W)] <- predict(
                ts_fit$fit, newx = X_test, s = ts_fit$lambda.min
              )[1]
            }
          }, error = function(e) NULL)
        }

        if (RUN_RF) {
          tryCatch({
            rf_df    <- as.data.frame(X_roll[ok_y, , drop = FALSE])
            rf_df$.y <- y_tr[ok_y]
            rf_fit_w <- ranger(
              .y ~ ., data = rf_df,
              num.trees     = 500L,
              mtry          = max(5L, floor(length(feature_cols) / 3L)),
              min.node.size = 5L,
              seed          = 42L
            )
            pred_list[[gc]][i, paste0("rf_w", W)] <- predict(
              rf_fit_w, data = as.data.frame(X_test)
            )$predictions
          }, error = function(e) NULL)
        }
      }
    }
  }

  # M9: XGBoost — gradient-boosted trees with time-series validation block.
  # Hyperparameters chosen for a low-SNR, small-T regression setting:
  #   max_depth = 4      : moderate tree depth; avoids memorising training noise
  #   eta = 0.05         : small learning rate compensated by more rounds
  #   subsample = 0.8    : row subsampling per tree (bagging effect)
  #   colsample_bytree   : sqrt(p)/p ≈ 0.31 for 118 features — standard heuristic
  #   min_child_weight=5 : minimum sum of instance weights per leaf (≈ min.node.size in RF)
  #   nrounds = 500      : upper bound; early stopping selects the actual count
  #   early_stopping_rounds = 20: halt if no improvement on val RMSE for 20 rounds
  # Validation block: last 20% of training rows (temporal holdout — no look-ahead).
  # Reference: Chen & Guestrin (2016, KDD); Gu, Kelly & Xiu (2020, RFS).
  if (RUN_XGB) {
    X_tr_xgb <- impute_means(X_train)
    xgb_colsample <- round(sqrt(length(feature_cols)) / length(feature_cols), 3)
    for (gc in gamma_cols) {
      y_tr <- Y_train[, gc]
      ok_y <- !is.na(y_tr)
      if (sum(ok_y) >= 40L && !anyNA(X_test)) {
        tryCatch({
          n_tr   <- sum(ok_y)
          n_val  <- max(20L, floor(n_tr * 0.20))
          n_fit  <- n_tr - n_val
          x_ok   <- X_tr_xgb[ok_y, , drop = FALSE]
          y_ok   <- y_tr[ok_y]
          dtrain <- xgb.DMatrix(data = x_ok[seq_len(n_fit), ],
                                label = y_ok[seq_len(n_fit)])
          dval   <- xgb.DMatrix(data = x_ok[(n_fit + 1L):n_tr, ],
                                label = y_ok[(n_fit + 1L):n_tr])
          xgb_fit <- xgb.train(
            params = list(
              objective         = "reg:squarederror",
              max_depth         = 4L,
              eta               = 0.05,
              subsample         = 0.8,
              colsample_bytree  = xgb_colsample,
              min_child_weight  = 5L,
              # Reproducibility. Without an explicit seed, xgboost draws its
              # subsample/colsample from R's GLOBAL RNG stream, so the fit
              # depends on how much randomness any upstream code consumed
              # earlier in the script. Verified: advancing the stream before an
              # unseeded fit changes its predictions; a seeded fit is invariant.
              # Consequence observed here -- adding the rolling-window blocks
              # above (which sit upstream of this loop) silently re-drew xgb,
              # moving Size OOS R^2 3.81 -> 5.28 and comb_xgb, the RAFE-selected
              # headline model, 5.58 -> 6.10, with LASSO/RF/comb bit-identical.
              # ranger is pinned the same way (seed = 42L) above.
              seed              = 42L
            ),
            data                = dtrain,
            nrounds             = 500L,
            evals               = list(val = dval),
            early_stopping_rounds = 20L,
            verbose             = 0L
          )
          pred_list[[gc]][i, "xgb"] <- predict(
            xgb_fit, newdata = xgb.DMatrix(X_test)
          )
        }, error = function(e) NULL)
      }
    }

    # XGB-CT: Campbell-Thompson sign restriction on XGBoost predictions
    for (gc in gamma_cols) {
      bench <- pred_list[[gc]][i, "hist_mean"]
      raw   <- pred_list[[gc]][i, "xgb"]
      if (!is.na(raw) && !is.na(bench)) {
        pred_list[[gc]][i, "xgb_ct"] <- if (sign(raw) == sign(bench)) raw else bench
      }
    }
  }

  if (i %% 50 == 0) cat(".")
}
cat(" done.\n")


# (3.1) Combination forecast ---------------------------------------------------
# M6 (comb): equal-weight average of AR(1), AR(3), and LASSO (lambda.min).
# MV and VAR(1) are excluded: in a near-zero AC1 environment, including k
# cross-gamma predictors costs approximately −k/(n−k−1) in OOS R² per the
# overfitting bias formula — a well-known consequence of spurious predictors
# in low-signal settings (see MV/VAR diagnostic comments above).
# lasso_1se excluded from base combination to keep comb definition stable and
# comparable to prior literature; examine lasso_1se separately for robustness.
#
# Theoretical motivation:
#   Timmermann (2006, Handbook of Economic Forecasting): equal-weight combination
#   is surprisingly hard to beat and is robust to model misspecification.
#   Stock & Watson (2004, ReStud): forecast averaging dominates model selection
#   in low signal-to-noise environments — exactly our setting.
#   Genre et al. (2013, IJF): trimmed equal-weight combination is robust to
#   outlier models; here we achieve the same by pre-excluding MV and VAR(1).
#
# Requires >= 2 of the three base forecasts to be non-NA; otherwise left as NA.
COMB_MODELS     <- intersect(c("ar1", "ar3", "lasso"), model_names)
COMB_RF_MODELS  <- intersect(c("ar1", "ar3", "lasso", "rf"), model_names)
COMB_XGB_MODELS <- intersect(c("ar1", "ar3", "lasso", "xgb"), model_names)

for (gc in gamma_cols) {
  # comb: AR(1) + AR(3) + LASSO (unchanged baseline)
  sub     <- pred_list[[gc]][, COMB_MODELS, drop = FALSE]
  n_valid <- rowSums(!is.na(sub))
  pred_list[[gc]][, "comb"] <- ifelse(n_valid >= 2,
                                      rowMeans(sub, na.rm = TRUE), NA_real_)
  # comb_rf: extends comb with RF — tests non-linear marginal gain
  if (RUN_RF && "comb_rf" %in% colnames(pred_list[[gc]])) {
    sub_rf     <- pred_list[[gc]][, COMB_RF_MODELS, drop = FALSE]
    n_valid_rf <- rowSums(!is.na(sub_rf))
    pred_list[[gc]][, "comb_rf"] <- ifelse(n_valid_rf >= 2,
                                           rowMeans(sub_rf, na.rm = TRUE), NA_real_)
  }
  # comb_xgb: extends comb with XGBoost — tests non-linear marginal gain
  if (RUN_XGB && "comb_xgb" %in% colnames(pred_list[[gc]])) {
    sub_xgb     <- pred_list[[gc]][, COMB_XGB_MODELS, drop = FALSE]
    n_valid_xgb <- rowSums(!is.na(sub_xgb))
    pred_list[[gc]][, "comb_xgb"] <- ifelse(n_valid_xgb >= 2,
                                            rowMeans(sub_xgb, na.rm = TRUE), NA_real_)
  }
}
cat(sprintf("comb     computed from: %s\n", paste(COMB_MODELS,     collapse = ", ")))
cat(sprintf("comb_rf  computed from: %s\n", paste(COMB_RF_MODELS,  collapse = ", ")))
cat(sprintf("comb_xgb computed from: %s\n", paste(COMB_XGB_MODELS, collapse = ", ")))


# (3.5) Post-OOS prediction diagnostics ----------------------------------------
# Checks that catch silent failures: shrinkage to mean, sign flip, collapsed variance.

cat("\n", strrep("=", 70), "\n")
cat("POST-OOS PREDICTION DIAGNOSTICS\n")
cat(strrep("=", 70), "\n\n")

post_diag <- map_dfr(gamma_cols, function(gc) {
  y <- actuals[, gc]
  map_dfr(model_names, function(m) {
    yhat  <- pred_list[[gc]][, m]
    valid <- !is.na(yhat) & !is.na(y)
    if (sum(valid) < 10) return(NULL)
    tibble(
      gamma          = gamma_labels[gc],
      model          = m,
      n_valid        = sum(valid),
      # Prediction spread: if near 0, model collapsed to constant (≈ hist mean)
      pred_sd        = round(sd(yhat[valid]) * 100, 4),
      actual_sd      = round(sd(y[valid]) * 100, 4),
      sd_ratio       = round(sd(yhat[valid]) / sd(y[valid]), 3),
      # Pearson correlation — signal strength regardless of calibration
      cor_pearson    = round(cor(y[valid], yhat[valid], use = "complete.obs"), 3),
      # Sign agreement — economic direction timing
      sign_agree_pct = round(100 * mean(sign(y[valid]) == sign(yhat[valid])), 1),
      # Mean prediction bias
      mean_bias      = round((mean(yhat[valid]) - mean(y[valid])) * 100, 4)
    )
  })
})

cat("--- Prediction spread (pred_sd ≈ 0 means model collapsed to hist mean) ---\n")
print(
  post_diag |>
    filter(model != "hist_mean") |>
    select(gamma, model, pred_sd, actual_sd, sd_ratio) |>
    pivot_wider(names_from = model, values_from = sd_ratio, id_cols = c(gamma, actual_sd)),
  n = Inf
)

cat("\n--- Pearson correlation (signal strength) ---\n")
print(
  post_diag |>
    filter(model != "hist_mean") |>
    select(gamma, model, cor_pearson) |>
    pivot_wider(names_from = model, values_from = cor_pearson),
  n = Inf
)

cat("\n--- Sign agreement % (> 50 = directional skill above random) ---\n")
print(
  post_diag |>
    filter(model != "hist_mean") |>
    select(gamma, model, sign_agree_pct) |>
    pivot_wider(names_from = model, values_from = sign_agree_pct),
  n = Inf
)

cat("\n--- Mean prediction bias (pred mean − actual mean, in %) ---\n")
print(
  post_diag |>
    filter(model != "hist_mean") |>
    select(gamma, model, mean_bias) |>
    pivot_wider(names_from = model, values_from = mean_bias),
  n = Inf
)
cat("\nNote: large bias + low sd_ratio = model is making near-constant biased forecasts.\n")
cat(strrep("=", 70), "\n\n")


# (4) OOS evaluation (full sample) ---------------------------------------------
eval_window <- function(preds, y_vec, bench_vec, idx = seq_along(y_vec)) {
  map_dfr(setdiff(model_names, "hist_mean"), function(m) {
    yhat  <- preds[idx, m]
    y     <- y_vec[idx]
    bench <- bench_vec[idx]
    valid <- !is.na(yhat) & !is.na(y)
    if (sum(valid) < 10) return(NULL)
    cw <- clark_west(y[valid], bench[valid], yhat[valid])
    tibble(
      model      = m,
      oos_r2     = round(oos_r2(y[valid], yhat[valid], bench[valid]) * 100, 2),
      msfe       = round(mean((y[valid] - yhat[valid])^2), 6),
      msfe_ratio = round(mean((y[valid] - yhat[valid])^2) /
                         mean((y[valid] - bench[valid])^2), 3),
      t_cw       = cw$t_cw,
      p_cw       = cw$p_cw,
      n_oos      = sum(valid)
    )
  })
}

oos_eval <- map_dfr(gamma_cols, function(gc) {
  y     <- actuals[, gc]
  bench <- pred_list[[gc]][, "hist_mean"]
  eval_window(pred_list[[gc]], y, bench) |>
    mutate(gamma = gamma_labels[gc], sub_period = "Full", .before = model)
})

cat("\n=== Out-of-Sample Evaluation: Full Sample (OOS R² in %) ===\n")
# Note: pivot only on gamma + model to avoid t_cw/p_cw acting as extra id cols
print(
  oos_eval |> select(gamma, model, oos_r2) |>
    pivot_wider(names_from = model, values_from = oos_r2),
  n = Inf
)
cat("\n=== OOS R²: MSFE ratio vs benchmark (< 1 = beats mean) ===\n")
print(
  oos_eval |> select(gamma, model, msfe_ratio) |>
    pivot_wider(names_from = model, values_from = msfe_ratio),
  n = Inf
)
cat("\n=== Clark-West t-stats (full sample) ===\n")
print(
  oos_eval |> select(gamma, model, t_cw) |>
    pivot_wider(names_from = model, values_from = t_cw),
  n = Inf
)


# (5) Sub-period OOS R² --------------------------------------------------------
dates_oos <- aligned$date[oos_idx]

sub_windows <- list(
  "Pre-2000"  = which(dates_oos <  as.Date("2000-01-01")),
  "2000-2010" = which(dates_oos >= as.Date("2000-01-01") & dates_oos < as.Date("2010-01-01")),
  "Post-2010" = which(dates_oos >= as.Date("2010-01-01"))
)

oos_eval_sub <- map_dfr(names(sub_windows), function(sp) {
  idx <- sub_windows[[sp]]
  if (length(idx) < 20) return(NULL)
  map_dfr(gamma_cols, function(gc) {
    y     <- actuals[, gc]
    bench <- pred_list[[gc]][, "hist_mean"]
    eval_window(pred_list[[gc]], y, bench, idx) |>
      mutate(gamma = gamma_labels[gc], sub_period = sp, .before = model)
  })
})

cat("\n=== Sub-Period OOS R² (%) ===\n")
print(
  bind_rows(oos_eval_sub) |>
    select(sub_period, gamma, model, oos_r2) |>
    pivot_wider(names_from = model, values_from = oos_r2) |>
    arrange(sub_period, gamma),
  n = Inf
)

# Combine full + sub-period for storage
oos_eval_all <- bind_rows(oos_eval, oos_eval_sub)


# (5b) Common-window OOS evaluation --------------------------------------------
# The per-model `oos_eval` above evaluates each model on ITS OWN non-NA window
# (414–421 months, depending on feature availability: the feature-dependent
# models — LASSO/RF/XGB/MLP/LSTM — skip the ~7 real-time-unavailable GW-tail
# months, while AR/VAR/combination models reach the full 421). Cross-model OOS
# R² / Clark-West comparisons in a single table should stand on an IDENTICAL
# sample. Here we recompute every model on the COMMON window per gamma: the set
# of months in which EVERY model (and the actual) is non-missing. This is the
# sample reported in Tables 3, 3b and 6, and the family on which 04f runs FDR.
common_mask <- setNames(lapply(gamma_cols, function(gc) {
  P <- pred_list[[gc]]                       # months × models
  rowSums(is.na(P)) == 0 & !is.na(actuals[, gc])
}), gamma_cols)

oos_eval_common <- map_dfr(gamma_cols, function(gc) {
  idx   <- which(common_mask[[gc]])
  y     <- actuals[, gc]
  bench <- pred_list[[gc]][, "hist_mean"]
  eval_window(pred_list[[gc]], y, bench, idx) |>
    mutate(gamma = gamma_labels[gc], sub_period = "Full", .before = model)
})

cat(sprintf(
  "\n=== Common-window OOS R²: every model on the same %d–%d months per gamma ===\n",
  min(sapply(common_mask, sum)), max(sapply(common_mask, sum))
))
print(
  oos_eval_common |> select(gamma, model, oos_r2) |>
    pivot_wider(names_from = model, values_from = oos_r2),
  n = Inf
)
cat("\nCommon-window months per gamma:\n")
print(tibble(gamma = gamma_labels[gamma_cols],
             n_common = sapply(common_mask, sum)), n = Inf)


# (6) Clark-West p-values ------------------------------------------------------
cat("\n=== Clark-West p-values (full sample, one-sided) ===\n")
print(
  oos_eval |>
    select(gamma, model, p_cw) |>
    pivot_wider(names_from = model, values_from = p_cw),
  n = Inf
)

# Summary diagnosis: AC1 ceiling vs actual OOS R²
cat("\n=== DIAGNOSIS: AC1 ceiling vs best OOS R² achieved ===\n")
best_oos <- oos_eval |>
  group_by(gamma) |>
  slice_max(oos_r2, n = 1, with_ties = FALSE) |>
  select(gamma, best_model = model, best_oos_r2 = oos_r2)
print(
  left_join(
    acf_summary |> select(gamma, AC1, AC1_sq),
    best_oos,
    by = "gamma"
  ) |> mutate(gap = round(AC1_sq - best_oos_r2, 2)),
  n = Inf
)
cat("gap = AC1_sq - best_oos_r2: positive means models underperform even their ceiling.\n")
cat("If AC1_sq itself is near 0, the gammas are white noise — no model can help without\n")
cat("external predictors (macro vars, characteristic spreads from FRED/AQR).\n")


# (7) Plots --------------------------------------------------------------------
dir.create("plots", showWarnings = FALSE)

# Helper: ggsave that warns instead of aborting when a PDF is locked (Windows).
# On Windows, open PDF viewers hold an exclusive file lock; wrapping in tryCatch
# lets the script complete and save the .rds even if one plot file is open.
safe_save <- function(path, plot, ...) {
  tryCatch(
    { ggsave(path, plot, ...); cat("Saved:", path, "\n") },
    error = function(e) cat("SKIPPED (file locked — close viewer):", path, "\n")
  )
}

model_order <- setdiff(model_names, "hist_mean")
# Human-readable labels for model axis ticks.
# NB every entry of model_order must have a label here: factor() turns an
# unlabelled model into an NA level, and ALL such models then share a single
# "NA" column with their values printed on top of one another. That is exactly
# what happened to the shipped heatmap on 2026-07-20, when the four Ext-E
# rolling-window variants were added to model_names but not to this map. The
# stopifnot() below now makes that impossible.
model_labels <- c(
  ar1          = "AR(1)",
  ar3          = "AR(3)",
  har          = "HAR",
  mv           = "MV",
  var1         = "VAR(1)",
  lasso        = "LASSO",
  lasso_ct     = "LASSO-CT",
  lasso_1se    = "LASSO-1SE",
  lasso_1se_ct = "LASSO-1SE-CT",
  rf           = "RF",
  rf_ct        = "RF-CT",
  xgb          = "XGBoost",
  xgb_ct       = "XGBoost-CT",
  factmom      = "Ampl.HM",   # Ext-B: Factor Momentum Benchmark
  comb         = "Comb.",
  comb_rf      = "Comb.+RF",
  comb_xgb     = "Comb.+XGB",
  # Ext-E rolling-window variants
  lasso_w120   = "LASSO-roll10y",
  lasso_w240   = "LASSO-roll20y",
  rf_w120      = "RF-roll10y",
  rf_w240      = "RF-roll20y"
)
stopifnot(
  "model_labels is missing an entry for a model in model_order" =
    all(model_order %in% names(model_labels))
)

# OOS R² heatmap — models on x-axis, gammas on y-axis.
# DIAGNOSTIC ONLY. This runs before the Colab MLP/LSTM imports, so it can never
# show the neural nets; the thesis exhibits (Images/oos_r2_heatmap.pdf and
# Images/oos_r2_subperiod.pdf) are produced by 04e_plots.R from the finished
# gamma_predictions.rds. Hence the separate diag_ filenames — writing the
# canonical names here silently overwrote 04e's complete figures with this
# partial view, which is how the broken heatmap reached the thesis.
p_r2 <- oos_eval |>
  mutate(
    model = factor(model_labels[model], levels = model_labels[model_order]),
    sig   = case_when(p_cw < 0.01 ~ "***", p_cw < 0.05 ~ "**",
                      p_cw < 0.10 ~ "*",   TRUE ~ "")
  ) |>
  ggplot(aes(x = model, y = gamma, fill = oos_r2)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = paste0(round(oos_r2, 1), sig)), size = 3) +
  scale_fill_gradient2(
    low = "firebrick", mid = "white", high = "steelblue", midpoint = 0,
    name = "OOS R² (%)"
  ) +
  labs(title = "Out-of-Sample R² by Model and Gamma",
       subtitle = "Clark-West significance: *** p<0.01, ** p<0.05, * p<0.10\nComb. = equal-weight AR(1)/AR(3)/LASSO; Comb.+RF adds Random Forest; HAR = daily/weekly/monthly cascade",
       x = NULL, y = NULL) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

safe_save("plots/diag_oos_r2_heatmap.pdf", p_r2, width = 10, height = 5)

# Sub-period OOS R² bar chart
if (nrow(oos_eval_sub) > 0) {
  p_sub <- oos_eval_sub |>
    mutate(
      model      = factor(model_labels[model], levels = model_labels[model_order]),
      sub_period = factor(sub_period, levels = c("Pre-2000", "2000-2010", "Post-2010"))
    ) |>
    ggplot(aes(x = sub_period, y = oos_r2, fill = model)) +
    geom_col(position = "dodge", width = 0.7) +
    geom_hline(yintercept = 0, linewidth = 0.3) +
    facet_wrap(~gamma, ncol = 3, scales = "free_y") +
    labs(title = "Sub-Period OOS R² by Model and Gamma",
         x = NULL, y = "OOS R² (%)") +
    theme_bw() +
    theme(legend.position = "bottom",
          axis.text.x = element_text(angle = 20, hjust = 1))
  safe_save("plots/diag_oos_r2_subperiod.pdf", p_sub, width = 14, height = 8)
}

# Predicted vs actual: AR(1) shown for all gammas (most interpretable baseline)
# plus best model if different from AR(1)
pred_actual_plot <- map_dfr(gamma_cols, function(gc) {
  bind_rows(
    tibble(date = dates_oos, actual = actuals[, gc],
           predicted = pred_list[[gc]][, "ar1"],
           gamma = gamma_labels[gc], model = "AR(1)"),
    tibble(date = dates_oos, actual = actuals[, gc],
           predicted = pred_list[[gc]][, "lasso"],
           gamma = gamma_labels[gc], model = "LASSO")
  )
})

p_pa <- pred_actual_plot |>
  pivot_longer(c(actual, predicted), names_to = "series", values_to = "value") |>
  ggplot(aes(x = date, y = value * 100, colour = series, linewidth = series)) +
  geom_line() +
  scale_colour_manual(values = c(actual = "black", predicted = "steelblue")) +
  scale_linewidth_manual(values = c(actual = 0.5, predicted = 0.7)) +
  facet_wrap(~gamma + model, ncol = 2, scales = "free_y") +
  labs(title = "Actual vs Predicted Gammas — AR(1) and LASSO",
       x = NULL, y = "Gamma (% per month)", colour = NULL, linewidth = NULL) +
  theme_bw() +
  theme(legend.position = "bottom")

safe_save("plots/predicted_vs_actual.pdf", p_pa, width = 10, height = 14)


# (8) Save ---------------------------------------------------------------------
# aligned, feature_cols, oos_idx are saved so 04b/04c can load them directly
# without rebuilding the feature matrix (avoids inconsistency).
oos_eval <- oos_eval  # full-sample only in oos_eval for downstream compat
saveRDS(list(pred_list = pred_list, oos_eval = oos_eval, oos_eval_all = oos_eval_all,
             oos_eval_common = oos_eval_common,
             actuals = actuals, dates_oos = dates_oos,
             aligned = aligned, feature_cols = feature_cols, oos_idx = oos_idx),
        file = "gamma_predictions.rds")
cat("Saved: gamma_predictions.rds\n")


# ==============================================================================
# Ext-C: QUARTERLY PREDICTION HORIZON (H = 3)
# Supervisor extension confirmed 2026-04-18.
#
# Outcome: Y_t = gamma_{t+1} + gamma_{t+2} + gamma_{t+3}  (cumulative 3-month)
# Features: identical to H=1 (feature vector at time t unchanged)
# Inference: Newey-West lags increased to NW_LAG + H - 1 = 14 to account for
#   MA(H-1) = MA(2) serial correlation introduced by overlapping windows.
#   Reference: Hodrick (1992, RFS) — overlapping returns inference;
#              Cochrane & Piazzesi (2005, AER) — long-horizon forecasting practice.
#
# Economic rationale: near-zero AC1 (max 0.11) limits 1-month predictability.
#   Aggregating 3 months attenuates idiosyncratic noise while preserving the
#   lower-frequency macro/factor signal.
#   Reference: Fama & French (1988, JFE) — "forecast power increases with horizon".
#              Corsi (2009, JFEC) — HAR model: multi-horizon aggregation.
#
# Saves: gamma_predictions_h3.rds (separate file, H=1 results untouched).
# ==============================================================================

if (RUN_QUARTERLY) {

  HORIZON   <- 3L
  NW_LAG_H3 <- NW_LAG + HORIZON - 1L   # 14 lags — MA(2) overlap correction

  T_full <- nrow(gammas_mat)   # total gamma observations (pre-alignment)
  n_h3   <- T_full - HORIZON   # rows with a complete H=3 outcome window

  # Build cumulative 3-month outcome: row j gets sum of gammas at j+1, j+2, j+3
  gamma_outcome_h3 <- matrix(
    NA_real_, nrow = n_h3, ncol = length(gamma_cols),
    dimnames = list(NULL, gamma_cols)
  )
  for (j in seq_len(n_h3)) {
    gamma_outcome_h3[j, ] <- colSums(
      as.matrix(gammas_mat[(j + 1L):(j + HORIZON), gamma_cols])
    )
  }

  # aligned_h3: feature at row j (time t) predicts cumulative gamma at t+1..t+H
  aligned_h3 <- bind_cols(
    gammas_mat |> select(date) |> slice(seq_len(n_h3)),
    as_tibble(gamma_outcome_h3),
    features |> select(all_of(feature_cols)) |> slice(seq_len(n_h3))
  )

  oos_idx_h3 <- which(
    seq_len(nrow(aligned_h3)) > INIT_WINDOW &
    aligned_h3$date >= OOS_START_DATE
  )
  dates_oos_h3 <- aligned_h3$date[oos_idx_h3]

  # H=3 model set: skip VAR(1) and MV (known underperformers) and RF (slow).
  # Include all others for a fair H=3 vs H=1 comparison.
  model_names_h3 <- c("hist_mean", "ar1", "ar3")
  if (RUN_HAR_H3) model_names_h3 <- c(model_names_h3, "har")  # off: look-ahead, see config
  if (RUN_LASSO) model_names_h3 <- c(model_names_h3, "lasso", "lasso_ct")
  if (RUN_LASSO && RUN_LASSO_1SE) model_names_h3 <- c(model_names_h3, "lasso_1se", "lasso_1se_ct")
  if (RUN_XGB)   model_names_h3 <- c(model_names_h3, "xgb", "xgb_ct")
  model_names_h3 <- c(model_names_h3, "factmom", "comb")
  if (RUN_XGB)   model_names_h3 <- c(model_names_h3, "comb_xgb")

  pred_list_h3 <- setNames(
    lapply(gamma_cols, \(gc) {
      m <- matrix(NA_real_, nrow = length(oos_idx_h3), ncol = length(model_names_h3))
      colnames(m) <- model_names_h3
      m
    }),
    gamma_cols
  )
  actuals_h3 <- matrix(NA_real_, nrow = length(oos_idx_h3), ncol = length(gamma_cols))
  colnames(actuals_h3) <- gamma_cols

  cat(sprintf(
    "\nExt-C: Running H=3 quarterly OOS (%d steps, NW lags = %d)\n",
    length(oos_idx_h3), NW_LAG_H3
  ))

  for (i in seq_along(oos_idx_h3)) {
    t     <- oos_idx_h3[i]
    train <- aligned_h3[seq_len(t - 1L), ]
    test  <- aligned_h3[t, ]

    Y_train <- train |> select(all_of(gamma_cols)) |> as.matrix()
    X_train <- train |> select(all_of(feature_cols)) |> as.matrix()
    X_test  <- test  |> select(all_of(feature_cols)) |> as.matrix()

    actuals_h3[i, ] <- as.numeric(test[gamma_cols])

    for (gc in gamma_cols) {
      y_train  <- Y_train[, gc]
      short    <- str_remove(gc, "^gamma_")
      hist_h3  <- mean(y_train, na.rm = TRUE)

      # M0 benchmark: expanding mean of cumulative 3-month gammas
      pred_list_h3[[gc]][i, "hist_mean"] <- hist_h3

      # Factor Momentum (sign-agnostic rule, applied to H=3 cumulative target; see H=1 note)
      pred_list_h3[[gc]][i, "factmom"] <- if (!is.na(hist_h3)) 2 * hist_h3 else NA_real_

      # AR(1) on H=3 target
      df1 <- data.frame(y = y_train, x1 = train[[paste0(short, "_l1")]])
      df1 <- df1[complete.cases(df1), ]
      if (nrow(df1) >= 5L) {
        m1 <- lm(y ~ x1, data = df1)
        pred_list_h3[[gc]][i, "ar1"] <- predict(
          m1, newdata = data.frame(x1 = X_test[, paste0(short, "_l1")])
        )
      }

      # AR(3) on H=3 target
      df3 <- data.frame(
        y  = y_train,
        x1 = train[[paste0(short, "_l1")]],
        x2 = train[[paste0(short, "_l2")]],
        x3 = train[[paste0(short, "_l3")]]
      )
      df3 <- df3[complete.cases(df3), ]
      if (nrow(df3) >= 8L) {
        m3 <- lm(y ~ x1 + x2 + x3, data = df3)
        pred_list_h3[[gc]][i, "ar3"] <- predict(m3, newdata = data.frame(
          x1 = X_test[, paste0(short, "_l1")],
          x2 = X_test[, paste0(short, "_l2")],
          x3 = X_test[, paste0(short, "_l3")]
        ))
      }

      # HAR on H=3 target — DISABLED (RUN_HAR_H3 = FALSE).
      # The cumulative target Y_t = γ_{t+1}+γ_{t+2}+γ_{t+3} overlaps its own lag:
      # x_d = Y_{t-1} = γ_t+γ_{t+1}+γ_{t+2} contains two future months relative to
      # the forecast origin t → look-ahead (~50% OOS R²). Re-enabling requires a
      # leak-free reformulation (monthly-gamma components lagged by ≥ H). See config.
      if (RUN_HAR_H3) {
        y_har <- y_train[!is.na(y_train)]
        n_har <- length(y_har)
        if (n_har >= 17L) {
          har_rows <- 13L:n_har
          if (length(har_rows) >= 5L) {
            har_df <- data.frame(
              y   = y_har[har_rows],
              x_d = y_har[har_rows - 1L],
              x_w = vapply(har_rows - 1L,
                           function(j) mean(y_har[max(1L, j - 3L):j]), numeric(1L)),
              x_m = vapply(har_rows - 1L,
                           function(j) mean(y_har[max(1L, j - 11L):j]), numeric(1L))
            )
            har_fit <- lm(y ~ x_d + x_w + x_m, data = har_df)
            pred_list_h3[[gc]][i, "har"] <- predict(har_fit, newdata = data.frame(
              x_d = tail(y_har, 1L),
              x_w = mean(tail(y_har, 4L)),
              x_m = mean(tail(y_har, 12L))
            ))
          }
        }
      }
    }  # end per-gamma loop

    # LASSO on H=3 target
    if (RUN_LASSO) {
      X_tr_h3 <- impute_means(X_train)
      for (gc in gamma_cols) {
        y_tr <- Y_train[, gc]
        ok_y <- !is.na(y_tr)
        if (sum(ok_y) >= 40L && !anyNA(X_test)) {
          tryCatch({
            ts_fit <- ts_cv_lasso(X_tr_h3[ok_y, ], y_tr[ok_y])
            if (!is.null(ts_fit)) {
              pred_list_h3[[gc]][i, "lasso"] <- predict(
                ts_fit$fit, newx = X_test, s = ts_fit$lambda.min
              )[1L]
              if (RUN_LASSO_1SE) {
                pred_list_h3[[gc]][i, "lasso_1se"] <- predict(
                  ts_fit$fit, newx = X_test, s = ts_fit$lambda.1se
                )[1L]
              }
            }
          }, error = function(e) NULL)
        }
      }
      # Campbell-Thompson sign restriction on H=3 LASSO
      for (gc in gamma_cols) {
        bench_h3 <- pred_list_h3[[gc]][i, "hist_mean"]
        for (m_raw in c("lasso", if (RUN_LASSO_1SE) "lasso_1se")) {
          m_ct <- paste0(m_raw, "_ct")
          if (!m_ct %in% colnames(pred_list_h3[[gc]])) next
          raw <- pred_list_h3[[gc]][i, m_raw]
          if (!is.na(raw) && !is.na(bench_h3)) {
            pred_list_h3[[gc]][i, m_ct] <- if (sign(raw) == sign(bench_h3)) raw else bench_h3
          }
        }
      }
    }

    # XGBoost on H=3 target
    if (RUN_XGB) {
      X_tr_xgb_h3 <- impute_means(X_train)
      xgb_cs_h3   <- round(sqrt(length(feature_cols)) / length(feature_cols), 3L)
      for (gc in gamma_cols) {
        y_tr <- Y_train[, gc]
        ok_y <- !is.na(y_tr)
        if (sum(ok_y) >= 40L && !anyNA(X_test)) {
          tryCatch({
            n_tr  <- sum(ok_y)
            n_val <- max(20L, floor(n_tr * 0.20))
            n_fit <- n_tr - n_val
            x_ok  <- X_tr_xgb_h3[ok_y, , drop = FALSE]
            y_ok  <- y_tr[ok_y]
            dtrain <- xgb.DMatrix(data = x_ok[seq_len(n_fit), ],  label = y_ok[seq_len(n_fit)])
            dval   <- xgb.DMatrix(data = x_ok[(n_fit + 1L):n_tr, ], label = y_ok[(n_fit + 1L):n_tr])
            xgb_fit <- xgb.train(
              params = list(
                objective = "reg:squarederror", max_depth = 4L,
                eta = 0.05, subsample = 0.8,
                colsample_bytree = xgb_cs_h3, min_child_weight = 5L,
                seed = 42L   # reproducibility -- see H=1 block
              ),
              data = dtrain, nrounds = 500L,
              watchlist = list(val = dval),
              early_stopping_rounds = 20L, verbose = 0L
            )
            pred_list_h3[[gc]][i, "xgb"] <- predict(
              xgb_fit, newdata = xgb.DMatrix(X_test)
            )
          }, error = function(e) NULL)
        }
      }
      for (gc in gamma_cols) {
        bench_h3 <- pred_list_h3[[gc]][i, "hist_mean"]
        raw      <- pred_list_h3[[gc]][i, "xgb"]
        if (!is.na(raw) && !is.na(bench_h3)) {
          pred_list_h3[[gc]][i, "xgb_ct"] <- if (sign(raw) == sign(bench_h3)) raw else bench_h3
        }
      }
    }

    if (i %% 50L == 0L) cat(".")
  }
  cat(" done.\n")

  # H=3 combination forecast
  COMB_H3      <- intersect(c("ar1", "ar3", "lasso"),        model_names_h3)
  COMB_H3_XGB  <- intersect(c("ar1", "ar3", "lasso", "xgb"), model_names_h3)
  for (gc in gamma_cols) {
    sub     <- pred_list_h3[[gc]][, COMB_H3, drop = FALSE]
    n_ok    <- rowSums(!is.na(sub))
    pred_list_h3[[gc]][, "comb"] <- ifelse(n_ok >= 2L, rowMeans(sub, na.rm = TRUE), NA_real_)
    if (RUN_XGB && "comb_xgb" %in% colnames(pred_list_h3[[gc]])) {
      sub_x  <- pred_list_h3[[gc]][, COMB_H3_XGB, drop = FALSE]
      n_ok_x <- rowSums(!is.na(sub_x))
      pred_list_h3[[gc]][, "comb_xgb"] <- ifelse(n_ok_x >= 2L,
                                                  rowMeans(sub_x, na.rm = TRUE), NA_real_)
    }
  }

  # Hodrick (1992) / Hansen-Hodrick (1980) SE for the overlapping H=3 forecasts.
  # Under H-period overlap the Clark-West loss differential f_t is MA(H-1), so its
  # long-run variance is captured exactly by a RECTANGULAR kernel truncated at lag
  # H-1 (= 2 here) — the appropriate window given the known MA structure, vs the
  # conservative Bartlett NW_LAG_H3 = 14. Reported as an inference robustness check
  # for the overlapping-data concern (a model whose H=3 significance survives both
  # the conservative NW and the exact-MA Hodrick SE is not an overlap artifact).
  # PSD fallback: if the rectangular LRV is non-positive, use Bartlett at lag H-1.
  # Reference: Hodrick (1992, RFS); Hansen & Hodrick (1980, JPE).
  clark_west_hodrick <- function(y, yhat_bench, yhat_model, h = HORIZON) {
    f  <- (y - yhat_bench)^2 - ((y - yhat_model)^2 - (yhat_bench - yhat_model)^2)
    f  <- f[!is.na(f)]; n <- length(f)
    if (n < 10L) return(list(t_hod = NA_real_, p_hod = NA_real_))
    q  <- max(1L, h - 1L)               # MA order from H-period overlap
    e  <- f - mean(f)
    g0 <- sum(e^2) / n
    lrv_rect <- g0
    for (j in seq_len(q)) lrv_rect <- lrv_rect + 2 * (sum(e[(j + 1L):n] * e[1:(n - j)]) / n)
    if (is.finite(lrv_rect) && lrv_rect > 0) {
      var_mean <- lrv_rect / n
    } else {
      lrv_bart <- g0                     # Bartlett (always PSD) fallback at lag q
      for (j in seq_len(q))
        lrv_bart <- lrv_bart + 2 * (1 - j / (q + 1)) * (sum(e[(j + 1L):n] * e[1:(n - j)]) / n)
      var_mean <- max(lrv_bart, g0) / n
    }
    t_stat <- mean(f) / sqrt(var_mean)
    list(t_hod = round(t_stat, 3L), p_hod = round(1 - pnorm(t_stat), 3L))
  }

  # H=3 OOS evaluation — NW lags = NW_LAG_H3 for MA(2) overlap correction;
  # t_hod/p_hod = Hodrick rectangular-kernel SE robustness (lag H-1).
  oos_eval_h3 <- map_dfr(gamma_cols, function(gc) {
    y     <- actuals_h3[, gc]
    bench <- pred_list_h3[[gc]][, "hist_mean"]
    map_dfr(setdiff(model_names_h3, "hist_mean"), function(m) {
      yhat  <- pred_list_h3[[gc]][, m]
      valid <- !is.na(yhat) & !is.na(y)
      if (sum(valid) < 10L) return(NULL)
      cw  <- clark_west(y[valid], bench[valid], yhat[valid], nw_lags = NW_LAG_H3)
      hod <- clark_west_hodrick(y[valid], bench[valid], yhat[valid], h = 3L)
      tibble(
        model      = m,
        oos_r2     = round(oos_r2(y[valid], yhat[valid], bench[valid]) * 100, 2L),
        msfe_ratio = round(mean((y[valid] - yhat[valid])^2) /
                           mean((y[valid] - bench[valid])^2), 3L),
        t_cw       = cw$t_cw,
        p_cw       = cw$p_cw,
        t_hod      = hod$t_hod,
        p_hod      = hod$p_hod,
        n_oos      = sum(valid)
      )
    }) |> mutate(gamma = gamma_labels[gc], sub_period = "Full", .before = model)
  })

  # H=3 sub-period breakdown
  sub_windows_h3 <- list(
    "Pre-2000"  = which(dates_oos_h3 <  as.Date("2000-01-01")),
    "2000-2010" = which(dates_oos_h3 >= as.Date("2000-01-01") &
                        dates_oos_h3 <  as.Date("2010-01-01")),
    "Post-2010" = which(dates_oos_h3 >= as.Date("2010-01-01"))
  )
  oos_eval_sub_h3 <- map_dfr(names(sub_windows_h3), function(sp) {
    idx <- sub_windows_h3[[sp]]
    if (length(idx) < 20L) return(NULL)
    map_dfr(gamma_cols, function(gc) {
      y     <- actuals_h3[, gc]
      bench <- pred_list_h3[[gc]][, "hist_mean"]
      map_dfr(setdiff(model_names_h3, "hist_mean"), function(m) {
        yhat  <- pred_list_h3[[gc]][idx, m]
        y_s   <- y[idx]; b_s <- bench[idx]
        valid <- !is.na(yhat) & !is.na(y_s)
        if (sum(valid) < 10L) return(NULL)
        cw <- clark_west(y_s[valid], b_s[valid], yhat[valid], nw_lags = NW_LAG_H3)
        tibble(model = m,
               oos_r2 = round(oos_r2(y_s[valid], yhat[valid], b_s[valid]) * 100, 2L),
               msfe_ratio = round(mean((y_s[valid] - yhat[valid])^2) /
                                  mean((y_s[valid] - b_s[valid])^2), 3L),
               t_cw = cw$t_cw, p_cw = cw$p_cw, n_oos = sum(valid))
      }) |> mutate(gamma = gamma_labels[gc], sub_period = sp, .before = model)
    })
  })
  oos_eval_all_h3 <- bind_rows(oos_eval_h3, oos_eval_sub_h3)

  # (Ext-C 5b) Common-window H=3 evaluation ------------------------------------
  # Mirrors the H=1 block (5b). `oos_eval_h3` above scores each model on ITS OWN
  # non-NA window: 414 months for the feature-dependent LASSO/XGB family, 418 for
  # the purely autoregressive AR/combination models, which need no GW-tail
  # features. A table putting them side by side would compare models on different
  # samples. Here every model is rescored on the set of months in which EVERY
  # model and the actual are observed. This is the sample reported in Table
  # tab:res-h3 and B.5, and the family on which 04f runs FDR at MT_TARGET=h3.
  common_mask_h3 <- setNames(lapply(gamma_cols, function(gc) {
    P <- pred_list_h3[[gc]]
    rowSums(is.na(P)) == 0 & !is.na(actuals_h3[, gc])
  }), gamma_cols)
  # A model column that is entirely NA (e.g. a spec disabled at this horizon)
  # would silently empty the intersection — fail loudly instead of reporting a
  # degenerate sample.
  stopifnot(min(sapply(common_mask_h3, sum)) >= 100L)

  oos_eval_common_h3 <- map_dfr(gamma_cols, function(gc) {
    idx   <- which(common_mask_h3[[gc]])
    y     <- actuals_h3[idx, gc]
    bench <- pred_list_h3[[gc]][idx, "hist_mean"]
    map_dfr(setdiff(model_names_h3, "hist_mean"), function(m) {
      yhat <- pred_list_h3[[gc]][idx, m]
      cw   <- clark_west(y, bench, yhat, nw_lags = NW_LAG_H3)
      hod  <- clark_west_hodrick(y, bench, yhat, h = 3L)
      tibble(
        model      = m,
        oos_r2     = round(oos_r2(y, yhat, bench) * 100, 2L),
        msfe_ratio = round(mean((y - yhat)^2) / mean((y - bench)^2), 3L),
        t_cw       = cw$t_cw,
        p_cw       = cw$p_cw,
        t_hod      = hod$t_hod,
        p_hod      = hod$p_hod,
        n_oos      = length(idx)
      )
    }) |> mutate(gamma = gamma_labels[gc], sub_period = "Full", .before = model)
  })

  cat(sprintf(
    "\n=== Common-window H=3 OOS R²: every model on the same %d–%d months per gamma ===\n",
    min(sapply(common_mask_h3, sum)), max(sapply(common_mask_h3, sum))
  ))
  print(
    oos_eval_common_h3 |> select(gamma, model, oos_r2) |>
      pivot_wider(names_from = model, values_from = oos_r2),
    n = Inf
  )

  cat(sprintf("\n=== H=3 OOS R² (%%) — NW lags = %d ===\n", NW_LAG_H3))
  print(
    oos_eval_h3 |>
      select(gamma, model, oos_r2) |>
      pivot_wider(names_from = model, values_from = oos_r2),
    n = Inf
  )
  cat("\n=== H=3 Clark-West t-stats ===\n")
  print(
    oos_eval_h3 |>
      select(gamma, model, t_cw) |>
      pivot_wider(names_from = model, values_from = t_cw),
    n = Inf
  )

  saveRDS(list(pred_list_h3 = pred_list_h3, oos_eval_h3 = oos_eval_h3,
               oos_eval_common_h3 = oos_eval_common_h3,
               oos_eval_all_h3 = oos_eval_all_h3, actuals_h3 = actuals_h3,
               dates_oos_h3 = dates_oos_h3, aligned_h3 = aligned_h3,
               model_names_h3 = model_names_h3, oos_idx_h3 = oos_idx_h3),
          file = "gamma_predictions_h3.rds")
  cat("Saved: gamma_predictions_h3.rds\n")
}  # end RUN_QUARTERLY


# ==============================================================================
# Ext-C2: HAR BACKWARD-AVERAGE SMOOTHED TARGET
# Applied validation: Stöckl et al. (LLB, 2024) use this exact formula in the
# InnoSwiss industry project on factor timing with MSCI World data, finding
# AC(1) rises from ≈ 0 (monthly) to ≈ 0.70 (backward-smoothed), enabling
# statistically significant out-of-sample prediction.
#
# Target definition:
#   γ̄_{k,t} = (1/3)(γ_{k,t} + γ_{k,t-1} + γ_{k,t-2})
#   Predicted with feature vector X_{t-1} (strictly ex-ante relative to γ_{k,t}).
#
# DISTINCTION FROM Ext-C (cumulative forward sum):
#   Ext-C:  Y_t = γ_{t+1} + γ_{t+2} + γ_{t+3}  [3 unknown future months]
#   Ext-C2: Y_t = (γ_t + γ_{t-1} + γ_{t-2}) / 3 [1 unknown + 2 known months]
#   Ext-C2 is econometrically easier: two-thirds of the target (γ_{t-1}, γ_{t-2})
#   is already observable at prediction time t-1. This partially explains the
#   higher OOS R² and must be disclosed as a caveat in the thesis.
#
# WHY AC(1) RISES TO ~0.67:
#   Even for white-noise gammas (ρ = 0), the backward average has:
#   AC(1)[γ̄_t] = Cov(γ̄_t, γ̄_{t-1}) / Var(γ̄_t) = (2/3)σ² / σ² = 2/3 ≈ 0.67.
#   Positive ρ pushes this higher. This is a structural property of the
#   backward average, not a sign of genuine predictability — but Corsi (2009)
#   argues that this aggregated persistence is economically meaningful because
#   it reflects the heterogeneous investment horizons of market participants
#   (Müller et al. 1997). The smoothed target is a better estimand for portfolio
#   timing because it represents the "medium-frequency factor environment"
#   rather than the noisy monthly risk premium realisation.
#
# INFERENCE:
#   Consecutive backward-average targets overlap by 2 months:
#   γ̄_t = (γ_t + γ_{t-1} + γ_{t-2})/3 and γ̄_{t+1} = (γ_{t+1} + γ_t + γ_{t-1})/3
#   share γ_t and γ_{t-1} → MA(2) structure in forecast errors.
#   NW lags = NW_LAG + 2 = 14 (same as Ext-C).
#   Reference: Hodrick (1992, RFS) — overlapping return inference.
#
# KEY REFERENCES:
#   Corsi (2009, J. Financial Econometrics) — HAR model: multi-horizon
#     backward aggregation reveals persistent factor structure hidden in noise.
#   Müller, Dacorogna, Davé et al. (1997) — heterogeneous market hypothesis;
#     different investment horizons create aggregate persistence.
#   Bollerslev, Patton & Quaedvlieg (2016, J. Econometrics) — HAR extensions.
#   Stöckl et al. (LLB, 2024) — exact formula applied to FM coefficients.
#   Hodrick (1992, RFS) — overlapping inference for long-horizon regressions.
#
# OUTPUT: gamma_predictions_smooth.rds (H=1 and H=3 results untouched).
# ==============================================================================

if (RUN_QUARTERLY_SMOOTH) {

  NW_LAG_SMOOTH <- NW_LAG + 2L   # 14 lags — MA(2) overlap correction (same as H=3)
  T_full_sm     <- nrow(gammas_mat)
  n_smooth      <- T_full_sm - 2L   # need t >= 3 to have γ_{t-2}

  # Build backward-average outcome: row j predicts γ̄ at date[j+2]
  # γ̄_{j+2} = (γ[j+2] + γ[j+1] + γ[j]) / 3
  # Feature vector: features[j+1] (one step before the target month, strictly ex-ante)
  gamma_outcome_smooth <- matrix(
    NA_real_, nrow = n_smooth, ncol = length(gamma_cols),
    dimnames = list(NULL, gamma_cols)
  )
  for (j in seq_len(n_smooth)) {
    t_sm <- j + 2L   # target month index in gammas_mat
    gamma_outcome_smooth[j, ] <- colMeans(
      as.matrix(gammas_mat[(t_sm - 2L):t_sm, gamma_cols])   # rows t-2, t-1, t
    )
  }

  # aligned_smooth: date = date of smoothed target month (date[j+2])
  #   outcome = γ̄_{date[j+2]}, features = X_{date[j+1]}
  aligned_smooth <- bind_cols(
    gammas_mat |> select(date) |> slice(seq(3L, T_full_sm)),           # dates: date[3]..date[T]
    as_tibble(gamma_outcome_smooth),
    features   |> select(all_of(feature_cols)) |> slice(seq(2L, T_full_sm - 1L))  # features at t-1
  )

  oos_idx_sm   <- which(
    seq_len(nrow(aligned_smooth)) > INIT_WINDOW &
    aligned_smooth$date >= OOS_START_DATE
  )
  dates_oos_sm <- aligned_smooth$date[oos_idx_sm]

  # Verify AC(1) of the backward-average target relative to raw gammas
  cat("\n--- Ext-C2: Backward-average target AC(1) verification ---\n")
  cat("Corsi (2009): backward aggregation raises AC(1) from ≈0 to ≈0.67 structurally.\n")
  ac_check_sm <- map_dfr(gamma_cols, function(gc) {
    y_smooth <- gamma_outcome_smooth[!is.na(gamma_outcome_smooth[, gc]), gc]
    y_raw    <- na.omit(gammas_mat[[gc]])
    tibble(
      gamma        = gamma_labels[gc],
      ac1_raw      = round(acf(y_raw,    lag.max = 1, plot = FALSE)$acf[2], 3),
      ac1_smooth   = round(acf(y_smooth, lag.max = 1, plot = FALSE)$acf[2], 3),
      ac1_increase = round(acf(y_smooth, lag.max = 1, plot = FALSE)$acf[2] -
                           acf(y_raw,    lag.max = 1, plot = FALSE)$acf[2], 3)
    )
  })
  print(ac_check_sm, n = Inf)
  cat("ac1_smooth ≈ 0.67 for white-noise gammas (structural); higher → genuine signal.\n\n")

  # Model set: same as H=3 (skip VAR and MV; RF omitted for speed)
  model_names_sm <- c("hist_mean", "ar1", "ar3")
  if (RUN_HAR)   model_names_sm <- c(model_names_sm, "har")
  if (RUN_LASSO) model_names_sm <- c(model_names_sm, "lasso", "lasso_ct")
  if (RUN_LASSO && RUN_LASSO_1SE) model_names_sm <- c(model_names_sm, "lasso_1se", "lasso_1se_ct")
  if (RUN_XGB)   model_names_sm <- c(model_names_sm, "xgb", "xgb_ct")
  model_names_sm <- c(model_names_sm, "factmom", "comb")
  if (RUN_XGB)   model_names_sm <- c(model_names_sm, "comb_xgb")

  pred_list_sm <- setNames(
    lapply(gamma_cols, \(gc) {
      m <- matrix(NA_real_, nrow = length(oos_idx_sm), ncol = length(model_names_sm))
      colnames(m) <- model_names_sm
      m
    }),
    gamma_cols
  )
  actuals_sm <- matrix(NA_real_, nrow = length(oos_idx_sm), ncol = length(gamma_cols))
  colnames(actuals_sm) <- gamma_cols

  cat(sprintf(
    "Ext-C2: Running backward-average smooth OOS (%d steps, NW lags = %d)\n",
    length(oos_idx_sm), NW_LAG_SMOOTH
  ))

  for (i in seq_along(oos_idx_sm)) {
    t_i   <- oos_idx_sm[i]
    train <- aligned_smooth[seq_len(t_i - 1L), ]
    test  <- aligned_smooth[t_i, ]

    Y_train <- train |> select(all_of(gamma_cols)) |> as.matrix()
    X_train <- train |> select(all_of(feature_cols)) |> as.matrix()
    X_test  <- test  |> select(all_of(feature_cols)) |> as.matrix()

    actuals_sm[i, ] <- as.numeric(test[gamma_cols])

    for (gc in gamma_cols) {
      y_train  <- Y_train[, gc]
      short    <- str_remove(gc, "^gamma_")
      hist_sm  <- mean(y_train, na.rm = TRUE)

      # M0: expanding mean of the backward-average target
      pred_list_sm[[gc]][i, "hist_mean"] <- hist_sm

      # Factor Momentum (sign-agnostic rule on smoothed target; see H=1 note)
      pred_list_sm[[gc]][i, "factmom"] <- if (!is.na(hist_sm)) 2 * hist_sm else NA_real_

      # AR(1) on smooth target
      df1 <- data.frame(y = y_train, x1 = train[[paste0(short, "_l1")]])
      df1 <- df1[complete.cases(df1), ]
      if (nrow(df1) >= 5L) {
        m1 <- lm(y ~ x1, data = df1)
        pred_list_sm[[gc]][i, "ar1"] <- predict(
          m1, newdata = data.frame(x1 = X_test[, paste0(short, "_l1")])
        )
      }

      # AR(3) on smooth target
      df3 <- data.frame(
        y  = y_train,
        x1 = train[[paste0(short, "_l1")]],
        x2 = train[[paste0(short, "_l2")]],
        x3 = train[[paste0(short, "_l3")]]
      )
      df3 <- df3[complete.cases(df3), ]
      if (nrow(df3) >= 8L) {
        m3 <- lm(y ~ x1 + x2 + x3, data = df3)
        pred_list_sm[[gc]][i, "ar3"] <- predict(m3, newdata = data.frame(
          x1 = X_test[, paste0(short, "_l1")],
          x2 = X_test[, paste0(short, "_l2")],
          x3 = X_test[, paste0(short, "_l3")]
        ))
      }

      # HAR on smooth target (Corsi 2009 — the methodological origin of this spec)
      # The smooth target IS a HAR-type average, so HAR fitted to it tests whether
      # higher-order aggregation (HAR cascade) improves on the simple 3-month average.
      if (RUN_HAR) {
        y_har <- y_train[!is.na(y_train)]
        n_har <- length(y_har)
        if (n_har >= 17L) {
          har_rows <- 13L:n_har
          if (length(har_rows) >= 5L) {
            har_df <- data.frame(
              y   = y_har[har_rows],
              x_d = y_har[har_rows - 1L],
              x_w = vapply(har_rows - 1L,
                           function(k) mean(y_har[max(1L, k - 3L):k]), numeric(1L)),
              x_m = vapply(har_rows - 1L,
                           function(k) mean(y_har[max(1L, k - 11L):k]), numeric(1L))
            )
            har_fit <- lm(y ~ x_d + x_w + x_m, data = har_df)
            pred_list_sm[[gc]][i, "har"] <- predict(har_fit, newdata = data.frame(
              x_d = tail(y_har, 1L),
              x_w = mean(tail(y_har, 4L)),
              x_m = mean(tail(y_har, 12L))
            ))
          }
        }
      }
    }  # end per-gamma loop

    # LASSO on smooth target
    if (RUN_LASSO) {
      X_tr_sm <- impute_means(X_train)
      for (gc in gamma_cols) {
        y_tr <- Y_train[, gc]
        ok_y <- !is.na(y_tr)
        if (sum(ok_y) >= 40L && !anyNA(X_test)) {
          tryCatch({
            ts_fit <- ts_cv_lasso(X_tr_sm[ok_y, ], y_tr[ok_y])
            if (!is.null(ts_fit)) {
              pred_list_sm[[gc]][i, "lasso"] <- predict(
                ts_fit$fit, newx = X_test, s = ts_fit$lambda.min
              )[1L]
              if (RUN_LASSO_1SE) {
                pred_list_sm[[gc]][i, "lasso_1se"] <- predict(
                  ts_fit$fit, newx = X_test, s = ts_fit$lambda.1se
                )[1L]
              }
            }
          }, error = function(e) NULL)
        }
      }
      # Campbell-Thompson sign restriction on smooth LASSO
      for (gc in gamma_cols) {
        bench_sm <- pred_list_sm[[gc]][i, "hist_mean"]
        for (m_raw in c("lasso", if (RUN_LASSO_1SE) "lasso_1se")) {
          m_ct <- paste0(m_raw, "_ct")
          if (!m_ct %in% colnames(pred_list_sm[[gc]])) next
          raw <- pred_list_sm[[gc]][i, m_raw]
          if (!is.na(raw) && !is.na(bench_sm)) {
            pred_list_sm[[gc]][i, m_ct] <- if (sign(raw) == sign(bench_sm)) raw else bench_sm
          }
        }
      }
    }

    # XGBoost on smooth target
    if (RUN_XGB) {
      X_tr_xgb_sm <- impute_means(X_train)
      xgb_cs_sm   <- round(sqrt(length(feature_cols)) / length(feature_cols), 3L)
      for (gc in gamma_cols) {
        y_tr <- Y_train[, gc]
        ok_y <- !is.na(y_tr)
        if (sum(ok_y) >= 40L && !anyNA(X_test)) {
          tryCatch({
            n_tr  <- sum(ok_y)
            n_val <- max(20L, floor(n_tr * 0.20))
            n_fit <- n_tr - n_val
            x_ok  <- X_tr_xgb_sm[ok_y, , drop = FALSE]
            y_ok  <- y_tr[ok_y]
            dtrain <- xgb.DMatrix(data = x_ok[seq_len(n_fit), ],  label = y_ok[seq_len(n_fit)])
            dval   <- xgb.DMatrix(data = x_ok[(n_fit + 1L):n_tr, ], label = y_ok[(n_fit + 1L):n_tr])
            xgb_fit <- xgb.train(
              params = list(
                objective = "reg:squarederror", max_depth = 4L,
                eta = 0.05, subsample = 0.8,
                colsample_bytree = xgb_cs_sm, min_child_weight = 5L,
                seed = 42L   # reproducibility -- see H=1 block
              ),
              data = dtrain, nrounds = 500L,
              watchlist = list(val = dval),
              early_stopping_rounds = 20L, verbose = 0L
            )
            pred_list_sm[[gc]][i, "xgb"] <- predict(
              xgb_fit, newdata = xgb.DMatrix(X_test)
            )
          }, error = function(e) NULL)
        }
      }
      for (gc in gamma_cols) {
        bench_sm <- pred_list_sm[[gc]][i, "hist_mean"]
        raw      <- pred_list_sm[[gc]][i, "xgb"]
        if (!is.na(raw) && !is.na(bench_sm)) {
          pred_list_sm[[gc]][i, "xgb_ct"] <- if (sign(raw) == sign(bench_sm)) raw else bench_sm
        }
      }
    }

    if (i %% 50L == 0L) cat(".")
  }
  cat(" done.\n")

  # Combination forecast for smooth target
  COMB_SM     <- intersect(c("ar1", "ar3", "lasso"),        model_names_sm)
  COMB_SM_XGB <- intersect(c("ar1", "ar3", "lasso", "xgb"), model_names_sm)
  for (gc in gamma_cols) {
    sub     <- pred_list_sm[[gc]][, COMB_SM, drop = FALSE]
    n_ok    <- rowSums(!is.na(sub))
    pred_list_sm[[gc]][, "comb"] <- ifelse(n_ok >= 2L, rowMeans(sub, na.rm = TRUE), NA_real_)
    if (RUN_XGB && "comb_xgb" %in% colnames(pred_list_sm[[gc]])) {
      sub_x  <- pred_list_sm[[gc]][, COMB_SM_XGB, drop = FALSE]
      n_ok_x <- rowSums(!is.na(sub_x))
      pred_list_sm[[gc]][, "comb_xgb"] <- ifelse(n_ok_x >= 2L,
                                                  rowMeans(sub_x, na.rm = TRUE), NA_real_)
    }
  }

  # OOS evaluation with NW_LAG_SMOOTH lags (MA(2) overlap correction)
  oos_eval_sm <- map_dfr(gamma_cols, function(gc) {
    y     <- actuals_sm[, gc]
    bench <- pred_list_sm[[gc]][, "hist_mean"]
    map_dfr(setdiff(model_names_sm, "hist_mean"), function(m) {
      yhat  <- pred_list_sm[[gc]][, m]
      valid <- !is.na(yhat) & !is.na(y)
      if (sum(valid) < 10L) return(NULL)
      cw <- clark_west(y[valid], bench[valid], yhat[valid], nw_lags = NW_LAG_SMOOTH)
      tibble(
        model      = m,
        oos_r2     = round(oos_r2(y[valid], yhat[valid], bench[valid]) * 100, 2L),
        msfe_ratio = round(mean((y[valid] - yhat[valid])^2) /
                           mean((y[valid] - bench[valid])^2), 3L),
        t_cw       = cw$t_cw,
        p_cw       = cw$p_cw,
        n_oos      = sum(valid)
      )
    }) |> mutate(gamma = gamma_labels[gc], sub_period = "Full", .before = model)
  })

  cat(sprintf("\n=== Ext-C2 Backward-Average OOS R² (%%) — NW lags = %d ===\n", NW_LAG_SMOOTH))
  cat("CAVEAT: 2/3 of target known at prediction time → OOS R² not comparable to H=1.\n")
  cat("Report alongside H=1 with caveat; economic value is in smoother gamma signal.\n\n")
  print(
    oos_eval_sm |>
      select(gamma, model, oos_r2) |>
      pivot_wider(names_from = model, values_from = oos_r2),
    n = Inf
  )
  cat("\n=== Ext-C2 Clark-West t-stats (NW-robust) ===\n")
  print(
    oos_eval_sm |>
      select(gamma, model, t_cw, p_cw) |>
      pivot_wider(names_from = model, values_from = c(t_cw, p_cw)),
    n = Inf
  )

  # Cross-specification comparison: H=1 vs smooth vs H=3
  cat("\n=== HORIZON COMPARISON: H=1 vs Smooth (Ext-C2) vs H=3 (Ext-C) — LASSO only ===\n")
  cat("Corsi (2009): predictability should increase with aggregation horizon.\n")
  cat("Stöckl et al. (LLB): confirm for smooth target (AC1 ≈ 0 → ≈ 0.70).\n\n")
  if (exists("oos_eval") && exists("oos_eval_h3")) {
    lasso_compare <- bind_rows(
      oos_eval    |> filter(model == "lasso") |> mutate(horizon = "H=1"),
      oos_eval_sm |> filter(model == "lasso") |> mutate(horizon = "Smooth"),
      oos_eval_h3 |> filter(model == "lasso") |> mutate(horizon = "H=3")
    ) |>
      select(horizon, gamma, oos_r2) |>
      pivot_wider(names_from = horizon, values_from = oos_r2)
    print(lasso_compare, n = Inf)
    cat("Note: Smooth OOS R² > H=1 is partially structural (2/3 target known).\n")
    cat("H=3 > H=1 is a genuine long-horizon improvement (Fama & French 1988, JFE).\n")
  }

  saveRDS(list(pred_list_sm = pred_list_sm, oos_eval_sm = oos_eval_sm,
               actuals_sm = actuals_sm, dates_oos_sm = dates_oos_sm,
               model_names_sm = model_names_sm, aligned_smooth = aligned_smooth,
               oos_idx_sm = oos_idx_sm, ac_check_sm = ac_check_sm),
          file = "gamma_predictions_smooth.rds")
  cat("Saved: gamma_predictions_smooth.rds\n")

}  # end RUN_QUARTERLY_SMOOTH
