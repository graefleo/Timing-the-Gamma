# ==============================================================================
# 02_fama_macbeth.R
# Runs monthly cross-sectional regressions (Fama-MacBeth step 1) and
# produces the gamma time series used for prediction in step 3.
#
# Primary specification : WLS with lagged market cap as weights.
#   Rationale: microcaps dominate the OLS cross-section but account for only
#   ~3% of total market cap; value-weighting (WLS) suppresses their influence.
#   Reference: Fama & French (2008, JFE); Lewellen, Nagel & Shanken (2010, JFE).
#
# Robustness            : OLS reported alongside for comparison.
#
# Shanken (1992) correction for beta EIV bias:
#   Market betas estimated in the first pass carry measurement error, which
#   attenuates the second-pass slope on beta and inflates other t-stats.
#   Correction factor: c = (r̄_mkt)^2 / Var(r_mkt), the market's squared Sharpe.
#   Variance inflates by (1 + c) => SE by sqrt(1 + c) => corrected t = t / sqrt(1 + c).
#   Reference: Shanken (1992, RFS); Kim (1995); Jagannathan & Wang (1998).
#
# Input:  panel_clean.rds, ff5_factors.rds  (from 01_data_pipeline.R)
# Output: gammas_ts.rds    (objects: gammas_ts, fm_table, gammas_ts_ols, fm_table_ols)
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(broom)
library(lmtest)
library(sandwich)

panel_clean <- readRDS("panel_clean.rds")   # -> panel_clean
ff5_factors <- readRDS("ff5_factors.rds")   # -> ff5_factors  (needed for Shanken correction)


# (1) Specification ------------------------------------------------------------
chars_std <- c(
  "beta_std", "bm_std", "mom12m_std",
  "oper_prof_std", "asset_growth_std", "log_mktcap_lag_std"
)

fm_formula <- as.formula(
  paste("ret_excess ~", paste(chars_std, collapse = " + "))
)


# (2) Monthly cross-sectional regressions: WLS (primary) + OLS (robustness) ----
cat("Running cross-sectional regressions (WLS primary, OLS robustness)...\n")

monthly_regs <- panel_clean |>
  group_by(date) |>
  nest() |>
  mutate(
    # Primary: value-weighted (WLS with lagged market cap)
    fit_wls   = map(data, \(d) lm(fm_formula, data = d, weights = mktcap_lag)),
    coefs_wls = map(fit_wls, tidy),
    r_sq_wls  = map_dbl(fit_wls, \(m) summary(m)$r.squared),
    # Robustness: equal-weighted (OLS)
    fit_ols   = map(data, \(d) lm(fm_formula, data = d)),
    coefs_ols = map(fit_ols, tidy),
    r_sq_ols  = map_dbl(fit_ols, \(m) summary(m)$r.squared),
    n         = map_int(data, nrow)
  ) |>
  ungroup()

cat(sprintf("  Done. %d months estimated.\n", nrow(monthly_regs)))


# (2b) VIF diagnostics — multicollinearity among cross-sectional regressors ----
# VIF_j = 1 / (1 - R²_j), where R²_j is from regressing characteristic j on
# all other characteristics within the same cross-section.
# Rule-of-thumb thresholds: VIF > 5 = moderate concern, VIF > 10 = severe.
# Computed monthly, then summarised over time.
# Reference: Gujarati & Porter (2009); Lewellen, Nagel & Shanken (2010, JFE)

vif_cross_section <- function(d, chars) {
  X <- as.matrix(d[, chars])
  X <- X[complete.cases(X), , drop = FALSE]
  if (nrow(X) < (ncol(X) + 2L)) return(rep(NA_real_, length(chars)))
  vapply(seq_along(chars), function(j) {
    y  <- X[, j]
    xj <- X[, -j, drop = FALSE]
    r2 <- summary(lm(y ~ xj))$r.squared
    1 / (1 - r2)
  }, numeric(1L))
}

cat("\nComputing monthly VIF for cross-sectional regressors...\n")

vif_monthly <- monthly_regs |>
  mutate(vif = map(data, \(d) {
    v <- vif_cross_section(d, chars_std)
    setNames(as.list(v), chars_std)
  })) |>
  select(date, vif) |>
  unnest_wider(vif)

vif_summary <- vif_monthly |>
  summarise(across(
    all_of(chars_std),
    list(
      mean = \(x) round(mean(x, na.rm = TRUE), 2),
      max  = \(x) round(max(x,  na.rm = TRUE), 2),
      pct_above5  = \(x) round(100 * mean(x > 5,  na.rm = TRUE), 1),
      pct_above10 = \(x) round(100 * mean(x > 10, na.rm = TRUE), 1)
    )
  )) |>
  pivot_longer(everything(),
               names_to = c("characteristic", ".value"),
               names_pattern = "(.+)_(mean|max|pct_above5|pct_above10)") |>
  mutate(characteristic = str_remove(characteristic, "_std$"))

cat("\n=== VIF Summary (cross-sectional regressors) ===\n")
cat("Thresholds: >5 = moderate concern, >10 = severe\n\n")
print(vif_summary, n = Inf)

# Flag months where any characteristic breaches VIF > 10
severe_months <- vif_monthly |>
  filter(if_any(all_of(chars_std), \(x) !is.na(x) & x > 10)) |>
  nrow()

cat(sprintf(
  "\nMonths with any VIF > 10: %d / %d (%.1f%%)\n",
  severe_months, nrow(vif_monthly),
  100 * severe_months / nrow(vif_monthly)
))


# (3) Helper: extract wide gamma time series from a coefs column ---------------
extract_gammas <- function(monthly_regs, coefs_col, rsq_col) {
  monthly_regs |>
    select(date, n, coefs = {{ coefs_col }}, r_sq = {{ rsq_col }}) |>
    unnest(coefs) |>
    select(date, term, estimate, r_sq, n) |>
    pivot_wider(names_from = term, values_from = estimate) |>
    rename(
      gamma_intercept    = `(Intercept)`,
      gamma_beta         = beta_std,
      gamma_bm           = bm_std,
      gamma_mom12m       = mom12m_std,
      gamma_oper_prof    = oper_prof_std,
      gamma_asset_growth = asset_growth_std,
      gamma_size         = log_mktcap_lag_std
    ) |>
    arrange(date)
}

gammas_ts     <- extract_gammas(monthly_regs, coefs_wls, r_sq_wls)
gammas_ts_ols <- extract_gammas(monthly_regs, coefs_ols, r_sq_ols)

gamma_cols <- c(
  "gamma_intercept", "gamma_beta", "gamma_bm", "gamma_mom12m",
  "gamma_oper_prof", "gamma_asset_growth", "gamma_size"
)


# (4) Fama-MacBeth inference ---------------------------------------------------
# NW t-stat: accounts for autocorrelation in gamma_t (lag = 12 months)
nw_tstat <- function(x, lag = 12) {
  x <- x[!is.na(x)]
  fit <- lm(x ~ 1)
  ct  <- coeftest(fit, vcov = NeweyWest(fit, lag = lag, prewhite = FALSE))
  ct["(Intercept)", "t value"]
}

build_fm_table <- function(gammas) {
  gammas |>
    summarise(across(
      all_of(gamma_cols),
      list(
        mean = \(x) mean(x, na.rm = TRUE),
        sd   = \(x) sd(x,   na.rm = TRUE),
        t_fm = \(x) {
          m <- mean(x, na.rm = TRUE)
          s <- sd(x,   na.rm = TRUE)
          n <- sum(!is.na(x))
          m / (s / sqrt(n))
        },
        t_nw = \(x) nw_tstat(x, lag = 12)
      )
    )) |>
    pivot_longer(
      everything(),
      names_to      = c("gamma", ".value"),
      names_pattern = "(.+)_(mean|sd|t_fm|t_nw)"
    ) |>
    mutate(gamma = str_remove(gamma, "^gamma_"))
}

fm_table     <- build_fm_table(gammas_ts)
fm_table_ols <- build_fm_table(gammas_ts_ols)


# (5) Shanken (1992) EIV correction -------------------------------------------
# c = (r̄_mkt)^2 / Var(r_mkt) — the squared monthly Sharpe ratio of the market.
#   Shanken's factor is c = γ̂₁' Σ_f^{-1} γ̂₁, where γ̂₁ is the price of COVARIANCE
#   risk. Under a correctly specified single-factor model the factor premium is
#   the market's own mean excess return, so c reduces to its squared Sharpe.
#   (Revised 2026-07-20: previously used γ̄_beta — the cross-sectional price of
#   the beta CHARACTERISTIC, which is a premium per unit of rank spread, not a
#   factor price. Numerically both are ≈ 0, so no reported figure moves; the
#   change buys the citable textbook object. Because beta enters here as a
#   rank-standardised characteristic rather than a covariance loading, the
#   correction is an upper-bound heuristic in this setting.)
#
# The correction inflates the VARIANCE by (1 + c); standard errors therefore
# scale by sqrt(1 + c), and corrected t = FM t / sqrt(1 + c).
# Note: correction is conservative — it reduces ALL t-stats, not just beta's,
# although only beta is a generated regressor (Daniel & Titman, 1997).

mkt_var    <- var(ff5_factors$mkt_excess,  na.rm = TRUE)
mkt_mean   <- mean(ff5_factors$mkt_excess, na.rm = TRUE)
shanken_c  <- mkt_mean^2 / mkt_var

cat(sprintf("\nShanken correction factor c = %.4f (sqrt(1+c) = %.4f)\n",
            shanken_c, sqrt(1 + shanken_c)))

fm_table <- fm_table |>
  mutate(
    t_nw_shanken = t_nw / sqrt(1 + shanken_c),
    t_fm_shanken = t_fm / sqrt(1 + shanken_c)
  )


# (6) Print results ------------------------------------------------------------
format_table <- function(tbl, label) {
  cat(sprintf("\n=== Fama-MacBeth Risk Premia [%s] ===\n", label))
  print(
    tbl |>
      filter(gamma != "intercept") |>
      mutate(
        across(c(mean, sd), \(x) round(x * 100, 3)),
        across(starts_with("t_"),  \(x) round(x, 2))
      ) |>
      rename(
        Characteristic      = gamma,
        `Mean (% / mo)`     = mean,
        `SD (% / mo)`       = sd,
        `t (FM)`            = t_fm,
        `t (NW-12)`         = t_nw
      ),
    n = Inf
  )
}

format_table(fm_table,     "WLS — primary")
format_table(fm_table_ols, "OLS — robustness")

cat(sprintf(
  "\nWLS — Mean cross-sectional R²: %.1f%%  (median: %.1f%%)\n",
  mean(gammas_ts$r_sq,   na.rm = TRUE) * 100,
  median(gammas_ts$r_sq, na.rm = TRUE) * 100
))
cat(sprintf(
  "OLS — Mean cross-sectional R²: %.1f%%  (median: %.1f%%)\n",
  mean(gammas_ts_ols$r_sq,   na.rm = TRUE) * 100,
  median(gammas_ts_ols$r_sq, na.rm = TRUE) * 100
))

cat("\n=== Shanken-corrected t-stats (WLS) ===\n")
print(
  fm_table |>
    filter(gamma != "intercept") |>
    select(gamma, t_nw, t_nw_shanken) |>
    mutate(across(c(t_nw, t_nw_shanken), \(x) round(x, 2))) |>
    rename(Characteristic = gamma, `t (NW-12)` = t_nw,
           `t (NW-12, Shanken)` = t_nw_shanken),
  n = Inf
)


# (7) Save ---------------------------------------------------------------------
# gammas_ts: WLS gammas (primary); fm_table: includes Shanken-corrected stats
# fm_table_ols / gammas_ts_ols: OLS robustness
saveRDS(list(gammas_ts = gammas_ts, fm_table = fm_table,
             gammas_ts_ols = gammas_ts_ols, fm_table_ols = fm_table_ols),
        file = "gammas_ts.rds")
cat("Saved: gammas_ts.rds\n")


# (8) CSV export ---------------------------------------------------------------
# Gamma values are in decimal (e.g. 0.002 = 0.2% per month).
# Both WLS (primary) and OLS (robustness) are exported side by side.
# Columns: date, n_stocks, r_sq, gamma_* (WLS), gamma_*_ols (OLS).
gammas_export <- gammas_ts |>
  select(date, n_stocks = n, r_sq_wls = r_sq,
         gamma_bm, gamma_mom12m, gamma_oper_prof,
         gamma_asset_growth, gamma_size, gamma_beta,
         gamma_intercept) |>
  left_join(
    gammas_ts_ols |>
      select(date, r_sq_ols = r_sq,
             gamma_bm_ols           = gamma_bm,
             gamma_mom12m_ols       = gamma_mom12m,
             gamma_oper_prof_ols    = gamma_oper_prof,
             gamma_asset_growth_ols = gamma_asset_growth,
             gamma_size_ols         = gamma_size,
             gamma_beta_ols         = gamma_beta),
    by = "date"
  )

write_csv(gammas_export, "gammas_ts.csv")
cat("Saved: gammas_ts.csv\n")
cat(sprintf("  Rows: %d | Columns: %d | Date range: %s to %s\n",
            nrow(gammas_export), ncol(gammas_export),
            format(min(gammas_export$date), "%Y-%m"),
            format(max(gammas_export$date), "%Y-%m")))
