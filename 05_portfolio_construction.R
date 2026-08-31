# ==============================================================================
# 05_portfolio_construction.R
# Constructs long-short characteristic portfolios and evaluates factor timing.
#
# For each characteristic k and each prediction model:
#   Static:     sign(γ̄_k,T0) × (Q5 return − Q1 return)      [no timing; γ̄ frozen
#               at the expanding mean as of evaluation start T0 — real-time]
#   Direction:  sign(γ̂_{k,t}) × (Q5 − Q1)                   [sign timing only — HEADLINE]
#   Scaled:     (γ̂_{k,t} / |γ̄_k|) × (Q5 − Q1)              [magnitude timing — leverage pathology]
#   VolManaged: (c_k · γ̂_{k,t} / σ²_{k,t-1}) × (Q5 − Q1)   [Moreira-Muir fix for Scaled]
#   Markowitz:  Σ⁻¹ γ̂_t (gross-1) combined                 [supervisor LLB dashboard method]
#
# Strategy hierarchy: Direction (equal-weight) is the headline; VolManaged repairs the
# Scaled leverage pathology; Markowitz tests Σ⁻¹ aggregation AGAINST equal-weight 1/N.
# References: Moreira & Muir (2017, JF); Markowitz (1952, JF); DeMiguel, Garlappi &
# Uppal (2009, RFS); Ledoit & Wolf (2004, JMVA); Frazzini, Israel & Moskowitz (2015).
#
# Evaluation: annualised Sharpe, FF5 alpha (t-stat), GRS test
#
# Input:  panel_clean.rds, gamma_predictions.rds, ff5_factors.rds
# Output: portfolio_results.rds  (objects: port_ret, port_eval, grs_results, tc_eval,
#         tc_strategy, rafe_results, rank_comparison, Sigma_hat, Sigma_inv,
#         Sigma_ls, Sigma_ls_inv, Sigma_inv_mkw, mkw_burnin)
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(broom)
library(lmtest)
library(sandwich)
library(GRS.test)

panel_clean <- readRDS("panel_clean.rds")        # -> panel_clean
list2env(readRDS("gamma_predictions.rds"), envir = environment())  # -> pred_list, oos_eval, actuals, dates_oos
ff5_factors <- readRDS("ff5_factors.rds")        # -> ff5_factors

dir.create("plots", showWarnings = FALSE)

# Mapping: standardised characteristic column <-> gamma name
char_gamma_map <- tribble(
  ~char_col,            ~gamma_col,
  "bm_std",             "gamma_bm",
  "mom12m_std",         "gamma_mom12m",
  "oper_prof_std",      "gamma_oper_prof",
  "asset_growth_std",   "gamma_asset_growth",
  "log_mktcap_lag_std", "gamma_size",
  "beta_std",           "gamma_beta"
)

char_labels <- c(
  bm_std             = "Value",
  mom12m_std         = "Momentum",
  oper_prof_std      = "Profitability",
  asset_growth_std   = "Asset Growth",
  log_mktcap_lag_std = "Size",
  beta_std           = "Beta"
)


# (1) Historical mean gammas (static benchmark weights) ------------------------
mean_gammas <- setNames(
  colMeans(
    pred_list |> map(\(m) m[, "hist_mean"]) |> do.call(cbind, args = _),
    na.rm = TRUE
  ),
  names(pred_list)
)


# (2) Long-short quintile returns for each characteristic each month -----------
chars_std <- char_gamma_map$char_col

# NOTE (2026-08-27, window fix): the long-short quintile returns are built on
# the FULL 1963-2025 panel, not just the OOS months. They require no forecasts,
# so restricting them to dates_oos threw away 26 years of history that a
# real-time investor demonstrably had. Two conditioning variables read that
# history and were silently truncated by it: the Markowitz Sigma_t (60-month
# burn-in, so weights only began 1995-02) and the trailing variance var_ls_lag
# (so vol-managed only began 1990-07). Both now start with the evaluation
# window like every other strategy. `ls_ret` itself is unchanged (quintiles are
# formed within each month independently); `var_ls_lag` DOES change for the
# first ~12 OOS months, which now use real 1989 data instead of a partial
# in-window window.
ls_returns_full <- panel_clean |>
  group_by(date) |>
  mutate(across(
    all_of(chars_std),
    \(x) ntile(x, 5),
    .names = "{.col}_q"
  )) |>
  ungroup() |>
  pivot_longer(
    cols      = ends_with("_q"),
    names_to  = "char_col",
    values_to = "quintile"
  ) |>
  mutate(char_col = str_remove(char_col, "_q")) |>
  filter(quintile %in% c(1L, 5L)) |>
  group_by(date, char_col, quintile) |>
  summarise(avg_ret = mean(ret_excess, na.rm = TRUE), .groups = "drop") |>
  pivot_wider(names_from = quintile, values_from = avg_ret,
              names_prefix = "q") |>
  mutate(ls_ret = q5 - q1) |>
  select(date, char_col, ls_ret) |>
  # Trailing realised variance of the long-short leg, σ²_{k,t-1}, over the prior
  # 12 months (strictly lagged, window [t-12, t-1], ≥6 obs) — the conditioning
  # variable for Moreira & Muir (2017, JF) volatility management.
  arrange(char_col, date) |>
  group_by(char_col) |>
  mutate(
    var_ls_lag = slider::slide_dbl(
      ls_ret,
      \(x) if (sum(!is.na(x)) >= 6) var(x, na.rm = TRUE) else NA_real_,
      .before = 12, .after = -1
    )
  ) |>
  ungroup()

# OOS slice used for the portfolios themselves (unchanged behaviour).
ls_returns <- ls_returns_full |> filter(date %in% dates_oos)


# (3) Predicted gamma time series (tidy) ---------------------------------------
model_cols <- colnames(pred_list[[1]])

gamma_preds <- map_dfr(names(pred_list), function(gc) {
  as_tibble(pred_list[[gc]]) |>
    mutate(date = dates_oos, gamma_col = gc)
}) |>
  pivot_longer(
    cols      = all_of(model_cols),
    names_to  = "model",
    values_to = "pred_gamma"
  )


# (4) Combine L/S returns with predicted gammas --------------------------------
port_base <- char_gamma_map |>
  left_join(ls_returns,    by = "char_col") |>
  left_join(gamma_preds,   by = c("gamma_col", "date")) |>
  filter(!is.na(ls_ret), !is.na(pred_gamma))

# ── Common evaluation window (2026-07-19, review fix) ─────────────────────────
# Every model's portfolio (and the static benchmark) is evaluated on the
# IDENTICAL months: those where EVERY model has a full 6-gamma forecast and the
# realised gamma vector is complete — the same discipline as oos_eval_common
# and the common-window RAFE (block 12). Without this, feature-free models were
# scored on 421 months and feature-dependent ones on 414, so cross-model tables
# mixed windows and the static Sharpe printed 0.49 or 0.50 depending on which
# model's row it came from. Filtering port_base here propagates the window to
# ALL downstream objects (combined book, Markowitz, GRS, turnover/cost tables).
common_eval <- Reduce(`&`, lapply(names(pred_list), function(gc) {
  rowSums(is.na(pred_list[[gc]])) == 0
})) & complete.cases(actuals)
common_dates <- dates_oos[common_eval]
cat(sprintf("Portfolio common evaluation window: %d months (of %d OOS)\n",
            length(common_dates), length(dates_oos)))
port_base <- port_base |> filter(date %in% common_dates)

# ── Real-time static sign (2026-07-20, critique fix B2) ──────────────────────
# PREVIOUSLY: the static direction was sign(mean_gammas), where mean_gammas is
# the average of the expanding hist_mean series over the ENTIRE OOS window — a
# single scalar per gamma computed with end-of-sample information. That is
# look-ahead IN THE BENCHMARK: an investor in 1990 could not have known it.
#
# NOW: the direction is frozen at the expanding historical mean as of the first
# evaluated month (i.e. information through the month before evaluation starts).
# This is genuinely static — zero trades over the whole window — and strictly
# real-time. Note the expanding-mean sign never flips during the evaluation
# sample (asserted below), so "frozen at evaluation start" and "expanding sign"
# are the SAME portfolio here; the frozen formulation is simply the cleaner
# thing to define in the text.
#
# NOT changed: mean_gammas remains the |γ̄| normaliser for the Scaled overlay.
# That is a pure scale constant (documented at the vol-managed block below),
# Sharpe-neutral per leg, and Scaled is a demoted robustness exhibit — the
# look-ahead objection is about the traded DIRECTION, not the leg scaling.
idx_start <- which(dates_oos == min(common_dates))[1]
static_gamma <- vapply(
  names(pred_list),
  \(gc) pred_list[[gc]][idx_start, "hist_mean"],
  numeric(1)
)

# Verification: does the expanding-mean sign ever flip inside the window?
# If it never does, the frozen and expanding definitions coincide (and the
# historical-mean direction overlay has zero turnover, as Table D.2 reports).
sign_flips <- vapply(names(pred_list), function(gc) {
  s <- sign(pred_list[[gc]][common_eval, "hist_mean"])
  sum(diff(s[!is.na(s)]) != 0)
}, numeric(1))
cat("Expanding-mean sign flips within evaluation window (per gamma):\n")
print(sign_flips)
if (all(sign_flips == 0)) {
  cat("  -> none: frozen-sign and expanding-sign static benchmarks are identical.\n")
} else {
  warning("Expanding-mean sign DOES flip; the footnote claiming the two static ",
          "definitions coincide is no longer valid — revise the text.")
}

# Sign of historical mean gamma determines "positive" direction per characteristic
port_base <- port_base |>
  mutate(
    gamma_hist_mean = mean_gammas[gamma_col],       # scale only (Scaled overlay)
    gamma_static    = static_gamma[gamma_col],      # real-time direction
    # Static: hold in the direction priced as of evaluation start (no timing)
    ret_static      = sign(gamma_static) * ls_ret,
    # Direction timing: flip with predicted gamma sign
    ret_direction   = sign(pred_gamma) * ls_ret,
    # Scaled timing: scale by predicted / |mean| (lever up/down)
    ret_scaled      = (pred_gamma / abs(gamma_hist_mean)) * ls_ret
  )

# Vol-managed timing (Moreira & Muir 2017, JF): replace the |γ̄| normaliser — which
# explodes for near-zero-premium factors (Beta) — with the inverse trailing variance
# of the long-short leg. Position weight w_{k,t} = c_k · γ̂_{k,t} / σ²_{k,t-1}; this is
# also the single-asset mean-variance weight w ∝ μ/σ². The constant c_k sets the managed
# leg to the SAME unconditional volatility as the static leg (Moreira-Muir normalisation),
# so the Sharpe comparison is pure-timing and Sharpe-neutral in c_k. c_k is a full-sample
# scale constant — same status as the existing |γ̄| in Scaled (scale only, not timing).
port_base <- port_base |>
  mutate(vm_raw = (pred_gamma / var_ls_lag) * ls_ret) |>   # pre-scaling managed return
  group_by(model, char_col) |>
  mutate(
    c_vm           = sd(ret_static, na.rm = TRUE) / sd(vm_raw, na.rm = TRUE),
    c_vm           = ifelse(is.finite(c_vm), c_vm, NA_real_),
    w_volmanaged   = c_vm * pred_gamma / var_ls_lag,        # timing weight (for turnover)
    ret_volmanaged = c_vm * vm_raw
  ) |>
  ungroup() |>
  select(-vm_raw)


# (5) Combined portfolio: equal-weight across all characteristics --------------
# Aggregated per model and strategy type
combined <- port_base |>
  group_by(date, model) |>
  summarise(
    ret_static     = mean(ret_static,     na.rm = TRUE),
    ret_direction  = mean(ret_direction,  na.rm = TRUE),
    ret_scaled     = mean(ret_scaled,     na.rm = TRUE),
    ret_volmanaged = mean(ret_volmanaged, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(char_col = "COMBINED")


# (5b) Markowitz Σ⁻¹ cross-factor aggregation ----------------------------------
# The supervisor's LLB dashboard combines the per-factor timing signals via a
# "post-hoc Markowitz mapping on the factor covariance matrix" rather than equal
# weighting. We replicate it: w_t = Σ_t⁻¹ γ̂_t, gross-normalised to |w|₁ = 1, where
# Σ_t is the covariance of the six long-short factor returns. Σ⁻¹ both risk-scales
# (diagonal — fixes the leverage pathology) and de-correlates (off-diagonal) the
# bets. Reported AGAINST equal-weight Direction as a robustness exhibit: DeMiguel,
# Garlappi & Uppal (2009, RFS) show Σ⁻¹ estimation error often lets naive 1/N win
# out of sample, so this TESTS — it does not replace — the equal-weight headline.
#
# Real-time estimation (no look-ahead): Σ_t is estimated on an EXPANDING window of
# LS returns strictly before t (≥ MKW_BURNIN months; earlier months get NA weights,
# consistent with the honest-NA treatment elsewhere). Ledoit-Wolf (2004) shrinkage
# toward the scaled identity keeps Σ_t⁻¹ well-conditioned in the early window.
# References: Markowitz (1952, JF); DeMiguel, Garlappi & Uppal (2009, RFS);
#             Ledoit & Wolf (2004, J. Multivariate Analysis).
gcols_ord <- char_gamma_map$gamma_col

# Built from ls_returns_full (1963-2025), NOT from port_base: Sigma_t needs only
# realised long-short returns, never a forecast, so it can and should use the
# pre-1990 history. Sourcing it from the common-window port_base left the
# 60-month burn-in biting inside the evaluation window (first weight 1995-02),
# which put Markowitz on 354 months while every other strategy had 414.
ls_by_gamma <- ls_returns_full |>
  inner_join(dplyr::select(char_gamma_map, char_col, gamma_col), by = "char_col") |>
  distinct(date, gamma_col, ls_ret) |>
  pivot_wider(names_from = gamma_col, values_from = ls_ret) |>
  arrange(date)

# Full-sample Σ retained for reference/reporting ONLY — weights use Sigma_inv_mkw.
# Kept on the EVALUATION window so its meaning is unchanged by the full-history
# switch above (it is saved for reference only and feeds nothing downstream).
Sigma_ls     <- cov(
  as.matrix(ls_by_gamma[ls_by_gamma$date %in% common_dates, gcols_ord]),
  use = "pairwise.complete.obs"
)
Sigma_ls_inv <- tryCatch(
  solve(Sigma_ls),
  error = function(e) {
    cat("WARNING: Sigma_ls singular — using pseudo-inverse (MASS::ginv)\n")
    MASS::ginv(Sigma_ls)
  }
)

MKW_BURNIN <- 60L   # months of LS-return history required before first weight

# Ledoit & Wolf (2004): shrink the sample covariance toward mu·I with the analytic
# intensity rho = b² / d², where d² measures S's dispersion around the target and
# b² the sampling noise in S. Guarantees a well-conditioned, invertible estimate.
lw_shrink_cov <- function(X) {
  n  <- nrow(X); p <- ncol(X)
  S  <- cov(X)
  mu <- sum(diag(S)) / p
  d2 <- sum((S - diag(mu, p))^2)
  Xc <- sweep(X, 2L, colMeans(X))
  b2bar <- sum(vapply(
    seq_len(n), \(i) sum((tcrossprod(Xc[i, ]) - S)^2), numeric(1L)
  )) / n^2
  rho <- if (d2 > 1e-12) min(b2bar, d2) / d2 else 1
  rho * diag(mu, p) + (1 - rho) * S
}

# Per-date expanding Σ_t⁻¹: rows 1:(i-1) of ls_by_gamma (strictly before date i).
ls_mat_all    <- as.matrix(ls_by_gamma[, gcols_ord])
mkw_dates     <- ls_by_gamma$date
Sigma_inv_mkw <- setNames(vector("list", length(mkw_dates)), as.character(mkw_dates))
for (di in seq_along(mkw_dates)) {
  H <- ls_mat_all[seq_len(di - 1L), , drop = FALSE]
  H <- H[complete.cases(H), , drop = FALSE]
  if (nrow(H) >= MKW_BURNIN) {
    Sh <- lw_shrink_cov(H)
    Sigma_inv_mkw[[di]] <- tryCatch(solve(Sh), error = function(e) MASS::ginv(Sh))
  }   # else NULL → burn-in month → NA weights below
}
# Report availability WITHIN the evaluation window - that is the number that
# determines whether Markowitz is comparable with the other strategies.
mkw_ok_eval <- sum(!vapply(Sigma_inv_mkw[as.character(common_dates)],
                           is.null, logical(1L)))
cat(sprintf(
  "Markowitz: expanding LW Sigma available for %d/%d evaluation months (burn-in %d, history from %s)\n",
  mkw_ok_eval, length(common_dates), MKW_BURNIN, format(min(mkw_dates))
))
if (mkw_ok_eval < length(common_dates)) {
  warning(sprintf(
    "Markowitz covers %d of %d evaluation months - Table 5 would mix windows.",
    mkw_ok_eval, length(common_dates)))
}

# Per (model, month): w = Σ_t⁻¹ γ̂, gross-1 normalised. NA whenever the month lacks
# a full 6-gamma prediction or is inside the Σ burn-in (honest real-time treatment).
markowitz_w <- port_base |>
  select(model, date, gamma_col, pred_gamma) |>
  group_by(model, date) |>
  group_modify(function(d, k) {
    d    <- d[match(gcols_ord, d$gamma_col), ]
    Sinv <- Sigma_inv_mkw[[as.character(k$date)]]
    if (is.null(Sinv) || anyNA(d$pred_gamma)) {
      return(tibble(gamma_col = gcols_ord, w = NA_real_))
    }
    w <- as.numeric(Sinv %*% d$pred_gamma)
    g <- sum(abs(w))
    if (!is.finite(g) || g < 1e-12) return(tibble(gamma_col = gcols_ord, w = NA_real_))
    tibble(gamma_col = gcols_ord, w = w / g)
  }) |>
  ungroup() |>
  left_join(distinct(port_base, date, gamma_col, ls_ret), by = c("date", "gamma_col"))

markowitz_combined <- markowitz_w |>
  group_by(model, date) |>
  summarise(ret_markowitz = sum(w * ls_ret), .groups = "drop")

combined <- combined |>
  left_join(markowitz_combined, by = c("model", "date"))


port_ret <- bind_rows(
  port_base |>
    select(date, model, char_col, ret_static, ret_direction, ret_scaled, ret_volmanaged) |>
    mutate(ret_markowitz = NA_real_),
  combined |>
    select(date, model, char_col, ret_static, ret_direction, ret_scaled,
           ret_volmanaged, ret_markowitz)
)

# (port_ret inherits the common evaluation window via the port_base filter above)


# (6) Rename FF5 for local use, and join the momentum factor for FF6 -----------
# umd_factor.rds is produced by 01e_momentum_factor.R from the French library.
# A left join keeps every FF5 month even if UMD were short; ff6_alpha() drops
# NA-umd rows itself, so the FF5 column is never silently re-windowed.
ff5 <- ff5_factors |>
  left_join(readRDS("umd_factor.rds"), by = "date")
stopifnot(!anyNA(ff5$umd[ff5$date %in% dates_oos]))


# (7) Evaluation functions -----------------------------------------------------
sharpe_annual <- function(r) {
  r <- r[!is.na(r)]
  if (length(r) < 12) return(NA_real_)
  mean(r) / sd(r) * sqrt(12)
}

vol_annual <- function(r) {
  r <- r[!is.na(r)]
  if (length(r) < 2) return(NA_real_)
  sd(r) * sqrt(12)
}

max_drawdown <- function(r) {
  r <- r[!is.na(r)]
  if (length(r) < 2) return(NA_real_)
  cum <- cumprod(1 + r)
  peak <- cummax(cum)
  dd   <- (cum - peak) / peak
  min(dd)   # most negative value
}

calmar_ratio <- function(r) {
  r   <- r[!is.na(r)]
  if (length(r) < 12) return(NA_real_)
  ann_ret <- mean(r) * 12
  mdd     <- abs(max_drawdown(r))
  if (mdd == 0) return(NA_real_)
  ann_ret / mdd
}

sortino_ratio <- function(r, mar = 0) {
  r <- r[!is.na(r)]
  if (length(r) < 12) return(NA_real_)
  downside <- r[r < mar] - mar
  if (length(downside) == 0) return(NA_real_)
  dd_sd <- sqrt(mean(downside^2))
  if (is.na(dd_sd) || dd_sd == 0) return(NA_real_)
  (mean(r) - mar) / dd_sd * sqrt(12)
}

ff5_alpha <- function(r, dates, ff) {
  df <- tibble(date = dates, ret = r) |>
    inner_join(ff, by = "date") |>
    filter(!is.na(ret))
  if (nrow(df) < 24) return(tibble(alpha = NA, t_alpha = NA, p_alpha = NA))
  fit <- lm(ret ~ mkt_excess + smb + hml + rmw + cma, data = df)
  ct  <- coeftest(fit, vcov = NeweyWest(fit, lag = 6, prewhite = FALSE))
  tibble(
    alpha   = ct["(Intercept)", "Estimate"] * 1200,   # annualised %
    t_alpha = ct["(Intercept)", "t value"],
    p_alpha = ct["(Intercept)", "Pr(>|t|)"]
  )
}

# Six-factor alpha (Fama & French 2018, JFE): FF5 plus the momentum factor.
# The combined book trades a momentum leg, and the FF5 cannot price that leg on
# its own (the static momentum leg earns a large FF5 alpha in its own right), so
# an FF5 alpha on the combined book partly measures an omitted-factor loading
# rather than timing skill. Adding UMD removes that channel; the gap between the
# two alphas is the part of the FF5 figure that momentum exposure explains.
ff6_alpha <- function(r, dates, ff) {
  df <- tibble(date = dates, ret = r) |>
    inner_join(ff, by = "date") |>
    filter(!is.na(ret), !is.na(umd))
  if (nrow(df) < 24) return(tibble(alpha6 = NA, t_alpha6 = NA, p_alpha6 = NA,
                                   b_umd = NA, t_umd = NA))
  fit <- lm(ret ~ mkt_excess + smb + hml + rmw + cma + umd, data = df)
  ct  <- coeftest(fit, vcov = NeweyWest(fit, lag = 6, prewhite = FALSE))
  tibble(
    alpha6   = ct["(Intercept)", "Estimate"] * 1200,   # annualised %
    t_alpha6 = ct["(Intercept)", "t value"],
    p_alpha6 = ct["(Intercept)", "Pr(>|t|)"],
    b_umd    = ct["umd", "Estimate"],                  # momentum loading
    t_umd    = ct["umd", "t value"]
  )
}


# (8) Compute evaluation metrics -----------------------------------------------
strategy_types <- c("ret_static", "ret_direction", "ret_scaled",
                    "ret_volmanaged", "ret_markowitz")

port_eval <- port_ret |>
  pivot_longer(
    cols      = all_of(strategy_types),
    names_to  = "strategy",
    values_to = "ret"
  ) |>
  group_by(char_col, model, strategy) |>
  group_modify(\(d, k) {
    r     <- d$ret
    dates <- d$date
    sr    <- sharpe_annual(r)
    alphas  <- ff5_alpha(r, dates, ff5)
    alphas6 <- ff6_alpha(r, dates, ff5)
    tibble(
      sharpe    = round(sr, 3),
      vol_pct   = round(vol_annual(r) * 100, 3),            # annualised vol %
      max_dd    = round(max_drawdown(r) * 100, 2),          # max drawdown %
      calmar    = round(calmar_ratio(r), 3),
      sortino   = round(sortino_ratio(r), 3),
      alpha_pct = round(alphas$alpha,   3),
      t_alpha   = round(alphas$t_alpha, 2),
      p_alpha   = round(alphas$p_alpha, 3),
      alpha6_pct = round(alphas6$alpha6,   3),              # FF5 + UMD
      t_alpha6   = round(alphas6$t_alpha6, 2),
      p_alpha6   = round(alphas6$p_alpha6, 3),
      b_umd      = round(alphas6$b_umd, 3),                 # momentum loading
      t_umd      = round(alphas6$t_umd, 2),
      mean_ret  = round(mean(r, na.rm = TRUE) * 1200, 3),   # annualised %
      n_months  = sum(!is.na(r))
    )
  }) |>
  ungroup() |>
  filter(n_months > 0) |>   # drop empty rows (e.g. markowitz at per-characteristic level)
  mutate(
    strategy  = str_remove(strategy, "^ret_"),
    char_label = ifelse(char_col == "COMBINED", "COMBINED",
                        char_labels[char_col])
  )

cat("\n=== Portfolio Evaluation: COMBINED strategy ===\n")
port_eval |>
  filter(char_col == "COMBINED") |>
  select(model, strategy, sharpe, alpha_pct, t_alpha, mean_ret) |>
  arrange(strategy, desc(sharpe)) |>
  print(n = Inf)


# (8b) H=3 quarterly-rebalanced direction timing (Ext-C portfolio wiring) ------
# The H=3 forecast at month t predicts the cumulative gamma over [t+1, t+2, t+3]
# (verified empirically: actuals_h3[t] = Σ γ_{t+1..t+3}). The matched-horizon
# portfolio sets the position sign from that forecast and HOLDS it for the next
# quarter, rebalancing every 3 months — NOT dividing the signal by 3, which would
# assume linear return decay and ignore √h volatility scaling (Lo & MacKinlay 1988,
# JPE). Sign timing is horizon-invariant, so no magnitude rescaling is required.
# Matching rebalancing to the forecast horizon also slashes turnover. Scaled/magnitude
# timing at H=3 is intentionally omitted (it is demoted at H=1 and would need √3 vol
# rescaling). References: Fama & French (1988, JFE); Hodrick (1992, RFS);
# Frazzini, Israel & Moskowitz (2015).
h3 <- readRDS("gamma_predictions_h3.rds")
pred_list_h3 <- h3$pred_list_h3
dates_h3     <- h3$dates_oos_h3
models_h3    <- colnames(pred_list_h3[[1]])

# Tidy H=3 predictions: (date, gamma_col, model, pred_h3)
gamma_preds_h3 <- map_dfr(names(pred_list_h3), function(gc) {
  as_tibble(pred_list_h3[[gc]]) |>
    mutate(date = dates_h3, gamma_col = gc)
}) |>
  pivot_longer(all_of(models_h3), names_to = "model", values_to = "pred_h3")

# Non-overlapping quarterly schedule: rebalance at index i, hold months i+1..i+3.
N_h3    <- length(dates_h3)
reb_idx <- seq.int(1L, N_h3 - 3L, by = 3L)

# Common quarterly schedule. A model that cannot forecast at a rebalance date
# takes no position for that whole quarter, so an unfiltered schedule would score
# the feature-dependent models (LASSO/XGB, which skip the Goyal-Welch-tail months)
# on 414 held months against 417 for the AR/combination set — the same mixed-window
# defect the H=1 `port_base` filter and the 04 Ext-C 5b common mask remove. Keep
# only rebalance dates at which EVERY model has a forecast for EVERY gamma, so all
# models are compared on an identical set of held months (417 -> 414; one quarter,
# anchored 2024-07, is dropped for everyone).
reb_ok  <- Reduce(`&`, lapply(pred_list_h3, \(P) rowSums(is.na(P)) == 0))
reb_idx <- reb_idx[reb_ok[reb_idx]]
stopifnot(length(reb_idx) >= 100L)
quarter_map <- map_dfr(reb_idx, \(i)
  tibble(reb_date = dates_h3[i], held_date = dates_h3[(i + 1L):(i + 3L)]))

ls_by_gc <- char_gamma_map |>
  left_join(ls_returns, by = "char_col") |>
  select(gamma_col, date, ls_ret)

# Direction returns over each held month, positions refreshed quarterly
h3_dir <- gamma_preds_h3 |>
  inner_join(quarter_map, by = c("date" = "reb_date")) |>
  inner_join(ls_by_gc, by = c("gamma_col", "held_date" = "date")) |>
  mutate(ret_dir_h3 = sign(pred_h3) * ls_ret)

# Combined equal-weight monthly series per model, then Sharpe + FF5 alpha
h3_combined <- h3_dir |>
  group_by(model, held_date) |>
  summarise(ret = mean(ret_dir_h3, na.rm = TRUE), .groups = "drop") |>
  rename(date = held_date)

h3_eval <- h3_combined |>
  group_by(model) |>
  group_modify(\(d, k) {
    a <- ff5_alpha(d$ret, d$date, ff5)
    tibble(sharpe_h3q = round(sharpe_annual(d$ret), 3),
           alpha_h3q  = round(a$alpha,   3),
           t_h3q      = round(a$t_alpha, 2),
           n_months   = sum(!is.na(d$ret)))
  }) |>
  ungroup()

# H=1 monthly direction (combined) for the same models, for side-by-side comparison
h1_dir_combined <- port_ret |>
  filter(char_col == "COMBINED") |>
  group_by(model) |>
  summarise(sharpe_h1 = round(sharpe_annual(ret_direction), 3), .groups = "drop")

h3_vs_h1 <- h3_eval |>
  left_join(h1_dir_combined, by = "model") |>
  select(model, sharpe_h1, sharpe_h3q, alpha_h3q, t_h3q, n_months) |>
  arrange(desc(sharpe_h3q))

static_sr_combined <- port_eval |>
  filter(char_col == "COMBINED", strategy == "static", model == "hist_mean") |>
  pull(sharpe)

cat("\n=== H=3 quarterly-rebalanced vs H=1 monthly direction timing (COMBINED) ===\n")
cat(sprintf("Static benchmark Sharpe = %.3f. Positions refreshed quarterly from the\n",
            static_sr_combined))
cat("H=3 forecast; sign timing only (horizon-invariant, no divide-by-3).\n\n")
print(h3_vs_h1, n = Inf)


# (9) GRS test on characteristic portfolios per model/strategy -----------------
# Markowitz is a combined-only strategy (no per-characteristic portfolios) → excluded.
grs_strategies <- setdiff(strategy_types, "ret_markowitz")
grs_results <- tidyr::expand_grid(
    mod   = unique(port_ret$model),
    strat = grs_strategies
  ) |>
  purrr::pmap_dfr(function(mod, strat) {

    ret_wide <- port_ret |>
      filter(model == mod, char_col != "COMBINED") |>
      select(date, char_col, all_of(strat)) |>
      pivot_wider(names_from = char_col, values_from = all_of(strat)) |>
      inner_join(ff5, by = "date") |>
      filter(complete.cases(across(everything())))

    if (nrow(ret_wide) < 30) return(NULL)

    port_mat   <- ret_wide |> select(all_of(chars_std)) |> as.matrix()
    factor_mat <- ret_wide |> select(mkt_excess, smb, hml, rmw, cma) |> as.matrix()

    tryCatch({
      grs_out <- GRS.test(port_mat, factor_mat)
      tibble(
        model    = mod,
        strategy = str_remove(strat, "^ret_"),
        grs_f    = round(grs_out$GRS.stat, 3),
        grs_p    = round(grs_out$GRS.pval, 3),
        n_months = nrow(port_mat)
      )
    }, error = function(e) NULL)
  })

cat("\n=== GRS Test: H0 = all FF5 alphas jointly zero ===\n")
print(grs_results |> arrange(strategy, grs_p), n = Inf)


# (10) Plots -------------------------------------------------------------------
# Note: the cumulative-returns plot and the new-strategy / cost / horizon plots are
# produced in block (12c) below, after RAFE is available, so the headline timed model
# is selected by RAFE (consistent with Table 4 in 06_tables_export.R).

# Sharpe comparison across models for COMBINED direction strategy
p_sharpe <- port_eval |>
  filter(char_col == "COMBINED", strategy == "direction") |>
  mutate(model = fct_reorder(model, sharpe)) |>
  ggplot(aes(x = model, y = sharpe, fill = sharpe > 0)) +
  geom_col(width = 0.6) +
  geom_hline(
    data = port_eval |> filter(char_col == "COMBINED", strategy == "static",
                                model == "hist_mean"),
    aes(yintercept = sharpe), linetype = "dashed", colour = "grey40"
  ) +
  scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "firebrick"),
                    guide = "none") +
  labs(
    title    = "Annualised Sharpe Ratio by Prediction Model (Direction Timing, Combined)",
    subtitle = "Dashed line = static benchmark",
    x = "Model", y = "Sharpe Ratio"
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("plots/sharpe_by_model.pdf", p_sharpe, width = 8, height = 4)
cat("Saved: plots/sharpe_by_model.pdf\n")

# Risk metrics comparison: Sharpe, Max Drawdown, Calmar — COMBINED direction
p_risk <- port_eval |>
  filter(char_col == "COMBINED", strategy == "direction") |>
  select(model, sharpe, max_dd, calmar, sortino) |>
  pivot_longer(-model, names_to = "metric", values_to = "value") |>
  mutate(
    metric = factor(metric,
                    levels = c("sharpe", "sortino", "calmar", "max_dd"),
                    labels = c("Sharpe Ratio", "Sortino Ratio",
                               "Calmar Ratio", "Max Drawdown (%)")),
    model  = fct_reorder(model, value, .fun = first, .desc = TRUE)
  ) |>
  ggplot(aes(x = model, y = value, fill = value > 0)) +
  geom_col(width = 0.6) +
  facet_wrap(~metric, scales = "free_y", ncol = 2) +
  scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "firebrick"),
                    guide = "none") +
  labs(
    title    = "Risk-Adjusted Performance Metrics by Model (Direction Timing, Combined)",
    subtitle = "Max Drawdown shown as negative percentage",
    x = "Model", y = NULL
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7))

ggsave("plots/risk_metrics.pdf", p_risk, width = 10, height = 7)
cat("Saved: plots/risk_metrics.pdf\n")

# Alpha heatmap: char × model for direction strategy
p_alpha <- port_eval |>
  filter(strategy == "direction", char_col != "COMBINED") |>
  mutate(
    sig = case_when(p_alpha < 0.01 ~ "***", p_alpha < 0.05 ~ "**",
                    p_alpha < 0.10 ~ "*",   TRUE ~ ""),
    label = paste0(round(alpha_pct, 1), sig)
  ) |>
  ggplot(aes(x = model, y = char_label, fill = alpha_pct)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = label), size = 2.4) +
  scale_fill_gradient2(
    low = "firebrick", mid = "white", high = "steelblue", midpoint = 0,
    name = "FF5 Alpha\n(ann. %)"
  ) +
  labs(title = "FF5 Alpha by Model and Characteristic (Direction Timing)",
       subtitle = "*** p<0.01, ** p<0.05, * p<0.10 (Newey-West SE)",
       x = "Prediction Model", y = NULL) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("plots/alpha_heatmap.pdf", p_alpha, width = 12, height = 5)
cat("Saved: plots/alpha_heatmap.pdf\n")


# (11) Transaction cost analysis -----------------------------------------------
# The scaled strategy levers the long-short leg by a time-varying timing weight
# w_t = pred_gamma / |hist_mean|, so its month-to-month turnover is |w_t - w_{t-1}|;
# direction timing turns over when the predicted sign flips; static timing holds a
# constant weight (≈ zero overlay turnover). We therefore charge cost proportional
# to EMPIRICAL turnover as the primary net-of-cost metric (so high- and low-turnover
# strategies are distinguished), and report a fixed 100%-turnover charge only as an
# illustrative reference. Note the scaled strategy's leverage can swing by more than
# one unit per month, so empirical turnover can exceed 100% — the flat charge is not
# a strict upper bound, which is precisely why the empirical measure is the honest one.
#   TC_10bp = 10 bps round-trip (large-cap, liquid stocks)
#   TC_50bp = 50 bps round-trip (full CRSP universe incl. microcaps)
# Turnover here is the TIMING-OVERLAY turnover; the underlying long-short quintile
# turnover is common to all three strategies and is not separately modelled.
# Reference: Frazzini, Israel & Moskowitz (2015); Novy-Marx & Velikov (2016, RFS).

TC_10BP <- 0.0010   # 10 basis points
TC_50BP <- 0.0050   # 50 basis points

# Empirical timing-overlay turnover per (model, characteristic): |w_t - w_{t-1}|,
# averaged across characteristics to the combined-portfolio level each month.
turnover_combined <- port_base |>
  mutate(
    w_scaled    = pred_gamma / abs(gamma_hist_mean),
    w_direction = sign(pred_gamma)
    # w_volmanaged is already computed in port_base
  ) |>
  arrange(model, char_col, date) |>
  group_by(model, char_col) |>
  mutate(
    to_scaled     = abs(w_scaled     - dplyr::lag(w_scaled)),
    to_direction  = abs(w_direction  - dplyr::lag(w_direction)),
    to_volmanaged = abs(w_volmanaged - dplyr::lag(w_volmanaged))
  ) |>
  group_by(model, date) |>
  summarise(to_scaled     = mean(to_scaled,     na.rm = TRUE),
            to_direction  = mean(to_direction,  na.rm = TRUE),
            to_volmanaged = mean(to_volmanaged, na.rm = TRUE),
            .groups = "drop")

# Markowitz overlay turnover: gross weight change Σ_k |w_{k,t} − w_{k,t-1}| per month.
markowitz_turnover <- markowitz_w |>
  arrange(model, gamma_col, date) |>
  group_by(model, gamma_col) |>
  mutate(dw = abs(w - dplyr::lag(w))) |>
  group_by(model, date) |>
  summarise(to_markowitz = sum(dw, na.rm = TRUE), .groups = "drop")

tc_eval <- combined |>
  left_join(turnover_combined, by = c("model", "date")) |>
  group_by(model) |>
  summarise(
    avg_turnover_scaled = round(mean(to_scaled, na.rm = TRUE), 3),  # mean monthly |Δ leverage|
    sharpe_scaled_gross = sharpe_annual(ret_scaled),
    sharpe_static_gross = sharpe_annual(ret_static),
    # PRIMARY: net Sharpe charging TC × realised turnover each month
    sharpe_scaled_10bp  = sharpe_annual(ret_scaled - TC_10BP * to_scaled),
    sharpe_scaled_50bp  = sharpe_annual(ret_scaled - TC_50BP * to_scaled),
    # REFERENCE: fixed 100%-turnover charge (illustrative, not a strict bound)
    sharpe_scaled_10bp_flat = sharpe_annual(ret_scaled - TC_10BP),
    sharpe_scaled_50bp_flat = sharpe_annual(ret_scaled - TC_50BP),
    # Breakeven: mean monthly scaled-vs-static outperformance (bps)
    breakeven_bp = round((mean(ret_scaled, na.rm = TRUE) -
                          mean(ret_static, na.rm = TRUE)) * 10000, 1),
    .groups = "drop"
  ) |>
  mutate(across(starts_with("sharpe_"), \(x) round(x, 3)))

cat("\n=== Transaction Cost Analysis: COMBINED Scaled vs Static ===\n")
cat("Primary net Sharpe charges TC × empirical turnover; _flat = fixed 100%-turnover reference.\n")
cat("avg_turnover_scaled = mean monthly |Δ leverage| of the timing overlay (can exceed 1).\n\n")
print(tc_eval |> arrange(desc(sharpe_scaled_gross)), n = Inf)

cat("\nbreakeven_bp = mean monthly scaled-over-static outperformance (bps); the strategy is\n")
cat("viable net of costs when this exceeds the per-trade cost × its turnover.\n")


# (11b) Net-of-cost Sharpe across all timed strategies (COMBINED) --------------
# Direction's structurally low turnover (positions only flip on a predicted sign
# change) is the MECHANISM by which it survives costs where magnitude-timing does
# not. Frazzini, Israel & Moskowitz (2015): turnover drag is the primary
# determinant of net factor-timing profitability. Costs charged as TC × empirical
# timing-overlay turnover each month (same convention as block 11).
tc_panel <- combined |>
  left_join(turnover_combined, by = c("model", "date")) |>
  left_join(markowitz_turnover, by = c("model", "date"))

tc_strategy <- tc_panel |>
  group_by(model) |>
  summarise(
    direction_to     = mean(to_direction,  na.rm = TRUE),
    direction_gross  = sharpe_annual(ret_direction),
    direction_10bp   = sharpe_annual(ret_direction  - TC_10BP * to_direction),
    direction_50bp   = sharpe_annual(ret_direction  - TC_50BP * to_direction),
    scaled_to        = mean(to_scaled,     na.rm = TRUE),
    scaled_gross     = sharpe_annual(ret_scaled),
    scaled_10bp      = sharpe_annual(ret_scaled     - TC_10BP * to_scaled),
    scaled_50bp      = sharpe_annual(ret_scaled     - TC_50BP * to_scaled),
    volmanaged_to    = mean(to_volmanaged, na.rm = TRUE),
    volmanaged_gross = sharpe_annual(ret_volmanaged),
    volmanaged_10bp  = sharpe_annual(ret_volmanaged - TC_10BP * to_volmanaged),
    volmanaged_50bp  = sharpe_annual(ret_volmanaged - TC_50BP * to_volmanaged),
    markowitz_to     = mean(to_markowitz,  na.rm = TRUE),
    markowitz_gross  = sharpe_annual(ret_markowitz),
    markowitz_10bp   = sharpe_annual(ret_markowitz  - TC_10BP * to_markowitz),
    markowitz_50bp   = sharpe_annual(ret_markowitz  - TC_50BP * to_markowitz),
    .groups = "drop"
  ) |>
  mutate(across(where(is.numeric), \(x) round(x, 3)))

cat("\n=== Net-of-cost Sharpe by timed strategy (COMBINED) ===\n")
cat("*_to = mean monthly timing-overlay turnover; *_50bp = net of 50 bps × turnover.\n")
cat("Direction's low turnover is the mechanism for net-of-cost survival vs Scaled.\n\n")
print(tc_strategy |> arrange(desc(direction_gross)), n = Inf)


# ==============================================================================
# (12) RAFE / C-RAFE — Risk-Adjusted Mean Forecast Error (Ext-D)
# Supervisor extension confirmed 2026-04-18.
#
# RAFE = Mahalanobis distance between predicted and realised gamma vectors:
#   RAFE_m = sqrt( (1/T) Σ_t (γ̂_t − γ_t)' Σ^{-1} (γ̂_t − γ_t) )   [Eq. 16]
#
# Key advantage over OOS R²: accounts for cross-gamma covariances. A model can
# have positive OOS R² on each gamma independently yet still be far from the
# tangency portfolio if it systematically mispredicts cross-factor correlations.
# Σ^{-1} weights errors more heavily in directions that damage the tangency portfolio.
#
# C-RAFE (covariance error component) = RAFE_full − RAFE_diagonal:
#   RAFE_full: uses full Σ^{-1} (variance + covariance structure)
#   RAFE_diag: uses diag(Σ)^{-1} only (ignores cross-gamma covariance)
#   C-RAFE: portion of forecast error attributable to covariance misspecification
#
# IMPORTANT: RAFE is used only in evaluation/reporting — NOT as a training loss.
#
# References:
#   Salcher, Stöckl & Hanke (2026, "Lost in Translation") — RAFE, C-RAFE definition.
#   Stöckl & Hanke (2014) — Mahalanobis distance as multivariate forecast metric.
# ==============================================================================

# Step 1 — build T_OOS × 6 matrix of realised gammas (column order = gamma_cols)
gamma_cols  <- colnames(actuals)   # e.g. gamma_bm, gamma_mom12m, ...
actuals_mat <- actuals[, gamma_cols, drop = FALSE]

# Step 1b — joint common evaluation window (2026-07-19, review fix):
# RAFE ranks models for the ex-ante headline selection, so every model must be
# scored on the IDENTICAL months (same discipline as oos_eval_common / §meth-
# missing). Previously feature-free models were scored on 421 months and
# feature-dependent ones on 414, mixing windows inside a selection-relevant
# table. Common window = months where EVERY model has a full 6-gamma forecast
# and the realised gamma vector is complete.
all_models_rafe <- unique(unlist(lapply(pred_list, colnames)))
rafe_common <- Reduce(`&`, lapply(names(pred_list), function(gc) {
  rowSums(is.na(pred_list[[gc]])) == 0
})) & complete.cases(actuals[, colnames(actuals), drop = FALSE])
cat(sprintf("RAFE common evaluation window: %d months (of %d OOS)\n",
            sum(rafe_common), length(rafe_common)))

# Step 2 — estimate Σ from the realised gammas on the common window (sample
# covariance; OOS period only — no look-ahead into pre-1990 training data).
complete_rows <- which(rafe_common)
Sigma_hat     <- cov(actuals_mat[complete_rows, ])   # 6×6 gamma covariance
Sigma_inv     <- tryCatch(
  solve(Sigma_hat),
  error = function(e) {
    cat("WARNING: Sigma_hat is singular — using pseudo-inverse (MASS::ginv)\n")
    MASS::ginv(Sigma_hat)
  }
)
# Diagonal precision matrix: used to decompose RAFE into variance vs covariance components
Sigma_diag_inv <- diag(1 / diag(Sigma_hat))   # 6×6 diagonal matrix, zeros off-diagonal

rafe_results <- map_dfr(all_models_rafe, function(mod) {
  # Build T_OOS × 6 prediction matrix for this model
  pred_mat <- do.call(cbind, lapply(gamma_cols, function(gc) {
    col <- pred_list[[gc]]
    if (mod %in% colnames(col)) col[, mod] else rep(NA_real_, nrow(col))
  }))

  errors  <- pred_mat - actuals_mat        # T_OOS × 6 forecast errors
  valid_t <- which(rafe_common & complete.cases(errors)) # identical window ∀ models
  if (length(valid_t) < 10L) return(NULL)

  E <- errors[valid_t, , drop = FALSE]

  # RAFE: Risk-Adjusted Mean Forecast Error (Salcher, Stöckl & Hanke 2026, Eq. 16)
  # Formula: RAFE = sqrt((1/T) * Σ_t e_t' Σ^{-1} e_t)
  # Analogous to RMSE: square root of the mean squared Mahalanobis distance.
  # Lower = better. Comparable across models and to the paper's simulation benchmarks.
  mahal_t  <- apply(E, 1L, function(e) as.numeric(t(e) %*% Sigma_inv      %*% e))
  mahal_d_t <- apply(E, 1L, function(e) as.numeric(t(e) %*% Sigma_diag_inv %*% e))

  # RAFE-CC0 (paper's Eq. 18): diagonal Σ^{-1} only — ignores cross-gamma covariance.
  # Equivalent to RAFE under the assumption that all gamma correlations = 0.
  # Difference RAFE - RAFE_CC0 quantifies how much cross-covariance structure matters.
  # Note: the paper's C-RAFE (Eq. 21) measures covariance *estimation* error
  # (||Σ^{1/2} Σ̂^{-1} Σ^{1/2} - I||_2) and is not directly applicable here since
  # we use a single Σ estimate for evaluation, not a separate forecast of Σ.

  tibble(
    model    = mod,
    RAFE     = round(sqrt(mean(mahal_t)),   4L),  # full Mahalanobis RAFE (lower = better)
    RAFE_CC0 = round(sqrt(mean(mahal_d_t)), 4L),  # diagonal-only (RAFE-CC0 in paper, ρ=0)
    RMSE     = round(sqrt(mean(E^2)),        4L),  # RAFE-I in paper (Σ=I, ignores all risk)
    cov_contribution = round(sqrt(mean(mahal_t)) - sqrt(mean(mahal_d_t)), 4L),
    n_oos    = length(valid_t)
  )
}) |>
  arrange(RAFE)

cat("\n=== RAFE by Prediction Model (Salcher, Stöckl & Hanke 2026) ===\n")
cat("RAFE     = sqrt(mean(e' Σ^{-1} e)) — full Mahalanobis, lower is better\n")
cat("RAFE_CC0 = diagonal Σ^{-1} only (ignores cross-gamma covariance, paper Eq.18)\n")
cat("RMSE     = Σ=I, no risk adjustment (RAFE-I in paper, standard benchmark)\n")
cat("cov_contribution = RAFE - RAFE_CC0: cost of cross-covariance in forecast errors\n\n")
print(rafe_results, n = Inf)

# Compare: does RAFE model ranking agree with Sharpe ratio ranking?
# Reference: Salcher et al. (2026) — RAFE should be better aligned with Sharpe than OOS R².
# Uses the DIRECTION strategy (the thesis's primary economic-value measure;
# switched from scaled 2026-07-19 — scaled is the demoted robustness exhibit,
# so basing the headline comparison stat on it was incongruent).
sharpe_rank <- port_eval |>
  filter(char_col == "COMBINED", strategy == "direction") |>
  select(model, sharpe) |>
  arrange(desc(sharpe)) |>
  mutate(sharpe_rank = row_number())

rafe_rank <- rafe_results |>
  select(model, RAFE) |>
  mutate(rafe_rank = row_number())   # already sorted ascending by RAFE

rank_comparison <- inner_join(sharpe_rank, rafe_rank, by = "model") |>
  mutate(rank_diff = abs(sharpe_rank - rafe_rank)) |>
  arrange(sharpe_rank)

cat("\n=== RAFE vs Sharpe Rank Comparison ===\n")
cat("rank_diff = |Sharpe rank − RAFE rank|; 0 = perfect agreement\n")
print(rank_comparison, n = Inf)


# (12b) Expanding-window Σ RAFE — robustness to full-sample Σ look-ahead --------
# The headline RAFE uses a full-OOS-sample Σ, which embeds future covariance
# information in the evaluation weight. Σ is common to all models, so it cannot
# bias the cross-model RANKING, but to close the door we recompute RAFE with a
# strictly expanding Σ_t estimated from realised gammas through t-1 only, and
# confirm the model ranking is unchanged (Spearman rank correlation).
MIN_SIGMA <- 36L                       # min months for a stable 6x6 covariance
T_oos     <- nrow(actuals_mat)

# Pre-compute expanding Σ_t^{-1} for each t using complete rows up to t-1.
sigma_inv_exp <- vector("list", T_oos)
for (t in (MIN_SIGMA + 1L):T_oos) {
  rows <- which(complete.cases(actuals_mat[seq_len(t - 1L), , drop = FALSE]))
  if (length(rows) >= MIN_SIGMA) {
    St <- cov(actuals_mat[rows, , drop = FALSE])
    sigma_inv_exp[[t]] <- tryCatch(solve(St), error = function(e) MASS::ginv(St))
  }
}

rafe_expanding <- map_dfr(all_models_rafe, function(mod) {
  pred_mat <- do.call(cbind, lapply(gamma_cols, function(gc) {
    col <- pred_list[[gc]]
    if (mod %in% colnames(col)) col[, mod] else rep(NA_real_, nrow(col))
  }))
  errors <- pred_mat - actuals_mat
  m_t <- vapply((MIN_SIGMA + 1L):T_oos, function(t) {
    e <- errors[t, ]; Si <- sigma_inv_exp[[t]]
    # same common-window restriction as the headline RAFE (identical months ∀ models)
    if (!rafe_common[t] || anyNA(e) || is.null(Si)) NA_real_
    else as.numeric(t(e) %*% Si %*% e)
  }, numeric(1L))
  m_t <- m_t[is.finite(m_t)]
  if (length(m_t) < 10L) return(NULL)
  tibble(model = mod, RAFE_exp = round(sqrt(mean(m_t)), 4L), n_oos_exp = length(m_t))
}) |>
  arrange(RAFE_exp)

rafe_compare <- inner_join(
  rafe_results   |> dplyr::select(model, RAFE)     |> mutate(rank_full = row_number()),
  rafe_expanding |> dplyr::select(model, RAFE_exp) |> mutate(rank_exp  = row_number()),
  by = "model"
) |>
  mutate(rank_diff = abs(rank_full - rank_exp)) |>
  arrange(rank_full)

rho_rafe <- suppressWarnings(
  cor(rafe_compare$rank_full, rafe_compare$rank_exp, method = "spearman")
)

cat("\n=== Expanding-Σ RAFE (robustness to full-sample Σ) ===\n")
cat(sprintf("Σ_t estimated through t-1 only (min %d months). Months used >= %d.\n",
            MIN_SIGMA, T_oos - MIN_SIGMA))
cat(sprintf("Spearman rank correlation (full-sample vs expanding Σ): %.3f\n",
            rho_rafe))
cat("If ~1, the full-sample Σ does not drive the model ranking → RAFE is robust.\n\n")
print(rafe_compare, n = Inf)


# (12b2) Traded-asset Σ: RAFE weighted by the long-short leg covariance --------
# The headline RAFE weights the gamma forecast-error vector by Σ_γ, the covariance
# of the REALISED GAMMAS. Salcher, Stöckl & Hanke (2026) define RAFE with the
# inverse covariance of the assets actually being TRADED, and the timing
# strategies here trade the six long-short quintile books, not the gammas
# themselves — the two objects correlate only 0.42-0.70 (Section res-ts), so Σ_γ
# and Σ_LS are genuinely different matrices. Inverse-covariance weighting is
# precisely where RAFE is sensitive, which puts the chapter's central
# methodological lesson (the random forest's best-R²/worst-RAFE inversion) at
# risk of being an artefact of scoring forecasts against a covariance structure
# the portfolio never faces. This block re-scores every model under Σ_LS.
#
# Σ_LS is re-estimated here on the SAME months as Σ_γ (the RAFE common window),
# NOT reused from the Markowitz block above, so the weighting matrix is the only
# thing that differs between the two rankings. Errors stay in gamma units; only
# the metric of the quadratic form changes.
ls_rafe_mat <- ls_by_gamma[match(dates_oos, ls_by_gamma$date), , drop = FALSE] |>
  dplyr::select(dplyr::all_of(gamma_cols)) |>
  as.matrix()
stopifnot(identical(colnames(ls_rafe_mat), gamma_cols),
          nrow(ls_rafe_mat) == nrow(actuals_mat))

Sigma_ls_rafe <- cov(ls_rafe_mat[complete_rows, , drop = FALSE],
                     use = "pairwise.complete.obs")
Sigma_ls_rafe_inv <- tryCatch(
  solve(Sigma_ls_rafe),
  error = function(e) {
    cat("WARNING: Sigma_ls_rafe singular — using pseudo-inverse (MASS::ginv)\n")
    MASS::ginv(Sigma_ls_rafe)
  }
)

rafe_ls <- map_dfr(all_models_rafe, function(mod) {
  pred_mat <- do.call(cbind, lapply(gamma_cols, function(gc) {
    col <- pred_list[[gc]]
    if (mod %in% colnames(col)) col[, mod] else rep(NA_real_, nrow(col))
  }))
  errors  <- pred_mat - actuals_mat
  valid_t <- which(rafe_common & complete.cases(errors))
  if (length(valid_t) < 10L) return(NULL)
  E <- errors[valid_t, , drop = FALSE]
  m_t <- apply(E, 1L, function(e) as.numeric(t(e) %*% Sigma_ls_rafe_inv %*% e))
  tibble(model = mod, RAFE_LS = round(sqrt(mean(m_t)), 4L), n_oos_ls = length(valid_t))
}) |>
  arrange(RAFE_LS)

rafe_ls_compare <- inner_join(
  rafe_results |> dplyr::select(model, RAFE)    |> mutate(rank_gamma = row_number()),
  rafe_ls      |> dplyr::select(model, RAFE_LS) |> mutate(rank_ls    = row_number()),
  by = "model"
) |>
  mutate(rank_diff = abs(rank_gamma - rank_ls)) |>
  arrange(rank_gamma)

rho_rafe_ls <- suppressWarnings(
  cor(rafe_ls_compare$rank_gamma, rafe_ls_compare$rank_ls, method = "spearman")
)

# Does the ex-ante headline selection survive the change of weighting matrix?
# best_model itself is defined in (12c) below; recompute the Σ_γ pick locally so
# this block does not depend on evaluation order.
best_model_gamma <- rafe_results |>
  filter(model != "hist_mean") |>
  slice_min(RAFE, n = 1) |>
  pull(model)
best_model_ls <- rafe_ls |>
  filter(model != "hist_mean") |>
  slice_min(RAFE_LS, n = 1) |>
  pull(model)
rf_rank_gamma <- rafe_ls_compare$rank_gamma[rafe_ls_compare$model == "rf"]
rf_rank_ls    <- rafe_ls_compare$rank_ls[rafe_ls_compare$model == "rf"]
n_rafe_models <- nrow(rafe_ls_compare)

cat("\n=== RAFE under the traded-asset covariance Σ_LS (robustness) ===\n")
cat("Errors are gamma forecast errors throughout; only the weighting matrix changes.\n")
cat(sprintf("Spearman rank correlation (Σ_gamma vs Σ_LS): %.3f\n", rho_rafe_ls))
cat(sprintf("Lowest-RAFE model:  Σ_gamma = %s  |  Σ_LS = %s\n",
            best_model_gamma, best_model_ls))
cat(sprintf("Random forest rank: Σ_gamma = %d/%d  |  Σ_LS = %d/%d (higher = worse)\n",
            rf_rank_gamma, n_rafe_models, rf_rank_ls, n_rafe_models))
cat("If the RF stays near the bottom under Σ_LS, the R²/RAFE inversion is a\n")
cat("property of the forecasts, not of the choice of covariance matrix.\n\n")
print(rafe_ls_compare, n = Inf)


# (12b3) Translation check: does the traded leg move with its gamma? -----------
# Direction timing applies sign(gamma_hat) to a long-short quintile book, so the
# forecast object (the FM slope) and the traded object (the Q5-Q1 spread) are not
# the same thing. Whenever their realised signs disagree, a correct gamma
# forecast still produces a losing month: the translation step injects noise on
# top of forecast error. That noise is invisible in both the OOS R² and the RAFE
# tables, so it is quantified here.
#
# Both series are RAW and share an orientation — ls_ret is q5-q1 on the same raw
# characteristic that the FM regression takes as its regressor — so no sign
# convention has to be undone and the comparison is direct. (Reconstructing
# ls_ret from ret_static instead would require dividing by sign(static_gamma),
# the FROZEN 1963-1989 mean, which differs in sign from the realised OOS mean for
# value and beta; see the sign-flip note below.)
sign_agreement <- map_dfr(gamma_cols, function(gc) {
  gam <- actuals_mat[, gc]
  lsr <- ls_rafe_mat[, gc]
  ok  <- rafe_common & is.finite(gam) & is.finite(lsr)
  g_ok <- gam[ok]; l_ok <- lsr[ok]
  agree <- sign(g_ok) == sign(l_ok)
  big   <- abs(g_ok) > median(abs(g_ok))   # months with a strong realised premium
  tibble(
    gamma        = gc,
    cor_gamma_ls = round(cor(g_ok, l_ok), 3L),
    agree_all    = round(100 * mean(agree),      1L),
    agree_big    = round(100 * mean(agree[big]), 1L),   # |gamma| above its median
    agree_small  = round(100 * mean(agree[!big]), 1L),
    n            = sum(ok)
  )
})

cat("\n=== Gamma / traded-leg translation check ===\n")
cat("cor_gamma_ls = correlation of the realised gamma with its raw q5-q1 leg\n")
cat("agree_*      = % of months where the two share a sign (all / |gamma| above\n")
cat("               its median / below). Direction timing only pays when they agree.\n\n")
print(sign_agreement, n = Inf)

# Frozen-sign diagnostic: static_gamma is the expanding mean AT THE FIRST
# evaluation month, i.e. the 1963-1989 premium a real-time investor would have
# held. For value and beta that sign is the OPPOSITE of the realised 1990-2025
# mean — the premia reversed between training and evaluation. This is why the
# static book earns so little on those two legs, and it is a property of the
# data, not of the benchmark construction.
static_sign_flip <- tibble(
  gamma      = gamma_cols,
  frozen_t0  = round(static_gamma[gamma_cols], 5L),
  oos_mean   = round(colMeans(actuals_mat[complete_rows, , drop = FALSE]), 5L),
  sign_flip  = sign(static_gamma[gamma_cols]) !=
               sign(colMeans(actuals_mat[complete_rows, , drop = FALSE]))
)
cat("\nFrozen (1963-1989) vs realised-OOS mean gamma, by sign:\n")
print(static_sign_flip, n = Inf)

# Split the evaluation window at the value break of Section res-breaks
# (estimated March 2001) to separate a SUSTAINED reversal from two opposing
# regimes averaging to zero: beta is positive on both sides, value is positive
# before and negative after, so only beta's flip is a level a forecaster could
# learn. Reported in §res-breaks; computed here so the numbers are reproducible.
VALUE_BREAK <- as.Date("2001-03-01")
regime_split <- tibble(
  gamma = gamma_cols,
  mean_pre  = round(colMeans(actuals_mat[rafe_common & dates_oos <  VALUE_BREAK, ,
                                         drop = FALSE], na.rm = TRUE), 5L),
  mean_post = round(colMeans(actuals_mat[rafe_common & dates_oos >= VALUE_BREAK, ,
                                         drop = FALSE], na.rm = TRUE), 5L)
) |>
  mutate(same_sign = sign(mean_pre) == sign(mean_post))
cat(sprintf("\nEvaluation window split at the value break (%s):\n",
            format(VALUE_BREAK, "%Y-%m")))
print(regime_split, n = Inf)


# (12c) New-results plots ------------------------------------------------------
# Produced here (after RAFE) so the headline timed model matches Table 4: lowest RAFE.
best_model <- rafe_results |>
  filter(model != "hist_mean") |>
  slice_min(RAFE, n = 1) |>
  pull(model)
cat(sprintf("\nHeadline timed model for plots (lowest RAFE): %s\n", best_model))


# (12d) Ledoit-Wolf (2008) test of the Sharpe-ratio DIFFERENCE -----------------
# The headline economic claim of the thesis is that direction timing raises the
# combined Sharpe ratio relative to the static benchmark. Until now that claim
# was never TESTED: the FF5 alpha t-stats and the GRS test address pricing (is
# there return the factors cannot absorb?), not the Sharpe difference itself,
# and the two Sharpe ratios are computed on the SAME months from OVERLAPPING
# positions, so they are strongly dependent — a naive comparison of two
# independent Sharpe standard errors would be badly mis-sized.
#
# Ledoit & Wolf (2008, JEF) test H0: SR_1 - SR_2 = 0 for dependent series.
# Writing SR_i = mu_i / sqrt(gamma_i - mu_i^2) with gamma_i = E[r_i^2], the
# difference is a smooth function f(v) of the moment vector
#   v = (mu_1, mu_2, gamma_1, gamma_2)'
# so by the delta method  sqrt(T)(Delta_hat - Delta) -> N(0, grad' Psi grad),
# where Psi is the long-run (HAC) covariance of (r_1t, r_2t, r_1t^2, r_2t^2)'.
# Two p-values are reported:
#   (a) HAC/delta-method, Psi from a Parzen-kernel prewhitened HAC estimator;
#   (b) studentised circular block bootstrap, which Ledoit-Wolf RECOMMEND as
#       their preferred variant because the delta-method interval is liberal in
#       finite samples with fat-tailed, autocorrelated returns — exactly our case.
# The bootstrap is the reported figure; the HAC value is a cross-check.
# Reference: Ledoit & Wolf (2008, Journal of Empirical Finance 15, 850-859).

# --- Long-run covariance of the moment vector (Parzen kernel, VAR(1) prewhiten)
lw_psi_hat <- function(V) {
  Tn <- nrow(V); p <- ncol(V)
  Vc <- scale(V, center = TRUE, scale = FALSE)
  # VAR(1) prewhitening: fit v_t = A v_{t-1} + e_t, HAC the residuals, recolour.
  A <- matrix(0, p, p); E <- Vc[-1, , drop = FALSE]
  ok_pw <- TRUE
  fit <- try({
    X <- Vc[-Tn, , drop = FALSE]
    A <- t(solve(crossprod(X), crossprod(X, Vc[-1, , drop = FALSE])))
    E <- Vc[-1, , drop = FALSE] - X %*% t(A)
  }, silent = TRUE)
  if (inherits(fit, "try-error") || max(abs(eigen(A, only.values = TRUE)$values)) > 0.97) {
    A <- matrix(0, p, p); E <- Vc; ok_pw <- FALSE   # skip prewhitening if unstable
  }
  Te <- nrow(E)
  # Andrews (1991) automatic bandwidth for the Parzen kernel, univariate-AR(1) rule
  rho <- vapply(seq_len(p), function(j) {
    e <- E[, j]; num <- sum(e[-1] * e[-Te]); den <- sum(e[-Te]^2)
    if (den <= 0) 0 else max(min(num / den, 0.97), -0.97)
  }, numeric(1))
  a2 <- sum(4 * rho^2 / ((1 - rho)^8)) / max(sum(1 / ((1 - rho)^4)), 1e-12)
  S  <- min(max(2.6614 * (a2 * Te)^(1 / 5), 1), Te - 1)
  parzen <- function(x) {
    ax <- abs(x)
    ifelse(ax <= 0.5, 1 - 6 * x^2 + 6 * ax^3,
           ifelse(ax <= 1, 2 * (1 - ax)^3, 0))
  }
  Psi_e <- crossprod(E) / Te
  for (l in seq_len(Te - 1)) {
    w <- parzen(l / S)
    if (w < 1e-8) next
    G <- crossprod(E[-seq_len(l), , drop = FALSE], E[seq_len(Te - l), , drop = FALSE]) / Te
    Psi_e <- Psi_e + w * (G + t(G))
  }
  D <- solve(diag(p) - A)
  list(Psi = D %*% Psi_e %*% t(D), prewhitened = ok_pw, bandwidth = S)
}

# --- Sharpe difference and its standard error (monthly units)
lw_sr_diff <- function(r1, r2) {
  ok <- !is.na(r1) & !is.na(r2)
  r1 <- r1[ok]; r2 <- r2[ok]; Tn <- length(r1)
  mu1 <- mean(r1); mu2 <- mean(r2)
  g1  <- mean(r1^2); g2 <- mean(r2^2)
  s1  <- sqrt(g1 - mu1^2); s2 <- sqrt(g2 - mu2^2)
  diff <- mu1 / s1 - mu2 / s2
  grad <- c(g1 / s1^3, -g2 / s2^3, -mu1 / (2 * s1^3), mu2 / (2 * s2^3))
  V    <- cbind(r1, r2, r1^2, r2^2)
  ph   <- lw_psi_hat(V)
  se   <- sqrt(max(as.numeric(t(grad) %*% ph$Psi %*% grad), 0) / Tn)
  list(diff = diff, se = se, T = Tn, bandwidth = ph$bandwidth)
}

# --- Studentised circular block bootstrap (Ledoit-Wolf's preferred variant)
lw_boot_p <- function(r1, r2, B = 4999L, block = 6L, seed = 42L) {
  set.seed(seed)
  ok <- !is.na(r1) & !is.na(r2)
  r1 <- r1[ok]; r2 <- r2[ok]; Tn <- length(r1)
  base <- lw_sr_diff(r1, r2)
  if (!is.finite(base$se) || base$se <= 0) return(list(p = NA_real_, base = base))
  stat0  <- base$diff / base$se
  nblk   <- ceiling(Tn / block)
  # circular extension so every observation is equally likely to be sampled
  idx_c  <- c(seq_len(Tn), seq_len(block))
  count  <- 0L; valid <- 0L
  for (b in seq_len(B)) {
    starts <- sample.int(Tn, nblk, replace = TRUE)
    idx    <- as.vector(vapply(starts, \(s) idx_c[s:(s + block - 1L)], numeric(block)))[seq_len(Tn)]
    bs <- try(lw_sr_diff(r1[idx], r2[idx]), silent = TRUE)
    if (inherits(bs, "try-error") || !is.finite(bs$se) || bs$se <= 0) next
    valid <- valid + 1L
    # centre on the observed difference: H0 imposed by recentring, not by resampling
    if (abs((bs$diff - base$diff) / bs$se) >= abs(stat0)) count <- count + 1L
  }
  list(p = (count + 1) / (valid + 1), base = base, B_valid = valid)
}

cat("\n=== Ledoit-Wolf (2008) Sharpe-ratio difference tests ===\n")
cat("H0: SR(timed) - SR(static) = 0, on the combined six-factor book,\n")
cat("common 414-month window, gross of costs. Studentised circular block\n")
cat("bootstrap (B = 4999, block = 6 months) with HAC delta-method cross-check.\n\n")

static_ret_vec <- port_ret |>
  filter(char_col == "COMBINED", model == "hist_mean") |>
  arrange(date) |>
  pull(ret_static)

lw_models <- unique(c(best_model, "rf", "lstm_r1", "mlp", "lasso", "comb"))
lw_models <- intersect(lw_models, unique(port_ret$model))

sharpe_tests <- map_dfr(lw_models, function(m) {
  map_dfr(c("direction", "scaled"), function(st) {
    col <- paste0("ret_", st)
    rv  <- port_ret |>
      filter(char_col == "COMBINED", model == m) |>
      arrange(date) |>
      pull(!!sym(col))
    if (all(is.na(rv))) return(tibble())
    bt <- lw_boot_p(rv, static_ret_vec)
    tibble(
      model        = m,
      strategy     = st,
      sharpe_timed = sharpe_annual(rv),
      sharpe_static= sharpe_annual(static_ret_vec),
      # annualise the monthly Sharpe difference for reporting
      diff_annual  = bt$base$diff * sqrt(12),
      se_annual    = bt$base$se   * sqrt(12),
      t_stat       = bt$base$diff / bt$base$se,
      p_hac        = 2 * pnorm(-abs(bt$base$diff / bt$base$se)),
      p_boot       = bt$p,
      # Minimum detectable effect: the smallest TRUE annualised Sharpe gap this
      # design could reject at 80% power and 5% size, MDE = (z_.975 + z_.80)*SE.
      # This is deliberately NOT "post hoc power" evaluated at the observed
      # effect, which is a monotone transform of the p-value and therefore
      # carries no information beyond it (Hoenig & Heisey, 2001, Am. Stat.).
      # The MDE is a property of the DESIGN (sample length and the second-moment
      # structure of the two return series), not of the realised estimate.
      mde_annual   = (qnorm(0.975) + qnorm(0.80)) * bt$base$se * sqrt(12),
      n_months     = bt$base$T
    )
  })
})

sharpe_tests <- sharpe_tests |> arrange(strategy, desc(sharpe_timed))
print(sharpe_tests |> mutate(across(where(is.numeric), \(x) round(x, 3))), n = Inf)

cat("\nNote: p_boot is the reported p-value (Ledoit-Wolf's preferred studentised\n")
cat("bootstrap); p_hac is the delta-method cross-check and is liberal in finite\n")
cat("samples. A significant DIRECTION row means the timing gain is not\n")
cat("attributable to sampling variation in the Sharpe ratios.\n")

cat("\n--- Minimum detectable effect (design resolution) ---\n")
mde_dir <- sharpe_tests |> filter(strategy == "direction")
cat(sprintf("Across direction-timed models the MDE spans %.2f to %.2f annualised\n",
            min(mde_dir$mde_annual), max(mde_dir$mde_annual)))
cat(sprintf("Sharpe units (median %.2f), against observed gaps of %.2f to %.2f.\n",
            median(mde_dir$mde_annual), min(mde_dir$diff_annual),
            max(mde_dir$diff_annual)))
cat("Read: at 80% power and 5% size this design can certify gaps of roughly that\n")
cat("size or larger. Observed gaps below the MDE are economically material but\n")
cat("below the resolution of the test — a statement about what the sample can\n")
cat("establish, NOT evidence that the gap is zero.\n")

# --- Deflated-Sharpe sanity check (Bailey & Lopez de Prado 2014) --------------
# We evaluated many model/strategy combinations, so the MAXIMUM observed Sharpe
# is upward-biased by selection even if no strategy has skill. The deflated
# Sharpe ratio asks whether the best Sharpe survives the multiplicity of trials.
# N_trials counts the model x strategy grid actually searched.
dsr_inputs <- port_eval |> filter(char_col == "COMBINED", strategy != "static")
n_trials   <- nrow(dsr_inputs)
best_row   <- dsr_inputs |> slice_max(sharpe, n = 1)
best_ret   <- port_ret |>
  filter(char_col == "COMBINED", model == best_row$model[1]) |>
  arrange(date) |>
  pull(!!sym(paste0("ret_", best_row$strategy[1])))
best_ret   <- best_ret[!is.na(best_ret)]

sr_m   <- mean(best_ret) / sd(best_ret)           # monthly, non-annualised
Tn_b   <- length(best_ret)
g3     <- mean((best_ret - mean(best_ret))^3) / sd(best_ret)^3   # skewness
g4     <- mean((best_ret - mean(best_ret))^4) / sd(best_ret)^4   # kurtosis
sr_sd  <- sd(dsr_inputs$sharpe / sqrt(12), na.rm = TRUE)         # across trials, monthly
emc    <- 0.5772156649
sr0    <- sr_sd * ((1 - emc) * qnorm(1 - 1 / n_trials) +
                     emc * qnorm(1 - 1 / (n_trials * exp(1))))   # expected max under H0
dsr    <- pnorm(((sr_m - sr0) * sqrt(Tn_b - 1)) /
                  sqrt(1 - g3 * sr_m + ((g4 - 1) / 4) * sr_m^2))

cat(sprintf("\n=== Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) ===\n"))
cat(sprintf("Best combined strategy: %s / %s (annualised SR = %.3f)\n",
            best_row$model[1], best_row$strategy[1], best_row$sharpe[1]))
cat(sprintf("Trials searched N = %d; expected max SR under H0 = %.3f ann.; DSR = %.4f\n",
            n_trials, sr0 * sqrt(12), dsr))
cat("DSR is the probability the true Sharpe exceeds zero AFTER deflating for the\n")
cat("number of trials, non-normality, and sample length. DSR > 0.95 => the best\n")
cat("strategy is not explained by selection over the model grid alone.\n")

sharpe_test_meta <- list(
  n_trials = n_trials, sr0_annual = sr0 * sqrt(12), dsr = dsr,
  best_model = best_row$model[1], best_strategy = best_row$strategy[1]
)


# (12e) Sub-period decomposition of the portfolio results ----------------------
# The statistical evidence (rolling OOS R²) shows the predictability is earned
# disproportionately before 2010. The full-window Sharpe/alpha figures must be
# decomposed on the SAME cuts, otherwise the obvious objection — "the gains are
# all from the 1990s" — is left unanswered. Cuts match the OOS sub-periods used
# throughout: pre-2000, 2000-2010, post-2010.
subperiod_of <- function(d) {
  dplyr::case_when(
    d <  as.Date("2000-01-01")                             ~ "Pre-2000",
    d >= as.Date("2000-01-01") & d < as.Date("2010-01-01") ~ "2000-2010",
    TRUE                                                   ~ "Post-2010"
  )
}

sub_levels <- c("Pre-2000", "2000-2010", "Post-2010", "Full")

port_sub <- port_ret |>
  filter(char_col == "COMBINED") |>
  mutate(sub_period = subperiod_of(date))

subperiod_eval <- map_dfr(c("Pre-2000", "2000-2010", "Post-2010", "Full"), function(sp) {
  d <- if (sp == "Full") port_sub else port_sub |> filter(sub_period == sp)
  map_dfr(c("static", "direction", "scaled", "volmanaged"), function(st) {
    col <- paste0("ret_", st)
    # static is model-invariant; report it once under the hist_mean row
    mods <- if (st == "static") "hist_mean" else lw_models
    map_dfr(intersect(mods, unique(d$model)), function(m) {
      dm <- d |> filter(model == m) |> arrange(date)
      r  <- dm[[col]]
      if (all(is.na(r))) return(tibble())
      a <- ff5_alpha(r, dm$date, ff5)
      tibble(sub_period = sp, strategy = st, model = m,
             n_months = sum(!is.na(r)),
             sharpe = sharpe_annual(r), vol = vol_annual(r),
             alpha_pct = a$alpha, t_alpha = a$t_alpha)
    })
  })
}) |>
  mutate(sub_period = factor(sub_period, levels = sub_levels)) |>
  arrange(sub_period, strategy, desc(sharpe))

cat("\n=== Sub-period decomposition: COMBINED portfolio ===\n")
cat("Static vs direction timing on the three OOS sub-period cuts.\n\n")
print(
  subperiod_eval |>
    filter(strategy %in% c("static", "direction")) |>
    mutate(across(where(is.numeric), \(x) round(x, 3))),
  n = Inf
)

# Explicit static-vs-direction gap per sub-period for the selected model
sub_gap <- subperiod_eval |>
  filter(strategy %in% c("static", "direction"),
         model %in% c("hist_mean", best_model)) |>
  select(sub_period, strategy, sharpe) |>
  distinct(sub_period, strategy, .keep_all = TRUE) |>
  pivot_wider(names_from = strategy, values_from = sharpe) |>
  mutate(gap = direction - static)
cat(sprintf("\nStatic vs direction (%s) by sub-period:\n", best_model))
print(sub_gap |> mutate(across(where(is.numeric), \(x) round(x, 3))), n = Inf)


# (12f) Split-sample RAFE selection — closing the last look-ahead channel ------
# The headline model is chosen by RAFE computed over the SAME window on which
# its portfolio performance is then reported. Even though RAFE is a
# forecast-error statistic rather than the portfolio objective, this is still a
# form of evaluation-window model selection: the selected model's reported
# Sharpe benefits from having been picked with knowledge of the whole window.
#
# The clean fix is a strict split:
#   SELECT  on RAFE over 1990-01 .. 2009-12   (selection sample)
#   EVALUATE the portfolio over 2010-01 ..    (holdout, never used to select)
# If the split-sample choice performs comparably out of sample, the headline is
# not an artefact of the selection channel.
split_date <- as.Date("2010-01-01")
sel_rows   <- which(rafe_common & dates_oos <  split_date)
hold_dates <- dates_oos[dates_oos >= split_date & rafe_common]

cat("\n=== Split-sample RAFE selection (C9) ===\n")
cat(sprintf("Selection window: %s .. %s (%d months)\n",
            format(min(dates_oos[sel_rows]), "%Y-%m"),
            format(max(dates_oos[sel_rows]), "%Y-%m"), length(sel_rows)))
cat(sprintf("Holdout window:   %s .. %s (%d months)\n",
            format(min(hold_dates), "%Y-%m"),
            format(max(hold_dates), "%Y-%m"), length(hold_dates)))

# Sigma re-estimated on the SELECTION window only — no holdout information.
Sigma_sel     <- cov(actuals_mat[sel_rows, ])
Sigma_sel_inv <- tryCatch(solve(Sigma_sel), error = function(e) MASS::ginv(Sigma_sel))

rafe_split <- map_dfr(all_models_rafe, function(mod) {
  pred_mat <- do.call(cbind, lapply(gamma_cols, function(gc) {
    col <- pred_list[[gc]]
    if (mod %in% colnames(col)) col[, mod] else rep(NA_real_, nrow(col))
  }))
  errors  <- pred_mat - actuals_mat
  valid_t <- intersect(sel_rows, which(complete.cases(errors)))
  if (length(valid_t) < 10L) return(NULL)
  E <- errors[valid_t, , drop = FALSE]
  tibble(model = mod,
         RAFE_sel = sqrt(mean(rowSums((E %*% Sigma_sel_inv) * E))),
         n_sel    = length(valid_t))
}) |> arrange(RAFE_sel)

best_split <- rafe_split |> filter(model != "hist_mean") |> slice_min(RAFE_sel, n = 1) |> pull(model)
cat(sprintf("\nSelected on 1990-2009 RAFE: %s (full-window pick was: %s)\n",
            best_split, best_model))
print(rafe_split |> head(8) |> mutate(across(where(is.numeric), \(x) round(x, 3))), n = Inf)

# Holdout portfolio performance of BOTH picks, plus the static benchmark
hold_eval <- map_dfr(unique(c(best_split, best_model)), function(m) {
  dm <- port_ret |>
    filter(char_col == "COMBINED", model == m, date >= split_date) |>
    arrange(date)
  a <- ff5_alpha(dm$ret_direction, dm$date, ff5)
  tibble(model = m,
         selected_by = paste(c(if (m == best_split) "split", if (m == best_model) "full"),
                             collapse = "+"),
         sharpe = sharpe_annual(dm$ret_direction),
         alpha_pct = a$alpha, t_alpha = a$t_alpha, n_months = nrow(dm))
})
static_hold <- port_ret |>
  filter(char_col == "COMBINED", model == "hist_mean", date >= split_date) |>
  arrange(date)
a_sh <- ff5_alpha(static_hold$ret_static, static_hold$date, ff5)
hold_eval <- bind_rows(
  hold_eval,
  tibble(model = "static benchmark", selected_by = "-",
         sharpe = sharpe_annual(static_hold$ret_static),
         alpha_pct = a_sh$alpha, t_alpha = a_sh$t_alpha, n_months = nrow(static_hold))
)
cat("\nHoldout (2010+) direction-timed performance of each pick:\n")
print(hold_eval |> mutate(across(where(is.numeric), \(x) round(x, 3))), n = Inf)
cat("\nIf the split-sample pick performs comparably to the full-window pick on the\n")
cat("holdout, the headline result is not driven by the selection channel.\n")


# --- Plot A: cumulative returns, Static vs the three timed strategies ----------
static_series <- port_ret |>
  filter(char_col == "COMBINED", model == "hist_mean") |>
  transmute(date, label = "Static", ret = ret_static)
timed_series <- port_ret |>
  filter(char_col == "COMBINED", model == best_model) |>
  select(date, ret_direction, ret_volmanaged, ret_markowitz) |>
  pivot_longer(-date, names_to = "strategy", values_to = "ret") |>
  mutate(label = recode(strategy, ret_direction = "Direction",
                        ret_volmanaged = "Vol-Managed", ret_markowitz = "Markowitz"))
# Net-of-10bp direction path (reviewer suggestion): the first thing a
# practitioner examiner looks for is a NET wealth curve; costs charged as
# TC × empirical monthly overlay turnover, same convention as block 11b.
net10_series <- tc_panel |>
  filter(model == best_model) |>
  transmute(date, label = "Direction (net 10 bp)",
            ret = ret_direction - TC_10BP * to_direction)
ret_long <- bind_rows(static_series, timed_series, net10_series) |>
  mutate(label = factor(label, levels = c("Static", "Direction",
                                          "Direction (net 10 bp)",
                                          "Vol-Managed", "Markowitz")))

strat_cols <- c("Static" = "grey40", "Direction" = "steelblue",
                "Direction (net 10 bp)" = "#9DC3E6",
                "Vol-Managed" = "darkorange", "Markowitz" = "forestgreen")

cumret_plot <- ret_long |>
  arrange(label, date) |>
  group_by(label) |>
  mutate(cum_ret = cumprod(1 + replace_na(ret, 0)) - 1) |>
  ungroup()

p_cum <- ggplot(cumret_plot, aes(x = date, y = cum_ret * 100, colour = label)) +
  geom_line(linewidth = 0.7) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  scale_colour_manual(values = strat_cols) +
  labs(title = sprintf("Cumulative Returns: Static vs Timed Combined Portfolio (%s)", best_model),
       x = NULL, y = "Cumulative Return (%)", colour = NULL) +
  theme_bw() + theme(legend.position = "bottom")
ggsave("plots/cumulative_returns.pdf", p_cum, width = 10, height = 5)
cat("Saved: plots/cumulative_returns.pdf\n")

# --- Plot A2: cumulative returns + drawdowns, aligned two-panel (thesis Fig.) --
# Same series as Plot A; adds a drawdown panel (dd = wealth / running max - 1)
# so the risk side of the timing gains is visible in the same exhibit.
wealth_df <- ret_long |>
  arrange(label, date) |>
  group_by(label) |>
  mutate(w   = cumprod(1 + replace_na(ret, 0)),
         dd  = 100 * (w / cummax(w) - 1),
         cum = 100 * (w - 1)) |>
  ungroup()

p_cum2_top <- ggplot(wealth_df, aes(date, cum, colour = label)) +
  geom_line(linewidth = 0.65) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  scale_colour_manual(values = strat_cols) +
  labs(x = NULL, y = "Cumulative return (%)", colour = NULL) +
  theme_bw(base_size = 10) +
  theme(legend.position = "top",
        axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        plot.margin = margin(2, 8, 0, 8))
p_cum2_bot <- ggplot(wealth_df, aes(date, dd, colour = label)) +
  geom_line(linewidth = 0.5, show.legend = FALSE) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  scale_colour_manual(values = strat_cols) +
  labs(x = NULL, y = "Drawdown (%)") +
  theme_bw(base_size = 10) +
  theme(plot.margin = margin(0, 8, 2, 8))
if (requireNamespace("ggpubr", quietly = TRUE)) {
  p_cum2 <- ggpubr::ggarrange(p_cum2_top, p_cum2_bot, ncol = 1,
                              heights = c(2.2, 1), align = "v")
  ggsave("plots/cumret_drawdown.pdf", p_cum2, width = 10, height = 6)
  cat("Saved: plots/cumret_drawdown.pdf\n")
}

# --- Plot B2: Sharpe-ratio decay under transaction costs (thesis Fig.) --------
# Gross vs net-of-10bp vs net-of-50bp Sharpe for the four timing overlays,
# the headline (lowest-RAFE) model and RF (the cost-fragile contrast case);
# dashed = static.
cost_df <- as.data.frame(tc_strategy) |>
  filter(model %in% c(best_model, "rf")) |>
  select(model, matches("^(direction|volmanaged|markowitz|scaled)_(gross|10bp|50bp)$")) |>
  pivot_longer(-model, names_to = c("strategy", "cost"), names_sep = "_",
               values_to = "sharpe") |>
  mutate(
    strategy = recode(strategy, direction = "Direction", volmanaged = "Vol-Managed",
                      markowitz = "Markowitz", scaled = "Scaled"),
    strategy = factor(strategy, levels = c("Direction", "Vol-Managed", "Markowitz", "Scaled")),
    cost  = recode(cost, gross = "Gross", `10bp` = "Net 10 bp", `50bp` = "Net 50 bp"),
    cost  = factor(cost, levels = c("Gross", "Net 10 bp", "Net 50 bp")),
    model = recode(model, lstm_r12 = "LSTM-12m", xgb = "XGBoost", rf = "Random Forest"),
    model = factor(model, levels = unique(c(recode(best_model, lstm_r12 = "LSTM-12m",
                                                   xgb = "XGBoost"), "Random Forest")))
  )
# Static benchmark on the TC window (= the 0.50 quoted in the net-of-cost table)
static_sharpe <- tc_eval |>
  filter(model == "hist_mean") |>
  pull(sharpe_static_gross) |> first()
p_cost <- ggplot(cost_df, aes(strategy, sharpe, fill = cost)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.65) +
  geom_hline(yintercept = static_sharpe, linetype = "dashed",
             colour = "grey30", linewidth = 0.4) +
  annotate("text", x = 4.28, y = static_sharpe + 0.03,
           label = sprintf("static (%.2f)", static_sharpe),
           size = 2.9, colour = "grey30", hjust = 1) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  facet_wrap(~model) +
  scale_fill_manual(values = c("Gross" = "steelblue", "Net 10 bp" = "steelblue3",
                               "Net 50 bp" = "lightsteelblue2")) +
  labs(x = NULL, y = "Annualised Sharpe ratio", fill = NULL) +
  theme_bw(base_size = 10) +
  theme(legend.position = "top", panel.grid.major.x = element_blank())
ggsave("plots/cost_decay.pdf", p_cost, width = 9, height = 4)
cat("Saved: plots/cost_decay.pdf\n")

# Vol-matched version: scale each series to Static's unconditional volatility before
# cumulating, so the wealth curves are risk-comparable (removes the leverage/level
# distortion — e.g. gross-1 Markowitz is low-vol, so its raw cumulative level
# understates its risk-adjusted quality). Scaling is constant per series (full-sample),
# so it does not alter the Sharpe ratio of any line.
target_sd <- sd(ret_long$ret[ret_long$label == "Static"], na.rm = TRUE)
cumret_vm <- ret_long |>
  arrange(label, date) |>
  group_by(label) |>
  mutate(ret_s   = ret * (target_sd / sd(ret, na.rm = TRUE)),
         cum_ret = cumprod(1 + replace_na(ret_s, 0)) - 1) |>
  ungroup()

p_cum_vm <- ggplot(cumret_vm, aes(x = date, y = cum_ret * 100, colour = label)) +
  geom_line(linewidth = 0.7) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  scale_colour_manual(values = strat_cols) +
  labs(title = sprintf("Vol-Matched Cumulative Returns: Static vs Timed (%s)", best_model),
       subtitle = sprintf("All series scaled to the static benchmark's volatility (%.1f%% ann.) — risk-comparable",
                          target_sd * sqrt(12) * 100),
       x = NULL, y = "Cumulative Return (%)", colour = NULL) +
  theme_bw() + theme(legend.position = "bottom")
ggsave("plots/cumulative_returns_volmatched.pdf", p_cum_vm, width = 10, height = 5)
cat("Saved: plots/cumulative_returns_volmatched.pdf\n")

# --- Plot B: combined Sharpe by timing strategy (gross) ------------------------
strat_map <- c(static = "Static", direction = "Direction", volmanaged = "Vol-Managed",
               scaled = "Scaled", markowitz = "Markowitz")
strat_sharpe <- port_eval |>
  filter(char_col == "COMBINED",
         (model == "hist_mean" & strategy == "static") |
         (model == best_model  & strategy %in% c("direction", "volmanaged", "scaled", "markowitz"))) |>
  mutate(strategy_lab = factor(strat_map[strategy],
                               levels = c("Static", "Direction", "Vol-Managed", "Scaled", "Markowitz")))

p_strat <- ggplot(strat_sharpe, aes(strategy_lab, sharpe, fill = strategy_lab)) +
  geom_col(width = 0.65) +
  geom_hline(yintercept = static_sr_combined, linetype = "dashed", colour = "grey40") +
  geom_text(aes(label = formatC(sharpe, format = "f", digits = 2)), vjust = -0.4, size = 3) +
  scale_fill_brewer(palette = "Set2", guide = "none") +
  labs(title = sprintf("Combined Sharpe by Timing Strategy (%s, gross)", best_model),
       subtitle = "Dashed = static benchmark", x = NULL, y = "Annualised Sharpe") +
  theme_bw()
ggsave("plots/strategy_comparison.pdf", p_strat, width = 8, height = 4.5)
cat("Saved: plots/strategy_comparison.pdf\n")

# --- Plot C: net-of-cost Sharpe by strategy (gross / 10bp / 50bp) --------------
tc_long <- tc_strategy |>
  filter(model == best_model) |>
  select(-ends_with("_to")) |>
  pivot_longer(-model, names_to = c("strategy", "cost"), names_sep = "_", values_to = "sharpe") |>
  mutate(cost = factor(cost, levels = c("gross", "10bp", "50bp"),
                       labels = c("Gross", "Net 10bp", "Net 50bp")),
         strategy = factor(strategy, levels = c("direction", "volmanaged", "scaled", "markowitz"),
                           labels = c("Direction", "Vol-Managed", "Scaled", "Markowitz")))

p_tc <- ggplot(tc_long, aes(strategy, sharpe, fill = cost)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  scale_fill_manual(values = c("Gross" = "steelblue", "Net 10bp" = "goldenrod",
                               "Net 50bp" = "firebrick")) +
  labs(title = sprintf("Net-of-Cost Sharpe by Strategy (%s)", best_model),
       subtitle = "Costs = bps × empirical timing-overlay turnover",
       x = NULL, y = "Annualised Sharpe", fill = NULL) +
  theme_bw() + theme(legend.position = "bottom")
ggsave("plots/net_of_cost.pdf", p_tc, width = 8, height = 4.5)
cat("Saved: plots/net_of_cost.pdf\n")

# --- Plot D: H=1 monthly vs H=3 quarterly direction timing ---------------------
h3_plot <- h3_vs_h1 |>
  select(model, sharpe_h1, sharpe_h3q) |>
  pivot_longer(-model, names_to = "horizon", values_to = "sharpe") |>
  mutate(horizon = recode(horizon, sharpe_h1 = "H=1 (monthly)", sharpe_h3q = "H=3 (quarterly)"))

p_h3 <- ggplot(h3_plot, aes(reorder(model, sharpe), sharpe, fill = horizon)) +
  geom_col(position = position_dodge(0.75), width = 0.65) +
  geom_hline(yintercept = static_sr_combined, linetype = "dashed", colour = "grey40") +
  coord_flip() +
  scale_fill_manual(values = c("H=1 (monthly)" = "grey60", "H=3 (quarterly)" = "steelblue")) +
  labs(title = "Direction-Timing Sharpe: H=1 Monthly vs H=3 Quarterly (Combined)",
       subtitle = "Dashed = static benchmark; H=3 positions held one quarter (matched horizon)",
       x = NULL, y = "Annualised Sharpe", fill = NULL) +
  theme_bw() + theme(legend.position = "bottom")
ggsave("plots/h3_vs_h1_sharpe.pdf", p_h3, width = 8, height = 5)
cat("Saved: plots/h3_vs_h1_sharpe.pdf\n")

# --- Thesis exhibit: Sharpe vs RAFE scatter (2026-07-19, reviewer suggestion) --
# Visualises the rank_comparison: direction-timed combined Sharpe against RAFE
# across all 32 models; the RF inversion (best R², worst RAFE) and the headline
# pick are annotated. Spearman quoted on the ranked series (best-to-worst both).
sr_df <- rank_comparison |> filter(!is.na(sharpe), !is.na(RAFE))
rho_sr <- cor(sr_df$sharpe_rank, sr_df$rafe_rank, method = "spearman")
sr_annot <- sr_df |>
  filter(model %in% c("rf", best_model, "hist_mean", "lstm_r1", "lasso")) |>
  mutate(label = recode(model, rf = "Random Forest", comb_xgb = "Comb+XGB",
                        hist_mean = "Hist. mean", lstm_r1 = "LSTM-1m",
                        lstm_r12 = "LSTM-12m", lasso = "LASSO"))
p_sr <- ggplot(sr_df, aes(RAFE, sharpe)) +
  geom_point(colour = "grey60", size = 1.7) +
  geom_point(data = sr_annot, colour = "#4477AA", size = 2.3) +
  geom_text(data = sr_annot, aes(label = label), size = 3,
            hjust = -0.12, vjust = 0.35) +
  scale_x_continuous(expand = expansion(mult = c(0.03, 0.14))) +
  labs(x = "RAFE (lower = better forecast)",
       y = "Direction-timed combined Sharpe",
       subtitle = sprintf("Spearman rank correlation (both best-to-worst): %.2f",
                          rho_sr)) +
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank())
ggsave("plots/sharpe_vs_rafe.pdf", p_sr, width = 7, height = 4.2)
cat("Saved: plots/sharpe_vs_rafe.pdf\n")

# --- Thesis exhibit: direction-timing position map (2026-07-19) ---------------
# sign(pred) per characteristic leg over the common window for the headline
# model: shows the low-turnover mechanism (positions change only at sign flips)
# and, for value, the break-age window during which the timed book stayed long.
gl_map <- c(gamma_bm = "Value", gamma_mom12m = "Momentum",
            gamma_oper_prof = "Profitability", gamma_asset_growth = "Asset Growth",
            gamma_size = "Size", gamma_beta = "Beta")
pos_df <- purrr::map_dfr(names(gl_map), function(gc) {
  tibble(date = dates_oos, gamma = gl_map[gc],
         pos  = sign(pred_list[[gc]][, best_model]))
}) |>
  filter(date %in% common_dates, pos != 0) |>
  mutate(gamma = factor(gamma, levels = rev(unname(gl_map))),
         Position = factor(ifelse(pos > 0, "Long", "Short"),
                           levels = c("Long", "Short")))
val_y <- which(levels(pos_df$gamma) == "Value")
p_pos <- ggplot(pos_df, aes(date, gamma, fill = Position)) +
  geom_tile(height = 0.72) +
  annotate("rect", xmin = as.Date("2001-03-01"), xmax = as.Date("2007-09-01"),
           ymin = val_y - 0.45, ymax = val_y + 0.45,
           fill = NA, colour = "grey20", linetype = "dashed", linewidth = 0.45) +
  scale_fill_manual(values = c(Long = "#4477AA", Short = "#EE6677")) +
  labs(x = NULL, y = NULL, fill = NULL) +
  theme_bw(base_size = 10) +
  theme(legend.position = "top", panel.grid = element_blank())
ggsave("plots/position_map.pdf", p_pos, width = 8, height = 3.2)
cat("Saved: plots/position_map.pdf\n")

# --- Thesis exhibit: breakeven-cost curves (2026-07-19) -----------------------
# Net Sharpe as a continuous function of round-trip cost (0-100 bp), charged as
# cost × empirical monthly overlay turnover, for the headline model's four
# overlays; dashed line = static benchmark. Generalises the two cost points of
# the cost-decay chart into the breakeven levels the methodology promises.
cost_grid <- seq(0, 100, by = 5)   # bp
be_panel <- tc_panel |> filter(model == best_model)
be_df <- purrr::map_dfr(cost_grid, function(bp) {
  tc <- bp / 1e4
  tibble(
    bp = bp,
    Direction     = sharpe_annual(be_panel$ret_direction  - tc * be_panel$to_direction),
    `Vol-Managed` = sharpe_annual(be_panel$ret_volmanaged - tc * be_panel$to_volmanaged),
    Markowitz     = sharpe_annual(be_panel$ret_markowitz  - tc * be_panel$to_markowitz),
    Scaled        = sharpe_annual(be_panel$ret_scaled     - tc * be_panel$to_scaled)
  )
}) |>
  pivot_longer(-bp, names_to = "Strategy", values_to = "sharpe") |>
  mutate(Strategy = factor(Strategy,
                           levels = c("Direction", "Vol-Managed", "Markowitz", "Scaled")))
static_sr_be <- sharpe_annual(be_panel$ret_static)
be_cross <- be_df |>
  filter(Strategy == "Direction", sharpe >= static_sr_be) |>
  summarise(bp = max(bp)) |> pull(bp)
cat(sprintf("Direction breakeven vs static (%.2f): between %d and %d bp\n",
            static_sr_be, be_cross, be_cross + 5L))
be_lab <- be_df |> group_by(Strategy) |> slice_max(bp, n = 1) |> ungroup()
p_be <- ggplot(be_df, aes(bp, sharpe, colour = Strategy, linetype = Strategy)) +
  geom_hline(yintercept = static_sr_be, linetype = "dashed",
             colour = "grey30", linewidth = 0.4) +
  annotate("text", x = 1, y = static_sr_be + 0.025, hjust = 0, size = 2.9,
           colour = "grey30", label = sprintf("static (%.2f)", static_sr_be)) +
  geom_line(linewidth = 0.7) +
  geom_text(data = be_lab, aes(label = Strategy), hjust = -0.08, size = 3,
            show.legend = FALSE) +
  scale_colour_manual(values = c(Direction = "#4477AA", `Vol-Managed` = "#EE6677",
                                 Markowitz = "#228833", Scaled = "#CCBB44"),
                      guide = "none") +
  scale_linetype_manual(values = c(Direction = "solid", `Vol-Managed` = "longdash",
                                   Markowitz = "dotdash", Scaled = "dotted"),
                        guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.22))) +
  labs(x = "Round-trip cost (bp)", y = "Net annualised Sharpe") +
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank())
ggsave("plots/breakeven_cost.pdf", p_be, width = 7.5, height = 4)
cat("Saved: plots/breakeven_cost.pdf\n")


# (13) Save --------------------------------------------------------------------
saveRDS(list(port_ret = port_ret, port_eval = port_eval, grs_results = grs_results,
             tc_eval = tc_eval, tc_strategy = tc_strategy,
             rafe_results = rafe_results, rank_comparison = rank_comparison,
             rafe_expanding = rafe_expanding, rafe_compare = rafe_compare,
             rho_rafe = rho_rafe, Sigma_hat = Sigma_hat, Sigma_inv = Sigma_inv,
             rafe_ls = rafe_ls, rafe_ls_compare = rafe_ls_compare,
             rho_rafe_ls = rho_rafe_ls, Sigma_ls_rafe = Sigma_ls_rafe,
             best_model_gamma = best_model_gamma, best_model_ls = best_model_ls,
             sign_agreement = sign_agreement,
             static_sign_flip = static_sign_flip, regime_split = regime_split,
             Sigma_ls = Sigma_ls, Sigma_ls_inv = Sigma_ls_inv,
             Sigma_inv_mkw = Sigma_inv_mkw, mkw_burnin = MKW_BURNIN,
             h3_vs_h1 = h3_vs_h1, h3_combined = h3_combined,
             sharpe_tests = sharpe_tests, sharpe_test_meta = sharpe_test_meta,
             subperiod_eval = subperiod_eval, sub_gap = sub_gap,
             rafe_split = rafe_split, hold_eval = hold_eval,
             best_split = best_split, best_model = best_model),
        file = "portfolio_results.rds")
cat("Saved: portfolio_results.rds\n")
