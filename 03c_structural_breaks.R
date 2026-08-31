# ==============================================================================
# 03c_structural_breaks.R
# Real-time structural break detection on the Fama-MacBeth gamma series using
# the online Change-Point Model (CPM) framework with the Generalized Likelihood
# Ratio (GLR) statistic.
#
# Method choice follows the supervisor's paper "Breaking Bad: Parameter
# Uncertainty Caused by Structural Breaks in Stocks" (Stöckl et al., 2026):
#   - Primary detector: GLR (joint break in mean AND variance) — best
#     power-to-false-positive tradeoff in the paper's simulation calibration
#     (detection prob. 59.4%, false-positive rate 6.8%; paper Table 2).
#   - Main specification: ARL0 = 20,000 (conservative in-control average run
#     length), startup = 20 observations — the paper's headline setting.
#   - Sensitivity: ARL0 ∈ {2,000, 5,000} (liberal / moderate; paper Table 24)
#     and four alternative statistics — Student-t (mean only), Bartlett
#     (variance only), Mann-Whitney / Mood (non-parametric location / scale).
#
# Why CPM/GLR instead of Bai-Perron: the CPM is an ONLINE detector — at each
# month t it uses only data up to t, so detected breaks carry a real-time
# detection date alongside the estimated break date. This is consistent with
# the thesis's expanding-window OOS design (no look-ahead), whereas Bai-Perron
# is a full-sample (offline) procedure. The gap between break date and
# detection date ("detection delay") is the paper's break-age / parameter-
# uncertainty window: within it, a real-time forecaster cannot yet know the
# regime has changed — the theoretical mechanism for why regime-shifting
# gammas (Value's 2007+ "value winter") resist real-time timing while
# persistent gammas (Size) remain predictable.
#
# CPM assumes serially independent observations under H0; the gammas are close
# to white noise (max AC1 ≈ 0.11, see 03_gamma_analysis.R), and the paper's
# calibration explicitly includes weakly autocorrelated processes, so the
# framework applies with minimal distortion.
#
# References:
#   Stöckl et al. (2026), "Breaking Bad" working paper — method + calibration
#   Ross (2015, J. Statistical Software 66/3) — the cpm R package
#   Hawkins, Qiu & Kang (2003, J. Quality Technology) — CPM framework
#   Andrews (1993, Econometrica) — caveat on Chow tests at estimated dates
#
# Input:  gammas_ts.rds  (from 02_fama_macbeth.R)
# Output: structural_breaks.rds  (breaks_glr, regime_stats, sens_grid, ...)
#         plots/structural_breaks_cpm.pdf
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(cpm)   # Ross (2015, JSS) — online change-point detection

list2env(readRDS("gammas_ts.rds"), envir = environment())  # -> gammas_ts, fm_table, ...

dir.create("plots", showWarnings = FALSE)

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

# Main specification (Breaking Bad headline setting)
ARL0_MAIN <- 20000L
STARTUP   <- 20L

# ------------------------------------------------------------------------------
# (1) CPM detection helper
# ------------------------------------------------------------------------------
# processStream() monitors the series sequentially; on each detection the CPM
# restarts after the detection point, so multiple breaks are handled natively.
# changePoints    = estimated break locations (indices into the stream)
# detectionTimes  = month in which the detector actually FLAGGED the break
# delay           = detectionTimes - changePoints (the break-age window)
run_cpm <- function(x, dates, cpm_type, arl0, startup = STARTUP) {
  res <- tryCatch(
    processStream(x, cpmType = cpm_type, ARL0 = arl0, startup = startup),
    error = function(e) NULL   # some (type, ARL0) combos lack lookup tables
  )
  if (is.null(res)) return(NULL)
  if (length(res$changePoints) == 0) {
    return(tibble(
      break_idx = integer(), detect_idx = integer(),
      break_date = as.Date(character()), detect_date = as.Date(character()),
      delay_months = integer()
    ))
  }
  tibble(
    break_idx    = res$changePoints,
    detect_idx   = res$detectionTimes,
    break_date   = dates[res$changePoints],
    detect_date  = dates[res$detectionTimes],
    delay_months = res$detectionTimes - res$changePoints
  )
}

# ------------------------------------------------------------------------------
# (2) Main run: GLR at ARL0 = 20,000 per gamma
# ------------------------------------------------------------------------------
breaks_glr <- map_dfr(gamma_cols, function(gc) {
  ok    <- !is.na(gammas_ts[[gc]])
  x     <- gammas_ts[[gc]][ok]
  dates <- gammas_ts$date[ok]
  out   <- run_cpm(x, dates, "GLR", ARL0_MAIN)
  if (is.null(out) || nrow(out) == 0) {
    return(tibble(
      gamma = gamma_labels[gc], break_idx = NA_integer_, detect_idx = NA_integer_,
      break_date = as.Date(NA), detect_date = as.Date(NA), delay_months = NA_integer_
    ))
  }
  out |> mutate(gamma = gamma_labels[gc], .before = 1)
})

cat("\n=== CPM/GLR Structural Breaks (ARL0 = 20,000, startup = 20) ===\n")
cat("break_date  = estimated break location | detect_date = real-time flag date\n")
cat("delay       = detection delay in months (break-age / ambiguity window)\n\n")
print(breaks_glr, n = Inf)

# ------------------------------------------------------------------------------
# (3) Regime statistics around each GLR break
# ------------------------------------------------------------------------------
# Segment each gamma at its estimated break dates; report mean / sd per adjacent
# regime pair so the economic content of each break (mean shift vs. vol shift)
# is visible. Units: % per month.
regime_stats <- map_dfr(gamma_cols, function(gc) {
  gl  <- gamma_labels[gc]
  brk <- breaks_glr |> filter(gamma == gl, !is.na(break_idx))
  if (nrow(brk) == 0) return(NULL)

  ok    <- !is.na(gammas_ts[[gc]])
  x     <- gammas_ts[[gc]][ok]
  n     <- length(x)
  edges <- c(0L, brk$break_idx, n)   # regime j = (edges[j]+1) : edges[j+1]

  map_dfr(seq_len(nrow(brk)), function(j) {
    pre  <- x[(edges[j] + 1L):edges[j + 1L]]
    post <- x[(edges[j + 1L] + 1L):edges[j + 2L]]
    tibble(
      gamma        = gl,
      break_date   = brk$break_date[j],
      detect_date  = brk$detect_date[j],
      delay_months = brk$delay_months[j],
      n_pre        = length(pre),
      n_post       = length(post),
      mean_pre     = round(mean(pre)  * 100, 3),
      mean_post    = round(mean(post) * 100, 3),
      sd_pre       = round(sd(pre)    * 100, 3),
      sd_post      = round(sd(post)   * 100, 3)
    )
  })
})

cat("\n=== Regime Statistics Around GLR Breaks (% per month) ===\n")
print(regime_stats, n = Inf)

# ------------------------------------------------------------------------------
# (4) Chow cross-check at the CPM-estimated break dates
# ------------------------------------------------------------------------------
# Complements the fixed-date Chow tests (2000 / 2010) in 00_diagnostics.R §12.
# CAVEAT: the break dates are estimated by a search, so the standard Chow
# F-distribution understates the true critical values (Andrews 1993 sup-F
# problem). Reported as a descriptive cross-check only — the CPM detection
# itself is the formal test (its threshold already controls the false-positive
# rate over the sequential search).
chow_test <- function(y, breakpoint) {
  n <- length(y)
  if (breakpoint < 3 || breakpoint > n - 3) return(NULL)
  x_full <- y[-n]; y_full <- y[-1]
  rss_r  <- sum(resid(lm(y_full ~ x_full))^2)
  x1 <- y[1:(breakpoint - 1)];   y1 <- y[2:breakpoint]
  x2 <- y[breakpoint:(n - 1)];   y2 <- y[(breakpoint + 1):n]
  rss_u <- sum(resid(lm(y1 ~ x1))^2) + sum(resid(lm(y2 ~ x2))^2)
  k <- 2L
  f_stat <- ((rss_r - rss_u) / k) / (rss_u / (n - 2L * k))
  list(f = round(f_stat, 3), p = round(1 - pf(f_stat, k, n - 2L * k), 4))
}

chow_at_breaks <- map_dfr(gamma_cols, function(gc) {
  gl  <- gamma_labels[gc]
  brk <- breaks_glr |> filter(gamma == gl, !is.na(break_idx))
  if (nrow(brk) == 0) return(NULL)
  x <- gammas_ts[[gc]][!is.na(gammas_ts[[gc]])]
  map_dfr(seq_len(nrow(brk)), function(j) {
    ct <- chow_test(x, brk$break_idx[j])
    tibble(
      gamma      = gl,
      break_date = brk$break_date[j],
      chow_f     = if (!is.null(ct)) ct$f else NA_real_,
      chow_p     = if (!is.null(ct)) ct$p else NA_real_
    )
  })
})

cat("\n=== Chow Test at CPM-Estimated Break Dates (descriptive cross-check) ===\n")
cat("Andrews (1993): p-values at estimated dates are anti-conservative.\n\n")
print(chow_at_breaks, n = Inf)

# ------------------------------------------------------------------------------
# (5) Sensitivity grid: 5 statistics × 3 ARL0 thresholds
# ------------------------------------------------------------------------------
# Mirrors Breaking Bad Table 24: break counts under liberal (2,000), moderate
# (5,000) and conservative (20,000) thresholds for all five CPM statistics.
cpm_types <- c("GLR", "Student", "Bartlett", "Mann-Whitney", "Mood")
arl0_grid <- c(2000L, 5000L, 20000L)

sens_grid <- map_dfr(gamma_cols, function(gc) {
  ok    <- !is.na(gammas_ts[[gc]])
  x     <- gammas_ts[[gc]][ok]
  dates <- gammas_ts$date[ok]
  map_dfr(cpm_types, function(tp) {
    map_dfr(arl0_grid, function(a0) {
      out <- run_cpm(x, dates, tp, a0)
      tibble(
        gamma    = gamma_labels[gc],
        cpm_type = tp,
        ARL0     = a0,
        n_breaks = if (is.null(out)) NA_integer_ else nrow(out),
        break_dates = if (is.null(out) || nrow(out) == 0) "" else
          paste(format(out$break_date, "%Y-%m"), collapse = "; ")
      )
    })
  })
})

cat("\n=== Sensitivity: Break Counts Across Statistics x ARL0 ===\n")
sens_wide <- sens_grid |>
  select(gamma, cpm_type, ARL0, n_breaks) |>
  pivot_wider(names_from = ARL0, values_from = n_breaks, names_prefix = "ARL0_")
print(sens_wide, n = Inf)

cat("\nGLR break dates by threshold:\n")
print(sens_grid |> filter(cpm_type == "GLR") |> select(gamma, ARL0, break_dates), n = Inf)

# ------------------------------------------------------------------------------
# (6) Validation of the sub-period cuts used in the OOS analysis
# ------------------------------------------------------------------------------
# The OOS R² sub-period breakdown (04_predict_gammas.R) cuts at 2000-01 and
# 2010-01. Check how far each ad-hoc cut lies from the nearest GLR break.
subperiod_cuts <- as.Date(c("2000-01-01", "2010-01-01"))

cut_validation <- map_dfr(unique(breaks_glr$gamma), function(gl) {
  bd <- breaks_glr |> filter(gamma == gl, !is.na(break_date)) |> pull(break_date)
  map_dfr(subperiod_cuts, function(cut) {
    tibble(
      gamma           = gl,
      cut_date        = cut,
      nearest_break   = if (length(bd) == 0) as.Date(NA) else bd[which.min(abs(bd - cut))],
      dist_months     = if (length(bd) == 0) NA_real_ else
        round(min(abs(as.numeric(bd - cut))) / 30.44, 1)
    )
  })
})

cat("\n=== Sub-Period Cut Validation (2000-01 / 2010-01 vs nearest GLR break) ===\n")
print(cut_validation, n = Inf)

# ------------------------------------------------------------------------------
# (7) Plot: gamma series with breaks (solid) and detection dates (dashed)
# ------------------------------------------------------------------------------
gammas_long <- gammas_ts |>
  select(date, all_of(gamma_cols)) |>
  pivot_longer(-date, names_to = "gamma", values_to = "value") |>
  mutate(gamma = factor(gamma_labels[gamma], levels = gamma_labels),
         value_pct = value * 100)

vlines <- breaks_glr |>
  filter(!is.na(break_date)) |>
  mutate(gamma = factor(gamma, levels = gamma_labels)) |>
  select(gamma, break_date, detect_date) |>
  pivot_longer(-gamma, names_to = "line_type", values_to = "date") |>
  mutate(line_type = recode(line_type,
                            break_date  = "Estimated break",
                            detect_date = "Real-time detection"))

p_breaks <- ggplot(gammas_long, aes(x = date, y = value_pct)) +
  geom_line(linewidth = 0.3, colour = "grey40") +
  geom_hline(yintercept = 0, linewidth = 0.3, linetype = "dotted") +
  geom_vline(data = vlines,
             aes(xintercept = date, colour = line_type, linetype = line_type),
             linewidth = 0.5) +
  scale_colour_manual(values = c("Estimated break"     = "firebrick",
                                 "Real-time detection" = "steelblue")) +
  scale_linetype_manual(values = c("Estimated break"     = "solid",
                                   "Real-time detection" = "dashed")) +
  facet_wrap(~gamma, ncol = 2, scales = "free_y") +
  labs(
    title    = "Structural Breaks in Gamma Series — Online CPM/GLR",
    subtitle = sprintf(
      "ARL0 = %d, startup = %d | gap between red and blue = detection delay (break-age)",
      ARL0_MAIN, STARTUP),
    x = NULL, y = "Gamma (% per month)", colour = NULL, linetype = NULL
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave("plots/structural_breaks_cpm.pdf", p_breaks, width = 10, height = 8)
cat("\nSaved: plots/structural_breaks_cpm.pdf\n")

# ------------------------------------------------------------------------------
# (8) Persist results
# ------------------------------------------------------------------------------
saveRDS(
  list(
    breaks_glr     = breaks_glr,
    regime_stats   = regime_stats,
    chow_at_breaks = chow_at_breaks,
    sens_grid      = sens_grid,
    cut_validation = cut_validation,
    spec           = list(cpm_type = "GLR", ARL0 = ARL0_MAIN, startup = STARTUP)
  ),
  "structural_breaks.rds"
)
cat("Saved: structural_breaks.rds\n")

# ------------------------------------------------------------------------------
# (9) Narrative summary for the writeup
# ------------------------------------------------------------------------------
n_brk <- breaks_glr |> filter(!is.na(break_idx)) |> count(gamma)
cat("\n=== Summary ===\n")
cat("Breaks per gamma (GLR, ARL0 = 20,000):\n")
print(n_brk, n = Inf)
med_delay <- median(breaks_glr$delay_months, na.rm = TRUE)
cat(sprintf(
  "\nMedian detection delay: %.0f months (Breaking Bad pooled median: 14-18).\n",
  med_delay))
cat("Interpretation: within the delay window a real-time forecaster cannot yet\n")
cat("know the regime changed — regime-shifting gammas resist real-time timing\n")
cat("(Value winter), persistent gammas (Size) remain timeable.\n")
