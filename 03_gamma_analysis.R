# ==============================================================================
# 03_gamma_analysis.R
# Descriptive analysis of the Fama-MacBeth gamma time series:
#   - Summary statistics (mean, sd, persistence, autocorrelations)
#   - Correlation with Fama-French factor returns
#   - Time series plots
#
# Input:  gammas_ts.rds  (from 02_fama_macbeth.R)
#         tidy_finance SQLite (for FF factors)
# Output: plots saved to /plots/, tables printed to console
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(kableExtra)
library(vars)
select <- dplyr::select   # vars loads MASS which masks dplyr::select

list2env(readRDS("gammas_ts.rds"), envir = environment())  # -> gammas_ts, fm_table, gammas_ts_ols, fm_table_ols
ff5_factors <- readRDS("ff5_factors.rds")  # -> ff5_factors

dir.create("plots", showWarnings = FALSE)

gamma_cols <- c(
  "gamma_bm", "gamma_mom12m", "gamma_oper_prof",
  "gamma_asset_growth", "gamma_size", "gamma_beta"
)

gamma_labels <- c(
  gamma_bm           = "Value (BM)",
  gamma_mom12m       = "Momentum",
  gamma_oper_prof    = "Profitability",
  gamma_asset_growth = "Asset Growth",
  gamma_size         = "Size",
  gamma_beta         = "Beta"
)


# (1) Summary statistics -------------------------------------------------------

# Newey-West t-stat for mean != 0
nw_tstat <- function(x, lag = 12) {
  library(lmtest); library(sandwich)
  x <- x[!is.na(x)]
  ct <- coeftest(lm(x ~ 1), vcov = NeweyWest(lm(x ~ 1), lag = lag, prewhite = FALSE))
  ct["(Intercept)", "t value"]
}

# AR(1) coefficient as a measure of persistence
ar1 <- function(x) {
  x <- x[!is.na(x)]
  coef(lm(x[-1] ~ x[-length(x)]))[2]
}

desc_stats <- gammas_ts |>
  summarise(across(
    all_of(gamma_cols),
    list(
      mean  = \(x) mean(x,   na.rm = TRUE) * 100,
      sd    = \(x) sd(x,     na.rm = TRUE) * 100,
      min   = \(x) min(x,    na.rm = TRUE) * 100,
      max   = \(x) max(x,    na.rm = TRUE) * 100,
      t_nw  = \(x) nw_tstat(x, lag = 12),
      ar1   = \(x) ar1(x)
    )
  )) |>
  pivot_longer(
    everything(),
    names_to      = c("gamma", ".value"),
    names_pattern = "(.+)_(mean|sd|min|max|t_nw|ar1)"
  ) |>
  mutate(
    gamma = recode(gamma, !!!gamma_labels),
    across(c(mean, sd, min, max), \(x) round(x, 3)),
    across(c(t_nw, ar1),          \(x) round(x, 2))
  )

cat("\n=== Gamma Descriptive Statistics (% per month) ===\n")
print(desc_stats, n = Inf)


# (1b) Sub-period analysis -----------------------------------------------------
# Structural instability: characteristics (especially size) have weakened
# post-publication. Reference: McLean & Pontiff (2016).
sub_periods <- list(
  "Pre-2000"  = \(d) d < as.Date("2000-01-01"),
  "2000-2010" = \(d) d >= as.Date("2000-01-01") & d < as.Date("2010-01-01"),
  "Post-2010" = \(d) d >= as.Date("2010-01-01")
)

sub_stats <- map_dfr(names(sub_periods), function(sp) {
  sub <- gammas_ts |> filter(sub_periods[[sp]](date))
  if (nrow(sub) < 12) return(NULL)
  map_dfr(gamma_cols, function(gc) {
    x <- sub[[gc]]
    t_s <- nw_tstat(x, lag = 12)
    tibble(
      sub_period     = sp,
      gamma          = gamma_labels[gc],
      n_months       = sum(!is.na(x)),
      mean_pct       = round(mean(x, na.rm = TRUE) * 100, 3),
      sd_pct         = round(sd(x,   na.rm = TRUE) * 100, 3),
      t_nw           = round(t_s, 2)
    )
  })
})

cat("\n=== Sub-Period Gamma Statistics (% per month, NW-12 t-stat) ===\n")
print(sub_stats |> arrange(gamma, sub_period), n = Inf)


# (2) Autocorrelation structure ------------------------------------------------
max_lag <- 12

acf_tbl <- map_dfr(gamma_cols, function(gc) {
  x <- gammas_ts[[gc]]
  x <- x[!is.na(x)]
  ac <- acf(x, lag.max = max_lag, plot = FALSE)$acf[-1]   # drop lag-0
  tibble(
    gamma = gamma_labels[gc],
    lag   = seq_len(max_lag),
    acf   = as.numeric(ac)
  )
})

p_acf <- ggplot(acf_tbl, aes(x = lag, y = acf)) +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  geom_hline(
    yintercept = c(-1, 1) * qnorm(0.975) / sqrt(nrow(gammas_ts)),
    linetype = "dashed", colour = "steelblue", linewidth = 0.4
  ) +
  geom_col(fill = "steelblue", width = 0.6) +
  facet_wrap(~gamma, ncol = 3) +
  scale_x_continuous(breaks = seq(2, max_lag, 2)) +
  labs(title = "Autocorrelation of Gamma Time Series",
       x = "Lag (months)", y = "ACF") +
  theme_bw()

ggsave("plots/acf_gammas.pdf", p_acf, width = 10, height = 6)
cat("Saved: plots/acf_gammas.pdf\n")


# (3) Correlation with Fama-French factors -------------------------------------
# Map: which FF factor is conceptually closest to each gamma
ff_labels <- c(
  mkt_excess = "MKT",
  smb        = "SMB (size)",
  hml        = "HML (value)",
  rmw        = "RMW (profit.)",
  cma        = "CMA (invest.)"
)

gamma_ff <- gammas_ts |>
  select(date, all_of(gamma_cols)) |>
  inner_join(ff5_factors, by = "date")

# Full correlation matrix: gammas × FF factors
cor_mat <- cor(
  gamma_ff |> select(all_of(gamma_cols)),
  gamma_ff |> select(mkt_excess, smb, hml, rmw, cma),
  use = "pairwise.complete.obs"
)
rownames(cor_mat) <- gamma_labels[rownames(cor_mat)]
colnames(cor_mat) <- ff_labels[colnames(cor_mat)]

cat("\n=== Gamma × Fama-French Factor Correlations ===\n")
print(round(cor_mat, 2))


# (4) Time series plots --------------------------------------------------------
gammas_long <- gammas_ts |>
  select(date, all_of(gamma_cols)) |>
  pivot_longer(-date, names_to = "gamma", values_to = "value") |>
  mutate(
    gamma      = recode(gamma, !!!gamma_labels),
    value_pct  = value * 100
  )

# Shade the value series' break-age window (break estimated 2001-03, detected
# in real time 2007-09; 03c_structural_breaks.R) — links this exhibit visually
# to the structural-breaks figure. Data-frame rect → drawn in the Value facet only.
value_break_win <- tibble(
  gamma = gamma_labels[["gamma_bm"]],   # must match the recoded facet label exactly
  xmin  = as.Date("2001-03-01"), xmax = as.Date("2007-09-01")
)
p_ts <- ggplot(gammas_long, aes(x = date, y = value_pct)) +
  geom_rect(data = value_break_win, aes(xmin = xmin, xmax = xmax),
            ymin = -Inf, ymax = Inf, inherit.aes = FALSE,
            fill = "grey50", alpha = 0.18) +
  geom_line(linewidth = 0.4, colour = "steelblue") +
  geom_hline(yintercept = 0, linewidth = 0.3, linetype = "dashed") +
  geom_smooth(method = "loess", span = 0.2, se = FALSE,
              colour = "firebrick", linewidth = 0.7) +
  facet_wrap(~gamma, ncol = 2, scales = "free_y") +
  labs(title = "Fama-MacBeth Risk Premia (Gammas) Over Time",
       x = NULL, y = "Gamma (% per month)") +
  theme_bw()

ggsave("plots/gammas_timeseries.pdf", p_ts, width = 10, height = 10)
cat("Saved: plots/gammas_timeseries.pdf\n")


# (5) Cross-sectional R² over time ---------------------------------------------
p_rsq <- ggplot(gammas_ts, aes(x = date, y = r_sq * 100)) +
  geom_line(linewidth = 0.4, colour = "steelblue") +
  geom_smooth(method = "loess", span = 0.2, se = FALSE,
              colour = "firebrick", linewidth = 0.7) +
  labs(title = "Cross-Sectional R² of Monthly FM Regressions",
       x = NULL, y = "R² (%)") +
  theme_bw()

ggsave("plots/rsq_timeseries.pdf", p_rsq, width = 10, height = 4)
cat("Saved: plots/rsq_timeseries.pdf\n")

cat(sprintf(
  "\nMean R²: %.1f%% | Median R²: %.1f%%\n",
  mean(gammas_ts$r_sq) * 100,
  median(gammas_ts$r_sq) * 100
))


# (6) VAR impulse response functions -------------------------------------------
# Fit a full-sample VAR on all six gammas to characterise dynamic spillovers.
# Identification: Cholesky decomposition (ordering = gamma_cols order, i.e.
#   Value → Momentum → Profitability → Asset Growth → Size → Beta).
# Ordering matters for contemporaneous effects; given the low cross-correlations
# among gammas the results are generally robust to reordering.
# Reference: Lütkepohl (2005), "New Introduction to Multiple Time Series Analysis".

gamma_var_data <- gammas_ts |>
  select(all_of(gamma_cols)) |>
  as.matrix()
gamma_var_data <- gamma_var_data[complete.cases(gamma_var_data), ]

# Select optimal lag order by AIC, capped at 6 to limit overparameterisation
lag_sel <- VARselect(gamma_var_data, lag.max = 6, type = "const")
cat("\n=== VAR lag-order selection ===\n")
print(lag_sel$selection)
p_opt <- min(lag_sel$selection["AIC(n)"], 3L)   # cap at 3 as robustness guard
cat(sprintf("Using p = %d\n", p_opt))

var_full <- VAR(gamma_var_data, p = p_opt, type = "const")

# Granger-causality summary: does any gamma Granger-cause another?
cat("\n=== Granger causality (joint test per equation) ===\n")
for (gc in gamma_cols) {
  gc_label <- gamma_labels[gc]
  res <- causality(var_full, cause = gc)
  cat(sprintf(
    "  %s → others: F = %.2f, p = %.3f\n",
    gc_label, res$Granger$statistic, res$Granger$p.value
  ))
}

# Compute orthogonalised IRFs with bootstrap confidence bands
N_AHEAD <- 24    # 2-year horizon
BOOTS   <- 500
set.seed(42)
cat(sprintf(
  "\nComputing IRFs (n.ahead = %d, bootstrap runs = %d)...\n",
  N_AHEAD, BOOTS
))
irf_res <- irf(var_full, n.ahead = N_AHEAD, boot = TRUE,
               runs = BOOTS, ci = 0.95, ortho = TRUE)

# Extract into tidy tibble — irf_res$irf / $Lower / $Upper are named lists
# where each element is a (N_AHEAD+1) × n_vars matrix (horizon 0..N_AHEAD)
extract_irf <- function(irf_obj, slot) {
  map_dfr(names(irf_obj[[slot]]), function(impulse) {
    mat <- irf_obj[[slot]][[impulse]]
    map_dfr(seq_len(ncol(mat)), function(j) {
      tibble(
        impulse  = impulse,
        response = colnames(mat)[j],
        horizon  = seq(0, N_AHEAD),
        value    = mat[, j]
      )
    })
  })
}

irf_tbl <- extract_irf(irf_res, "irf")   |> rename(irf   = value)
lb_tbl  <- extract_irf(irf_res, "Lower") |> rename(lower = value)
ub_tbl  <- extract_irf(irf_res, "Upper") |> rename(upper = value)

irf_plot <- irf_tbl |>
  left_join(lb_tbl, by = c("impulse", "response", "horizon")) |>
  left_join(ub_tbl, by = c("impulse", "response", "horizon")) |>
  mutate(
    # Scale from decimal to % per month (same units as the time-series plots)
    across(c(irf, lower, upper), \(x) x * 100),
    impulse  = factor(gamma_labels[impulse],  levels = gamma_labels),
    response = factor(gamma_labels[response], levels = gamma_labels)
  )

# Full 6×6 grid: rows = response, columns = impulse
p_irf <- ggplot(irf_plot, aes(x = horizon)) +
  geom_hline(yintercept = 0, linewidth = 0.3, linetype = "dashed") +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              fill = "steelblue", alpha = 0.2) +
  geom_line(aes(y = irf), colour = "steelblue", linewidth = 0.6) +
  facet_grid(response ~ impulse, scales = "free_y") +
  scale_x_continuous(breaks = seq(0, N_AHEAD, 6)) +
  labs(
    title    = sprintf(
      "Orthogonalised Impulse Response Functions — VAR(%d)", p_opt),
    subtitle = sprintf(
      "95%% bootstrap CI (%d runs) | Cholesky ordering: %s",
      BOOTS, paste(gamma_labels, collapse = " → ")),
    x = "Horizon (months)",
    y = "Response (pp / month)"
  ) +
  theme_bw(base_size = 9) +
  theme(
    strip.text   = element_text(size = 7),
    panel.spacing = unit(0.3, "lines")
  )

ggsave("plots/irf_var.pdf", p_irf, width = 14, height = 12)
cat("Saved: plots/irf_var.pdf\n")

# Compact version: diagonal only (own-shock responses)
irf_diag <- irf_plot |> filter(as.character(impulse) == as.character(response))

p_irf_diag <- ggplot(irf_diag, aes(x = horizon)) +
  geom_hline(yintercept = 0, linewidth = 0.3, linetype = "dashed") +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              fill = "steelblue", alpha = 0.25) +
  geom_line(aes(y = irf), colour = "steelblue", linewidth = 0.7) +
  facet_wrap(~impulse, ncol = 3, scales = "free_y") +
  scale_x_continuous(breaks = seq(0, N_AHEAD, 6)) +
  labs(
    title    = sprintf("Own-shock IRFs — VAR(%d)", p_opt),
    subtitle = "Response of each gamma to a 1 SD shock in itself | 95% bootstrap CI",
    x = "Horizon (months)",
    y = "Response (pp / month)"
  ) +
  theme_bw()

ggsave("plots/irf_var_diagonal.pdf", p_irf_diag, width = 10, height = 6)
cat("Saved: plots/irf_var_diagonal.pdf\n")
