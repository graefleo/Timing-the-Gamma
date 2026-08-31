# ==============================================================================
# 03b_visual_exploration.R
# Visual diagnostics for gammas and their predictors — run AFTER
# 04_predict_gammas.R (needs gamma_predictions.rds for the aligned matrix).
#
# Plots produced:
#   1.  Gamma–gamma correlation heatmap
#   2.  Gamma time series — zoomed decade panels (1963-79, 1980-99, 2000-09, 2010-23)
#   3.  All gammas overlaid — 1990-2023 on one panel per sub-period
#   4.  Macro predictor time series (raw values)
#   5.  Macro × gamma correlation heatmap
#   6.  Own-feature × gamma correlation heatmap
#          (each gamma's l1/l2/l3/vol12/spread against all 6 gammas)
#   7.  Full feature × gamma correlation heatmap (42 × 6)
#   8.  Feature–feature correlation heatmap (macro + vol + spread block)
#
# Input:  gammas_ts.rds, macro_predictors.rds, gamma_predictions.rds
# Output: plots/exploration/*.pdf
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(ggcorrplot)   # install.packages("ggcorrplot") if missing
library(car)          # vif()

list2env(readRDS("gammas_ts.rds"), envir = environment())          # -> gammas_ts, fm_table, ...
macro_pred <- readRDS("macro_predictors.rds")                      # -> macro_pred
gw_pred    <- readRDS("goyal_welch_predictors.rds")                # -> gw_pred   (from 01d_goyal_welch.R)
list2env(readRDS("gamma_predictions.rds"), envir = environment())  # -> pred_list, aligned, feature_cols, oos_idx

dir.create("plots/exploration", showWarnings = FALSE, recursive = TRUE)

safe_save <- function(path, plot, ...) {
  tryCatch(
    { ggsave(path, plot, ...); cat("Saved:", path, "\n") },
    error = function(e) cat("SKIPPED (file locked):", path, "\n")
  )
}

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
macro_cols <- c("default_spread", "term_spread", "short_rate",
                "vix", "indpro_gap", "mkt_lag1", "infl", "lty")
macro_labels <- c(
  default_spread = "Default spread",
  term_spread    = "Term spread",
  short_rate     = "Short rate (3M)",
  vix            = "VIX",
  indpro_gap     = "IP growth (12m)",
  mkt_lag1       = "Mkt ret (lag 1)",
  infl           = "CPI inflation",
  lty            = "LT yield (20Y)"
)

# Goyal-Welch (2008) equity premium predictors — subset to what exists in gw_pred
gw_cols_all <- c("dp", "ep", "de", "bm", "ntis", "svar",
                  "ltr", "dfr", "infl", "lty", "ik")
# dy omitted: near-perfectly collinear with dp (VIF > 80); dp is the standard GW spec
gw_cols <- intersect(gw_cols_all, names(gw_pred))
gw_labels_all <- c(
  dp   = "D/P (log dividend-price)",
  dy   = "D/Y (log dividend-yield)",
  ep   = "E/P (earnings-price)",
  de   = "D/E (payout ratio)",
  bm   = "B/M (S&P 500)",
  ntis = "Net equity issuance",
  svar = "Stock return variance",
  ltr  = "LT gov't bond return",
  dfr  = "Default return spread",
  infl = "CPI inflation (GW)",
  lty  = "LT yield (GW)",
  ik   = "Investment-to-capital"
)
gw_labels <- gw_labels_all[gw_cols]


# ==============================================================================
# 1. Gamma–gamma correlation heatmap
# ==============================================================================
cat("\n--- Plot 1: Gamma–gamma correlations ---\n")

cor_gg <- cor(
  gammas_ts[, gamma_cols],
  use = "pairwise.complete.obs"
)
rownames(cor_gg) <- colnames(cor_gg) <- gamma_labels[gamma_cols]

# Melt to tidy
cor_gg_tbl <- as.data.frame(cor_gg) |>
  rownames_to_column("gamma_y") |>
  pivot_longer(-gamma_y, names_to = "gamma_x", values_to = "r") |>
  mutate(
    gamma_y = factor(gamma_y, levels = rev(gamma_labels)),
    gamma_x = factor(gamma_x, levels = gamma_labels),
    label   = sprintf("%.2f", r)
  )

p1 <- ggplot(cor_gg_tbl, aes(x = gamma_x, y = gamma_y, fill = r)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = label), size = 3.5, fontface = "bold") +
  scale_fill_gradient2(
    low = "firebrick", mid = "white", high = "steelblue",
    midpoint = 0, limits = c(-1, 1), name = "Pearson r"
  ) +
  labs(
    title    = "Gamma–Gamma Correlation Matrix",
    subtitle = "Pairwise Pearson correlations | Full sample (1963–2023)",
    x = NULL, y = NULL
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

safe_save("plots/exploration/p1_gamma_gamma_corr.pdf", p1, width = 7, height = 6)


# ==============================================================================
# 2. Gamma time series — zoomed decade panels
# ==============================================================================
cat("\n--- Plot 2: Decade-panel time series ---\n")

decade_breaks <- list(
  "1963–1979" = c(as.Date("1963-01-01"), as.Date("1979-12-31")),
  "1980–1999" = c(as.Date("1980-01-01"), as.Date("1999-12-31")),
  "2000–2009" = c(as.Date("2000-01-01"), as.Date("2009-12-31")),
  "2010–2023" = c(as.Date("2010-01-01"), as.Date("2023-12-31"))
)

gammas_long <- gammas_ts |>
  select(date, all_of(gamma_cols)) |>
  pivot_longer(-date, names_to = "gamma", values_to = "value") |>
  mutate(
    gamma     = factor(gamma_labels[gamma], levels = gamma_labels),
    value_pct = value * 100
  )

# One page per decade, all 6 gammas in facets
for (dname in names(decade_breaks)) {
  drange <- decade_breaks[[dname]]
  sub <- gammas_long |>
    filter(date >= drange[1], date <= drange[2])

  p2 <- ggplot(sub, aes(x = date, y = value_pct)) +
    geom_hline(yintercept = 0, linewidth = 0.3, linetype = "dashed", colour = "grey50") +
    geom_line(linewidth = 0.4, colour = "steelblue") +
    geom_smooth(method = "loess", span = 0.3, se = TRUE,
                colour = "firebrick", fill = "firebrick", alpha = 0.12, linewidth = 0.7) +
    facet_wrap(~gamma, ncol = 2, scales = "free_y") +
    scale_x_date(date_labels = "%Y", date_breaks = "2 years") +
    labs(
      title    = paste0("Fama-MacBeth Risk Premia — ", dname),
      subtitle = "Monthly WLS gammas (% per month) with LOESS trend ± SE",
      x = NULL, y = "Gamma (% / month)"
    ) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  fname <- paste0("plots/exploration/p2_gammas_",
                  gsub("[^0-9]", "", dname), ".pdf")
  safe_save(fname, p2, width = 10, height = 10)
}


# ==============================================================================
# 3. All gammas overlaid — three sub-periods, one line per gamma
# ==============================================================================
cat("\n--- Plot 3: Gammas overlaid by sub-period ---\n")

sub_windows <- list(
  "Pre-2000 (OOS: Jan 1990 – Dec 1999)" =
    c(as.Date("1990-01-01"), as.Date("1999-12-31")),
  "2000–2009" =
    c(as.Date("2000-01-01"), as.Date("2009-12-31")),
  "2010–2023" =
    c(as.Date("2010-01-01"), as.Date("2023-12-31"))
)

gammas_sub <- map_dfr(names(sub_windows), function(sp) {
  rng <- sub_windows[[sp]]
  gammas_long |>
    filter(date >= rng[1], date <= rng[2]) |>
    mutate(sub_period = factor(sp, levels = names(sub_windows)))
})

p3 <- ggplot(gammas_sub, aes(x = date, y = value_pct, colour = gamma)) +
  geom_hline(yintercept = 0, linewidth = 0.3, linetype = "dashed", colour = "grey60") +
  geom_line(linewidth = 0.45, alpha = 0.85) +
  facet_wrap(~sub_period, ncol = 1, scales = "free_x") +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "1 year") +
  scale_colour_brewer(palette = "Set1", name = NULL) +
  labs(
    title    = "All Six Gammas Overlaid — OOS Sub-periods",
    subtitle = "Each line = one risk premium (% per month)",
    x = NULL, y = "Gamma (% / month)"
  ) +
  theme_bw() +
  theme(
    legend.position  = "bottom",
    axis.text.x      = element_text(angle = 45, hjust = 1),
    panel.spacing    = unit(0.8, "lines")
  )

safe_save("plots/exploration/p3_gammas_overlaid_subperiods.pdf",
          p3, width = 14, height = 12)


# ==============================================================================
# 4. Macro predictor time series
# ==============================================================================
cat("\n--- Plot 4: Macro predictor time series ---\n")

macro_long <- macro_pred |>
  select(ym, all_of(macro_cols)) |>
  mutate(date = as.Date(paste0(ym, "-01"))) |>
  pivot_longer(all_of(macro_cols), names_to = "variable", values_to = "value") |>
  mutate(variable = factor(macro_labels[variable], levels = macro_labels))

p4 <- ggplot(macro_long, aes(x = date, y = value)) +
  geom_line(linewidth = 0.4, colour = "steelblue", na.rm = TRUE) +
  geom_smooth(method = "loess", span = 0.2, se = FALSE,
              colour = "firebrick", linewidth = 0.6, na.rm = TRUE) +
  facet_wrap(~variable, ncol = 2, scales = "free_y") +
  scale_x_date(date_labels = "%Y", date_breaks = "5 years") +
  labs(
    title    = "Macro Conditioning Variables Over Time",
    subtitle = "FRED series used as predictors in LASSO / MLP",
    x = NULL, y = NULL
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

safe_save("plots/exploration/p4_macro_timeseries.pdf", p4, width = 10, height = 10)


# ==============================================================================
# 4b. Goyal-Welch predictor time series
# ==============================================================================
cat("\n--- Plot 4b: Goyal-Welch predictor time series ---\n")

if (length(gw_cols) > 0) {
  gw_long <- gw_pred |>
    select(ym, all_of(gw_cols)) |>
    mutate(date = as.Date(paste0(ym, "-01"))) |>
    pivot_longer(all_of(gw_cols), names_to = "variable", values_to = "value") |>
    mutate(variable = factor(gw_labels[variable], levels = gw_labels))

  p4b <- ggplot(gw_long, aes(x = date, y = value)) +
    geom_line(linewidth = 0.4, colour = "steelblue", na.rm = TRUE) +
    geom_smooth(method = "loess", span = 0.2, se = FALSE,
                colour = "firebrick", linewidth = 0.6, na.rm = TRUE) +
    facet_wrap(~variable, ncol = 3, scales = "free_y") +
    scale_x_date(date_labels = "%Y", date_breaks = "10 years") +
    labs(
      title    = "Goyal-Welch (2008) Equity Premium Predictors Over Time",
      subtitle = "Used as additional features in LASSO / RF alongside macro variables",
      x = NULL, y = NULL
    ) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  safe_save("plots/exploration/p4b_gw_timeseries.pdf", p4b,
            width = 12, height = ceiling(length(gw_cols) / 3) * 3 + 2)
} else {
  cat("No GW predictors found in gw_pred — skipping plot 4b.\n")
}


# ==============================================================================
# 5. Macro × gamma correlation heatmap
# ==============================================================================
cat("\n--- Plot 5: Macro × gamma correlations ---\n")

# Join on year-month string (same approach as 00_diagnostics.R to avoid off-by-one)
gammas_ym <- gammas_ts |>
  mutate(ym = format(date, "%Y-%m")) |>
  select(ym, all_of(gamma_cols))

macro_ym <- macro_pred |> select(ym, all_of(macro_cols))

joint <- inner_join(gammas_ym, macro_ym, by = "ym")

cor_mg <- map_dfr(macro_cols, function(mc) {
  map_dfr(gamma_cols, function(gc) {
    tibble(
      macro = macro_labels[mc],
      gamma = gamma_labels[gc],
      r     = cor(joint[[mc]], joint[[gc]], use = "complete.obs")
    )
  })
}) |>
  mutate(
    macro = factor(macro, levels = macro_labels),
    gamma = factor(gamma, levels = gamma_labels),
    label = sprintf("%.2f", r)
  )

p5 <- ggplot(cor_mg, aes(x = gamma, y = macro, fill = r)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = label), size = 3.5) +
  scale_fill_gradient2(
    low = "firebrick", mid = "white", high = "steelblue",
    midpoint = 0, limits = c(-1, 1), name = "Pearson r"
  ) +
  labs(
    title    = "Macro Predictor × Gamma Correlations",
    subtitle = "In-sample Pearson r | |r| > 0.10 = weak signal; |r| > 0.20 = moderate",
    x = NULL, y = NULL
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

safe_save("plots/exploration/p5_macro_gamma_corr.pdf", p5, width = 7, height = 5)


# ==============================================================================
# 5b. Goyal-Welch predictor × gamma correlation heatmap
# ==============================================================================
cat("\n--- Plot 5b: GW predictor × gamma correlations ---\n")

if (length(gw_cols) > 0) {
  gw_ym <- gw_pred |>
    select(ym, all_of(gw_cols)) |>
    mutate(ym = as.character(ym))

  joint_gw <- inner_join(gammas_ym, gw_ym, by = "ym")

  cor_gw <- map_dfr(gw_cols, function(gc) {
    map_dfr(gamma_cols, function(target) {
      tibble(
        gw_var = gw_labels[gc],
        gamma  = gamma_labels[target],
        r      = cor(joint_gw[[gc]], joint_gw[[target]], use = "complete.obs")
      )
    })
  }) |>
    mutate(
      gw_var = factor(gw_var, levels = gw_labels),
      gamma  = factor(gamma, levels = gamma_labels),
      label  = sprintf("%.2f", r)
    )

  p5b <- ggplot(cor_gw, aes(x = gamma, y = gw_var, fill = r)) +
    geom_tile(colour = "white", linewidth = 0.6) +
    geom_text(aes(label = label), size = 3.5) +
    scale_fill_gradient2(
      low = "firebrick", mid = "white", high = "steelblue",
      midpoint = 0, limits = c(-1, 1), name = "Pearson r"
    ) +
    labs(
      title    = "Goyal-Welch Predictor × Gamma Correlations",
      subtitle = "In-sample Pearson r | |r| > 0.10 = weak signal; |r| > 0.20 = moderate",
      x = NULL, y = NULL
    ) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  safe_save("plots/exploration/p5b_gw_gamma_corr.pdf", p5b,
            width = 7, height = max(4, length(gw_cols) * 0.5 + 2))
} else {
  cat("No GW predictors found in gw_pred — skipping plot 5b.\n")
}


# ==============================================================================
# 6. Own-feature × gamma heatmap
#    Rows = feature type (l1, l2, l3, vol12, spread) × gamma
#    Cols = target gamma
# ==============================================================================
cat("\n--- Plot 6: Own-feature × gamma correlations ---\n")

short_names <- c(
  gamma_bm           = "bm",
  gamma_mom12m       = "mom12m",
  gamma_oper_prof    = "oper_prof",
  gamma_asset_growth = "asset_growth",
  gamma_size         = "size",
  gamma_beta         = "beta"
)

feature_types <- c("l1", "l2", "l3", "vol12", "spread")
feature_type_labels <- c(
  l1     = "Own lag 1  (γ_t)",
  l2     = "Own lag 2  (γ_{t-1})",
  l3     = "Own lag 3  (γ_{t-2})",
  vol12  = "12m volatility",
  spread = "Char. IQR spread"
)

own_cor <- map_dfr(gamma_cols, function(gc_feat) {
  sname <- short_names[gc_feat]
  map_dfr(gamma_cols, function(gc_target) {
    map_dfr(feature_types, function(ft) {
      feat_col <- paste0(sname, "_", ft)
      if (!feat_col %in% colnames(aligned)) return(NULL)
      r <- cor(aligned[[feat_col]], aligned[[gc_target]],
               use = "complete.obs")
      tibble(
        feature_gamma = gamma_labels[gc_feat],
        feature_type  = feature_type_labels[ft],
        target_gamma  = gamma_labels[gc_target],
        r             = r
      )
    })
  })
}) |>
  mutate(
    feature       = paste0(feature_type, "\n(", feature_gamma, ")"),
    feature_gamma = factor(feature_gamma, levels = gamma_labels),
    feature_type  = factor(feature_type,  levels = feature_type_labels),
    target_gamma  = factor(target_gamma,  levels = gamma_labels),
    label         = sprintf("%.2f", r)
  )

p6 <- ggplot(own_cor, aes(x = target_gamma, y = feature_type, fill = r)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = label), size = 2.2) +
  scale_fill_gradient2(
    low = "firebrick", mid = "white", high = "steelblue",
    midpoint = 0, limits = c(-1, 1), name = "r"
  ) +
  facet_wrap(~feature_gamma, ncol = 3) +
  labs(
    title    = "Own-Feature × Gamma Correlations",
    subtitle = "Rows = feature type; columns = target gamma; facets = feature's source gamma",
    x = "Target gamma", y = "Feature type"
  ) +
  theme_bw(base_size = 9) +
  theme(
    axis.text.x   = element_text(angle = 30, hjust = 1),
    strip.text    = element_text(size = 8, face = "bold"),
    panel.spacing = unit(0.4, "lines")
  )

safe_save("plots/exploration/p6_own_feature_gamma_corr.pdf",
          p6, width = 12, height = 10)


# ==============================================================================
# 7. Full feature × gamma correlation heatmap (42 × 6)
# ==============================================================================
cat("\n--- Plot 7: Full feature × gamma correlation heatmap ---\n")

full_cor <- map_dfr(feature_cols, function(fc) {
  map_dfr(gamma_cols, function(gc) {
    tibble(
      feature = fc,
      gamma   = gamma_labels[gc],
      r       = cor(aligned[[fc]], aligned[[gc]], use = "complete.obs")
    )
  })
})

# Determine feature group for ordering / colouring.
# Groups: Gamma lag, Gamma vol12, Char spread, Macro, GW Predictor, Factor Signal.
factor_prefixes <- paste0("^(", paste(c("mkt","smb","hml","rmw","cma"), collapse="|"), ")_")
gamma_short_pat <- paste0("^(", paste(c("bm","mom12m","oper_prof","asset_growth","size","beta"), collapse="|"), ")_vol12$")
gw_col_names    <- c("dp","dy","ep","de","bm","ntis","svar","ltr","dfr","ik")
macro_col_names <- c("default_spread","term_spread","short_rate","vix",
                     "indpro_gap","mkt_lag1","infl","lty")

classify_feature <- function(fn) {
  if (grepl("_l[123]$", fn))                 return("Gamma lag")
  if (grepl(gamma_short_pat, fn))             return("Gamma vol12")
  # Macro before "_spread$": default_spread/term_spread are macro but end in
  # "_spread" and would otherwise be misfiled as characteristic spreads.
  if (fn %in% macro_col_names)               return("Macro")
  if (grepl("_spread$", fn))                  return("Char spread")
  if (fn %in% gw_col_names)                  return("GW Predictor")
  if (grepl(factor_prefixes, fn))            return("Factor Signal")
  return("Other")
}

group_order <- c("Gamma lag", "Gamma vol12", "Char spread",
                 "Macro", "GW Predictor", "Factor Signal", "Other")

full_cor <- full_cor |>
  mutate(
    group   = factor(sapply(feature, classify_feature), levels = group_order),
    gamma   = factor(gamma, levels = gamma_labels),
    feature = factor(feature, levels = feature_cols),  # preserve original order
    label   = ifelse(abs(r) >= 0.10, sprintf("%.2f", r), "")
  )

p7 <- ggplot(full_cor, aes(x = gamma, y = feature, fill = r)) +
  geom_tile(colour = "white", linewidth = 0.3) +
  geom_text(aes(label = label), size = 2, colour = "black") +
  scale_fill_gradient2(
    low = "firebrick", mid = "white", high = "steelblue",
    midpoint = 0, limits = c(-1, 1), name = "r"
  ) +
  facet_grid(group ~ ., scales = "free_y", space = "free_y") +
  labs(
    title    = sprintf("Full Feature × Gamma Correlation Heatmap (%d predictors)", length(feature_cols)),
    subtitle = "Numbers shown only where |r| ≥ 0.10 | Grouped by feature type",
    x = NULL, y = NULL
  ) +
  theme_bw(base_size = 8) +
  theme(
    axis.text.x   = element_text(angle = 30, hjust = 1),
    axis.text.y   = element_text(size = 6),
    strip.text.y  = element_text(angle = 0, size = 8, face = "bold"),
    panel.spacing = unit(0.3, "lines")
  )

safe_save("plots/exploration/p7_full_feature_gamma_corr.pdf",
          p7, width = 8, height = max(16, length(feature_cols) * 0.18 + 4))


# ==============================================================================
# 8. Feature–feature correlation heatmap (macro + cross-gamma lag-1 block)
# ==============================================================================
cat("\n--- Plot 8: Feature–feature correlation (macro + cross-gamma l1) ---\n")

# Focus on the 12 most interpretable: 6 macro + 6 gamma lag-1
lag1_cols    <- paste0(short_names, "_l1")
focus_cols   <- c(lag1_cols, macro_cols)
focus_labels <- c(
  setNames(paste0(gamma_labels, " l1"), lag1_cols),
  macro_labels
)

ff_mat <- cor(
  na.omit(aligned[, focus_cols]),
  use = "pairwise.complete.obs"
)
rownames(ff_mat) <- colnames(ff_mat) <- focus_labels[colnames(ff_mat)]

ff_tbl <- as.data.frame(ff_mat) |>
  rownames_to_column("var_y") |>
  pivot_longer(-var_y, names_to = "var_x", values_to = "r") |>
  mutate(
    var_y = factor(var_y, levels = rev(focus_labels)),
    var_x = factor(var_x, levels = focus_labels),
    label = sprintf("%.2f", r)
  )

p8 <- ggplot(ff_tbl, aes(x = var_x, y = var_y, fill = r)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = label), size = 2.8) +
  scale_fill_gradient2(
    low = "firebrick", mid = "white", high = "steelblue",
    midpoint = 0, limits = c(-1, 1), name = "r"
  ) +
  labs(
    title    = "Feature–Feature Correlations (Gamma l1 + Macro variables)",
    subtitle = "Diagonal = 1.0 | Off-diagonal multicollinearity",
    x = NULL, y = NULL
  ) +
  theme_bw(base_size = 9) +
  theme(axis.text.x = element_text(angle = 40, hjust = 1))

safe_save("plots/exploration/p8_feature_feature_corr.pdf",
          p8, width = 9, height = 8)


# ==============================================================================
# 9. VIF — interpretable subset (gamma lag-1 + macro + GW)
# ==============================================================================
# With ~130 features, OLS is rank-deficient (more predictors than complete-case
# rows for some combos) and full-set VIF is not meaningful. Factor signals and
# gamma lags-2/3 are handled by LASSO / RF regularisation. VIF is only
# interpretable for the low-dimensional "interpretable" block used in the linear
# models (MV, AR): own lag-1 per gamma + macro conditioning + GW predictors.
cat(sprintf("\n--- Plot 9: VIF — interpretable subset (lag-1 + macro + GW; %d total features in full set) ---\n",
            length(feature_cols)))

# Build interpretable subset: 6 gamma lag-1 + 8 macro + GW (up to ~12) columns
vif_subset_cols <- intersect(
  c(paste0(short_names, "_l1"), macro_col_names, gw_col_names),
  names(aligned)
)
cat(sprintf("VIF subset: %d features (%s)\n", length(vif_subset_cols),
            paste(vif_subset_cols, collapse = ", ")))

vif_all <- map_dfr(gamma_cols, function(gc) {
  df_v <- aligned |>
    select(all_of(c(gc, vif_subset_cols))) |>
    filter(complete.cases(pick(everything())))

  if (nrow(df_v) < length(vif_subset_cols) + 5) {
    cat(sprintf("  Too few complete rows for %s (%d rows, need %d) — skipping.\n",
                gamma_labels[gc], nrow(df_v), length(vif_subset_cols) + 5))
    return(tibble(gamma = gamma_labels[gc], feature = NA_character_, vif = NA_real_))
  }

  # Drop any columns that are perfect linear combinations of others (aliased).
  # Common cause: dp and dy in Goyal-Welch are near-identical S&P ratio series.
  fit_check <- lm(as.formula(paste(gc, "~ .")), data = df_v)
  al <- tryCatch(alias(fit_check)$Complete, error = function(e) NULL)
  if (!is.null(al) && nrow(al) > 0) {
    drop_alias <- rownames(al)
    cat(sprintf("  Dropping %d aliased col(s) for %s: %s\n",
                length(drop_alias), gamma_labels[gc], paste(drop_alias, collapse = ", ")))
    df_v <- df_v |> select(-any_of(drop_alias))
  }

  fml      <- as.formula(paste(gc, "~ ."))
  vif_vals <- tryCatch(
    car::vif(lm(fml, data = df_v)),
    error = function(e) { cat("  VIF lm error:", e$message, "\n"); NULL }
  )
  if (is.null(vif_vals)) return(tibble(gamma = gamma_labels[gc],
                                        feature = NA_character_, vif = NA_real_))

  tibble(
    gamma   = gamma_labels[gc],
    feature = names(vif_vals),
    vif     = as.numeric(vif_vals)
  )
})

if (!("feature" %in% names(vif_all)) || all(is.na(vif_all$feature))) {
  cat("VIF computation returned no results — skipping plot 9.\n")
} else {
  vif_all <- vif_all |>
    filter(!is.na(feature), !is.na(vif)) |>
    mutate(
      group = case_when(
        grepl("_l[123]$", feature)    ~ "Gamma lag",
        feature %in% macro_col_names  ~ "Macro",
        feature %in% gw_col_names     ~ "GW Predictor",
        TRUE                          ~ "Other"
      ),
      group   = factor(group, levels = c("Gamma lag", "Macro", "GW Predictor", "Other")),
      feature = fct_reorder(feature, vif, .fun = max)
    )

  vif_plot_data <- vif_all |> filter(gamma == "Value")

  p9 <- ggplot(vif_plot_data, aes(x = vif, y = feature, fill = group)) +
    geom_col(width = 0.7) +
    geom_vline(xintercept = c(5, 10), linetype = c("dashed", "solid"),
               colour = c("orange", "firebrick"), linewidth = 0.6) +
    annotate("text", x = 5.2,  y = Inf, label = "VIF = 5",  hjust = 0, vjust = 1.5,
             size = 2.8, colour = "orange") +
    annotate("text", x = 10.2, y = Inf, label = "VIF = 10", hjust = 0, vjust = 1.5,
             size = 2.8, colour = "firebrick") +
    scale_fill_brewer(palette = "Set2", name = "Feature group") +
    labs(
      title    = sprintf("Variance Inflation Factors — Interpretable Subset (%d features)",
                         length(vif_subset_cols)),
      subtitle = paste0("Gamma lag-1 + macro + GW predictors | Value gamma as representative\n",
                        sprintf("(Full feature set has %d cols — factor signals not shown; LASSO handles collinearity)",
                                length(feature_cols))),
      x = "VIF", y = NULL
    ) +
    theme_bw(base_size = 9) +
    theme(legend.position = "bottom")

  safe_save("plots/exploration/p9_vif_interpretable_subset.pdf",
            p9, width = 8, height = max(6, length(vif_subset_cols) * 0.35 + 3))

  cat("\n--- VIF summary: features with VIF > 5 (Value gamma) ---\n")
  vif_plot_data |>
    filter(vif > 5) |>
    arrange(desc(vif)) |>
    mutate(vif = round(vif, 1)) |>
    print(n = Inf)

  cat("\nMax VIF by feature group:\n")
  vif_all |>
    filter(gamma == "Value") |>
    group_by(group) |>
    summarise(max_vif = round(max(vif), 1), .groups = "drop") |>
    arrange(desc(max_vif)) |>
    print()
}


cat("\n=== All exploration plots saved to plots/exploration/ ===\n")
