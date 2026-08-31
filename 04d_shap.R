# ==============================================================================
# 04d_shap.R
# Model interpretation for RF and LASSO: SHAP, partial dependence, and ALE.
#
# Fits a FINAL model on all training data available at the end of the OOS period
# (i.e., the last expanding window), then explains that model.
#
# This answers: "which predictors drive the gamma forecasts, and are the
# relationships consistent with economic theory?"
#
# SHAP approach:
#   RF:    Two-step — KernelSHAP (CRAN) for beeswarm (sign/direction), impurity
#          importance for the importance heatmap (instant, already in the model).
#          KernelSHAP uses a tiny sample (bg=20, X=30, max_iter=10) to keep
#          runtime to seconds; approximate SHAP values are sufficient for
#          directional interpretation in a thesis context.
#   LASSO: Coefficient plot only — for a linear model, coefficients ARE the SHAP
#          values (up to scaling), so KernelSHAP adds nothing over direct inspection.
#
# Shape of the relationship (Sec 3–4): SHAP/importance say WHICH features and HOW
# MUCH; partial dependence and ALE say in WHAT SHAPE. PDP + ICE (Friedman 2001;
# Goldstein et al. 2015) is the main exhibit; ALE (Apley & Zhu 2020) is a
# correlation-robust robustness appendix on the same features — PDP marginalises
# assuming feature independence, which is violated by the 118 collinear predictors,
# whereas ALE uses local differences within conditional bins. Both computed by hand
# (base R + ranger) to stay dependency-free and fully reproducible; the citations
# are to the methods, not any package.
#
# Outputs:
#   plots/shap_rf_beeswarm.pdf    — SHAP beeswarm per gamma (top 15 features)
#   plots/shap_lasso_coef.pdf     — LASSO coefficients at last OOS lambda
#   plots/shap_rf_importance.pdf  — Mean |SHAP| bar chart (all gammas combined)
#   plots/pdp_rf.pdf              — RF partial dependence + ICE, top 4 feats / gamma
#   plots/ale_rf.pdf              — RF accumulated local effects (same panels)
#
# Input:  gamma_predictions.rds (aligned, feature_cols, oos_idx)
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(ranger)
library(glmnet)
library(kernelshap)  # KernelSHAP (CRAN) — model-agnostic SHAP approximation
library(shapviz)     # SHAP plots: beeswarm, importance, dependence
select <- dplyr::select

# Load aligned + feature_cols directly from gamma_predictions.rds.
# These were built and saved by 04_predict_gammas.R from the full 118-feature
# set (gamma lags/vol, char spreads, 8 macro vars, GW predictors, factor signals).
# Loading them here avoids rebuilding and guarantees consistency with the OOS loop.
list2env(readRDS("gamma_predictions.rds"), envir = environment())  # -> pred_list, oos_eval, actuals, dates_oos,
                                  #    aligned, feature_cols, oos_idx

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

cat(sprintf(
  "Feature matrix loaded: %d features, %d total months, %d OOS months (%s – %s)\n",
  length(feature_cols), nrow(aligned), length(oos_idx),
  format(aligned$date[min(oos_idx)], "%Y-%m"),
  format(aligned$date[max(oos_idx)], "%Y-%m")
))

# Column-mean imputation (must match 04_predict_gammas.R)
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

# Use ALL training data available at the last OOS step (full expanding window)
t_last   <- max(oos_idx)
train_all <- aligned[1:(t_last - 1), ]
Y_all    <- train_all |> select(all_of(gamma_cols)) |> as.matrix()
X_all    <- train_all |> select(all_of(feature_cols)) |> as.matrix()
X_imp    <- impute_means(X_all)

# Clean feature labels for plots (remove trailing _l1 etc.)
feat_labels <- feature_cols |>
  str_replace("_l1$", " (lag 1)") |>
  str_replace("_l2$", " (lag 2)") |>
  str_replace("_l3$", " (lag 3)") |>
  str_replace("_vol12$", " (vol)") |>
  str_replace("_spread$", " (spread)") |>
  str_replace("_", " ")

rf_mtry <- max(5L, floor(length(feature_cols) / 3L))
dir.create("plots", showWarnings = FALSE)

safe_save <- function(path, plot, ...) {
  tryCatch(
    { ggsave(path, plot, ...); cat("Saved:", path, "\n") },
    error = function(e) cat("SKIPPED (file locked):", path, "\n")
  )
}


# ==============================================================================
# 1. RF KernelSHAP (beeswarm) + impurity importance
# ==============================================================================
cat("Computing RF KernelSHAP values...\n")

shap_rf_list  <- setNames(vector("list", length(gamma_cols)), gamma_cols)
rf_importance <- setNames(vector("list", length(gamma_cols)), gamma_cols)
# Retain the per-gamma fit and its training matrix so the PDP/ALE sections below
# explain the SAME final-window model (no refit → guaranteed consistency).
rf_fits       <- setNames(vector("list", length(gamma_cols)), gamma_cols)
rf_X          <- setNames(vector("list", length(gamma_cols)), gamma_cols)

for (gc in gamma_cols) {
  cat(" ", gamma_labels[gc], "...")
  y_all_gc <- Y_all[, gc]
  ok       <- !is.na(y_all_gc)
  X_gc     <- X_imp[ok, , drop = FALSE]
  y_gc     <- y_all_gc[ok]

  rf_df <- as.data.frame(X_gc)
  rf_df$.y <- y_gc
  rf_fit <- ranger(
    .y ~ ., data = rf_df,
    num.trees     = 500L,
    mtry          = rf_mtry,
    min.node.size = 5L,
    importance    = "impurity",
    seed          = 42L
  )

  # Store impurity importance for the importance heatmap (instant, no SHAP needed)
  rf_importance[[gc]] <- rf_fit$variable.importance
  rf_fits[[gc]]       <- rf_fit
  rf_X[[gc]]          <- X_gc

  # KernelSHAP for beeswarm (directional interpretation only).
  # Use small sample to keep runtime short (~seconds per gamma):
  #   bg=20 background rows, X=30 explain rows, max_iter=10 convergence rounds.
  # KernelSHAP default m = 2 * p * (1 + 3 * (hybrid_degree==0)) = 72 for p=36.
  # Total budget: 30 × 20 × 72 × 10 ≈ 432K evaluations vs 144M in the original.
  set.seed(42)
  bg_idx    <- sample(nrow(X_gc), min(20L, nrow(X_gc)))
  shap_idx  <- sample(nrow(X_gc), min(30L,  nrow(X_gc)))
  X_bg_df   <- as.data.frame(X_gc[bg_idx,   , drop = FALSE])
  X_shap_df <- as.data.frame(X_gc[shap_idx, , drop = FALSE])

  pred_fn <- function(model, newdata)
    predict(model, data = as.data.frame(newdata))$predictions

  ks  <- kernelshap(rf_fit, X = X_shap_df, bg_X = X_bg_df,
                    pred_fun = pred_fn, max_iter = 10L,
                    verbose = FALSE, seed = 42L)
  shp <- shapviz(ks)
  shap_rf_list[[gc]] <- shp
  cat(" done\n")
}

# Beeswarm plots — one per gamma, top 15 features by mean |SHAP|
cat("Generating beeswarm plots...\n")
beeswarm_plots <- map(gamma_cols, function(gc) {
  shp    <- shap_rf_list[[gc]]
  colnames(shp$S) <- feat_labels
  colnames(shp$X) <- feat_labels
  sv_importance(shp, kind = "beeswarm", max_display = 15) +
    ggtitle(paste0("RF SHAP — ", gamma_labels[gc])) +
    theme_bw(base_size = 9) +
    theme(plot.title = element_text(size = 10))
})

if (requireNamespace("ggpubr", quietly = TRUE)) {
  library(ggpubr)
  p_beeswarm <- ggarrange(plotlist = beeswarm_plots, ncol = 2, nrow = 3)
  safe_save("plots/shap_rf_beeswarm.pdf", p_beeswarm, width = 14, height = 18)
} else {
  cat("ggpubr not installed — saving individual beeswarm plots\n")
  for (k in seq_along(gamma_cols)) {
    gc <- gamma_cols[k]
    safe_save(
      sprintf("plots/shap_rf_%s.pdf", str_remove(gc, "^gamma_")),
      beeswarm_plots[[k]], width = 7, height = 5
    )
  }
}


# Impurity importance heatmap — all gammas, top 20 features.
# Impurity importance (already computed during RF training) is more stable than
# mean |SHAP| when the SHAP explain set is small (30 obs). Both rank features
# the same way; impurity importance covers the full training set.
cat("Computing impurity importance heatmap...\n")
shap_imp <- map_dfr(gamma_cols, function(gc) {
  imp <- rf_importance[[gc]]
  tibble(
    gamma    = gamma_labels[gc],
    feature  = feat_labels,
    mean_abs = imp
  )
})

top_feats <- shap_imp |>
  group_by(feature) |>
  summarise(total_imp = sum(mean_abs), .groups = "drop") |>
  slice_max(total_imp, n = 20) |>
  pull(feature)

p_imp <- shap_imp |>
  filter(feature %in% top_feats) |>
  mutate(feature = factor(feature, levels = rev(top_feats))) |>
  ggplot(aes(x = gamma, y = feature, fill = mean_abs)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  scale_fill_gradient(low = "white", high = "steelblue", name = "Impurity\nImportance") +
  labs(title = "RF Feature Importance (Impurity) — Top 20 Features",
       x = NULL, y = NULL) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

safe_save("plots/shap_rf_importance.pdf", p_imp, width = 8, height = 8)


# ==============================================================================
# 1b. Family-level importance heatmap (predictor family × gamma) — thesis Fig.
# ==============================================================================
# Aggregates the per-feature impurity importance into the six predictor families
# (classification identical to 03b_visual_exploration.R). Two panels:
#   (a) total family share of importance per gamma (sums to 100%)
#   (b) average share per predictor in the family — corrects for family size
#       (75 of 118 features are factor signals, so (a) alone overstates them;
#        proportional benchmark = 100/118 ≈ 0.85% per predictor).
cat("Computing family-level importance heatmap...\n")

factor_prefixes <- paste0("^(", paste(c("mkt","smb","hml","rmw","cma"), collapse = "|"), ")_")
gamma_short_pat <- paste0("^(", paste(c("bm","mom12m","oper_prof","asset_growth","size","beta"), collapse = "|"), ")_vol12$")
gw_col_names    <- c("dp","dy","ep","de","bm","ntis","svar","ltr","dfr","ik")
macro_col_names <- c("default_spread","term_spread","short_rate","vix",
                     "indpro_gap","mkt_lag1","infl","lty")
classify_feature <- function(fn) {
  if (grepl("_l[123]$", fn))      return("Gamma lag")
  if (grepl(gamma_short_pat, fn)) return("Gamma vol.")
  # Macro is tested BEFORE the generic "_spread$" pattern: default_spread and
  # term_spread are macro variables whose names also end in "_spread" and would
  # otherwise be misclassified as characteristic spreads (inflating that group
  # to 8 features and shrinking Macro to 6).
  if (fn %in% macro_col_names)    return("Macro")
  if (grepl("_spread$", fn))      return("Char. spread")
  if (fn %in% gw_col_names)       return("Goyal-Welch")
  if (grepl(factor_prefixes, fn)) return("Factor signal")
  return("Other")
}
feat_groups <- vapply(feature_cols, classify_feature, character(1L))
stopifnot(sum(feat_groups == "Other") == 0L)

group_order <- c("Gamma lag", "Gamma vol.", "Char. spread",
                 "Macro", "Goyal-Welch", "Factor signal")
heat_grp <- map_dfr(gamma_cols, function(gc) {
  tibble(gamma = gamma_labels[gc], group = feat_groups,
         imp = rf_importance[[gc]])
}) |>
  group_by(gamma, group) |>
  summarise(imp = sum(imp), n_feat = n(), .groups = "drop_last") |>
  mutate(share = 100 * imp / sum(imp),
         share_per_feat = share / n_feat) |>
  ungroup() |>
  mutate(group = factor(group, levels = rev(group_order)),
         gamma = factor(gamma, levels = unname(gamma_labels[gamma_cols])))

theme_heat <- theme_bw(base_size = 10) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1),
        panel.grid = element_blank(),
        legend.position = "none",
        plot.title = element_text(size = 10))

p_grp_a <- ggplot(heat_grp, aes(x = gamma, y = group, fill = share)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.1f", share)), size = 2.9) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "(a) Total family share (%)", x = NULL, y = NULL) +
  theme_heat

p_grp_b <- ggplot(heat_grp, aes(x = gamma, y = group, fill = share_per_feat)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", share_per_feat)), size = 2.9) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "(b) Average share per predictor (%)", x = NULL, y = NULL) +
  theme_heat + theme(axis.text.y = element_blank(),
                     axis.ticks.y = element_blank())

if (requireNamespace("ggpubr", quietly = TRUE)) {
  p_grp <- ggpubr::ggarrange(p_grp_a, p_grp_b, ncol = 2, widths = c(1.18, 1))
  safe_save("plots/shap_rf_group_heatmap.pdf", p_grp, width = 9.5, height = 3.6)
} else {
  safe_save("plots/shap_rf_group_heatmap.pdf", p_grp_a, width = 8, height = 4.4)
}
write_csv(heat_grp |> select(gamma, group, n_feat, share, share_per_feat),
          "shap_group_shares.csv")


# ==============================================================================
# 2. LASSO coefficient plot at lambda.min (final window)
# ==============================================================================
cat("\nComputing LASSO coefficients at final OOS window...\n")

lasso_coef_list <- map_dfr(gamma_cols, function(gc) {
  y_gc <- Y_all[, gc]
  ok   <- !is.na(y_gc)
  X_gc <- X_imp[ok, , drop = FALSE]

  # ts holdout CV on the full training set
  n       <- sum(ok)
  n_val   <- max(20L, floor(n * 0.20))
  n_tr    <- n - n_val
  if (n_tr < 20L) return(NULL)

  path <- tryCatch(
    glmnet(X_gc[1:n_tr, ], y_gc[ok][1:n_tr], alpha = 1),
    error = function(e) NULL
  )
  if (is.null(path)) return(NULL)

  val_pred <- predict(path, newx = X_gc[(n_tr + 1):n, ])
  val_mse  <- colMeans((y_gc[ok][(n_tr + 1):n] - val_pred)^2, na.rm = TRUE)
  lam_min  <- path$lambda[which.min(val_mse)]

  # Refit on full training set at lambda.min
  fit_full <- tryCatch(
    glmnet(X_gc, y_gc[ok], alpha = 1, lambda = lam_min),
    error = function(e) NULL
  )
  if (is.null(fit_full)) return(NULL)

  coefs <- as.numeric(coef(fit_full))[-1]  # drop intercept
  tibble(
    gamma   = gamma_labels[gc],
    feature = feat_labels,
    coef    = coefs
  )
})

# Plot the top-15 non-zero coefficients per gamma (LASSO is non-sparse for some
# gammas, e.g. Profitability has ~50 non-zero features which crush the panel).
N_TOP_COEF <- 15L
top_lasso <- lasso_coef_list |>
  filter(coef != 0) |>
  group_by(gamma) |>
  slice_max(abs(coef), n = N_TOP_COEF, with_ties = FALSE) |>
  ungroup() |>
  mutate(feature  = str_trunc(feature, 28),
         feat_key = paste(feature, gamma, sep = "___"))   # unique per facet

# Order the feature axis WITHIN each facet by |coef| (manual reorder_within;
# tidytext not available). The "___gamma" suffix is stripped in the axis labels.
feat_levels <- top_lasso |> arrange(gamma, abs(coef)) |> pull(feat_key)
top_lasso   <- top_lasso |> mutate(feat_key = factor(feat_key, levels = feat_levels))

p_lasso_coef <- top_lasso |>
  ggplot(aes(x = coef, y = feat_key, fill = coef > 0)) +
  geom_col(show.legend = FALSE) +
  scale_fill_manual(values = c(`TRUE` = "steelblue", `FALSE` = "firebrick")) +
  scale_y_discrete(labels = function(x) sub("___.*$", "", x)) +
  geom_vline(xintercept = 0, linewidth = 0.3) +
  facet_wrap(~gamma, scales = "free", ncol = 2) +
  labs(title = "LASSO Coefficients (lambda.min, final OOS window)",
       subtitle = paste0("Top ", N_TOP_COEF,
                         " non-zero coefficients per gamma by |coef|; blue = positive, red = negative"),
       x = "Coefficient", y = NULL) +
  theme_bw(base_size = 10) +
  theme(axis.text.y = element_text(size = 8))

safe_save("plots/shap_lasso_coef.pdf", p_lasso_coef, width = 13, height = 10)


# ==============================================================================
# 3. Partial Dependence + ICE (RF) — the SHAPE of the top relationships
# ==============================================================================
# For the top N_TOP_SHAPE features (by impurity importance) of each gamma, trace
# how the RF forecast moves as that feature is swept across its 5th–95th
# percentile, holding the others at their observed values and averaging:
#   PDP(v) = mean_i f( x_j = v, x_{i,-j} )            [Friedman 2001]
# ICE curves show the same sweep for individual observations (one line each),
# exposing heterogeneity / interactions the average PDP would hide
# [Goldstein, Kapelner, Bleich & Pitkin 2015, JCGS]. Grid trimmed to the
# 5th–95th percentile so the average is not dominated by sparse tail regions.
cat("\nComputing RF partial dependence + ICE...\n")

N_TOP_SHAPE <- 4L      # features per gamma (matched across PDP and ALE)
PDP_GRID    <- 25L     # sweep points per feature
N_ICE       <- 40L     # individual ICE curves per panel
RUG_MAX     <- 300L    # observed values drawn as a rug (subsampled)

# Predict with the same data-frame convention as the KernelSHAP block above.
rf_pred <- function(fit, X) predict(fit, data = as.data.frame(X))$predictions

# Top features per gamma: highest impurity importance among features with enough
# distinct values to define a sensible sweep (skips near-constant signals).
top_idx <- setNames(lapply(gamma_cols, function(gc) {
  imp   <- rf_importance[[gc]]
  Xg    <- rf_X[[gc]]
  ord   <- order(imp, decreasing = TRUE)
  ord   <- ord[vapply(ord, function(j) length(unique(Xg[, j])) >= 5L, logical(1L))]
  head(ord, N_TOP_SHAPE)
}), gamma_cols)

# One PDP + ICE sweep for feature column j of matrix X under fitted model.
pdp_ice_rf <- function(fit, X, j, grid_n = PDP_GRID, n_ice = N_ICE, seed = 42L) {
  grid <- unique(quantile(X[, j], probs = seq(0.05, 0.95, length.out = grid_n),
                          names = FALSE))
  set.seed(seed)
  ice_id  <- sort(sample(nrow(X), min(n_ice, nrow(X))))
  pd      <- numeric(length(grid))
  ice_mat <- matrix(NA_real_, nrow = length(ice_id), ncol = length(grid))
  for (g in seq_along(grid)) {
    Xg      <- X; Xg[, j] <- grid[g]
    pr      <- rf_pred(fit, Xg)
    pd[g]        <- mean(pr)
    ice_mat[, g] <- pr[ice_id]
  }
  list(
    pdp = tibble(x = grid, pd = pd),
    ice = tibble(x    = rep(grid, each = length(ice_id)),
                 id   = rep(seq_along(ice_id), times = length(grid)),
                 yhat = as.numeric(ice_mat))
  )
}

pdp_parts <- list(); ice_parts <- list(); rug_parts <- list(); facet_levels <- character()
for (gc in gamma_cols) {
  fit <- rf_fits[[gc]]; Xg <- rf_X[[gc]]
  for (j in top_idx[[gc]]) {
    key   <- sprintf("%s: %s", gamma_labels[gc], feat_labels[j])
    facet_levels <- c(facet_levels, key)
    res   <- pdp_ice_rf(fit, Xg, j)
    meta  <- list(gamma = unname(gamma_labels[gc]), feature = feat_labels[j], facet = key)
    pdp_parts[[key]] <- res$pdp |> mutate(!!!meta)
    ice_parts[[key]] <- res$ice |> mutate(!!!meta)
    rv <- Xg[, j]; if (length(rv) > RUG_MAX) rv <- sample(rv, RUG_MAX)
    rug_parts[[key]] <- tibble(x = rv, !!!meta)
  }
  cat(" ", gamma_labels[gc], "done\n")
}

fac <- function(df) mutate(df, facet = factor(facet, levels = facet_levels))
pdp_df <- fac(bind_rows(pdp_parts))
ice_df <- fac(bind_rows(ice_parts)) |> mutate(grp = paste(facet, id))
rug_df <- fac(bind_rows(rug_parts))

p_pdp <- ggplot() +
  geom_line(data = ice_df, aes(x, yhat, group = grp),
            colour = "grey70", alpha = 0.25, linewidth = 0.2) +
  geom_line(data = pdp_df, aes(x, pd), colour = "firebrick", linewidth = 0.9) +
  geom_rug(data = rug_df, aes(x), sides = "b", alpha = 0.12) +
  facet_wrap(~facet, scales = "free", ncol = N_TOP_SHAPE) +
  labs(title = "RF partial dependence (red) with ICE curves (grey) — top 4 features per gamma",
       subtitle = paste("Final expanding-window fit; sweep over 5th-95th pctile.",
                         "PDP: Friedman (2001); ICE: Goldstein et al. (2015)."),
       x = "Feature value", y = "Predicted gamma") +
  theme_bw(base_size = 9) +
  theme(strip.text = element_text(size = 7.3))

safe_save("plots/pdp_rf.pdf", p_pdp, width = 13, height = 16)


# ==============================================================================
# 4. Accumulated Local Effects (RF) — correlation-robust robustness appendix
# ==============================================================================
# ALE (Apley & Zhu 2020, JRSS-B) replaces PDP's marginal integration — which
# evaluates unrealistic feature combinations when predictors are correlated — with
# accumulated LOCAL differences computed within conditional quantile bins:
#   for each bin k, average f(z_k, x_{-j}) - f(z_{k-1}, x_{-j}) over obs in bin k,
#   accumulate across bins, then centre so E[ALE] = 0.
# Same features/panels as the PDP so the two are read side by side.
cat("\nComputing RF accumulated local effects (ALE)...\n")

ALE_BINS <- 20L

ale_rf <- function(fit, X, j, K = ALE_BINS) {
  z  <- unique(quantile(X[, j], probs = seq(0, 1, length.out = K + 1L), names = FALSE))
  Kq <- length(z) - 1L
  if (Kq < 2L) return(NULL)
  bin  <- findInterval(X[, j], z, rightmost.closed = TRUE, all.inside = TRUE)
  eff  <- numeric(Kq); cnt <- numeric(Kq)
  for (k in seq_len(Kq)) {
    in_k <- which(bin == k)
    if (!length(in_k)) next
    Xlo <- X[in_k, , drop = FALSE]; Xlo[, j] <- z[k]
    Xhi <- X[in_k, , drop = FALSE]; Xhi[, j] <- z[k + 1L]
    eff[k] <- mean(rf_pred(fit, Xhi) - rf_pred(fit, Xlo))
    cnt[k] <- length(in_k)
  }
  acc     <- c(0, cumsum(eff))                       # value at edges z_0..z_Kq
  bin_mid <- (acc[-1] + acc[-length(acc)]) / 2       # per-bin ALE level
  centre  <- sum(cnt * bin_mid) / sum(cnt)           # count-weighted mean
  tibble(x = z, ale = acc - centre)
}

ale_parts <- list()
for (gc in gamma_cols) {
  fit <- rf_fits[[gc]]; Xg <- rf_X[[gc]]
  for (j in top_idx[[gc]]) {
    res <- ale_rf(fit, Xg, j)
    if (is.null(res)) next
    key <- sprintf("%s: %s", gamma_labels[gc], feat_labels[j])
    ale_parts[[key]] <- res |>
      mutate(gamma = unname(gamma_labels[gc]), feature = feat_labels[j], facet = key)
  }
  cat(" ", gamma_labels[gc], "done\n")
}

ale_df <- fac(bind_rows(ale_parts))

p_ale <- ggplot(ale_df, aes(x, ale)) +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = "grey60") +
  geom_line(colour = "darkgreen", linewidth = 0.8) +
  geom_point(colour = "darkgreen", size = 0.7) +
  geom_rug(data = rug_df, aes(x), inherit.aes = FALSE, sides = "b", alpha = 0.12) +
  facet_wrap(~facet, scales = "free", ncol = N_TOP_SHAPE) +
  labs(title = "RF accumulated local effects (ALE) — top 4 features per gamma",
       subtitle = paste("Correlation-robust counterpart to the PDP (same panels).",
                        "Apley & Zhu (2020). Centred so E[ALE] = 0."),
       x = "Feature value", y = "ALE (centred)") +
  theme_bw(base_size = 9) +
  theme(strip.text = element_text(size = 7.3))

safe_save("plots/ale_rf.pdf", p_ale, width = 13, height = 16)


cat("\nDone. Outputs:\n")
cat("  plots/shap_rf_beeswarm.pdf   — SHAP beeswarm (top 15 features per gamma, approximate)\n")
cat("  plots/shap_rf_importance.pdf — impurity importance heatmap (top 20 features)\n")
cat("  plots/shap_lasso_coef.pdf    — LASSO non-zero coefficients (final window)\n")
cat("  plots/pdp_rf.pdf             — RF partial dependence + ICE (top 4 features per gamma)\n")
cat("  plots/ale_rf.pdf             — RF accumulated local effects (correlation-robust)\n")
