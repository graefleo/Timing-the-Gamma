# ==============================================================================
# 04g_ablation.R — Drop-one-family-out OOS R² ablation for LASSO and RF
# ------------------------------------------------------------------------------
# Answers RQ3 ("which predictor families matter") inferentially rather than
# through a single in-sample impurity-importance decomposition. For each of the
# five predictor families, the family's columns are removed from the feature
# matrix and LASSO and the random forest are re-estimated over the identical
# expanding-window OOS design, on the identical common evaluation sample. The
# marginal contribution of a family is the drop in out-of-sample R² caused by
# its removal (full minus ablated): a large positive drop means the family
# carries unique, non-redundant predictive information for that gamma.
#
# This generalises the MLP/LSTM "NoMacro" variants (which drop only the macro
# block, and only for the networks) into a single drop-one-family-out design
# spanning all five families and both interpretable learners.
#
# Estimators, hyperparameters and the common window are held identical to
# 04_predict_gammas.R; the only thing that changes across runs is which family
# of columns is withheld. mtry for the random forest is recomputed as
# floor(p/3) on the reduced feature set, i.e. the estimator adapts to the
# features actually available — the honest "run the same estimator on fewer
# predictors" counterfactual.
#
# Inputs : gamma_predictions.rds (aligned, feature_cols, oos_idx, actuals,
#          pred_list — for the full-model baseline, hist_mean, common window)
# Outputs: ablation_results.rds, ablation_results.csv, tables/b7_ablation.tex
# Run    : after 04_predict_gammas.R (no other dependency; reuses stored matrix)
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(glmnet)
  library(ranger)
  library(sandwich)
  library(lmtest)
})

set.seed(42L)

# ── Load stored pipeline objects ──────────────────────────────────────────────
gp <- readRDS("gamma_predictions.rds")
aligned      <- gp$aligned
feature_cols <- gp$feature_cols
oos_idx      <- gp$oos_idx
actuals      <- gp$actuals
pred_list    <- gp$pred_list

gamma_cols <- c("gamma_bm", "gamma_mom12m", "gamma_oper_prof",
                "gamma_asset_growth", "gamma_size", "gamma_beta")
gamma_labels <- c(gamma_bm = "Value", gamma_mom12m = "Momentum",
                  gamma_oper_prof = "Profitability",
                  gamma_asset_growth = "Asset Growth",
                  gamma_size = "Size", gamma_beta = "Beta")

# ── Helpers copied verbatim from 04_predict_gammas.R ──────────────────────────
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

ts_cv_lasso <- function(X, y, val_frac = 0.20, alpha = 1) {
  n <- length(y); n_val <- max(20L, floor(n * val_frac)); n_train <- n - n_val
  if (n_train < 20L) return(NULL)
  tr_idx <- seq_len(n_train); val_idx <- seq(n_train + 1L, n)
  fit_path <- tryCatch(glmnet(X[tr_idx, ], y[tr_idx], alpha = alpha),
                       error = function(e) NULL)
  if (is.null(fit_path)) return(NULL)
  val_pred <- predict(fit_path, newx = X[val_idx, ])
  val_mse  <- colMeans((y[val_idx] - val_pred)^2, na.rm = TRUE)
  idx_min  <- which.min(val_mse); lam_min <- fit_path$lambda[idx_min]
  fit_full <- tryCatch(glmnet(X, y, alpha = alpha, lambda = lam_min),
                       error = function(e) NULL)
  if (is.null(fit_full)) return(NULL)
  list(fit = fit_full, lambda.min = lam_min)
}

oos_r2 <- function(y, yhat, yhat_bench) {
  1 - sum((y - yhat)^2, na.rm = TRUE) / sum((y - yhat_bench)^2, na.rm = TRUE)
}

clark_west <- function(y, yhat_bench, yhat_model, nw_lags = 12L) {
  f  <- (y - yhat_bench)^2 - ((y - yhat_model)^2 - (yhat_bench - yhat_model)^2)
  ok <- !is.na(f); n <- sum(ok)
  if (n < 10L) return(list(t_cw = NA_real_, p_cw = NA_real_))
  f_ok <- f[ok]
  nw_var <- tryCatch(
    as.numeric(NeweyWest(lm(f_ok ~ 1), lag = nw_lags, prewhite = FALSE, adjust = TRUE)),
    error = function(e) var(f_ok) / n)
  if (!is.finite(nw_var) || nw_var <= 0) nw_var <- var(f_ok) / n
  t_stat <- mean(f_ok) / sqrt(nw_var)
  list(t_cw = round(t_stat, 3), p_cw = round(1 - pnorm(t_stat), 3))
}

# ── Feature-family definitions (five families, partition of feature_cols) ─────
gamma_short_pat <- paste0("^(", paste(c("bm","mom12m","oper_prof",
                          "asset_growth","size","beta"), collapse = "|"), ")_vol12$")
factor_prefixes <- "^(mkt|smb|hml|rmw|cma)_"
gw_col_names    <- c("dp","dy","ep","de","bm","ntis","svar","ltr","dfr","ik")
macro_col_names <- c("default_spread","term_spread","short_rate","vix",
                     "indpro_gap","mkt_lag1","infl","lty")

family_of <- function(fn) {
  if (grepl("_l[123]$", fn) || grepl(gamma_short_pat, fn)) return("Own-gamma history")
  # Macro is tested BEFORE the "_spread$" pattern: default_spread and
  # term_spread are macro variables that also end in "_spread" and would
  # otherwise be misfiled as characteristic spreads (the ordering bug present
  # in the 04d_shap.R classifier).
  if (fn %in% macro_col_names)     return("Macro")
  if (grepl("_spread$", fn))       return("Characteristic spread")
  if (fn %in% gw_col_names)        return("Goyal--Welch")
  if (grepl(factor_prefixes, fn))  return("Factor signal")
  return("Other")
}
feat_family <- vapply(feature_cols, family_of, character(1L))
stopifnot(sum(feat_family == "Other") == 0L)   # families must partition
families <- c("Own-gamma history", "Characteristic spread", "Macro",
              "Goyal--Welch", "Factor signal")
cat("Feature-family sizes:\n"); print(table(feat_family))

# ── Common evaluation window (identical to 04 §5b) ────────────────────────────
common_mask <- setNames(lapply(gamma_cols, function(gc) {
  rowSums(is.na(pred_list[[gc]])) == 0 & !is.na(actuals[, gc])
}), gamma_cols)

# ── Expanding-window OOS loop for a given feature subset ──────────────────────
# Returns a months × gammas matrix of predictions for the requested learner.
run_subset <- function(keep_cols, learner) {
  preds <- matrix(NA_real_, nrow = length(oos_idx), ncol = length(gamma_cols),
                  dimnames = list(NULL, gamma_cols))
  rf_mtry <- max(5L, floor(length(keep_cols) / 3L))
  for (i in seq_along(oos_idx)) {
    t     <- oos_idx[i]
    train <- aligned[seq_len(t - 1L), ]
    test  <- aligned[t, ]
    X_test <- as.matrix(test[, keep_cols])
    if (anyNA(X_test)) next                       # honest real-time skip
    Y_train <- as.matrix(train[, gamma_cols])
    X_train <- impute_means(as.matrix(train[, keep_cols]))
    for (gc in gamma_cols) {
      y_tr <- Y_train[, gc]; ok <- !is.na(y_tr)
      if (sum(ok) < 30L) next
      if (learner == "lasso") {
        fit <- tryCatch(ts_cv_lasso(X_train[ok, , drop = FALSE], y_tr[ok]),
                        error = function(e) NULL)
        if (!is.null(fit))
          preds[i, gc] <- predict(fit$fit, newx = X_test, s = fit$lambda.min)[1]
      } else {                                    # rf
        rf_df <- as.data.frame(X_train[ok, , drop = FALSE]); rf_df$.y <- y_tr[ok]
        fit <- tryCatch(ranger(.y ~ ., data = rf_df, num.trees = 500L,
                               mtry = rf_mtry, min.node.size = 5L, seed = 42L),
                        error = function(e) NULL)
        if (!is.null(fit))
          preds[i, gc] <- predict(fit, data = as.data.frame(X_test))$predictions
      }
    }
    if (i %% 100L == 0L) cat(".")
  }
  preds
}

# ── Evaluate a prediction matrix on the common window ─────────────────────────
eval_common <- function(preds, learner, family) {
  map_dfr(gamma_cols, function(gc) {
    idx  <- which(common_mask[[gc]])
    y    <- actuals[idx, gc]
    yhat <- preds[idx, gc]
    bench <- pred_list[[gc]][idx, "hist_mean"]
    valid <- !is.na(yhat) & !is.na(y) & !is.na(bench)
    if (sum(valid) < 10L) return(NULL)
    cw <- clark_west(y[valid], bench[valid], yhat[valid])
    tibble(gamma = gamma_labels[gc], learner = learner, dropped = family,
           oos_r2 = round(oos_r2(y[valid], yhat[valid], bench[valid]) * 100, 2),
           t_cw = cw$t_cw, p_cw = cw$p_cw, n = sum(valid))
  })
}

# ── Run: full baseline (recomputed here for an apples-to-apples comparison) ───
cat("\nBaseline (all families):\n  lasso "); base_lasso <- run_subset(feature_cols, "lasso")
cat(" rf ");                                base_rf    <- run_subset(feature_cols, "rf")

results <- bind_rows(
  eval_common(base_lasso, "lasso", "None (full)"),
  eval_common(base_rf,    "rf",    "None (full)")
)

# ── Run: drop each family in turn ─────────────────────────────────────────────
for (fam in families) {
  keep <- feature_cols[feat_family != fam]
  cat(sprintf("\nDrop %-22s (%d cols left):\n  lasso ", fam, length(keep)))
  p_l <- run_subset(keep, "lasso"); cat(" rf "); p_r <- run_subset(keep, "rf")
  results <- bind_rows(results,
    eval_common(p_l, "lasso", fam),
    eval_common(p_r, "rf",    fam))
}
cat("\n")

# ── Marginal contribution = full OOS R² − ablated OOS R² ──────────────────────
base_r2 <- results |> filter(dropped == "None (full)") |>
  select(gamma, learner, full_r2 = oos_r2)
marginal <- results |> filter(dropped != "None (full)") |>
  left_join(base_r2, by = c("gamma", "learner")) |>
  mutate(delta = round(full_r2 - oos_r2, 2))    # >0 : family helps

saveRDS(list(results = results, marginal = marginal, base_r2 = base_r2),
        "ablation_results.rds")
write_csv(marginal, "ablation_results.csv")

cat("\n=== Marginal contribution (full − ablated OOS R²), Size & Beta ===\n")
print(marginal |> filter(gamma %in% c("Size", "Beta")) |>
        select(gamma, learner, dropped, full_r2, ablated = oos_r2, delta) |>
        arrange(gamma, learner, desc(delta)), n = Inf)

cat("\nSaved: ablation_results.rds, ablation_results.csv\n")

# ── LaTeX appendix table (b7) ─────────────────────────────────────────────────
if (requireNamespace("kableExtra", quietly = TRUE)) {
  library(kableExtra)
  char_order <- c("Value","Momentum","Profitability","Asset Growth","Size","Beta")
  wide <- marginal |>
    mutate(learner = recode(learner, lasso = "LASSO", rf = "Random forest"),
           gamma   = factor(gamma, levels = char_order),
           dropped = factor(dropped, levels = families)) |>
    select(learner, gamma, full_r2, dropped, delta) |>
    pivot_wider(names_from = dropped, values_from = delta) |>
    arrange(learner, gamma)
  fmtnum <- function(x) ifelse(is.na(x), "", sprintf("%+.2f", x))
  disp <- wide |>
    mutate(Full = sprintf("%.2f", full_r2)) |>
    select(learner, Gamma = gamma, Full, all_of(families)) |>
    mutate(across(all_of(families), fmtnum))
  cap <- paste0(
    "Drop-one-family-out out-of-sample $R^2$ ablation. Each cell is the marginal ",
    "contribution of a predictor family: the full-model out-of-sample $R^2$ (column ",
    "``Full'') minus the $R^2$ obtained when that family is removed and the model ",
    "re-estimated over the identical expanding-window design and common 414-month ",
    "sample. A positive value means removing the family lowers $R^2$, so the ",
    "family carries unique predictive information for that gamma; a negative value means ",
    "the family is, on net, noise the model forecasts better without. Family sizes: ",
    "own-gamma history 24, characteristic spread 6, macro 8, Goyal--Welch 5, factor ",
    "signal 75.")
  kbl(disp |> select(-learner), format = "latex", booktabs = TRUE, escape = FALSE,
      caption = cap,
      # Short form for the List of Tables (the full caption is far too long there).
      caption.short = "Drop-one-family-out out-of-sample $R^2$ ablation",
      label = "app-ablation",
      align = c("l","r","r","r","r","r","r")) |>
    pack_rows(index = table(factor(disp$learner,
              levels = c("LASSO","Random forest"))), escape = FALSE) |>
    # HOLD_position ([H]) pins the float: with the default specifier this table
    # drifted out of Annex B into Annex D, so "Table B.7" appeared after D.3.
    kable_styling(latex_options = c("scale_down", "HOLD_position"), font_size = 9) |>
    as.character() |> writeLines("tables/b7_ablation.tex")
  if (dir.exists("thesis_drafts/tables"))
    file.copy("tables/b7_ablation.tex", "thesis_drafts/tables/b7_ablation.tex",
              overwrite = TRUE)
  cat("Saved: tables/b7_ablation.tex\n")
}
