# ==============================================================================
# 04i_ls_horserace.R — Does forecasting the gamma beat forecasting the factor?
# ------------------------------------------------------------------------------
# The thesis argues that forecasting Fama-MacBeth slopes is a different and more
# primitive exercise than forecasting the returns of pre-constructed factor
# portfolios. That claim needs a counterfactual, not an assertion: run the
# IDENTICAL design on the factor-return target and compare. If forecasting the
# long-short quintile spreads directly produces the same timing gains, the
# gamma-level framing is a relabelling; if it does not, the framing earns its
# place.
#
# Design — everything is held fixed except the target:
#   * the same feature matrix (`aligned`, `feature_cols`) built in 04, so the
#     information set is identical month by month;
#   * the same expanding-window OOS loop, the same initial window and the same
#     evaluation months;
#   * the same estimators and hyperparameters (LASSO with a time-series
#     validation split, ranger with mtry = p/3, xgboost with early stopping);
#   * the same historical-mean benchmark, computed on whichever target is being
#     forecast.
#
# Two feature variants are run, because the fairness of the comparison turns on
# one choice. The 118-column design contains 24 own-gamma history columns (three
# lags plus a 12-month volatility for each of the six premia).
#   Variant A ("same features"): the literal counterfactual — the LS model gets
#     exactly the 118 columns the gamma model had, own-gamma lags included.
#   Variant B ("own history"): the 24 own-gamma columns are replaced by the
#     equivalent own-LS-return columns, so each target is conditioned on its own
#     past rather than on the other object's past.
# Reporting only A would invite the objection that the LS model was denied its
# own lags; reporting only B would change two things at once. Both are reported.
#
# Neural networks are excluded: they run on Colab through the CSV bridge and
# re-running that grid for a counterfactual is disproportionate. The comparison
# is therefore between the penalised-linear and tree learners, which is where the
# gamma pipeline's own headline sits (Comb+XGB, LASSO, RF).
#
# Inputs : gamma_predictions.rds (aligned, feature_cols, oos_idx, actuals,
#          pred_list, dates_oos), panel_clean.rds, ff5_factors.rds,
#          umd_factor.rds, portfolio_results.rds (gamma-side book to compare to)
# Outputs: ls_horserace.rds, ls_horserace.csv, tables/b9_ls_horserace.tex
# Run    : after 04_predict_gammas.R and 05_portfolio_construction.R
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(glmnet)
  library(ranger)
  library(xgboost)
  library(slider)
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
dates_oos    <- gp$dates_oos

panel_clean <- readRDS("panel_clean.rds")
ff5 <- readRDS("ff5_factors.rds") |>
  left_join(readRDS("umd_factor.rds"), by = "date")
pr  <- readRDS("portfolio_results.rds")

char_gamma_map <- tribble(
  ~char_col,            ~gamma_col,
  "bm_std",             "gamma_bm",
  "mom12m_std",         "gamma_mom12m",
  "oper_prof_std",      "gamma_oper_prof",
  "asset_growth_std",   "gamma_asset_growth",
  "log_mktcap_lag_std", "gamma_size",
  "beta_std",           "gamma_beta"
)
gamma_cols   <- char_gamma_map$gamma_col
gamma_labels <- c(gamma_bm = "Value", gamma_mom12m = "Momentum",
                  gamma_oper_prof = "Profitability",
                  gamma_asset_growth = "Asset Growth",
                  gamma_size = "Size", gamma_beta = "Beta")

# ── Long-short quintile returns, built exactly as in 05 ───────────────────────
# Quintiles on the rank-standardised characteristic, q5 minus q1, formed within
# each month on the full 1963-2025 panel.
ls_wide <- panel_clean |>
  group_by(date) |>
  mutate(across(all_of(char_gamma_map$char_col), \(x) ntile(x, 5),
                .names = "{.col}_q")) |>
  ungroup() |>
  pivot_longer(cols = ends_with("_q"), names_to = "char_col",
               values_to = "quintile") |>
  mutate(char_col = str_remove(char_col, "_q")) |>
  filter(quintile %in% c(1L, 5L)) |>
  group_by(date, char_col, quintile) |>
  summarise(avg_ret = mean(ret_excess, na.rm = TRUE), .groups = "drop") |>
  pivot_wider(names_from = quintile, values_from = avg_ret, names_prefix = "q") |>
  mutate(ls_ret = q5 - q1) |>
  inner_join(char_gamma_map, by = "char_col") |>
  select(date, gamma_col, ls_ret) |>
  pivot_wider(names_from = gamma_col, values_from = ls_ret) |>
  arrange(date)

# ── Attach the LS target to the aligned matrix ────────────────────────────────
# aligned$date is the OUTCOME date (features sit at t, outcome at t+1), so the
# LS return is joined on that same date and inherits the alignment unchanged.
ls_target <- as.matrix(
  ls_wide[match(aligned$date, ls_wide$date), gamma_cols, drop = FALSE]
)
colnames(ls_target) <- gamma_cols
stopifnot(nrow(ls_target) == nrow(aligned))
cat(sprintf("LS target attached: %d rows, %d complete\n",
            nrow(ls_target), sum(complete.cases(ls_target))))

gamma_target <- as.matrix(aligned[, gamma_cols])

# ── Variant B features: own-LS history in place of own-gamma history ──────────
gamma_short <- c(gamma_bm = "bm", gamma_mom12m = "mom12m",
                 gamma_oper_prof = "oper_prof",
                 gamma_asset_growth = "asset_growth",
                 gamma_size = "size", gamma_beta = "beta")
own_hist_cols <- feature_cols[
  grepl("_l[123]$", feature_cols) |
  grepl(paste0("^(", paste(gamma_short, collapse = "|"), ")_vol12$"), feature_cols)
]
cat(sprintf("Own-gamma history columns to be swapped in variant B: %d\n",
            length(own_hist_cols)))

# The LS analogues are built on the SAME timing rule as the gamma features: at
# outcome-row i the features describe information through the previous month.
# ls_wide is indexed by outcome date, so lag k of the LS return at outcome row i
# is the LS return k months before aligned$date[i].
ls_by_date <- ls_wide[match(aligned$date, ls_wide$date), gamma_cols, drop = FALSE]
ls_feat <- map_dfc(gamma_cols, function(gc) {
  s <- gamma_short[[gc]]
  x <- ls_by_date[[gc]]
  out <- tibble(
    !!paste0(s, "_ls_l1") := dplyr::lag(x, 1L),
    !!paste0(s, "_ls_l2") := dplyr::lag(x, 2L),
    !!paste0(s, "_ls_l3") := dplyr::lag(x, 3L),
    !!paste0(s, "_ls_vol12") := slider::slide_dbl(
      x, \(v) if (sum(!is.na(v)) >= 6L) sd(v, na.rm = TRUE) else NA_real_,
      .before = 12L, .after = -1L)
  )
  out
})
stopifnot(ncol(ls_feat) == length(own_hist_cols))

aligned_B <- bind_cols(aligned[, setdiff(names(aligned), own_hist_cols)], ls_feat)
feature_cols_B <- c(setdiff(feature_cols, own_hist_cols), names(ls_feat))

# ── Helpers (verbatim from 04 / 04g) ──────────────────────────────────────────
impute_means <- function(X) {
  for (j in seq_len(ncol(X))) {
    na_j <- is.na(X[, j])
    if (any(na_j)) X[na_j, j] <- if (any(!na_j)) mean(X[!na_j, j]) else 0
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
  val_mse <- colMeans((y[val_idx] - predict(fit_path, newx = X[val_idx, ]))^2,
                      na.rm = TRUE)
  lam_min <- fit_path$lambda[which.min(val_mse)]
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

# ── Expanding-window OOS loop, target-agnostic ────────────────────────────────
# `Y` is the outcome matrix (gamma or LS); `A` and `fc` the feature source.
# Also returns the expanding historical mean of the target, which is this
# target's own M0 benchmark.
run_oos <- function(Y, A, fc, learner) {
  preds <- matrix(NA_real_, length(oos_idx), length(gamma_cols),
                  dimnames = list(NULL, gamma_cols))
  bench <- matrix(NA_real_, length(oos_idx), length(gamma_cols),
                  dimnames = list(NULL, gamma_cols))
  rf_mtry   <- max(5L, floor(length(fc) / 3L))
  colsample <- round(sqrt(length(fc)) / length(fc), 3)
  for (i in seq_along(oos_idx)) {
    t <- oos_idx[i]
    X_test <- as.matrix(A[t, fc])
    if (anyNA(X_test)) next                         # honest real-time skip
    X_train <- impute_means(as.matrix(A[seq_len(t - 1L), fc]))
    for (gc in gamma_cols) {
      y_tr <- Y[seq_len(t - 1L), gc]
      ok   <- !is.na(y_tr)
      if (sum(ok) < 30L) next
      bench[i, gc] <- mean(y_tr[ok])                # expanding historical mean
      if (learner == "lasso") {
        fit <- tryCatch(ts_cv_lasso(X_train[ok, , drop = FALSE], y_tr[ok]),
                        error = function(e) NULL)
        if (!is.null(fit))
          preds[i, gc] <- predict(fit$fit, newx = X_test, s = fit$lambda.min)[1]
      } else if (learner == "rf") {
        rf_df <- as.data.frame(X_train[ok, , drop = FALSE]); rf_df$.y <- y_tr[ok]
        fit <- tryCatch(ranger(.y ~ ., data = rf_df, num.trees = 500L,
                               mtry = rf_mtry, min.node.size = 5L, seed = 42L),
                        error = function(e) NULL)
        if (!is.null(fit))
          preds[i, gc] <- predict(fit, data = as.data.frame(X_test))$predictions
      } else {                                      # xgb
        tryCatch({
          x_ok  <- X_train[ok, , drop = FALSE]; y_ok <- y_tr[ok]
          n_tr  <- length(y_ok)
          n_val <- max(20L, floor(n_tr * 0.20)); n_fit <- n_tr - n_val
          if (n_fit < 30L) stop("too short")
          dtrain <- xgb.DMatrix(x_ok[seq_len(n_fit), , drop = FALSE],
                                label = y_ok[seq_len(n_fit)])
          dval   <- xgb.DMatrix(x_ok[(n_fit + 1L):n_tr, , drop = FALSE],
                                label = y_ok[(n_fit + 1L):n_tr])
          fit <- xgb.train(
            params = list(objective = "reg:squarederror", max_depth = 4L,
                          eta = 0.05, subsample = 0.8,
                          colsample_bytree = colsample,
                          min_child_weight = 5L, seed = 42L),
            data = dtrain, nrounds = 500L, evals = list(val = dval),
            early_stopping_rounds = 20L, verbose = 0L)
          preds[i, gc] <- predict(fit, newdata = xgb.DMatrix(X_test))
        }, error = function(e) NULL)
      }
    }
    if (i %% 100L == 0L) cat(".")
  }
  list(pred = preds, bench = bench)
}

# ── Common evaluation window ──────────────────────────────────────────────────
# The gamma pipeline's 414-month window, so every Sharpe ratio quoted here is on
# the months the thesis reports elsewhere.
common_mask <- Reduce(`&`, lapply(gamma_cols, function(gc)
  rowSums(is.na(pred_list[[gc]])) == 0 & !is.na(actuals[, gc])))
ls_oos <- ls_target[oos_idx, , drop = FALSE]     # realised LS over OOS rows
gm_oos <- gamma_target[oos_idx, , drop = FALSE]  # realised gammas over OOS rows
cat(sprintf("\nCommon window: %d of %d OOS months\n",
            sum(common_mask), length(common_mask)))

# ── Portfolio evaluation of a set of direction signals ────────────────────────
# The traded object is the SAME long-short book in every case; only the signal
# that sets its sign changes. Static uses the frozen 1963-1989 gamma sign, as in
# 05, so the benchmark is identical to the one used throughout the thesis.
static_gamma <- vapply(gamma_cols, function(gc)
  pred_list[[gc]][min(which(common_mask)), "hist_mean"], numeric(1))

ff_alpha <- function(r, dates, six = FALSE) {
  df <- tibble(date = dates, ret = r) |> inner_join(ff5, by = "date") |>
    filter(!is.na(ret))
  if (nrow(df) < 24L) return(c(alpha = NA, t = NA))
  form <- if (six) ret ~ mkt_excess + smb + hml + rmw + cma + umd
          else     ret ~ mkt_excess + smb + hml + rmw + cma
  fit <- lm(form, data = df)
  ct  <- coeftest(fit, vcov = NeweyWest(fit, lag = 6L, prewhite = FALSE))
  c(alpha = ct["(Intercept)", "Estimate"] * 1200,
    t     = ct["(Intercept)", "t value"])
}

eval_book <- function(signal_mat, label) {
  idx   <- which(common_mask)
  dts   <- dates_oos[idx]
  legs  <- matrix(NA_real_, length(idx), length(gamma_cols),
                  dimnames = list(NULL, gamma_cols))
  for (gc in gamma_cols) {
    s <- sign(signal_mat[idx, gc])
    s[is.na(s) | s == 0] <- NA_real_
    legs[, gc] <- s * ls_oos[idx, gc]
  }
  comb <- rowMeans(legs, na.rm = TRUE)
  ok   <- is.finite(comb)
  a5 <- ff_alpha(comb[ok], dts[ok], six = FALSE)
  a6 <- ff_alpha(comb[ok], dts[ok], six = TRUE)
  tibble(book = label,
         sharpe   = round(mean(comb[ok]) / sd(comb[ok]) * sqrt(12), 3),
         alpha5   = round(a5[["alpha"]], 2), t5 = round(a5[["t"]], 2),
         alpha6   = round(a6[["alpha"]], 2), t6 = round(a6[["t"]], 2),
         n_months = sum(ok))
}

# ── Run both variants for each learner ────────────────────────────────────────
learners <- c("lasso", "rf", "xgb")
stat_rows <- list(); book_rows <- list()

# Static benchmark book (frozen sign, no forecast at all)
static_sig <- matrix(rep(static_gamma, each = length(oos_idx)),
                     nrow = length(oos_idx), dimnames = list(NULL, gamma_cols))
book_rows[["static"]] <- eval_book(static_sig, "Static (frozen sign)")

for (lr in learners) {
  cat(sprintf("\n[%s] gamma target ", lr));    g_out <- run_oos(gamma_target, aligned,   feature_cols,   lr)
  cat(sprintf("\n[%s] LS target (A) ", lr));   a_out <- run_oos(ls_target,    aligned,   feature_cols,   lr)
  cat(sprintf("\n[%s] LS target (B) ", lr));   b_out <- run_oos(ls_target,    aligned_B, feature_cols_B, lr)

  # Statistical accuracy, each target against its OWN historical mean
  for (cfg in list(list(o = g_out, y = gm_oos, tgt = "Gamma",   var = "Same features"),
                   list(o = a_out, y = ls_oos, tgt = "LS return", var = "Same features"),
                   list(o = b_out, y = ls_oos, tgt = "LS return", var = "Own history"))) {
    for (gc in gamma_cols) {
      idx <- which(common_mask)
      y <- cfg$y[idx, gc]; yh <- cfg$o$pred[idx, gc]; bm <- cfg$o$bench[idx, gc]
      v <- is.finite(y) & is.finite(yh) & is.finite(bm)
      if (sum(v) < 10L) next
      cw <- clark_west(y[v], bm[v], yh[v])
      stat_rows[[length(stat_rows) + 1L]] <- tibble(
        learner = lr, target = cfg$tgt, variant = cfg$var,
        characteristic = gamma_labels[[gc]],
        oos_r2 = round(oos_r2(y[v], yh[v], bm[v]) * 100, 2),
        t_cw = cw$t_cw, p_cw = cw$p_cw, n = sum(v))
    }
  }

  book_rows[[paste0(lr, "_g")]] <- eval_book(g_out$pred, sprintf("%s / gamma", lr))
  book_rows[[paste0(lr, "_a")]] <- eval_book(a_out$pred, sprintf("%s / LS (same features)", lr))
  book_rows[[paste0(lr, "_b")]] <- eval_book(b_out$pred, sprintf("%s / LS (own history)", lr))
}

ls_stats <- bind_rows(stat_rows)
ls_books <- bind_rows(book_rows)

cat("\n\n=== Statistical accuracy: gamma target vs LS-return target ===\n")
cat("Each target is scored against its OWN expanding historical mean, so the\n")
cat("two OOS R2 columns are comparable as ratios of explained to benchmark error.\n\n")
ls_stats |>
  pivot_wider(id_cols = c(learner, characteristic),
              names_from = c(target, variant), values_from = oos_r2) |>
  as.data.frame() |> print(row.names = FALSE)

cat("\n=== Economic value: direction timing on the SAME long-short book ===\n")
cat("Only the signal that sets each leg's sign differs across rows.\n\n")
print(as.data.frame(ls_books), row.names = FALSE)

saveRDS(list(stats = ls_stats, books = ls_books,
             static_gamma = static_gamma, n_common = sum(common_mask)),
        "ls_horserace.rds")
write_csv(ls_stats, "ls_horserace.csv")
cat("\nSaved: ls_horserace.rds, ls_horserace.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX appendix tables (b9 statistical, d12 economic)
# ── EXPORT BLOCK ──  (re-runnable on its own from ls_horserace.rds)
# ══════════════════════════════════════════════════════════════════════════════
if (requireNamespace("kableExtra", quietly = TRUE)) {
  library(kableExtra)
  hr <- readRDS("ls_horserace.rds")
  char_order <- c("Value", "Momentum", "Profitability", "Asset Growth", "Size", "Beta")
  lrn_lab <- c(lasso = "LASSO", rf = "Random forest", xgb = "XGBoost")

  # ── b9: statistical accuracy, three targets side by side ────────────────────
  b9 <- hr$stats |>
    mutate(cfg = paste0(target, " / ", variant),
           characteristic = factor(characteristic, levels = char_order),
           learner = factor(lrn_lab[learner],
                            levels = c("LASSO", "Random forest", "XGBoost"))) |>
    select(learner, characteristic, cfg, oos_r2) |>
    pivot_wider(names_from = cfg, values_from = oos_r2) |>
    arrange(learner, characteristic)
  b9_disp <- b9 |>
    transmute(learner,
              Characteristic = characteristic,
              `Gamma` = sprintf("%.2f", `Gamma / Same features`),
              `LS, same features` = sprintf("%.2f", `LS return / Same features`),
              `LS, own history`   = sprintf("%.2f", `LS return / Own history`))
  cap_b9 <- paste0(
    "Out-of-sample $R^2$ (\\%) when the identical design is pointed at a ",
    "different target. ``Gamma'' forecasts the Fama--MacBeth slope, as ",
    "everywhere else in this thesis; the two ``LS'' columns forecast the ",
    "q5$-$q1 quintile spread on the same characteristic. Features, expanding ",
    "window, estimators, hyperparameters and the common 414-month sample are ",
    "held identical; each target is scored against its own expanding ",
    "historical mean. ``Same features'' gives the LS model the identical 118 ",
    "predictors, own-gamma lags included; ``own history'' replaces the 24 ",
    "own-gamma history columns with the equivalent own-LS-return columns, so ",
    "that each target is conditioned on its own past. Supports ",
    "Section~\\ref{sec:res-horserace}.")
  kbl(b9_disp |> select(-learner), format = "latex", booktabs = TRUE,
      escape = FALSE, caption = cap_b9,
      caption.short = "Gamma versus factor-return target: out-of-sample $R^2$",
      label = "app-horserace", align = c("l", "r", "r", "r")) |>
    pack_rows(index = table(b9_disp$learner), escape = FALSE) |>
    kable_styling(latex_options = c("HOLD_position"), font_size = 9) |>
    as.character() |> writeLines("tables/b9_ls_horserace.tex")

  # ── d12: economic value of each signal on the same book ─────────────────────
  d12 <- hr$books |>
    mutate(book = str_replace(book, "^lasso", "LASSO") |>
                  str_replace("^rf", "Random forest") |>
                  str_replace("^xgb", "XGBoost") |>
                  str_replace(" / gamma", " / gamma target")) |>
    transmute(Signal = book,
              Sharpe = sprintf("%.3f", sharpe),
              `$\\alpha_{5}$ (\\%)` = sprintf("%.2f", alpha5),
              `$t$`  = sprintf("%.2f", t5),
              `$\\alpha_{6}$ (\\%)` = sprintf("%.2f", alpha6),
              `$t$ ` = sprintf("%.2f", t6))
  cap_d12 <- paste0(
    "Direction timing driven by gamma forecasts versus by factor-return ",
    "forecasts, common 414-month sample, gross of costs. Every row trades the ",
    "same six long-short quintile books against the same frozen-sign ",
    "static benchmark; only the signal that sets each leg's sign differs, so ",
    "any difference is attributable to the forecast target alone. ",
    "$\\alpha_{5}$ and $\\alpha_{6}$ are annualised five- and six-factor ",
    "alphas with Newey--West (6 lags) $t$-statistics. No pairwise gap in this ",
    "table reaches the minimum detectable effect of roughly 0.44 Sharpe units ",
    "reported in Table~\\ref{tab:app-sharpe-tests}, so none of the ",
    "differences shown is statistically certifiable in either direction. ",
    "Supports Section~\\ref{sec:res-horserace}.")
  kbl(d12, format = "latex", booktabs = TRUE, escape = FALSE, caption = cap_d12,
      caption.short = "Gamma versus factor-return signals: direction timing",
      label = "app-horserace-book", align = c("l", rep("r", 5))) |>
    kable_styling(latex_options = c("HOLD_position"), font_size = 9) |>
    as.character() |> writeLines("tables/d12_ls_horserace_book.tex")

  if (dir.exists("thesis_drafts/tables")) {
    file.copy("tables/b9_ls_horserace.tex",
              "thesis_drafts/tables/b9_ls_horserace.tex", overwrite = TRUE)
    file.copy("tables/d12_ls_horserace_book.tex",
              "thesis_drafts/tables/d12_ls_horserace_book.tex", overwrite = TRUE)
  }
  cat("Saved: tables/b9_ls_horserace.tex, tables/d12_ls_horserace_book.tex\n")
}
