# ==============================================================================
# colab/import_lstm_from_colab.R
# Bridge step 3 of 3 (LSTM): merge Colab LSTM predictions into gamma_predictions.rds.
#
# Replaces ONLY the training section of 04c_predict_lstm.R: predictions are read
# from colab_io/lstm_preds.csv instead of being trained locally. There is NO
# Campbell-Thompson restriction on LSTM (raw only), matching 04c. The Clark-West
# inference and saveRDS logic are copied verbatim from 04c.
#
# Prereq: run colab/export_for_colab.R, then the 04c notebook on Colab, then
# download colab_io/lstm_preds.csv back to colab/colab_io/.
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(sandwich)   # NeweyWest() for Clark-West inference

PROJECT_DIR <- Sys.getenv("THESIS_DIR",
  unset = "D:/OneDrive - University of Liechtenstein/Master-Studium/07_Thesis/Code")
IO_DIR      <- file.path(PROJECT_DIR, "colab", "colab_io")
NW_LAG      <- 12L

gamma_cols <- c(
  "gamma_bm", "gamma_mom12m", "gamma_oper_prof",
  "gamma_asset_growth", "gamma_size", "gamma_beta"
)
gamma_labels <- c(
  gamma_bm = "Value", gamma_mom12m = "Momentum", gamma_oper_prof = "Profitability",
  gamma_asset_growth = "Asset Growth", gamma_size = "Size", gamma_beta = "Beta"
)

# ── Load the R pipeline state and strip any prior lstm* columns ───────────────
gp <- readRDS(file.path(PROJECT_DIR, "gamma_predictions.rds"))
list2env(gp, envir = environment())   # -> pred_list, oos_eval, actuals, dates_oos, ...
for (.gc in names(pred_list)) {
  .drop <- grepl("^lstm(_|$)", colnames(pred_list[[.gc]]))
  pred_list[[.gc]] <- pred_list[[.gc]][, !.drop, drop = FALSE]
}
rm(.gc, .drop)

# ── Read Colab predictions and rebuild lstm_variants ──────────────────────────
preds_csv <- readr::read_csv(file.path(IO_DIR, "lstm_preds.csv"),
                             show_col_types = FALSE)
n_oos     <- nrow(preds_csv)
pred_cols <- setdiff(colnames(preds_csv), c("step", "date"))
# Per-seed columns ({variant}_seed{N}__…) come from the GKX 5-seed ensemble in
# the notebook. They are used ONLY for the across-seed dispersion report below
# and are NEVER merged into pred_list / oos_eval, so the 04f FDR family and the
# 05 portfolio loop see the ensemble averages only.
vnames      <- unique(sub("__.*$", "", pred_cols))
seed_vnames <- grep("_seed[0-9]+$", vnames, value = TRUE)
vnames      <- setdiff(vnames, seed_vnames)
cat("LSTM variants found in CSV:", paste(vnames, collapse = ", "), "\n")
cat("Per-seed columns:", length(seed_vnames), "variant-seed combinations\n")

.build_mat <- function(v) {
  m <- matrix(NA_real_, nrow = n_oos, ncol = length(gamma_cols),
              dimnames = list(NULL, gamma_cols))
  for (gc in gamma_cols) {
    col <- paste0(v, "__", gc)
    if (col %in% pred_cols) m[, gc] <- preds_csv[[col]]
  }
  m
}
lstm_variants      <- setNames(lapply(vnames, .build_mat), vnames)
lstm_seed_variants <- setNames(lapply(seed_vnames, .build_mat), seed_vnames)

# ── Common-window mask: restrict LSTM to the rows the MLP actually evaluates ───
# The Colab LSTM imputes its feature NAs internally and therefore scores all 421
# OOS months, whereas the full-feature MLP (and LASSO) skip any month with an NA
# in the feature row (anyNA(X_te) -> skip). Those skipped months are the recent
# Goyal-Welch earnings-publication-lag tail that we decided NOT to mean-impute
# (CLAUDE.md, DECISION 2026-06-19). Scoring the LSTM on 421 while MLP sees 414
# (a) puts the two models on different evaluation windows and (b) lets the LSTM
# predict the recent tail from imputed macro/GW features we would not have had
# live. We therefore NA-out every LSTM prediction whose paired MLP variant is NA,
# so all models share one honest, impute-free OOS window.
#   pairing: lstm -> mlp ; lstm_nogeo -> mlp_nogeo  (sub "^lstm" -> "mlp")
# Seed variants inherit the mask of their base variant (lstm_seed43 -> mlp), so
# the dispersion report is computed on the same honest window as the ensemble.
for (vname in names(lstm_variants)) {
  mlp_pair <- sub("^lstm", "mlp", vname)
  for (gc in gamma_cols) {
    if (!mlp_pair %in% colnames(pred_list[[gc]])) {
      stop(sprintf(
        "Mask source '%s' absent from pred_list[['%s']]; run import_mlp_from_colab.R before this script.",
        mlp_pair, gc))
    }
    lstm_variants[[vname]][is.na(pred_list[[gc]][, mlp_pair]), gc] <- NA_real_
  }
}
for (vname in names(lstm_seed_variants)) {
  mlp_pair <- sub("^lstm", "mlp", sub("_seed[0-9]+$", "", vname))
  for (gc in gamma_cols) {
    if (!mlp_pair %in% colnames(pred_list[[gc]])) {
      stop(sprintf(
        "Mask source '%s' absent from pred_list[['%s']]; run import_mlp_from_colab.R before this script.",
        mlp_pair, gc))
    }
    lstm_seed_variants[[vname]][is.na(pred_list[[gc]][, mlp_pair]), gc] <- NA_real_
  }
}
rm(vname, mlp_pair, gc)

# ── Append raw LSTM columns (no CT — verbatim from 04c) ────────────────────────
for (gc in gamma_cols) {
  for (vname in names(lstm_variants)) {
    raw <- lstm_variants[[vname]][, gc]
    m   <- pred_list[[gc]]
    if (vname %in% colnames(m)) m <- m[, colnames(m) != vname, drop = FALSE]
    pred_list[[gc]] <- cbind(m, setNames(data.frame(raw), vname))
  }
}

# ── Clark-West (verbatim from 04c) ────────────────────────────────────────────
clark_west <- function(y, yhat_bench, yhat_model, nw_lags = NW_LAG) {
  f  <- (y - yhat_bench)^2 - ((y - yhat_model)^2 - (yhat_bench - yhat_model)^2)
  ok <- !is.na(f)
  n  <- sum(ok)
  if (n < 10L) return(list(t_cw = NA_real_, p_cw = NA_real_))
  f_ok   <- f[ok]
  nw_var <- tryCatch(
    as.numeric(NeweyWest(lm(f_ok ~ 1), lag = nw_lags,
                         prewhite = FALSE, adjust = TRUE)),
    error = function(e) var(f_ok) / n
  )
  if (!is.finite(nw_var) || nw_var <= 0) nw_var <- var(f_ok) / n
  t_stat <- mean(f_ok) / sqrt(nw_var)
  list(t_cw = round(t_stat, 3L), p_cw = round(1 - pnorm(t_stat), 3L))
}

oos_r2 <- function(y, yhat, yhat_bench) {
  1 - sum((y - yhat)^2,       na.rm = TRUE) /
      sum((y - yhat_bench)^2, na.rm = TRUE)
}

lstm_model_cols <- names(lstm_variants)

new_eval <- map_dfr(gamma_cols, function(gc) {
  y     <- actuals[, gc]
  bench <- pred_list[[gc]][, "hist_mean"]
  map_dfr(lstm_model_cols, function(m) {
    if (!m %in% colnames(pred_list[[gc]])) return(NULL)
    yhat  <- pred_list[[gc]][, m]
    valid <- !is.na(yhat) & !is.na(y)
    if (sum(valid) < 10L) return(NULL)
    cw <- clark_west(y[valid], bench[valid], yhat[valid])
    tibble(
      gamma      = gamma_labels[gc],
      sub_period = "Full",
      model      = m,
      oos_r2     = round(oos_r2(y[valid], yhat[valid], bench[valid]) * 100, 2L),
      msfe       = round(mean((y[valid] - yhat[valid])^2), 6L),
      msfe_ratio = round(
        mean((y[valid] - yhat[valid])^2) / mean((y[valid] - bench[valid])^2), 3L
      ),
      t_cw  = cw$t_cw,
      p_cw  = cw$p_cw,
      n_oos = sum(valid)
    )
  })
})

oos_eval <- bind_rows(
  oos_eval |> filter(!model %in% lstm_model_cols),
  new_eval
)

cat("\n=== LSTM OOS R² (%) ===\n")
print(
  new_eval |>
    dplyr::select(gamma, model, oos_r2, t_cw, p_cw, n_oos) |>
    pivot_wider(names_from = model, values_from = c(oos_r2, t_cw, p_cw, n_oos)),
  n = Inf
)

# ── Across-seed dispersion report (GKX ensemble; Meeting 3 mandate) ───────────
# Per-seed OOS R² vs. the ensemble average, on the common-window mask. Written
# to CSV for the thesis writeup; deliberately NOT merged into pred_list/oos_eval.
if (length(seed_vnames) > 0L) {
  seed_eval <- map_dfr(gamma_cols, function(gc) {
    y     <- actuals[, gc]
    bench <- pred_list[[gc]][, "hist_mean"]
    map_dfr(seed_vnames, function(v) {
      yhat  <- lstm_seed_variants[[v]][, gc]
      valid <- !is.na(yhat) & !is.na(y)
      if (sum(valid) < 10L) return(NULL)
      tibble(
        gamma   = gamma_labels[gc],
        variant = sub("_seed[0-9]+$", "", v),
        seed    = as.integer(sub("^.*_seed", "", v)),
        oos_r2  = round(oos_r2(y[valid], yhat[valid], bench[valid]) * 100, 2L),
        n_oos   = sum(valid)
      )
    })
  })
  seed_disp <- seed_eval |>
    group_by(gamma, variant) |>
    summarise(n_seeds = n(),
              mean_r2 = round(mean(oos_r2), 2L), sd_r2 = round(sd(oos_r2), 2L),
              min_r2  = min(oos_r2),             max_r2 = max(oos_r2),
              .groups = "drop") |>
    left_join(new_eval |> filter(model %in% names(lstm_variants)) |>
                dplyr::select(gamma, variant = model, ensemble_r2 = oos_r2),
              by = c("gamma", "variant"))
  readr::write_csv(seed_eval, file.path(PROJECT_DIR, "seed_dispersion_lstm_per_seed.csv"))
  readr::write_csv(seed_disp, file.path(PROJECT_DIR, "seed_dispersion_lstm.csv"))
  cat("\n=== LSTM across-seed OOS R² dispersion ===\n")
  print(seed_disp, n = Inf)
  cat("Saved: seed_dispersion_lstm.csv / seed_dispersion_lstm_per_seed.csv\n")
}

# ── Rebuild common-window evaluation (oos_eval_common) ────────────────────────
# pred_list now carries the freshly-imported MLP + LSTM ensemble columns, so the
# common-window table that feeds Tables 3/3b/6/8 and the 04f FDR family MUST be
# recomputed here. The import steps above update `oos_eval` (each model on its
# own non-NA window) but NOT `oos_eval_common`, which is otherwise a stale
# snapshot persisted by the last full 04 run — i.e. it would still report the
# single-seed NN numbers while pred_list (and the 05 portfolios) carry the
# ensemble. Logic mirrors 04_predict_gammas.R §5b exactly: every model scored on
# the per-gamma set of months where EVERY model (and the actual) is non-missing.
model_names_common <- Reduce(union, lapply(pred_list, colnames))
eval_window_common <- function(preds, y_vec, bench_vec, idx) {
  map_dfr(setdiff(model_names_common, "hist_mean"), function(m) {
    if (!m %in% colnames(preds)) return(NULL)
    yhat  <- preds[idx, m]
    y     <- y_vec[idx]
    bench <- bench_vec[idx]
    valid <- !is.na(yhat) & !is.na(y)
    if (sum(valid) < 10L) return(NULL)
    cw <- clark_west(y[valid], bench[valid], yhat[valid])
    tibble(
      model      = m,
      oos_r2     = round(oos_r2(y[valid], yhat[valid], bench[valid]) * 100, 2L),
      msfe       = round(mean((y[valid] - yhat[valid])^2), 6L),
      msfe_ratio = round(mean((y[valid] - yhat[valid])^2) /
                         mean((y[valid] - bench[valid])^2), 3L),
      t_cw       = cw$t_cw,
      p_cw       = cw$p_cw,
      n_oos      = sum(valid)
    )
  })
}
common_mask <- setNames(lapply(gamma_cols, function(gc) {
  P <- pred_list[[gc]]
  rowSums(is.na(P)) == 0 & !is.na(actuals[, gc])
}), gamma_cols)
oos_eval_common <- map_dfr(gamma_cols, function(gc) {
  idx   <- which(common_mask[[gc]])
  y     <- actuals[, gc]
  bench <- pred_list[[gc]][, "hist_mean"]
  eval_window_common(pred_list[[gc]], y, bench, idx) |>
    mutate(gamma = gamma_labels[gc], sub_period = "Full", .before = model)
})
cat(sprintf("\nRebuilt oos_eval_common: common window = %d-%d months per gamma.\n",
            min(sapply(common_mask, sum)), max(sapply(common_mask, sum))))

# ── Sub-period evaluation for the imported NN models ──────────────────────────
# 04's oos_eval_all carries Pre-2000 / 2000-2010 / Post-2010 rows only for the
# models computed natively in 04; the Colab-imported MLP/LSTM columns had
# Full-window rows only, so sub-period exhibits (plots/oos_r2_subperiod.pdf)
# showed no NN entries. Mirror 04 §5 exactly on the imported columns. Runs here
# (last import) so pred_list already carries BOTH networks' ensemble columns.
sub_windows <- list(
  "Pre-2000"  = which(dates_oos <  as.Date("2000-01-01")),
  "2000-2010" = which(dates_oos >= as.Date("2000-01-01") &
                      dates_oos <  as.Date("2010-01-01")),
  "Post-2010" = which(dates_oos >= as.Date("2010-01-01"))
)
nn_pattern <- "^(mlp|lstm)"
nn_sub <- map_dfr(names(sub_windows), function(sp) {
  idx <- sub_windows[[sp]]
  if (length(idx) < 20L) return(NULL)
  map_dfr(gamma_cols, function(gc) {
    P     <- pred_list[[gc]]
    y     <- actuals[idx, gc]
    bench <- P[idx, "hist_mean"]
    map_dfr(grep(nn_pattern, colnames(P), value = TRUE), function(m) {
      yhat  <- P[idx, m]
      valid <- !is.na(yhat) & !is.na(y)
      if (sum(valid) < 10L) return(NULL)
      cw <- clark_west(y[valid], bench[valid], yhat[valid])
      tibble(
        gamma      = gamma_labels[gc],
        sub_period = sp,
        model      = m,
        oos_r2     = round(oos_r2(y[valid], yhat[valid], bench[valid]) * 100, 2L),
        msfe       = round(mean((y[valid] - yhat[valid])^2), 6L),
        msfe_ratio = round(mean((y[valid] - yhat[valid])^2) /
                           mean((y[valid] - bench[valid])^2), 3L),
        t_cw       = cw$t_cw,
        p_cw       = cw$p_cw,
        n_oos      = sum(valid)
      )
    })
  })
})
cat("\n=== NN sub-period OOS R² (%) — Size ===\n")
print(nn_sub |> filter(gamma == "Size", model %in% c("mlp", "lstm")) |>
        dplyr::select(sub_period, model, oos_r2, t_cw, p_cw, n_oos), n = Inf)

# ── Save ──────────────────────────────────────────────────────────────────────
gp <- readRDS(file.path(PROJECT_DIR, "gamma_predictions.rds"))
gp$pred_list <- pred_list; gp$oos_eval <- oos_eval
gp$actuals   <- actuals;   gp$dates_oos <- dates_oos
gp$oos_eval_common <- oos_eval_common
# oos_eval_all = Full rows for every model (incl. fresh NN ensembles) + native
# sub-period rows from the last 04 run + the NN sub-period rows computed above.
gp$oos_eval_all <- bind_rows(
  oos_eval,
  gp$oos_eval_all |> filter(sub_period != "Full", !grepl(nn_pattern, model)),
  nn_sub
)
saveRDS(gp, file = file.path(PROJECT_DIR, "gamma_predictions.rds"))
cat("\nSaved: gamma_predictions.rds (lstm columns + oos_eval_common + NN sub-periods in oos_eval_all)\n")
