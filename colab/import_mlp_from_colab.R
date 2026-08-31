# ==============================================================================
# colab/import_mlp_from_colab.R
# Bridge step 3 of 3 (MLP): merge Colab MLP predictions into gamma_predictions.rds.
#
# Replaces ONLY the training section of 04b_predict_mlp.R: predictions are read
# from colab_io/mlp_preds.csv instead of being trained locally. The Campbell-
# Thompson (_ct) sign restriction, Clark-West HAC inference, and saveRDS logic
# are copied verbatim from 04b so the result is identical to a local MLP run.
#
# Prereq: run colab/export_for_colab.R, then the 04b notebook on Colab, then
# download colab_io/mlp_preds.csv back to colab/colab_io/.
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(sandwich)   # NeweyWest() for HAC-robust Clark-West

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

# ── Load the R pipeline state and strip any prior mlp* columns ─────────────────
gp <- readRDS(file.path(PROJECT_DIR, "gamma_predictions.rds"))
list2env(gp, envir = environment())   # -> pred_list, oos_eval, actuals, dates_oos, ...
for (.gc in names(pred_list)) {
  .drop <- grepl("^mlp(_|$)", colnames(pred_list[[.gc]]))
  pred_list[[.gc]] <- pred_list[[.gc]][, !.drop, drop = FALSE]
}
rm(.gc, .drop)

# ── Read Colab predictions and rebuild mlp_variants (list of n_oos x 6 mats) ──
preds_csv <- readr::read_csv(file.path(IO_DIR, "mlp_preds.csv"),
                             show_col_types = FALSE)
n_oos     <- nrow(preds_csv)
pred_cols <- setdiff(colnames(preds_csv), c("step", "date"))

# Variant names are the prefixes before "__"; preserve first-seen order.
# Per-seed columns ({variant}_seed{N}__…) come from the GKX 5-seed ensemble in
# the notebook. They are used ONLY for the across-seed dispersion report below
# and are NEVER merged into pred_list / oos_eval, so the 04f FDR family and the
# 05 portfolio loop see the ensemble averages only.
vnames      <- unique(sub("__.*$", "", pred_cols))
seed_vnames <- grep("_seed[0-9]+$", vnames, value = TRUE)
vnames      <- setdiff(vnames, seed_vnames)
cat("MLP variants found in CSV:", paste(vnames, collapse = ", "), "\n")
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
mlp_variants      <- setNames(lapply(vnames, .build_mat), vnames)
mlp_seed_variants <- setNames(lapply(seed_vnames, .build_mat), seed_vnames)

# ── Campbell-Thompson sign restriction (verbatim from 04b) ────────────────────
for (gc in gamma_cols) {
  bench <- pred_list[[gc]][, "hist_mean"]
  for (vname in names(mlp_variants)) {
    raw    <- mlp_variants[[vname]][, gc]
    ct_raw <- ifelse(!is.na(raw) & !is.na(bench) & sign(raw) != sign(bench),
                     bench, raw)
    pred_list[[gc]] <- cbind(
      pred_list[[gc]],
      setNames(data.frame(raw, ct_raw), c(vname, paste0(vname, "_ct")))
    )
  }
}

# ── Clark-West (2007) HAC-robust test (verbatim from 04b) ──────────────────────
clark_west <- function(y, yhat_bench, yhat_model, nw_lags = NW_LAG) {
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
  if (!is.finite(nw_var) || nw_var <= 0) nw_var <- var(f_ok) / n
  t_stat <- mean(f_ok) / sqrt(nw_var)
  list(t_cw = round(t_stat, 3L), p_cw = round(1 - pnorm(t_stat), 3L))
}

oos_r2 <- function(y, yhat, yhat_bench) {
  1 - sum((y - yhat)^2,      na.rm = TRUE) /
      sum((y - yhat_bench)^2, na.rm = TRUE)
}

mlp_model_cols <- unlist(lapply(names(mlp_variants),
                                \(v) c(v, paste0(v, "_ct"))))

new_eval <- map_dfr(gamma_cols, function(gc) {
  y     <- actuals[, gc]
  bench <- pred_list[[gc]][, "hist_mean"]
  map_dfr(mlp_model_cols, function(m) {
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
  oos_eval |> filter(!model %in% mlp_model_cols),
  new_eval
)

cat("\n=== MLP OOS R² (%) ===\n")
if (nrow(new_eval) == 0L) {
  cat("WARNING: new_eval empty — all MLP predictions NA. Check mlp_preds.csv.\n")
} else {
  print(
    new_eval |>
      dplyr::select(gamma, model, oos_r2, t_cw, p_cw) |>
      pivot_wider(names_from = model, values_from = c(oos_r2, t_cw, p_cw)),
    n = Inf
  )
}

# ── Across-seed dispersion report (GKX ensemble; Meeting 3 mandate) ───────────
# Per-seed OOS R² (raw, no CT) vs. the ensemble average. Written to CSV for the
# thesis writeup; deliberately NOT merged into pred_list / oos_eval.
if (length(seed_vnames) > 0L) {
  seed_eval <- map_dfr(gamma_cols, function(gc) {
    y     <- actuals[, gc]
    bench <- pred_list[[gc]][, "hist_mean"]
    map_dfr(seed_vnames, function(v) {
      yhat  <- mlp_seed_variants[[v]][, gc]
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
    left_join(new_eval |> filter(model %in% names(mlp_variants)) |>
                dplyr::select(gamma, variant = model, ensemble_r2 = oos_r2),
              by = c("gamma", "variant"))
  readr::write_csv(seed_eval, file.path(PROJECT_DIR, "seed_dispersion_mlp_per_seed.csv"))
  readr::write_csv(seed_disp, file.path(PROJECT_DIR, "seed_dispersion_mlp.csv"))
  cat("\n=== MLP across-seed OOS R² dispersion (raw, no CT) ===\n")
  print(seed_disp, n = Inf)
  cat("Saved: seed_dispersion_mlp.csv / seed_dispersion_mlp_per_seed.csv\n")
}

# ── Save (merge back, preserve everything else) ───────────────────────────────
gp <- readRDS(file.path(PROJECT_DIR, "gamma_predictions.rds"))
gp$pred_list <- pred_list; gp$oos_eval <- oos_eval
gp$actuals   <- actuals;   gp$dates_oos <- dates_oos
saveRDS(gp, file = file.path(PROJECT_DIR, "gamma_predictions.rds"))
cat("Saved: gamma_predictions.rds (mlp columns merged from Colab)\n")
