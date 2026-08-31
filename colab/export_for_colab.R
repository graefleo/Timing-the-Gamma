# ==============================================================================
# colab/export_for_colab.R
# Bridge step 1 of 3: dump the objects the MLP / LSTM Colab notebooks need into
# plain CSVs that Python can read.
#
# Run this LOCALLY, once, AFTER 04_predict_gammas.R has finished (i.e. once
# gamma_predictions.rds exists). It does NOT touch 04_predict_gammas.R or the
# .rds file — it only reads.
#
# Workflow:
#   1. Rscript colab/export_for_colab.R              -> writes colab/colab_io/*.csv
#   2. Upload colab/colab_io/ to Colab (or Google Drive); run the two notebooks
#      (04b_predict_mlp.ipynb, 04c_predict_lstm.ipynb) -> they write
#      mlp_preds.csv / lstm_preds.csv back into colab_io/
#   3. Download colab_io/ back; run colab/import_mlp_from_colab.R and
#      colab/import_lstm_from_colab.R -> merges predictions into gamma_predictions.rds
#
# The notebooks reproduce exactly the R objects:
#   aligned[t, feature_cols] = features at time t ; aligned[t, gamma_cols] = gamma_{t+1}
#   oos_idx                  = 1-based row indices of aligned that are evaluated OOS
#   hist_mean                = expanding-mean benchmark (for Campbell-Thompson + CW)
#   actuals                  = realised gamma at each OOS step
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))

# Resolve paths relative to the project root regardless of where Rscript is launched.
# Override with env var THESIS_DIR (used by the round-trip test); defaults to the project root.
PROJECT_DIR <- Sys.getenv("THESIS_DIR",
  unset = "D:/OneDrive - University of Liechtenstein/Master-Studium/07_Thesis/Code")
IO_DIR      <- file.path(PROJECT_DIR, "colab", "colab_io")
dir.create(IO_DIR, recursive = TRUE, showWarnings = FALSE)

rds_path <- file.path(PROJECT_DIR, "gamma_predictions.rds")
if (!file.exists(rds_path)) {
  stop("gamma_predictions.rds not found — run 04_predict_gammas.R first.")
}

gp <- readRDS(rds_path)
invisible(list2env(gp, envir = environment()))   # -> pred_list, actuals, dates_oos,
                                                  #    aligned, feature_cols, oos_idx, ...

gamma_cols <- c(
  "gamma_bm", "gamma_mom12m", "gamma_oper_prof",
  "gamma_asset_growth", "gamma_size", "gamma_beta"
)

# ── aligned: date + features + gamma outcomes (full series, all rows) ──────────
aligned_out <- aligned |>
  dplyr::select(date, dplyr::all_of(feature_cols), dplyr::all_of(gamma_cols))
readr::write_csv(aligned_out, file.path(IO_DIR, "aligned.csv"))

# ── feature column names (preserve order; notebook reads by name) ─────────────
writeLines(feature_cols, file.path(IO_DIR, "feature_cols.txt"))

# ── oos_idx: 1-based row indices into aligned that are evaluated OOS ───────────
readr::write_csv(tibble(oos_idx = as.integer(oos_idx)),
                 file.path(IO_DIR, "oos_idx.csv"))

# ── hist_mean benchmark per gamma (n_oos x 6) ─────────────────────────────────
hist_mean_mat <- sapply(gamma_cols, function(gc) pred_list[[gc]][, "hist_mean"])
colnames(hist_mean_mat) <- gamma_cols
readr::write_csv(as_tibble(hist_mean_mat), file.path(IO_DIR, "hist_mean.csv"))

# ── actuals: realised gamma at each OOS step (n_oos x 6) ───────────────────────
actuals_mat <- as.matrix(actuals)
colnames(actuals_mat) <- gamma_cols
readr::write_csv(as_tibble(actuals_mat), file.path(IO_DIR, "actuals.csv"))

# ── dates_oos: date label for each OOS step ───────────────────────────────────
readr::write_csv(tibble(date = dates_oos), file.path(IO_DIR, "dates_oos.csv"))

cat(sprintf(
  "Exported to %s\n  aligned: %d rows x %d cols (%d features + %d gammas + date)\n  oos steps: %d  | OOS window: %s .. %s\n",
  IO_DIR, nrow(aligned_out), ncol(aligned_out),
  length(feature_cols), length(gamma_cols), length(oos_idx),
  format(min(dates_oos), "%Y-%m"), format(max(dates_oos), "%Y-%m")
))
cat("Next: upload colab/colab_io/ to Colab and run the two notebooks.\n")
