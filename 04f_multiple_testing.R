# ==============================================================================
# 04f_multiple_testing.R
# Multiple-testing controls for OOS predictability.
#
# Motivation (referee/defense point): with ~17 models x 6 gammas of Clark-West
# tests, per-test significance does NOT control the family-wise error / false
# discovery rate. A handful of "significant" OOS R² can arise by chance alone.
# This script adds two standard controls:
#
#   PART 1 — False Discovery Rate (FDR) on the Clark-West p-values
#            Benjamini-Hochberg (1995) and Benjamini-Yekutieli (2001, robust to
#            arbitrary dependence) across the whole family of (model x gamma)
#            tests, and within each gamma. Harvey, Liu & Zhu (2016, RFS) advocate
#            exactly this for the cross-section of predictability tests.
#
#   PART 2 — Model Confidence Set (Hansen, Lunde & Nason 2011, Econometrica)
#            Per gamma, the subset of models that contains the best forecaster
#            with (1 - alpha) confidence, eliminating significantly inferior
#            models via a stationary-bootstrap range statistic. The naive
#            benchmark (hist_mean) is included as a candidate: if it survives in
#            the MCS, no model significantly beats it for that gamma.
#
# Robust to whichever model columns are present in gamma_predictions.rds, so it
# can be re-run after the Colab MLP/LSTM predictions are merged.
#
# The FDR family runs on the COMMON-window evaluation (oos_eval_common) so every
# (model x gamma) test shares one 414-month sample; the MCS reports the 90% and
# 75% confidence sets plus per-model MCS p-values (low-power corroboration —
# FDR is the primary control).
#
# Input  : gamma_predictions.rds  (pred_list, actuals, oos_eval_common, ...)
# Output : multiple_testing_results.rds  (fdr_table, mcs_table, mcs_pval_table, mcs_objects)
#          + console summary; CSVs (…_fdr, …_mcs, …_mcs_pvals) for the thesis tables.
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(MCS))
suppressPackageStartupMessages(library(sandwich))   # NeweyWest() for Part 1b

set.seed(42L)
MCS_ALPHA <- 0.10     # primary MCS confidence level: 90% confidence set
MCS_ALPHA2<- 0.25     # secondary (stricter) level: 75% set — shows the set shrinks
MCS_B     <- 3000L    # bootstrap replications
MCS_STAT  <- "Tmax"   # range statistic (T_max,M of Hansen-Lunde-Nason)
FDR_LEVEL <- 0.10     # target false discovery rate
BENCH     <- "hist_mean"

# Target horizon: "h1" (default, gamma_predictions.rds), "h3" (quarterly Ext-C),
# or "smooth" (HAR backward-smoothed Ext-C2). Override with env var MT_TARGET.
# CAVEAT: at "smooth", 2/3 of the target is known at prediction time, so the
# survivor counts are not comparable to h1/h3 (validation of the HAR spec only).
TARGET <- Sys.getenv("MT_TARGET", unset = "h1")
cfg <- switch(TARGET,
  h1     = list(file = "gamma_predictions.rds",         pl = "pred_list",     ac = "actuals",     ev = "oos_eval",     evc = "oos_eval_common"),
  h3     = list(file = "gamma_predictions_h3.rds",       pl = "pred_list_h3",  ac = "actuals_h3",  ev = "oos_eval_h3",  evc = "oos_eval_common_h3"),
  smooth = list(file = "gamma_predictions_smooth.rds",   pl = "pred_list_sm",  ac = "actuals_sm",  ev = "oos_eval_sm",  evc = NA_character_),
  stop("MT_TARGET must be one of: h1, h3, smooth")
)
cat(sprintf("Target: %s  (%s)\n", TARGET, cfg$file))

gamma_cols <- c("gamma_bm", "gamma_mom12m", "gamma_oper_prof",
                "gamma_asset_growth", "gamma_size", "gamma_beta")
gamma_labels <- c(gamma_bm = "Value", gamma_mom12m = "Momentum",
                  gamma_oper_prof = "Profitability", gamma_asset_growth = "Asset Growth",
                  gamma_size = "Size", gamma_beta = "Beta")

gp <- readRDS(cfg$file)
pred_list <- gp[[cfg$pl]]
actuals   <- gp[[cfg$ac]]

# FDR family: run on the COMMON-window evaluation when available (h1), so every
# (model x gamma) Clark-West p-value in the family is measured on an identical
# 414-month sample. The per-model `oos_eval` (414–421 months) would otherwise
# mix samples across the family. h3 has its own common-window object (04 Ext-C
# 5b, 414 months); smooth has none yet → falls back to per-model eval.
use_common <- !is.na(cfg$evc) && !is.null(gp[[cfg$evc]])
if (use_common) {
  oos_eval <- gp[[cfg$evc]]
  cat(sprintf("FDR family: common-window evaluation (%s, identical sample per gamma).\n", cfg$evc))
} else {
  oos_eval <- gp[[cfg$ev]]
  cat("FDR family: per-model evaluation (common-window object not available for this target).\n")
}

# ── PART 1: FDR on Clark-West p-values ────────────────────────────────────────
# Each row of oos_eval (sub_period "Full") is one CW test of a model vs the
# historical-mean benchmark. The benchmark itself is not a test (no self-CW).
cat("==============================================================\n")
cat("PART 1 — False Discovery Rate control on Clark-West tests\n")
cat("==============================================================\n")

fdr_table <- oos_eval |>
  filter(sub_period == "Full", model != BENCH, !is.na(p_cw)) |>
  mutate(
    p_bh_family = p.adjust(p_cw, method = "BH"),   # BH across ALL model x gamma tests
    p_by_family = p.adjust(p_cw, method = "BY")    # BY (robust to dependence)
  ) |>
  group_by(gamma) |>
  mutate(p_bh_within = p.adjust(p_cw, method = "BH")) |>   # BH within each gamma
  ungroup() |>
  mutate(
    sig_raw10    = p_cw        < FDR_LEVEL,
    sig_bh_fam   = p_bh_family < FDR_LEVEL,
    sig_by_fam   = p_by_family < FDR_LEVEL,
    sig_bh_within= p_bh_within < FDR_LEVEL
  ) |>
  arrange(gamma, p_cw)

n_tests <- nrow(fdr_table)
cat(sprintf("Family size: %d (model x gamma) Clark-West tests, FDR target %.0f%%\n\n",
            n_tests, 100 * FDR_LEVEL))
cat(sprintf("Discoveries (one-sided CW, p < %.2f):\n", FDR_LEVEL))
cat(sprintf("  Raw (uncorrected)            : %d / %d\n", sum(fdr_table$sig_raw10), n_tests))
cat(sprintf("  BH across whole family       : %d / %d\n", sum(fdr_table$sig_bh_fam), n_tests))
cat(sprintf("  BH within each gamma         : %d / %d\n", sum(fdr_table$sig_bh_within), n_tests))
cat(sprintf("  BY (dependence-robust) family: %d / %d\n\n", sum(fdr_table$sig_by_fam), n_tests))

cat("Models surviving BH-family FDR control (positive OOS R²):\n")
surv <- fdr_table |>
  filter(sig_bh_fam, oos_r2 > 0) |>
  transmute(gamma, model, oos_r2,
            p_cw = round(p_cw, 3), p_bh_family = round(p_bh_family, 3))
if (nrow(surv) == 0L) cat("  (none)\n") else print(surv, n = Inf)


# ── PART 1b: HAC bandwidth sensitivity of the Clark-West family ───────────────
# The p-values entering Part 1 are produced in 04_predict_gammas.R with a FIXED
# Newey-West truncation of NW_LAG (12 at H=1; 14 at H=3 / smooth, for the MA(2)
# overlap). That choice is conventional, not tested: the CW loss differential
# f_t carries autocorrelation up to |rho| ~ 0.35 within the first twelve lags,
# and for several gammas a majority of models still show significant
# autocorrelation BEYOND lag 12 — so an understated long-run variance would
# inflate t_cw and manufacture discoveries.
#
# This block recomputes the ENTIRE family at the base lag, base+6, base+12 and
# under Newey-West automatic bandwidth selection, and re-applies BH to each
# column. It is a pure recomputation from the stored predictions — no model is
# re-estimated — and it is deliberately self-checking: the base-lag column must
# reproduce the p_cw already in oos_eval (see the stopifnot below), so a
# mismatch flags that the two paths have drifted apart.
#
# Reference: Newey & West (1987, 1994); Andrews (1991) on bandwidth choice.
cat("\n==============================================================\n")
cat("PART 1b — HAC bandwidth sensitivity of the Clark-West p-values\n")
cat("==============================================================\n")

BASE_LAG <- if (TARGET == "h1") 12L else 14L   # mirrors NW_LAG / NW_LAG_H3 in 04
LAG_GRID <- c(BASE_LAG, BASE_LAG + 6L, BASE_LAG + 12L)

# Newey-West long-run variance of mean(f), fixed lag; `lag = NULL` requests the
# package's automatic bandwidth. Mirrors clark_west() in 04_predict_gammas.R,
# including the var(f)/n fallback and the 3-decimal rounding of p, so the base
# column is comparable to oos_eval$p_cw test-for-test.
cw_p_bw <- function(f, lag = NULL) {
  f <- f[!is.na(f)]; n <- length(f)
  if (n < 10L) return(NA_real_)
  v <- tryCatch(
    if (is.null(lag)) as.numeric(NeweyWest(lm(f ~ 1), prewhite = FALSE, adjust = TRUE))
    else              as.numeric(NeweyWest(lm(f ~ 1), lag = lag, prewhite = FALSE, adjust = TRUE)),
    error = function(e) var(f) / n)
  if (!is.finite(v) || v <= 0) v <- var(f) / n
  round(1 - pnorm(mean(f) / sqrt(v)), 3)
}

bw_rows <- list()
for (gc in gamma_cols) {
  P <- pred_list[[gc]]; y <- actuals[, gc]; bench <- P[, BENCH]
  # Same sample as the FDR family: the common window when Part 1 uses it
  # (rebuilt exactly as common_mask in 04 section 5b), per-model otherwise.
  keep <- if (use_common) (rowSums(is.na(P)) == 0 & !is.na(y)) else rep(TRUE, length(y))
  for (mm in setdiff(colnames(P), BENCH)) {
    f <- ((y - bench)^2 - ((y - P[, mm])^2 - (bench - P[, mm])^2))[keep]
    if (sum(!is.na(f)) < 10L) next
    bw_rows[[length(bw_rows) + 1L]] <- tibble(
      gamma = gamma_labels[gc], model = mm, n_oos = sum(!is.na(f)),
      !!!setNames(lapply(LAG_GRID, \(L) cw_p_bw(f, L)), paste0("p_nw", LAG_GRID)),
      p_nw_auto = cw_p_bw(f, lag = NULL))
  }
}
bw_table <- bind_rows(bw_rows) |>
  mutate(across(starts_with("p_nw"), \(p) p.adjust(p, "BH"), .names = "q_{.col}")) |>
  mutate(worst_q = do.call(pmax, across(starts_with("q_p_nw"))))

q_cols  <- grep("^q_p_nw", names(bw_table), value = TRUE)
bw_surv <- sapply(q_cols, \(q) sum(bw_table[[q]] < FDR_LEVEL, na.rm = TRUE))

# Self-check: the base-lag recomputation must match the stored p_cw family.
base_chk <- fdr_table |>
  select(gamma, model, p_cw) |>
  inner_join(bw_table |> select(gamma, model, p_base = !!paste0("p_nw", BASE_LAG)),
             by = c("gamma", "model"))
max_dev <- max(abs(base_chk$p_cw - base_chk$p_base), na.rm = TRUE)
cat(sprintf("Self-check vs stored p_cw: %d tests matched, max |deviation| = %.3f\n",
            nrow(base_chk), max_dev))
stopifnot(nrow(base_chk) == n_tests, max_dev <= 0.001)

cat(sprintf("\nBH-family survivors (q < %.2f) by Newey-West truncation:\n", FDR_LEVEL))
for (i in seq_along(q_cols))
  cat(sprintf("  %-12s : %d / %d\n",
              if (grepl("auto", q_cols[i])) "NW-auto" else sub("^q_p_nw", "lag ", q_cols[i]), bw_surv[i], nrow(bw_table)))
cat(sprintf("  %-12s : %d / %d  (survive at EVERY bandwidth)\n",
            "all four", sum(bw_table$worst_q < FDR_LEVEL, na.rm = TRUE), nrow(bw_table)))

cat("\nSurvivors per gamma:\n")
print(bw_table |> group_by(gamma) |>
        summarise(across(all_of(q_cols), \(x) sum(x < FDR_LEVEL, na.rm = TRUE)),
                  all_four = sum(worst_q < FDR_LEVEL, na.rm = TRUE)), n = Inf)

lapsed <- bw_table |>
  filter(.data[[paste0("q_p_nw", BASE_LAG)]] < FDR_LEVEL, worst_q >= FDR_LEVEL) |>
  select(gamma, model, all_of(q_cols)) |> arrange(gamma, model)
cat("\nDiscoveries that LAPSE at a wider bandwidth (base-lag survivors only):\n")
if (nrow(lapsed) == 0L) cat("  (none — the discovery set is bandwidth-invariant)\n") else print(lapsed, n = Inf)
cat("\nReading: a discovery that holds across all four columns is not an artefact of\n")
cat("the truncation-lag convention. Lapsing entries are marginal at the base lag and\n")
cat("should be described as bandwidth-sensitive in the text.\n")

# ── PART 2: Model Confidence Set per gamma ────────────────────────────────────
cat("\n==============================================================\n")
cat(sprintf("PART 2 — Model Confidence Set (Hansen-Lunde-Nason), %.0f%% conf.\n",
            100 * (1 - MCS_ALPHA)))
cat("==============================================================\n")
# FRAMING: the MCS is a LOW-POWER corroboration, not the primary inference.
# With T ≈ 414 and ~30 highly-correlated candidate models, the range statistic
# rarely eliminates anyone at 90% confidence — so a full-set MCS is expected and
# does NOT contradict the FDR discoveries in Part 1 (FDR is the primary control).
# To make the (low) power visible we (i) also report the stricter 75% set and
# (ii) export each model's MCS p-value = the largest confidence 1−α at which it
# survives. A model is in the (1−α) set iff its MCS p-value ≥ α, so the 75% set
# is exactly {MCS p-value ≥ 0.25}. The elimination ORDER (via the p-values) is
# informative even when the 90% set retains everything.

mcs_objects <- list()
mcs_rows    <- list()
mcs_pval_rows <- list()

for (gc in gamma_cols) {
  mdf    <- pred_list[[gc]]
  models <- colnames(mdf)                  # candidates incl. benchmark
  y      <- actuals[, gc]

  # Squared-error loss matrix: rows = OOS months, cols = models. Lower = better.
  L <- sapply(models, function(m) (y - mdf[, m])^2)
  colnames(L) <- models

  # MCS needs a balanced loss panel: keep months where ALL models predicted.
  ok <- stats::complete.cases(L)
  L_ok <- L[ok, , drop = FALSE]

  # Drop any zero-variance (degenerate) loss columns to avoid singular bootstrap.
  keep <- apply(L_ok, 2L, function(x) stats::sd(x) > 1e-12)
  L_ok <- L_ok[, keep, drop = FALSE]

  # Run at the liberal (75%) alpha so more models are eliminated and thus appear
  # in @show WITH their MCS p-values; the 90% set is the p-value≥0.10 subset.
  res <- tryCatch(
    MCS::MCSprocedure(Loss = L_ok, alpha = MCS_ALPHA2, B = MCS_B,
                      statistic = MCS_STAT, verbose = FALSE),
    error = function(e) { cat(sprintf("  [%s] MCS error: %s\n",
                                       gamma_labels[gc], conditionMessage(e))); NULL }
  )
  mcs_objects[[gc]] <- res

  # Extract per-model MCS p-values. The MCS package @show matrix carries columns
  # "Avg.Loss", "p-Value for H_{0,M_k}", "MCS p-Value" (rownames = models); the
  # membership-determining column is the "MCS p-Value" one. @show lists all
  # candidates with their p-value, so set membership at level (1-α) is simply
  # {MCS p-Value ≥ α}.
  if (!is.null(res)) {
    sh   <- tryCatch(as.matrix(res@show), error = function(e) NULL)
    pcol <- if (!is.null(sh)) grep("MCS", colnames(sh), value = TRUE)[1] else NA
  } else { sh <- NULL; pcol <- NA }

  if (!is.null(sh) && !is.na(pcol)) {
    pvals <- setNames(as.numeric(sh[, pcol]), rownames(sh))
    # Models NOT in @show were eliminated before the 75% cut → p-value < 0.25.
    # Assigning 0 is exact for the 75% set but CONSERVATIVE for the 90% set (a
    # model eliminated with true MCS-p in [0.10, 0.25) belongs in the 90% set).
    # In every run to date @show has contained ALL candidates (verified: zero
    # models hit this fallback), so the approximation has never been binding;
    # if it ever binds, re-run MCSprocedure at alpha = MCS_ALPHA instead.
    missing_models <- setdiff(colnames(L_ok), names(pvals))
    pvals <- c(pvals, setNames(rep(0, length(missing_models)), missing_models))
    in90  <- names(pvals)[pvals >= MCS_ALPHA]
    in75  <- names(pvals)[pvals >= MCS_ALPHA2]
    mcs_pval_rows[[gc]] <- tibble(
      gamma    = gamma_labels[gc],
      model    = names(pvals),
      mcs_pval = round(as.numeric(pvals), 3),
      in_mcs_90 = pvals >= MCS_ALPHA,
      in_mcs_75 = pvals >= MCS_ALPHA2
    ) |> arrange(desc(mcs_pval))
  } else {
    in90 <- in75 <- NA
  }

  bench_in90 <- BENCH %in% in90
  bench_in75 <- BENCH %in% in75

  cat(sprintf("\n%-13s | months used: %d/%d | in 90%% MCS: %d | in 75%% MCS: %d\n",
              gamma_labels[gc], sum(ok), length(y),
              if (all(is.na(in90))) 0L else length(in90),
              if (all(is.na(in75))) 0L else length(in75)))
  cat("  Benchmark in 90% / 75% MCS:", bench_in90, "/", bench_in75, "\n")
  if (!is.null(mcs_pval_rows[[gc]])) {
    top <- mcs_pval_rows[[gc]] |> filter(model != BENCH) |> slice_head(n = 3)
    cat("  Lowest-loss models (MCS p-value):",
        paste(sprintf("%s=%.2f", top$model, top$mcs_pval), collapse = ", "), "\n")
  }

  mcs_rows[[gc]] <- tibble(
    gamma        = gamma_labels[gc],
    months_used  = sum(ok),
    n_in_mcs_90  = if (all(is.na(in90))) 0L else length(in90),
    n_in_mcs_75  = if (all(is.na(in75))) 0L else length(in75),
    bench_in_90  = bench_in90,
    bench_in_75  = bench_in75,
    mcs_models_90 = if (all(is.na(in90))) NA_character_ else paste(in90, collapse = "; ")
  )
}

mcs_table      <- bind_rows(mcs_rows)
mcs_pval_table <- bind_rows(mcs_pval_rows)

cat("\n=== MCS summary (90% and 75% confidence) ===\n")
print(mcs_table |> dplyr::select(gamma, months_used, n_in_mcs_90, n_in_mcs_75,
                                 bench_in_90, bench_in_75), n = Inf)
cat("\nReading: FDR (Part 1) is the primary multiple-testing control; the MCS is a\n")
cat("low-power corroboration. Even where the 90% set retains all models, tightening\n")
cat("to 75% thins the set, and the per-model MCS p-values (mcs_pval CSV) rank models\n")
cat("by how close they are to elimination.\n")

# ── Save ──────────────────────────────────────────────────────────────────────
sfx <- if (TARGET == "h1") "" else paste0("_", TARGET)
saveRDS(list(fdr_table = fdr_table, bw_table = bw_table, mcs_table = mcs_table,
             mcs_pval_table = mcs_pval_table, mcs_objects = mcs_objects,
             target = TARGET),
        file = sprintf("multiple_testing_results%s.rds", sfx))
readr::write_csv(fdr_table, sprintf("multiple_testing_fdr%s.csv", sfx))
readr::write_csv(bw_table, sprintf("multiple_testing_bandwidth%s.csv", sfx))
readr::write_csv(mcs_table, sprintf("multiple_testing_mcs%s.csv", sfx))
readr::write_csv(mcs_pval_table, sprintf("multiple_testing_mcs_pvals%s.csv", sfx))
cat(sprintf("\nSaved: multiple_testing_results%s.rds, multiple_testing_fdr%s.csv,\n       multiple_testing_mcs%s.csv, multiple_testing_mcs_pvals%s.csv\n",
            sfx, sfx, sfx, sfx))
