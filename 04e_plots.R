# ==============================================================================
# 04d_plots.R
# Regenerate all OOS prediction plots from the final gamma_predictions.rds.
#
# Run this AFTER 04_predict_gammas.R, 04b_predict_mlp.R, and 04c_predict_lstm.R
# so that pred_list and oos_eval contain all models.
#
# Model groups:
#   Linear baselines : ar1, ar3, mv, var1
#   LASSO variants   : lasso, lasso_ct  (CT = Campbell-Thompson sign restriction)
#   Combination      : comb             (equal-weight AR1 + AR3 + LASSO)
#   ML models        : mlp, lstm        (added by 04b / 04c if run)
#
# Output: plots/oos_r2_heatmap.pdf
#         plots/oos_r2_barchart.pdf
#         plots/predicted_vs_actual.pdf
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))

list2env(readRDS("gamma_predictions.rds"), envir = environment())   # -> pred_list, oos_eval, actuals, dates_oos

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

dir.create("plots", showWarnings = FALSE)

# Helper: warn instead of abort when a PDF is locked by the viewer (Windows).
safe_save <- function(path, plot, ...) {
  tryCatch(
    { ggsave(path, plot, ...); cat("Saved:", path, "\n") },
    error = function(e) cat("SKIPPED (file locked — close viewer):", path, "\n")
  )
}

# ── Model order and labels ────────────────────────────────────────────────────
# CANON_ORDER / CANON_LABELS are the single label authority for every exhibit in
# this script and MUST stay in sync with FULL_ORDER / model_labels in
# 07_appendix_export.R and 06_tables_export.R. Any model present in oos_eval but
# absent from CANON_LABELS is silently coerced to NA by factor(), which collapses
# ALL such models into one "NA" column with their values overprinted on top of
# each other — the defect that shipped in the 2026-07-20 heatmap, where the four
# Ext-E rolling-window variants were missing from the map. lbl() therefore hard-
# fails instead of warning.
CANON_ORDER <- c(
  "factmom", "ar1", "ar3", "mv", "var1", "har",
  "lasso", "lasso_ct", "lasso_1se", "lasso_1se_ct",
  "lasso_w120", "lasso_w240",
  "rf", "rf_ct", "rf_w120", "rf_w240", "xgb", "xgb_ct",
  "mlp", "mlp_ct", "mlp_r1", "mlp_r1_ct", "mlp_r12", "mlp_r12_ct",
  "mlp_nogeo", "mlp_nogeo_ct", "mlp_sr", "mlp_sr_ct",
  "lstm", "lstm_r1", "lstm_r12", "lstm_nogeo",
  "comb", "comb_rf", "comb_xgb"
)
# Verbatim copy of model_labels in 07_appendix_export.R — keep the two in sync.
CANON_LABELS <- c(
  hist_mean = "Hist. mean", ar1 = "AR(1)", ar3 = "AR(3)", mv = "Multivariate OLS",
  var1 = "VAR(1)", factmom = "Ampl. hist. mean", har = "HAR",
  lasso = "LASSO", lasso_ct = "LASSO-CT", lasso_1se = "LASSO-1se",
  lasso_1se_ct = "LASSO-1se-CT",
  xgb = "XGBoost", xgb_ct = "XGBoost-CT", rf = "Random Forest", rf_ct = "RF-CT",
  mlp = "MLP", mlp_ct = "MLP-CT", mlp_nogeo = "MLP-NoMacro", mlp_nogeo_ct = "MLP-NoMacro-CT",
  mlp_sr = "MLP-SR", mlp_sr_ct = "MLP-SR-CT", lstm = "LSTM", lstm_nogeo = "LSTM-NoMacro",
  lstm_ct = "LSTM-CT",
  comb = "Combination", comb_rf = "Comb+RF", comb_xgb = "Comb+XGB",
  mlp_r1 = "MLP-1m", mlp_r12 = "MLP-12m", lstm_r1 = "LSTM-1m", lstm_r12 = "LSTM-12m",
  mlp_r1_ct = "MLP-1m-CT", mlp_r12_ct = "MLP-12m-CT",
  lasso_w120 = "LASSO-roll10y", lasso_w240 = "LASSO-roll20y",
  rf_w120 = "RF-roll10y", rf_w240 = "RF-roll20y"
)

# Axis-tick overrides. Table 3b prints the same two abbreviations for column
# width ("Random Forest" -> RF, "Combination" -> Comb.), so the heatmap axis and
# the table headers stay identical.
AXIS_SHORT <- c(rf = "RF", comb = "Comb.")

# Label lookup that refuses to produce NA levels.
lbl <- function(models, short = FALSE) {
  miss <- setdiff(models, names(CANON_LABELS))
  if (length(miss))
    stop("Models missing from CANON_LABELS (would become an NA column): ",
         paste(miss, collapse = ", "), call. = FALSE)
  out <- unname(CANON_LABELS[models])
  if (short) {
    hit <- models %in% names(AXIS_SHORT)
    out[hit] <- unname(AXIS_SHORT[models[hit]])
  }
  out
}
# Turn a model column into an ordered, labelled factor over `keep`.
as_model_factor <- function(x, keep, short = FALSE)
  factor(lbl(x, short), levels = lbl(keep, short))

# Main-text heatmap: exactly the union of Table 3 (linear) and Table 3b (ML and
# combination), because the caption states the figure condenses those two tables.
# The broken vintage instead showed 04's internal model list, which omitted the
# MLP and LSTM the surrounding text discusses.
HEATMAP_MODELS <- c("ar1", "ar3", "mv", "var1",       # Table 3  — linear
                    "lasso", "lasso_ct", "rf", "xgb", # Table 3b — ML / trees
                    "mlp", "lstm", "comb", "comb_xgb")

# Kept for the H=3 heatmap and the bar chart further down: the compact ML-vs-
# linear subset, filtered to what is actually present in the object being drawn.
FULL_MODEL_ORDER <- c("ar1", "ar3", "mv", "var1", "lasso", "lasso_ct",
                      "rf", "xgb", "comb", "mlp", "lstm")
model_order  <- intersect(FULL_MODEL_ORDER, unique(oos_eval$model))
model_labels <- CANON_LABELS

cat("Models in oos_eval:  ", paste(unique(oos_eval$model), collapse = ", "), "\n")
cat("Models in heatmap:   ", paste(intersect(HEATMAP_MODELS, unique(oos_eval$model)),
                                   collapse = ", "), "\n")

# ── OOS R² heatmap (main text) ────────────────────────────────────────────────
# Scored on oos_eval_common — the identical 414-month sample used by Tables 3/3b
# and by 04f — so the grid and the tables report the same numbers. The broken
# vintage used the per-model oos_eval (414/415/421 months), contradicting the
# figure caption's "common 414-month evaluation sample".
oos_eval_fig <- if (exists("oos_eval_common")) oos_eval_common else oos_eval
n_common_fig <- if (exists("oos_eval_common")) unique(oos_eval_common$n_oos)[1] else NA_integer_
heatmap_models <- intersect(HEATMAP_MODELS, unique(oos_eval_fig$model))

p_r2 <- oos_eval_fig |>
  filter(model %in% heatmap_models) |>
  mutate(
    model = as_model_factor(model, heatmap_models, short = TRUE),
    sig   = case_when(p_cw < 0.01 ~ "***", p_cw < 0.05 ~ "**",
                      p_cw < 0.10 ~ "*",   TRUE ~ "")
  ) |>
  ggplot(aes(x = model, y = gamma, fill = oos_r2)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = paste0(round(oos_r2, 1), sig)), size = 3) +
  scale_fill_gradient2(
    low = "firebrick", mid = "white", high = "steelblue", midpoint = 0,
    name = "OOS R² (%)"
  ) +
  labs(
    title    = "Out-of-Sample R² by Model and Gamma",
    subtitle = paste0(
      "Clark-West significance: *** p<0.01, ** p<0.05, * p<0.10",
      if (!is.na(n_common_fig))
        sprintf("; common %d-month evaluation sample", n_common_fig) else "",
      "\nLASSO-CT = LASSO + Campbell-Thompson (2008) sign restriction; ",
      "Comb. = equal-weight AR(1)/AR(3)/LASSO; Comb+XGB adds XGBoost"
    ),
    x = NULL, y = NULL
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

safe_save("plots/oos_r2_heatmap.pdf", p_r2, width = 11, height = 5)

# ── Predicted vs actual — AR(1) and LASSO for every gamma ─────────────────────
# Fixed choice of AR(1) and LASSO rather than "best model per gamma" so that
# the comparison is consistent across characteristics.
pred_actual_plot <- map_dfr(gamma_cols, function(gc) {
  bind_rows(
    tibble(date = dates_oos, actual = actuals[, gc],
           predicted = pred_list[[gc]][, "ar1"],
           gamma = gamma_labels[gc], model = "AR(1)"),
    tibble(date = dates_oos, actual = actuals[, gc],
           predicted = pred_list[[gc]][, "lasso"],
           gamma = gamma_labels[gc], model = "LASSO")
  )
})

p_pa <- pred_actual_plot |>
  pivot_longer(c(actual, predicted), names_to = "series", values_to = "value") |>
  ggplot(aes(x = date, y = value * 100, colour = series, linewidth = series)) +
  geom_line() +
  scale_colour_manual(values = c(actual = "black", predicted = "steelblue")) +
  scale_linewidth_manual(values = c(actual = 0.5, predicted = 0.7)) +
  facet_wrap(~gamma + model, ncol = 2, scales = "free_y") +
  labs(title = "Actual vs Predicted Gammas — AR(1) and LASSO",
       x = NULL, y = "Gamma (% per month)", colour = NULL, linewidth = NULL) +
  theme_bw() +
  theme(legend.position = "bottom")

safe_save("plots/predicted_vs_actual.pdf", p_pa, width = 10, height = 14)

# ── OOS R² bar chart — all models side by side per gamma ──────────────────────
p_bar <- oos_eval |>
  filter(model %in% model_order) |>
  mutate(model = factor(model_labels[model], levels = model_labels[model_order])) |>
  ggplot(aes(x = model, y = oos_r2, fill = model)) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = 0, linewidth = 0.4, linetype = "dashed") +
  # NOTE: no brewer palette here — Set2 caps at 8 colours, and with 11 models
  # the overflow levels (Comb/MLP/LSTM) were silently rendered with NA fill.
  scale_fill_hue(guide = "none") +
  facet_wrap(~gamma, ncol = 2, scales = "free_y") +
  labs(title = "OOS R² by Model and Characteristic",
       x = NULL, y = "OOS R² (%)") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))

safe_save("plots/oos_r2_barchart.pdf", p_bar, width = 10, height = 10)

# ── Sub-period OOS R² bar chart ───────────────────────────────────────────────
# Same exhibit as the one 04 draws natively, but regenerated from the persisted
# oos_eval_all so it picks up the Colab-imported NN models' sub-period rows
# (added by import_lstm_from_colab.R) without re-running the 04 OOS loop.
if (exists("oos_eval_all") && any(oos_eval_all$sub_period != "Full")) {
  # Appendix figure: ALL model variants, matching its caption and Table B.4.
  sub_models <- intersect(CANON_ORDER, unique(oos_eval_all$model))
  p_sub <- oos_eval_all |>
    filter(sub_period != "Full", model %in% sub_models) |>
    mutate(
      model      = as_model_factor(model, sub_models),
      sub_period = factor(sub_period, levels = c("Pre-2000", "2000-2010", "Post-2010"))
    ) |>
    ggplot(aes(x = sub_period, y = oos_r2, fill = model)) +
    geom_col(position = "dodge", width = 0.7) +
    geom_hline(yintercept = 0, linewidth = 0.3) +
    scale_fill_hue() +
    facet_wrap(~gamma, ncol = 3, scales = "free_y") +
    labs(title = "Sub-Period OOS R² by Model and Gamma",
         x = NULL, y = "OOS R² (%)", fill = NULL) +
    theme_bw() +
    theme(legend.position = "bottom",
          axis.text.x = element_text(angle = 20, hjust = 1))
  safe_save("plots/oos_r2_subperiod.pdf", p_sub, width = 14, height = 8)
}


# ══════════════════════════════════════════════════════════════════════════════
# Thesis exhibits added 2026-07-19 (reviewer suggestions): SED plot, forecast
# with across-seed band, monthly-vs-quarterly R² panel. Colours: Tol bright
# trio (#4477AA/#EE6677/#228833) — CVD-safe; linetype as secondary encoding.
# ══════════════════════════════════════════════════════════════════════════════

# Joint common months (all models × all gammas non-NA + complete actuals) —
# the same 414-month discipline as oos_eval_common / 05 / RAFE.
common_all <- Reduce(`&`, lapply(gamma_cols, function(gc)
  rowSums(is.na(pred_list[[gc]])) == 0)) & complete.cases(actuals)

# ── Cumulative squared-error difference (SED), Size ───────────────────────────
# Goyal-Welch (2008)-style: cum Σ[(γ−bench)² − (γ−model)²]; upward drift =
# model beating the expanding historical mean. Shows WHEN size predictability
# is earned. Gammas in % per month → SED in %-squared.
sed_models <- c(lasso = "LASSO", rf = "Random Forest", mlp = "MLP")
sed_df <- purrr::map_dfr(names(sed_models), function(m) {
  y  <- actuals[common_all, "gamma_size"] * 100
  b  <- pred_list[["gamma_size"]][common_all, "hist_mean"] * 100
  yh <- pred_list[["gamma_size"]][common_all, m] * 100
  tibble(date = dates_oos[common_all], model = sed_models[m],
         sed  = cumsum((y - b)^2 - (y - yh)^2))
})
sed_lab <- sed_df |> group_by(model) |> slice_max(date, n = 1) |> ungroup() |>
  mutate(vjust = case_when(model == "MLP" ~ -0.6,        # dodge the two lines
                           model == "Random Forest" ~ 1.4,  # ending near each other
                           TRUE ~ 0.35))
p_sed <- ggplot(sed_df, aes(date, sed, colour = model, linetype = model)) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  geom_vline(xintercept = as.Date(c("2000-01-01", "2010-01-01")),
             linetype = "dotted", colour = "grey55", linewidth = 0.35) +
  geom_line(linewidth = 0.7) +
  geom_text(data = sed_lab, aes(label = model, vjust = vjust),
            hjust = -0.05, size = 3, show.legend = FALSE) +
  scale_colour_manual(values = c("LASSO" = "#4477AA", "Random Forest" = "#EE6677",
                                 "MLP" = "#228833"), guide = "none") +
  scale_linetype_manual(values = c("LASSO" = "solid", "Random Forest" = "longdash",
                                   "MLP" = "dotdash"), guide = "none") +
  scale_x_date(expand = expansion(mult = c(0.01, 0.14)),
               breaks = as.Date(paste0(seq(1990, 2020, 10), "-01-01")),
               date_labels = "%Y") +
  labs(x = NULL, y = expression("Cumulative SED (%"^2*")")) +
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank())
safe_save("plots/sed_size.pdf", p_sed, width = 8, height = 3.6)

# ── Size gamma: realised vs MLP-ensemble forecast with across-seed band ───────
# Band = min–max across the five seed forecasts (from the Colab bridge CSV);
# makes the promised seed-dispersion transparency visible and shows what an
# 8-9% OOS R² looks like against the realised series.
mlp_csv <- "colab/colab_io/mlp_preds.csv"
if (file.exists(mlp_csv)) {
  mp <- suppressMessages(readr::read_csv(mlp_csv, show_col_types = FALSE))
  seed_cols <- grep("^mlp_seed[0-9]+__gamma_size$", names(mp), value = TRUE)
  if (length(seed_cols) >= 2 && nrow(mp) == length(dates_oos)) {
    S <- as.matrix(mp[, seed_cols]) * 100
    fc <- tibble(
      date  = dates_oos,
      real  = actuals[, "gamma_size"] * 100,
      mlp   = pred_list[["gamma_size"]][, "mlp"] * 100,
      bench = pred_list[["gamma_size"]][, "hist_mean"] * 100,
      lo    = apply(S, 1, min), hi = apply(S, 1, max)
    ) |> filter(common_all)
    p_fc <- ggplot(fc, aes(date)) +
      geom_hline(yintercept = 0, linewidth = 0.3) +
      geom_line(aes(y = real), colour = "grey65", linewidth = 0.35) +
      geom_ribbon(aes(ymin = lo, ymax = hi), fill = "#4477AA", alpha = 0.25) +
      geom_line(aes(y = bench), colour = "grey30", linetype = "dashed",
                linewidth = 0.45) +
      geom_line(aes(y = mlp), colour = "#4477AA", linewidth = 0.6) +
      labs(x = NULL, y = "Size gamma (% per month)") +
      theme_bw(base_size = 10) +
      theme(panel.grid.minor = element_blank())
    safe_save("plots/size_forecast_seedband.pdf", p_fc, width = 8, height = 3.6)
  } else cat("Skipped seed-band plot: seed columns or row alignment missing.\n")
} else cat("Skipped seed-band plot: colab/colab_io/mlp_preds.csv not found.\n")

# ── Monthly vs quarterly OOS R², LASSO family ─────────────────────────────────
# The H=3 results ("where the economic content is concentrated") get a visual:
# two horizon bars per gamma, faceted by model. Fill = two steps of one hue
# (ordered horizon); H=1 from the common-window table, H=3 from the H=3 run.
if (file.exists("gamma_predictions_h3.rds") && exists("oos_eval_common")) {
  ev3 <- readRDS("gamma_predictions_h3.rds")$oos_eval_h3
  h13 <- bind_rows(
    oos_eval_common |> filter(model %in% c("lasso", "lasso_1se")) |>
      transmute(gamma, model, oos_r2, Horizon = "Monthly (H=1)"),
    ev3 |> filter(sub_period == "Full", model %in% c("lasso", "lasso_1se")) |>
      transmute(gamma, model, oos_r2, Horizon = "Quarterly (H=3)")
  ) |>
    mutate(gamma   = factor(gamma, levels = c("Value", "Momentum", "Profitability",
                                              "Asset Growth", "Size", "Beta")),
           model   = recode(model, lasso = "LASSO", lasso_1se = "LASSO-1se"),
           Horizon = factor(Horizon, levels = c("Monthly (H=1)", "Quarterly (H=3)")))
  p_h13 <- ggplot(h13, aes(gamma, oos_r2, fill = Horizon)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.62) +
    geom_hline(yintercept = 0, linewidth = 0.3) +
    scale_fill_manual(values = c("Monthly (H=1)" = "#A3C1E0",
                                 "Quarterly (H=3)" = "#34618F")) +
    facet_wrap(~model) +
    labs(x = NULL, y = expression("OOS R"^2*" (%)"), fill = NULL) +
    theme_bw(base_size = 10) +
    theme(legend.position = "top", panel.grid.minor = element_blank(),
          panel.grid.major.x = element_blank(),
          axis.text.x = element_text(angle = 25, hjust = 1))
  safe_save("plots/oos_r2_h1_vs_h3.pdf", p_h13, width = 8, height = 3.8)
}

# ── Rolling 60-month OOS R², Size and Beta ────────────────────────────────────
# Real-time complement to the SED plot: R² over a trailing 60-month window,
# same models/colours as the SED exhibit. Shows predictability switching off
# after 2010 (Size) and the nets' beta signal being episodic.
ROLL_W <- 60L
roll_df <- purrr::map_dfr(c(gamma_size = "Size", gamma_beta = "Beta") |> names(),
  function(gc) {
  glab <- c(gamma_size = "Size", gamma_beta = "Beta")[gc]
  idx  <- which(common_all)
  y    <- actuals[idx, gc] * 100
  b    <- pred_list[[gc]][idx, "hist_mean"] * 100
  dts  <- dates_oos[idx]
  purrr::map_dfr(names(sed_models), function(m) {
    yh <- pred_list[[gc]][idx, m] * 100
    r2 <- sapply(seq_along(y), function(i) {
      if (i < ROLL_W) return(NA_real_)
      w <- (i - ROLL_W + 1L):i
      100 * (1 - sum((y[w] - yh[w])^2) / sum((y[w] - b[w])^2))
    })
    tibble(date = dts, gamma = glab, model = sed_models[m], r2 = r2)
  })
}) |>
  filter(!is.na(r2)) |>
  mutate(gamma = factor(gamma, levels = c("Size", "Beta")))
p_roll <- ggplot(roll_df, aes(date, r2, colour = model, linetype = model)) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  geom_line(linewidth = 0.55) +
  scale_colour_manual(values = c("LASSO" = "#4477AA", "Random Forest" = "#EE6677",
                                 "MLP" = "#228833")) +
  scale_linetype_manual(values = c("LASSO" = "solid", "Random Forest" = "longdash",
                                   "MLP" = "dotdash")) +
  facet_wrap(~gamma, ncol = 1, scales = "free_y") +
  labs(x = NULL, y = expression("Rolling 60-month OOS R"^2*" (%)"),
       colour = NULL, linetype = NULL) +
  theme_bw(base_size = 10) +
  theme(legend.position = "top", panel.grid.minor = element_blank())
safe_save("plots/rolling_oos_r2.pdf", p_roll, width = 8, height = 5)

# ── H=3 OOS R² heatmap (diagnostic companion to the H=1 heatmap) ──────────────
if (file.exists("gamma_predictions_h3.rds")) {
  ev3_full <- readRDS("gamma_predictions_h3.rds")$oos_eval_h3 |>
    filter(sub_period == "Full")
  h3_models <- intersect(FULL_MODEL_ORDER, unique(ev3_full$model))
  p_r2_h3 <- ev3_full |>
    filter(model %in% h3_models) |>
    mutate(
      model = factor(model_labels[model], levels = model_labels[h3_models]),
      sig   = case_when(p_cw < 0.01 ~ "***", p_cw < 0.05 ~ "**",
                        p_cw < 0.10 ~ "*",   TRUE ~ "")
    ) |>
    ggplot(aes(x = model, y = gamma, fill = oos_r2)) +
    geom_tile(colour = "white", linewidth = 0.5) +
    geom_text(aes(label = paste0(round(oos_r2, 1), sig)), size = 3) +
    scale_fill_gradient2(low = "firebrick", mid = "white", high = "steelblue",
                         midpoint = 0, name = "OOS R² (%)") +
    labs(title = "Out-of-Sample R² by Model and Gamma — Quarterly Horizon (H=3)",
         subtitle = "Clark-West significance (NW-14): *** p<0.01, ** p<0.05, * p<0.10",
         x = NULL, y = NULL) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  safe_save("plots/oos_r2_heatmap_h3.pdf", p_r2_h3, width = 10, height = 5)
}
