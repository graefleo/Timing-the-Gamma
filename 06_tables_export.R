# ==============================================================================
# 06_tables_export.R
# Exports all thesis tables as publication-ready LaTeX (.tex) files.
# Each file can be included in the thesis with \input{tables/tableX.tex}.
#
# Input:  gammas_ts.rds, gamma_predictions.rds, portfolio_results.rds
# Output: tables/ directory with one .tex file per table
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(kableExtra)

list2env(readRDS("gammas_ts.rds"), envir = environment())           # -> gammas_ts, fm_table, ...
list2env(readRDS("gamma_predictions.rds"), envir = environment())   # -> pred_list, oos_eval, actuals, dates_oos
list2env(readRDS("portfolio_results.rds"), envir = environment())   # -> port_ret, port_eval, grs_results, ...

dir.create("tables", showWarnings = FALSE)

# Helpers ----------------------------------------------------------------------

# Add significance stars to a numeric t-statistic
stars <- function(t, df = Inf) {
  p <- 2 * pt(-abs(t), df = df)
  case_when(p < 0.01 ~ "***", p < 0.05 ~ "**", p < 0.10 ~ "*", TRUE ~ "")
}

# Format number + stars into a single string
fmt_stars <- function(x, digits = 2, t_stat = NULL) {
  s <- if (!is.null(t_stat)) stars(t_stat) else ""
  paste0(formatC(x, digits = digits, format = "f"), s)
}

# Save a kable object to a .tex file
save_tex <- function(tbl, filename, caption, label) {
  tex <- tbl |>
    kable_styling(
      latex_options = c("HOLD_position", "striped"),
      stripe_color  = "gray!10",
      font_size     = 10
    ) |>
    footnote(
      general           = "*** p<0.01, ** p<0.05, * p<0.10.",
      general_title     = "",
      footnote_as_chunk = TRUE,
      escape            = FALSE
    )
  writeLines(as.character(tex), file.path("tables", filename))
  cat(sprintf("Saved: tables/%s\n", filename))
}

model_order  <- c("ar1", "ar3", "mv", "var1", "lasso")
model_labels <- c(
  # linear / benchmark
  hist_mean = "Hist. mean", ar1 = "AR(1)", ar3 = "AR(3)", mv = "Multivariate OLS",
  var1 = "VAR(1)", factmom = "Ampl. hist. mean", har = "HAR",
  # penalised linear
  lasso = "LASSO", lasso_ct = "LASSO-CT", lasso_1se = "LASSO-1se",
  lasso_1se_ct = "LASSO-1se-CT",
  # trees
  xgb = "XGBoost", xgb_ct = "XGBoost-CT", rf = "Random Forest", rf_ct = "RF-CT",
  # neural nets
  mlp = "MLP", mlp_ct = "MLP-CT", mlp_nogeo = "MLP-NoMacro", mlp_nogeo_ct = "MLP-NoMacro-CT",
  mlp_sr = "MLP-SR", mlp_sr_ct = "MLP-SR-CT", lstm = "LSTM", lstm_nogeo = "LSTM-NoMacro",
  # combinations
  comb = "Combination", comb_rf = "Comb+RF", comb_xgb = "Comb+XGB",
  # retraining-frequency variants (imported from Colab; can win best-model selections)
  mlp_r1 = "MLP-1m", mlp_r12 = "MLP-12m",
  lstm_r1 = "LSTM-1m", lstm_r12 = "LSTM-12m",
  # Ext-E rolling-window variants
  lasso_w120 = "LASSO-roll10y", lasso_w240 = "LASSO-roll20y",
  rf_w120 = "RF-roll10y", rf_w240 = "RF-roll20y"
)

char_order  <- c("Value", "Momentum", "Profitability", "Asset Growth", "Size", "Beta")


# ==============================================================================
# Table 1 — Fama-MacBeth Risk Premia
# ==============================================================================
fm_gamma_labels <- c(
  bm = "Value", mom12m = "Momentum", oper_prof = "Profitability",
  asset_growth = "Asset Growth", size = "Size", beta = "Beta"
)
t1 <- fm_table |>
  filter(gamma != "intercept") |>
  mutate(
    gamma      = factor(fm_gamma_labels[gamma], levels = char_order),
    mean_fmt   = fmt_stars(mean * 100, 2, t_stat = t_nw),
    sd_fmt     = formatC(sd * 100,       digits = 2, format = "f"),
    t_fm_fmt   = formatC(t_fm,           digits = 2, format = "f"),
    t_nw_fmt   = formatC(t_nw,           digits = 2, format = "f"),
    t_sh_fmt   = formatC(
      if ("t_nw_shanken" %in% names(fm_table)) t_nw_shanken else NA_real_,
      digits = 2, format = "f"
    )
  ) |>
  arrange(gamma) |>
  select(gamma, mean_fmt, sd_fmt, t_fm_fmt, t_nw_fmt, t_sh_fmt) |>
  rename(
    Characteristic              = gamma,
    `Mean$^{a}$`               = mean_fmt,
    `SD$^{a}$`                 = sd_fmt,
    `$t$ (FM)`                 = t_fm_fmt,
    `$t$ (NW-12)`              = t_nw_fmt,
    `$t$ (NW-12, Shanken)`     = t_sh_fmt
  )

kbl_t1 <- kbl(t1,
  format   = "latex",
  booktabs = TRUE,
  escape   = FALSE,
  caption  = "Fama--MacBeth Risk Premia (WLS primary specification). Monthly
              value-weighted cross-sectional regressions of excess returns on
              rank-standardised characteristics. Columns report the time-series
              mean and SD of monthly gamma estimates (in \\% per month), the
              standard FM $t$-statistic, the Newey--West (12 lags) $t$-statistic,
              and the Shanken (1992) EIV-corrected NW $t$-statistic.",
  # Short form for the List of Tables (the full caption is far too long there).
  caption.short = "Fama--MacBeth risk premia (WLS specification)",
  label    = "res-fm"   # matches the \ref{tab:res-fm} references in thesis_drafts
) |>
  footnote(
    alphabet          = c("In \\\\% per month."),
    general           = paste0(
      "*** p<0.01, ** p<0.05, * p<0.10 based on NW-12 $t$-statistic. ",
      "Shanken correction: $c = \\\\bar{r}_{mkt}^2 / \\\\mathrm{Var}(r_{mkt})$, the ",
      "squared monthly Sharpe ratio of the market ($c = 0.018$); the sampling ",
      "variance inflates by $(1+c)$, so corrected $t = t_{NW} / \\\\sqrt{1+c}$."
    ),
    general_title     = "",
    footnote_as_chunk = TRUE,
    threeparttable    = TRUE,
    escape            = FALSE
  ) |>
  kable_styling(latex_options = c("HOLD_position"), font_size = 10)

writeLines(as.character(kbl_t1), "tables/t1_fm_risk_premia.tex")
cat("Saved: tables/t1_fm_risk_premia.tex\n")


# ==============================================================================
# Table 2 — Gamma Descriptive Statistics
# ==============================================================================
# Rebuild from gammas_ts directly
library(lmtest); library(sandwich)

nw_t <- function(x, lag = 12) {
  x   <- x[!is.na(x)]
  fit <- lm(x ~ 1)
  ct  <- lmtest::coeftest(
    fit, vcov = sandwich::NeweyWest(fit, lag = lag, prewhite = FALSE)
  )
  ct["(Intercept)", "t value"]
}

ar1_coef <- function(x) {
  x <- x[!is.na(x)]
  coef(lm(x[-1] ~ x[-length(x)]))[2]
}

gamma_cols_named <- c(
  gamma_bm = "Value", gamma_mom12m = "Momentum", gamma_oper_prof = "Profitability",
  gamma_asset_growth = "Asset Growth", gamma_size = "Size", gamma_beta = "Beta"
)

t2 <- map_dfr(names(gamma_cols_named), function(gc) {
  x   <- gammas_ts[[gc]]
  t_s <- nw_t(x)
  tibble(
    Characteristic = gamma_cols_named[gc],
    Mean           = fmt_stars(mean(x, na.rm = TRUE) * 100, 3, t_stat = t_s),
    SD             = formatC(sd(x,   na.rm = TRUE) * 100, digits = 3, format = "f"),
    Min            = formatC(min(x,  na.rm = TRUE) * 100, digits = 3, format = "f"),
    Max            = formatC(max(x,  na.rm = TRUE) * 100, digits = 3, format = "f"),
    `AR(1)`        = formatC(ar1_coef(x),                 digits = 3, format = "f"),
    `$t$ (NW-12)`  = formatC(t_s,                         digits = 2, format = "f")
  )
}) |>
  mutate(Characteristic = factor(Characteristic, levels = char_order)) |>
  arrange(Characteristic)

kbl_t2 <- kbl(t2,
  format   = "latex",
  booktabs = TRUE,
  escape   = FALSE,
  caption  = "Descriptive statistics of Fama--MacBeth gamma time series.
              Mean and SD are in \\% per month. AR(1) is the first-order
              autocorrelation coefficient. $t$ (NW-12) is the Newey--West
              $t$-statistic for the null hypothesis that the mean equals zero.",
  caption.short = "Descriptive statistics of the gamma time series",
  label    = "gamma_desc"
) |>
  footnote(
    general           = "*** p<0.01, ** p<0.05, * p<0.10.",
    general_title     = "",
    footnote_as_chunk = TRUE,
    escape            = FALSE
  ) |>
  kable_styling(latex_options = c("HOLD_position"), font_size = 10)

writeLines(as.character(kbl_t2), "tables/t2_gamma_descriptives.tex")
cat("Saved: tables/t2_gamma_descriptives.tex\n")


# ==============================================================================
# Table 3 — Out-of-Sample R² (%) by Model and Characteristic
# ==============================================================================
# Evaluation source: the COMMON-window table (oos_eval_common) when present, so
# every model in Tables 3/3b is scored on the identical 414-month sample. Falls
# back to per-model oos_eval if the common-window object is absent (older rds).
oos_eval_tab <- if (exists("oos_eval_common")) oos_eval_common else oos_eval
n_common <- if (exists("oos_eval_common"))
  unique(oos_eval_common$n_oos)[1] else NA_integer_

# Shared caption clause explaining why stars can attach to negative OOS R².
# Clark-West adds back the benchmark's estimation-noise penalty, so a model can
# be significantly informative (stars) even when its unadjusted OOS R² is < 0.
cw_neg_note <- "The Clark--West statistic adjusts for the estimation noise the
    larger model introduces under the null; a model can therefore be
    significantly informative (stars) while its unadjusted out-of-sample $R^2$
    is negative."

# Cell formatter: OOS R² with Clark-West stars.
r2_cell <- function(oos_r2, p_cw) {
  paste0(formatC(oos_r2, digits = 2, format = "f"),
         ifelse(p_cw < 0.01, "***",
         ifelse(p_cw < 0.05, "**",
         ifelse(p_cw < 0.10, "*", ""))))
}

available_models <- intersect(model_order, unique(oos_eval_tab$model))

t3 <- oos_eval_tab |>
  filter(gamma %in% char_order, model %in% available_models) |>
  mutate(
    gamma = factor(gamma, levels = char_order),
    model = factor(model, levels = available_models, labels = model_labels[available_models]),
    cell  = r2_cell(oos_r2, p_cw)
  ) |>
  select(gamma, model, cell) |>
  pivot_wider(names_from = model, values_from = cell) |>
  arrange(gamma) |>
  rename(Characteristic = gamma)

kbl_t3 <- kbl(t3,
  format   = "latex",
  booktabs = TRUE,
  escape   = FALSE,
  caption  = sprintf(
    "Out-of-sample $R^2$ (\\%%) of the linear gamma-prediction models relative to
     the historical-mean benchmark. Expanding estimation window with a minimum of
     60 months%s. Stars denote significance of the one-sided Clark--West (2007) test
     of equal predictive accuracy. %s",
    if (!is.na(n_common)) sprintf("; all models evaluated on the common %d-month sample", n_common) else "",
    cw_neg_note),
  caption.short = "Out-of-sample $R^2$: linear models",
  label    = "oos_r2"
) |>
  footnote(
    general           = "*** p<0.01, ** p<0.05, * p<0.10 (Clark--West test, one-sided).",
    general_title     = "",
    footnote_as_chunk = TRUE,
    escape            = FALSE
  ) |>
  kable_styling(latex_options = c("HOLD_position"), font_size = 10)

writeLines(as.character(kbl_t3), "tables/t3_oos_r2.tex")
cat("Saved: tables/t3_oos_r2.tex\n")


# ==============================================================================
# Table 3b — Out-of-Sample R² (%): ML, tree and combination models
# ==============================================================================
# The headline predictability evidence (RF/MLP/LSTM/XGB and their combinations)
# lives here, alongside penalised-linear LASSO for reference. Same common-window
# sample and Clark-West stars as Table 3.
ml_model_order <- c("lasso", "lasso_ct", "rf", "xgb", "mlp", "lstm", "comb", "comb_xgb")
ml_available   <- intersect(ml_model_order, unique(oos_eval_tab$model))

t3b <- oos_eval_tab |>
  filter(gamma %in% char_order, model %in% ml_available) |>
  mutate(
    gamma = factor(gamma, levels = char_order),
    model = factor(model, levels = ml_available, labels = model_labels[ml_available]),
    cell  = r2_cell(oos_r2, p_cw)
  ) |>
  select(gamma, model, cell) |>
  pivot_wider(names_from = model, values_from = cell) |>
  arrange(gamma) |>
  rename(Characteristic = gamma)

# Compact headers for THIS table only. With nine columns, "Random Forest"
# (60.8pt) and "Combination" (52.0pt) are the two columns whose *header* rather
# than whose data sets the column width -- their widest cell is only 32.4pt --
# and together they pushed the table about 57pt past the right edge of the text
# block. RF / Comb. are already the house abbreviations in model_labels above
# (rf_ct = "RF-CT", comb_rf = "Comb+RF"), so this stays consistent; both are
# spelled out in the footnote. Base-R renaming, so the table still builds if
# either model is missing from ml_available.
names(t3b)[names(t3b) == "Random Forest"] <- "RF"
names(t3b)[names(t3b) == "Combination"]   <- "Comb."

kbl_t3b <- kbl(t3b,
  format   = "latex",
  booktabs = TRUE,
  escape   = FALSE,
  caption  = sprintf(
    "Out-of-sample $R^2$ (\\%%) of the penalised-linear, tree-based, neural-network
     and forecast-combination models relative to the historical-mean benchmark%s.
     RF denotes the random forest and Comb. the equal-weight combination of AR(1), AR(3) and LASSO.
     Stars denote significance of the one-sided Clark--West (2007) test. %s",
    if (!is.na(n_common)) sprintf(" (common %d-month sample)", n_common) else "",
    cw_neg_note),
  caption.short = "Out-of-sample $R^2$: machine-learning and combination models",
  label    = "oos_r2_ml"
) |>
  footnote(
    # NB keep this to ONE short line: kableExtra emits it as \multicolumn{9}{l}{},
    # which cannot wrap, so a long note runs straight past the right margin.
    # Anything longer belongs in the caption above.
    general           = "*** p<0.01, ** p<0.05, * p<0.10 (Clark--West test, one-sided).",
    general_title     = "",
    footnote_as_chunk = TRUE,
    escape            = FALSE
  ) |>
  kable_styling(latex_options = c("HOLD_position"), font_size = 10)

writeLines(as.character(kbl_t3b), "tables/t3b_oos_r2_ml.tex")
cat("Saved: tables/t3b_oos_r2_ml.tex\n")


# ==============================================================================
# Table 4 — Portfolio Performance: Static vs Best Timed Model
# ==============================================================================
# Headline timed model is selected by RAFE (lowest), NOT by OOS R². RAFE is the
# economically-aligned forecast-error metric (Mahalanobis, Σ⁻¹-weighted); selecting on
# it rather than R² operationalises the "Lost in Translation" finding that statistical
# fit and economic value diverge — and it is non-circular w.r.t. the reported Sharpe
# (RAFE is a forecast metric, not the portfolio return). Benchmarks (hist_mean) excluded.
# References: Salcher, Stöckl & Hanke (Lost in Translation); Gu, Kelly & Xiu (2020, RFS).
best_model <- rafe_results |>
  filter(model != "hist_mean") |>
  slice_min(RAFE, n = 1) |>
  pull(model)

# For transparency, also note the OOS-R²-best model (the criterion we deliberately do NOT use)
r2_best_model <- oos_eval |>
  filter(gamma %in% char_order) |>
  group_by(model) |>
  summarise(avg_r2 = mean(oos_r2, na.rm = TRUE), .groups = "drop") |>
  slice_max(avg_r2, n = 1) |>
  pull(model)

cat(sprintf("\nHeadline timed model (lowest RAFE): %s  [OOS-R²-best would be: %s]\n",
            best_model, r2_best_model))

# Side-by-side: static (hist_mean) | direction (best_model)
# Direction timing is the primary economic-value strategy (sign-only, leverage-free).
# The magnitude-scaled strategy is reported separately as a robustness exhibit because
# its 1/|gamma_bar| normalisation produces extreme, leverage-driven volatility for
# near-zero-premium factors (esp. Beta) rather than timing skill.
char_labels <- c(
  bm_std = "Value", mom12m_std = "Momentum", oper_prof_std = "Profitability",
  asset_growth_std = "Asset Growth", log_mktcap_lag_std = "Size",
  beta_std = "Beta"
)
# Both panels from best_model's rows so Static and Timed sit on the IDENTICAL
# common evaluation window (the static legs are the same portfolio for every
# model; only the scored months differ). Using hist_mean's static row instead
# put Static on 421 months vs Timed on 414 → the 0.49-vs-0.50 static Sharpe
# inconsistency flagged in the 2026-07-19 review.
t4_raw <- port_eval |>
  filter(
    char_col %in% c(names(char_labels), "COMBINED"),
    model    == best_model,
    strategy %in% c("static", "direction")
  ) |>
  mutate(
    char_label = ifelse(char_col == "COMBINED", "Combined",
                        char_labels[char_col]),
    char_label = factor(char_label, levels = c(char_order, "Combined")),
    panel      = ifelse(strategy == "static", "Static", paste0("Timed (", model_labels[best_model], ")"))
  ) |>
  select(panel, char_label, mean_ret, sharpe, alpha_pct, t_alpha) |>
  mutate(
    mean_fmt  = formatC(mean_ret,  digits = 2, format = "f"),
    sharpe_fmt = formatC(sharpe,   digits = 2, format = "f"),
    alpha_fmt = fmt_stars(alpha_pct, digits = 2, t_stat = t_alpha),
    t_fmt     = formatC(t_alpha,   digits = 2, format = "f")
  ) |>
  select(panel, char_label, mean_fmt, sharpe_fmt, alpha_fmt, t_fmt) |>
  arrange(char_label)

t4_wide <- t4_raw |>
  pivot_wider(
    names_from  = panel,
    values_from = c(mean_fmt, sharpe_fmt, alpha_fmt, t_fmt)
  )

# Reorder columns: Static first, then Timed
static_label <- "Static"
timed_label  <- paste0("Timed (", model_labels[best_model], ")")
col_order <- c("char_label",
               paste0(c("mean_fmt", "sharpe_fmt", "alpha_fmt", "t_fmt"), "_", static_label),
               paste0(c("mean_fmt", "sharpe_fmt", "alpha_fmt", "t_fmt"), "_", timed_label))
t4_wide <- t4_wide[, col_order]

colnames(t4_wide) <- c(
  "Characteristic",
  "Ret$^{a}$", "Sharpe", "$\\alpha^{a}$", "$t(\\alpha)$",
  "Ret$^{a}$", "Sharpe", "$\\alpha^{a}$", "$t(\\alpha)$"
)

kbl_t4 <- kbl(t4_wide,
  format   = "latex",
  booktabs = TRUE,
  escape   = FALSE,
  caption  = sprintf(
    "Portfolio performance: static vs factor-timed long-short strategies.
    Static portfolio holds each characteristic long-short in the historically
    priced direction. Timed portfolio flips each position to the predicted
    sign of the gamma (direction timing) from the %s model, which is selected
    as the model with the lowest RAFE (risk-adjusted forecast error), the
    economically-aligned forecast metric, rather than the highest out-of-sample
    $R^2$. $\\alpha$ is the Fama--French five-factor alpha. $t(\\alpha)$ is based
    on Newey--West (6 lags) standard errors. The five-factor model contains no
    momentum factor, and the momentum leg loads on one heavily;
    Table~\\ref{tab:app-ff6} reports six-factor alphas for the combined book.",
    model_labels[best_model]
  ),
  caption.short = "Portfolio performance: static versus factor-timed long-short strategies",
  label    = "portfolio_performance"
) |>
  add_header_above(c(
    " " = 1,
    "Static"                            = 4,
    setNames(4, timed_label)
  ), escape = FALSE) |>
  row_spec(nrow(t4_wide), bold = TRUE) |>   # bold Combined row
  footnote(
    alphabet          = c("Annualised \\\\%, Newey--West standard errors."),
    general           = "*** p<0.01, ** p<0.05, * p<0.10.",
    general_title     = "",
    footnote_as_chunk = TRUE,
    escape            = FALSE
  ) |>
  kable_styling(latex_options = c("HOLD_position"), font_size = 10)

writeLines(as.character(kbl_t4), "tables/t4_portfolio_performance.tex")
cat("Saved: tables/t4_portfolio_performance.tex\n")


# ==============================================================================
# Table 5 — GRS Test Results
# ==============================================================================
# The main text needs a curated three-model exhibit — one penalised linear model,
# one tree, and the headline combination — with a SINGLE SPANNING static row,
# because the static strategy ignores the forecast and is therefore identical
# across models. Until 2026-08-30 this file was a generic all-model grid that
# nothing ever \input, while Results.tex carried a hand-maintained copy of the
# curated version: two exhibits of the same statistic, free to drift apart. This
# file is now that curated exhibit and Results.tex \inputs it as tab:res-grs;
# the appendix (tab:app-grs-full, from 07) keeps the full all-model grid.
# Built with sprintf rather than kbl() because kableExtra offers no clean way to
# emit the spanning static row.
# Availability is governed by grs_results, NOT by `available_models` — the latter
# is Table 3's LINEAR model ordering (hist_mean/ar1/ar3/mv/var1/lasso), which is
# precisely why the old generic version of this file showed AR/VAR rows and could
# never match the main text.
grs_main_models <- intersect(c("lasso", "rf", "comb_xgb"), unique(grs_results$model))
stopifnot(length(grs_main_models) == 3L)

grs_cell <- function(f, p) sprintf("%s (%.3f)", formatC(f, digits = 2, format = "f"), p)
grs_get  <- function(m, s) {
  r <- grs_results |> filter(model == m, strategy == s)
  if (nrow(r) == 0L) NA_character_ else grs_cell(r$grs_f[1], r$grs_p[1])
}
# The spanning row is only honest if static really is model-invariant.
grs_static <- unique(vapply(grs_main_models, grs_get, character(1), s = "static"))
stopifnot(length(grs_static) == 1L, !is.na(grs_static))

grs_row <- function(label, s) sprintf(
  "%s & %s\\\\", label,
  paste(vapply(grs_main_models, grs_get, character(1), s = s), collapse = " & ")
)

t5_lines <- c(
  "\\begin{table}[H]",
  "\\centering",
  paste0(
    "\\caption[GRS tests of jointly zero five-factor alphas]{GRS tests of jointly\n",
    "zero Fama--French five-factor alphas across the\n",
    "six characteristic long-short portfolios, January 1990--2025. The static\n",
    "row is identical across forecast models on the common window. $F$-statistics\n",
    "with degrees of freedom $(6, T-11)$; $p$-values in parentheses.}"
  ),
  "\\label{tab:res-grs}",
  "\\fontsize{10}{12}\\selectfont",
  "\\begin{tabular}{lrrr}",
  "\\toprule",
  sprintf("Strategy & %s\\\\",
          paste(unname(model_labels[grs_main_models]), collapse = " & ")),
  "\\midrule",
  sprintf("Static      & \\multicolumn{3}{c}{%s}\\\\", grs_static),
  grs_row("Direction  ", "direction"),
  grs_row("Scaled     ", "scaled"),
  grs_row("Vol-managed", "volmanaged"),
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}"
)
writeLines(t5_lines, "tables/t5_grs_test.tex")
cat("Saved: tables/t5_grs_test.tex\n")


# ==============================================================================
# Table 6 — Neural-Net Retraining-Frequency Robustness
# ==============================================================================
# Robustness of the MLP/LSTM OOS R² to how often the network is re-estimated.
# Primary spec retrains every 6 months; here we compare 1-month and 12-month
# retraining (variants imported as {model}_r1 / {model}_r12). Stable signs and
# magnitudes — especially for Size — show the result is not an artefact of a
# frozen-model window. See "Known Issue: Neural Net Retraining Frequency".
retrain_models <- c("mlp_r1", "mlp", "mlp_r12", "lstm_r1", "lstm", "lstm_r12")

# Same common-window source as Tables 3/3b (identical sample across frequencies).
oos_eval_full <- if ("sub_period" %in% names(oos_eval_tab)) {
  filter(oos_eval_tab, sub_period == "Full")
} else oos_eval_tab

t6 <- oos_eval_full |>
  filter(gamma %in% char_order, model %in% retrain_models) |>
  mutate(
    gamma = factor(gamma, levels = char_order),
    model = factor(model, levels = retrain_models),
    cell  = r2_cell(oos_r2, p_cw)
  ) |>
  select(gamma, model, cell) |>
  pivot_wider(names_from = model, values_from = cell) |>
  arrange(gamma) |>
  rename(Characteristic = gamma)

# Defensive: only proceed if both nets imported with all three frequencies.
if (all(retrain_models %in% colnames(t6)[-1]) || all(retrain_models %in% oos_eval_tab$model)) {
  t6 <- t6[, c("Characteristic", retrain_models)]
  colnames(t6) <- c("Characteristic", "1m", "6m", "12m", "1m", "6m", "12m")

  kbl_t6 <- kbl(t6,
    format   = "latex",
    booktabs = TRUE,
    escape   = FALSE,
    caption  = paste0("Robustness of neural-net out-of-sample $R^2$ (\\%) to retraining
                frequency. Each network is re-estimated every 1, 6 (primary), or
                12 months on an expanding window; intervening months reuse the
                frozen model. The 6-month columns reproduce the primary
                specification of Table~\\ref{tab:oos_r2_ml}. Stars denote the
                one-sided Clark--West (2007) test of equal predictive accuracy
                versus the historical mean. ", cw_neg_note),
    caption.short = "Out-of-sample $R^2$ by neural-net retraining frequency",
    label    = "retrain_robustness"
  ) |>
    add_header_above(c(" " = 1, "MLP" = 3, "LSTM" = 3)) |>
    footnote(
      general           = "*** p<0.01, ** p<0.05, * p<0.10 (Clark--West test, one-sided).",
      general_title     = "",
      footnote_as_chunk = TRUE,
      escape            = FALSE
    ) |>
    kable_styling(latex_options = c("HOLD_position"), font_size = 10)

  writeLines(as.character(kbl_t6), "tables/t6_retrain_robustness.tex")
  cat("Saved: tables/t6_retrain_robustness.tex\n")
} else {
  cat("Skipped Table 6: retraining-frequency variants (mlp_r1/_r12, lstm_r1/_r12) not all present in oos_eval.\n")
}


# ==============================================================================
# Table 7 — Structural Breaks (CPM/GLR, real-time detection)
# ==============================================================================
# Input produced by 03c_structural_breaks.R. Reports the estimated break date,
# the REAL-TIME detection date (online CPM), the detection delay (break-age /
# ambiguity window), and the adjacent-regime means/vols per gamma.
# Reference: Stöckl et al. (2026, "Breaking Bad"); Ross (2015, JSS).
if (file.exists("structural_breaks.rds")) {
  sb <- readRDS("structural_breaks.rds")

  t7 <- sb$regime_stats |>
    mutate(
      gamma = factor(gamma, levels = char_order),
      `Break`      = format(break_date,  "%Y-%m"),
      `Detected`   = format(detect_date, "%Y-%m"),
      `Delay (m)`  = delay_months,
      `Mean pre`   = formatC(mean_pre,  digits = 2, format = "f"),
      `Mean post`  = formatC(mean_post, digits = 2, format = "f"),
      `SD pre`     = formatC(sd_pre,    digits = 2, format = "f"),
      `SD post`    = formatC(sd_post,   digits = 2, format = "f")
    ) |>
    arrange(gamma, break_date) |>
    select(Characteristic = gamma, Break, Detected, `Delay (m)`,
           `Mean pre`, `Mean post`, `SD pre`, `SD post`)

  # group-aligned row spacing: \addlinespace at each characteristic boundary
  # (kableExtra's default inserts it every 5 rows, splitting groups mid-table)
  grp_sizes <- t7 |> count(Characteristic, name = "n") |>
    arrange(match(Characteristic, char_order)) |> pull(n)
  t7_linesep <- unlist(map(grp_sizes, \(n) c(rep("", n - 1), "\\addlinespace")))
  t7_linesep <- head(t7_linesep, -1)   # no trailing space after last group

  # collapse repeated characteristic labels for readability
  t7 <- t7 |>
    group_by(Characteristic) |>
    mutate(Characteristic = if_else(row_number() == 1,
                                    as.character(Characteristic), "")) |>
    ungroup()

  kbl_t7 <- kbl(t7,
    format   = "latex",
    booktabs = TRUE,
    escape   = FALSE,
    linesep  = t7_linesep,
    caption  = sprintf(
      "Structural breaks in the gamma series detected by the online change-point
       model (CPM) with the generalized likelihood ratio (GLR) statistic
       (ARL$_0$ = %s, startup = %d; Ross 2015). Break is the estimated
       break date; Detected is the month in which the sequential detector
       first flagged the break using only real-time data; Delay is the
       detection delay in months, the window during which the new regime is
       statistically ambiguous to a real-time observer. Regime means and standard
       deviations are in \\%% per month for the segments adjacent to each break.",
      format(sb$spec$ARL0, big.mark = ","), sb$spec$startup),
    caption.short = "Structural breaks in the gamma series",
    label    = "structural_breaks"
  ) |>
    add_header_above(c(" " = 4, "Regime mean" = 2, "Regime SD" = 2)) |>
    footnote(
      general           = "Break dates estimated by maximising the GLR statistic; detection dates are look-ahead-free.",
      general_title     = "",
      footnote_as_chunk = TRUE,
      escape            = FALSE
    ) |>
    kable_styling(latex_options = c("HOLD_position"), font_size = 10)

  writeLines(as.character(kbl_t7), "tables/t7_structural_breaks.tex")
  cat("Saved: tables/t7_structural_breaks.tex\n")
} else {
  cat("Skipped Table 7: structural_breaks.rds not found (run 03c_structural_breaks.R).\n")
}


# ==============================================================================
# Table 8 — Regime Persistence vs. Out-of-Sample Predictability
# ==============================================================================
# Confronts the break statistics (03c) with the best OOS R² per gamma at H=1
# (common 414-month window) and H=3. Key result: break FREQUENCY tracks
# (un)predictability (Spearman rho ≈ -0.7/-0.8), while detection DELAY carries
# the OPPOSITE sign to the single-stock Breaking Bad reading — at the
# aggregated-premium level a long delay is a symptom of regime persistence
# (quiet background → detector needs more evidence), not of forecasting cost.
# Discussed in thesis Section "Regime Persistence and the Cross-Section of
# Predictability" (tab:breaks_pred).
if (file.exists("structural_breaks.rds") && exists("oos_eval_common")) {
  sb <- readRDS("structural_breaks.rds")

  # retraining-frequency variants are absent from model_labels above
  model_labels_t8 <- c(model_labels,
    mlp_r1 = "MLP-1m", mlp_r12 = "MLP-12m",
    lstm_r1 = "LSTM-1m", lstm_r12 = "LSTM-12m",
  # Ext-E rolling-window variants
  lasso_w120 = "LASSO-roll10y", lasso_w240 = "LASSO-roll20y",
  rf_w120 = "RF-roll10y", rf_w240 = "RF-roll20y"
  )
  lbl <- function(m) coalesce(model_labels_t8[m], m)

  # break statistics per gamma: count, median delay, median regime length (TBB)
  brk_stats <- sb$breaks_glr |>
    filter(!is.na(break_idx)) |>
    arrange(gamma, break_date) |>
    group_by(gamma) |>
    summarise(
      n_breaks  = n(),
      med_delay = median(delay_months),
      regime_m  = ifelse(n() >= 2,
                         median(diff(as.numeric(break_date)) / 30.44),
                         NA_real_),
      .groups = "drop"
    )

  # best H=1 OOS R² across all models (common window)
  best_h1 <- oos_eval_common |>
    filter(sub_period == "Full") |>
    group_by(gamma) |>
    slice_max(oos_r2, n = 1, with_ties = FALSE) |>
    summarise(h1_r2 = oos_r2, h1_model = model, .groups = "drop")

  # best H=3 OOS R² (optional input)
  best_h3 <- tibble(gamma = character(), h3_r2 = double(), h3_model = character())
  if (file.exists("gamma_predictions_h3.rds")) {
    h3_obj <- readRDS("gamma_predictions_h3.rds")
    # common-window eval (04 Ext-C 5b); older vintages fall back to per-model
    ev3 <- h3_obj$oos_eval_common_h3 %||% h3_obj$oos_eval_h3
    best_h3 <- ev3 |>
      filter(sub_period == "Full") |>
      group_by(gamma) |>
      slice_max(oos_r2, n = 1, with_ties = FALSE) |>
      summarise(h3_r2 = oos_r2, h3_model = model, .groups = "drop")
  }

  t8_raw <- brk_stats |>
    left_join(best_h1, by = "gamma") |>
    left_join(best_h3, by = "gamma") |>
    arrange(desc(h1_r2))

  # Spearman rank correlations (N = 6 — descriptive only)
  rho <- function(x, y) cor(x, y, method = "spearman", use = "complete.obs")
  rho_brk_h1   <- rho(t8_raw$n_breaks,  t8_raw$h1_r2)
  rho_delay_h1 <- rho(t8_raw$med_delay, t8_raw$h1_r2)
  rho_brk_h3   <- if (nrow(best_h3) > 0) rho(t8_raw$n_breaks, t8_raw$h3_r2) else NA_real_

  fmt_rho <- function(r) sprintf("%s%.2f", ifelse(r < 0, "$-$", "$+$"), abs(r))
  fmt_r2  <- function(r2, m) ifelse(is.na(r2), "",
    sprintf("%s%.2f (%s)", ifelse(r2 < 0, "$-$", ""), abs(r2), lbl(m)))

  t8 <- t8_raw |>
    transmute(
      Characteristic = gamma,
      Breaks         = as.character(n_breaks),
      `Delay (m)`    = formatC(med_delay, digits = 1, format = "f"),
      `Regime (m)`   = formatC(regime_m,  digits = 0, format = "f"),
      `$H=1$`        = fmt_r2(h1_r2, h1_model),
      `$H=3$`        = fmt_r2(h3_r2, h3_model)
    ) |>
    add_row(Characteristic = "Rank correlation with breaks",
            Breaks = "", `Delay (m)` = "", `Regime (m)` = "",
            `$H=1$` = fmt_rho(rho_brk_h1),
            `$H=3$` = if (is.na(rho_brk_h3)) "" else fmt_rho(rho_brk_h3)) |>
    add_row(Characteristic = "Rank correlation with delay",
            Breaks = "", `Delay (m)` = "", `Regime (m)` = "",
            `$H=1$` = fmt_rho(rho_delay_h1), `$H=3$` = "")

  # \midrule between the gamma rows and the correlation summary rows
  t8_linesep <- c(rep("", nrow(t8_raw) - 1), "\\midrule", "")

  kbl_t8 <- kbl(t8,
    format   = "latex",
    booktabs = TRUE,
    escape   = FALSE,
    linesep  = t8_linesep,
    align    = c("l", "r", "r", "r", "r", "r"),
    caption  = sprintf(
      "Structural-break statistics and out-of-sample predictability by gamma.
       Breaks is the number of breaks detected by the online CPM/GLR
       procedure (ARL$_0$ = %s); Delay is the median detection delay in
       months; Regime is the median number of months between consecutive
       breaks. Best OOS $R^2$ is the largest Campbell--Thompson out-of-sample
       $R^2$ (\\%%) achieved by any model for that gamma at the monthly
       (common 414-month sample) and quarterly horizon, with the corresponding
       model in parentheses. Rows are ordered by monthly predictability.",
      format(sb$spec$ARL0, big.mark = ",")),
    caption.short = "Structural breaks and out-of-sample predictability by gamma",
    label    = "breaks_pred"
  ) |>
    add_header_above(c(" " = 4, "Best OOS $R^2$ (\\\\%)" = 2), escape = FALSE) |>
    footnote(
      general           = "Spearman rank correlations across the six gammas; N = 6, descriptive only.",
      general_title     = "",
      footnote_as_chunk = TRUE,
      escape            = FALSE
    ) |>
    kable_styling(latex_options = c("HOLD_position"), font_size = 10)

  writeLines(as.character(kbl_t8), "tables/t8_breaks_vs_predictability.tex")
  cat("Saved: tables/t8_breaks_vs_predictability.tex\n")

  # --- Thesis exhibit: breaks-vs-predictability scatter (2026-07-19) ----------
  # The monotone rho = -0.79 pattern of Table 8 as a labelled scatter; both
  # horizons as facets, six labelled points each. Cheap companion to the table.
  sc_df <- bind_rows(
    t8_raw |> transmute(gamma, n_breaks, r2 = h1_r2, Horizon = "Monthly (H=1)"),
    t8_raw |> transmute(gamma, n_breaks, r2 = h3_r2, Horizon = "Quarterly (H=3)")
  ) |> filter(!is.na(r2))
  rho_lab <- sc_df |>
    group_by(Horizon) |>
    summarise(r = cor(n_breaks, r2, method = "spearman"), .groups = "drop") |>
    mutate(lab = sprintf("rho == %.2f", r))
  sc_df <- sc_df |>   # dodge the near-coincident Asset Growth / Beta labels
    mutate(lab_vjust = case_when(gamma == "Beta" & Horizon == "Monthly (H=1)" ~ 1.7,
                                 gamma == "Asset Growth" ~ -0.7,
                                 TRUE ~ 0.35))
  p_bp <- ggplot(sc_df, aes(n_breaks, r2)) +
    geom_hline(yintercept = 0, linewidth = 0.3) +
    geom_point(colour = "#4477AA", size = 2.2) +
    ggplot2::geom_text(aes(label = gamma, vjust = lab_vjust), size = 2.9,
                       hjust = -0.12) +
    ggplot2::geom_text(data = rho_lab, aes(x = Inf, y = Inf, label = lab),
                       parse = TRUE, hjust = 1.25, vjust = 1.8, size = 3.2) +
    facet_wrap(~Horizon, scales = "free_y") +
    scale_x_continuous(breaks = 3:9, expand = expansion(mult = c(0.06, 0.22))) +
    labs(x = "Number of detected breaks (CPM/GLR)",
         y = expression("Best OOS R"^2*" (%)")) +
    theme_bw(base_size = 10) +
    theme(panel.grid.minor = element_blank())
  ggsave("plots/breaks_vs_pred_scatter.pdf", p_bp, width = 8, height = 3.4)
  cat("Saved: plots/breaks_vs_pred_scatter.pdf\n")

  cat("\n=== Table 8 sanity ===\n")
  print(as.data.frame(t8_raw), row.names = FALSE)
  cat(sprintf("rho(breaks, H1) = %+.2f | rho(breaks, H3) = %+.2f | rho(delay, H1) = %+.2f\n",
              rho_brk_h1, rho_brk_h3, rho_delay_h1))
} else {
  cat("Skipped Table 8: needs structural_breaks.rds and oos_eval_common (run 03c + 04).\n")
}


# ==============================================================================
# Tables 9 and 10 — main-text counterparts of the D.1 / D.11 appendix grids
# ==============================================================================
# Added 2026-08-30 after reviewer feedback: Section 4.5.1 quoted direction-timed
# Sharpe ratios and six-factor alphas in prose with only an appendix pointer
# ("Wo ist die Tabelle??"). The appendix versions are 36-row longtables, which
# cannot be placed [H] and are too long for the main text, so these are curated
# subsets holding exactly the models the prose names. The full grids stay in the
# appendix as tab:app-direction-full and tab:app-ff6.
num_t <- function(x, d = 2) formatC(x, digits = d, format = "f")
star_t <- function(p) ifelse(is.na(p), "",
  ifelse(p < 0.01, "***", ifelse(p < 0.05, "**", ifelse(p < 0.10, "*", ""))))

# Models quoted in Section 4.5.1: the ex-ante RAFE pick and the rest of the
# near-tied RAFE group, then the stronger single forecasters named in the text.
T9_MODELS <- c("comb_xgb", "xgb", "lstm_r12", "comb", "mlp", "rf", "lasso", "lstm_r1")
t9_avail  <- intersect(T9_MODELS, unique(port_eval$model))

pe_comb_t <- port_eval |> filter(char_col == "COMBINED")
t9_static <- pe_comb_t |>
  filter(strategy == "static", model == "hist_mean") |>
  transmute(Model = "Static benchmark", sharpe, vol_pct, max_dd, alpha_pct, p_alpha, t_alpha)
t9 <- pe_comb_t |>
  filter(strategy == "direction", model %in% t9_avail) |>
  mutate(model = factor(model, levels = t9_avail)) |>
  arrange(model) |>
  transmute(Model = model_labels[as.character(model)],
            sharpe, vol_pct, max_dd, alpha_pct, p_alpha, t_alpha) |>
  bind_rows(t9_static, y = _) |>
  transmute(Model,
            Sharpe                = num_t(sharpe),
            `Vol (\\%)`           = num_t(vol_pct, 1),
            `Max DD (\\%)`        = num_t(max_dd, 1),
            `$\\alpha$ (\\%)`     = paste0(num_t(alpha_pct), star_t(p_alpha)),
            `$t(\\alpha)$`        = num_t(t_alpha))

kbl_t9 <- kbl(t9,
  format = "latex", booktabs = TRUE, escape = FALSE,
  caption = paste0(
    "Direction-timed combined six-characteristic portfolio across forecasting ",
    "models, common 414-month sample, gross of costs. The first row is the ",
    "static benchmark, which is model-independent. Comb+XGB is the ex-ante ",
    "RAFE-selected model of Table~\\ref{tab:portfolio_performance}; Comb+XGB, ",
    "XGBoost, LSTM-12m, Comb.\\ and MLP form the near-tied best-RAFE group of ",
    "Section~\\ref{sec:res-rafe}, and the remaining rows are the stronger ",
    "single forecasters discussed in the text. ",
    "$\\alpha$ is the annualised Fama--French ",
    "five-factor alpha with Newey--West (6 lags) $t$-statistics. ",
    "Table~\\ref{tab:app-direction-full} reports all 36 model variants."),
  caption.short = "Direction timing across forecasting models",
  label = "res-direction-models", align = c("l", rep("r", 5))) |>
  footnote(general = "*** p<0.01, ** p<0.05, * p<0.10.", general_title = "",
           footnote_as_chunk = TRUE, escape = FALSE) |>
  kable_styling(latex_options = c("HOLD_position"), font_size = 10)
writeLines(as.character(kbl_t9), "tables/t9_direction_models.tex")
cat("Saved: tables/t9_direction_models.tex\n")

# Table 10 — five- vs six-factor alphas. The momentum leg is included because
# it is the whole diagnosis: the static book's five-factor alpha is largely a
# momentum loading, which is what the reviewer objection turns on.
T10_MODELS <- c("comb_xgb", "lasso", "mlp", "rf", "lstm_r1")
t10_avail  <- intersect(T10_MODELS, unique(port_eval$model))
mom_col    <- unique(port_eval$char_col[port_eval$char_label == "Momentum"])

t10_head <- port_eval |>
  filter(strategy == "static", model == "hist_mean",
         char_col %in% c(mom_col, "COMBINED")) |>
  mutate(Model = ifelse(char_col == "COMBINED",
                        "Combined book, static", "Momentum leg, static"),
         ord = ifelse(char_col == "COMBINED", 2, 1)) |>
  arrange(ord) |>
  select(Model, alpha_pct, p_alpha, t_alpha, alpha6_pct, p_alpha6, t_alpha6, b_umd)
t10 <- pe_comb_t |>
  filter(strategy == "direction", model %in% t10_avail) |>
  mutate(model = factor(model, levels = t10_avail)) |>
  arrange(model) |>
  transmute(Model = paste0("Combined, timed: ", model_labels[as.character(model)]),
            alpha_pct, p_alpha, t_alpha, alpha6_pct, p_alpha6, t_alpha6, b_umd) |>
  bind_rows(t10_head, y = _) |>
  transmute(Model,
            `$\\alpha_{5}$ (\\%)`         = paste0(num_t(alpha_pct), star_t(p_alpha)),
            `$t$`                          = num_t(t_alpha),
            `$\\alpha_{6}$ (\\%)`         = paste0(num_t(alpha6_pct), star_t(p_alpha6)),
            `$t$ `                         = num_t(t_alpha6),
            `$\\beta_{\\text{UMD}}$`      = num_t(b_umd))

kbl_t10 <- kbl(t10,
  format = "latex", booktabs = TRUE, escape = FALSE,
  caption = paste0(
    "Five- versus six-factor alphas of the combined book and of the static ",
    "momentum leg, common 414-month sample, gross of costs. ",
    "$\\alpha_{5}$ is the Fama--French five-factor alpha reported throughout ",
    "the chapter; $\\alpha_{6}$ adds the momentum factor of the French data ",
    "library, giving the six-factor model of \\citet{FamaFrench2018}, and ",
    "$\\beta_{\\text{UMD}}$ is the resulting momentum loading. All ",
    "$t$-statistics use Newey--West (6 lags) standard errors. The momentum leg ",
    "is close to a repackaging of the momentum factor, which is why pricing it ",
    "removes two-thirds of the static book's five-factor alpha while leaving ",
    "the timed books largely intact. ",
    "Table~\\ref{tab:app-ff6} reports all 36 model variants."),
  caption.short = "Five- versus six-factor alphas",
  label = "res-ff6", align = c("l", rep("r", 5))) |>
  footnote(general = "*** p<0.01, ** p<0.05, * p<0.10.", general_title = "",
           footnote_as_chunk = TRUE, escape = FALSE) |>
  kable_styling(latex_options = c("HOLD_position"), font_size = 10)
writeLines(as.character(kbl_t10), "tables/t10_ff6_alphas.tex")
cat("Saved: tables/t10_ff6_alphas.tex\n")


# ==============================================================================
# Summary
# ==============================================================================
cat("\n=== Tables export complete ===\n")
cat("LaTeX files in tables/. Include in thesis with:\n")
cat("  \\input{tables/t1_fm_risk_premia.tex}\n")
cat("  \\input{tables/t2_gamma_descriptives.tex}\n")
cat("  \\input{tables/t3_oos_r2.tex}\n")
cat("  \\input{tables/t3b_oos_r2_ml.tex}\n")
cat("  \\input{tables/t4_portfolio_performance.tex}\n")
cat("  \\input{tables/t5_grs_test.tex}\n")
if (file.exists("tables/t6_retrain_robustness.tex"))
  cat("  \\input{tables/t6_retrain_robustness.tex}\n")
if (file.exists("tables/t7_structural_breaks.tex"))
  cat("  \\input{tables/t7_structural_breaks.tex}\n")
if (file.exists("tables/t8_breaks_vs_predictability.tex"))
  cat("  \\input{tables/t8_breaks_vs_predictability.tex}\n")
cat("\nRequired LaTeX packages: booktabs, caption, float, xcolor, colortbl\n")

# ==============================================================================
# Sync into the Overleaf project (thesis_drafts/tables/)
# ==============================================================================
# Results.tex \input{tables/tX_...} these files directly, so re-running 06 keeps
# the thesis tables in lockstep with the pipeline — no hand-copied numbers.
# t5 is NOT yet wired into Results.tex (its generator needs a per-model pivot
# rewrite before it matches the hand-built GRS table); it is copied anyway so
# the folder is complete once that rewrite lands.
thesis_tables_dir <- "thesis_drafts/tables"
if (dir.exists("thesis_drafts")) {
  dir.create(thesis_tables_dir, showWarnings = FALSE)
  tex_files <- list.files("tables", pattern = "\\.tex$", full.names = TRUE)
  copied <- file.copy(tex_files, thesis_tables_dir, overwrite = TRUE)
  cat(sprintf("\nSynced %d/%d table files to %s/\n",
              sum(copied), length(tex_files), thesis_tables_dir))
} else {
  cat("\nthesis_drafts/ not found — skipped Overleaf table sync.\n")
}
