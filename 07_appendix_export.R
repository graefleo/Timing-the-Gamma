# ==============================================================================
# 07_appendix_export.R
# Exports all THESIS APPENDIX tables as publication-ready LaTeX (.tex) files,
# one per exhibit, and syncs them into thesis_drafts/tables/. The appendix holds
# the EXTENDED (all-model) versions of the main-text tables plus the data and
# robustness documentation; the main text keeps the selected-model views.
#
# Appendix chapters (see Anhang.tex):
#   A  Data and Predictor Set      -> a1_features, a2_factor_signals
#   B  Extended Statistical Results-> b1_oos_full, b2_fdr, b3_mcs,
#                                      b4_subperiod, b5_h3_full, b6_smooth
#   C  Estimation Robustness       -> c1_seed_mlp, c2_seed_lstm, c3_seed_rafe
#   D  Extended Portfolio Results  -> d1_direction_full, d2_netcost_full,
#                                      d3_rafe_full, d4_grs_full
#
# Input:  gamma_predictions.rds, gamma_predictions_h3.rds,
#         gamma_predictions_smooth.rds, portfolio_results.rds,
#         multiple_testing_fdr.csv, multiple_testing_mcs.csv,
#         multiple_testing_mcs_pvals.csv,
#         seed_dispersion_{mlp,lstm}.csv, colab/colab_io/{mlp,lstm}_preds.csv
# Output: tables/{a,b,c,d}*.tex  (+ synced to thesis_drafts/tables/)
#
# Conventions mirror 06_tables_export.R: kableExtra LaTeX, booktabs, en-dashed
# names in captions, Clark-West stars, common 414-month window for every
# cross-model statistic. Labels use the "app-*" prefix (kable adds "tab:").
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(kableExtra)

gp <- readRDS("gamma_predictions.rds")
list2env(readRDS("portfolio_results.rds"), envir = environment())
gts <- readRDS("gammas_ts.rds")   # -> fm_table (WLS), fm_table_ols (OLS)

dir.create("tables", showWarnings = FALSE)

# ── Shared conventions (mirrors 06) ───────────────────────────────────────────
char_order <- c("Value", "Momentum", "Profitability", "Asset Growth", "Size", "Beta")

model_labels <- c(
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
lbl <- function(m) ifelse(is.na(model_labels[m]), m, model_labels[m])

# Full model ordering for the extended grids: benchmarks -> linear -> penalised
# -> trees -> nets (+ variants) -> combinations.
FULL_ORDER <- c(
  "hist_mean", "factmom", "ar1", "ar3", "mv", "var1", "har",
  "lasso", "lasso_ct", "lasso_1se", "lasso_1se_ct",
  "lasso_w120", "lasso_w240",
  "rf", "rf_ct", "rf_w120", "rf_w240", "xgb", "xgb_ct",
  "mlp", "mlp_ct", "mlp_r1", "mlp_r1_ct", "mlp_r12", "mlp_r12_ct",
  "mlp_nogeo", "mlp_nogeo_ct", "mlp_sr", "mlp_sr_ct",
  "lstm", "lstm_r1", "lstm_r12", "lstm_nogeo",
  "comb", "comb_rf", "comb_xgb"
)

star <- function(p) ifelse(is.na(p), "",
  ifelse(p < 0.01, "***", ifelse(p < 0.05, "**", ifelse(p < 0.10, "*", ""))))
r2_cell <- function(r2, p) paste0(formatC(r2, digits = 2, format = "f"), star(p))
num  <- function(x, d = 2) formatC(x, digits = d, format = "f")

cw_note <- "*** p<0.01, ** p<0.05, * p<0.10 (Clark--West test, one-sided)."
cw_neg  <- "The Clark--West statistic adjusts for the estimation noise the larger
    model introduces under the null, so a model can be significantly informative
    (stars) while its unadjusted out-of-sample $R^2$ is negative."

save_app <- function(kbl_obj, filename) {
  writeLines(as.character(kbl_obj), file.path("tables", filename))
  cat(sprintf("Saved: tables/%s\n", filename))
}

# Order a model column by FULL_ORDER, keeping only present models, apply labels.
order_models <- function(df, col = "model") {
  present <- intersect(FULL_ORDER, unique(df[[col]]))
  df[[col]] <- factor(df[[col]], levels = present, labels = lbl(present))
  df
}


# ==============================================================================
# A.1 — Full predictor set (118 features), grouped
# ==============================================================================
# Group assignment: macro membership is tested BEFORE the "_spread$" rule so the
# two macro spreads (default_spread, term_spread) are NOT mis-bucketed as
# characteristic spreads — this reproduces the Methodology group counts
# (24/6/8/5/75) exactly, unlike the exploratory classifier in 03b which orders
# the spread rule first (a cosmetic quirk that only affects a heatmap ordering).
fc <- gp$feature_cols
macro_names <- c("default_spread","term_spread","short_rate","vix",
                 "indpro_gap","mkt_lag1","infl","lty")
gw_names    <- c("dp","ep","de","ntis","svar")
classify <- function(fn) {
  if (grepl("_l[123]$", fn))                                        return("Own-gamma lag")
  if (grepl("^(bm|mom12m|oper_prof|asset_growth|size|beta)_vol12$", fn)) return("Own-gamma volatility")
  if (fn %in% macro_names)                                          return("Macro conditioning")
  if (grepl("_spread$", fn))                                        return("Characteristic spread")
  if (fn %in% gw_names)                                             return("Goyal--Welch predictor")
  if (grepl("^(mkt|smb|hml|rmw|cma)_", fn))                         return("Factor signal")
  return("Other")
}
grp <- sapply(fc, classify)

# Per-variable descriptions for the non-factor-signal blocks (metadata, not data).
desc_map <- c(
  # own-gamma lags (pattern) filled programmatically below
  bm_spread             = "Cross-sectional IQR of book-to-market each month",
  mom12m_spread         = "Cross-sectional IQR of 12-month momentum each month",
  oper_prof_spread      = "Cross-sectional IQR of operating profitability each month",
  asset_growth_spread   = "Cross-sectional IQR of asset growth each month",
  log_mktcap_lag_spread = "Cross-sectional IQR of log market capitalisation each month",
  beta_spread           = "Cross-sectional IQR of market beta each month",
  default_spread        = "Moody's BAA minus AAA corporate yield (credit / recession proxy)",
  term_spread           = "10-year Treasury minus 3-month T-bill yield (business cycle)",
  short_rate            = "3-month Treasury bill rate",
  vix                   = "CBOE implied volatility index, month-end (from 1990)",
  indpro_gap            = "12-month log growth in industrial production",
  mkt_lag1              = "One-month lagged market excess return",
  infl                  = "Consumer price index inflation, month $t-1$",
  lty                   = "Long-term Treasury yield (20Y/30Y splice)",
  dp                    = "Log dividend-price ratio of the S\\&P 500",
  ep                    = "Log earnings-price ratio of the S\\&P 500",
  de                    = "Log dividend-payout ratio (dividend-earnings)",
  ntis                  = "Net equity issuance ratio",
  svar                  = "Stock-market realised variance"
)
src_map <- c(
  bm_spread = "Own construction", mom12m_spread = "Own construction",
  oper_prof_spread = "Own construction", asset_growth_spread = "Own construction",
  log_mktcap_lag_spread = "Own construction", beta_spread = "Own construction",
  default_spread = "\\citet{FamaFrench1989}", term_spread = "\\citet{FamaFrench1989}",
  short_rate = "\\citet{AngBekaert2007}", vix = "\\citet{MoreiraMuir2017}",
  indpro_gap = "\\citet{CooperPriestley2009}", mkt_lag1 = "\\citet{EhsaniLinnainmaa2022}",
  infl = "\\citet{WelchGoyal2008}", lty = "\\citet{GurkaynakSackWright2007}",
  dp = "\\citet{WelchGoyal2008}", ep = "\\citet{WelchGoyal2008}",
  de = "\\citet{WelchGoyal2008}", ntis = "\\citet{WelchGoyal2008}",
  svar = "\\citet{WelchGoyal2008}"
)
gamma_pretty <- c(bm = "book-to-market", mom12m = "12-month momentum",
                  oper_prof = "operating profitability", asset_growth = "asset growth",
                  size = "size (log market cap)", beta = "market beta")
describe <- function(fn) {
  if (!is.na(desc_map[fn])) return(unname(desc_map[fn]))
  if (grepl("_l[123]$", fn)) {
    base <- sub("_l[123]$", "", fn); k <- sub(".*_l([123])$", "\\1", fn)
    return(sprintf("Own %s gamma, lagged %s month%s", gamma_pretty[base], k,
                   ifelse(k == "1", "", "s")))
  }
  if (grepl("_vol12$", fn)) {
    base <- sub("_vol12$", "", fn)
    return(sprintf("12-month rolling volatility of the %s gamma", gamma_pretty[base]))
  }
  ""
}

non_fs <- fc[grp != "Factor signal"]
a1_df <- tibble(
  Variable    = paste0("\\texttt{", gsub("_", "\\\\_", non_fs), "}"),
  Description = vapply(non_fs, describe, character(1)),
  Source      = vapply(non_fs, function(f) ifelse(is.na(src_map[f]), "Own construction", unname(src_map[f])), character(1)),
  grp         = grp[grp != "Factor signal"]
)
grp_levels <- c("Own-gamma lag", "Own-gamma volatility", "Characteristic spread",
                "Macro conditioning", "Goyal--Welch predictor")
a1_df <- a1_df |> mutate(grp = factor(grp, levels = grp_levels)) |> arrange(grp)

kbl_a1 <- a1_df |>
  select(Variable, Description, Source) |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE, longtable = TRUE,
      caption = paste0(
        "The 43 gamma-history, characteristic-spread, macroeconomic, and ",
        "\\citet{WelchGoyal2008} equity-premium predictors (predictor groups 1--4 of ",
        "Section~\\ref{sec:meth-features}). The 75 factor signals that complete the ",
        "118-predictor set are documented in Table~\\ref{tab:app-factor-signals}."),
      # Short form for the List of Tables (full captions are far too long there).
      caption.short = "Gamma-history, spread, macro and Goyal--Welch predictors",
      label = "app-features",
      col.names = c("Variable", "Description", "Source"),
      align = c("l", "l", "l")) |>
  pack_rows(index = table(a1_df$grp), escape = FALSE) |>
  column_spec(2, width = "7.2cm") |>
  kable_styling(latex_options = c("repeat_header"), font_size = 9)
save_app(kbl_a1, "a1_features.tex")


# ==============================================================================
# A.2 — Factor-signal construction (75 = 5 FF factors x 15 signal types)
# ==============================================================================
sig_df <- tribble(
  ~Signal, ~Description,
  "\\texttt{\\_mom1}",   "1-month raw return of the factor",
  "\\texttt{\\_mom3}",   "3-month cumulative raw return",
  "\\texttt{\\_mom6}",   "6-month cumulative raw return",
  "\\texttt{\\_mom12}",  "12-month cumulative raw return",
  "\\texttt{\\_tsmom1}", "1-month time-series momentum sign",
  "\\texttt{\\_tsmom3}", "3-month time-series momentum sign",
  "\\texttt{\\_tsmom6}", "6-month time-series momentum sign",
  "\\texttt{\\_tsmom12}","12-month time-series momentum sign",
  "\\texttt{\\_vol12}",  "12-month rolling return volatility",
  "\\texttt{\\_vol36}",  "36-month rolling return volatility",
  "\\texttt{\\_gk\\_mom1}",  "1-month volatility-scaled momentum",
  "\\texttt{\\_gk\\_mom3}",  "3-month volatility-scaled momentum",
  "\\texttt{\\_gk\\_mom6}",  "6-month volatility-scaled momentum",
  "\\texttt{\\_gk\\_mom12}", "12-month volatility-scaled momentum",
  "\\texttt{\\_rev60}",  "60-month long-run reversal"
)
kbl_a2 <- sig_df |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE,
      caption = paste0(
        "Construction of the 75 factor-signal predictors. Each of the 15 signal ",
        "types below is computed for all five Fama--French factors ",
        "(\\texttt{mkt}, \\texttt{smb}, \\texttt{hml}, \\texttt{rmw}, \\texttt{cma}), ",
        "giving $5 \\times 15 = 75$ predictors, named ",
        "\\texttt{\\{factor\\}\\{signal\\}}. ",
        "Signal definitions follow \\citet{TimingFactorZoo_Missing}."),
      caption.short = "Construction of the 75 factor-signal predictors",
      label = "app-factor-signals",
      col.names = c("Signal suffix", "Description"), align = c("l", "l")) |>
  column_spec(2, width = "9cm") |>
  kable_styling(latex_options = c("HOLD_position"), font_size = 9)
save_app(kbl_a2, "a2_factor_signals.tex")


# ==============================================================================
# A.3 — First-stage robustness: WLS vs OLS Fama-MacBeth premia
# ==============================================================================
# Documents that the WLS primary specification is not load-bearing: OLS gives
# equal weight to microcaps, inflating every premium (and its t-statistic)
# relative to the value-weighted WLS estimates. Both from gammas_ts.rds.
fm_gamma_labels <- c(bm = "Value", mom12m = "Momentum", oper_prof = "Profitability",
                     asset_growth = "Asset Growth", size = "Size", beta = "Beta")
join_fm <- function(tab) tab |>
  filter(gamma %in% names(fm_gamma_labels)) |>
  transmute(gamma, mean_pct = mean * 100, t_nw)
a3 <- inner_join(
    join_fm(gts$fm_table)     |> rename(w_mean = mean_pct, w_t = t_nw),
    join_fm(gts$fm_table_ols) |> rename(o_mean = mean_pct, o_t = t_nw),
    by = "gamma") |>
  mutate(gamma = factor(fm_gamma_labels[gamma], levels = char_order)) |>
  arrange(gamma) |>
  transmute(
    Characteristic = as.character(gamma),
    `WLS mean` = paste0(num(w_mean), star(2 * pnorm(-abs(w_t)))),
    `WLS $t$`  = num(w_t),
    `OLS mean` = paste0(num(o_mean), star(2 * pnorm(-abs(o_t)))),
    `OLS $t$`  = num(o_t)
  )
kbl_a3 <- a3 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE,
      caption = paste0(
        "First-stage robustness: Fama--MacBeth risk premia under the primary ",
        "value-weighted (WLS) specification and under ordinary least squares (OLS). ",
        "Means are in \\% per month; $t$ is the Newey--West (12 lags) statistic. ",
        "OLS weights every stock equally and so loads heavily on microcaps, ",
        "inflating each premium and its $t$-statistic relative to WLS; the ",
        "value-weighted estimates used throughout the thesis are the conservative ",
        "choice. Extends Table~\\ref{tab:res-fm}."),
      caption.short = "First-stage robustness: risk premia under WLS and OLS",
      label = "app-wls-ols", align = c("l", "r", "r", "r", "r")) |>
  add_header_above(c(" " = 1, "WLS (primary)" = 2, "OLS" = 2), escape = FALSE) |>
  footnote(general = "*** p<0.01, ** p<0.05, * p<0.10 (Newey--West, 12 lags).",
           general_title = "", footnote_as_chunk = TRUE, escape = FALSE) |>
  kable_styling(latex_options = c("HOLD_position"), font_size = 10)
save_app(kbl_a3, "a3_wls_vs_ols.tex")


# ==============================================================================
# B.1 — Full out-of-sample R^2 grid: all models x 6 gammas (common 414 window)
# ==============================================================================
oe <- gp$oos_eval_common
n_common <- unique(oe$n_oos)[1]

grid_by_gamma <- function(ev, order = FULL_ORDER, val = "cell") {
  ev |>
    filter(gamma %in% char_order, model %in% order) |>
    mutate(cell = r2_cell(oos_r2, p_cw)) |>
    select(gamma, model, cell) |>
    pivot_wider(names_from = gamma, values_from = cell) |>
    mutate(model = factor(model, levels = intersect(order, model))) |>
    arrange(model) |>
    mutate(Model = lbl(as.character(model))) |>
    select(Model, all_of(char_order))
}

b1 <- grid_by_gamma(oe)
kbl_b1 <- b1 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE, longtable = TRUE,
      caption = sprintf(paste0(
        "Out-of-sample $R^2$ (\\%%) against the historical-mean benchmark for all ",
        "%d model variants and six gammas, common %d-month evaluation sample. ",
        "Extends Tables~\\ref{tab:oos_r2} and~\\ref{tab:oos_r2_ml}. %s"),
        nrow(b1), n_common, cw_neg),
      caption.short = "Out-of-sample $R^2$, all model variants",
      label = "app-oos-full", align = c("l", rep("r", 6))) |>
  kable_styling(latex_options = c("repeat_header"), font_size = 8) |>
  footnote(general = cw_note, general_title = "", footnote_as_chunk = TRUE, escape = FALSE)
save_app(kbl_b1, "b1_oos_full.tex")


# ==============================================================================
# B.2 — Multiple-testing detail: per-test CW and FDR-adjusted p-values
# ==============================================================================
fdr <- readr::read_csv("multiple_testing_fdr.csv", show_col_types = FALSE)
b2 <- fdr |>
  filter(p_cw < 0.10) |>
  mutate(gamma = factor(gamma, levels = char_order),
         model = factor(model, levels = intersect(FULL_ORDER, model))) |>
  arrange(gamma, model) |>
  transmute(
    Gamma = gamma, Model = lbl(as.character(model)),
    `$R^2$` = num(oos_r2),
    `$p_{\\text{CW}}$`  = num(p_cw, 3),
    `$p_{\\text{BH}}$`  = num(p_bh_family, 3),
    `$p_{\\text{BY}}$`  = num(p_by_family, 3),
    `$p_{\\text{BH-w}}$`= num(p_bh_within, 3)
  )
kbl_b2 <- b2 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE, longtable = TRUE,
      caption = sprintf(paste0(
        "Every model$\\times$gamma test with a raw one-sided Clark--West $p$-value ",
        "below 0.10 (%d of the %d-test family), with false-discovery-rate ",
        "adjustments: Benjamini--Hochberg across the whole family ($p_{\\text{BH}}$), ",
        "Benjamini--Yekutieli dependence-robust ($p_{\\text{BY}}$), and ",
        "Benjamini--Hochberg within each gamma ($p_{\\text{BH-w}}$). The remaining ",
        "%d tests have raw $p \\geq 0.10$. Supports Section~\\ref{sec:res-mt}."),
        nrow(b2), nrow(fdr), nrow(fdr) - nrow(b2)),
      caption.short = "False-discovery-rate adjustments to the Clark--West family",
      label = "app-fdr", align = c("l","l","r","r","r","r","r")) |>
  kable_styling(latex_options = c("repeat_header"), font_size = 9)
save_app(kbl_b2, "b2_fdr.tex")


# ==============================================================================
# B.3 — Model Confidence Set: per-model MCS p-values (gamma columns)
# ==============================================================================
mcs_p  <- readr::read_csv("multiple_testing_mcs_pvals.csv", show_col_types = FALSE)
mcs_sm <- readr::read_csv("multiple_testing_mcs.csv", show_col_types = FALSE)
b3 <- mcs_p |>
  filter(model %in% FULL_ORDER) |>
  mutate(gamma = factor(gamma, levels = char_order),
         model = factor(model, levels = intersect(FULL_ORDER, model))) |>
  select(gamma, model, mcs_pval) |>
  pivot_wider(names_from = gamma, values_from = mcs_pval) |>
  arrange(model) |>
  mutate(Model = lbl(as.character(model))) |>
  select(Model, all_of(char_order)) |>
  mutate(across(all_of(char_order), ~ num(.x, 2)))
kbl_b3 <- b3 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE, longtable = TRUE,
      caption = paste0(
        "Model-confidence-set $p$-values \\citep{HansenLundeNason2011} by model and ",
        "gamma, common 414-month sample. A model remains in the 90\\% (75\\%) ",
        "confidence set when its $p$-value exceeds 0.10 (0.25). The benchmark is ",
        "retained in every gamma, consistent with the low power of the MCS at ",
        "$T \\approx 414$ discussed in Section~\\ref{sec:res-mt}; the ",
        "FDR-controlled Clark--West family (Table~\\ref{tab:app-fdr}) is the primary ",
        "inference."),
      caption.short = "Model-confidence-set $p$-values by model and gamma",
      label = "app-mcs", align = c("l", rep("r", 6))) |>
  kable_styling(latex_options = c("repeat_header"), font_size = 8)
save_app(kbl_b3, "b3_mcs.tex")


# ==============================================================================
# B.4 — Sub-period out-of-sample R^2, all models (Pre-2000 / 2000-2010 / Post-2010)
# ==============================================================================
sub_ev <- gp$oos_eval_all |> filter(sub_period != "Full")
b4_long <- sub_ev |>
  filter(gamma %in% char_order, model %in% FULL_ORDER) |>
  mutate(cell = r2_cell(oos_r2, p_cw),
         sub_period = factor(sub_period, levels = c("Pre-2000","2000-2010","Post-2010"),
                             # match on the data key, display the range with an en-dash
                             labels = c("Pre-2000","2000--2010","Post-2010")),
         model = factor(model, levels = intersect(FULL_ORDER, model))) |>
  select(sub_period, model, gamma, cell) |>
  pivot_wider(names_from = gamma, values_from = cell) |>
  arrange(sub_period, model) |>
  mutate(Model = lbl(as.character(model)))
b4 <- b4_long |> select(Model, all_of(char_order))
kbl_b4 <- b4 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE, longtable = TRUE,
      caption = paste0(
        "Sub-period out-of-sample $R^2$ (\\%) against the historical mean for all ",
        "model variants and six gammas, split Pre-2000 / 2000--2010 / Post-2010. ",
        "Extends Table~\\ref{tab:res-subperiod} (which reports LASSO and the random ",
        "forest) to the full model set, including the seed-ensembled networks. ",
        "Stars from the one-sided Clark--West test within each sub-window. ", cw_neg),
      caption.short = "Sub-period out-of-sample $R^2$, all model variants",
      label = "app-subperiod", align = c("l", rep("r", 6))) |>
  pack_rows(index = table(b4_long$sub_period), escape = FALSE) |>
  kable_styling(latex_options = c("repeat_header"), font_size = 8) |>
  footnote(general = cw_note, general_title = "", footnote_as_chunk = TRUE, escape = FALSE)
save_app(kbl_b4, "b4_subperiod.tex")


# ==============================================================================
# B.5 — Quarterly (H=3) out-of-sample R^2, full model set
# ==============================================================================
# Common-window evaluation (04 Ext-C 5b): at H=3 the feature-dependent LASSO/XGB
# family forecasts 414 months and the autoregressive/combination models 418, so
# the per-model `oos_eval_h3` would put models on different samples inside one
# grid. Fall back only for .rds vintages written before that object existed.
h3_obj <- readRDS("gamma_predictions_h3.rds")
h3 <- (h3_obj$oos_eval_common_h3 %||% h3_obj$oos_eval_h3) |> filter(sub_period == "Full")
b5 <- grid_by_gamma(h3)
kbl_b5 <- b5 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE,
      caption = paste0(
        "Quarterly-horizon ($H=3$) out-of-sample $R^2$ (\\%) for the full model set ",
        "run at that horizon, against the historical mean, every model scored on ",
        "the common 414-month sample. Extends ",
        "Table~\\ref{tab:res-h3}. Stars from the one-sided Clark--West test with ",
        "Newey--West (14 lags) standard errors; significance was verified against ",
        "the \\citet[p.~835]{HansenHodrick1980} rectangular-kernel long-run variance with materially identical ",
        "results. ", cw_neg),
      caption.short = "Quarterly-horizon out-of-sample $R^2$, all model variants",
      label = "app-h3-full", align = c("l", rep("r", 6))) |>
  kable_styling(latex_options = c("HOLD_position"), font_size = 9) |>
  footnote(general = cw_note, general_title = "", footnote_as_chunk = TRUE, escape = FALSE)
save_app(kbl_b5, "b5_h3_full.tex")


# ==============================================================================
# B.6 — Backward-smoothed (HAR) target: validation exhibit (NOT predictability)
# ==============================================================================
sm  <- readRDS("gamma_predictions_smooth.rds")
sm_ev <- sm$oos_eval_sm |> filter(sub_period == "Full")
n_sm <- unique(sm_ev$n_oos)[1]
b6 <- grid_by_gamma(sm_ev)
kbl_b6 <- b6 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE,
      caption = sprintf(paste0(
        "Out-of-sample $R^2$ (\\%%) for the backward-smoothed HAR-type target ",
        "$\\bar{\\gamma}_{k,t} = \\tfrac{1}{3}(\\gamma_t + \\gamma_{t-1} + ",
        "\\gamma_{t-2})$ (%d-month sample). \\textbf{These figures are NOT ",
        "predictability evidence and are not comparable to ",
        "Tables~\\ref{tab:app-oos-full} or~\\ref{tab:app-h3-full}}: two-thirds of ",
        "the target is known at prediction time, so the high $R^2$ validates the ",
        "HAR specification (the models recover the mechanical ",
        "$\\mathrm{AC}(1) \\approx 2/3$ structure) rather than measuring genuine ",
        "forecastability. See the note in Section~\\ref{sec:res-robust}."), n_sm),
      caption.short = "Out-of-sample $R^2$ for the backward-smoothed HAR target",
      label = "app-smooth", align = c("l", rep("r", 6))) |>
  kable_styling(latex_options = c("HOLD_position"), font_size = 9) |>
  footnote(general = cw_note, general_title = "", footnote_as_chunk = TRUE, escape = FALSE)
save_app(kbl_b6, "b6_smooth.tex")


# ==============================================================================
# C.1 / C.2 — Seed-dispersion tables (MLP, LSTM)
# ==============================================================================
seed_table <- function(csv, label, netname, filename) {
  d <- readr::read_csv(csv, show_col_types = FALSE) |>
    mutate(gamma = factor(gamma, levels = char_order),
           variant = factor(variant, levels = intersect(FULL_ORDER, variant))) |>
    arrange(variant, gamma) |>
    transmute(
      variant, Gamma = gamma,
      Mean = num(mean_r2), SD = num(sd_r2),
      Min = num(min_r2), Max = num(max_r2), Ensemble = num(ensemble_r2)
    )
  idx <- table(lbl(as.character(d$variant)))
  idx <- idx[unique(lbl(as.character(d$variant)))]      # preserve order
  k <- d |> select(-variant) |>
    kbl(format = "latex", booktabs = TRUE, escape = FALSE, longtable = TRUE,
        caption = sprintf(paste0(
          "Across-seed dispersion of the %s out-of-sample $R^2$ (\\%%): mean, ",
          "standard deviation, minimum, and maximum across the five random seeds, ",
          "with the equal-weight ensemble value used in the main text. Documents ",
          "the seed-ensemble design of Section~\\ref{sec:meth-nn} ",
          "\\citep{GuKellyXiu2020} and shows the ensemble stabilises the ",
          "single-seed draws (common 414-month sample)."), netname),
        caption.short = sprintf("Across-seed dispersion of the %s out-of-sample $R^2$", netname),
        label = label, align = c("l", rep("r", 5))) |>
    pack_rows(index = idx, escape = FALSE) |>
    kable_styling(latex_options = c("repeat_header"), font_size = 9)
  save_app(k, filename)
}
seed_table("seed_dispersion_mlp.csv",  "app-seed-mlp",  "MLP",  "c1_seed_mlp.tex")
seed_table("seed_dispersion_lstm.csv", "app-seed-lstm", "LSTM", "c2_seed_lstm.tex")


# ==============================================================================
# C.3 — Per-seed RAFE (computed from the Colab per-seed forecasts)
# ==============================================================================
# Mirrors 05 block 12 exactly: RAFE = sqrt(mean_t e_t' Sigma^{-1} e_t) on the
# common 414-month window, Sigma = sample covariance of realised gammas there.
# Gives the RF best-R^2/worst-RAFE inversion a dispersion context: the network
# RAFE is stable across seeds, so the forest's poor RAFE is structural, not a
# seed draw (RF/XGBoost are deterministic single fits with no seed dispersion).
gamma_cols <- colnames(gp$actuals)
common_mask <- Reduce(`&`, lapply(names(gp$pred_list), function(gc)
  rowSums(is.na(gp$pred_list[[gc]])) == 0)) & complete.cases(gp$actuals)
Amat  <- gp$actuals[common_mask, gamma_cols, drop = FALSE]
Sig   <- cov(Amat); Sinv <- solve(Sig)
rafe_of <- function(pred_mat) {           # pred_mat: full T x 6, aligned to actuals
  E <- (pred_mat - gp$actuals)[common_mask, , drop = FALSE]
  ok <- complete.cases(E); E <- E[ok, , drop = FALSE]
  sqrt(mean(rowSums((E %*% Sinv) * E)))
}
seed_rafe_block <- function(csv, variants) {
  mp <- readr::read_csv(csv, show_col_types = FALSE)
  purrr::map_dfr(variants, function(v) {
    seeds <- as.integer(unique(na.omit(as.integer(
      sub(paste0("^", v, "_seed([0-9]+)__.*$"), "\\1",
          grep(paste0("^", v, "_seed[0-9]+__"), names(mp), value = TRUE))))))
    if (length(seeds) < 2) return(NULL)
    rs <- vapply(seeds, function(s) {
      cols <- paste0(v, "_seed", s, "__", gamma_cols)
      if (!all(cols %in% names(mp))) return(NA_real_)
      rafe_of(as.matrix(mp[, cols]))
    }, numeric(1))
    ens_cols <- paste0(v, "__", gamma_cols)   # ensemble mean columns if present
    ens <- if (all(ens_cols %in% names(mp))) rafe_of(as.matrix(mp[, ens_cols])) else NA_real_
    tibble(variant = v, mean = mean(rs), sd = sd(rs), min = min(rs), max = max(rs),
           ensemble = ens)
  })
}
c3 <- bind_rows(
  seed_rafe_block("colab/colab_io/mlp_preds.csv",  c("mlp","mlp_r1","mlp_r12","mlp_nogeo","mlp_sr")),
  seed_rafe_block("colab/colab_io/lstm_preds.csv", c("lstm","lstm_r1","lstm_r12","lstm_nogeo"))
)
if (nrow(c3) > 0) {
  c3_tab <- c3 |>
    mutate(variant = factor(variant, levels = intersect(FULL_ORDER, variant))) |>
    arrange(variant) |>
    transmute(Model = lbl(as.character(variant)),
              Mean = num(mean, 3), SD = num(sd, 3),
              Min = num(min, 3), Max = num(max, 3),
              Ensemble = ifelse(is.na(ensemble), "--", num(ensemble, 3)))
  kbl_c3 <- c3_tab |>
    kbl(format = "latex", booktabs = TRUE, escape = FALSE,
        caption = paste0(
          "Across-seed dispersion of the risk-adjusted mean forecast error (RAFE) ",
          "for the neural-network variants, computed on the common 414-month sample ",
          "with the same $\\boldsymbol{\\Sigma}$ as Table~\\ref{tab:res-rafe}. The ",
          "network RAFE is stable across seeds, confirming that the random forest's ",
          "poor RAFE (Table~\\ref{tab:res-rafe}) is a structural property of its ",
          "aggressive forecasts rather than an unlucky draw; the random forest and ",
          "XGBoost are deterministic single fits and carry no seed dispersion."),
        caption.short = "Across-seed dispersion of the neural-network RAFE",
        label = "app-seed-rafe", align = c("l", rep("r", 5))) |>
    kable_styling(latex_options = c("HOLD_position"), font_size = 9)
  save_app(kbl_c3, "c3_seed_rafe.tex")
} else cat("WARNING: C.3 per-seed RAFE produced no rows (check Colab CSV columns).\n")


# ==============================================================================
# D.1 — Direction timing, all 32 models (combined book)
# ==============================================================================
pe_comb <- port_eval |> filter(char_col == "COMBINED")
d1 <- pe_comb |>
  filter(strategy == "direction", model %in% FULL_ORDER) |>
  mutate(model = factor(model, levels = intersect(FULL_ORDER, model))) |>
  arrange(model) |>
  transmute(
    Model = lbl(as.character(model)),
    Sharpe = num(sharpe), `Vol (\\%)` = num(vol_pct, 1),
    `Max DD (\\%)` = num(max_dd, 1),
    `$\\alpha$ (\\%)` = paste0(num(alpha_pct), star(p_alpha)),
    `$t(\\alpha)$` = num(t_alpha)
  )
kbl_d1 <- d1 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE, longtable = TRUE,
      caption = paste0(
        "Direction-timed combined six-characteristic portfolio for all model ",
        "variants, common 414-month sample, gross of costs. Extends ",
        "Table~\\ref{tab:portfolio_performance} (which reports the ex-ante ",
        "RAFE-selected model). $\\alpha$ is the annualised Fama--French five-factor ",
        "alpha with Newey--West (6 lags) $t$-statistics. The near-tied ",
        "best-RAFE group spans direction-timed Sharpe ratios of roughly 0.68--0.98."),
      caption.short = "Direction-timed combined portfolio, all model variants",
      label = "app-direction-full", align = c("l", rep("r", 5))) |>
  kable_styling(latex_options = c("repeat_header"), font_size = 9) |>
  footnote(general = "*** p<0.01, ** p<0.05, * p<0.10.", general_title = "",
           footnote_as_chunk = TRUE, escape = FALSE)
save_app(kbl_d1, "d1_direction_full.tex")


# ==============================================================================
# D.2 — Net-of-cost Sharpe, all 32 models
# ==============================================================================
d2 <- tc_strategy |>
  mutate(model = factor(model, levels = intersect(FULL_ORDER, model))) |>
  filter(!is.na(model)) |>
  arrange(model) |>
  transmute(
    Model = lbl(as.character(model)),
    `Turn.` = num(direction_to),
    Gross = num(direction_gross), `10\\,bp` = num(direction_10bp),
    `50\\,bp` = num(direction_50bp),
    `VM gross` = num(volmanaged_gross), `VM 50\\,bp` = num(volmanaged_50bp)
  )
kbl_d2 <- d2 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE, longtable = TRUE,
      caption = paste0(
        "Net-of-cost annualised Sharpe ratios for all model variants, combined ",
        "portfolio, common 414-month sample. Turn. is the mean monthly ",
        "direction-overlay turnover; costs are charged as basis points $\\times$ ",
        "turnover. The last two columns give the vol-managed overlay gross and net ",
        "of 50\\,bp for contrast. Extends Table~\\ref{tab:res-costs}. The static ",
        "benchmark Sharpe is 0.50."),
      caption.short = "Net-of-cost Sharpe ratios, all model variants",
      label = "app-netcost-full", align = c("l", rep("r", 6))) |>
  kable_styling(latex_options = c("repeat_header"), font_size = 9)
save_app(kbl_d2, "d2_netcost_full.tex")


# ==============================================================================
# D.3 — Full RAFE ranking, all 32 models
# ==============================================================================
d3 <- rafe_results |>
  arrange(RAFE) |>
  transmute(
    Model = lbl(model),
    RAFE = num(RAFE, 3), `RAFE$_{\\text{CC0}}$` = num(RAFE_CC0, 3),
    RMSE = num(RMSE, 4), `Cov.\\ contr.` = num(cov_contribution, 4),
    `$N$` = n_oos
  )
kbl_d3 <- d3 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE, longtable = TRUE,
      caption = paste0(
        "Risk-adjusted mean forecast error (RAFE) for all model variants, sorted ",
        "by RAFE, common 414-month sample. Extends Table~\\ref{tab:res-rafe}. ",
        "RAFE uses the full inverse covariance of the realised gamma vector; ",
        "RAFE$_{\\text{CC0}}$ its diagonal; RMSE the identity weighting; ",
        "Cov.\\ contribution $=$ RAFE $-$ RAFE$_{\\text{CC0}}$. Forecast combinations ",
        "average over their available member forecasts (at least two required)."),
      caption.short = "Risk-adjusted mean forecast error, all model variants",
      label = "app-rafe-full", align = c("l", rep("r", 5))) |>
  kable_styling(latex_options = c("repeat_header"), font_size = 9)
save_app(kbl_d3, "d3_rafe_full.tex")


# ==============================================================================
# D.4 — GRS test, all models x strategies (proper pivot)
# ==============================================================================
# NB: the main-text generated t5 is malformed (raw pivot names, duplicated N
# rows). Here we pivot cleanly: one row per model, one "F (p)" column per
# strategy, each strategy on its own evaluation window (noted in the caption).
grs_cell <- function(f, p) sprintf("%s%s (%.3f)",
  formatC(f, digits = 3, format = "f"),
  ifelse(p < 0.01, "***", ifelse(p < 0.05, "**", ifelse(p < 0.10, "*", ""))), p)
d4 <- grs_results |>
  filter(model %in% FULL_ORDER) |>
  mutate(cell = grs_cell(grs_f, grs_p),
         strategy = factor(str_to_title(strategy),
                           levels = c("Static","Direction","Scaled","Volmanaged")),
         model = factor(model, levels = intersect(FULL_ORDER, model))) |>
  select(model, strategy, cell) |>
  pivot_wider(names_from = strategy, values_from = cell) |>
  arrange(model) |>
  mutate(Model = lbl(as.character(model))) |>
  select(Model, Static, Direction, Scaled, `Vol-managed` = Volmanaged)
kbl_d4 <- d4 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE, longtable = TRUE,
      caption = paste0(
        "GRS \\citep{GibbonsRossShanken1989} $F$-statistics (with $p$-values in ",
        "parentheses) for jointly zero Fama--French five-factor alphas across the ",
        "six characteristic portfolios, all model variants and four strategies. ",
        "Extends Table~\\ref{tab:res-grs}. The static column is identical across ",
        "models. Vol-managed is evaluated on its post-burn-in window; all others on ",
        "the common sample."),
      caption.short = "GRS tests, all model variants and strategies",
      label = "app-grs-full", align = c("l", rep("r", 4))) |>
  kable_styling(latex_options = c("repeat_header"), font_size = 8) |>
  footnote(general = "*** p<0.01, ** p<0.05, * p<0.10.", general_title = "",
           footnote_as_chunk = TRUE, escape = FALSE)
save_app(kbl_d4, "d4_grs_full.tex")


# ==============================================================================
# ==============================================================================
# D.5 — Quarterly-horizon (H=3) portfolios, ALL models run at that horizon
# ==============================================================================
# Addresses the cherry-picking reading: the main text reports H=3 portfolios for
# the two LASSO variants only. The full H=3 model set is reported here.
d5 <- h3_vs_h1 |>
  filter(model %in% FULL_ORDER) |>
  mutate(model = factor(model, levels = intersect(FULL_ORDER, model))) |>
  arrange(model) |>
  transmute(
    Model          = lbl(as.character(model)),
    `SR ($H=1$)`   = num(sharpe_h1,  3),
    `SR ($H=3$)`   = num(sharpe_h3q, 3),
    `$\\Delta$ SR` = num(sharpe_h3q - sharpe_h1, 3),
    `$\\alpha$ (\\%)` = num(alpha_h3q, 2),
    `$t(\\alpha)$` = num(t_h3q, 2),
    `$N$`          = n_months
  )
kbl_d5 <- d5 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE,
      caption = paste0(
        "Quarterly-horizon ($H=3$) direction-timed combined portfolio for every ",
        "model run at that horizon, against the corresponding monthly ($H=1$) ",
        "Sharpe ratio. Positions are set from the three-month forecast and held ",
        "for the quarter, so the rebalancing frequency matches the forecast ",
        "horizon. Every model is scored on the same 414 held months: a quarter is kept only if all models forecast at its rebalancing date. $\\alpha$ is the annualised Fama--French five-factor alpha with ",
        "Newey--West (6 lags) $t$-statistics. Extends the two-model exhibit of ",
        "Section~\\ref{sec:res-robust}."),
      caption.short = "Quarterly-horizon direction-timed portfolio, all model variants",
      label = "app-h3-portfolio", align = c("l", rep("r", 6))) |>
  kable_styling(latex_options = c("HOLD_position"), font_size = 9)
save_app(kbl_d5, "d5_h3_portfolio.tex")


# ==============================================================================
# D.6 — Ledoit-Wolf Sharpe-difference tests and design resolution
# ==============================================================================
d6 <- sharpe_tests |>
  mutate(model = factor(model, levels = intersect(FULL_ORDER, unique(model)))) |>
  arrange(strategy, model) |>
  transmute(
    Model            = lbl(as.character(model)),
    Strategy         = str_to_title(strategy),
    `SR (timed)`     = num(sharpe_timed, 3),
    `SR (static)`    = num(sharpe_static, 3),
    `$\\Delta$ SR`   = num(diff_annual, 3),
    `SE`             = num(se_annual, 3),
    `$p$ (boot)`     = num(p_boot, 3),
    `$p$ (HAC)`      = num(p_hac, 3),
    `MDE`            = num(mde_annual, 3)
  )
kbl_d6 <- d6 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE,
      caption = paste0(
        "\\citet[p.~853]{LedoitWolf2008} tests of the Sharpe-ratio difference ",
        "between each timed strategy and the static benchmark on the combined ",
        "six-characteristic book, common 414-month window, gross of costs. ",
        "$p$ (boot) is the studentised circular block bootstrap ($B=4999$, block ",
        "length 6) that \\citet{LedoitWolf2008} recommend; $p$ (HAC) is the ",
        "delta-method cross-check using a prewhitened Parzen-kernel long-run ",
        "covariance. MDE is the minimum detectable effect: the smallest true ",
        "annualised Sharpe gap the design could reject at 80\\% power and 5\\% ",
        "size. It is a property of the design, not of the estimate, and is ",
        "reported in place of post-hoc power \\citep{HoenigHeisey2001}."),
      caption.short = "Ledoit--Wolf tests of Sharpe-ratio differences versus the static benchmark",
      label = "app-sharpe-tests", align = c("l", "l", rep("r", 7))) |>
  kable_styling(latex_options = c("HOLD_position"), font_size = 8)
save_app(kbl_d6, "d6_sharpe_tests.tex")


# ==============================================================================
# D.7 — Sub-period decomposition of the portfolio results
# ==============================================================================
d7 <- subperiod_eval |>
  filter(strategy %in% c("static", "direction"), sub_period != "Full") |>
  mutate(Strategy = ifelse(strategy == "static", "Static", "Direction"),
         Model    = ifelse(strategy == "static", "-", lbl(model))) |>
  arrange(sub_period, Strategy, desc(sharpe)) |>
  transmute(
    # display the range with an en-dash; the data key keeps its hyphen
    Period            = sub("^2000-2010$", "2000--2010", as.character(sub_period)),
    Strategy, Model,
    `Sharpe`          = num(sharpe, 3),
    `Vol (\\%)`       = num(vol * 100, 1),
    `$\\alpha$ (\\%)` = num(alpha_pct, 2),
    `$t(\\alpha)$`    = num(t_alpha, 2),
    `$N$`             = n_months
  )
kbl_d7 <- d7 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE, longtable = TRUE,
      caption = paste0(
        "Sub-period decomposition of the combined six-characteristic portfolio, ",
        "static versus direction timing, on the same pre-2000 / 2000--2010 / ",
        "post-2010 cuts used for the out-of-sample $R^2$. $\\alpha$ is the ",
        "annualised Fama--French five-factor alpha with Newey--West (6 lags) ",
        "$t$-statistics. The static benchmark is identical across models and is ",
        "therefore reported once per period."),
      caption.short = "Sub-period decomposition of the combined portfolio",
      label = "app-subperiod-portfolio", align = c("l", "l", "l", rep("r", 5))) |>
  kable_styling(latex_options = c("repeat_header"), font_size = 8)
save_app(kbl_d7, "d7_subperiod_portfolio.tex")


# ==============================================================================
# D.9 — RAFE under the traded-asset covariance (robustness)
# ==============================================================================
# The headline RAFE weights gamma forecast errors by the covariance of the
# realised gammas; the strategies actually trade the six long-short books. This
# table re-scores every model with the LS covariance so the ranking can be shown
# not to depend on which of the two matrices supplies the metric. (d8 is taken
# by the value-weighted-leg robustness exported from 05b_vw_robustness.R.)
# The caption names the RAFE winner under each matrix. The two picks are
# separate slice_min() calls in 05 and are NOT guaranteed to agree -- which is
# precisely what this exhibit probes -- so the phrase is built conditionally
# rather than asserting "under both". best_model_gamma is exported by 05; older
# portfolio_results.rds vintages predate it, so recompute from rafe_results.
if (!exists("best_model_gamma")) {
  best_model_gamma <- rafe_results |>
    filter(model != "hist_mean") |>
    slice_min(RAFE, n = 1) |>
    pull(model)
}
best_rafe_phrase <- if (identical(best_model_gamma, best_model_ls)) {
  sprintf("the lowest-RAFE model is %s under both", lbl(best_model_ls))
} else {
  sprintf(paste0("the lowest-RAFE model is %s under $\\Sigma_{\\gamma}$ but ",
                 "%s under $\\Sigma_{LS}$"),
          lbl(best_model_gamma), lbl(best_model_ls))
}

d9 <- rafe_ls_compare |>
  arrange(rank_gamma) |>
  transmute(
    Model = lbl(model),
    `RAFE ($\\Sigma_{\\gamma}$)`  = num(RAFE, 3),    `Rank` = rank_gamma,
    `RAFE ($\\Sigma_{LS}$)`       = num(RAFE_LS, 3), `Rank ` = rank_ls,
    `$\\Delta$ rank` = rank_diff
  )
kbl_d9 <- d9 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE, longtable = TRUE,
      caption = sprintf(paste0(
        "Risk-adjusted mean forecast error under two weighting matrices, all ",
        "model variants, common 414-month sample. $\\Sigma_{\\gamma}$ is the ",
        "covariance of the realised gamma vector, the weighting used in ",
        "Table~\\ref{tab:res-rafe}; $\\Sigma_{LS}$ is the covariance of the six ",
        "long-short quintile books that the timing strategies actually trade, ",
        "estimated on the same months. Forecast errors are gamma errors ",
        "throughout, so only the metric of the quadratic form changes; the two ",
        "RAFE columns are therefore not comparable in level, only in ranking. ",
        "The Spearman rank correlation between the two orderings is %.3f and ",
        "%s. Supports Section~\\ref{sec:res-rafe}."),
        rho_rafe_ls, best_rafe_phrase),
      caption.short = "RAFE under the traded-asset covariance matrix",
      label = "app-rafe-ls", align = c("l", rep("r", 5))) |>
  kable_styling(latex_options = c("repeat_header"), font_size = 9)
save_app(kbl_d9, "d9_rafe_sigma_ls.tex")


# ==============================================================================
# D.10 — Gamma / traded-leg translation check
# ==============================================================================
# Quantifies the step between the forecast object (the FM slope) and the traded
# object (the q5-q1 quintile spread), which neither the OOS R² nor the RAFE
# tables can see.
gamma_col_labels <- c(gamma_bm = "Value", gamma_mom12m = "Momentum",
                      gamma_oper_prof = "Profitability",
                      gamma_asset_growth = "Asset Growth",
                      gamma_size = "Size", gamma_beta = "Beta")
d10 <- sign_agreement |>
  mutate(gamma = factor(gamma_col_labels[gamma], levels = char_order)) |>
  arrange(gamma) |>
  transmute(
    Characteristic = gamma,
    `$\\rho(\\gamma, LS)$` = num(cor_gamma_ls, 3),
    `All` = num(agree_all, 1),
    `$|\\gamma|$ above median` = num(agree_big, 1),
    `$|\\gamma|$ below median` = num(agree_small, 1),
    `$N$` = n
  )
kbl_d10 <- d10 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE,
      caption = paste0(
        "Translation from the forecast object to the traded object, common ",
        "414-month sample. $\\rho(\\gamma, LS)$ is the correlation between the ",
        "realised Fama--MacBeth slope and the raw q5$-$q1 quintile spread on the ",
        "same characteristic; both are stated in the raw-characteristic ",
        "orientation, so no sign convention is undone. The three agreement ",
        "columns give the percentage of months in which the two realisations ",
        "share a sign: over all months, and split at the median of ",
        "$|\\gamma_{k,t}|$. Direction timing earns the premium only in months ",
        "where the two agree, so the split shows where the translation loses ",
        "least. Supports Section~\\ref{sec:res-econ}."),
      caption.short = "Gamma / traded-leg translation check",
      label = "app-signagree", align = c("l", rep("r", 5))) |>
  # HOLD_position -> [H]: this table was promoted into Results.tex (2026-08-30),
  # where the house convention is [H] so floats cannot drift from their
  # reference. Without a specifier it defaulted to [tbp] and floated away.
  kable_styling(latex_options = c("HOLD_position"), font_size = 9)
save_app(kbl_d10, "d10_sign_agreement.tex")


# ==============================================================================
# D.11 — Five- versus six-factor alphas
# ==============================================================================
# The combined book trades a momentum leg, and the FF5 contains no momentum
# factor, so part of any FF5 alpha is a loading on UMD rather than timing skill.
# This table reports both alphas and the momentum loading that separates them.
d11_static <- port_eval |>
  filter(char_col == "COMBINED", strategy == "static", model == "hist_mean") |>
  transmute(Model = "Static (any model)",
            alpha_pct, p_alpha, t_alpha, alpha6_pct, p_alpha6, t_alpha6, b_umd)
d11 <- port_eval |>
  filter(char_col == "COMBINED", strategy == "direction", model %in% FULL_ORDER) |>
  mutate(model = factor(model, levels = intersect(FULL_ORDER, model))) |>
  arrange(model) |>
  transmute(Model = lbl(as.character(model)),
            alpha_pct, p_alpha, t_alpha, alpha6_pct, p_alpha6, t_alpha6, b_umd) |>
  bind_rows(d11_static, y = _) |>
  transmute(
    Model,
    `$\\alpha_{5}$ (\\%)` = paste0(num(alpha_pct), star(p_alpha)),
    `$t$` = num(t_alpha),
    `$\\alpha_{6}$ (\\%)` = paste0(num(alpha6_pct), star(p_alpha6)),
    `$t$ ` = num(t_alpha6),
    `$\\beta_{\\text{UMD}}$` = num(b_umd)
  )
kbl_d11 <- d11 |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE, longtable = TRUE,
      caption = paste0(
        "Five- and six-factor alphas of the combined six-characteristic book, ",
        "common 414-month sample, gross of costs. $\\alpha_{5}$ is the ",
        "annualised Fama--French five-factor alpha reported throughout the ",
        "chapter; $\\alpha_{6}$ adds the momentum factor of the French data ",
        "library, giving the six-factor model of \\citet{FamaFrench2018}, and ",
        "$\\beta_{\\text{UMD}}$ is the resulting momentum loading. All ",
        "$t$-statistics use Newey--West (6 lags) standard errors. The first row ",
        "is the static benchmark, which is model-independent. Supports ",
        "Section~\\ref{sec:res-econ}."),
      caption.short = "Five- versus six-factor alphas",
      label = "app-ff6", align = c("l", rep("r", 5))) |>
  kable_styling(latex_options = c("repeat_header"), font_size = 9) |>
  footnote(general = "*** p<0.01, ** p<0.05, * p<0.10.", general_title = "",
           footnote_as_chunk = TRUE, escape = FALSE)
save_app(kbl_d11, "d11_ff6_alphas.tex")


# ==============================================================================
# Sync into the Overleaf project (self-contained; mirrors 06's block)
# ==============================================================================
thesis_tables_dir <- "thesis_drafts/tables"
if (dir.exists("thesis_drafts")) {
  dir.create(thesis_tables_dir, showWarnings = FALSE)
  app_files <- list.files("tables", pattern = "^[abcd][0-9].*\\.tex$", full.names = TRUE)
  copied <- file.copy(app_files, thesis_tables_dir, overwrite = TRUE)
  cat(sprintf("\nSynced %d/%d appendix table files to %s/\n",
              sum(copied), length(app_files), thesis_tables_dir))
} else cat("\nthesis_drafts/ not found — skipped Overleaf sync.\n")

cat("\n=== Appendix tables export complete ===\n")
