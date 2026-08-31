# ==============================================================================
# 04h_tail_robustness.R — Fat-tail / crisis-month robustness of OOS R² (C7)
# ------------------------------------------------------------------------------
# Single-month gammas reach +/-40-54%, and because the historical-mean benchmark
# forecasts near zero, a handful of crisis months dominate its squared-error
# sum. An out-of-sample R^2 could therefore be inflated by the model merely
# leaning toward those few extreme realisations. This script quantifies that
# exposure two ways, for every model on the common 414-month sample:
#
#   (1) Winsorised target — the realised gamma is capped at its 1st/99th
#       percentile before the squared errors are formed (both the model and the
#       benchmark error use the capped realisation). This extends the Huber-loss
#       logic already used in network TRAINING to the EVALUATION step, without
#       any arbitrary date selection.
#   (2) Crisis-excluded — the dot-com window (1998-2002) and the global
#       financial crisis (2008-2009) are dropped from the R^2 sums entirely,
#       exactly the periods the critique names.
#
# A size (or beta) result that survives both is not a crisis artefact.
#
# Predictions are left untouched throughout — only the evaluation target is
# tamed — so this is a pure robustness recomputation, not a re-run.
# Input : gamma_predictions.rds   Output: tail_robustness.{rds,csv}, tables/b8_tail.tex
# ==============================================================================

suppressPackageStartupMessages({library(tidyverse); library(sandwich); library(lmtest)})

gp <- readRDS("gamma_predictions.rds")
list2env(gp, envir = environment())   # pred_list, actuals, dates_oos, ...

gamma_cols <- c("gamma_bm","gamma_mom12m","gamma_oper_prof",
                "gamma_asset_growth","gamma_size","gamma_beta")
gamma_labels <- c(gamma_bm="Value", gamma_mom12m="Momentum", gamma_oper_prof="Profitability",
                  gamma_asset_growth="Asset Growth", gamma_size="Size", gamma_beta="Beta")

report_models <- c("lasso","rf","xgb","comb_xgb","comb","mlp","lstm")
model_labels  <- c(lasso="LASSO", rf="Random forest", xgb="XGBoost",
                   comb_xgb="Comb+XGB", comb="Combination", mlp="MLP", lstm="LSTM")

# Crisis windows named in the critique
in_crisis <- function(d) (d >= as.Date("1998-01-01") & d <= as.Date("2002-12-31")) |
                         (d >= as.Date("2008-01-01") & d <= as.Date("2009-12-31"))

oos_r2 <- function(y, yhat, bench)
  1 - sum((y - yhat)^2, na.rm = TRUE) / sum((y - bench)^2, na.rm = TRUE)

clark_west <- function(y, bench, model, nw = 12L) {
  f <- (y - bench)^2 - ((y - model)^2 - (bench - model)^2)
  f <- f[!is.na(f)]; if (length(f) < 10L) return(NA_real_)
  v <- tryCatch(as.numeric(NeweyWest(lm(f ~ 1), lag = nw, prewhite = FALSE, adjust = TRUE)),
                error = function(e) var(f)/length(f))
  if (!is.finite(v) || v <= 0) v <- var(f)/length(f)
  round(1 - pnorm(mean(f)/sqrt(v)), 3)
}

winsor <- function(y, p = 0.01) {
  q <- quantile(y, c(p, 1 - p), na.rm = TRUE)
  pmin(pmax(y, q[1]), q[2])
}

results <- map_dfr(gamma_cols, function(gc) {
  mask  <- rowSums(is.na(pred_list[[gc]])) == 0 & !is.na(actuals[, gc])
  idx   <- which(mask)
  y     <- actuals[idx, gc]
  dts   <- dates_oos[idx]
  bench <- pred_list[[gc]][idx, "hist_mean"]
  y_w   <- winsor(y)                      # winsorised realised gamma
  keep  <- !in_crisis(dts)                # crisis-excluded mask
  map_dfr(report_models, function(m) {
    if (!m %in% colnames(pred_list[[gc]])) return(NULL)
    yhat <- pred_list[[gc]][idx, m]
    ok   <- !is.na(yhat)
    # Only the REALISED gamma is winsorised; all forecasts (model and benchmark)
    # are left untouched — capping a forecast would alter the object being
    # evaluated rather than the evaluation target.
    tibble(
      gamma = gamma_labels[gc], model = model_labels[m],
      base       = round(100 * oos_r2(y[ok],   yhat[ok], bench[ok]),  2),
      winsor     = round(100 * oos_r2(y_w[ok], yhat[ok], bench[ok]), 2),
      no_crisis  = round(100 * oos_r2(y[ok & keep], yhat[ok & keep], bench[ok & keep]), 2),
      p_base     = clark_west(y[ok], bench[ok], yhat[ok]),
      p_winsor   = clark_west(y_w[ok], bench[ok], yhat[ok]),
      p_nocrisis = clark_west(y[ok & keep], bench[ok & keep], yhat[ok & keep])
    )
  })
})

# How much of the benchmark's squared-error mass sits in the crisis months?
crisis_share <- map_dfr(c("gamma_size","gamma_beta"), function(gc) {
  mask <- rowSums(is.na(pred_list[[gc]])) == 0 & !is.na(actuals[, gc])
  idx  <- which(mask); y <- actuals[idx, gc]; dts <- dates_oos[idx]
  bse  <- (y - pred_list[[gc]][idx, "hist_mean"])^2
  tibble(gamma = gamma_labels[gc],
         crisis_months = sum(in_crisis(dts)),
         pct_bench_sse = round(100 * sum(bse[in_crisis(dts)]) / sum(bse), 1))
})

saveRDS(list(results = results, crisis_share = crisis_share), "tail_robustness.rds")
write_csv(results, "tail_robustness.csv")

cat("=== OOS R^2 under baseline / winsorised target / crisis-excluded ===\n")
print(as.data.frame(results |> filter(gamma %in% c("Size","Beta"))), row.names = FALSE)
cat("\n=== Crisis months' share of benchmark squared-error mass ===\n")
print(as.data.frame(crisis_share), row.names = FALSE)

# ── Table b8 (Size + Beta) ────────────────────────────────────────────────────
star <- function(p) ifelse(is.na(p), "", ifelse(p<0.01,"^{***}", ifelse(p<0.05,"^{**}", ifelse(p<0.10,"^{*}",""))))
cell <- function(r2, p) sprintf("$%.2f%s$", r2, star(p))
tab <- results |> filter(gamma %in% c("Size","Beta")) |>
  mutate(gamma = factor(gamma, levels = c("Size","Beta")),
         B = cell(base, p_base), W = cell(winsor, p_winsor), N = cell(no_crisis, p_nocrisis)) |>
  arrange(gamma) |>
  select(gamma, model, B, W, N)
rows <- tab |> group_split(gamma) |> map(function(d) {
  c(sprintf("\\addlinespace[0.2em]\\multicolumn{4}{l}{\\textbf{%s}}\\\\", d$gamma[1]),
    sprintf("\\hspace{1em}%s & %s & %s & %s\\\\", d$model, d$B, d$W, d$N))
}) |> unlist()
tex <- c("\\begin{table}[H]", "\\centering",   # pinned: [!h] let this float drift out of Annex B
  # Short caption (bracketed) keeps the List of Tables readable.
  "\\caption[Fat-tail robustness of the size and beta out-of-sample $R^2$]",
  "{Fat-tail robustness of the size and beta out-of-sample $R^2$ (\\%). ",
  "Baseline is the headline figure; Winsorised caps the realised gamma ",
  "at its 1st/99th percentile before the squared errors are formed (extending the ",
  "networks' Huber training loss to the evaluation step); Crisis-excl.\\ drops ",
  "the 1998--2002 and 2008--2009 windows from the sums. Predictions are unchanged; ",
  "only the evaluation target is tamed. Stars from the one-sided Clark--West test ",
  "($^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$).}",
  "\\label{tab:app-tail}", "\\fontsize{10}{12}\\selectfont",
  "\\begin{tabular}{lrrr}", "\\toprule",
  "Model & Baseline & Winsorised & Crisis-excl.\\\\", "\\midrule", rows,
  "\\bottomrule", "\\end{tabular}", "\\end{table}")
writeLines(tex, "tables/b8_tail.tex")
if (dir.exists("thesis_drafts/tables"))
  file.copy("tables/b8_tail.tex", "thesis_drafts/tables/b8_tail.tex", overwrite = TRUE)
cat("\nSaved: tail_robustness.{rds,csv}, tables/b8_tail.tex\n")
