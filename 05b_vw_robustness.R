# ==============================================================================
# 05b_vw_robustness.R — Value-weighted quintile-leg robustness (critique C8(i))
# ------------------------------------------------------------------------------
# The headline portfolios (05_portfolio_construction.R) form each quintile leg
# by EQUAL-weighting its stocks, while the gammas that generate the timing
# signal are estimated by WLS with lagged market cap (a value-weighted
# cross-section). The two objects estimate the same premium but weight stocks
# differently, so equal-weighted legs sit slightly at odds with the estimation.
# This script closes that mismatch: it rebuilds the long-short legs with
# VALUE weights (lagged market cap — the identical weight used in the WLS
# Fama-MacBeth regressions) and re-computes the direction-timing result, so the
# portfolio weighting and the estimation weighting coincide.
#
# Equal-weighted legs remain the headline (comparable to the rest of the
# thesis); the value-weighted book is reported alongside as the robustness
# exhibit that shows the direction-timing conclusion is not an artefact of the
# equal-weighting choice. Everything else — the sign-timing rule, the combined
# equal-weight-across-characteristics book, the common 414-month window, the
# Newey-West FF5 alpha — is held identical to 05.
#
# Standalone: reads panel_clean / gamma_predictions / ff5_factors and does NOT
# touch portfolio_results.rds or any headline output.
# Output: vw_robustness.rds, tables/d8_vw_robustness.tex
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse); library(sandwich); library(lmtest); library(slider)
})

panel_clean <- readRDS("panel_clean.rds")
list2env(readRDS("gamma_predictions.rds"), envir = environment())  # pred_list, dates_oos, ...
ff5 <- readRDS("ff5_factors.rds")

char_gamma_map <- tribble(
  ~char_col,            ~gamma_col,
  "bm_std",             "gamma_bm",
  "mom12m_std",         "gamma_mom12m",
  "oper_prof_std",      "gamma_oper_prof",
  "asset_growth_std",   "gamma_asset_growth",
  "log_mktcap_lag_std", "gamma_size",
  "beta_std",           "gamma_beta")
chars_std <- char_gamma_map$char_col

# ── Common 414-month evaluation window (identical to 05) ──────────────────────
common_eval  <- Reduce(`&`, lapply(names(pred_list), function(gc)
  rowSums(is.na(pred_list[[gc]])) == 0 & !is.na(actuals[, gc])))
common_dates <- dates_oos[common_eval]

# ── Long-short quintile legs: equal-weighted AND value-weighted ───────────────
# EW leg  : mean(ret_excess)                         (headline)
# VW leg  : weighted.mean(ret_excess, w = mktcap_lag) (matches WLS estimation)
legs <- panel_clean |>
  filter(date %in% dates_oos) |>
  group_by(date) |>
  mutate(across(all_of(chars_std), \(x) ntile(x, 5), .names = "{.col}_q")) |>
  ungroup() |>
  pivot_longer(ends_with("_q"), names_to = "char_col", values_to = "quintile") |>
  mutate(char_col = str_remove(char_col, "_q")) |>
  filter(quintile %in% c(1L, 5L)) |>
  group_by(date, char_col, quintile) |>
  summarise(ew = mean(ret_excess, na.rm = TRUE),
            vw = weighted.mean(ret_excess, w = mktcap_lag, na.rm = TRUE),
            .groups = "drop") |>
  pivot_wider(names_from = quintile, values_from = c(ew, vw), names_sep = "_q") |>
  mutate(ls_ew = ew_q5 - ew_q1,
         ls_vw = vw_q5 - vw_q1) |>
  select(date, char_col, ls_ew, ls_vw)

# ── Direction / static returns for a given model, under a given weighting ─────
sharpe_annual <- function(r) { r <- r[!is.na(r)]
  if (length(r) < 12) return(NA_real_); mean(r) / sd(r) * sqrt(12) }
ff5_alpha <- function(r, dates) {
  df <- tibble(date = dates, ret = r) |> inner_join(ff5, by = "date") |> filter(!is.na(ret))
  if (nrow(df) < 24) return(tibble(alpha = NA, t_alpha = NA))
  ct <- coeftest(lm(ret ~ mkt_excess + smb + hml + rmw + cma, data = df),
                 vcov = NeweyWest(lm(ret ~ mkt_excess + smb + hml + rmw + cma, data = df),
                                  lag = 6, prewhite = FALSE))
  tibble(alpha = ct["(Intercept)", "Estimate"] * 1200,
         t_alpha = ct["(Intercept)", "t value"])
}

# static direction: frozen expanding-mean sign at evaluation start (as in 05)
idx_start    <- which(dates_oos == min(common_dates))[1]
static_gamma <- vapply(names(pred_list), \(gc) pred_list[[gc]][idx_start, "hist_mean"], numeric(1))
names(static_gamma) <- char_gamma_map$char_col[match(names(static_gamma), char_gamma_map$gamma_col)]

# Build the combined equal-weight-across-characteristics book for one model,
# separately under EW and VW legs and under static vs direction timing.
combined_book <- function(model, weighting) {
  ls_col <- if (weighting == "ew") "ls_ew" else "ls_vw"
  map_dfr(chars_std, function(cc) {
    gc  <- char_gamma_map$gamma_col[char_gamma_map$char_col == cc]
    pg  <- tibble(date = dates_oos, pred = pred_list[[gc]][, model])
    legs |> filter(char_col == cc) |>
      inner_join(pg, by = "date") |>
      transmute(date, char_col = cc,
                ls = .data[[ls_col]],
                ret_static    = sign(static_gamma[[cc]]) * ls,
                ret_direction = sign(pred) * ls)
  }) |>
    filter(date %in% common_dates) |>
    group_by(date) |>
    summarise(static = mean(ret_static, na.rm = TRUE),
              direction = mean(ret_direction, na.rm = TRUE), .groups = "drop")
}

models <- c(comb_xgb = "Comb+XGB", rf = "Random forest",
            lstm_r1 = "LSTM-1m", comb = "Combination")

results <- map_dfr(names(models), function(m) {
  map_dfr(c("ew", "vw"), function(w) {
    cb <- combined_book(m, w)
    map_dfr(c("static", "direction"), function(s) {
      r <- cb[[s]]; a <- ff5_alpha(r, cb$date)
      tibble(model = models[[m]], weighting = toupper(w), strategy = s,
             sharpe = round(sharpe_annual(r), 3),
             alpha_pct = round(a$alpha, 2), t_alpha = round(a$t_alpha, 2),
             n = sum(!is.na(r)))
    })
  })
})

saveRDS(results, "vw_robustness.rds")
cat("=== Combined book: equal- vs value-weighted legs ===\n")
print(as.data.frame(results), row.names = FALSE)

# ── Table d8 ──────────────────────────────────────────────────────────────────
# Static is model-independent within a weighting; report it once, then the
# direction-timed Sharpe/alpha for each model under EW and VW.
stat <- results |> filter(strategy == "static") |> distinct(weighting, sharpe, alpha_pct, t_alpha)
dir  <- results |> filter(strategy == "direction")

wide <- dir |>
  select(model, weighting, sharpe, alpha_pct, t_alpha) |>
  pivot_wider(names_from = weighting, values_from = c(sharpe, alpha_pct, t_alpha))

fmt_a <- function(a, t) sprintf("%.2f\\,(%.2f)", a, t)
rows <- c(
  sprintf("Static (any model) & %.2f & %s & %.2f & %s\\\\",
          stat$sharpe[stat$weighting=="EW"], fmt_a(stat$alpha_pct[stat$weighting=="EW"], stat$t_alpha[stat$weighting=="EW"]),
          stat$sharpe[stat$weighting=="VW"], fmt_a(stat$alpha_pct[stat$weighting=="VW"], stat$t_alpha[stat$weighting=="VW"])),
  "\\addlinespace",
  vapply(seq_len(nrow(wide)), function(i) sprintf(
    "%s & %.2f & %s & %.2f & %s\\\\", wide$model[i],
    wide$sharpe_EW[i], fmt_a(wide$alpha_pct_EW[i], wide$t_alpha_EW[i]),
    wide$sharpe_VW[i], fmt_a(wide$alpha_pct_VW[i], wide$t_alpha_VW[i])), character(1))
)

tex <- c(
  "\\begin{table}[H]",   # pinned, consistent with the other appendix floats "\\centering",
  # Short caption (bracketed) keeps the List of Tables readable.
  "\\caption[Value-weighted-leg robustness of direction timing]",
  "{Value-weighted-leg robustness of direction timing. The combined",
  "six-characteristic long-short book, common 414-month sample, with each",
  "quintile leg formed by equal weighting (EW, the headline) and by value",
  "weighting with lagged market cap (VW - the same weight used in the WLS",
  "Fama--MacBeth estimation). Sharpe is annualised; $\\alpha$ is the annualised",
  "Fama--French five-factor alpha in \\% with Newey--West (6 lags) $t$ in",
  "parentheses. Static is the frozen-sign benchmark (model-independent).}",
  "\\label{tab:app-vw}", "\\fontsize{10}{12}\\selectfont",
  "\\begin{tabular}{lrrrr}", "\\toprule",
  " & \\multicolumn{2}{c}{Equal-weighted} & \\multicolumn{2}{c}{Value-weighted}\\\\",
  "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
  "Timing (direction) & Sharpe & $\\alpha$ ($t$) & Sharpe & $\\alpha$ ($t$)\\\\",
  "\\midrule", rows, "\\bottomrule", "\\end{tabular}", "\\end{table}")

writeLines(tex, "tables/d8_vw_robustness.tex")
if (dir.exists("thesis_drafts/tables"))
  file.copy("tables/d8_vw_robustness.tex", "thesis_drafts/tables/d8_vw_robustness.tex", overwrite = TRUE)
cat("\nSaved: vw_robustness.rds, tables/d8_vw_robustness.tex\n")
