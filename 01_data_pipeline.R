# ==============================================================================
# 01_data_pipeline.R
# Loads stock returns / market cap from tidy_finance SQLite (CRSP) and the five
# accounting/return-based cross-sectional characteristics from the Chen-Zimmermann
# (2020) Open Source Asset Pricing signed firm-level predictors, merges them, and
# prepares a clean panel for Fama-MacBeth cross-sectional regressions.
# (Size is built from CRSP market cap; CZ excludes Size by design.)
#
# Data-cleaning protocol (per thesis cleaning specification):
#   Phase 1 – Delisting / survivorship bias   (handled by Tidy Finance DB)
#   Phase 2 – Deduplication & universe definition (sector exclusion)
#   Phase 3 – Outlier treatment (winsorising at 1%/99%)
#   Phase 4 – Look-ahead alignment (t-1 lagging of all characteristics)
#   Phase 5 – Variable transformation (rank-standardisation, log-size)
#   Diagnostics – NA audit per variable/year, cross-section size check
#
# NOTE — Alderson dividend filter (fake dividend cuts): requires quarterly
#   Compustat dividend history, which is not part of this pipeline's data
#   sources. Excluded from implementation; document this limitation in thesis.
#
# Output: panel_clean.rds   (object: panel_clean)
#         ff5_factors.rds   (object: ff5_factors)
#         char_spreads.rds  (object: char_spreads)
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(RSQLite)
library(dbplyr)
library(lubridate)

db_path <- "/home/shared/data/tidy_finance.sqlite"


# ── Phase 1: Delisting / Survivorship Bias ─────────────────────────────────────
# Delisting returns are already folded into ret_excess by the Tidy Finance
# SQLite build; nothing is imputed here. The rule Tidy Finance implements is:
#   dlret present                      -> use dlret
#   performance-related delisting with
#   missing dlret (codes 500, 520,
#   551-574, 580, 584)                 -> -30%   (Shumway 1997, JF)
#   any other delisting, no return     -> -100%
# The -30% substitution is applied UNIFORMLY across exchanges. It is NOT
# exchange-specific: Shumway & Warther (1999, JF) estimate roughly -55% for
# Nasdaq, and applying -30% there understates the loss on delisted Nasdaq
# firms. Those firms are disproportionately small, so the size premium
# estimated in 02_fama_macbeth.R is, in magnitude, an upper rather than a
# lower estimate. Disclosed in Methodology section "Data and Sample
# Construction"; see also the sign convention note there.
#
# NB: an earlier version of this comment claimed a -30%/-55% exchange split
# was applied upstream. That was wrong — no such split exists in the Tidy
# Finance build, and the thesis text now documents the uniform rule.
# Reference: Shumway (1997, JF); Shumway & Warther (1999, JF)
cat("Phase 1: Delisting returns inherited from Tidy Finance",
    "(uniform -30% substitution, Shumway 1997).\n")


# ── Phase 2a: Load raw data ────────────────────────────────────────────────────
tidy_finance <- dbConnect(SQLite(), db_path, extended_types = TRUE)

# Characteristics from the Chen-Zimmermann Open Source Asset Pricing project
# (signed firm-level predictors, wide CSV — openassetpricing.com/data). Replaces
# the former crsp_monthly_signal source with an independently constructed, fully
# documented set: each signal maps to a published paper via SignalDoc.csv.
# Returns / market cap / industry still come from CRSP (stock_data, below).
#
# CZ values are SIGN-ALIGNED so a higher signal implies a higher expected return,
# i.e. signed = raw x Sign  (SignalDoc "Sign" column). To preserve the raw-
# characteristic sign convention used throughout the thesis we undo this:
#   raw = signed x Sign.  Only AssetGrowth carries Sign = -1 (Cooper, Gulen &
# Schill 2008); BM / Mom12m / OperProf / Beta are +1, so unchanged.
#
# Reads the compact snapshot cz_signals_subset.rds (permno, yyyymm + 5 signal
# columns) produced once, locally, by 00_trim_cz.R from the full ~8 GB CZ file.
# Only the small subset is uploaded to the server.
#
# ⚠️ IMPLICIT SIZE SCREEN — read before interpreting any premium.
# CZ's OperProf.do sets the signal to missing for the SMALLEST market-cap
# tercile each month ("dirty simulation of NYSE size breakpoints", their
# comment), to mimic the FF(2006) portfolio sorts. Combined with the listwise
# deletion below, this removes the bottom size tercile from EVERY monthly
# cross-section. Magnitude: among CZ rows where beta/bm/mom12m/asset_growth
# are all present, 53.7% still lack OperProf; the surviving panel has a median
# of 1,554 stocks/month with a 10th-pct market cap of ~USD 55m. Not a bug, and
# it pulls the same way as the WLS primary spec, but it must be disclosed —
# documented in Methodology §"Cross-Sectional Characteristics".
#
# Source attributions, per the PUBLISHED paper's predictor table (Chen &
# Zimmermann 2022, CFR 11(2), Appendix) — verified 2026-08-04. These are not
# the papers one would guess from the acronyms:
#   BM          Rosenberg, Reid & Lanstein (1985)  "BM using most recent ME";
#                 NOT Fama-French — their December-ME variant is `BMdec` (FF 1992).
#                 Released data computes it as log(ceqt / ME matched at FYE).
#   Mom12m      Jegadeesh & Titman (1993)     ret over t-11..t-1 (skips month t)
#   OperProf    Fama & French (2006, JFE)     (revt-cogs-xsga-xint)/ceq
#   AssetGrowth Cooper, Gulen & Schill (2008) annual growth in `at`
#   Beta        Fama & MacBeth (1973)         60-obs rolling, min 20, vs EW index
# CAUTION: the live SignalDoc.csv has drifted from the published paper — it now
# credits BM to Stattman (1980) per a post-publication GitHub decision. Cite the
# published table, not SignalDoc, for anything that goes in the thesis.
#
# Date shifted +1 month: characteristic at t aligns with return at t+1 (Phase 4).
cz_path <- "cz_signals_subset.rds"           # built by 00_trim_cz.R (run locally)
cz_map  <- c(beta = "Beta", bm = "BM", mom12m = "Mom12m",
             oper_prof = "OperProf", asset_growth = "AssetGrowth")
cz_sign <- c(beta = 1, bm = 1, mom12m = 1, oper_prof = 1, asset_growth = -1)

signal_data <- readRDS(cz_path) |>
  as_tibble() |>
  select(permno, yyyymm, all_of(unname(cz_map))) |>
  rename(all_of(cz_map)) |>                                         # CZ acronym -> pipeline name
  mutate(across(names(cz_sign), \(x) x * cz_sign[cur_column()])) |> # signed -> raw
  mutate(
    year  = yyyymm %/% 100L,
    month = yyyymm %%  100L,
    date  = make_date(year, month, 1L) %m+% months(1L)
  ) |>
  select(-year, -month, -yyyymm)

# Returns, market cap, exchange, and FF12 industry classification.
# `exchange` is carried through for the NYSE-percentile size filter (robustness);
# it is NOT used for delisting, which is uniform across exchanges (Phase 1).
# `industry` needed for sector exclusion (Phase 2c).
stock_data <- tbl(tidy_finance, "crsp_monthly") |>
  select(permno, date, mktcap, ret_excess, exchange, industry) |>
  collect()

dbDisconnect(tidy_finance)
rm(tidy_finance)

cat(sprintf(
  "Raw data loaded: %s stock-month obs | %s signal obs\n",
  format(nrow(stock_data),  big.mark = ","),
  format(nrow(signal_data), big.mark = ",")
))


# ── Phase 2b: Deduplication ────────────────────────────────────────────────────
# Remove duplicate (permno, date) pairs that can arise from Compustat
# restatements or data-vendor processing artefacts.
# Strategy: keep first occurrence after sorting by permno, date (no
# information advantage in choosing last — restatement timing is ambiguous).
n_before <- nrow(stock_data)
stock_data <- stock_data |>
  arrange(permno, date) |>
  distinct(permno, date, .keep_all = TRUE)
cat(sprintf(
  "Deduplication (returns):  %d duplicate permno-date pairs removed.\n",
  n_before - nrow(stock_data)
))

n_before <- nrow(signal_data)
signal_data <- signal_data |>
  arrange(permno, date) |>
  distinct(permno, date, .keep_all = TRUE)
cat(sprintf(
  "Deduplication (signals):  %d duplicate permno-date pairs removed.\n",
  n_before - nrow(signal_data)
))


# ── Phase 2c: Sector exclusion ─────────────────────────────────────────────────
# Exclude Financials and Utilities: regulated leverage and pricing structures
# make their BM ratios and profitability measures non-comparable to industrials,
# producing mechanically different cross-sectional slopes.
# The `industry` column uses the Fama-French 12-industry classification as
# constructed by Tidy Finance (based on CRSP SIC codes).
# Equivalent SIC-code filter:  siccd %in% 6000:6999  |  siccd %in% 4900:4999
# Reference: Fama & French (1992, JF); Lewellen, Nagel & Shanken (2010, JFE)
excluded_industries <- c("Finance", "Utilities")
n_before <- nrow(stock_data)
stock_data <- stock_data |>
  filter(!industry %in% excluded_industries)
cat(sprintf(
  "Sector filter:  %s obs removed (%.1f%%) — Financials & Utilities excluded.\n",
  format(n_before - nrow(stock_data), big.mark = ","),
  100 * (n_before - nrow(stock_data)) / n_before
))


# ── Phase 4: Lagged market cap — look-ahead safe size proxy ───────────────────
# lag() within each stock gives mktcap_{t-1}:
#   • mktcap_lag  : raw lagged market cap → WLS weight
#   • log_mktcap_lag : log-size characteristic (Phase 5 transformation)
# First observation per stock is intentionally dropped (lag = NA), avoiding
# any forward-looking contamination from price adjustments at listing.
# Reference: Fama & French (1992, JF) — size measured at June of year t
#   applied to returns July t through June t+1 (analogous spirit here)
stock_data <- stock_data |>
  arrange(permno, date) |>
  group_by(permno) |>
  mutate(
    mktcap_lag     = lag(mktcap),
    log_mktcap_lag = log(mktcap_lag)
  ) |>
  ungroup()


# ── Merge panel ────────────────────────────────────────────────────────────────
# Inner join: only (permno, date) pairs present in both tables are kept.
# This implicitly drops stocks without characteristic coverage and vice versa.
panel_data <- signal_data |>
  inner_join(
    stock_data |> select(permno, date, ret_excess, mktcap_lag, log_mktcap_lag),
    by = c("permno", "date")
  )

cat(sprintf("After merge: %s obs\n", format(nrow(panel_data), big.mark = ",")))


# ── Phase 3: Winsorise at 1%/99% each month ───────────────────────────────────
# Preferred over trimming: retains sample size while dampening bad ticks and
# data-entry errors. Applied cross-sectionally within each month.
# Reference: Fama & French (2008, JFE); Lewellen, Nagel & Shanken (2010, JFE)
winsorize <- function(x, probs = c(0.01, 0.99)) {
  b <- quantile(x, probs, na.rm = TRUE)
  pmax(pmin(x, b[2]), b[1])
}

chars <- c("beta", "bm", "mom12m", "oper_prof", "asset_growth", "log_mktcap_lag")

panel_data <- panel_data |>
  group_by(date) |>
  mutate(across(all_of(chars), winsorize)) |>
  ungroup()


# ── NA Diagnostic ─────────────────────────────────────────────────────────────
# Run BEFORE standardisation and final drop to identify structurally sparse
# characteristics (e.g. oper_prof/asset_growth missing in early years).
cat("\n══ NA Audit (post-winsorise, pre-drop) ══════════════════════════════════\n")

audit_vars <- c(chars, "ret_excess", "mktcap_lag")
n_total <- nrow(panel_data)

na_by_var <- panel_data |>
  summarise(across(all_of(audit_vars), \(x) sum(is.na(x)))) |>
  pivot_longer(everything(), names_to = "variable", values_to = "n_na") |>
  mutate(pct_na = round(100 * n_na / n_total, 2)) |>
  arrange(desc(n_na))

cat(sprintf("Total obs before drop: %s\n\n", format(n_total, big.mark = ",")))
print(na_by_var, n = Inf)

# NA in ret_excess by year — reveals structural coverage gaps early in sample
cat("\n── ret_excess NA by year ────────────────────────────────────────────────\n")
panel_data |>
  mutate(year = year(date)) |>
  group_by(year) |>
  summarise(
    n        = n(),
    na_ret   = sum(is.na(ret_excess)),
    pct_na   = round(100 * na_ret / n, 1),
    .groups  = "drop"
  ) |>
  filter(na_ret > 0) |>
  print(n = Inf)

# NA in key characteristics by year — flags when oper_prof/asset_growth became
# available in Compustat (typically mid-1960s through 1970s ramp-up)
cat("\n── oper_prof NA by year (Compustat coverage ramp-up) ───────────────────\n")
panel_data |>
  mutate(year = year(date)) |>
  group_by(year) |>
  summarise(
    n          = n(),
    na_op      = sum(is.na(oper_prof)),
    na_ag      = sum(is.na(asset_growth)),
    pct_op     = round(100 * na_op / n, 1),
    pct_ag     = round(100 * na_ag / n, 1),
    .groups    = "drop"
  ) |>
  filter(na_op > 0 | na_ag > 0) |>
  print(n = Inf)

cat("════════════════════════════════════════════════════════════════════════\n\n")


# ── Monthly characteristic IQR spreads ────────────────────────────────────────
# Computed on winsorised but not yet rank-standardised values so that the
# spread reflects economically meaningful cross-sectional dispersion.
# Used as a predictor in the gamma prediction models (Step 3 feature set).
char_spreads <- panel_data |>
  group_by(date) |>
  summarise(across(
    all_of(chars),
    \(x) IQR(x, na.rm = TRUE),
    .names = "{.col}_spread"
  )) |>
  ungroup()

cat(sprintf("Characteristic spreads: %d months computed.\n", nrow(char_spreads)))


# ── Phase 5: Cross-sectional rank-standardise to [-0.5, 0.5] ─────────────────
# Makes gamma estimates directly comparable in magnitude across characteristics.
# na.last = "keep" propagates NAs so they are caught by the final drop.
rank_std <- function(x) {
  r <- rank(x, ties.method = "average", na.last = "keep")
  n <- sum(!is.na(x))
  (r - 1) / (n - 1) - 0.5
}

panel_data <- panel_data |>
  group_by(date) |>
  mutate(across(all_of(chars), rank_std, .names = "{.col}_std")) |>
  ungroup()


# ── Sample attrition tracker ───────────────────────────────────────────────────
# Records obs count after each filter step for the thesis Sample Selection table.
attrition <- tibble(
  step = character(),
  n_obs = integer(),
  n_dropped = integer(),
  pct_dropped = numeric()
)
record_step <- function(tbl, label, n_prev) {
  n <- nrow(tbl)
  tibble(step = label, n_obs = n,
         n_dropped = n_prev - n,
         pct_dropped = round(100 * (n_prev - n) / n_prev, 1))
}

n_raw <- nrow(panel_data)
attrition <- bind_rows(
  tibble(step = "Raw merge (post-sector filter)", n_obs = n_raw,
         n_dropped = 0L, pct_dropped = 0)
)


# ── Final listwise drop ────────────────────────────────────────────────────────
# Drop rows missing any standardised characteristic, ret_excess, or WLS weight.
# Listwise deletion ensures all six gammas are estimated on the same cross-
# section each month (Option C — consistent sample design).
# Reference: Novy-Marx (2013, JFE); Fama & French (2008, JFE)
chars_std <- paste0(chars, "_std")

panel_clean <- panel_data |>
  filter(
    if_all(all_of(c(chars_std, "ret_excess")), \(x) !is.na(x)),
    !is.na(mktcap_lag),
    mktcap_lag > 0
  )

attrition <- bind_rows(attrition,
  record_step(panel_clean, "Listwise deletion (all 6 chars + ret + mktcap)", n_raw))
n_after_listwise <- nrow(panel_clean)

cat(sprintf(
  "Listwise deletion: %s obs removed (%.1f%%) → %s obs remaining.\n",
  format(n_raw - n_after_listwise, big.mark = ","),
  100 * (n_raw - n_after_listwise) / n_raw,
  format(n_after_listwise, big.mark = ",")
))


# ── Sample start fix: enforce date >= 1963-07-01 ──────────────────────────────
# Jan–Jun 1963 have only 8–36 stocks — far below the 50-stock minimum for
# reliable cross-sectional regression. Dropping these 6 months also aligns
# the sample start with the Fama-French 5-factor series (available from
# 1963-07), eliminating any FF5-alignment issue in downstream steps.
# Reference: Fama & French (1993, JFE) — standard CRSP sample start
SAMPLE_START <- as.Date("1963-07-01")
panel_clean <- panel_clean |> filter(date >= SAMPLE_START)

attrition <- bind_rows(attrition,
  record_step(panel_clean, "Sample start fix (drop pre-1963-07)", n_after_listwise))

cat(sprintf(
  "Sample start fix: dropped %d obs in Jan–Jun 1963.\n",
  n_after_listwise - nrow(panel_clean)
))


# ── Minimum cross-section guard ────────────────────────────────────────────────
MIN_N <- 50L
xsec_counts <- panel_clean |> count(date, name = "n_stocks")
thin_months  <- filter(xsec_counts, n_stocks < MIN_N)

if (nrow(thin_months) > 0) {
  cat(sprintf(
    "\nWARNING: %d month(s) still have fewer than %d stocks — investigate:\n",
    nrow(thin_months), MIN_N
  ))
  print(thin_months)
} else {
  cat(sprintf(
    "Cross-section check: all %d months have >= %d stocks. OK\n",
    nrow(xsec_counts), MIN_N
  ))
}

cat(sprintf(
  "Cross-section size — median: %d | min: %d | max: %d stocks/month\n",
  as.integer(median(xsec_counts$n_stocks)),
  min(xsec_counts$n_stocks),
  max(xsec_counts$n_stocks)
))


# ── Bias analysis: size of dropped vs retained observations ───────────────────
# Checks whether listwise deletion induces a large-firm bias (expected: yes,
# because Compustat coverage of oper_prof skews toward larger companies).
# Report the ratio in the thesis Sample Selection section.
size_comparison <- bind_rows(
  panel_data |>
    filter(date >= SAMPLE_START, !is.na(mktcap_lag), mktcap_lag > 0) |>
    mutate(retained = permno %in% panel_clean$permno |
             paste(permno, date) %in% paste(panel_clean$permno, panel_clean$date)) |>
    group_by(retained) |>
    summarise(
      n               = n(),
      median_mktcap   = median(mktcap_lag, na.rm = TRUE),
      mean_mktcap     = mean(mktcap_lag,   na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(group = if_else(retained, "Retained (panel_clean)", "Dropped (missing chars)"))
) |>
  select(group, n, median_mktcap, mean_mktcap)

cat("\n── Size bias check: retained vs dropped observations ────────────────────\n")
cat("(median/mean lagged market cap in USD millions)\n\n")
print(size_comparison)
cat("════════════════════════════════════════════════════════════════════════\n")


# ── Sample attrition table ─────────────────────────────────────────────────────
cat("\n══ Sample attrition (for thesis Table 1) ════════════════════════════════\n")
print(attrition, n = Inf)
cat("════════════════════════════════════════════════════════════════════════\n\n")


# ── Summary ────────────────────────────────────────────────────────────────────
cat(sprintf(
  "\nFinal panel: %s obs | %s stocks | %s months | %s to %s\n",
  format(nrow(panel_clean),              big.mark = ","),
  format(n_distinct(panel_clean$permno), big.mark = ","),
  format(n_distinct(panel_clean$date),   big.mark = ","),
  format(min(panel_clean$date), "%Y-%m"),
  format(max(panel_clean$date), "%Y-%m")
))


# ── Fama-French 5 factors ──────────────────────────────────────────────────────
tidy_finance2 <- dbConnect(SQLite(), db_path, extended_types = TRUE)
ff5_factors <- tbl(tidy_finance2, "factors_ff5_monthly") |>
  select(date, mkt_excess, smb, hml, rmw, cma) |>
  collect()
dbDisconnect(tidy_finance2)
rm(tidy_finance2)

cat(sprintf(
  "FF5 factors: %d months | %s to %s\n",
  nrow(ff5_factors),
  format(min(ff5_factors$date), "%Y-%m"),
  format(max(ff5_factors$date), "%Y-%m")
))


# ── Save ───────────────────────────────────────────────────────────────────────
saveRDS(panel_clean,  file = "panel_clean.rds")
cat("Saved: panel_clean.rds\n")

saveRDS(ff5_factors,  file = "ff5_factors.rds")
cat("Saved: ff5_factors.rds\n")

saveRDS(char_spreads, file = "char_spreads.rds")
cat("Saved: char_spreads.rds\n")


# ── Cross-section breadth figure (thesis, Methodology data section) ────────────
# Number of stocks per monthly cross-section after all filters. Documents that
# every FM regression rests on hundreds to thousands of stocks and visualises
# the post-1997 contraction in US listings.
dir.create("plots", showWarnings = FALSE)
p_breadth <- panel_clean |>
  count(date) |>
  ggplot(aes(date, n)) +
  geom_line(colour = "grey25", linewidth = 0.45) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  labs(x = NULL, y = "Stocks per monthly cross-section") +
  theme_bw(base_size = 10)
ggsave("plots/cross_section_breadth.pdf", p_breadth, width = 9, height = 3.4)
cat("Saved: plots/cross_section_breadth.pdf\n")
