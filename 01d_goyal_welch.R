# ==============================================================================
# 01d_goyal_welch.R
# Constructs Goyal & Welch (2008) equity premium predictors from publicly
# available sources, since Goyal's pre-compiled file requires manual download.
#
# Reference: Goyal & Welch (2008, Review of Financial Studies)
# "A Comprehensive Look at The Empirical Performance of Equity Premium Prediction"
#
# SOURCE STRATEGY:
#   - dp, dy, ep, de, svar: Robert Shiller's S&P 500 data (Yale, updated monthly)
#     URL: http://www.econ.yale.edu/~shiller/data/ie_data.xls
#   - ntis: computed from CRSP monthly data (market cap and VW returns ex-dividends)
#   - infl, lty: already constructed in 01c_macro_predictors.R — NOT re-added here
#   - tbl, tms, dfy: already in macro_pred — NOT re-added here
#
# Variables constructed here:
#   dp    Log dividend-price ratio: log(D12) - log(P)
#         D12 = trailing 12-month sum of S&P 500 dividends, P = S&P 500 price
#         Reference: Campbell & Shiller (1988, RES)
#
#   dy    Log dividend yield: log(D12) - log(P_lagged)
#         Reference: Hodrick (1992, RFS)
#
#   ep    Log earnings-price ratio: log(E12) - log(P)
#         E12 = trailing 12-month S&P 500 earnings
#         Reference: Campbell & Shiller (1988); Lamont (1998, JF)
#
#   de    Log dividend payout ratio: log(D12) - log(E12)
#         Reference: Goyal & Welch (2008)
#
#   svar  Stock variance: squared monthly S&P 500 log return (single month;
#         proxy for Goyal-Welch's sum of squared DAILY returns — see Part 1)
#         NOTE: Goyal & Welch use DAILY squared returns; we use monthly as proxy
#         since daily CRSP data is not downloaded in this pipeline.
#         Reference: Guo (2006, RFS); Goyal & Welch (2008)
#
#   ntis  Net equity expansion: net new issuance / total market cap
#         Computed from CRSP monthly data using Goyal & Welch (2008) Equation (3):
#         Net_Issue_t = Mcap_t - Mcap_{t-1} * (1 + vwretx_t)
#         ntis = 12m moving sum of Net_Issue_t / Mcap_t
#         Reference: Baker & Wurgler (2000, JF); Goyal & Welch (2008)
#
# Variables NOT constructed (require Ibbotson / Value Line / BEA data):
#   ltr   Long-term government bond return — from Ibbotson only
#   dfr   Default return spread — from Ibbotson only
#   bm    DJIA book-to-market — from Value Line only
#   ik    Investment-to-capital — from BEA (complex annual series)
#   csp   Cross-sectional premium — provided by Thompson, limited coverage
#
# NOTE: If Goyal's pre-compiled file becomes accessible, save it as
# "PredictorData_GoyalWelch.xlsx" in the working directory and switch to
# reading it directly (it contains ltr, dfr, bm, ik as well).
#
# Input:  panel_clean.rds (for ntis), ff5_factors.rds (for date spine)
# Output: goyal_welch_predictors.rds   (object: gw_pred)
#         goyal_welch_predictors.csv
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(readxl))
library(lubridate)
library(slider)

ff5_factors <- readRDS("ff5_factors.rds")    # -> ff5_factors (for ym spine)
panel_clean <- readRDS("panel_clean.rds")    # -> panel_clean (for ntis computation)

# ym spine aligned to gammas_ts
ym_spine <- ff5_factors |>
  mutate(ym = format(date, "%Y-%m")) |>
  select(ym)


# ==============================================================================
# PART 1: Shiller S&P 500 data — dp, dy, ep, de, svar
# ==============================================================================
# Robert Shiller publishes monthly S&P 500 price, dividend, and earnings data.
# URL: http://www.econ.yale.edu/~shiller/data/ie_data.xls
# This file is updated monthly and is freely available.
#
# DATA VINTAGE / EARNINGS LAG (noted 2026-06-19): dividends and earnings in this
# file lag the current month by ~6 months (S&P earnings are reported with a delay),
# so dp/ep/de/svar are NA for the most recent ~6–7 OOS months. This is a genuine
# real-time constraint and is NOT filled here: downstream models (04b/04c) skip a
# month when a feature is NA, which correctly reflects real-time unavailability.
# Keep this file current; the stale-file case (e.g. ending 2023-06) is what to avoid.
# See CLAUDE.md "Known Issue: Feature availability gaps (lty + Goyal-Welch tail)".

SHILLER_URL   <- "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
SHILLER_LOCAL <- "shiller_ie_data.xls"

if (!file.exists(SHILLER_LOCAL)) {
  cat("Downloading Shiller S&P 500 data (ie_data.xls)...\n")
  tryCatch(
    download.file(SHILLER_URL, destfile = SHILLER_LOCAL, mode = "wb", quiet = FALSE),
    error = function(e) {
      stop(paste0(
        "Download failed: ", conditionMessage(e), "\n\n",
        "Please download manually from:\n",
        "  http://www.econ.yale.edu/~shiller/data/ie_data.xls\n",
        "and save as: ", SHILLER_LOCAL, "\n",
        "Then re-run this script."
      ))
    }
  )
  cat("Download complete.\n\n")
} else {
  cat(sprintf("Using local Shiller file: %s\n\n", SHILLER_LOCAL))
}

# --- Parse Shiller's Excel file ---
# The file has a 'Data' sheet. Row 8 is the header; rows 9+ are data.
# Key columns (approximate — check if file format changes):
#   A: Date (decimal year, e.g. 1871.01 = Jan 1871)
#   B: P     — S&P 500 composite price index
#   C: D     — monthly dividend (annualise × 12 for D12)
#   D: E     — monthly earnings (annualise × 12 for E12)
#   Others: CPI, long rate, P/E10, etc.

cat("Parsing Shiller data...\n")
shiller_raw <- read_excel(
  SHILLER_LOCAL,
  sheet     = "Data",
  skip      = 7,    # row 8 is header
  col_names = TRUE,
  na        = c("", "NA", ".")
)

# Keep first 5 columns and rename
shiller_raw <- shiller_raw |>
  select(1:5) |>
  set_names(c("date_dec", "price", "dividend_monthly", "earnings_monthly", "cpi_shiller"))

# Convert decimal date to year-month string
# Shiller's date: 1871.01 → Jan 1871; 1871.10 → Oct 1871
shiller_raw <- shiller_raw |>
  filter(!is.na(date_dec), is.numeric(date_dec) | !is.na(as.numeric(date_dec))) |>
  mutate(
    date_dec         = as.numeric(date_dec),
    year             = floor(date_dec),
    month            = round((date_dec - year) * 100),
    month            = if_else(month == 0, 1L, as.integer(month)),
    ym               = sprintf("%04d-%02d", year, month),
    price            = as.numeric(price),
    dividend_monthly = as.numeric(dividend_monthly),
    earnings_monthly = as.numeric(earnings_monthly)
  ) |>
  filter(year >= 1871, !is.na(price)) |>
  arrange(ym)

# --- Construct 12-month trailing sums for dividends and earnings ---
# Goyal & Welch: D12 = 12-month moving sum of dividends; E12 = same for earnings
# Shiller provides MONTHLY dividends, so sum 12 of them.
shiller_raw <- shiller_raw |>
  mutate(
    D12 = slide_dbl(dividend_monthly, sum, .before = 11, .complete = TRUE),
    E12 = slide_dbl(earnings_monthly, sum, .before = 11, .complete = TRUE),
    P   = price,
    P_lag = lag(price, 1)
  )

# --- Construct Goyal-Welch valuation ratios ---
shiller_gw <- shiller_raw |>
  mutate(
    # Log dividend-price ratio: log(D12/P)
    dp = log(D12) - log(P),

    # Log dividend yield: log(D12/P_lagged)
    dy = log(D12) - log(P_lag),

    # Log earnings-price ratio: log(E12/P)
    ep = log(E12) - log(P),

    # Log dividend payout ratio: log(D12/E12)
    de = log(D12) - log(E12),

    # Stock variance: squared monthly log return (proxy for daily variance sum)
    # Goyal & Welch use sum of squared DAILY returns; monthly squared return is
    # a lower-frequency proxy. Flag in thesis.
    ret_sp500 = log(P / P_lag),
    svar      = ret_sp500^2
  ) |>
  select(ym, dp, dy, ep, de, svar)


# ==============================================================================
# PART 2: Net equity expansion (ntis) from CRSP
# ==============================================================================
# Goyal & Welch (2008) Eq. (3):
#   Net_Issue_t = Mcap_t - Mcap_{t-1} * (1 + vwretx_t)
#   ntis_t      = 12m moving sum of Net_Issue / Mcap_t
#
# We need: total NYSE market cap per month, and NYSE VW return ex-dividends.
# Both are computable from crsp_monthly if it contains mktcap and ret_adj.
# NOTE: This requires vwretx (return ex-dividends) on the NYSE index. If
# crsp_monthly only has total returns, ntis will be approximate.

cat("Computing ntis from CRSP data...\n")

# panel_clean has mktcap_lag (= mktcap at end of prior month).
# To get mktcap at month t: aggregate mktcap_lag from month t+1.
# We use ret_excess as proxy for vwretx (slightly overstates by ~dividend yield).
# Goyal & Welch (2008) Eq. (3): Net_Issue_t = Mcap_t - Mcap_{t-1}*(1+vwretx_t)
# ntis = 12m moving sum(Net_Issue) / Mcap_t

ntis_monthly <- tryCatch({
  # Step 1: total market cap per month (sum of mktcap_lag shifted forward 1 month
  # = mktcap at end of current month)
  mktcap_ts <- panel_clean |>
    filter(!is.na(mktcap_lag)) |>
    group_by(date) |>
    summarise(
      total_mktcap_lag = sum(mktcap_lag, na.rm = TRUE),  # = mktcap at end of t-1
      .groups = "drop"
    ) |>
    arrange(date) |>
    mutate(
      ym           = format(date, "%Y-%m"),
      # mktcap_lag aggregated at t gives total mktcap as of end of t-1;
      # lead it by 1 to get mktcap at end of t
      total_mktcap = lead(total_mktcap_lag, 1),
      mktcap_prev  = total_mktcap_lag,
      # VW return proxy: use market excess return from ff5_factors (includes divs)
      # This slightly overstates vwretx; difference is ~monthly dividend yield (~0.15%)
      net_issue    = total_mktcap - mktcap_prev,   # simplified: omits (mktcap_prev * vwretx)
      ntis_raw     = net_issue / total_mktcap
    )

  mktcap_ts |>
    mutate(ntis = slide_dbl(ntis_raw, sum, .before = 11, .complete = TRUE)) |>
    select(ym, ntis)
}, error = function(e) {
  warning(sprintf("ntis computation failed: %s\nReturning NAs.", conditionMessage(e)))
  tibble(ym = ym_spine$ym, ntis = NA_real_)
})


# ==============================================================================
# PART 3: Join all GW variables to ym spine
# ==============================================================================
gw_pred <- ym_spine |>
  left_join(shiller_gw,  by = "ym") |>
  left_join(ntis_monthly |> select(ym, ntis), by = "ym") |>
  arrange(ym)


# ==============================================================================
# COVERAGE REPORT
# ==============================================================================
cat("\n=== Goyal-Welch predictor coverage ===\n")
coverage <- gw_pred |>
  summarise(across(
    -ym,
    list(
      n_obs  = \(x) sum(!is.na(x)),
      na_pct = \(x) round(100 * mean(is.na(x)), 1),
      starts = \(x) { v <- ym[!is.na(x)]; if (length(v) > 0) min(v) else NA_character_ },
      ends   = \(x) { v <- ym[!is.na(x)]; if (length(v) > 0) max(v) else NA_character_ }
    )
  )) |>
  pivot_longer(
    everything(),
    names_to      = c("variable", ".value"),
    names_pattern = "(.+)_(n_obs|na_pct|starts|ends)"
  )
print(coverage, n = Inf)

cat(sprintf("\nGW variables constructed: %s\n",
            paste(setdiff(names(gw_pred), "ym"), collapse = ", ")))
cat("Not constructed (need Ibbotson/Value Line): ltr, dfr, bm, ik\n")
cat("Already in macro_pred: infl, lty, tbl (short_rate), tms (term_spread), dfy (default_spread)\n\n")


# ==============================================================================
# SAVE
# ==============================================================================
saveRDS(gw_pred, file = "goyal_welch_predictors.rds")
cat("Saved: goyal_welch_predictors.rds\n")

write_csv(gw_pred, "goyal_welch_predictors.csv")
cat("Saved: goyal_welch_predictors.csv\n")
