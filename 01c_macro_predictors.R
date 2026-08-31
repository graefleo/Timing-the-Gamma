# ==============================================================================
# 01c_macro_predictors.R
# Downloads and prepares macro conditioning variables from FRED.
#
# These are the external state variables identified in the factor-timing
# literature as primary drivers of time-varying cross-sectional risk premia.
# Without these variables, AR-based models see near-white-noise gammas
# (AC1 < 0.11) and produce near-zero or negative OOS R².
#
# Variables constructed:
#
#   default_spread  Moody's BAA yield − AAA yield  [DBAA − DAAA]
#                   Credit risk / recession proxy. Wide spread = high risk aversion.
#                   Reference: Fama & French (1989, JFE); Chen, Roll & Ross (1986)
#                   Coverage: 1919-01 onward — full sample coverage ✓
#
#   term_spread     10Y Treasury − 3M T-bill        [GS10 − TB3MS]
#                   Business cycle expectations. Inverts before recessions.
#                   Reference: Fama & French (1989, JFE); Campbell (1987, JFE)
#                   Coverage: 1953-04 onward — slight NAs in early sample
#
#   short_rate      3-Month T-bill rate              [TB3MS]
#                   Most robust short-run return predictor across intl. markets.
#                   Reference: Ang & Bekaert (2007, ReStud)
#                   Coverage: 1934-01 onward
#
#   vix             CBOE Volatility Index (monthly avg of daily) [VIXCLS]
#                   Fear proxy / risk appetite. Key variable in supervisor paper.
#                   Reference: Reschenhofer & Zechner (2022, Working Paper)
#                   Coverage: 1990-01 onward — NAs pre-1990 (LASSO handles via
#                   complete.cases; VIX enriches post-1990 OOS sub-period)
#
#   indpro_gap      Year-over-year log growth in industrial production [INDPRO]
#                   Production-based business cycle indicator; look-ahead free.
#                   Avoids HP-filter end-point bias (uses 12-month difference).
#                   Reference: Cooper & Priestley (2007); Stock & Watson (1999)
#                   Coverage: 1920-01 onward (needs 12 months of INDPRO)
#
#   mkt_lag1        1-month lagged market excess return [from ff5_factors]
#                   Already available — no FRED download needed.
#                   Reference: Reschenhofer & Zechner (2022)
#
# DATE ALIGNMENT NOTE:
#   FRED monthly series use first-of-month dates (e.g. 2023-01-01).
#   gammas_ts uses end-of-month dates. We match on year-month string to avoid
#   off-by-one join errors.
#
# LAG NOTE:
#   Macro variables are joined to the `features` matrix at their natural date t.
#   The aligned() construction in 04_predict_gammas.R then places features at t
#   opposite outcome at t+1, creating the correct 1-month prediction lag.
#   No additional lag is applied here.
#
# FRED API KEY:
#   Register free at: https://fred.stlouisfed.org/docs/api/api_key.html
#   Then set your key below or via environment variable:
#     Sys.setenv(FRED_API_KEY = "your_key_here")
#
# Input:  ff5_factors.rds
# Output: macro_predictors.rds   (object: macro_pred)
#         macro_predictors.csv     (for inspection)
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(fredr)
library(lubridate)
select <- dplyr::select  # guard against masking by vars/MASS if loaded in session

ff5_factors <- readRDS("ff5_factors.rds")   # -> ff5_factors

# ── FRED API key ───────────────────────────────────────────────────────────────
# Set your key here OR via Sys.setenv(FRED_API_KEY = "...") in .Renviron
FRED_API_KEY <- Sys.getenv("FRED_API_KEY")
if (nchar(FRED_API_KEY) == 0) {
  stop(paste(
    "FRED API key not found.",
    "Register free at https://fred.stlouisfed.org/docs/api/api_key.html",
    "Then run: Sys.setenv(FRED_API_KEY = '123bec3f8834dd87d758577ba84acc4b')",
    sep = "\n"
  ))
}
fredr_set_key(FRED_API_KEY)


# ── Helper: download monthly FRED series ──────────────────────────────────────
# Returns a two-column tibble: ym (year-month string) | value
fred_monthly <- function(series_id, start = "1950-01-01") {
  cat(sprintf("  Downloading %s from FRED...\n", series_id))
  fredr(
    series_id         = series_id,
    observation_start = as.Date(start),
    frequency         = "m",
    aggregation_method = "eop"   # end-of-period for daily series
  ) |>
    select(date, value) |>
    mutate(ym = format(date, "%Y-%m")) |>
    select(ym, value) |>
    rename(!!series_id := value)
}


# ── Download raw series ────────────────────────────────────────────────────────
cat("Downloading macro series from FRED...\n")

dbaa    <- fred_monthly("DBAA",   start = "1919-01-01")  # Moody's BAA yield
daaa    <- fred_monthly("DAAA",   start = "1919-01-01")  # Moody's AAA yield
gs10    <- fred_monthly("GS10",   start = "1950-01-01")  # 10Y Treasury
tb3ms   <- fred_monthly("TB3MS",  start = "1934-01-01")  # 3M T-bill
vixcls  <- fred_monthly("VIXCLS", start = "1990-01-01")  # VIX (daily → monthly eop)
# NOTE: CBOE SKEW index is NOT available on FRED (tried series "SKEW" — 400 error).
# Download from CBOE directly if needed; omitted for now.
cpiaucsl <- fred_monthly("CPIAUCSL", start = "1919-01-01")  # CPI all urban consumers
gs20     <- fred_monthly("GS20",     start = "1950-01-01")  # 20Y Treasury yield (lty proxy)
gs30     <- fred_monthly("GS30",     start = "1977-02-01")  # 30Y Treasury yield (bridges GS20 1987–1993 gap)
indpro  <- fred_monthly("INDPRO", start = "1919-01-01")  # Industrial production

cat("All downloads complete.\n\n")


# ── Construct macro variables ──────────────────────────────────────────────────
# Year-month spine from ff5_factors (ensures alignment with gammas_ts)
ym_spine <- ff5_factors |>
  mutate(ym = format(date, "%Y-%m")) |>
  select(ym, mkt_excess)

macro_raw <- ym_spine |>
  left_join(dbaa,     by = "ym") |>
  left_join(daaa,     by = "ym") |>
  left_join(gs10,     by = "ym") |>
  left_join(gs20,     by = "ym") |>
  left_join(gs30,     by = "ym") |>
  left_join(tb3ms,    by = "ym") |>
  left_join(vixcls,   by = "ym") |>
  left_join(cpiaucsl, by = "ym") |>
  left_join(indpro,   by = "ym") |>
  arrange(ym)

macro_pred <- macro_raw |>
  mutate(
    # Default spread: BAA − AAA (in percentage points)
    # Wide spread → high credit risk → predicts large value/size premia
    default_spread = DBAA - DAAA,

    # Term spread: 10Y − 3M (in percentage points)
    # Inverted yield curve precedes recessions; positive → expansion expected
    # STATIONARITY NOTE: GS10 and TB3MS are individually I(1) (ADF, 00_diagnostics.R),
    # but their difference is I(0) because the two rates are cointegrated with vector
    # (1, -1) — the spread cannot drift to ±∞ by the no-arbitrage constraint.
    # KPSS on term_spread gives borderline results (~0.09) that vary with bandwidth,
    # but the cointegration argument overrides; the literature universally uses levels
    # (Fama & French 1989; Campbell 1987). No transformation applied.
    term_spread = GS10 - TB3MS,

    # Short rate change: first difference of 3M T-bill (in percentage points/month)
    # ADF test (00_diagnostics.R) confirms TB3MS is I(1) over 1963-2023 — the
    # secular decline from ~15% (1981) to ~0% (2010s) makes the level non-stationary.
    # Regressing I(0) gammas on an I(1) predictor produces spurious in-sample fit
    # that collapses OOS. First-differencing restores stationarity.
    # Interpretation: a rising short rate signals tightening monetary policy.
    # Reference: Campbell & Shiller (1991); Ang & Bekaert (2007, ReStud)
    short_rate = TB3MS - lag(TB3MS, 1),

    # VIX: end-of-period monthly level (in volatility points)
    # High VIX → elevated fear → predicts higher subsequent risk premia
    vix = VIXCLS,


    # Output gap proxy: 12-month log growth in industrial production (in %)
    # Positive = expansion; negative = contraction
    # Uses 12-month difference to avoid HP-filter look-ahead bias
    # STATIONARITY NOTE: KPSS rejects H0 of stationarity (p ≈ 0.013). This is a
    # known false positive caused by the MA(11) autocorrelation structure induced by
    # the 12-month log difference (overlapping 11 monthly INDPRO observations). The
    # default KPSS bandwidth (≈4 lags) severely underestimates the long-run variance.
    # With bandwidth ≥ 12 lags, KPSS fails to reject. ADF clearly rejects a unit root
    # (p < 0.001). Conclusion: indpro_gap is stationary; KPSS rejection is a size
    # distortion from bandwidth mismatch. No transformation applied.
    # Reference: Hobijn, Franses & Ooms (2004, J. Applied Econometrics) on KPSS
    # size distortion under persistent MA structure.
    indpro_gap = 100 * (log(INDPRO) - log(lag(INDPRO, 12))),

    # Lagged market excess return (already available, included for completeness)
    # Negative mkt_lag1 → bad times → predicts higher subsequent premia
    # NOTE: No pre-lag here. The aligned() construction in 04_predict_gammas.R
    # uses slice(-nrow(features)) which places features at t opposite outcome at t+1,
    # giving the correct 1-month prediction lag automatically. Pre-lagging here
    # would create a spurious 2-month lag (the prior bug).
    mkt_lag1 = mkt_excess,

    # Inflation: log monthly CPI growth rate (%)
    # Lagged 1 additional month for real-time availability (CPI released ~2 weeks
    # after month-end, so month t CPI only known in month t+1).
    # Higher inflation → tighter monetary policy → lower equity premia
    # Reference: Fama (1981, AER); Goyal & Welch (2008, RFS)
    # Coverage: 1919-01 onward (full sample)
    infl = 100 * (log(CPIAUCSL) - log(lag(CPIAUCSL, 1))),
    infl = lag(infl, 1),   # real-time lag

    # Long-term government bond yield (lty proxy via 20Y Treasury).
    # GS20 from FRED; Goyal & Welch use Ibbotson's long-term yield (similar).
    # SPLICE: the 20Y constant-maturity series is unpublished 1987-01 … 1993-09
    # (Treasury suspended 20Y issuance 1986–1993), leaving a gap inside the OOS
    # window. Bridge it with the 30Y (GS30, available 1977–2002 & 2006–present);
    # GS30's own 2002–2006 gap is in turn covered by GS20, so coalesce(GS20, GS30)
    # is continuous across 1990–2025. The 20Y/30Y yields track within a few bp.
    # Reference: Keim & Stambaugh (1986, JFE); Goyal & Welch (2008, RFS);
    #            Gürkaynak, Sack & Wright (2007, JME) on Treasury yield curve gaps.
    # Coverage: 1953-09 onward (continuous post-splice)
    lty = coalesce(GS20, GS30)
  ) |>
  select(ym, default_spread, term_spread, short_rate, vix, indpro_gap, mkt_lag1,
         infl, lty)


# ── Coverage report ────────────────────────────────────────────────────────────
cat("=== Macro predictor coverage ===\n")
coverage <- macro_pred |>
  summarise(across(
    -ym,
    list(
      n_obs    = \(x) sum(!is.na(x)),
      na_pct   = \(x) round(100 * mean(is.na(x)), 1),
      starts   = \(x) { v <- ym[!is.na(x)]; if (length(v) > 0) min(v) else NA_character_ },
      ends     = \(x) { v <- ym[!is.na(x)]; if (length(v) > 0) max(v) else NA_character_ }
    )
  )) |>
  pivot_longer(
    everything(),
    names_to      = c("variable", ".value"),
    names_pattern = "(.+)_(n_obs|na_pct|starts|ends)"
  )
print(coverage, n = Inf)

cat(sprintf("\nTotal months in spine: %d (%s to %s)\n",
            nrow(macro_pred),
            min(macro_pred$ym), max(macro_pred$ym)))
cat("Note: VIX NAs pre-1990 are expected; LASSO uses complete.cases per iteration.\n\n")


# ── Save ───────────────────────────────────────────────────────────────────────
saveRDS(macro_pred, file = "macro_predictors.rds")
cat("Saved: macro_predictors.rds\n")

write_csv(macro_pred, "macro_predictors.csv")
cat("Saved: macro_predictors.csv\n")
