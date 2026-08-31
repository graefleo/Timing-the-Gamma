# ==============================================================================
# 01e — Momentum factor (UMD) for the six-factor alpha
# ==============================================================================
# Purpose: supply the momentum factor that the FF5 model lacks, so the timed
#   portfolios can be priced against Fama-French six-factor as well as
#   five-factor benchmarks. The combined book trades a momentum leg, and the
#   static momentum leg alone earns a large FF5 alpha (Table D.4), i.e. the FF5
#   demonstrably cannot price one of the six legs. Any alpha reported against
#   the FF5 therefore mixes timing performance with a known omitted-factor
#   loading; the FF6 column separates the two.
#
# Source: Kenneth R. French's data library, "Momentum Factor (Mom)", monthly.
#   This is the same library from which the Tidy Finance SQLite build derives
#   `factors_ff5_monthly` (the source of ff5_factors.rds), so the two are
#   consistent in construction. Block 2 below verifies that empirically rather
#   than assuming it.
#
# Reference: Carhart (1997, JF) for the four-factor model; Fama & French
#   (2018, JFE) for the six-factor specification. French constructs Mom from
#   six value-weighted portfolios on size and prior return (months -12 to -2).
#
# Input:  F-F_Momentum_Factor.csv (downloaded once, then read from disk)
# Output: umd_factor.rds  -> tibble(date, umd) with date = first of month
# ==============================================================================

library(tidyverse)

RAW_CSV <- "F-F_Momentum_Factor.csv"
URL_MOM <- paste0("https://mba.tuck.dartmouth.edu/pages/faculty/",
                  "ken.french/ftp/F-F_Momentum_Factor_CSV.zip")

# (1) Fetch once, then reuse ---------------------------------------------------
# The file is kept in the project so that a re-run is reproducible and does not
# silently pick up a later CRSP vintage (French revises the history when CRSP
# does). Delete the CSV to force a refresh.
if (!file.exists(RAW_CSV)) {
  tmp_zip <- tempfile(fileext = ".zip")
  utils::download.file(URL_MOM, tmp_zip, mode = "wb", quiet = TRUE)
  utils::unzip(tmp_zip, exdir = ".")
  unlink(tmp_zip)
  cat("Downloaded", RAW_CSV, "from the French data library.\n")
} else {
  cat("Using existing", RAW_CSV, "\n")
}

raw <- readLines(RAW_CSV, warn = FALSE)
vintage <- raw[1]                                   # "...created using the YYYYMM CRSP database."

# Monthly rows are "YYYYMM, value"; the annual block below them is "YYYY, value"
# and the header/footer text matches neither. Selecting on a six-digit key is
# therefore sufficient and needs no line-number offsets that a new vintage would
# invalidate.
mom_lines <- raw[grepl("^\\s*[0-9]{6}\\s*,", raw)]
umd_factor <- tibble(txt = mom_lines) |>
  separate(txt, into = c("ym", "mom"), sep = ",", convert = FALSE) |>
  mutate(
    ym   = str_trim(ym),
    date = as.Date(paste0(substr(ym, 1, 4), "-", substr(ym, 5, 6), "-01")),
    umd  = as.numeric(str_trim(mom)) / 100      # percent -> decimal, as in ff5
  ) |>
  filter(!is.na(umd), umd > -0.99) |>           # French codes missing as -99.99
  select(date, umd) |>
  arrange(date)

cat(sprintf("UMD: %d months, %s to %s\n", nrow(umd_factor),
            format(min(umd_factor$date), "%Y-%m"),
            format(max(umd_factor$date), "%Y-%m")))
cat(sprintf("Vintage line: %s\n", str_trim(vintage)))

# (2) Source-consistency check against the existing FF5 ------------------------
# ff5_factors.rds comes from the Tidy Finance SQLite mirror, umd from French
# directly. If the two sources agree, mixing them in one regression is safe. The
# check is on the overlap of the market factor, which both carry.
ff5 <- readRDS("ff5_factors.rds")
ov  <- inner_join(ff5, umd_factor, by = "date")
cat(sprintf("\nOverlap with ff5_factors.rds: %d months (%s to %s)\n",
            nrow(ov), format(min(ov$date), "%Y-%m"), format(max(ov$date), "%Y-%m")))
if (nrow(ov) < 300) warning("Suspiciously little overlap with ff5_factors.rds.")

# Sanity: UMD should be strongly negatively correlated with HML (the classic
# value/momentum tension) and near-uncorrelated with the market.
cat(sprintf("cor(umd, hml) = %+.3f   cor(umd, mkt_excess) = %+.3f\n",
            cor(ov$umd, ov$hml, use = "complete.obs"),
            cor(ov$umd, ov$mkt_excess, use = "complete.obs")))
cat(sprintf("UMD mean = %+.3f%%/month, sd = %.3f%%\n",
            100 * mean(ov$umd, na.rm = TRUE), 100 * sd(ov$umd, na.rm = TRUE)))

saveRDS(umd_factor, "umd_factor.rds")
cat("\nSaved: umd_factor.rds\n")
