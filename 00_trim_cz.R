# ==============================================================================
# 00_trim_cz.R   —   RUN ONCE, LOCALLY (not on the server)
# ------------------------------------------------------------------------------
# The Chen-Zimmermann "signed_predictors_dl_wide.csv" is ~8 GB because it holds
# 200+ signal columns. The thesis uses only five of them, so this script reads
# just the needed columns from the big local file and writes a compact snapshot
# (cz_signals_subset.rds, typically a few tens of MB). Upload ONLY that small
# file to the server; 01_data_pipeline.R reads it via readRDS().
#
# Input : signed_predictors_dl_wide.csv   (the full ~8 GB CZ download, local)
# Output: cz_signals_subset.rds           (permno, yyyymm + 5 signal columns)
#
# Reference: Chen & Zimmermann (2022, Critical Finance Review)
# ==============================================================================

big_csv <- "signed_predictors_dl_wide.csv"   # path to the full local download
out_rds  <- "cz_signals_subset.rds"

# Keep the CZ acronyms exactly; renaming/sign-flipping happens in 01_data_pipeline.R
cz_cols <- c("permno", "yyyymm", "Beta", "BM", "Mom12m", "OperProf", "AssetGrowth")

# fread reads only these columns, so memory stays low despite the 8 GB file.
cz <- data.table::fread(big_csv, select = cz_cols)

# Drop rows where ALL five signals are NA — these can never enter the analysis
# (01_data_pipeline.R requires every characteristic to be present) and they
# contribute nothing to the cross-sectional IQR spreads. Safe size reduction.
sig_cols <- c("Beta", "BM", "Mom12m", "OperProf", "AssetGrowth")
keep <- rowSums(!is.na(cz[, ..sig_cols])) > 0L
cz   <- cz[keep]

# xz compression for the smallest possible upload (read once, so speed is fine).
saveRDS(cz, out_rds, compress = "xz")

cat(sprintf(
  "Wrote %s: %s rows x %d cols (%.1f MB on disk).\n",
  out_rds, format(nrow(cz), big.mark = ","), ncol(cz),
  file.info(out_rds)$size / 1024^2
))
