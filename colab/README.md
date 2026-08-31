# Colab MLP / LSTM bridge

Python (PyTorch) re-implementations of `04b_predict_mlp.R` and `04c_predict_lstm.R`
so the GPU-heavy neural-net training can run on Google Colab. The R pipeline stays
authoritative: notebooks only produce raw predictions; all Campbell-Thompson and
Clark-West evaluation is recomputed in R on import.

## Files

| File | Where it runs | Purpose |
|---|---|---|
| `export_for_colab.R`      | local | Dump `gamma_predictions.rds` objects to `colab_io/*.csv` |
| `04b_predict_mlp.ipynb`   | Colab | MLP: primary(6m) + r1 + r12 + NoGeo + MLP-SR |
| `04c_predict_lstm.ipynb`  | Colab | LSTM: primary(6m) + r1 + r12 + NoGeo |
| `import_mlp_from_colab.R` | local | Merge `mlp_preds.csv` into `gamma_predictions.rds` |
| `import_lstm_from_colab.R`| local | Merge `lstm_preds.csv` into `gamma_predictions.rds` |
| `_build_notebooks.py`     | local | Regenerates the two `.ipynb` (edit code here, not the JSON) |

## Workflow

1. **Run `04_predict_gammas.R`** locally so `gamma_predictions.rds` exists.
2. **Export:** `Rscript colab/export_for_colab.R` → writes `colab/colab_io/`
   (`aligned.csv`, `feature_cols.txt`, `oos_idx.csv`, `hist_mean.csv`,
   `actuals.csv`, `dates_oos.csv`).
3. **Colab:** upload the `colab_io/` folder (Drive mount or direct upload), open
   each notebook, set `DATA_DIR` in the first code cell, pick a **GPU runtime**,
   Run all. Each notebook writes `mlp_preds.csv` / `lstm_preds.csv` into `DATA_DIR`.
4. **Download** those two CSVs back into `colab/colab_io/`.
5. **Import:** `Rscript colab/import_mlp_from_colab.R` and
   `Rscript colab/import_lstm_from_colab.R` → merges the `mlp*` / `lstm*` columns
   into `gamma_predictions.rds`, exactly as the original `04b`/`04c` would.

Then continue with `05_portfolio_construction.R` as usual.

## Notes

- Do **not** run both the original R `04b`/`04c` *and* the import scripts — they
  write the same columns. Pick one path (Colab = the new one).
- The in-notebook OOS R² / Clark-West table is a sanity check only. The numbers
  used in the thesis come from the R import step (`sandwich::NeweyWest`).
- Exact figures will differ slightly from a local R-`torch` run (different RNG /
  CUDA kernels); the architectures, losses, regularisation, retraining schedule
  and OOS alignment are identical.
- `colab_io/` is intended as scratch — safe to add to `.gitignore`.
