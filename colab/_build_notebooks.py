#!/usr/bin/env python3
"""Generate the two Colab notebooks (04b_predict_mlp.ipynb, 04c_predict_lstm.ipynb).

Run locally:  python colab/_build_notebooks.py
This only writes the .ipynb files; it does not run any training.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def kernel_meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    }


# ============================================================================
# Shared cells (data loading, eval helpers) — identical across both notebooks
# ============================================================================
SETUP = r'''
# --- Colab setup: point DATA_DIR at the uploaded colab_io/ folder ------------
# Option A (Google Drive): mount and set the path to wherever you uploaded colab_io.
# Option B (manual upload): upload the 6 CSVs into /content and set DATA_DIR="/content".
#
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_DIR = '/content/drive/MyDrive/thesis/colab_io'

DATA_DIR = '/content/colab_io'   # <-- EDIT THIS to match where you put the CSVs

import os
assert os.path.isdir(DATA_DIR), f"DATA_DIR not found: {DATA_DIR}"
print("DATA_DIR:", DATA_DIR)
print("files   :", sorted(os.listdir(DATA_DIR)))
'''

IMPORTS = r'''
# CUBLAS determinism must be configured BEFORE the CUDA context is created,
# i.e. before importing torch. Required by torch.use_deterministic_algorithms.
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- Full determinism (supervisor mandate, Meeting 3 2026-07-10) -------------
# cuDNN's default RNN/conv kernels are non-deterministic on GPU; without these
# switches every rerun of the SAME notebook is a fresh draw (the 2026-07-06
# LSTM Size flip +1.88 -> -3.48 was exactly this). Deterministic mode makes a
# rerun on the same hardware/software stack bit-identical; across different
# GPUs results may still differ marginally (acknowledged by supervisor).
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.use_deterministic_algorithms(True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- GKX seed ensemble (Gu, Kelly & Xiu 2020, RFS, Sec. 1.9) ------------------
# "we use multiple random seeds to initialize neural network estimation and
#  construct predictions by averaging forecasts from all networks" (GKX p.2246;
# see also Hansen & Salamon 1990; Dietterich 2000). GKX's appendix and exact
# replications (e.g. Drobetz & Otto 2021) use 10 seeds; we use 5 per supervisor
# specification (Meeting 3) and report across-seed dispersion via the per-seed
# columns exported alongside the ensemble average. Early stopping runs WITHIN
# each seed (as in GKX); the ensemble is the equal-weight forecast average.
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]

def set_seed(seed):
    """Re-seed numpy + torch (incl. CUDA) so each ensemble member is an
    independent, individually reproducible draw."""
    np.random.seed(seed)
    torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("torch", torch.__version__, "| device:", device)
if device.type == 'cuda':
    print("GPU:", torch.cuda.get_device_name(0),
          "| CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())
print("deterministic:", torch.are_deterministic_algorithms_enabled(),
      "| ensemble seeds:", ENSEMBLE_SEEDS)
'''

LOAD = r'''
# --- Load the exported objects ----------------------------------------------
aligned    = pd.read_csv(os.path.join(DATA_DIR, 'aligned.csv'), parse_dates=['date'])
feature_cols = [l.strip() for l in open(os.path.join(DATA_DIR, 'feature_cols.txt')) if l.strip()]
oos_idx    = pd.read_csv(os.path.join(DATA_DIR, 'oos_idx.csv'))['oos_idx'].to_numpy(dtype=int)  # R 1-based
hist_mean  = pd.read_csv(os.path.join(DATA_DIR, 'hist_mean.csv'))
actuals    = pd.read_csv(os.path.join(DATA_DIR, 'actuals.csv'))
dates_oos  = pd.read_csv(os.path.join(DATA_DIR, 'dates_oos.csv'), parse_dates=['date'])['date']

gamma_cols = ['gamma_bm','gamma_mom12m','gamma_oper_prof',
              'gamma_asset_growth','gamma_size','gamma_beta']
gamma_labels = {'gamma_bm':'Value','gamma_mom12m':'Momentum','gamma_oper_prof':'Profitability',
                'gamma_asset_growth':'Asset Growth','gamma_size':'Size','gamma_beta':'Beta'}
MACRO_COLS = ['default_spread','term_spread','short_rate','vix',
              'indpro_gap','mkt_lag1','infl','lty']

INIT_WINDOW = 60
NW_LAG      = 12
n_oos       = len(oos_idx)

# Feature / gamma matrices over the full aligned series (numpy, NaNs preserved).
F_all = aligned[feature_cols].to_numpy(dtype=float)   # [T_aligned, p]
G_all = aligned[gamma_cols].to_numpy(dtype=float)     # [T_aligned, 6]
T_aligned = F_all.shape[0]
print(f"aligned: {T_aligned} rows x {len(feature_cols)} features | oos steps: {n_oos}")
print(f"OOS window: {dates_oos.min():%Y-%m} .. {dates_oos.max():%Y-%m}")
'''

EVAL = r'''
# --- Evaluation helpers (SANITY CHECK ONLY) ---------------------------------
# The authoritative OOS R2 / Clark-West numbers for the thesis are recomputed in
# R (import_*_from_colab.R) with sandwich::NeweyWest. This Newey-West below is a
# faithful manual re-implementation (Bartlett kernel, fixed lag, small-sample
# n/(n-1) adjustment) so you can sanity-check the GPU run before downloading.

def oos_r2(y, yhat, bench):
    m = ~(np.isnan(y) | np.isnan(yhat) | np.isnan(bench))
    return 1.0 - np.sum((y[m]-yhat[m])**2) / np.sum((y[m]-bench[m])**2)

def clark_west(y, bench, model, L=NW_LAG):
    f = (y-bench)**2 - ((y-model)**2 - (bench-model)**2)
    f = f[~np.isnan(f)]
    n = len(f)
    if n < 10:
        return np.nan, np.nan
    e  = f - f.mean()
    g0 = (e @ e) / n
    s  = g0
    for l in range(1, L+1):
        if l >= n: break
        gl = (e[l:] @ e[:-l]) / n
        s += 2.0 * (1.0 - l/(L+1)) * gl
    var_mean = s / (n - 1)            # sandwich adjust=TRUE on intercept-only model
    if not np.isfinite(var_mean) or var_mean <= 0:
        var_mean = np.var(f, ddof=1) / n
    t = f.mean() / np.sqrt(var_mean)
    from math import erf, sqrt
    p = 1.0 - 0.5*(1.0 + erf(t/sqrt(2)))
    return round(float(t), 3), round(float(p), 3)

def eval_table(pred_dict, apply_ct=False):
    rows = []
    for gc in gamma_cols:
        y     = actuals[gc].to_numpy()
        bench = hist_mean[gc].to_numpy()
        for name, mat in pred_dict.items():
            yhat = mat[:, gamma_cols.index(gc)].copy()
            if apply_ct:
                bad = (~np.isnan(yhat)) & (~np.isnan(bench)) & (np.sign(yhat) != np.sign(bench))
                yhat = np.where(bad, bench, yhat)
            valid = (~np.isnan(yhat)) & (~np.isnan(y))
            if valid.sum() < 10:
                continue
            t_cw, p_cw = clark_west(y[valid], bench[valid], yhat[valid])
            rows.append(dict(gamma=gamma_labels[gc], model=name,
                             oos_r2=round(100*oos_r2(y[valid], yhat[valid], bench[valid]), 2),
                             t_cw=t_cw, p_cw=p_cw, n=int(valid.sum())))
    return pd.DataFrame(rows)
'''


# ============================================================================
# MLP-specific cells
# ============================================================================
MLP_CONFIG = r'''
# --- MLP hyperparameters (match 04b_predict_mlp.R) --------------------------
RETRAIN_FREQ  = 6
RETRAIN_FREQS = [1, 6, 12]
HIDDEN        = (32, 16)
DROPOUT       = 0.40
LR, LR_MIN    = 1e-3, 1e-5
WEIGHT_DECAY  = 1e-3
HUBER_DELTA   = 1.0          # nn.SmoothL1Loss(beta=1.0) == Huber delta=1
MAX_EPOCHS    = 300
PATIENCE      = 25
VAL_FRAC      = 0.15
BATCH_SIZE    = 32

# MLP-SR (Ext-A) hyperparameters
BATCH_SIZE_SR   = 64
PRETRAIN_EPOCHS = 50
MAX_EPOCHS_SR   = 250
EPS_SR          = 1e-4
K_GAMMAS        = len(gamma_cols)

RUN_MLP_NOGEO   = True
RUN_MLP_SR      = True
RUN_ROBUSTNESS  = True    # mlp_r1 / mlp_r12 retraining-frequency variants.
                          # ON: each is a full 5-seed GKX ensemble like the primary,
                          # so r1 (monthly retrain) is ~30x the primary run. Budget
                          # an overnight GPU pass. Flip False to skip the robustness set.

huber = nn.SmoothL1Loss(beta=HUBER_DELTA)

def impute_means(X):
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        na = np.isnan(col)
        if na.any():
            fill = col[~na].mean() if (~na).any() else 0.0
            X[na, j] = fill
    return X
'''

MLP_MODEL = r'''
# --- Scalar MLP: input -> 32 -> ELU -> Dropout -> 16 -> ELU -> Dropout -> 1 --
class GammaMLP(nn.Module):
    def __init__(self, n_in, h1, h2, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h1), nn.ELU(), nn.Dropout(p),
            nn.Linear(h1, h2),   nn.ELU(), nn.Dropout(p),
            nn.Linear(h2, 1),
        )
    def forward(self, x):
        return self.net(x)

def _std_cols(X):
    mu  = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0, ddof=1)
    sig = np.where((np.isnan(sig)) | (sig < 1e-8), 1.0, sig)
    return mu, sig

def fit_mlp(X_tr, y_tr, seed=SEED):
    set_seed(seed)          # seed-specific init + batch shuffling (GKX ensemble)
    x_mu, x_sig = _std_cols(X_tr)
    y_mu = float(np.nanmean(y_tr)); y_sig = float(np.nanstd(y_tr, ddof=1))
    if not np.isfinite(y_sig) or y_sig < 1e-8: y_sig = 1.0
    Xs = (X_tr - x_mu) / x_sig
    ys = (y_tr - y_mu) / y_sig

    n = Xs.shape[0]; n_val = max(5, int(np.floor(n*VAL_FRAC)))
    it = np.arange(0, n-n_val); iv = np.arange(n-n_val, n)
    x_t = torch.tensor(Xs[it], dtype=torch.float32, device=device)
    y_t = torch.tensor(ys[it].reshape(-1,1), dtype=torch.float32, device=device)
    x_v = torch.tensor(Xs[iv], dtype=torch.float32, device=device)
    y_v = torch.tensor(ys[iv].reshape(-1,1), dtype=torch.float32, device=device)

    dl = DataLoader(TensorDataset(x_t, y_t),
                    batch_size=min(BATCH_SIZE, len(it)), shuffle=True)
    model = GammaMLP(X_tr.shape[1], HIDDEN[0], HIDDEN[1], DROPOUT).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=LR_MIN)

    best, best_state, no_imp = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = huber(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = huber(model(x_v), y_v).item()
        if not np.isfinite(vl): break
        if vl < best - 1e-7:
            best, best_state, no_imp = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    return dict(model=model, x_mu=x_mu, x_sig=x_sig, y_mu=y_mu, y_sig=y_sig)

def predict_mlp(fit, X_new):
    Xs = (np.asarray(X_new, float) - fit['x_mu']) / fit['x_sig']
    with torch.no_grad():
        p = fit['model'](torch.tensor(Xs, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    return p * fit['y_sig'] + fit['y_mu']
'''

MLP_RUN = r'''
# --- Generic MLP OOS loop (one scalar net per gamma, GKX seed ensemble) ------
# At each re-estimation date we train len(ENSEMBLE_SEEDS) networks per gamma
# (independent seeds, early stopping within each) and average their forecasts
# equally (GKX 2020, Sec. 1.9). Per-seed prediction paths are kept so the R
# import can report across-seed dispersion.
def run_mlp_full(retrain_freq, feat_cols, label):
    fc_idx = [feature_cols.index(c) for c in feat_cols]
    ens_preds  = np.full((n_oos, len(gamma_cols)), np.nan)
    seed_preds = {s: np.full((n_oos, len(gamma_cols)), np.nan) for s in ENSEMBLE_SEEDS}
    fits = {gc: None for gc in gamma_cols}          # gc -> {seed: fit}
    print(f"\n[{label}] {n_oos} steps, {len(feat_cols)} features, "
          f"retrain every {retrain_freq}m, {len(ENSEMBLE_SEEDS)}-seed ensemble")
    for i, t in enumerate(oos_idx):                 # t is R 1-based
        tr_rows = slice(0, t-1)                      # aligned[1:(t-1)] in R
        X_tr = impute_means(F_all[tr_rows][:, fc_idx])
        X_te = F_all[t-1, fc_idx]                    # aligned[t] in R
        do_retrain = (i % retrain_freq == 0)
        for gc in gamma_cols:
            j = gamma_cols.index(gc)
            y_full = G_all[tr_rows, j]
            ok = ~np.isnan(y_full)
            if do_retrain and ok.sum() >= INIT_WINDOW:
                new_fits = {}
                for s in ENSEMBLE_SEEDS:
                    try:
                        new_fits[s] = fit_mlp(X_tr[ok], y_full[ok], seed=s)
                    except Exception as e:
                        print(f"  fit_mlp[{gc}] step {i} seed {s}: {e}")
                if new_fits:
                    fits[gc] = new_fits
            if np.isnan(X_te).any() or not fits[gc]:
                continue
            vals = []
            for s, f in fits[gc].items():
                v = float(predict_mlp(f, X_te.reshape(1, -1))[0])
                seed_preds[s][i, j] = v
                vals.append(v)
            ens_preds[i, j] = float(np.mean(vals))   # equal-weight forecast average
        if (i+1) % 50 == 0: print(f"  step {i+1}/{n_oos}")
    print(f"[{label}] filled {np.sum(~np.isnan(ens_preds[:,0]))}/{n_oos}")
    return ens_preds, seed_preds
'''

MLP_SR = r'''
# --- MLP-SR (Ext-A): joint 6-output net, negative-Sharpe loss ----------------
class GammaMLPSharpe(nn.Module):
    def __init__(self, n_in, h1, h2, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h1), nn.ELU(), nn.Dropout(p),
            nn.Linear(h1, h2),   nn.ELU(), nn.Dropout(p),
            nn.Linear(h2, K_GAMMAS),
        )
    def forward(self, x):
        return self.net(x)

def sharpe_loss(pred, y):
    r_p = (pred * y).sum(dim=1)
    mu  = r_p.mean()
    sig = r_p.std(unbiased=False).clamp(min=EPS_SR)   # population std, matches correction=0L
    return -(mu / sig)

def fit_mlp_sr(X_tr, Y_tr, seed=SEED):
    set_seed(seed)          # seed-specific init + batch shuffling (GKX ensemble)
    x_mu, x_sig = _std_cols(X_tr)
    y_mu = np.nanmean(Y_tr, axis=0)
    y_sig = np.nanstd(Y_tr, axis=0, ddof=1); y_sig = np.where(y_sig < 1e-8, 1.0, y_sig)
    Xs = (X_tr - x_mu) / x_sig
    Ys = (Y_tr - y_mu) / y_sig

    n = Xs.shape[0]; n_val = max(5, int(np.floor(n*VAL_FRAC)))
    it = np.arange(0, n-n_val); iv = np.arange(n-n_val, n)
    x_t = torch.tensor(Xs[it], dtype=torch.float32, device=device)
    y_t = torch.tensor(Ys[it], dtype=torch.float32, device=device)
    x_v = torch.tensor(Xs[iv], dtype=torch.float32, device=device)
    y_v = torch.tensor(Ys[iv], dtype=torch.float32, device=device)
    dl  = DataLoader(TensorDataset(x_t, y_t),
                     batch_size=min(BATCH_SIZE_SR, len(it)), shuffle=True)

    model = GammaMLPSharpe(X_tr.shape[1], HIDDEN[0], HIDDEN[1], DROPOUT).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS_SR, eta_min=LR_MIN)

    # Phase 1: Huber warm-start
    for _ in range(PRETRAIN_EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = huber(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    # Phase 2: Sharpe fine-tuning (val on Huber for stable early stopping)
    best, best_state, no_imp = np.inf, None, 0
    for _ in range(MAX_EPOCHS_SR):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = sharpe_loss(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = huber(model(x_v), y_v).item()
        if not np.isfinite(vl): break
        if vl < best - 1e-7:
            best, best_state, no_imp = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    return dict(model=model, x_mu=x_mu, x_sig=x_sig, y_mu=y_mu, y_sig=y_sig)

def predict_mlp_sr(fit, X_new):
    Xs = (np.asarray(X_new, float) - fit['x_mu']) / fit['x_sig']
    with torch.no_grad():
        p = fit['model'](torch.tensor(Xs, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    return p * fit['y_sig'] + fit['y_mu']

def run_mlp_sr(retrain_freq=RETRAIN_FREQ):
    fc_idx = [feature_cols.index(c) for c in feature_cols]
    ens_preds  = np.full((n_oos, len(gamma_cols)), np.nan)
    seed_preds = {s: np.full((n_oos, len(gamma_cols)), np.nan) for s in ENSEMBLE_SEEDS}
    fits = {}                                          # seed -> fit
    print(f"\n[MLP-SR] {n_oos} steps, retrain every {retrain_freq}m, "
          f"{len(ENSEMBLE_SEEDS)}-seed ensemble")
    for i, t in enumerate(oos_idx):
        tr_rows = slice(0, t-1)
        X_tr = impute_means(F_all[tr_rows][:, fc_idx])
        X_te = F_all[t-1, fc_idx]
        if i % retrain_freq == 0:
            Y = G_all[tr_rows]
            ok = ~np.isnan(Y).any(axis=1)              # complete.cases over 6 gammas
            if ok.sum() >= INIT_WINDOW:
                new_fits = {}
                for s in ENSEMBLE_SEEDS:
                    try:
                        new_fits[s] = fit_mlp_sr(X_tr[ok], Y[ok], seed=s)
                    except Exception as e:
                        print(f"  fit_mlp_sr step {i} seed {s}: {e}")
                if new_fits:
                    fits = new_fits
        if np.isnan(X_te).any() or not fits:
            continue
        vals = []
        for s, f in fits.items():
            p = predict_mlp_sr(f, X_te.reshape(1, -1))
            seed_preds[s][i, :] = p
            vals.append(p)
        ens_preds[i, :] = np.mean(vals, axis=0)        # equal-weight forecast average
        if (i+1) % 50 == 0: print(f"  step {i+1}/{n_oos}")
    print(f"[MLP-SR] filled {np.sum(~np.isnan(ens_preds[:,0]))}/{n_oos}")
    return ens_preds, seed_preds
'''

MLP_DRIVE = r'''
# --- Run all MLP variants (each is a 5-seed GKX ensemble) ---------------------
variants      = {}   # name -> ensemble-average predictions (n_oos x 6)
seed_variants = {}   # name -> {seed: per-seed predictions} for dispersion report
def _add(name, result):
    variants[name], seed_variants[name] = result

_add('mlp', run_mlp_full(RETRAIN_FREQ, feature_cols, f"MLP primary (retrain={RETRAIN_FREQ}m)"))
if RUN_ROBUSTNESS:
    for rf in [f for f in RETRAIN_FREQS if f != RETRAIN_FREQ]:
        _add(f'mlp_r{rf}', run_mlp_full(rf, feature_cols, f"MLP robustness (retrain={rf}m)"))
if RUN_MLP_NOGEO:
    feat_nogeo = [c for c in feature_cols if c not in MACRO_COLS]
    _add('mlp_nogeo', run_mlp_full(RETRAIN_FREQ, feat_nogeo, "MLP-NoGeo (no macro)"))
if RUN_MLP_SR:
    _add('mlp_sr', run_mlp_sr(RETRAIN_FREQ))
print("\nVariants:", list(variants.keys()))
'''

MLP_SAVE = r'''
# --- Save raw predictions for the R import step (THIS is the important part) --
# Wide CSV: one column per {model}__{gamma}. CT and Clark-West are (re)done in R.
# The save runs first and unconditionally, so a skipped helper cell can never
# block it; the sanity print below is best-effort.
out = pd.DataFrame({'step': np.arange(1, n_oos+1), 'date': dates_oos.values})
for name, mat in variants.items():
    for j, gc in enumerate(gamma_cols):
        out[f'{name}__{gc}'] = mat[:, j]
# Per-seed columns ({name}_seed{s}__{gamma}): the R import keeps these OUT of
# pred_list / downstream tables and uses them only for the across-seed
# dispersion report (supervisor mandate, Meeting 3).
for name, sd in seed_variants.items():
    for s, mat in sd.items():
        for j, gc in enumerate(gamma_cols):
            out[f'{name}_seed{s}__{gc}'] = mat[:, j]
save_path = os.path.join(DATA_DIR, 'mlp_preds.csv')
out.to_csv(save_path, index=False)
print(f"Saved {save_path}  ({out.shape[0]} rows x {out.shape[1]} cols)")
print("Download mlp_preds.csv, then run colab/import_mlp_from_colab.R locally.")

# --- Sanity check (best-effort; needs the 'Evaluation helpers' cell) ----------
try:
    print("\n=== RAW (no CT) OOS R2 sanity ===")
    print(eval_table(variants, apply_ct=False).pivot(index='gamma', columns='model', values='oos_r2'))
    print("\n=== CT (Campbell-Thompson) OOS R2 sanity ===")
    print(eval_table(variants, apply_ct=True).pivot(index='gamma', columns='model', values='oos_r2'))
    print("\n=== Across-seed OOS R2 dispersion (primary mlp) ===")
    disp = {f"seed{s}": m for s, m in seed_variants['mlp'].items()}
    disp['ensemble'] = variants['mlp']
    print(eval_table(disp, apply_ct=False).pivot(index='gamma', columns='model', values='oos_r2'))
except NameError:
    print("\n(Sanity table skipped: run the 'Evaluation helpers' cell for OOS R2 — CSV is already saved.)")
'''


# ============================================================================
# LSTM-specific cells
# ============================================================================
LSTM_CONFIG = r'''
# --- LSTM hyperparameters (match 04c_predict_lstm.R) ------------------------
SEQ_LEN       = 12
RETRAIN_FREQ  = 6
RETRAIN_FREQS = [1, 6, 12]
PROJ_DIM      = 32
HIDDEN_SIZE   = 32
N_LAYERS      = 2
DROPOUT       = 0.2
LR, LR_MIN    = 1e-3, 1e-5
WEIGHT_DECAY  = 1e-3
MAX_EPOCHS    = 150
PATIENCE      = 15
VAL_FRAC      = 0.15
VAL_MIN       = 5
BATCH_SIZE    = 32
CLIP_GRAD     = 1.0
RUN_LSTM_NOGEO = True
RUN_ROBUSTNESS = True    # lstm_r1 / lstm_r12 retraining-frequency variants.
                         # ON: each is a full 5-seed GKX ensemble like the primary,
                         # so lstm_r1 (monthly retrain) is ~30x the primary run.
                         # Budget an overnight GPU pass. Flip False to skip.

huber = nn.SmoothL1Loss(beta=1.0)
n_gc  = len(gamma_cols)
'''

LSTM_MODEL = r'''
# --- LSTM: per-timestep Linear projection -> LSTM -> LayerNorm -> Linear(1) --
class GammaLSTM(nn.Module):
    def __init__(self, n_feat_in, proj_dim, hidden, n_layers, p):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_feat_in, proj_dim), nn.ELU())
        self.lstm = nn.LSTM(proj_dim, hidden, num_layers=n_layers,
                            batch_first=True, dropout=(p if n_layers > 1 else 0.0))
        self.ln   = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(p)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):                    # x: [B, SEQ_LEN, n_feat_in]
        xp = self.proj(x)                    # per-timestep projection
        out, _ = self.lstm(xp)
        last = out[:, -1, :]                 # last hidden [B, hidden]
        return self.fc(self.drop(self.ln(last)))

def make_sequences_k(F_std, G_std, t, k, nf):
    # R: n_seq = t - SEQ_LEN ; sequence ss uses rows ss:ss+SEQ_LEN, target row ss+SEQ_LEN-1
    n_seq = t - SEQ_LEN
    if n_seq < 1: return None
    X = np.zeros((n_seq, SEQ_LEN, nf), dtype=np.float32)
    Y = np.zeros(n_seq, dtype=np.float32)
    for ss in range(n_seq):
        X[ss] = F_std[ss:ss+SEQ_LEN, :]
        Y[ss] = G_std[ss+SEQ_LEN-1, k]
    ok = ~np.isnan(Y)
    if ok.sum() < 15: return None
    return X[ok], Y[ok]

def fit_lstm_k(X, Y, nf, seed=SEED):
    set_seed(seed)          # seed-specific init + batch shuffling (GKX ensemble)
    n_seq = len(Y)
    n_val = max(VAL_MIN, int(np.floor(n_seq*VAL_FRAC)))
    n_tr  = n_seq - n_val
    if n_tr < 10: return None
    x_tr = torch.tensor(X[:n_tr], dtype=torch.float32, device=device)
    y_tr = torch.tensor(Y[:n_tr].reshape(-1,1), dtype=torch.float32, device=device)
    x_vl = torch.tensor(X[n_tr:], dtype=torch.float32, device=device)
    y_vl = torch.tensor(Y[n_tr:].reshape(-1,1), dtype=torch.float32, device=device)

    bs = min(BATCH_SIZE, n_tr)
    model = GammaLSTM(nf, PROJ_DIM, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=LR_MIN)

    best, best_state, no_imp = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        perm = np.random.permutation(n_tr)
        for b0 in range(0, n_tr, bs):
            bi = perm[b0:b0+bs]
            opt.zero_grad()
            loss = huber(model(x_tr[bi]), y_tr[bi])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = huber(model(x_vl), y_vl).item()
        if not np.isfinite(vl): break
        if vl < best - 1e-7:
            best, best_state, no_imp = vl, {k_: v.clone() for k_, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    return model
'''

LSTM_RUN = r'''
# --- LSTM OOS loop (expanding-window standardisation recomputed each retrain) -
# GKX seed ensemble adapted to the recurrent architecture: at each re-estimation
# date, len(ENSEMBLE_SEEDS) LSTMs per gamma (independent seeds, early stopping
# within each) are trained and their forecasts averaged equally. Note GKX (2020)
# use feed-forward networks only; applying their ensemble to an LSTM is our
# adaptation. Per-seed paths are kept for the dispersion report.
import time

def run_lstm_oos(feat_cols, label, retrain_freq=RETRAIN_FREQ):
    fc_idx = [feature_cols.index(c) for c in feat_cols]
    nf     = len(feat_cols)
    F_raw  = F_all[:, fc_idx]
    G_raw  = G_all
    ens_preds  = np.full((n_oos, n_gc), np.nan)
    seed_preds = {s: np.full((n_oos, n_gc), np.nan) for s in ENSEMBLE_SEEDS}
    models = [None]*n_gc               # per gamma: {seed: model} or None
    cache  = None                      # (F_std, G_std, g_mu, g_sig)
    print(f"\n[{label}] {n_oos} steps, {nf} features, retrain every {retrain_freq}m, "
          f"{len(ENSEMBLE_SEEDS)}-seed ensemble")
    t0 = time.time()
    for i, t in enumerate(oos_idx):    # t is R 1-based row count
        do_retrain = (cache is None) or (i % retrain_freq == 0)
        if do_retrain and t > SEQ_LEN + 5:
            # Standardise features using rows 1:t (R) == F_raw[:t] (Python);
            # features at time t are observed at prediction time.
            Ftr = F_raw[:t]
            f_mu = np.nanmean(Ftr, axis=0); f_sig = np.nanstd(Ftr, axis=0, ddof=1)
            f_sig = np.where((np.isnan(f_sig)) | (f_sig < 1e-8), 1.0, f_sig)
            F_std = (F_raw - f_mu) / f_sig; F_std = np.nan_to_num(F_std, nan=0.0)

            # Target moments use rows 1:(t-1) only: G_raw row t (R) = gamma_{t+1}
            # is the value being predicted and must not enter g_mu / g_sig.
            Gtr = G_raw[:t-1]
            g_mu = np.nanmean(Gtr, axis=0); g_sig = np.nanstd(Gtr, axis=0, ddof=1)
            g_sig = np.where((np.isnan(g_sig)) | (g_sig < 1e-8), 1.0, g_sig)
            G_std = (G_raw - g_mu) / g_sig; G_std = np.nan_to_num(G_std, nan=0.0)
            cache = (F_std, G_std, g_mu, g_sig)

            for k in range(n_gc):
                seqs = make_sequences_k(F_std, G_std, t, k, nf)
                if seqs is None:
                    models[k] = None
                else:
                    new_models = {}
                    for s in ENSEMBLE_SEEDS:
                        try:
                            m = fit_lstm_k(seqs[0], seqs[1], nf, seed=s)
                            if m is not None:
                                new_models[s] = m
                        except Exception as e:
                            print(f"  fit_lstm_k[{gamma_cols[k]}] step {i} seed {s}: {e}")
                    models[k] = new_models if new_models else None

        if cache is not None and t >= SEQ_LEN:
            F_std, G_std, g_mu, g_sig = cache
            x_raw = F_std[t-SEQ_LEN:t, :]           # R rows (t-SEQ_LEN+1):t
            if not np.isnan(x_raw).any():
                x_te = torch.tensor(x_raw.reshape(1, SEQ_LEN, nf),
                                    dtype=torch.float32, device=device)
                for k in range(n_gc):
                    if models[k]:
                        vals = []
                        for s, m in models[k].items():
                            with torch.no_grad():
                                raw = float(m(x_te).cpu().numpy().ravel()[0])
                            v = raw * g_sig[k] + g_mu[k]
                            seed_preds[s][i, k] = v
                            vals.append(v)
                        ens_preds[i, k] = float(np.mean(vals))   # equal-weight average

        if (i+1) % 50 == 0 or (i+1) == n_oos:
            el = (time.time()-t0)/60
            eta = el/(i+1)*(n_oos-(i+1))
            print(f"  step {i+1}/{n_oos} | {el:.1f} min | ETA {eta:.1f} min")
    print(f"[{label}] done. non-NA per gamma: "
          + str({gamma_labels[gamma_cols[k]]: int(np.sum(~np.isnan(ens_preds[:,k]))) for k in range(n_gc)}))
    return ens_preds, seed_preds
'''

LSTM_DRIVE = r'''
# --- Run all LSTM variants (each is a 5-seed GKX ensemble) --------------------
# NOTE: the 5-seed ensemble is ~5x a single-seed run, and monthly retraining
# (lstm_r1) is another ~6x on top. The r1/r12 robustness variants are gated
# behind RUN_ROBUSTNESS (set in the config cell).
variants      = {}   # name -> ensemble-average predictions (n_oos x 6)
seed_variants = {}   # name -> {seed: per-seed predictions} for dispersion report
def _add(name, result):
    variants[name], seed_variants[name] = result

_add('lstm', run_lstm_oos(feature_cols, f"LSTM primary (retrain={RETRAIN_FREQ}m)", RETRAIN_FREQ))
if RUN_ROBUSTNESS:
    for rf in [f for f in RETRAIN_FREQS if f != RETRAIN_FREQ]:
        _add(f'lstm_r{rf}', run_lstm_oos(feature_cols, f"LSTM robustness (retrain={rf}m)", rf))
if RUN_LSTM_NOGEO:
    feat_nogeo = [c for c in feature_cols if c not in MACRO_COLS]
    _add('lstm_nogeo', run_lstm_oos(feat_nogeo, "LSTM-NoGeo (no macro)", RETRAIN_FREQ))
print("\nVariants:", list(variants.keys()))
'''

LSTM_SAVE = r'''
# --- Save raw predictions for the R import step (THIS is the important part) --
# LSTM has no Campbell-Thompson restriction in the R pipeline (raw only).
# Save runs first and unconditionally; the sanity print below is best-effort.
out = pd.DataFrame({'step': np.arange(1, n_oos+1), 'date': dates_oos.values})
for name, mat in variants.items():
    for j, gc in enumerate(gamma_cols):
        out[f'{name}__{gc}'] = mat[:, j]
# Per-seed columns ({name}_seed{s}__{gamma}): the R import keeps these OUT of
# pred_list / downstream tables and uses them only for the across-seed
# dispersion report (supervisor mandate, Meeting 3).
for name, sd in seed_variants.items():
    for s, mat in sd.items():
        for j, gc in enumerate(gamma_cols):
            out[f'{name}_seed{s}__{gc}'] = mat[:, j]
save_path = os.path.join(DATA_DIR, 'lstm_preds.csv')
out.to_csv(save_path, index=False)
print(f"Saved {save_path}  ({out.shape[0]} rows x {out.shape[1]} cols)")
print("Download lstm_preds.csv, then run colab/import_lstm_from_colab.R locally.")

# --- Sanity check (best-effort; needs the 'Evaluation helpers' cell) ----------
try:
    print("\n=== LSTM OOS R2 sanity ===")
    print(eval_table(variants, apply_ct=False).pivot(index='gamma', columns='model', values='oos_r2'))
    print("\n=== Across-seed OOS R2 dispersion (primary lstm) ===")
    disp = {f"seed{s}": m for s, m in seed_variants['lstm'].items()}
    disp['ensemble'] = variants['lstm']
    print(eval_table(disp, apply_ct=False).pivot(index='gamma', columns='model', values='oos_r2'))
except NameError:
    print("\n(Sanity table skipped: run the 'Evaluation helpers' cell for OOS R2 — CSV is already saved.)")
'''


def build_mlp():
    nb = new_notebook(metadata=kernel_meta())
    nb.cells = [
        new_markdown_cell(
            "# 04b — MLP gamma predictions (Colab / GPU)\n\n"
            "Python translation of `04b_predict_mlp.R`. Trains one scalar MLP per gamma plus the "
            "MLP-NoGeo diagnostic and the MLP-SR (Sharpe-loss) joint network, across retraining "
            "frequencies 1/6/12 months.\n\n"
            "**Seed ensemble (Gu, Kelly & Xiu 2020, RFS, Sec. 1.9):** every variant trains "
            "5 networks with independent random seeds at each re-estimation date (early stopping "
            "within each seed) and averages their forecasts equally. GKX use 10 seeds; we use 5 per "
            "supervisor specification and export per-seed columns (`{model}_seed{N}__…`) so the R "
            "import can report across-seed dispersion. Deterministic cuDNN/CUBLAS is enabled, so a "
            "rerun on the same GPU/software stack reproduces exactly.\n\n"
            "**Workflow:** run `colab/export_for_colab.R` locally → upload `colab_io/` here → run all "
            "cells → download `mlp_preds.csv` → run `colab/import_mlp_from_colab.R` locally to merge "
            "into `gamma_predictions.rds`.\n\n"
            "The R import step recomputes Campbell-Thompson `_ct` columns and Clark-West stats with "
            "`sandwich::NeweyWest`, so those numbers — not the in-notebook sanity check — are authoritative."
        ),
        new_code_cell(SETUP.strip()),
        new_code_cell(IMPORTS.strip()),
        new_code_cell(LOAD.strip()),
        new_code_cell(EVAL.strip()),
        new_code_cell(MLP_CONFIG.strip()),
        new_code_cell(MLP_MODEL.strip()),
        new_code_cell(MLP_RUN.strip()),
        new_code_cell(MLP_SR.strip()),
        new_code_cell(MLP_DRIVE.strip()),
        new_code_cell(MLP_SAVE.strip()),
    ]
    path = os.path.join(HERE, "04b_predict_mlp.ipynb")
    nbf.write(nb, path)
    print("wrote", path)


def build_lstm():
    nb = new_notebook(metadata=kernel_meta())
    nb.cells = [
        new_markdown_cell(
            "# 04c — LSTM gamma predictions (Colab / GPU)\n\n"
            "Python translation of `04c_predict_lstm.R`. Per-gamma LSTM with a per-timestep linear "
            "projection (n_features → 32) before a 2-layer LSTM, across retraining frequencies "
            "1/6/12 months, plus the NoGeo diagnostic.\n\n"
            "**Seed ensemble, adapted from Gu, Kelly & Xiu (2020, RFS, Sec. 1.9):** every variant "
            "trains 5 LSTMs with independent random seeds at each re-estimation date (early stopping "
            "within each seed) and averages their forecasts equally. Note GKX use *feed-forward* "
            "networks (NN1–NN5) only — applying their seed-ensemble procedure to a recurrent "
            "architecture is our adaptation, motivated by the LSTM's demonstrated seed sensitivity. "
            "GKX use 10 seeds; we use 5 per supervisor specification and export per-seed columns "
            "(`{model}_seed{N}__…`) for the across-seed dispersion report. Deterministic cuDNN is "
            "enabled (`torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`), which "
            "fixes the previous non-reproducibility of GPU LSTM training: a rerun on the same "
            "GPU/software stack now reproduces exactly.\n\n"
            "**Workflow:** run `colab/export_for_colab.R` locally → upload `colab_io/` here → run all "
            "cells → download `lstm_preds.csv` → run `colab/import_lstm_from_colab.R` locally to merge "
            "into `gamma_predictions.rds`.\n\n"
            "Use a **GPU runtime** (Runtime → Change runtime type → GPU). Monthly retraining (`lstm_r1`) "
            "is the slow one; comment it out in the driver cell if pressed for time."
        ),
        new_code_cell(SETUP.strip()),
        new_code_cell(IMPORTS.strip()),
        new_code_cell(LOAD.strip()),
        new_code_cell(EVAL.strip()),
        new_code_cell(LSTM_CONFIG.strip()),
        new_code_cell(LSTM_MODEL.strip()),
        new_code_cell(LSTM_RUN.strip()),
        new_code_cell(LSTM_DRIVE.strip()),
        new_code_cell(LSTM_SAVE.strip()),
    ]
    path = os.path.join(HERE, "04c_predict_lstm.ipynb")
    nbf.write(nb, path)
    print("wrote", path)



if __name__ == "__main__":
    build_mlp()
    build_lstm()
