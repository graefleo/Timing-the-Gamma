# ==============================================================================
# 04c_predict_lstm.R  (v3 — full feature set)
#
# Changes from v2:
#   [1] Full feature set  : input is now aligned[, feature_cols] (~118 cols)
#       instead of the hand-crafted M matrix (6 gammas + 6 macro = 12 cols).
#       gamma_predictions.rds already contains aligned + feature_cols, so
#       gammas_ts.rds and macro_predictors.rds are no longer loaded here.
#   [2] Input projection  : Linear(n_feat_full → PROJ_DIM=32) + ELU added
#       before the LSTM. With 118 features the raw LSTM would have ~80K params
#       per model on ~300 training sequences (ratio ≈ 0.004 — severe overfitting
#       risk). The projection bottleneck reduces this to ~20K params.
#       nn_linear applies to the last dim so [B, SEQ_LEN, 118] → [B, SEQ_LEN, 32]
#       with no reshape needed — per-timestep projection is automatic.
#   [3] HIDDEN_SIZE = 32  : reduced from 64 to match the smaller projected space
#   [4] RETRAIN_FREQS    : robustness check over retraining frequency.
#       RETRAIN_FREQ = 6 (semi-annual) is the primary; RETRAIN_FREQS = c(1,6,12)
#       produces `lstm_r1`, `lstm`, `lstm_r12`. Monthly retraining × 6 gammas ×
#       397 steps ≈ 2400 fits; slow on CPU but feasible overnight.
#   [5] CUDA support      : same smoke-test pattern as 04b_predict_mlp.R
#   [6] Separate F / G    : feature matrix F and target matrix G are standardised
#       independently. G_std is used only for target scaling; targets are
#       unstandardised via g_mu / g_sig before storage. Feature NAs → 0 after
#       standardisation (= imputed column mean), same as v2.
#   [7] NoMacro variant     : optional run without macro columns, matching MLP
#       diagnostic. Controlled by RUN_LSTM_NOGEO flag.
#
# Architecture (one instance per gamma k):
#   Input  [B, SEQ_LEN, n_feat_full]
#   → Linear(n_feat_full, PROJ_DIM) + ELU              [B, SEQ_LEN, PROJ_DIM]
#   → LSTM(PROJ_DIM, HIDDEN_SIZE=32, layers=2,
#           inter-layer dropout=0.2)
#   → last hidden [B, HIDDEN_SIZE]
#   → LayerNorm(HIDDEN_SIZE) → Dropout(0.2) → Linear(HIDDEN_SIZE → 1)
#
# Alignment:
#   aligned[t, feature_cols] = features at time t  (predict gamma at t+1)
#   aligned[t, gamma_cols]   = gamma at time t+1   (outcome)
#   Training sequences at step i (t = oos_idx[i]):
#     rows 1 … t of aligned; last target = G_std[t-1, k] = gamma_t
#   Prediction: x = F_std[(t-SEQ_LEN+1):t, :]  → gamma_{t+1} = actuals[i,k]
#
# Loss:   Huber / nn_smooth_l1_loss (δ=1; robust to fat-tailed gamma spikes)
# Optim:  Adam(lr=1e-3, wd=1e-3) + cosine LR annealing (floor 1e-5)
# Val:    last 15% of sequences (temporal holdout, min 5 sequences)
#
# Inputs:  gamma_predictions.rds  (aligned, feature_cols, oos_idx, pred_list, …)
# Output:  gamma_predictions.rds  (lstm column added to pred_list + oos_eval)
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(torch)
library(sandwich)   # NeweyWest() for Clark-West inference

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LEN       <- 12L              # look-back window (months of history)
RETRAIN_FREQ  <-  6L              # primary retraining frequency (semi-annual)
RETRAIN_FREQS <- c(1L, 6L, 12L)   # robustness check: monthly / semi-annual / annual
INIT_WINDOW   <- 60L              # must match 04_predict_gammas.R
PROJ_DIM     <- 32L   # input projection dimension (118 → 32 before LSTM)
HIDDEN_SIZE  <- 32L   # LSTM hidden units (was 64; matched to projected input size)
N_LAYERS     <-  2L   # LSTM layers; activates inter-layer dropout
DROPOUT      <-  0.2
LR           <-  1e-3
LR_MIN       <-  1e-5
WEIGHT_DECAY <-  1e-3   # increased vs v2 (larger feature space → more regularisation)
MAX_EPOCHS   <- 150L
PATIENCE     <-  15L
VAL_FRAC     <-  0.15
VAL_MIN      <-   5L
BATCH_SIZE   <-  32L
CLIP_GRAD    <-  1.0
NW_LAG       <-  12L

RUN_LSTM_NOGEO <- TRUE   # also run LSTM on gamma-only features (no macro)
MACRO_COLS_LSTM <- c("default_spread", "term_spread", "short_rate",
                     "vix", "indpro_gap", "mkt_lag1", "infl", "lty")

set.seed(42L)
torch_manual_seed(42L)

# ── Device selection ───────────────────────────────────────────────────────────
# GPU disabled — using CPU only (CUDA runtime issues)
device <- torch_device("cpu")
cat("Using device: cpu\n")

# ── Load data ──────────────────────────────────────────────────────────────────
# All features come from aligned (built and saved by 04_predict_gammas.R).
# No need to reload gammas_ts or macro_predictors separately.
list2env(readRDS("gamma_predictions.rds"), envir = environment())  # -> pred_list, oos_eval, actuals, dates_oos,
                                   #    aligned, feature_cols, oos_idx

# Strip any previously saved lstm columns so we rebuild from scratch
# Matches: lstm, lstm_nogeo, lstm_r1, lstm_r6, lstm_r12, ...
for (.gc in names(pred_list)) {
  .drop <- grepl("^lstm(_|$)", colnames(pred_list[[.gc]]))
  pred_list[[.gc]] <- pred_list[[.gc]][, !.drop, drop = FALSE]
}
rm(.gc, .drop)

gamma_cols <- c(
  "gamma_bm", "gamma_mom12m", "gamma_oper_prof",
  "gamma_asset_growth", "gamma_size", "gamma_beta"
)
gamma_labels <- c(
  gamma_bm           = "Value",
  gamma_mom12m       = "Momentum",
  gamma_oper_prof    = "Profitability",
  gamma_asset_growth = "Asset Growth",
  gamma_size         = "Size",
  gamma_beta         = "Beta"
)
n_gc <- length(gamma_cols)

# ── Feature and target matrices ────────────────────────────────────────────────
# F_mat [T_aligned × n_feat_full]: LSTM input features (includes gamma lags,
#   char spreads, macro, GW predictors, factor signals)
# G_mat [T_aligned × 6]          : targets — gamma at time t+1 for each row t
F_mat      <- as.matrix(aligned[, feature_cols])
G_mat      <- as.matrix(aligned[, gamma_cols])
T_aligned  <- nrow(F_mat)
n_feat_full <- ncol(F_mat)

cat(sprintf(
  "Feature matrix F: %d rows x %d cols | target G: %d cols\n",
  T_aligned, n_feat_full, n_gc
))
cat(sprintf(
  "OOS: %d steps, retrain every %d month(s), %d models\n",
  length(oos_idx), RETRAIN_FREQ, n_gc
))

# ── LSTM module (per-gamma, scalar output) ─────────────────────────────────────
# Input projection applied per-timestep: nn_linear operates on the last
# dimension, so [B, SEQ_LEN, n_feat_full] → [B, SEQ_LEN, PROJ_DIM] without
# any explicit reshape.
lstm_module <- nn_module(
  "GammaLSTM",
  initialize = function(n_feat_in, proj_dim, hidden, n_layers, p_drop) {
    # Compress large input to lower-dimensional representation per timestep
    self$proj <- nn_sequential(
      nn_linear(n_feat_in, proj_dim),
      nn_elu()
    )
    self$lstm <- nn_lstm(
      input_size  = proj_dim,
      hidden_size = hidden,
      num_layers  = n_layers,
      batch_first = TRUE,
      dropout     = if (n_layers > 1L) p_drop else 0
    )
    self$ln   <- nn_layer_norm(hidden)
    self$drop <- nn_dropout(p_drop)
    self$fc   <- nn_linear(hidden, 1L)
  },
  forward = function(x) {
    # x: [B, SEQ_LEN, n_feat_in]
    xp   <- self$proj(x)              # [B, SEQ_LEN, proj_dim] — applied per timestep
    out  <- self$lstm(xp)
    last <- out[[1L]][, out[[1L]]$size(2L), ]   # last hidden: [B, hidden]
    self$fc(self$drop(self$ln(last)))            # [B, 1]
  }
)

# ── Sequence builder (per-gamma k, using feature_cols from F_mat) ──────────────
# For training rows 1 … t-1 of aligned:
#   Sequence s: X = F_std[s : s+SEQ_LEN-1, :]   (SEQ_LEN feature rows)
#               Y = G_std[s+SEQ_LEN-1,     k]   (gamma at time s+SEQ_LEN)
#   s ranges 1 … t-SEQ_LEN  →  n_seq = t - SEQ_LEN sequences
#   (last target row is t-1; no leakage of test observation)
# Returns NULL if fewer than 15 clean sequences are available.
make_sequences_k <- function(sp, t, k,
                              F_std = sp$F_std,
                              G_std = sp$G_std,
                              nf    = n_feat_full) {
  n_seq <- t - SEQ_LEN
  if (n_seq < 1L) return(NULL)

  X_arr <- array(0, dim = c(n_seq, SEQ_LEN, nf))
  Y_vec <- numeric(n_seq)

  for (s in seq_len(n_seq)) {
    r_last     <- s + SEQ_LEN - 1L        # last row of this sequence
    X_arr[s,,] <- F_std[s:r_last, ]
    Y_vec[s]   <- G_std[r_last, k]        # gamma at time r_last + 1
  }

  ok <- !is.na(Y_vec)
  if (sum(ok) < 15L) return(NULL)
  list(X = X_arr[ok,,, drop = FALSE], Y = Y_vec[ok])
}

# ── Training helper (single gamma) ────────────────────────────────────────────
loss_fn <- nn_smooth_l1_loss()   # Huber loss, δ=1

fit_lstm_k <- function(seqs, nf = n_feat_full) {
  n_seq <- length(seqs$Y)
  n_val <- max(VAL_MIN, floor(n_seq * VAL_FRAC))
  n_tr  <- n_seq - n_val
  if (n_tr < 10L) return(NULL)

  idx_t <- seq_len(n_tr)
  idx_v <- (n_tr + 1L):n_seq

  x_tr <- torch_tensor(seqs$X[idx_t,,, drop = FALSE])$float()$to(device = device)
  y_tr <- torch_tensor(matrix(seqs$Y[idx_t], ncol = 1L))$float()$to(device = device)
  x_vl <- torch_tensor(seqs$X[idx_v,,, drop = FALSE])$float()$to(device = device)
  y_vl <- torch_tensor(matrix(seqs$Y[idx_v], ncol = 1L))$float()$to(device = device)

  bs    <- min(BATCH_SIZE, n_tr)
  model <- lstm_module(nf, PROJ_DIM, HIDDEN_SIZE, N_LAYERS, DROPOUT)$to(device = device)
  opt   <- optim_adam(model$parameters, lr = LR, weight_decay = WEIGHT_DECAY)
  sched <- lr_cosine_annealing(opt, T_max = MAX_EPOCHS, eta_min = LR_MIN)

  best_val <- Inf; best_state <- NULL; no_imp <- 0L

  for (epoch in seq_len(MAX_EPOCHS)) {
    model$train()
    perm <- sample(n_tr)
    for (b0 in seq(1L, n_tr, by = bs)) {
      bi <- perm[b0:min(b0 + bs - 1L, n_tr)]
      opt$zero_grad()
      pred <- model(x_tr[bi,,])
      loss <- loss_fn(pred, y_tr[bi, , drop = FALSE])
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = CLIP_GRAD)
      opt$step()
    }
    sched$step()

    model$eval()
    with_no_grad({
      vl_loss <- loss_fn(model(x_vl), y_vl)$item()
    })

    if (!is.finite(vl_loss)) break
    if (vl_loss < best_val - 1e-7) {
      best_val   <- vl_loss
      best_state <- lapply(model$state_dict(), \(p) p$clone()$cpu())
      no_imp     <- 0L
    } else {
      no_imp <- no_imp + 1L
      if (no_imp >= PATIENCE) break
    }
  }

  if (!is.null(best_state)) model$load_state_dict(best_state)
  model$eval()

  rm(x_tr, y_tr, x_vl, y_vl)
  gc()
  if (exists("torch_gc")) torch_gc()
  model
}

# ── OOS loop helper ────────────────────────────────────────────────────────────
# Takes raw (unstandardised) F_raw and G_raw matrices.
# Standardisation is recomputed inside each retrain block (strict expanding
# window; no look-ahead bias): features use rows 1:t (features at time t are
# observed at prediction time), targets use rows 1:(t-1) (row t of G_raw is
# gamma_{t+1}, the value being predicted).
run_lstm_oos <- function(F_raw, G_raw, nf, label, retrain_freq = RETRAIN_FREQ) {
  n_oos      <- length(oos_idx)
  preds      <- matrix(NA_real_, nrow = n_oos, ncol = n_gc,
                       dimnames = list(NULL, gamma_cols))
  cur_models <- vector("list", n_gc)
  sp_cached  <- NULL   # list(F_std, G_std, g_mu, g_sig) from last retrain

  cat(sprintf(
    "\nRunning %s OOS (%d steps, %d features, retrain every %d month%s)\n",
    label, n_oos, nf, retrain_freq, if (retrain_freq == 1L) "" else "s"
  ))
  t0 <- proc.time()

  for (i in seq_along(oos_idx)) {
    t          <- oos_idx[i]
    do_retrain <- is.null(sp_cached) || ((i - 1L) %% retrain_freq == 0L)

    if (do_retrain && t > SEQ_LEN + 5L) {
      for (k in seq_len(n_gc)) {
        if (!is.null(cur_models[[k]])) cur_models[[k]]$cpu()
        cur_models[k] <- list(NULL)
      }
      gc(); if (exists("torch_gc")) torch_gc()

      # ── Expanding-window standardisation (features 1:t, targets 1:t-1) ────
      F_tr  <- F_raw[seq_len(t), , drop = FALSE]
      f_mu  <- colMeans(F_tr, na.rm = TRUE)
      f_sig <- apply(F_tr, 2L, sd, na.rm = TRUE)
      f_sig[is.na(f_sig) | f_sig < 1e-8] <- 1
      F_std <- sweep(sweep(F_raw, 2L, f_mu, "-"), 2L, f_sig, "/")
      F_std[is.na(F_std)] <- 0

      # Target moments use rows 1:(t-1) only: G_raw[t, ] = gamma_{t+1} is the
      # value being predicted at this step and must not enter g_mu / g_sig.
      G_tr  <- G_raw[seq_len(t - 1L), , drop = FALSE]
      g_mu  <- colMeans(G_tr, na.rm = TRUE)
      g_sig <- apply(G_tr, 2L, sd, na.rm = TRUE)
      g_sig[is.na(g_sig) | g_sig < 1e-8] <- 1
      G_std <- sweep(sweep(G_raw, 2L, g_mu, "-"), 2L, g_sig, "/")
      G_std[is.na(G_std)] <- 0

      sp_cached <- list(F_std = F_std, G_std = G_std, g_mu = g_mu, g_sig = g_sig)

      for (k in seq_len(n_gc)) {
        seqs_k <- make_sequences_k(sp_cached, t, k,
                                   F_std = F_std, G_std = G_std, nf = nf)
        fitted_k <- if (!is.null(seqs_k)) {
          tryCatch(
            fit_lstm_k(seqs_k, nf = nf),
            error = function(e) {
              warning(sprintf("fit_lstm_k[%s] step %d: %s",
                              gamma_cols[k], i, conditionMessage(e)))
              NULL
            }
          )
        } else NULL
        cur_models[k] <- list(fitted_k)
      }
      gc()
    }

    # Predict using standardisation from last retrain
    if (!is.null(sp_cached) && t >= SEQ_LEN) {
      x_raw <- sp_cached$F_std[(t - SEQ_LEN + 1L):t, , drop = FALSE]
      if (!anyNA(x_raw)) {
        x_te <- torch_tensor(
          array(x_raw, dim = c(1L, SEQ_LEN, nf))
        )$float()$to(device = device)

        for (k in seq_len(n_gc)) {
          if (!is.null(cur_models[[k]])) {
            raw_k <- tryCatch(
              with_no_grad(as.numeric(cur_models[[k]](x_te)$cpu())),
              error = function(e) NA_real_
            )
            preds[i, k] <- raw_k * sp_cached$g_sig[k] + sp_cached$g_mu[k]
          }
        }
        rm(x_te)
      }
    }

    if (i %% 10L == 0L) { gc(); if (exists("torch_gc")) torch_gc() }

    if (i %% 50L == 0L || i == n_oos) {
      elapsed <- (proc.time() - t0)["elapsed"]
      eta     <- if (i < n_oos) elapsed / i * (n_oos - i) else 0
      cat(sprintf("  step %4d / %d | %.1f min elapsed | ETA %.1f min\n",
                  i, n_oos, elapsed / 60, eta / 60))
    }
  }

  cat(sprintf("Done (%s). Total: %.1f min\n",
              label, (proc.time() - t0)["elapsed"] / 60))
  cat("Non-NA predictions per gamma:\n")
  print(setNames(colSums(!is.na(preds)), gamma_labels[gamma_cols]))
  preds
}

# ── Primary LSTM + retraining-frequency robustness variants ───────────────────
# Primary (stored as `lstm`, matching RETRAIN_FREQ) + robustness variants at
# the other frequencies in RETRAIN_FREQS (stored as `lstm_r{freq}`).
# Reference: Gu, Kelly & Xiu (2020, RFS) — NN retraining frequency as robustness.
lstm_variants <- list(
  lstm = run_lstm_oos(
    F_raw = F_mat, G_raw = G_mat, nf = n_feat_full,
    label = sprintf("LSTM primary (retrain=%dm)", RETRAIN_FREQ),
    retrain_freq = RETRAIN_FREQ
  )
)
for (rf in setdiff(RETRAIN_FREQS, RETRAIN_FREQ)) {
  vname <- sprintf("lstm_r%d", rf)
  lstm_variants[[vname]] <- run_lstm_oos(
    F_raw = F_mat, G_raw = G_mat, nf = n_feat_full,
    label = sprintf("LSTM robustness (retrain=%dm)", rf),
    retrain_freq = rf
  )
}

# ── NoMacro variant (primary freq only — diagnostic, not a robustness target) ──
if (RUN_LSTM_NOGEO) {
  feat_nogeo <- setdiff(feature_cols, MACRO_COLS_LSTM)
  F_mat_ng   <- as.matrix(aligned[, feat_nogeo])
  nf_ng      <- ncol(F_mat_ng)

  lstm_variants$lstm_nogeo <- run_lstm_oos(
    F_raw = F_mat_ng, G_raw = G_mat, nf = nf_ng,
    label = sprintf("LSTM-NoMacro (%d features, no macro)", nf_ng),
    retrain_freq = RETRAIN_FREQ
  )
}

for (gc in gamma_cols) {
  for (vname in names(lstm_variants)) {
    raw <- lstm_variants[[vname]][, gc]
    m   <- pred_list[[gc]]
    if (vname %in% colnames(m)) m <- m[, colnames(m) != vname, drop = FALSE]
    pred_list[[gc]] <- cbind(m, setNames(data.frame(raw), vname))
  }
}

# ── Evaluation ────────────────────────────────────────────────────────────────
clark_west <- function(y, yhat_bench, yhat_model, nw_lags = NW_LAG) {
  f  <- (y - yhat_bench)^2 - ((y - yhat_model)^2 - (yhat_bench - yhat_model)^2)
  ok <- !is.na(f)
  n  <- sum(ok)
  if (n < 10L) return(list(t_cw = NA_real_, p_cw = NA_real_))
  f_ok   <- f[ok]
  nw_var <- tryCatch(
    as.numeric(NeweyWest(lm(f_ok ~ 1), lag = nw_lags,
                         prewhite = FALSE, adjust = TRUE)),
    error = function(e) var(f_ok) / n
  )
  if (!is.finite(nw_var) || nw_var <= 0) nw_var <- var(f_ok) / n
  t_stat <- mean(f_ok) / sqrt(nw_var)
  list(t_cw = round(t_stat, 3L), p_cw = round(1 - pnorm(t_stat), 3L))
}

oos_r2 <- function(y, yhat, yhat_bench) {
  1 - sum((y - yhat)^2,       na.rm = TRUE) /
      sum((y - yhat_bench)^2, na.rm = TRUE)
}

lstm_model_cols <- names(lstm_variants)

new_eval <- map_dfr(gamma_cols, function(gc) {
  y     <- actuals[, gc]
  bench <- pred_list[[gc]][, "hist_mean"]
  map_dfr(lstm_model_cols, function(m) {
    if (!m %in% colnames(pred_list[[gc]])) return(NULL)
    yhat  <- pred_list[[gc]][, m]
    valid <- !is.na(yhat) & !is.na(y)
    if (sum(valid) < 10L) return(NULL)
    cw <- clark_west(y[valid], bench[valid], yhat[valid])
    tibble(
      gamma      = gamma_labels[gc],
      sub_period = "Full",
      model      = m,
      oos_r2     = round(oos_r2(y[valid], yhat[valid], bench[valid]) * 100, 2L),
      msfe       = round(mean((y[valid] - yhat[valid])^2), 6L),
      msfe_ratio = round(
        mean((y[valid] - yhat[valid])^2) / mean((y[valid] - bench[valid])^2),
        3L
      ),
      t_cw  = cw$t_cw,
      p_cw  = cw$p_cw,
      n_oos = sum(valid)
    )
  })
})

oos_eval <- bind_rows(
  oos_eval |> filter(!model %in% lstm_model_cols),
  new_eval
)

cat("\n=== LSTM OOS R² (%) — v3 full features ===\n")
print(
  new_eval |>
    dplyr::select(gamma, model, oos_r2, t_cw, p_cw, n_oos) |>
    pivot_wider(names_from = model, values_from = c(oos_r2, t_cw, p_cw, n_oos)),
  n = Inf
)

# ── Save ───────────────────────────────────────────────────────────────────────
# Merge updated objects back into the existing file so aligned / feature_cols /
# oos_idx / oos_eval_all (written by 04_predict_gammas.R) are preserved.
gp <- readRDS("gamma_predictions.rds")
gp$pred_list <- pred_list; gp$oos_eval <- oos_eval
gp$actuals   <- actuals;   gp$dates_oos <- dates_oos
saveRDS(gp, file = "gamma_predictions.rds")
cat("\nSaved: gamma_predictions.rds (lstm v3 added)\n")
