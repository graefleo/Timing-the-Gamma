# ==============================================================================
# 04b_predict_mlp.R
# MLP (multi-layer perceptron) out-of-sample gamma predictions.
#
# KEY DESIGN DECISIONS (vs. prior version):
#
#   1. One network per gamma (6 models, scalar output each).
#      The prior multi-output design coupled all 6 gammas in a single loss,
#      averaging out individual signals. With AC1 near zero, any spurious
#      cross-gamma coupling amplifies noise. Independent models let each
#      network specialise on its own target.
#
#   2. Smaller architecture: n_in -> 32 -> 16 -> 1.
#      Prior: 42 -> 64 -> 32 -> 6 ≈ 5 000 parameters on ~270 training rows
#             (ratio: 0.05 samples / parameter — extreme overfitting risk).
#      With 42 features: 42 -> 32 -> 16 -> 1 ≈ 1 900 parameters (ratio ≈ 0.14).
#      With ~118 features: 118 -> 32 -> 16 -> 1 ≈ 4 400 parameters (ratio ≈ 0.07
#      at the start of OOS). Strong regularisation (Dropout=0.40, L2=1e-3) is
#      the primary defence; the 32-unit bottleneck forces aggressive compression.
#      L2 weight decay and Dropout provide additional regularisation.
#
#   3. BatchNorm removed.
#      With BATCH_SIZE=32 and ~8 batches per epoch, running statistics never
#      converge. BatchNorm and Dropout also interact destructively in small-
#      sample settings (Dropout disturbs the variance BatchNorm normalises).
#      Replaced with plain Dropout; L2 via weight_decay handles scale control.
#
#   4. Retraining-frequency robustness: RETRAIN_FREQS = c(1L, 6L, 12L).
#      Primary RETRAIN_FREQ = 6L (semi-annual, stored as `mlp`). Robustness
#      variants at 1-month (`mlp_r1`) and 12-month (`mlp_r12`) retraining
#      test sensitivity to staleness. Annual retraining leaves up to 11
#      consecutive months of stale predictions; monthly retraining is the
#      cleanest spec but 6× the compute.
#
# Architecture : input(n_in) -> Linear(32) -> ELU -> Dropout(0.20)
#                            -> Linear(16) -> ELU -> Dropout(0.20)
#                            -> Linear(1)          [linear, no activation]
# Loss         : Huber (smooth_l1, delta=1.0) — robust to fat-tailed residuals
# Optimiser    : Adam + L2 weight decay (1e-4) + cosine LR annealing
# Features     : feature_cols loaded from gamma_predictions.rds (~118 columns:
#                30 gamma lags/vol + 6 char spreads + 8 macro + ~10 GW + 75 factor signals)
# OOS scheme   : expanding window; model retrained every RETRAIN_FREQ months
#
# Input  : gamma_predictions.rds  (aligned, feature_cols, oos_idx, pred_list, …)
# Output : gamma_predictions.rds  (pred_list + oos_eval extended with "mlp")
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))
library(torch)
library(sandwich)  # NeweyWest() for HAC-robust CW inference

# ── Hyperparameters ────────────────────────────────────────────────────────────
RETRAIN_FREQ   <- 6L               # primary retraining frequency (semi-annually)
RETRAIN_FREQS  <- c(1L, 6L, 12L)   # robustness check: monthly / semi-annual / annual
INIT_WINDOW    <- 60L              # must match 04_predict_gammas.R
NW_LAG         <- 12L
RUN_MLP_NOGEO  <- TRUE   # also run MLP without the 8 macro predictors
                         # Object keys stay `mlp_nogeo` / `mlp_nogeo_ct` (Colab CSV
                         # bridge + stored .rds depend on them); the reader-facing
                         # label in 06/07 and the thesis is "MLP-NoMacro".
# Ext-A: MLP with Sharpe Ratio Objective (supervisor extension 2026-04-18).
# Replaces per-gamma scalar MLPs with a single 6-output joint network whose
# loss directly maximises portfolio Sharpe ratio instead of minimising MSE.
# Economic rationale: MSE and Sharpe ratio can diverge substantially
# (Salcher, Stöckl & Hanke 2026 "Lost in Translation").
# Two-phase training: Huber warm-start → SR fine-tuning avoids non-convex
# SR loss landscape issues from random initialisation.
# Reference: Gu, Kelly & Xiu (2020, RFS) — neural nets in asset pricing.
RUN_MLP_SR     <- TRUE
                          # Diagnostic: isolates whether macro features add or subtract
                          # value in the nonlinear model. In near-zero AC1 environments,
                          # 6 macro features may provide more overfitting surface than
                          # predictive signal for 5 of 6 gammas (Size expected to benefit
                          # most from mkt_lag1 after the double-lag fix).
# Macro column names — used to define the nogeo feature subset.
# Must match macro_cols in 04_predict_gammas.R (8 variables).
MACRO_COLS_MLP <- c("default_spread", "term_spread", "short_rate",
                    "vix", "indpro_gap", "mkt_lag1", "infl", "lty")
HIDDEN       <- c(32L, 16L)   # smaller than before: fewer params relative to N
DROPOUT      <- 0.40   # increased from 0.20: macro correlations near zero → strong regularisation needed
LR           <- 1e-3
LR_MIN       <- 1e-5
WEIGHT_DECAY <- 1e-3   # increased from 1e-4: same rationale
HUBER_DELTA  <- 1.0
MAX_EPOCHS   <- 300L
PATIENCE     <- 25L
VAL_FRAC     <- 0.15
BATCH_SIZE   <- 32L
set.seed(42)
torch_manual_seed(42)

loss_fn <- nn_smooth_l1_loss()

# ── Ext-A: Sharpe Ratio loss and multi-output architecture ────────────────────
# Architecture change for MLP-SR: one joint 6-output network (vs six 1-output
# scalar networks). Joint prediction allows the Sharpe loss to be computed over
# the full factor portfolio each batch step.
#
# Hyperparameters for SR training:
BATCH_SIZE_SR   <- 64L    # larger batches → more stable SR gradient estimate
                           # CLT requires ~30+ portfolio returns per SR estimate
PRETRAIN_EPOCHS <- 50L    # Huber warm-start epochs before switching to SR loss
MAX_EPOCHS_SR   <- 250L   # SR fine-tuning epochs (early stopping with PATIENCE)
EPS_SR          <- 1e-4   # denominator floor: prevents NaN when batch SR vol → 0
                           # Gradient challenge: std(r_p) → 0 causes ∂L/∂θ → ∞
                           # clamp(min = EPS_SR) is the standard fix.

# Joint 6-output MLP for Sharpe optimisation
# Architecture: same hidden layers as scalar MLP, final layer outputs 6 gammas.
# No BatchNorm (same reasoning as scalar MLP — small batches, destructive with Dropout).
K_GAMMAS <- length(gamma_cols)   # = 6

mlp_module_sr <- nn_module(
  "GammaMLPSharpe",
  initialize = function(n_in, h1, h2, p_drop) {
    self$net <- nn_sequential(
      nn_linear(n_in, h1),
      nn_elu(),
      nn_dropout(p_drop),
      nn_linear(h1, h2),
      nn_elu(),
      nn_dropout(p_drop),
      nn_linear(h2, K_GAMMAS)   # joint output: all 6 gammas simultaneously
    )
  },
  forward = function(x) self$net(x)
)

# Sharpe ratio loss function.
# pred:       [batch, 6] predicted gammas (standardised — network output space)
# y_realized: [batch, 6] realised gammas (standardised — same space)
#
# Portfolio return logic: predicted γ̂_k,t serves as the position weight on
# factor k. The realised γ_k,t is the "return" earned by holding that position.
# Since both are standardised by the same per-gamma training statistics,
# the ratio (SR in standardised space) = SR in natural units (scale-invariant).
# Reference: Salcher, Stöckl & Hanke (2026) — portfolio Sharpe as primary objective.
sharpe_loss_fn <- function(pred, y_realized) {
  r_p  <- (pred * y_realized)$sum(dim = 2L)     # [batch] portfolio return each step
  mu   <- r_p$mean()
  # correction = 0L: population std (divide by N, not N-1). Avoids the
  # "degrees of freedom <= 0" warning when batch size is small. Using N vs N-1
  # in the SR denominator is standard practice (SR formula is scale-consistent).
  sig  <- r_p$std(correction = 0L)$clamp(min = EPS_SR)
  -(mu / sig)                                    # negative SR → minimise
}

# Fit MLP-SR: joint 6-output network with two-phase training.
# Phase 1 (Huber warm-start): stabilises weight initialisation — the SR loss
#   landscape is non-convex from random init; Huber provides a good starting basin.
# Phase 2 (SR fine-tuning): switches loss to negative Sharpe ratio.
# References: Gu et al. (2020, RFS) — curriculum learning for asset pricing NNs.
fit_mlp_sr <- function(X_tr, Y_tr) {
  # X_tr: [T, p] feature matrix    Y_tr: [T, 6] all-gamma target matrix
  stopifnot(ncol(Y_tr) == K_GAMMAS)

  # Standardise inputs (same as scalar MLP)
  x_mu  <- colMeans(X_tr, na.rm = TRUE)
  x_sig <- apply(X_tr, 2L, sd, na.rm = TRUE)
  x_sig[x_sig < 1e-8] <- 1
  Xs <- scale(X_tr, center = x_mu, scale = x_sig)

  # Standardise all 6 gamma targets jointly
  y_mu  <- colMeans(Y_tr, na.rm = TRUE)
  y_sig <- apply(Y_tr, 2L, sd, na.rm = TRUE)
  y_sig[y_sig < 1e-8] <- 1
  Ys <- scale(Y_tr, center = y_mu, scale = y_sig)

  # Temporal validation split (last VAL_FRAC of rows)
  n     <- nrow(Xs)
  n_val <- max(5L, floor(n * VAL_FRAC))
  idx_t <- seq_len(n - n_val)
  idx_v <- (n - n_val + 1L):n

  x_t <- torch_tensor(Xs[idx_t, , drop = FALSE], dtype = torch_float())$to(device = device)
  y_t <- torch_tensor(Ys[idx_t, , drop = FALSE], dtype = torch_float())$to(device = device)
  x_v <- torch_tensor(Xs[idx_v, , drop = FALSE], dtype = torch_float())$to(device = device)
  y_v <- torch_tensor(Ys[idx_v, , drop = FALSE], dtype = torch_float())$to(device = device)

  bs   <- min(BATCH_SIZE_SR, length(idx_t))
  dl_p <- dataloader(tensor_dataset(x_t, y_t), batch_size = bs, shuffle = TRUE)

  model <- mlp_module_sr(ncol(X_tr), HIDDEN[1L], HIDDEN[2L], DROPOUT)$to(device = device)
  opt   <- optim_adam(model$parameters, lr = LR, weight_decay = WEIGHT_DECAY)
  sched <- lr_cosine_annealing(opt, T_max = MAX_EPOCHS_SR, eta_min = LR_MIN)

  # ── Phase 1: Huber warm-start ─────────────────────────────────────────────
  huber_fn <- nn_smooth_l1_loss()
  for (epoch in seq_len(PRETRAIN_EPOCHS)) {
    model$train()
    coro::loop(for (batch in dl_p) {
      opt$zero_grad()
      loss <- huber_fn(model(batch[[1L]]), batch[[2L]])
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      opt$step()
    })
  }

  # ── Phase 2: SR fine-tuning ───────────────────────────────────────────────
  best_val   <- Inf
  best_state <- NULL
  no_imp     <- 0L

  for (epoch in seq_len(MAX_EPOCHS_SR)) {
    model$train()
    coro::loop(for (batch in dl_p) {
      opt$zero_grad()
      loss <- sharpe_loss_fn(model(batch[[1L]]), batch[[2L]])
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      opt$step()
    })
    sched$step()

    # Validation: use Huber for early-stopping stability
    # (SR on the small val set is too noisy to use as stopping criterion)
    model$eval()
    with_no_grad({
      vl_loss <- huber_fn(model(x_v), y_v)$item()
    })
    if (!is.finite(vl_loss)) break
    if (vl_loss < best_val - 1e-7) {
      best_val   <- vl_loss
      best_state <- lapply(model$state_dict(), \(p) p$clone())
      no_imp     <- 0L
    } else {
      no_imp <- no_imp + 1L
      if (no_imp >= PATIENCE) break
    }
  }
  if (!is.null(best_state)) model$load_state_dict(best_state)
  model$eval()

  list(model = model, x_mu = x_mu, x_sig = x_sig, y_mu = y_mu, y_sig = y_sig)
}

# Predict all 6 gammas jointly from a fitted MLP-SR model.
# Returns named numeric vector of length 6 (one prediction per gamma, natural units).
predict_mlp_sr <- function(fit, X_new) {
  Xs <- scale(as.matrix(X_new), center = fit$x_mu, scale = fit$x_sig)
  with_no_grad({
    pred_s <- as.numeric(
      fit$model(torch_tensor(Xs, dtype = torch_float())$to(device = device))$cpu()
    )
  })
  # Unstandardise to natural gamma units
  setNames(pred_s * fit$y_sig + fit$y_mu, gamma_cols)
}

# Column-mean imputation — same logic as 04_predict_gammas.R.
# Completely-NA columns (e.g. VIX in training windows before 1990) are filled
# with 0. After standardisation in fit_mlp, a constant-0 column has sd ≈ 0,
# which triggers the x_sig floor → scaled to 0 throughout → effectively a
# zero-weight feature. This is the correct behaviour: no VIX signal exists yet.
impute_means <- function(X) {
  for (j in seq_len(ncol(X))) {
    na_j <- is.na(X[, j])
    if (any(na_j)) {
      fill <- if (any(!na_j)) mean(X[!na_j, j]) else 0
      X[na_j, j] <- fill
    }
  }
  X
}

# Use CUDA if available and passes a smoke-test; fall back to CPU otherwise.
device <- tryCatch({
  if (cuda_is_available()) {
    d <- torch_device("cuda")
    # Smoke-test: small tensor op to catch driver mismatches before the OOS loop
    tmp <- torch_randn(4L, 4L, device = d)
    rm(tmp)
    cat("CUDA available — using GPU\n")
    d
  } else {
    cat("CUDA not available — using CPU\n")
    torch_device("cpu")
  }
}, error = function(e) {
  cat("CUDA smoke-test failed (", conditionMessage(e), ") — falling back to CPU\n")
  torch_device("cpu")
})
cat("Using device:", device$type, "\n")

# ── Load data ──────────────────────────────────────────────────────────────────
list2env(readRDS("gamma_predictions.rds"), envir = environment())  # -> pred_list, oos_eval, actuals, dates_oos,
                                  #    aligned, feature_cols, oos_idx
# Strip any previously saved mlp columns so we rebuild from scratch.
# Matches: mlp, mlp_ct, mlp_nogeo*, mlp_sr*, mlp_r1*, mlp_r6*, mlp_r12*, ...
for (.gc in names(pred_list)) {
  .drop <- grepl("^mlp(_|$)", colnames(pred_list[[.gc]]))
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

T_aligned <- nrow(aligned)

# ── MLP module (scalar output — one instance per gamma) ───────────────────────
# No BatchNorm: unstable with small batches; interacts destructively with Dropout.
# ELU: smooth, allows negative activations, avoids dead-neuron problem of ReLU.
mlp_module <- nn_module(
  "GammaMLP",
  initialize = function(n_in, h1, h2, p_drop) {
    self$net <- nn_sequential(
      nn_linear(n_in, h1),
      nn_elu(),
      nn_dropout(p_drop),
      nn_linear(h1, h2),
      nn_elu(),
      nn_dropout(p_drop),
      nn_linear(h2, 1L)   # scalar output per gamma
    )
  },
  forward = function(x) self$net(x)
)

# ── Training helper (single target gamma) ─────────────────────────────────────
fit_mlp <- function(X_tr, y_tr) {
  # Standardise inputs
  x_mu  <- colMeans(X_tr, na.rm = TRUE)
  x_sig <- apply(X_tr, 2L, sd, na.rm = TRUE)
  x_sig[x_sig < 1e-8] <- 1
  y_mu  <- mean(y_tr, na.rm = TRUE)
  y_sig <- sd(y_tr, na.rm = TRUE)
  if (is.na(y_sig) || y_sig < 1e-8) y_sig <- 1

  Xs <- scale(X_tr, center = x_mu, scale = x_sig)
  ys <- (y_tr - y_mu) / y_sig

  # Temporal validation split
  n     <- nrow(Xs)
  n_val <- max(5L, floor(n * VAL_FRAC))
  idx_t <- seq_len(n - n_val)
  idx_v <- (n - n_val + 1L):n

  x_t <- torch_tensor(Xs[idx_t, , drop = FALSE], dtype = torch_float())$to(device = device)
  y_t <- torch_tensor(matrix(ys[idx_t], ncol = 1L), dtype = torch_float())$to(device = device)
  x_v <- torch_tensor(Xs[idx_v, , drop = FALSE], dtype = torch_float())$to(device = device)
  y_v <- torch_tensor(matrix(ys[idx_v], ncol = 1L), dtype = torch_float())$to(device = device)

  dl <- dataloader(
    tensor_dataset(x_t, y_t),
    batch_size = min(BATCH_SIZE, length(idx_t)),
    shuffle    = TRUE
  )

  model <- mlp_module(ncol(X_tr), HIDDEN[1L], HIDDEN[2L], DROPOUT)$to(device = device)
  opt   <- optim_adam(model$parameters, lr = LR, weight_decay = WEIGHT_DECAY)
  sched <- lr_cosine_annealing(opt, T_max = MAX_EPOCHS, eta_min = LR_MIN)

  best_val   <- Inf
  best_state <- NULL
  no_imp     <- 0L

  for (epoch in seq_len(MAX_EPOCHS)) {
    model$train()
    coro::loop(for (batch in dl) {
      opt$zero_grad()
      loss <- loss_fn(model(batch[[1L]]), batch[[2L]])
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)  # gradient clipping
      opt$step()
    })
    sched$step()

    model$eval()
    with_no_grad({
      vl_loss <- loss_fn(model(x_v), y_v)$item()
    })

    if (!is.finite(vl_loss)) break   # catches NA, NaN, Inf
    if (vl_loss < best_val - 1e-7) {
      best_val   <- vl_loss
      best_state <- lapply(model$state_dict(), \(p) p$clone())
      no_imp     <- 0L
    } else {
      no_imp <- no_imp + 1L
      if (no_imp >= PATIENCE) break
    }
  }

  if (!is.null(best_state)) model$load_state_dict(best_state)
  model$eval()

  list(model = model, x_mu = x_mu, x_sig = x_sig, y_mu = y_mu, y_sig = y_sig)
}

# ── Prediction helper ─────────────────────────────────────────────────────────
predict_mlp <- function(fit, X_new) {
  Xs <- scale(as.matrix(X_new), center = fit$x_mu, scale = fit$x_sig)
  with_no_grad({
    pred_s <- as.numeric(
      fit$model(torch_tensor(Xs, dtype = torch_float())$to(device = device))$cpu()
    )
  })
  pred_s * fit$y_sig + fit$y_mu
}

# ── OOS loop (wrapped in a function for retraining-frequency robustness) ─────
n_oos <- length(oos_idx)

run_mlp_full <- function(retrain_freq, label = "MLP") {
  preds_mlp <- matrix(NA_real_, nrow = n_oos, ncol = length(gamma_cols))
  colnames(preds_mlp) <- gamma_cols

  cat(sprintf(
    "\nRunning %s OOS predictions (%d steps, retrain every %d month%s)\n",
    label, n_oos, retrain_freq, if (retrain_freq == 1L) "" else "s"
  ))

  # One fit object per gamma, updated every retrain_freq steps
  current_fits  <- setNames(vector("list", length(gamma_cols)), gamma_cols)
  n_skipped_na  <- 0L
  n_pred_errors <- 0L
  n_pred_nan    <- 0L

  for (i in seq_along(oos_idx)) {
    t     <- oos_idx[i]
    train <- aligned[seq_len(t - 1L), ]
    test  <- aligned[t, ]

    # Impute training NAs with column means so pre-1990 rows (VIX = NA) are
    # retained. Only rows with missing outcome are excluded.
    X_tr_raw <- as.matrix(train[, feature_cols])
    X_tr     <- impute_means(X_tr_raw)
    X_te <- as.matrix(test[, feature_cols])

    do_retrain <- (i == 1L) || (i - 1L) %% retrain_freq == 0L

    for (gc in gamma_cols) {
      ok_y <- !is.na(train[[gc]])        # only drop rows where outcome is NA
      y_tr <- train[[gc]][ok_y]

      # Retrain this gamma's model when due
      if (do_retrain && sum(ok_y) >= INIT_WINDOW) {
        # Only update the stored fit on success — assigning NULL to a named list
        # element in R deletes it, which breaks [[1L]] indexing on later steps.
        result <- tryCatch(
          fit_mlp(X_tr[ok_y, ], y_tr),
          error = function(e) {
            warning(sprintf("fit_mlp[%s] failed at step %d (%s): %s",
                            gc, i, label, conditionMessage(e)))
            NULL
          }
        )
        if (!is.null(result)) current_fits[[gc]] <- result
      }

      if (anyNA(X_te)) {
        n_skipped_na <- n_skipped_na + 1L
        next
      }
      if (is.null(current_fits[[gc]])) next

      pred <- tryCatch(
        predict_mlp(current_fits[[gc]], X_te),
        error = function(e) {
          n_pred_errors <<- n_pred_errors + 1L
          NA_real_
        }
      )
      if (is.na(pred)) {
        n_pred_nan <- n_pred_nan + 1L
      } else {
        preds_mlp[i, gc] <- pred
      }
    }

    if (i %% 50L == 0L) cat(sprintf("  step %d / %d\n", i, n_oos))
  }

  cat(sprintf(
    "Done (%s). skipped(NA in X_te)=%d  pred_errors=%d  pred_NaN=%d  filled=%d/%d\n",
    label, n_skipped_na, n_pred_errors, n_pred_nan,
    sum(!is.na(preds_mlp[, 1L])), n_oos
  ))
  preds_mlp
}

# Primary MLP run (stored as `mlp`, matching RETRAIN_FREQ)
preds_mlp <- run_mlp_full(RETRAIN_FREQ,
                          sprintf("MLP primary (retrain=%dm)", RETRAIN_FREQ))

# Robustness variants at alternative retraining frequencies
preds_mlp_robust <- list()
for (rf in setdiff(RETRAIN_FREQS, RETRAIN_FREQ)) {
  preds_mlp_robust[[sprintf("mlp_r%d", rf)]] <- run_mlp_full(
    rf, sprintf("MLP robustness (retrain=%dm)", rf)
  )
}

# ── MLP-NoMacro OOS loop (gamma-only features, no macro predictors) ─────────────
# Diagnostic variant: excludes the 6 macro columns (VIX, default_spread,
# term_spread, short_rate, indpro_gap, mkt_lag1) from the MLP input. Tests
# whether macro features add or subtract OOS value in the nonlinear model.
# Same architecture, regularisation, and retraining schedule as the full MLP.
# No macro features → no VIX imputation discontinuity → cleaner feature space.
if (RUN_MLP_NOGEO) {
  feature_cols_nogeo <- setdiff(feature_cols, MACRO_COLS_MLP)
  preds_nogeo <- matrix(NA_real_, nrow = n_oos, ncol = length(gamma_cols))
  colnames(preds_nogeo) <- gamma_cols

  cat(sprintf(
    "\nRunning MLP-NoMacro OOS predictions (%d predictors, no macro, retrain every %d months)\n",
    length(feature_cols_nogeo), RETRAIN_FREQ
  ))

  current_fits_ng <- setNames(vector("list", length(gamma_cols)), gamma_cols)
  n_skip_ng <- 0L; n_err_ng <- 0L; n_nan_ng <- 0L

  for (i in seq_along(oos_idx)) {
    t     <- oos_idx[i]
    train <- aligned[seq_len(t - 1L), ]
    test  <- aligned[t, ]

    X_tr_ng <- impute_means(as.matrix(train[, feature_cols_nogeo]))
    X_te_ng <- as.matrix(test[, feature_cols_nogeo])

    do_retrain <- (i == 1L) || (i - 1L) %% RETRAIN_FREQ == 0L

    for (gc in gamma_cols) {
      ok_y <- !is.na(train[[gc]])
      y_tr <- train[[gc]][ok_y]

      if (do_retrain && sum(ok_y) >= INIT_WINDOW) {
        result <- tryCatch(
          fit_mlp(X_tr_ng[ok_y, ], y_tr),
          error = function(e) {
            warning(sprintf("fit_mlp_nogeo[%s] step %d: %s", gc, i, conditionMessage(e)))
            NULL
          }
        )
        if (!is.null(result)) current_fits_ng[[gc]] <- result
      }

      if (anyNA(X_te_ng)) { n_skip_ng <- n_skip_ng + 1L; next }
      if (is.null(current_fits_ng[[gc]])) next

      pred <- tryCatch(
        predict_mlp(current_fits_ng[[gc]], X_te_ng),
        error = function(e) { n_err_ng <<- n_err_ng + 1L; NA_real_ }
      )
      if (is.na(pred)) {
        n_nan_ng <- n_nan_ng + 1L
      } else {
        preds_nogeo[i, gc] <- pred
      }
    }

    if (i %% 50L == 0L) cat(sprintf("  step %d / %d\n", i, n_oos))
  }

  cat(sprintf(
    "NoMacro done. skipped=%d  pred_errors=%d  pred_NaN=%d  filled=%d/%d\n",
    n_skip_ng, n_err_ng, n_nan_ng,
    sum(!is.na(preds_nogeo[, 1L])), n_oos
  ))
}

# ── Ext-A: MLP-SR OOS loop ────────────────────────────────────────────────────
# Joint 6-output MLP trained with negative Sharpe ratio loss (two-phase).
# One fit object for all 6 gammas simultaneously (retrained every RETRAIN_FREQ months).
# Reference: Gu, Kelly & Xiu (2020, RFS); Salcher, Stöckl & Hanke (2026).
if (RUN_MLP_SR) {
  preds_sr <- matrix(NA_real_, nrow = n_oos, ncol = length(gamma_cols))
  colnames(preds_sr) <- gamma_cols

  cat(sprintf(
    "\nExt-A: Running MLP-SR OOS predictions (%d steps, retrain every %d months)\n",
    n_oos, RETRAIN_FREQ
  ))

  current_fit_sr <- NULL
  n_skip_sr <- 0L; n_err_sr <- 0L; n_nan_sr <- 0L

  for (i in seq_along(oos_idx)) {
    t     <- oos_idx[i]
    train <- aligned[seq_len(t - 1L), ]
    test  <- aligned[t, ]

    X_tr_raw <- as.matrix(train[, feature_cols])
    X_tr     <- impute_means(X_tr_raw)
    X_te     <- as.matrix(test[, feature_cols])

    do_retrain <- (i == 1L) || (i - 1L) %% RETRAIN_FREQ == 0L

    if (do_retrain) {
      # Build joint Y matrix: all rows where ALL 6 gammas are observed
      ok_y  <- complete.cases(train[, gamma_cols])
      Y_tr  <- as.matrix(train[ok_y, gamma_cols])
      X_tr_ok <- X_tr[ok_y, , drop = FALSE]

      if (nrow(Y_tr) >= INIT_WINDOW) {
        result <- tryCatch(
          fit_mlp_sr(X_tr_ok, Y_tr),
          error = function(e) {
            warning(sprintf("fit_mlp_sr failed at step %d: %s", i, conditionMessage(e)))
            NULL
          }
        )
        if (!is.null(result)) current_fit_sr <- result
      }
    }

    if (anyNA(X_te)) {
      n_skip_sr <- n_skip_sr + 1L
      next
    }
    if (is.null(current_fit_sr)) next

    preds_6 <- tryCatch(
      predict_mlp_sr(current_fit_sr, X_te),
      error = function(e) { n_err_sr <<- n_err_sr + 1L; NULL }
    )
    if (!is.null(preds_6)) {
      if (any(is.na(preds_6))) {
        n_nan_sr <- n_nan_sr + 1L
      } else {
        preds_sr[i, ] <- preds_6
      }
    }

    if (i %% 50L == 0L) cat(sprintf("  step %d / %d\n", i, n_oos))
  }

  cat(sprintf(
    "MLP-SR done. skipped(NA)=%d  errors=%d  NaN=%d  filled=%d/%d\n",
    n_skip_sr, n_err_sr, n_nan_sr,
    sum(!is.na(preds_sr[, 1L])), n_oos
  ))
}

# ── Append to pred_list ───────────────────────────────────────────────────────
# Campbell-Thompson (2008) sign restriction applied to all raw MLP variants:
# if the prediction disagrees in sign with hist_mean, revert to hist_mean.
# Primary MLP + retraining-frequency robustness variants (diagnostics NoMacro/SR
# run at primary freq only).
mlp_variants <- c(list(mlp = preds_mlp), preds_mlp_robust)
if (RUN_MLP_NOGEO) mlp_variants$mlp_nogeo <- preds_nogeo
if (RUN_MLP_SR)    mlp_variants$mlp_sr    <- preds_sr

for (gc in gamma_cols) {
  bench <- pred_list[[gc]][, "hist_mean"]

  for (vname in names(mlp_variants)) {
    raw    <- mlp_variants[[vname]][, gc]
    ct_raw <- ifelse(!is.na(raw) & !is.na(bench) & sign(raw) != sign(bench),
                     bench, raw)
    pred_list[[gc]] <- cbind(
      pred_list[[gc]],
      setNames(data.frame(raw, ct_raw), c(vname, paste0(vname, "_ct")))
    )
  }
}

# ── Update oos_eval ───────────────────────────────────────────────────────────
# HAC-robust Clark-West (2007) test.
# Uses Newey-West SE with fixed lag = nw_lags (default 12, consistent with
# Fama-MacBeth inference). sandwich::NeweyWest() is used directly because
# lrvar(..., lag=k) does NOT honour the lag argument — it routes through
# bandwidth selection (bwNeweyWest) and silently ignores lag, producing a
# near-zero SE. NeweyWest(lm(f~1), lag=k) correctly applies the Bartlett
# kernel with fixed truncation.
# Plain-SE version is available as a robustness check: replace nw_se with
#   sd(f, na.rm=TRUE) / sqrt(n).
clark_west <- function(y, yhat_bench, yhat_model, nw_lags = 12L) {
  f    <- (y - yhat_bench)^2 - ((y - yhat_model)^2 - (yhat_bench - yhat_model)^2)
  ok   <- !is.na(f)
  n    <- sum(ok)
  if (n < 10L) return(list(t_cw = NA_real_, p_cw = NA_real_))
  f_ok <- f[ok]
  nw_var <- tryCatch(
    as.numeric(NeweyWest(lm(f_ok ~ 1), lag = nw_lags,
                         prewhite = FALSE, adjust = TRUE)),
    error = function(e) var(f_ok) / n
  )
  if (!is.finite(nw_var) || nw_var <= 0) nw_var <- var(f_ok) / n
  nw_se  <- sqrt(nw_var)
  t_stat <- mean(f_ok) / nw_se
  list(t_cw = round(t_stat, 3L), p_cw = round(1 - pnorm(t_stat), 3L))
}

oos_r2 <- function(y, yhat, yhat_bench) {
  1 - sum((y - yhat)^2,      na.rm = TRUE) /
      sum((y - yhat_bench)^2, na.rm = TRUE)
}

# Build list dynamically so retraining-frequency variants (mlp_r1, mlp_r12, …)
# and their _ct twins are included in the evaluation automatically.
mlp_model_cols <- unlist(lapply(names(mlp_variants),
                                \(v) c(v, paste0(v, "_ct"))))

new_eval <- map_dfr(gamma_cols, function(gc) {
  y     <- actuals[, gc]
  bench <- pred_list[[gc]][, "hist_mean"]
  map_dfr(mlp_model_cols, function(m) {
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
        mean((y[valid] - yhat[valid])^2) / mean((y[valid] - bench[valid])^2), 3L
      ),
      t_cw  = cw$t_cw,
      p_cw  = cw$p_cw,
      n_oos = sum(valid)
    )
  })
})

oos_eval <- bind_rows(
  oos_eval |> filter(!model %in% mlp_model_cols),
  new_eval
)

cat("\n=== MLP OOS R² (%) ===\n")
if (nrow(new_eval) == 0L) {
  cat("WARNING: new_eval is empty — all MLP predictions are NA.\n",
      "Check warnings() for fit_mlp failure messages.\n")
} else {
  print(
    new_eval |>
      dplyr::select(gamma, model, oos_r2, t_cw, p_cw) |>
      pivot_wider(names_from = model, values_from = c(oos_r2, t_cw, p_cw)),
    n = Inf
  )
}

# ── Save ──────────────────────────────────────────────────────────────────────
# Merge updated objects back into the existing file so aligned / feature_cols /
# oos_idx / oos_eval_all (written by 04_predict_gammas.R) are preserved.
gp <- readRDS("gamma_predictions.rds")
gp$pred_list <- pred_list; gp$oos_eval <- oos_eval
gp$actuals   <- actuals;   gp$dates_oos <- dates_oos
saveRDS(gp, file = "gamma_predictions.rds")
cat("Saved: gamma_predictions.rds (mlp added)\n")
