#!/usr/bin/env python3
"""
Multi-chain Bayesian Model Averaging imputation of ionic + electronic dielectric
for ~123K materials without DFPT data.

Inserts new rows into ALL 4 MCMC chains, runs batch_impute_column on each,
and averages predictions across chains for better calibrated uncertainty.

Input:  preprocessed/new_materials_data.npy + train_data.npy (from preprocess)
        multichain_results/chain_{0-3}.jxc (trained chains)
Output: multichain_results/predicted_dielectric_123k.csv

Usage: uv run python examples/materials_project/impute_dielectric_bma.py
"""

import gc
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from crosscat import batch_impute_column, packed_insert_rows
from crosscat.serialization import load_packed_state

# ── Configuration ──────────────────────────────────────────────
CACHE_DIR = Path("examples/materials_project/results")
PREPROCESSED_DIR = CACHE_DIR / "preprocessed"
RESULTS_DIR = CACHE_DIR / "multichain_results"
OUTPUT_CSV = RESULTS_DIR / "predicted_dielectric_123k.csv"

N_CHAINS = 4
INSERT_BATCH = 5000
PREDICT_BATCH = 500
N_SAMPLES = 500  # Higher than default (100) for better estimates
SEED = 42

print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

# ── Load preprocessed data ───────────────────────────────────
print("\nLoading preprocessed data...")
new_data = np.load(str(PREPROCESSED_DIR / "new_materials_data.npy"))
train_data = np.load(str(PREPROCESSED_DIR / "train_data.npy"))
new_meta = pd.read_parquet(PREPROCESSED_DIR / "new_materials_meta.parquet")

n_new, n_cols = new_data.shape
n_train = train_data.shape[0]
print(f"  Training data: {n_train} x {n_cols}")
print(f"  New materials: {n_new} x {n_cols}")
print(f"  New data NaN fraction: {np.isnan(new_data).mean():.1%}")

# Column indices (must match COLUMN_CATALOG order)
IONIC_COL = 3  # e_ionic
ELEC_COL = 2  # e_electronic

train_data_jax = jnp.array(train_data)

# ══════════════════════════════════════════════════════════════
# For each chain: insert new rows + predict
# ══════════════════════════════════════════════════════════════
key = jax.random.key(SEED)

# Store per-chain predictions for BMA averaging
chain_ionic_preds = []
chain_elec_preds = []

total_start = time.time()

for chain_idx in range(N_CHAINS):
    print(f"\n{'=' * 60}")
    print(f"CHAIN {chain_idx + 1}/{N_CHAINS}")
    print(f"{'=' * 60}")

    # Load chain
    chain_path = RESULTS_DIR / f"chain_{chain_idx}.jxc"
    packed, _ = load_packed_state(str(chain_path))
    print(f"  Loaded: {packed.n_rows} rows x {packed.n_cols} cols")

    # ── Insert new rows in batches ────────────────────────────
    print(f"  Inserting {n_new} rows (batches of {INSERT_BATCH})...")
    data_current = train_data_jax
    packed_current = packed
    t0 = time.time()

    for batch_start in range(0, n_new, INSERT_BATCH):
        batch_end = min(batch_start + INSERT_BATCH, n_new)
        batch_rows = jnp.array(new_data[batch_start:batch_end])

        key, subkey = jax.random.split(key)
        packed_current, data_current = packed_insert_rows(
            subkey,
            packed_current,
            data_current,
            batch_rows,
        )

        if batch_end % 25000 == 0 or batch_end == n_new:
            elapsed = time.time() - t0
            print(
                f"    {batch_end}/{n_new} inserted ({elapsed:.0f}s, "
                f"{packed_current.n_rows} total rows)",
                flush=True,
            )
        gc.collect()

    t_insert = time.time() - t0
    print(f"  Insertion done: {t_insert:.0f}s")

    # ── Predict dielectric for new rows ───────────────────────
    new_row_ids = jnp.arange(n_train, n_train + n_new)

    for col_idx, col_label in [(IONIC_COL, "ionic"), (ELEC_COL, "electronic")]:
        print(f"  Predicting {col_label} dielectric (n_samples={N_SAMPLES})...")
        all_pred = []
        t0 = time.time()

        for batch_start in range(0, n_new, PREDICT_BATCH):
            batch_end = min(batch_start + PREDICT_BATCH, n_new)
            batch_ids = new_row_ids[batch_start:batch_end]

            key, subkey = jax.random.split(key)
            pred, _ = batch_impute_column(
                subkey,
                packed_current,
                data_current,
                query_col=col_idx,
                row_ids=batch_ids,
                n_samples=N_SAMPLES,
            )
            all_pred.append(np.array(pred))

            if batch_end % 10000 == 0 or batch_end == n_new:
                elapsed = time.time() - t0
                print(f"    {batch_end}/{n_new} ({elapsed:.0f}s)", flush=True)

            gc.collect()

        preds = np.concatenate(all_pred)
        if col_label == "ionic":
            chain_ionic_preds.append(preds)
        else:
            chain_elec_preds.append(preds)

        t_pred = time.time() - t0
        print(f"  {col_label} prediction done: {t_pred:.0f}s")

    # Free memory before next chain
    del packed_current, data_current
    gc.collect()

# ══════════════════════════════════════════════════════════════
# Bayesian Model Averaging: mean + std across chains
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("BAYESIAN MODEL AVERAGING (4 chains)")
print(f"{'=' * 60}")

ionic_stack = np.stack(chain_ionic_preds)  # (4, n_new)
elec_stack = np.stack(chain_elec_preds)  # (4, n_new)

# BMA point estimate = mean across chains
ionic_bma_mean = ionic_stack.mean(axis=0)
ionic_bma_std = ionic_stack.std(axis=0)
elec_bma_mean = elec_stack.mean(axis=0)
elec_bma_std = elec_stack.std(axis=0)

# Confidence = 1 / (1 + std) — same convention as CrossCat
ionic_conf = 1.0 / (1.0 + ionic_bma_std)
elec_conf = 1.0 / (1.0 + elec_bma_std)

# Approximate 90% CI from BMA: mean +/- 1.645 * std
ionic_ci_lo = ionic_bma_mean - 1.645 * ionic_bma_std
ionic_ci_hi = ionic_bma_mean + 1.645 * ionic_bma_std
elec_ci_lo = elec_bma_mean - 1.645 * elec_bma_std
elec_ci_hi = elec_bma_mean + 1.645 * elec_bma_std

# ══════════════════════════════════════════════════════════════
# Save publishable CSV
# ══════════════════════════════════════════════════════════════
output_df = new_meta.copy()
output_df["pred_ionic_dielectric"] = ionic_bma_mean
output_df["ionic_std"] = ionic_bma_std
output_df["ionic_ci_lo"] = ionic_ci_lo
output_df["ionic_ci_hi"] = ionic_ci_hi
output_df["ionic_confidence"] = ionic_conf
output_df["pred_electronic_dielectric"] = elec_bma_mean
output_df["elec_std"] = elec_bma_std
output_df["elec_ci_lo"] = elec_ci_lo
output_df["elec_ci_hi"] = elec_ci_hi
output_df["elec_confidence"] = elec_conf

output_df.to_csv(OUTPUT_CSV, index=False)

total_time = time.time() - total_start
print(f"\nSaved: {OUTPUT_CSV}")
print(f"  {len(output_df)} materials with BMA predictions")

# Summary stats
print("\n  Ionic Dielectric (BMA):")
print(f"    Mean: {ionic_bma_mean.mean():.2f}, Median: {np.median(ionic_bma_mean):.2f}")
print(f"    Mean std across chains: {ionic_bma_std.mean():.2f}")
print(f"    Mean confidence: {ionic_conf.mean():.3f}")
print(f"    High-confidence (>0.5): {(ionic_conf > 0.5).sum()}")

print("\n  Electronic Dielectric (BMA):")
print(f"    Mean: {elec_bma_mean.mean():.2f}, Median: {np.median(elec_bma_mean):.2f}")
print(f"    Mean std across chains: {elec_bma_std.mean():.2f}")
print(f"    Mean confidence: {elec_conf.mean():.3f}")
print(f"    High-confidence (>0.5): {(elec_conf > 0.5).sum()}")

print(f"\nTotal time: {total_time:.0f}s ({total_time / 60:.1f} min)")
print("Done!")
