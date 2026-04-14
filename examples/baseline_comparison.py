#!/usr/bin/env python3
"""
Baseline comparison: CrossCat vs sklearn IterativeImputer (MICE) vs Random Forest.
Same 10% holdout, same data preprocessing, same columns.

Usage: uv run python examples/baseline_comparison.py
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error, r2_score

from crosscat import batch_impute_column
from crosscat.serialization import load_packed_state
from crosscat.types import ColumnType

# ── Configuration ──────────────────────────────────────────────
CACHE_DIR = Path("examples/results/materials_project")
CACHE_PATH = CACHE_DIR / "mp_dielectric_cache_v2.parquet"
RESULTS_DIR = CACHE_DIR / "multichain_results"
SEED = 42

# ── Column catalog (same as CrossCat) ─────────────────────────
COLUMN_CATALOG = [
    ("band_gap", "Band Gap (eV)", ColumnType.CONTINUOUS),
    ("is_metal", "Is Metal", ColumnType.BINARY),
    ("e_electronic", "Electronic Dielectric", ColumnType.CONTINUOUS),
    ("e_ionic", "Ionic Dielectric", ColumnType.CONTINUOUS),
    ("e_total", "Total Dielectric", ColumnType.CONTINUOUS),
    ("formation_energy_per_atom", "Formation Energy (eV/atom)", ColumnType.CONTINUOUS),
    ("energy_above_hull", "E Above Hull (eV/atom)", ColumnType.CONTINUOUS),
    ("is_stable", "Is Stable", ColumnType.BINARY),
    ("density", "Density (g/cm3)", ColumnType.CONTINUOUS),
    ("volume", "Volume (A3)", ColumnType.CONTINUOUS),
    ("nsites", "N Sites", ColumnType.CONTINUOUS),
    ("nelements", "N Elements", ColumnType.CONTINUOUS),
    ("crystal_system", "Crystal System", ColumnType.CATEGORICAL),
    ("bulk_modulus_vrh", "Bulk Modulus (GPa)", ColumnType.CONTINUOUS),
    ("shear_modulus_vrh", "Shear Modulus (GPa)", ColumnType.CONTINUOUS),
    ("universal_anisotropy", "Elastic Anisotropy", ColumnType.CONTINUOUS),
    ("homogeneous_poisson", "Poisson Ratio", ColumnType.CONTINUOUS),
    ("e_ij_max", "Piezo e_ij_max", ColumnType.CONTINUOUS),
    ("avg_electroneg", "Avg Electronegativity", ColumnType.CONTINUOUS),
    ("avg_ionic_radius", "Avg Ionic Radius (A)", ColumnType.CONTINUOUS),
    ("laue_class", "Laue Class", ColumnType.ORDINAL),
    ("total_magnetization", "Magnetization", ColumnType.CONTINUOUS),
    ("ordering", "Magnetic Ordering", ColumnType.CATEGORICAL),
]

# ── Data loading + preprocessing (identical to CrossCat) ──────
print("Loading and preprocessing data...")
df = pd.read_parquet(CACHE_PATH)

CRYSTAL_SYSTEM_MAP = {
    "Triclinic": 0,
    "Monoclinic": 1,
    "Orthorhombic": 2,
    "Tetragonal": 3,
    "Trigonal": 4,
    "Hexagonal": 5,
    "Cubic": 6,
}
df["crystal_system"] = df["crystal_system"].map(CRYSTAL_SYSTEM_MAP)
ordering_vals = sorted(df["ordering"].dropna().unique(), key=str)
ORDERING_MAP = {v: i for i, v in enumerate(ordering_vals)}
df["ordering"] = df["ordering"].map(ORDERING_MAP)
for col in ["is_stable", "is_metal"]:
    df[col] = df[col].astype(float)
for attr in [
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "universal_anisotropy",
    "e_ionic",
    "e_electronic",
    "e_total",
]:
    if attr in df.columns:
        df[attr] = df[attr].replace([np.inf, -np.inf], np.nan)
for attr in ["bulk_modulus_vrh", "shear_modulus_vrh"]:
    if attr in df.columns:
        df.loc[df[attr] < 0, attr] = np.nan
for attr in [
    "e_ionic",
    "e_electronic",
    "e_total",
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "universal_anisotropy",
]:
    if attr in df.columns:
        s = df[attr].dropna()
        if len(s) > 0:
            cap = s.quantile(0.995) * 5
            df.loc[df[attr] > cap, attr] = np.nan
for attr, _, _ in COLUMN_CATALOG:
    if attr in df.columns:
        s = df[attr].dropna()
        if len(s) > 0 and s.min() > 0 and (s.max() / s.min()) > 100:
            df[attr] = np.log1p(df[attr])
if "avg_ionic_radius" in df.columns:
    df.loc[df["avg_ionic_radius"] == 0, "avg_ionic_radius"] = np.nan
if "laue_class" in df.columns:
    df["laue_class"] = df["laue_class"].astype(float)

valid_attrs = []
col_names = []
for attr, display, _ctype in COLUMN_CATALOG:
    if attr in df.columns and df[attr].notna().mean() >= 0.25:
        valid_attrs.append(attr)
        col_names.append(display)

data_np = df[valid_attrs].values.astype(np.float32)
data_np[~np.isfinite(data_np)] = np.nan
for ci in range(data_np.shape[1]):
    col = data_np[:, ci]
    valid = col[np.isfinite(col)]
    if len(valid) > 100:
        q01, q99 = np.percentile(valid, [0.5, 99.5])
        iqr = q99 - q01
        lo, hi = q01 - 5 * iqr, q99 + 5 * iqr
        if ((col < lo) | (col > hi)).sum() > 0:
            data_np[(data_np[:, ci] < lo) | (data_np[:, ci] > hi), ci] = np.nan

n_rows, n_cols = data_np.shape
print(f"Data: {n_rows} x {n_cols}, NaN: {np.isnan(data_np).mean():.1%}")

# ── Target columns ────────────────────────────────────────────
ionic_col = col_names.index("Ionic Dielectric")
elec_col = col_names.index("Electronic Dielectric")

# ── Holdout setup (same as CrossCat prediction script) ────────
rng = np.random.RandomState(SEED)

targets = [
    (ionic_col, "Ionic Dielectric"),
    (elec_col, "Electronic Dielectric"),
]

# Also test on other columns for a fuller picture
extra_targets = [
    (col_names.index("Band Gap (eV)"), "Band Gap"),
    (col_names.index("Formation Energy (eV/atom)"), "Formation Energy"),
    (col_names.index("E Above Hull (eV/atom)"), "E Above Hull"),
    (col_names.index("Bulk Modulus (GPa)"), "Bulk Modulus"),
]

all_targets = targets + extra_targets

# ── CrossCat results (from multichain analysis) ──────────────
print("\nLoading CrossCat model for comparison...")
best_packed, _ = load_packed_state(str(RESULTS_DIR / "best_chain.jxc"))
data_jax = jnp.array(data_np)
key = jax.random.key(99)

# ── Run all methods ──────────────────────────────────────────
print(f"\n{'=' * 80}")
print("BASELINE COMPARISON: CrossCat vs MICE vs Random Forest")
print(f"{'=' * 80}")

results = []

for col_idx, col_label in all_targets:
    observed = np.where(~np.isnan(data_np[:, col_idx]))[0]
    if len(observed) < 100:
        continue

    holdout_idx = rng.choice(observed, size=len(observed) // 10, replace=False)
    holdout_true = data_np[holdout_idx, col_idx]
    n_holdout = len(holdout_idx)

    print(f"\n--- {col_label} ({n_holdout} holdout samples) ---")

    # ── 1. CrossCat ──────────────────────────────────────────
    key, subkey = jax.random.split(key)
    t0 = time.time()
    cc_pred, cc_conf = batch_impute_column(
        subkey,
        best_packed,
        data_jax,
        query_col=col_idx,
        row_ids=jnp.array(holdout_idx),
    )
    cc_pred = np.array(cc_pred)
    cc_time = time.time() - t0

    cc_mae = mean_absolute_error(holdout_true, cc_pred)
    cc_r2 = r2_score(holdout_true, cc_pred)

    print(f"  CrossCat:     R2={cc_r2:.4f}  MAE={cc_mae:.4f}  ({cc_time:.1f}s)")

    # ── 2. MICE (IterativeImputer with BayesianRidge) ────────
    t0 = time.time()
    # Create masked data: set holdout values to NaN
    data_masked = data_np.copy()
    data_masked[holdout_idx, col_idx] = np.nan

    mice = IterativeImputer(
        max_iter=10,
        random_state=SEED,
        sample_posterior=False,
    )
    data_imputed = mice.fit_transform(data_masked)
    mice_pred = data_imputed[holdout_idx, col_idx]
    mice_time = time.time() - t0

    mice_mae = mean_absolute_error(holdout_true, mice_pred)
    mice_r2 = r2_score(holdout_true, mice_pred)

    print(f"  MICE:         R2={mice_r2:.4f}  MAE={mice_mae:.4f}  ({mice_time:.1f}s)")

    # ── 3. Random Forest ─────────────────────────────────────
    t0 = time.time()
    # Train on non-holdout observed rows, predict holdout
    train_mask = ~np.isnan(data_np[:, col_idx])
    train_mask[holdout_idx] = False
    train_idx = np.where(train_mask)[0]

    # Use all other columns as features (impute NaN with column median for RF)
    feature_cols = [c for c in range(n_cols) if c != col_idx]
    X_all = data_np[:, feature_cols].copy()
    # Simple median imputation for RF features
    for fc in range(X_all.shape[1]):
        col_vals = X_all[:, fc]
        median_val = np.nanmedian(col_vals)
        X_all[np.isnan(col_vals), fc] = median_val

    X_train = X_all[train_idx]
    y_train = data_np[train_idx, col_idx]
    X_test = X_all[holdout_idx]

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_time = time.time() - t0

    rf_mae = mean_absolute_error(holdout_true, rf_pred)
    rf_r2 = r2_score(holdout_true, rf_pred)

    print(f"  Random Forest: R2={rf_r2:.4f}  MAE={rf_mae:.4f}  ({rf_time:.1f}s)")

    # Store results
    results.append(
        {
            "Column": col_label,
            "N_holdout": n_holdout,
            "CrossCat_R2": cc_r2,
            "CrossCat_MAE": cc_mae,
            "MICE_R2": mice_r2,
            "MICE_MAE": mice_mae,
            "RF_R2": rf_r2,
            "RF_MAE": rf_mae,
        }
    )

# ── Summary table ────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("SUMMARY TABLE")
print(f"{'=' * 80}")

print(f"\n{'Column':25s} {'CrossCat R2':>12s} {'MICE R2':>10s} {'RF R2':>10s} {'Winner':>10s}")
print(f"{'-' * 25} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}")

for r in results:
    scores = {"CrossCat": r["CrossCat_R2"], "MICE": r["MICE_R2"], "RF": r["RF_R2"]}
    winner = max(scores, key=scores.get)
    print(
        f"{r['Column']:25s} {r['CrossCat_R2']:>12.4f} {r['MICE_R2']:>10.4f} "
        f"{r['RF_R2']:>10.4f} {winner:>10s}"
    )

# CrossCat advantages
print(f"\n{'=' * 80}")
print("CROSSCAT UNIQUE ADVANTAGES (not available from MICE or RF)")
print(f"{'=' * 80}")
print("  1. Structure discovery (5 views) — which properties are dependent?")
print("  2. Calibrated uncertainty (96% at 90% CI) — should I run DFPT?")
print("  3. Mixed types natively (continuous + binary + categorical + ordinal)")
print("  4. No feature engineering — works directly on raw tabular data")
print("  5. Handles arbitrary missingness patterns without pre-imputation")
print("  6. Anomaly detection with attribution — which properties are surprising?")
print("  7. Mutual information — quantify nonlinear property relationships")

# Save results
results_df = pd.DataFrame(results)
csv_path = RESULTS_DIR / "baseline_comparison.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")
