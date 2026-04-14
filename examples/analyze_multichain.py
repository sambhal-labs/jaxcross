#!/usr/bin/env python3
"""
Full multi-chain analysis of 4 chains from local GTX 1650 run.
Produces: Rhat convergence, Z-matrix, view structure, anomalies,
imputation evaluation, mutual information, classification, typicality.

Usage: uv run python examples/analyze_multichain.py
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from crosscat import (
    batch_anomaly_score,
    batch_classify_column,
    batch_impute_column,
    batch_row_typicality,
    packed_dependence_matrix,
    packed_log_joint,
    packed_mutual_information,
)
from crosscat.diagnostics import effective_sample_size, gelman_rubin_rhat
from crosscat.packed import unpack_state
from crosscat.serialization import load_packed_state
from crosscat.types import ColumnType

# ── Configuration ──────────────────────────────────────────────
CACHE_DIR = Path("examples/results/materials_project")
CACHE_PATH = CACHE_DIR / "mp_dielectric_cache_v2.parquet"
RESULTS_DIR = CACHE_DIR / "multichain_results"
SEED = 99

print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

# ── Column catalog ────────────────────────────────────────────
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

LAUE_MAP = {
    "1": 0,
    "-1": 0,
    "2": 1,
    "m": 1,
    "2/m": 1,
    "222": 2,
    "mm2": 2,
    "mmm": 2,
    "4": 3,
    "-4": 3,
    "4/m": 3,
    "422": 4,
    "4mm": 4,
    "-42m": 4,
    "-4m2": 4,
    "4/mmm": 4,
    "3": 5,
    "-3": 5,
    "32": 6,
    "3m": 6,
    "-3m": 6,
    "6": 7,
    "-6": 7,
    "6/m": 7,
    "622": 8,
    "6mm": 8,
    "-6m2": 8,
    "-62m": 8,
    "6/mmm": 8,
    "23": 9,
    "m-3": 9,
    "432": 10,
    "-43m": 10,
    "m-3m": 10,
}

# ── Load and preprocess data ─────────────────────────────────
print(f"\nLoading data from {CACHE_PATH}")
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

valid_attrs, column_types, col_names = [], [], []
for attr, display, ctype in COLUMN_CATALOG:
    if attr in df.columns and df[attr].notna().mean() >= 0.25:
        valid_attrs.append(attr)
        column_types.append(ctype)
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
        n_clamped = int(((col < lo) | (col > hi)).sum())
        if n_clamped > 0:
            data_np[(data_np[:, ci] < lo) | (data_np[:, ci] > hi), ci] = np.nan

data_jax = jnp.array(data_np)
n_rows, n_cols = data_jax.shape
print(f"Data: {n_rows} rows x {n_cols} cols, NaN: {float(jnp.isnan(data_jax).mean()):.1%}")

# ── Load all chains ──────────────────────────────────────────
print(f"\nLoading chains from {RESULTS_DIR}")
all_chains = []
for i in range(4):
    path = RESULTS_DIR / f"chain_{i}.jxc"
    packed, ct = load_packed_state(str(path))
    all_chains.append(packed)
    lj = float(packed_log_joint(packed, data_jax))
    print(f"  Chain {i}: log_joint={lj:,.1f}")

best_packed, _ = load_packed_state(str(RESULTS_DIR / "best_chain.jxc"))
best_lj = float(packed_log_joint(best_packed, data_jax))
print(f"  Best chain: log_joint={best_lj:,.1f}")

log_joint_traces = np.load(str(RESULTS_DIR / "log_joint_traces.npy"))
print(f"  Traces shape: {log_joint_traces.shape}")

t_start = time.time()

# ══════════════════════════════════════════════════════════════
# 1. CONVERGENCE DIAGNOSTICS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("1. CONVERGENCE DIAGNOSTICS")
print(f"{'=' * 70}")

traces_jax = jnp.array(log_joint_traces)
rhat = float(gelman_rubin_rhat(traces_jax))
ess = float(effective_sample_size(traces_jax))
print(f"  Gelman-Rubin Rhat: {rhat:.4f}")
print(f"  Effective Sample Size: {ess:.1f}")
print(f"  Log-joint range: [{float(traces_jax.min()):,.1f}, {float(traces_jax.max()):,.1f}]")
print(f"  Log-joint std across chains: {float(traces_jax[:, -1].std()):.1f}")

if rhat < 1.1:
    print("  Status: CONVERGED (Rhat < 1.1)")
elif rhat < 1.2:
    print("  Status: APPROXIMATELY CONVERGED (1.1 < Rhat < 1.2)")
else:
    print("  Status: NOT CONVERGED (Rhat > 1.2)")
    print("  Note: CrossCat's combinatorial partition space means chains explore")
    print("  different posterior modes. Stable log-joint traces are more informative.")

# ══════════════════════════════════════════════════════════════
# 2. VIEW STRUCTURE (per chain + consensus)
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("2. VIEW STRUCTURE")
print(f"{'=' * 70}")

for ci, packed in enumerate(all_chains):
    state = unpack_state(packed, column_types)
    print(f"\n  Chain {ci} ({len(state.views)} views):")
    for vi, view in enumerate(state.views):
        cols = [col_names[c] for c in view.column_indices]
        n_clusters = len(set(int(x) for x in np.unique(np.array(view.row_assignments))))
        print(f"    View {vi}: {n_clusters} clusters, {len(cols)} cols: {cols}")

# ══════════════════════════════════════════════════════════════
# 3. DEPENDENCE STRUCTURE (Z-Matrix averaged over all chains)
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("3. DEPENDENCE STRUCTURE (Z-Matrix, 4-chain average)")
print(f"{'=' * 70}")

z_matrix = np.array(packed_dependence_matrix(all_chains))
print(f"  Z-matrix shape: {z_matrix.shape}")

pairs = []
for i in range(n_cols):
    for j in range(i + 1, n_cols):
        pairs.append((col_names[i], col_names[j], z_matrix[i, j]))
pairs.sort(key=lambda x: x[2], reverse=True)

print("\n  Top 15 dependencies:")
for a, b, dep in pairs[:15]:
    print(f"    {dep:.3f}  {a} <-> {b}")

print("\n  Top 10 independent pairs:")
for a, b, dep in pairs[-10:]:
    print(f"    {dep:.3f}  {a} <-> {b}")

# Count views per chain to show structural diversity
print("\n  Structural diversity:")
for ci, packed in enumerate(all_chains):
    state = unpack_state(packed, column_types)
    print(f"    Chain {ci}: {len(state.views)} views")

# ══════════════════════════════════════════════════════════════
# 4. ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("4. ANOMALY DETECTION (best chain)")
print(f"{'=' * 70}")

anom_scores = np.array(batch_anomaly_score(best_packed, data_jax, jnp.arange(n_rows)))
top_idx = np.argsort(anom_scores)[:20]

print(f"  Score range: [{anom_scores.min():.4f}, {anom_scores.max():.4f}]")
print(f"  Mean: {anom_scores.mean():.4f}, Std: {anom_scores.std():.4f}")
print("\n  Top 20 anomalous materials:")
for rank, idx in enumerate(top_idx):
    formula = df.iloc[idx].get("formula_pretty", "N/A")
    print(f"    #{rank + 1}: {formula:20s} (row {idx}, score={anom_scores[idx]:.4f})")

# ══════════════════════════════════════════════════════════════
# 5. IMPUTATION EVALUATION (10% holdout)
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("5. IMPUTATION EVALUATION (10% holdout, best chain)")
print(f"{'=' * 70}")

key = jax.random.key(SEED)
test_cols = [
    ("Band Gap (eV)", 0),
    ("Electronic Dielectric", 2),
    ("Ionic Dielectric", 3),
    ("Formation Energy (eV/atom)", 5),
    ("E Above Hull (eV/atom)", 6),
    ("Density (g/cm3)", 8),
    ("Crystal System", 12),
    ("Bulk Modulus (GPa)", 13),
    ("Shear Modulus (GPa)", 14),
    ("Elastic Anisotropy", 15),
    ("Poisson Ratio", 16),
    ("Avg Electronegativity", 18),
    ("Laue Class", 20),
]

print(f"  {'Column':30s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s} {'Conf':>8s} {'N':>6s}")
print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6}")

all_r2 = []
for col_label, col_idx in test_cols:
    if col_idx >= n_cols:
        continue
    observed_mask = ~np.isnan(data_np[:, col_idx])
    observed_rows = np.where(observed_mask)[0]
    if len(observed_rows) < 100:
        continue

    rng = np.random.RandomState(42)
    holdout_idx = rng.choice(observed_rows, size=len(observed_rows) // 10, replace=False)
    holdout_true = data_np[holdout_idx, col_idx]

    key, subkey = jax.random.split(key)
    imputed, conf = batch_impute_column(
        subkey,
        best_packed,
        data_jax,
        query_col=col_idx,
        row_ids=jnp.array(holdout_idx),
    )
    imputed = np.array(imputed)
    conf = np.array(conf)

    mae = np.mean(np.abs(imputed - holdout_true))
    rmse = np.sqrt(np.mean((imputed - holdout_true) ** 2))
    ss_res = np.sum((holdout_true - imputed) ** 2)
    ss_tot = np.sum((holdout_true - np.mean(holdout_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    avg_conf = np.mean(conf)
    all_r2.append(r2)

    n_ho = len(holdout_idx)
    print(f"  {col_label:30s} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {avg_conf:>8.3f} {n_ho:>6d}")

print(f"\n  Mean R2: {np.mean(all_r2):.4f}")
print(f"  Median R2: {np.median(all_r2):.4f}")
print(f"  Columns with R2 > 0: {sum(1 for r in all_r2 if r > 0)}/{len(all_r2)}")

# ══════════════════════════════════════════════════════════════
# 6. MUTUAL INFORMATION (all chains)
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("6. MUTUAL INFORMATION (4-chain average)")
print(f"{'=' * 70}")

mi_pairs = [
    (0, 2, "Band Gap <-> Electronic Dielectric"),
    (0, 1, "Band Gap <-> Is Metal"),
    (13, 14, "Bulk Modulus <-> Shear Modulus"),
    (0, 12, "Band Gap <-> Crystal System"),
    (5, 6, "Formation Energy <-> E Above Hull"),
    (8, 9, "Density <-> Volume"),
    (12, 20, "Crystal System <-> Laue Class"),
    (18, 19, "Avg Electronegativity <-> Avg Ionic Radius"),
    (0, 20, "Band Gap <-> Laue Class"),
    (13, 20, "Bulk Modulus <-> Laue Class"),
    (0, 5, "Band Gap <-> Formation Energy"),
    (2, 3, "Electronic Dielectric <-> Ionic Dielectric"),
]

print(f"  {'Pair':45s} {'MI':>8s} {'Linfoot':>8s}")
print(f"  {'-' * 45} {'-' * 8} {'-' * 8}")

for col_i, col_j, pair_label in mi_pairs:
    if col_i >= n_cols or col_j >= n_cols:
        continue
    mi_val, mi_std = packed_mutual_information(
        all_chains,
        column_types,
        col_i=col_i,
        col_j=col_j,
        rng_key=jax.random.key(SEED),
    )
    mi = float(mi_val)
    linfoot = float(np.sqrt(1 - np.exp(-2 * mi))) if mi > 0 else 0.0
    print(f"  {pair_label:45s} {mi:>8.4f} {linfoot:>8.3f}")

# ══════════════════════════════════════════════════════════════
# 7. CLASSIFICATION (metallicity)
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("7. METALLICITY CLASSIFICATION (best chain)")
print(f"{'=' * 70}")

is_metal_col = 1
candidate_vals = jnp.array([0.0, 1.0])
log_probs = np.array(
    batch_classify_column(
        best_packed,
        data_jax,
        target_col=is_metal_col,
        candidate_vals=candidate_vals,
        row_ids=jnp.arange(n_rows),
    )
)
probs = np.exp(log_probs)
probs = probs / probs.sum(axis=1, keepdims=True)
predictions = probs[:, 1]

true_labels = data_np[:, is_metal_col]
valid = ~np.isnan(true_labels)
pred_valid = predictions[valid]
true_valid = true_labels[valid]

best_f1, best_thresh = 0, 0.5
for thresh in np.arange(0.05, 0.95, 0.05):
    pred_binary = (pred_valid >= thresh).astype(float)
    tp = np.sum((pred_binary == 1) & (true_valid == 1))
    fp = np.sum((pred_binary == 1) & (true_valid == 0))
    fn = np.sum((pred_binary == 0) & (true_valid == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

pred_binary = (pred_valid >= best_thresh).astype(float)
tp = np.sum((pred_binary == 1) & (true_valid == 1))
fp = np.sum((pred_binary == 1) & (true_valid == 0))
fn = np.sum((pred_binary == 0) & (true_valid == 1))
tn = np.sum((pred_binary == 0) & (true_valid == 0))

n_metal = int(true_valid.sum())
n_nonmetal = int(len(true_valid) - n_metal)
print(
    f"  Class balance: {n_metal} metals ({n_metal / len(true_valid):.1%}), {n_nonmetal} non-metals"
)
print(f"  Optimal threshold: {best_thresh:.2f}")
print(f"  Accuracy:  {(tp + tn) / len(true_valid):.4f}")
print(f"  Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
print(f"  Recall:    {tp / (tp + fn) if (tp + fn) > 0 else 0:.4f}")
print(f"  F1:        {best_f1:.4f}")
print(f"  TP={int(tp)}, FP={int(fp)}, FN={int(fn)}, TN={int(tn)}")

# ══════════════════════════════════════════════════════════════
# 8. ROW TYPICALITY
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("8. ROW TYPICALITY (all chains)")
print(f"{'=' * 70}")

typ_scores = np.array(batch_row_typicality(all_chains, jnp.arange(n_rows)))
print(f"  Mean: {typ_scores.mean():.4f}, Std: {typ_scores.std():.4f}")
print(f"  Min: {typ_scores.min():.4f}, Max: {typ_scores.max():.4f}")

bottom_idx = np.argsort(typ_scores)[:10]
print("\n  Least typical materials:")
for idx in bottom_idx:
    formula = df.iloc[idx].get("formula_pretty", "N/A")
    print(f"    {formula:20s} (row {idx}): {typ_scores[idx]:.4f}")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
elapsed = time.time() - t_start
print(f"\n{'=' * 70}")
print("SUMMARY: MATERIALS PROJECT CrossCat MODEL ASSESSMENT")
print(f"{'=' * 70}")
print(f"""
Dataset: {n_rows} materials, {n_cols} properties (mixed types)
  - {sum(1 for ct in column_types if ct == ColumnType.CONTINUOUS)} continuous, \
{sum(1 for ct in column_types if ct == ColumnType.BINARY)} binary, \
{sum(1 for ct in column_types if ct == ColumnType.CATEGORICAL)} categorical, \
{sum(1 for ct in column_types if ct == ColumnType.ORDINAL)} ordinal
  - NaN fraction: {float(jnp.isnan(data_jax).mean()):.1%}

Inference: 4 chains x 100 sweeps from sweep-300 checkpoint (GTX 1650)
  - Rhat: {rhat:.4f}, ESS: {ess:.1f}
  - Best log-joint: {best_lj:,.1f}

Imputation (10% holdout):
  - Mean R2: {np.mean(all_r2):.4f}, Median R2: {np.median(all_r2):.4f}
  - Columns with R2 > 0: {sum(1 for r in all_r2 if r > 0)}/{len(all_r2)}

Classification (metallicity):
  - F1: {best_f1:.4f} (threshold: {best_thresh:.2f})
  - {n_metal} metals / {n_nonmetal} non-metals ({n_metal / len(true_valid):.1%} class imbalance)

Analysis time: {elapsed:.0f}s
""")
