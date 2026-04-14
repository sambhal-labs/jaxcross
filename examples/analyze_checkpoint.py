#!/usr/bin/env python3
"""
Analyze v2 checkpoints: sweep-500 (Kaggle) vs sweep-650 (local GPU).
Runs inference queries and produces a detailed comparison report.

Usage: uv run python examples/analyze_checkpoint.py
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pymatgen.symmetry.groups import SpaceGroup

from crosscat import (
    batch_anomaly_score,
    batch_classify_column,
    batch_impute_column,
    batch_row_typicality,
    packed_dependence_matrix,
    packed_log_joint,
    packed_mutual_information,
)
from crosscat.packed import unpack_state
from crosscat.serialization import load_packed_state
from crosscat.types import ColumnType

# ── Configuration ──────────────────────────────────────────────
CACHE_DIR = Path("examples/results/materials_project")
CACHE_PATH = CACHE_DIR / "mp_dielectric_cache_v2.parquet"
CKPT_DIR = CACHE_DIR / "checkpoints_v2"
SEED = 99

print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

# ── Column catalog (must match v2 notebook) ───────────────────
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

# ── Laue class mapping ────────────────────────────────────────
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


def sg_to_laue_int(sg_number):
    try:
        sg = SpaceGroup.from_int_number(int(sg_number))
        return LAUE_MAP.get(sg.point_group)
    except Exception:
        return None


# ── Load and preprocess data ─────────────────────────────────
print(f"\nLoading cached data from {CACHE_PATH}")
df = pd.read_parquet(CACHE_PATH)
print(f"Loaded: {df.shape}")

# Preprocess (same as run_local_sweeps.py)
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
    if col in df.columns:
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

# Build JAX array
MIN_COVERAGE = 0.25
valid_attrs = []
column_types = []
col_names = []
for attr, display, ctype in COLUMN_CATALOG:
    if attr in df.columns and df[attr].notna().mean() >= MIN_COVERAGE:
        valid_attrs.append(attr)
        column_types.append(ctype)
        col_names.append(display)

df_model = df[valid_attrs]
data_np = df_model.values.astype(np.float32)

# Sanitize after float32 cast: replace inf/-inf and extreme outliers with NaN
data_np[~np.isfinite(data_np)] = np.nan
for ci in range(data_np.shape[1]):
    col = data_np[:, ci]
    valid_col = col[np.isfinite(col)]
    if len(valid_col) > 100:
        q01, q99 = np.percentile(valid_col, [0.5, 99.5])
        iqr = q99 - q01
        lo, hi = q01 - 5 * iqr, q99 + 5 * iqr
        n_clamped = int(((col < lo) | (col > hi)).sum())
        if n_clamped > 0:
            data_np[(data_np[:, ci] < lo) | (data_np[:, ci] > hi), ci] = np.nan
            print(f"  Clamped {n_clamped} extreme values in col {ci} ({col_names[ci]})")

data_jax = jnp.array(data_np)
n_rows, n_cols = data_jax.shape
print(f"Data array: {n_rows} rows x {n_cols} cols")
print(f"Columns: {col_names}")


# ── Load both checkpoints ────────────────────────────────────
LOCAL_CKPT_DIR = CACHE_DIR / "checkpoints_v2_local"

print("\n" + "=" * 70)
print("LOADING CHECKPOINTS")
print("=" * 70)

kaggle_path = CKPT_DIR / "checkpoint_sweep_000500.jxc"
local_path = LOCAL_CKPT_DIR / "checkpoint_sweep_000300.jxc"

print(f"  Kaggle: {kaggle_path}")
print(f"  Local:  {local_path}")

packed_500, ct_500 = load_packed_state(str(kaggle_path))
packed_local, ct_local = load_packed_state(str(local_path))

print(f"\nKaggle sweep-500: {packed_500.n_rows} rows x {packed_500.n_cols} cols")
print(f"Local  sweep-100: {packed_local.n_rows} rows x {packed_local.n_cols} cols")


# ── Analysis functions ────────────────────────────────────────
def analyze_structure(packed, label):
    """Analyze view structure of a checkpoint."""
    print(f"\n--- View Structure ({label}) ---")
    state = unpack_state(packed, column_types)
    for i, view in enumerate(state.views):
        cols_in_view = [col_names[c] for c in view.column_indices]
        n_clusters = len(set(int(x) for x in np.unique(np.array(view.row_assignments))))
        print(f"  View {i}: {n_clusters} clusters, {len(cols_in_view)} cols: {cols_in_view}")
    return state


def analyze_log_joint(packed, label):
    """Compute log-joint score."""
    lj = float(packed_log_joint(packed, data_jax))
    print(f"  Log-joint ({label}): {lj:,.1f}")
    return lj


def analyze_dependence(packed, label):
    """Compute dependence matrix."""
    z = np.array(packed_dependence_matrix([packed]))
    print(f"\n--- Top Dependencies ({label}) ---")
    pairs = []
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            pairs.append((col_names[i], col_names[j], z[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for a, b, dep in pairs[:15]:
        print(f"  {dep:.3f}  {a} <-> {b}")

    print(f"\n--- Top Independent Pairs ({label}) ---")
    for a, b, dep in pairs[-10:]:
        print(f"  {dep:.3f}  {a} <-> {b}")
    return z


def analyze_anomalies(packed, label, top_n=15):
    """Score all rows for anomalies."""
    scores = np.array(batch_anomaly_score(packed, data_jax, jnp.arange(n_rows)))
    print(f"\n--- Top {top_n} Anomalies ({label}) ---")
    top_idx = np.argsort(scores)[:top_n]
    for rank, idx in enumerate(top_idx):
        formula = df.iloc[idx].get("formula_pretty", "N/A")
        print(f"  #{rank + 1}: row {idx} ({formula}), score={scores[idx]:.4f}")
    return scores


def analyze_imputation(packed, label):
    """Run holdout imputation evaluation on key columns."""
    key = jax.random.key(SEED)
    print(f"\n--- Imputation Quality ({label}) ---")
    results = {}

    # Test columns: band_gap, bulk_modulus, e_total, crystal_system, laue_class
    test_cols = {
        "Band Gap (eV)": 0,
        "E Above Hull (eV/atom)": 6,
        "Bulk Modulus (GPa)": 13,
        "Shear Modulus (GPa)": 14,
        "Crystal System": 12,
        "Laue Class": 20,
    }

    for col_label, col_idx in test_cols.items():
        if col_idx >= n_cols:
            continue

        # Get rows with observed values
        observed_mask = ~np.isnan(data_np[:, col_idx])
        observed_rows = np.where(observed_mask)[0]

        if len(observed_rows) < 100:
            print(f"  {col_label}: too few observed values ({len(observed_rows)})")
            continue

        # 10% holdout
        rng = np.random.RandomState(42)
        holdout_idx = rng.choice(observed_rows, size=len(observed_rows) // 10, replace=False)
        holdout_true = data_np[holdout_idx, col_idx]

        # Impute
        key, subkey = jax.random.split(key)
        imputed, conf = batch_impute_column(
            subkey,
            packed,
            data_jax,
            query_col=col_idx,
            row_ids=jnp.array(holdout_idx),
        )
        imputed = np.array(imputed)
        conf = np.array(conf)

        # Metrics
        mae = np.mean(np.abs(imputed - holdout_true))
        rmse = np.sqrt(np.mean((imputed - holdout_true) ** 2))
        ss_res = np.sum((holdout_true - imputed) ** 2)
        ss_tot = np.sum((holdout_true - np.mean(holdout_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        avg_conf = np.mean(conf)

        results[col_label] = {"MAE": mae, "RMSE": rmse, "R2": r2, "Conf": avg_conf}
        print(
            f"  {col_label:30s}: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  Conf={avg_conf:.3f}"
        )

    return results


def analyze_mutual_information(packed, label):
    """Compute MI for key physics pairs."""
    print(f"\n--- Mutual Information ({label}) ---")
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
    ]

    results = {}
    for col_i, col_j, pair_label in mi_pairs:
        if col_i >= n_cols or col_j >= n_cols:
            continue
        mi_val, mi_std = packed_mutual_information(
            [packed],
            column_types,
            col_i=col_i,
            col_j=col_j,
            rng_key=jax.random.key(SEED),
        )
        mi = float(mi_val)
        linfoot = float(np.sqrt(1 - np.exp(-2 * mi))) if mi > 0 else 0.0
        results[pair_label] = {"MI": mi, "Linfoot": linfoot}
        print(f"  {pair_label:45s}: MI={mi:.4f}  Linfoot={linfoot:.3f}")

    return results


def analyze_classification(packed, label):
    """Run metallicity classification."""
    is_metal_col = 1  # is_metal column index
    candidate_vals = jnp.array([0.0, 1.0])  # binary: not metal, metal

    # batch_classify_column returns log P(target=v | row) for all rows x candidates
    log_probs = np.array(
        batch_classify_column(
            packed,
            data_jax,
            target_col=is_metal_col,
            candidate_vals=candidate_vals,
            row_ids=jnp.arange(n_rows),
        )
    )
    # log_probs shape: (n_rows, 2) — convert to P(metal)
    probs = np.exp(log_probs)
    probs = probs / probs.sum(axis=1, keepdims=True)
    predictions = probs[:, 1]  # P(is_metal=1)

    true_labels = data_np[:, is_metal_col]
    valid = ~np.isnan(true_labels)

    pred_valid = predictions[valid]
    true_valid = true_labels[valid]

    # Optimal threshold via F1
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.05):
        pred_binary = (pred_valid >= thresh).astype(float)
        tp = np.sum((pred_binary == 1) & (true_valid == 1))
        fp = np.sum((pred_binary == 1) & (true_valid == 0))
        fn = np.sum((pred_binary == 0) & (true_valid == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    pred_binary = (pred_valid >= best_thresh).astype(float)
    tp = np.sum((pred_binary == 1) & (true_valid == 1))
    fp = np.sum((pred_binary == 1) & (true_valid == 0))
    fn = np.sum((pred_binary == 0) & (true_valid == 1))
    tn = np.sum((pred_binary == 0) & (true_valid == 0))
    accuracy = (tp + tn) / len(true_valid)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n--- Metallicity Classification ({label}) ---")
    print(f"  Threshold: {best_thresh:.2f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP={int(tp)}, FP={int(fp)}, FN={int(fn)}, TN={int(tn)}")

    return {
        "threshold": best_thresh,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def analyze_typicality(packed, label, bottom_n=10):
    """Score row typicality."""
    scores = np.array(batch_row_typicality([packed], jnp.arange(n_rows)))
    print(f"\n--- Row Typicality ({label}) ---")
    print(f"  Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
    print(f"  Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}")
    bottom_idx = np.argsort(scores)[:bottom_n]
    print("  Least typical rows:")
    for idx in bottom_idx:
        formula = df.iloc[idx].get("formula_pretty", "N/A")
        print(f"    row {idx} ({formula}): {scores[idx]:.4f}")
    return scores


# ── Run full analysis on both checkpoints ─────────────────────
print("\n" + "=" * 70)
print("ANALYSIS: KAGGLE SWEEP-500 (2xT4, 10 chains, best chain)")
print("=" * 70)

t0 = time.time()
lj_500 = analyze_log_joint(packed_500, "kaggle-500")
struct_500 = analyze_structure(packed_500, "kaggle-500")
dep_500 = analyze_dependence(packed_500, "kaggle-500")
anom_500 = analyze_anomalies(packed_500, "kaggle-500")
imp_500 = analyze_imputation(packed_500, "kaggle-500")
mi_500 = analyze_mutual_information(packed_500, "kaggle-500")
cls_500 = analyze_classification(packed_500, "kaggle-500")
typ_500 = analyze_typicality(packed_500, "kaggle-500")
t_500 = time.time() - t0
print(f"\n[Kaggle-500 analysis took {t_500:.1f}s]")

print("\n" + "=" * 70)
print("ANALYSIS: LOCAL SWEEP-100 (GTX 1650, single chain, clean data)")
print("=" * 70)

t0 = time.time()
lj_local = analyze_log_joint(packed_local, "local-100")
struct_local = analyze_structure(packed_local, "local-100")
dep_local = analyze_dependence(packed_local, "local-100")
anom_local = analyze_anomalies(packed_local, "local-100")
imp_local = analyze_imputation(packed_local, "local-100")
mi_local = analyze_mutual_information(packed_local, "local-100")
cls_local = analyze_classification(packed_local, "local-100")
typ_local = analyze_typicality(packed_local, "local-100")
t_local = time.time() - t0
print(f"\n[Local-100 analysis took {t_local:.1f}s]")


# ── Comparison Report ─────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON REPORT: KAGGLE SWEEP-500 vs LOCAL SWEEP-100")
print("=" * 70)

print("\n1. LOG-JOINT SCORE")
print(f"   Kaggle-500: {lj_500:>15,.1f}")
print(f"   Local-100:  {lj_local:>15,.1f}")
delta_lj = lj_local - lj_500
print(f"   Delta:      {delta_lj:>15,.1f} ({'improved' if delta_lj > 0 else 'decreased'})")

print("\n2. IMPUTATION QUALITY (R2 comparison)")
print(f"   {'Column':30s} {'Kaggle':>10s} {'Local':>10s} {'Delta':>10s}")
print(f"   {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10}")
for col_label in imp_500:
    if col_label in imp_local:
        r2_500 = imp_500[col_label]["R2"]
        r2_loc = imp_local[col_label]["R2"]
        delta = r2_loc - r2_500
        marker = "+" if delta > 0 else ""
        print(f"   {col_label:30s} {r2_500:>10.4f} {r2_loc:>10.4f} {marker}{delta:>9.4f}")

print("\n3. MUTUAL INFORMATION (Linfoot comparison)")
print(f"   {'Pair':45s} {'Kaggle':>8s} {'Local':>8s} {'Delta':>8s}")
print(f"   {'-' * 45} {'-' * 8} {'-' * 8} {'-' * 8}")
for pair_label in mi_500:
    if pair_label in mi_local:
        l500 = mi_500[pair_label]["Linfoot"]
        l_loc = mi_local[pair_label]["Linfoot"]
        delta = l_loc - l500
        marker = "+" if delta > 0 else ""
        print(f"   {pair_label:45s} {l500:>8.3f} {l_loc:>8.3f} {marker}{delta:>7.3f}")

print("\n4. CLASSIFICATION (Metallicity)")
print(f"   {'Metric':15s} {'Kaggle':>10s} {'Local':>10s} {'Delta':>10s}")
print(f"   {'-' * 15} {'-' * 10} {'-' * 10} {'-' * 10}")
for metric in ["accuracy", "precision", "recall", "f1"]:
    v500 = cls_500[metric]
    v_loc = cls_local[metric]
    delta = v_loc - v500
    marker = "+" if delta > 0 else ""
    print(f"   {metric:15s} {v500:>10.4f} {v_loc:>10.4f} {marker}{delta:>9.4f}")

print("\n5. ANOMALY SCORE CORRELATION")
corr = np.corrcoef(anom_500, anom_local)[0, 1]
print(f"   Pearson correlation: {corr:.4f}")
rank_500 = np.argsort(anom_500)
rank_local = np.argsort(anom_local)
top20_overlap = len(set(rank_500[:20]) & set(rank_local[:20]))
print(f"   Top-20 anomaly overlap: {top20_overlap}/20")

print("\n6. DEPENDENCE MATRIX SIMILARITY")
dep_corr = np.corrcoef(dep_500.flatten(), dep_local.flatten())[0, 1]
print(f"   Pearson correlation (flattened Z-matrix): {dep_corr:.4f}")
max_diff = np.max(np.abs(dep_500 - dep_local))
mean_diff = np.mean(np.abs(dep_500 - dep_local))
print(f"   Max absolute difference: {max_diff:.4f}")
print(f"   Mean absolute difference: {mean_diff:.4f}")

print("\n7. TYPICALITY CORRELATION")
typ_corr = np.corrcoef(typ_500, typ_local)[0, 1]
print(f"   Pearson correlation: {typ_corr:.4f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print(f"Total time: kaggle {t_500:.0f}s + local {t_local:.0f}s")
print("=" * 70)
