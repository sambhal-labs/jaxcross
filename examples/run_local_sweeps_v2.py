#!/usr/bin/env python3
"""
Run 100 additional Gibbs sweeps on local GPU starting from sweep-500 checkpoint.
Loads sweep-500 explicitly (clean state, no NaN hypers).
Prints log-joint every diagnostic step to monitor convergence.

Usage: uv run python examples/run_local_sweeps_v2.py
"""

import gc
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pymatgen.symmetry.groups import SpaceGroup

from crosscat import (
    initialize,
    load_latest_checkpoint,
    packed_log_joint,
    save_checkpoint,
)
from crosscat.packed import pack_state, packed_gibbs_step
from crosscat.types import ColumnType

# ── Configuration ──────────────────────────────────────────────
CACHE_DIR = Path("examples/results/materials_project")
CACHE_PATH = CACHE_DIR / "mp_dielectric_cache_v2.parquet"
CKPT_DIR = CACHE_DIR / "checkpoints_v2_local"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
N_SWEEPS = 200
DIAG_EVERY = 10
SEED = 84

print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

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

# ── Load and preprocess data ─────────────────────────────────
print(f"\nLoading cached data from {CACHE_PATH}")
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

# Sanitize after float32 cast: replace inf/-inf and extreme values with NaN
data_np[~np.isfinite(data_np)] = np.nan
# Clamp extreme outliers that survived preprocessing (float32 overflow artifacts)
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
            print(f"  Clamped {n_clamped} extreme values in col {ci} ({col_names[ci]})")

data_jax = jnp.array(data_np)
n_rows, n_cols = data_jax.shape

print(f"Data array: {n_rows} rows x {n_cols} cols")
print(f"NaN fraction: {float(jnp.isnan(data_jax).mean()):.1%}")

# ── Load checkpoint or initialize fresh ──────────────────────
print(f"\nLoading checkpoint from {CKPT_DIR}...")
try:
    packed, ckpt_col_types, start_sweep = load_latest_checkpoint(str(CKPT_DIR))
    print(f"Resumed from sweep {start_sweep}: {packed.n_rows} rows x {packed.n_cols} cols")
except (FileNotFoundError, StopIteration):
    print("No checkpoint found, initializing fresh...")
    key_init = jax.random.key(SEED)
    result = initialize(key_init, data_jax, column_types, n_chains=1)
    packed = pack_state(result.state, max_views=16, max_clusters=32, data=data_jax)
    start_sweep = 0
    print(f"Initialized: {packed.n_rows} rows x {packed.n_cols} cols")

# Verify log-joint is finite
lj = float(packed_log_joint(packed, data_jax))
print(f"Log-joint after patch: {lj:,.1f}")
assert np.isfinite(lj), f"Log-joint still not finite: {lj}"

# ── Run sweeps ───────────────────────────────────────────────
print(f"\nRunning {N_SWEEPS} additional sweeps (sweep {start_sweep} -> {start_sweep + N_SWEEPS})")
print("Using packed_gibbs_step (4 sub-kernels, memory-friendly)")
print(f"Diagnostics every {DIAG_EVERY} sweeps")
print(flush=True)

key = jax.random.key(SEED + start_sweep)
total_start = time.time()

for i in range(N_SWEEPS):
    key, subkey = jax.random.split(key)

    t0 = time.time()
    packed = packed_gibbs_step(subkey, packed, data_jax)
    jax.block_until_ready(packed.view_row_assignments)
    elapsed = time.time() - t0

    current_sweep = start_sweep + i + 1

    if (i + 1) % DIAG_EVERY == 0 or i == 0:
        total_elapsed = time.time() - total_start
        lj = float(packed_log_joint(packed, data_jax))

        # Check for NaN hypers
        n_nan_s = int(jnp.isnan(packed.hyper_s).sum())
        n_nan_mu = int(jnp.isnan(packed.hyper_mu).sum())
        nan_flag = (
            f" [!NaN hypers: s={n_nan_s}, mu={n_nan_mu}]" if (n_nan_s + n_nan_mu) > 0 else ""
        )

        print(
            f"Sweep {current_sweep} ({elapsed:.1f}s/step, total {total_elapsed:.0f}s) "
            f"log_joint={lj:,.1f}{nan_flag}",
            flush=True,
        )

        # Checkpoint
        save_checkpoint(packed, str(CKPT_DIR), current_sweep, column_types=column_types)
        print(f"  Checkpoint saved: sweep_{current_sweep}", flush=True)

        gc.collect()

final_sweep = start_sweep + N_SWEEPS
total_time = time.time() - total_start
final_lj = float(packed_log_joint(packed, data_jax))
print(f"\nDone! {N_SWEEPS} sweeps in {total_time:.0f}s ({total_time / N_SWEEPS:.1f}s/sweep)")
print(f"Final checkpoint: sweep {final_sweep}, log_joint={final_lj:,.1f}")
