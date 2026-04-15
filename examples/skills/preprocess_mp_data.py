#!/usr/bin/env python3
"""
Preprocess all MP materials for CrossCat imputation.
Applies identical pipeline as training: encoding, cleaning, log transform, IQR clamp.
Separates materials WITH dielectric (training set) from WITHOUT (prediction targets).

Input:  mp_all_summary_cache.parquet (from fetch_mp_data.py)
Output: mp_new_materials_preprocessed.npy  (float32 array, 23 cols)
        mp_new_materials_meta.parquet      (material_id, formula, crystal_system)
        mp_train_data.npy                  (training data array for insertion)

Usage: uv run python examples/skills/preprocess_mp_data.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from pymatgen.symmetry.groups import SpaceGroup

from crosscat.types import ColumnType

CACHE_DIR = Path("examples/results/materials_project")
ALL_CACHE = CACHE_DIR / "mp_all_summary_cache.parquet"
DIELECTRIC_CACHE = CACHE_DIR / "mp_dielectric_cache_v2.parquet"
OUTPUT_DIR = CACHE_DIR / "preprocessed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


# Column order must match training exactly
COLUMN_CATALOG = [
    ("band_gap", ColumnType.CONTINUOUS),
    ("is_metal", ColumnType.BINARY),
    ("e_electronic", ColumnType.CONTINUOUS),
    ("e_ionic", ColumnType.CONTINUOUS),
    ("e_total", ColumnType.CONTINUOUS),
    ("formation_energy_per_atom", ColumnType.CONTINUOUS),
    ("energy_above_hull", ColumnType.CONTINUOUS),
    ("is_stable", ColumnType.BINARY),
    ("density", ColumnType.CONTINUOUS),
    ("volume", ColumnType.CONTINUOUS),
    ("nsites", ColumnType.CONTINUOUS),
    ("nelements", ColumnType.CONTINUOUS),
    ("crystal_system", ColumnType.CATEGORICAL),
    ("bulk_modulus_vrh", ColumnType.CONTINUOUS),
    ("shear_modulus_vrh", ColumnType.CONTINUOUS),
    ("universal_anisotropy", ColumnType.CONTINUOUS),
    ("homogeneous_poisson", ColumnType.CONTINUOUS),
    ("e_ij_max", ColumnType.CONTINUOUS),
    ("avg_electroneg", ColumnType.CONTINUOUS),
    ("avg_ionic_radius", ColumnType.CONTINUOUS),
    ("laue_class", ColumnType.ORDINAL),
    ("total_magnetization", ColumnType.CONTINUOUS),
    ("ordering", ColumnType.CATEGORICAL),
]
VALID_ATTRS = [attr for attr, _ in COLUMN_CATALOG]

# ══════════════════════════════════════════════════════════════
# STEP 1: Load raw data
# ══════════════════════════════════════════════════════════════
print("Loading data...")
t0 = time.time()

if not ALL_CACHE.exists():
    raise FileNotFoundError(f"{ALL_CACHE} not found. Run fetch_mp_data.py first.")

df_all = pd.read_parquet(ALL_CACHE)
df_diel = pd.read_parquet(DIELECTRIC_CACHE)
diel_mpids = set(df_diel["material_id"].values)

print(f"  All materials: {len(df_all)}")
print(f"  With dielectric (training): {len(diel_mpids)}")

# ══════════════════════════════════════════════════════════════
# STEP 2: Compute compositional features from formula
# ══════════════════════════════════════════════════════════════
print("\nComputing compositional features from formulas...")

avg_electroneg = []
avg_ionic_radius = []
laue_class = []

for i, row in df_all.iterrows():
    # Electronegativity + ionic radius from formula
    try:
        comp = Composition(row["formula_pretty"])
        avg_electroneg.append(comp.average_electroneg)
        comp_dict = comp.as_dict()
        total = sum(comp_dict.values())
        if total > 0:
            wir = (
                sum(float(Element(el).average_ionic_radius) * amt for el, amt in comp_dict.items())
                / total
            )
            avg_ionic_radius.append(wir if wir > 0 else None)
        else:
            avg_ionic_radius.append(None)
    except Exception:
        avg_electroneg.append(None)
        avg_ionic_radius.append(None)

    # Laue class from space group (if available)
    sg_num = row.get("spacegroup_number", None)
    laue_class.append(sg_to_laue_int(sg_num) if pd.notna(sg_num) else None)

    if (i + 1) % 25000 == 0:
        print(f"  {i + 1}/{len(df_all)} formulas processed...")

df_all["avg_electroneg"] = avg_electroneg
df_all["avg_ionic_radius"] = avg_ionic_radius
df_all["laue_class"] = laue_class

# Add missing columns as NaN (dielectric, elasticity, piezo, crystal system)
for col in [
    "e_electronic",
    "e_ionic",
    "e_total",
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "universal_anisotropy",
    "homogeneous_poisson",
    "e_ij_max",
    "crystal_system",
]:
    if col not in df_all.columns:
        df_all[col] = np.nan

print(f"  Done ({time.time() - t0:.0f}s)")

# ══════════════════════════════════════════════════════════════
# STEP 3: Apply same preprocessing as training
# ══════════════════════════════════════════════════════════════
print("\nApplying training-identical preprocessing...")


def preprocess(df):
    """Apply the same cleaning pipeline as CrossCat training."""
    df = df.copy()

    # Crystal system encoding
    cs_map = {
        "Triclinic": 0,
        "Monoclinic": 1,
        "Orthorhombic": 2,
        "Tetragonal": 3,
        "Trigonal": 4,
        "Hexagonal": 5,
        "Cubic": 6,
    }
    df["crystal_system"] = df["crystal_system"].map(cs_map)

    # Ordering encoding (use same values as dielectric training set)
    df_diel_tmp = pd.read_parquet(DIELECTRIC_CACHE)
    ordering_vals = sorted(df_diel_tmp["ordering"].dropna().unique(), key=str)
    ord_map = {v: i for i, v in enumerate(ordering_vals)}
    df["ordering"] = df["ordering"].map(ord_map)

    # Booleans to float
    for col in ["is_stable", "is_metal"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Inf replacement
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

    # Negative moduli
    for attr in ["bulk_modulus_vrh", "shear_modulus_vrh"]:
        if attr in df.columns:
            df.loc[df[attr] < 0, attr] = np.nan

    # Upper outlier clamping
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

    # Log transform
    for attr, _ in COLUMN_CATALOG:
        if attr in df.columns:
            s = df[attr].dropna()
            if len(s) > 0 and s.min() > 0 and (s.max() / s.min()) > 100:
                df[attr] = np.log1p(df[attr])

    # Ionic radius zero guard
    if "avg_ionic_radius" in df.columns:
        df.loc[df["avg_ionic_radius"] == 0, "avg_ionic_radius"] = np.nan

    # Laue class as float
    if "laue_class" in df.columns:
        df["laue_class"] = df["laue_class"].astype(float)

    return df


df_all = preprocess(df_all)

# Build array with exact column order
data_np = df_all[VALID_ATTRS].values.astype(np.float32)

# Post-cast IQR clamp
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
            print(f"  Clamped {n_clamped} in col {ci} ({VALID_ATTRS[ci]})")

# ══════════════════════════════════════════════════════════════
# STEP 4: Split into training set and new materials
# ══════════════════════════════════════════════════════════════
is_training = df_all["material_id"].isin(diel_mpids).values
new_mask = ~is_training

new_data = data_np[new_mask]
new_meta = df_all.loc[
    new_mask,
    [
        "material_id",
        "formula_pretty",
        "crystal_system",
        "nelements",
        "band_gap",
        "density",
        "formation_energy_per_atom",
    ],
].reset_index(drop=True)

# Build training data from DIELECTRIC cache (has actual DFT values)
# NOT from all-materials (which has NaN for dielectric columns)
print("\n  Building training data from dielectric cache...")
df_train = preprocess(df_diel)
train_data = df_train[VALID_ATTRS].values.astype(np.float32)
train_data[~np.isfinite(train_data)] = np.nan
# Apply same IQR clamp
for ci in range(train_data.shape[1]):
    col = train_data[:, ci]
    v = col[np.isfinite(col)]
    if len(v) > 100:
        q01, q99 = np.percentile(v, [0.5, 99.5])
        iqr = q99 - q01
        lo, hi = q01 - 5 * iqr, q99 + 5 * iqr
        if ((col < lo) | (col > hi)).sum() > 0:
            train_data[(train_data[:, ci] < lo) | (train_data[:, ci] > hi), ci] = np.nan

# Verify training data has dielectric values
ionic_col = VALID_ATTRS.index("e_ionic")
n_ionic = (~np.isnan(train_data[:, ionic_col])).sum()
print(f"  Train ionic dielectric: {n_ionic}/{len(train_data)} non-NaN")

print(f"\n  Training materials: {train_data.shape[0]}")
print(f"  New materials (to predict): {new_data.shape[0]}")
print(f"  Columns: {data_np.shape[1]}")
print(f"  New data NaN fraction: {np.isnan(new_data).mean():.1%}")

# ══════════════════════════════════════════════════════════════
# STEP 5: Save outputs
# ══════════════════════════════════════════════════════════════
np.save(str(OUTPUT_DIR / "new_materials_data.npy"), new_data)
new_meta.to_parquet(OUTPUT_DIR / "new_materials_meta.parquet", index=False)
np.save(str(OUTPUT_DIR / "train_data.npy"), train_data)

elapsed = time.time() - t0
print(f"\nSaved to {OUTPUT_DIR}/:")
print(f"  new_materials_data.npy: {new_data.shape}")
print(f"  new_materials_meta.parquet: {len(new_meta)} rows")
print(f"  train_data.npy: {train_data.shape}")
print(f"Total time: {elapsed:.0f}s")
