#!/usr/bin/env python3
"""
Fetch v2 data from MP API, load checkpoint, run additional Gibbs sweeps on local GPU.
Usage: uv run python examples/run_local_sweeps.py
"""

import gc
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from emmet.core.summary import HasProps
from mp_api.client import MPRester
from pymatgen.core import Element
from pymatgen.symmetry.groups import SpaceGroup

from crosscat import (
    load_latest_checkpoint,
    save_checkpoint,
)
from crosscat.packed import packed_gibbs_step
from crosscat.types import ColumnType

# ── Configuration ──────────────────────────────────────────────
MP_API_KEY = "yGA3US2qaVGQG51xrjLTDidG80JqvG5e"
CACHE_DIR = Path("examples/results/materials_project")
CACHE_PATH = CACHE_DIR / "mp_dielectric_cache_v2.parquet"
CKPT_DIR = CACHE_DIR / "checkpoints_v2"
N_SWEEPS = 100  # Additional sweeps to run
DIAG_EVERY = 10  # Print diagnostics every N sweeps
SEED = 42

CACHE_DIR.mkdir(parents=True, exist_ok=True)

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


# ── Column catalog (must match v2 notebook exactly) ───────────
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

# ── Step 1: Fetch or load data ────────────────────────────────
if CACHE_PATH.exists():
    print(f"Loading cached data from {CACHE_PATH}")
    df = pd.read_parquet(CACHE_PATH)
    print(f"Loaded: {df.shape}")
else:
    print("Fetching data from Materials Project API...")
    t0 = time.time()

    with MPRester(MP_API_KEY) as mpr:
        print("  Fetching summary docs (has_props=dielectric)...")
        summary_docs = mpr.materials.summary.search(has_props=[HasProps.dielectric])
        print(f"  Got {len(summary_docs)} summary docs")

        mpids = [str(doc.material_id) for doc in summary_docs]

        print("  Fetching dielectric data...")
        dielectric_docs = mpr.materials.dielectric.search(material_ids=mpids)
        print(f"  Got {len(dielectric_docs)} dielectric docs")

        print("  Fetching elasticity data...")
        elastic_docs = mpr.materials.elasticity.search(material_ids=mpids)
        print(f"  Got {len(elastic_docs)} elasticity docs")

        print("  Fetching piezoelectric data...")
        piezo_docs = mpr.materials.piezoelectric.search(material_ids=mpids)
        print(f"  Got {len(piezo_docs)} piezoelectric docs")

    # Build summary DataFrame
    summary_records = []
    for doc in summary_docs:
        rec = {"material_id": str(doc.material_id)}
        for field in [
            "band_gap",
            "formation_energy_per_atom",
            "energy_above_hull",
            "density",
            "volume",
            "nsites",
            "is_stable",
            "is_metal",
            "total_magnetization",
            "ordering",
            "nelements",
        ]:
            rec[field] = getattr(doc, field, None)
        if hasattr(doc, "symmetry") and doc.symmetry is not None:
            rec["crystal_system"] = str(getattr(doc.symmetry, "crystal_system", None))
        else:
            rec["crystal_system"] = None

        # v2: compositional features
        if doc.composition is not None:
            rec["avg_electroneg"] = doc.composition.average_electroneg
            comp_dict = doc.composition.as_dict()
            total = sum(comp_dict.values())
            if total > 0:
                weighted_ir = (
                    sum(
                        float(Element(el).average_ionic_radius) * amt
                        for el, amt in comp_dict.items()
                    )
                    / total
                )
                rec["avg_ionic_radius"] = weighted_ir if weighted_ir > 0 else None
            else:
                rec["avg_ionic_radius"] = None
        else:
            rec["avg_electroneg"] = None
            rec["avg_ionic_radius"] = None

        # v2: Laue class
        if hasattr(doc, "symmetry") and doc.symmetry is not None:
            sg_num = getattr(doc.symmetry, "number", None)
            rec["laue_class"] = sg_to_laue_int(sg_num) if sg_num else None
        else:
            rec["laue_class"] = None

        rec["formula_pretty"] = str(getattr(doc, "formula_pretty", ""))
        summary_records.append(rec)

    df_summary = pd.DataFrame(summary_records)

    # Dielectric
    df_diel = pd.DataFrame(
        [
            {
                "material_id": str(d.material_id),
                "e_total": getattr(d, "e_total", None),
                "e_ionic": getattr(d, "e_ionic", None),
                "e_electronic": getattr(d, "e_electronic", None),
            }
            for d in dielectric_docs
        ]
    )

    # Elasticity
    elastic_records = []
    for d in elastic_docs:
        rec = {"material_id": str(d.material_id)}
        if hasattr(d, "bulk_modulus") and d.bulk_modulus is not None:
            rec["bulk_modulus_vrh"] = getattr(d.bulk_modulus, "vrh", None)
        if hasattr(d, "shear_modulus") and d.shear_modulus is not None:
            rec["shear_modulus_vrh"] = getattr(d.shear_modulus, "vrh", None)
        rec["universal_anisotropy"] = getattr(d, "universal_anisotropy", None)
        rec["homogeneous_poisson"] = getattr(d, "homogeneous_poisson", None)
        elastic_records.append(rec)
    df_elastic = pd.DataFrame(elastic_records)

    # Piezoelectric
    df_piezo = pd.DataFrame(
        [
            {"material_id": str(d.material_id), "e_ij_max": getattr(d, "e_ij_max", None)}
            for d in piezo_docs
        ]
    )

    # Merge
    df = df_summary.merge(df_diel, on="material_id", how="left")
    df = df.merge(df_elastic, on="material_id", how="left")
    df = df.merge(df_piezo, on="material_id", how="left")

    elapsed = time.time() - t0
    print(f"Fetched and merged in {elapsed:.1f}s: {df.shape}")

    df.to_parquet(CACHE_PATH, index=False)
    print(f"Cached to {CACHE_PATH}")

# ── Step 2: Preprocess (same as notebook cells 11-15) ─────────
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

# Boolean -> float
for col in ["is_stable", "is_metal"]:
    if col in df.columns:
        df[col] = df[col].astype(float)

# Clean infinities and outliers
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

# Outlier clamping
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

# Log-transform skewed columns
for attr, _, _ in COLUMN_CATALOG:
    if attr in df.columns:
        s = df[attr].dropna()
        if len(s) > 0 and s.min() > 0 and (s.max() / s.min()) > 100:
            df[attr] = np.log1p(df[attr])

# Ionic radius zero-guard
if "avg_ionic_radius" in df.columns:
    df.loc[df["avg_ionic_radius"] == 0, "avg_ionic_radius"] = np.nan

# Laue class as float
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
data_jax = jnp.array(data_np)
n_rows, n_cols = data_jax.shape

print(f"\nData array: {n_rows} rows x {n_cols} cols")
print(f"NaN fraction: {float(jnp.isnan(data_jax).mean()):.1%}")
print(f"Column types: {dict(pd.Series([ct.name for ct in column_types]).value_counts())}")

# ── Step 3: Load checkpoint ───────────────────────────────────
print(f"\nLoading checkpoint from {CKPT_DIR}...")
packed, ckpt_col_types, start_sweep = load_latest_checkpoint(str(CKPT_DIR))
print(f"Loaded: sweep {start_sweep}, {packed.n_rows} rows x {packed.n_cols} cols")

assert packed.n_cols == n_cols, (
    f"Column mismatch: checkpoint has {packed.n_cols}, data has {n_cols}"
)
assert packed.n_rows == n_rows, f"Row mismatch: checkpoint has {packed.n_rows}, data has {n_rows}"

# ── Step 4: Run additional sweeps ─────────────────────────────
print(f"\nRunning {N_SWEEPS} additional sweeps (single chain, local GPU)...")
print("Using packed_gibbs_step (4 smaller JIT compilations, memory-friendly)")
print(f"Diagnostics every {DIAG_EVERY} sweeps")
print(flush=True)

key = jax.random.key(SEED + start_sweep)
total_start = time.time()

for i in range(N_SWEEPS):
    key, subkey = jax.random.split(key)

    t0 = time.time()
    packed = packed_gibbs_step(subkey, packed, data_jax)
    # Block until computation completes (for accurate timing)
    jax.block_until_ready(packed.view_row_assignments)
    elapsed = time.time() - t0

    current_sweep = start_sweep + i + 1

    if (i + 1) % DIAG_EVERY == 0 or i == 0:
        total_elapsed = time.time() - total_start
        print(
            f"Sweep {current_sweep} ({elapsed:.1f}s/step, total {total_elapsed:.0f}s)",
            flush=True,
        )

        # Checkpoint
        save_checkpoint(packed, str(CKPT_DIR), current_sweep, column_types=column_types)
        print(f"  Checkpoint saved: sweep_{current_sweep}", flush=True)

        gc.collect()

final_sweep = start_sweep + N_SWEEPS
total_time = time.time() - total_start
print(f"\nDone! {N_SWEEPS} sweeps in {total_time:.0f}s ({total_time / N_SWEEPS:.1f}s/sweep)")
print(f"Final checkpoint: sweep {final_sweep}")
