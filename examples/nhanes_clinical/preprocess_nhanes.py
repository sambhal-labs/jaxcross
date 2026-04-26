#!/usr/bin/env python3
"""Build the mixed-type NHANES design matrix for jaxcross structure-discovery.

Joins 12 NHANES 2017-2018 tables on SEQN (left join into DEMO_J's universe),
selects ~29 clinically meaningful columns, applies type-aware transforms,
and writes a jaxcross-ready matrix. Missing values are preserved as NaN —
that's jaxcross's native idiom and lets the unsupervised mixture model
naturally handle the irregular sub-sampling NHANES uses (e.g. fasting labs
are only collected for ~3000 of 9000 participants).

Column layout (29 columns total):

    CONTINUOUS (log1p + z-score where heavy-tailed; z-score otherwise):
      RIDAGEYR, INDFMPIR, BMXBMI, BMXWAIST, BPXSY1, BPXDI1, BPXPLS,
      LBXSCR (log1p), LBXSGL (log1p), LBXGH, LBXTC, LBDHDD,
      LBXTR (log1p), LBDLDL, LBXSAL, LBXSASSI (log1p), LBXSATSI (log1p),
      LBXSBU, LBXWBCSI, LBXRBCSI, LBXHGB, LBXPLTSI, LBXMCVSI

    CATEGORICAL:
      RIAGENDR (sex), RIDRETH3 (race/Hispanic origin)

    ORDINAL:
      DMDEDUC2 (education attainment, 5 levels)

    BINARY:
      DIQ010 (told had diabetes), BPQ020 (told had hypertension),
      MCQ160C (told had coronary heart disease)

Outputs (results/preprocessed/):
    train_data.npy        (n_rows, 29) float32 with NaN for missing values
    seqn.npy              (n_rows,) int64 — participant ids, for traceability
    column_info.json      schema + transform stats + value mapping for cats

Usage:
    uv run python examples/nhanes_clinical/preprocess_nhanes.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pyreadstat

RAW_DIR = Path("examples/nhanes_clinical/results/raw")
OUT_DIR = Path("examples/nhanes_clinical/results/preprocessed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-6


# ---------------------------------------------------------------------------
# Column schema — drives type tagging + transforms in the build loop
# ---------------------------------------------------------------------------
# Each entry: (nhanes_code, jaxcross_type, transform, source_table, label)
COLUMN_SCHEMA: list[tuple[str, str, str, str, str]] = [
    # Demographics
    ("RIDAGEYR", "CONTINUOUS", "zscore", "DEMO_J", "Age in years at screening"),
    ("INDFMPIR", "CONTINUOUS", "zscore", "DEMO_J", "Family income to poverty ratio"),
    # Anthropometrics
    ("BMXBMI", "CONTINUOUS", "zscore", "BMX_J", "Body Mass Index (kg/m^2)"),
    ("BMXWAIST", "CONTINUOUS", "zscore", "BMX_J", "Waist circumference (cm)"),
    # Vitals
    ("BPXSY1", "CONTINUOUS", "zscore", "BPX_J", "Systolic BP (1st reading, mmHg)"),
    ("BPXDI1", "CONTINUOUS", "zscore", "BPX_J", "Diastolic BP (1st reading, mmHg)"),
    ("BPXPLS", "CONTINUOUS", "zscore", "BPX_J", "60-second pulse"),
    # Biochem (heavy-tailed → log1p first)
    ("LBXSCR", "CONTINUOUS", "log1p_zscore", "BIOPRO_J", "Creatinine (mg/dL)"),
    ("LBXSGL", "CONTINUOUS", "log1p_zscore", "BIOPRO_J", "Glucose, refrigerated (mg/dL)"),
    ("LBXGH", "CONTINUOUS", "zscore", "GHB_J", "Glycohemoglobin / HbA1c (%)"),
    ("LBXTC", "CONTINUOUS", "zscore", "TCHOL_J", "Total cholesterol (mg/dL)"),
    ("LBDHDD", "CONTINUOUS", "zscore", "HDL_J", "Direct HDL-Cholesterol (mg/dL)"),
    ("LBXTR", "CONTINUOUS", "log1p_zscore", "TRIGLY_J", "Triglycerides (mg/dL)"),
    ("LBDLDL", "CONTINUOUS", "zscore", "TRIGLY_J", "LDL-Cholesterol, Friedewald (mg/dL)"),
    ("LBXSAL", "CONTINUOUS", "zscore", "BIOPRO_J", "Albumin, refrigerated serum (g/dL)"),
    ("LBXSASSI", "CONTINUOUS", "log1p_zscore", "BIOPRO_J", "AST (U/L)"),
    ("LBXSATSI", "CONTINUOUS", "log1p_zscore", "BIOPRO_J", "ALT (U/L)"),
    ("LBXSBU", "CONTINUOUS", "zscore", "BIOPRO_J", "Blood urea nitrogen (mg/dL)"),
    # CBC
    ("LBXWBCSI", "CONTINUOUS", "zscore", "CBC_J", "White blood cell count (1000 cells/uL)"),
    ("LBXRBCSI", "CONTINUOUS", "zscore", "CBC_J", "Red blood cell count (million/uL)"),
    ("LBXHGB", "CONTINUOUS", "zscore", "CBC_J", "Hemoglobin (g/dL)"),
    ("LBXPLTSI", "CONTINUOUS", "zscore", "CBC_J", "Platelet count (1000 cells/uL)"),
    ("LBXMCVSI", "CONTINUOUS", "zscore", "CBC_J", "Mean cell volume (fL)"),
    # Categorical demographics
    ("RIAGENDR", "CATEGORICAL", "remap", "DEMO_J", "Gender (1=Male, 2=Female)"),
    ("RIDRETH3", "CATEGORICAL", "remap", "DEMO_J", "Race/Hispanic origin (Asian split)"),
    # Ordinal demographics
    ("DMDEDUC2", "ORDINAL", "remap", "DEMO_J", "Education (5 levels, 7/9 -> NaN)"),
    # Binary self-reported conditions
    ("DIQ010", "BINARY", "yes_no", "DIQ_J", "Doctor told had diabetes"),
    ("BPQ020", "BINARY", "yes_no", "BPQ_J", "Ever told had high blood pressure"),
    ("MCQ160C", "BINARY", "yes_no", "MCQ_J", "Doctor told had coronary heart disease"),
]

# Special-coded missing values: NHANES uses 7=Refused, 9=Don't know for many
# self-report items. These collapse to NaN.
NHANES_MISSING = {7.0, 9.0, 77.0, 99.0, 777.0, 999.0, 7777.0, 9999.0}

# Map RIDRETH3 (NHANES 2011+ extended race) into a contiguous 0..5 range
# (the NHANES code 5 was reserved and never used; we collapse 6/7).
RIDRETH3_REMAP = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5}
# Sex: 1=Male, 2=Female -> 0/1
RIAGENDR_REMAP = {1: 0, 2: 1}
# Education: keep 1..5 contiguous; 7,9 -> NaN
DMDEDUC2_REMAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}


def _read_xport(table: str) -> pl.DataFrame:
    fp = RAW_DIR / f"{table}.xpt"
    df, _ = pyreadstat.read_xport(str(fp))
    pdf = pl.from_pandas(df)
    if "SEQN" not in pdf.columns:
        raise RuntimeError(f"{table} is missing SEQN")
    pdf = pdf.with_columns(pl.col("SEQN").cast(pl.Int64))
    return pdf


def _zscore(values: np.ndarray) -> tuple[np.ndarray, dict]:
    m = values[~np.isnan(values)]
    mean = float(m.mean()) if m.size else 0.0
    std = float(m.std()) if m.size and m.std() > EPS else 1.0
    return ((values - mean) / std).astype(np.float32), {"mean": mean, "std": std}


def _log1p_zscore(values: np.ndarray) -> tuple[np.ndarray, dict]:
    log_vals = np.log1p(np.maximum(values, 0.0))
    z, stats = _zscore(log_vals)
    return z, {"transform": "log1p_then_zscore", **stats}


def _binary_yes_no(values: np.ndarray) -> tuple[np.ndarray, dict]:
    """1=Yes -> 1, 2=No -> 0, 7/9/borderline -> NaN."""
    out = np.full_like(values, np.nan, dtype=np.float32)
    out[values == 1] = 1.0
    out[values == 2] = 0.0
    return out, {"transform": "1_yes_2_no"}


def _categorical_remap(values: np.ndarray, mapping: dict[int, int]) -> tuple[np.ndarray, dict]:
    out = np.full_like(values, np.nan, dtype=np.float32)
    for src, dst in mapping.items():
        out[values == src] = float(dst)
    return out, {"mapping": {str(k): v for k, v in mapping.items()}}


def main() -> int:
    # Load all tables (lazy via polars but pyreadstat is eager — fine for ~17MB total)
    tables: dict[str, pl.DataFrame] = {}
    for spec in COLUMN_SCHEMA:
        t = spec[3]
        if t not in tables:
            print(f"Loading {t}...")
            tables[t] = _read_xport(t)

    # Anchor on DEMO_J (every NHANES participant), left-join everything else.
    base = tables["DEMO_J"].select(["SEQN"])
    print(f"Anchor: DEMO_J has {base.height:,} participants")

    for code, _jtype, _transform, table, _label in COLUMN_SCHEMA:
        src = tables[table]
        if code not in src.columns:
            raise RuntimeError(f"{code} not in {table}")
        base = base.join(src.select(["SEQN", code]), on="SEQN", how="left")

    # Cast all clinical columns to Float64 so NHANES-special values can become NaN
    base = base.with_columns(
        [pl.col(code).cast(pl.Float64, strict=False) for code, *_ in COLUMN_SCHEMA]
    )

    # Replace NHANES-coded "Refused" / "Don't know" markers with NaN for
    # self-reported items only (DIQ010, BPQ020, MCQ160C — the other tables
    # already use SAS . missingness which polars converted to NaN).
    for code, _jtype, transform, *_ in COLUMN_SCHEMA:
        if transform in ("yes_no", "remap"):
            base = base.with_columns(
                pl.when(pl.col(code).is_in(list(NHANES_MISSING)))
                .then(None)
                .otherwise(pl.col(code))
                .alias(code)
            )

    # Build the design matrix column by column
    n_rows = base.height
    n_cols = len(COLUMN_SCHEMA)
    train = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    column_info: list[dict] = []

    for j, (code, jtype, transform, table, label) in enumerate(COLUMN_SCHEMA):
        raw = base[code].to_numpy()
        if transform == "zscore":
            col, stats = _zscore(raw)
        elif transform == "log1p_zscore":
            col, stats = _log1p_zscore(raw)
        elif transform == "yes_no":
            col, stats = _binary_yes_no(raw)
        elif transform == "remap":
            mapping = {
                "RIAGENDR": RIAGENDR_REMAP,
                "RIDRETH3": RIDRETH3_REMAP,
                "DMDEDUC2": DMDEDUC2_REMAP,
            }[code]
            col, stats = _categorical_remap(raw, mapping)
        else:
            raise RuntimeError(f"Unknown transform {transform}")
        train[:, j] = col
        column_info.append(
            {
                "index": j,
                "name": code,
                "label": label,
                "type": jtype,
                "source_table": table,
                "transform": transform,
                "n_observed": int(np.isfinite(col).sum()),
                "stats": stats,
            }
        )

    seqn = base["SEQN"].to_numpy().astype(np.int64)

    # Diagnostics
    nan_frac_per_col = np.isnan(train).mean(axis=0)
    print("\n=== Per-column missingness ===")
    for c in column_info:
        nan = nan_frac_per_col[c["index"]]
        print(
            f"  [{c['index']:2d}] {c['name']:10s}  ({c['type']:11s})  "
            f"observed={c['n_observed']:5d}/{n_rows}  NaN={nan:.1%}  — {c['label']}"
        )
    print(f"\nOverall NaN fraction: {np.isnan(train).mean():.1%}")
    n_any = int(np.isfinite(train).any(axis=1).sum())
    n_all = int(np.isfinite(train).all(axis=1).sum())
    print(f"Rows with at least one observation: {n_any:,}/{n_rows}")
    print(f"Rows with all observations: {n_all:,}/{n_rows}")

    np.save(OUT_DIR / "train_data.npy", train)
    np.save(OUT_DIR / "seqn.npy", seqn)
    (OUT_DIR / "column_info.json").write_text(
        json.dumps({"n_rows": n_rows, "n_cols": n_cols, "columns": column_info}, indent=2)
    )
    print(f"\nSaved to {OUT_DIR}/:")
    print(f"  train_data.npy   shape={train.shape} dtype={train.dtype}")
    print(f"  seqn.npy         shape={seqn.shape}")
    print(f"  column_info.json {n_cols} columns + transform stats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
