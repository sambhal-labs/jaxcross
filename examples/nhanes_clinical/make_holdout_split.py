#!/usr/bin/env python3
"""Build a held-out split for the NHANES clinical demo.

Produces a deterministic stratified 80/20 split of the 9,254-row preprocessed
matrix and additionally masks 5 % of cells in 6 biomarker columns within the
training fold. The masked cells become a ground-truth held-out set for
imputation-calibration evaluation; the 1,851 test rows become a held-out set
for diabetes (DIQ010) classification.

Stratification: by DIQ010 observed-status × DIQ010 value (so test prevalence
of diabetes = train prevalence of diabetes = full-cohort prevalence).

Outputs (results/preprocessed_holdout/):
    train_data.npy           (7403, 29) — train rows; biomarker cells in
                             5 % of train rows have been replaced with NaN
                             so inference cannot peek at the held-out values
    column_info.json         identical schema to results/preprocessed/, so
                             run_inference.py works against this dir as-is
    seqn.npy                 (7403,) train SEQNs
    test_data.npy            (1851, 29) — test rows; DIQ010 column has been
                             replaced with NaN so the model cannot peek at
                             diabetes labels at inference time
    test_seqn.npy            (1851,) test SEQNs
    test_indices.npy         (1851,) original-row indices into the 9254-row
                             preprocessed matrix (for joining back later)
    train_indices.npy        (7403,) original-row indices
    holdout_meta.json        all the masking metadata + ground-truth values
                             needed to compute held-out CI coverage

Usage:
    uv run python examples/nhanes_clinical/make_holdout_split.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SRC_DIR = Path("examples/nhanes_clinical/results/preprocessed")
OUT_DIR = Path("examples/nhanes_clinical/results/preprocessed_holdout")

# Columns where we mask 5 % of cells for held-out CI coverage. These are the
# six clinical biomarkers we already report per-column 90 % CI coverage for in
# discover_structure.py / NHANES_RESULTS.md, so the held-out story stays
# directly comparable to the in-sample story.
MASK_COLUMNS = ["LBXGH", "LBXSGL", "BMXBMI", "BPXSY1", "LBXTC", "LBDLDL"]
MASK_FRACTION = 0.05
TEST_FRACTION = 0.20
SEED = 7


def main() -> int:
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Missing {SRC_DIR} — run preprocess_nhanes.py first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    info = json.loads((SRC_DIR / "column_info.json").read_text())
    column_names = [c["name"] for c in info["columns"]]
    name_to_idx = {n: i for i, n in enumerate(column_names)}
    train_data = np.load(SRC_DIR / "train_data.npy")  # (9254, 29)
    seqn = np.load(SRC_DIR / "seqn.npy")
    n_rows, n_cols = train_data.shape
    print(f"Loaded {n_rows:,} x {n_cols} matrix from {SRC_DIR}")

    # ── Stratified 80/20 split on DIQ010 status × value ───────────────────
    rng = np.random.default_rng(SEED)
    diq_idx = name_to_idx["DIQ010"]
    diq = train_data[:, diq_idx]

    # Stratum keys: 'missing', '0_no_diabetes', '1_diabetes'
    strata = np.full(n_rows, fill_value="?", dtype=object)
    strata[np.isnan(diq)] = "missing"
    strata[diq == 0.0] = "negative"
    strata[diq == 1.0] = "positive"

    train_idx_list, test_idx_list = [], []
    for stratum in ["positive", "negative", "missing"]:
        ids = np.where(strata == stratum)[0]
        rng.shuffle(ids)
        n_test = int(round(len(ids) * TEST_FRACTION))
        test_idx_list.append(ids[:n_test])
        train_idx_list.append(ids[n_test:])
        print(
            f"  Stratum '{stratum}': {len(ids):5d} rows -> "
            f"{len(ids) - n_test:5d} train + {n_test:5d} test"
        )
    train_indices = np.sort(np.concatenate(train_idx_list))
    test_indices = np.sort(np.concatenate(test_idx_list))
    print(f"\nFinal split: {len(train_indices):,} train + {len(test_indices):,} test")

    train_data_split = train_data[train_indices].copy()
    test_data_split = train_data[test_indices].copy()
    train_seqn = seqn[train_indices]
    test_seqn = seqn[test_indices]

    # ── Mask 5 % of biomarker cells in train ──────────────────────────────
    # For each column independently, choose 5 % of CURRENTLY-OBSERVED rows in
    # train and set their values to NaN. We save the (row_in_train, col_idx,
    # ground_truth_value) tuples so we can compute held-out CI coverage later.
    masked_records: list[dict] = []
    for col_name in MASK_COLUMNS:
        if col_name not in name_to_idx:
            continue
        col_idx = name_to_idx[col_name]
        col = train_data_split[:, col_idx]
        observed_in_train = np.where(~np.isnan(col))[0]
        n_to_mask = int(round(len(observed_in_train) * MASK_FRACTION))
        if n_to_mask == 0:
            continue
        # Use a column-specific RNG stream so the test masking is deterministic
        # regardless of column ordering.
        col_rng = np.random.default_rng(SEED + col_idx)
        mask_local_rows = col_rng.choice(observed_in_train, size=n_to_mask, replace=False)
        for r in mask_local_rows:
            masked_records.append(
                {
                    "train_row_idx": int(r),
                    "train_seqn": int(train_seqn[r]),
                    "col_idx": int(col_idx),
                    "col_name": col_name,
                    "ground_truth": float(col[r]),
                }
            )
        # Apply the mask AFTER reading ground truth values
        train_data_split[mask_local_rows, col_idx] = np.nan
        print(
            f"  Masked {n_to_mask:4d} cells in {col_name:8s} "
            f"(5 % of {len(observed_in_train):4d} observed in train)"
        )
    print(f"Masked {len(masked_records):,} biomarker cells total in train")

    # ── Mask DIQ010 in test rows so the model cannot peek at diabetes ────
    # (We keep the ground truth in test_indices for evaluation.)
    test_diq_truth = test_data_split[:, diq_idx].copy()
    test_data_split[:, diq_idx] = np.nan
    n_test_diq_observed = int((~np.isnan(test_diq_truth)).sum())
    print(
        f"Test rows: {len(test_seqn):,} total, "
        f"{n_test_diq_observed:,} with observed DIQ010 (held-out diabetes labels)"
    )

    # ── Persist all artifacts ────────────────────────────────────────────
    np.save(OUT_DIR / "train_data.npy", train_data_split)
    np.save(OUT_DIR / "seqn.npy", train_seqn)
    np.save(OUT_DIR / "test_data.npy", test_data_split)
    np.save(OUT_DIR / "test_seqn.npy", test_seqn)
    np.save(OUT_DIR / "train_indices.npy", train_indices)
    np.save(OUT_DIR / "test_indices.npy", test_indices)
    # Re-emit column_info.json verbatim so run_inference.py + discover_*.py
    # work against this dir without changes
    (OUT_DIR / "column_info.json").write_text(json.dumps(info, indent=2))

    holdout_meta = {
        "src_dir": str(SRC_DIR),
        "split_seed": SEED,
        "test_fraction": TEST_FRACTION,
        "mask_fraction": MASK_FRACTION,
        "mask_columns": MASK_COLUMNS,
        "n_total": int(n_rows),
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
        "n_masked_cells_in_train": len(masked_records),
        "n_test_diq010_observed": n_test_diq_observed,
        "test_diq010_prevalence": (
            float(test_diq_truth[~np.isnan(test_diq_truth)].mean())
            if n_test_diq_observed
            else None
        ),
        "test_diq010_ground_truth": [
            (float(v) if not np.isnan(v) else None) for v in test_diq_truth
        ],
        "masked_cells": masked_records,
    }
    (OUT_DIR / "holdout_meta.json").write_text(json.dumps(holdout_meta, indent=2))

    print(f"\nSaved holdout artifacts to {OUT_DIR}/")
    for f in sorted(OUT_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name:25s} ({f.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
