#!/usr/bin/env python3
"""Preprocess C-MAPSS turbofan degradation data for jaxcross.

Reads raw NASA C-MAPSS text files and produces a unified mixed-type design
matrix suitable for the CrossCat packed-state pipeline.

For each sub-dataset (FD001-FD004):
  - Load train/test/RUL files
  - Compute ground-truth RUL for every training row:
        RUL = max_cycle[engine] - current_cycle
    Cap at RUL_CAP (standard convention in the RUL literature; defaults to 125)
  - Drop sensors that are constant or near-constant (std < EPS)
  - Standardize remaining sensors to mean-0 / std-1 using training statistics
  - Build a query matrix from the test set: last observed cycle per engine,
    with the RUL column left NaN (to be imputed). Ground-truth RUL from
    RUL_FD00X.txt is saved alongside for evaluation.

Outputs (per sub-dataset, under examples/c_mapss/results/preprocessed/<fd>/):
  train_data.npy          (n_train_rows, n_cols)  float32
  test_query.npy          (n_test_engines, n_cols)  float32  (RUL col = NaN)
  test_rul_truth.npy      (n_test_engines,)  float32
  column_info.json        column names, types, standardization stats

Column layout (index in output):
  0           time_in_cycles         ORDINAL
  1..3        op_setting_1..3        CONTINUOUS  (or CATEGORICAL in FD002/FD004)
  4..(4+K-1)  kept sensors (z-scored) CONTINUOUS
  last        rul                    CONTINUOUS  (target)

Usage:
    uv run python examples/c_mapss/preprocess_cmapss.py [FD001 FD002 ...]
If no sub-datasets are given, all four are processed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

RAW_DIR = Path("examples/c_mapss/results/raw")
OUT_ROOT = Path("examples/c_mapss/results/preprocessed")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

RUL_CAP = 125.0  # standard RUL cap from the literature (e.g. Heimes 2008)
CONST_EPS = 1e-6  # sensors with std < EPS are dropped

RAW_COLUMNS = (
    ["unit_id", "time_in_cycles"]
    + [f"op_setting_{i}" for i in (1, 2, 3)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OPSET_COLS = [f"op_setting_{i}" for i in (1, 2, 3)]
# FD002 / FD004 have 6 discrete operating regimes; treat op settings as
# CATEGORICAL for those to let jaxcross discover the regimes.
REGIME_SETS = {"FD002", "FD004"}


def _load_raw(fd: str) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray]:
    """Return (train_df, test_df, rul_truth). Whitespace-delimited via numpy."""
    train_path = RAW_DIR / f"train_{fd}.txt"
    test_path = RAW_DIR / f"test_{fd}.txt"
    rul_path = RAW_DIR / f"RUL_{fd}.txt"

    for p in (train_path, test_path, rul_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} — run fetch_cmapss.py first")

    # np.loadtxt handles the variable-width whitespace separator natively
    # and drops the two trailing empty columns present in the NASA files.
    train_arr = np.loadtxt(train_path, dtype=np.float64)
    test_arr = np.loadtxt(test_path, dtype=np.float64)
    rul_truth = np.loadtxt(rul_path, dtype=np.float32)

    train_df = pl.DataFrame({name: train_arr[:, i] for i, name in enumerate(RAW_COLUMNS)})
    test_df = pl.DataFrame({name: test_arr[:, i] for i, name in enumerate(RAW_COLUMNS)})
    # unit_id is always an integer
    train_df = train_df.with_columns(pl.col("unit_id").cast(pl.Int64))
    test_df = test_df.with_columns(pl.col("unit_id").cast(pl.Int64))
    return train_df, test_df, rul_truth


def _compute_rul(df: pl.DataFrame, cap: float) -> np.ndarray:
    """Compute RUL per row: max(cycle within engine) - current cycle, capped."""
    with_max = df.with_columns(pl.col("time_in_cycles").max().over("unit_id").alias("_max_cycle"))
    rul = (with_max["_max_cycle"] - with_max["time_in_cycles"]).cast(pl.Float32).to_numpy()
    return np.minimum(rul, cap)


def _select_sensors(train_df: pl.DataFrame) -> list[str]:
    """Keep sensors with non-trivial variance in the training set."""
    stds = train_df.select([pl.col(s).std().alias(s) for s in SENSOR_COLS]).row(0)
    return [
        s for s, sd in zip(SENSOR_COLS, stds, strict=True) if sd is not None and sd >= CONST_EPS
    ]


def _zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std < CONST_EPS:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def _build_columns(fd: str, kept_sensors: list[str]) -> list[dict]:
    # time_in_cycles is a count (1..362 in FD001); modeling it as CONTINUOUS
    # avoids blowing up the ordinal cutpoint budget. RUL is the real target.
    cols: list[dict] = [{"name": "time_in_cycles", "type": "CONTINUOUS"}]
    opset_type = "CATEGORICAL" if fd in REGIME_SETS else "CONTINUOUS"
    for s in OPSET_COLS:
        cols.append({"name": s, "type": opset_type})
    for s in kept_sensors:
        cols.append({"name": s, "type": "CONTINUOUS"})
    cols.append({"name": "rul", "type": "CONTINUOUS"})
    return cols


def _encode_regimes(
    train_vals: np.ndarray, test_vals: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list]:
    """For FD002/FD004: map op-setting triples to discrete regime ids.

    C-MAPSS FD002 and FD004 have six flight regimes. Rather than running
    k-means, we deduplicate rounded triples from the training set, which
    recovers the six regimes reliably for this dataset.
    """
    rounded_train = np.round(train_vals, 2)
    rounded_test = np.round(test_vals, 2)
    regimes = np.unique(rounded_train, axis=0)
    regime_map = {tuple(r.tolist()): i for i, r in enumerate(regimes)}

    def _lookup(arr: np.ndarray) -> np.ndarray:
        out = np.empty(len(arr), dtype=np.float32)
        for i, row in enumerate(arr):
            out[i] = regime_map.get(tuple(row.tolist()), len(regime_map))
        return out

    return _lookup(rounded_train), _lookup(rounded_test), [r.tolist() for r in regimes]


def _last_cycle_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Pick the row with the maximum time_in_cycles per unit_id."""
    return (
        df.sort(["unit_id", "time_in_cycles"])
        .group_by("unit_id", maintain_order=True)
        .last()
        .sort("unit_id")
    )


def process_subdataset(fd: str) -> None:
    print(f"\n{'=' * 70}\n{fd}\n{'=' * 70}")
    train_df, test_df, rul_truth = _load_raw(fd)
    print(
        f"  Raw: train={train_df.height} rows / {train_df['unit_id'].n_unique()} engines, "
        f"test={test_df.height} rows / {test_df['unit_id'].n_unique()} engines"
    )

    # Training RUL
    train_rul = _compute_rul(train_df, RUL_CAP)

    # Sensor filtering (training-only statistics)
    kept_sensors = _select_sensors(train_df)
    dropped = [s for s in SENSOR_COLS if s not in kept_sensors]
    print(f"  Kept {len(kept_sensors)} sensors, dropped {len(dropped)}: {dropped}")

    # Standardization stats
    means = train_df.select([pl.col(s).mean().alias(s) for s in kept_sensors]).row(0)
    stds = train_df.select([pl.col(s).std().alias(s) for s in kept_sensors]).row(0)
    stats = dict(zip(kept_sensors, zip(means, stds, strict=True), strict=True))

    # Build training matrix column-by-column
    n_train = train_df.height
    cycle_mean = float(train_df["time_in_cycles"].mean())
    cycle_std = float(train_df["time_in_cycles"].std())
    train_cols: list[np.ndarray] = [
        _zscore(train_df["time_in_cycles"].to_numpy(), cycle_mean, cycle_std)
    ]

    if fd in REGIME_SETS:
        train_op, test_op_full, regimes = _encode_regimes(
            np.stack([train_df[c].to_numpy() for c in OPSET_COLS], axis=1),
            np.stack([test_df[c].to_numpy() for c in OPSET_COLS], axis=1),
        )
        print(f"  Discovered {len(regimes)} operating regimes (categorical encoding)")
        # All three op-setting slots hold the same regime id so downstream
        # column-index math stays identical across sub-datasets.
        for _ in OPSET_COLS:
            train_cols.append(train_op.copy())
    else:
        for s in OPSET_COLS:
            train_cols.append(train_df[s].cast(pl.Float32).to_numpy())
        regimes = None
        test_op_full = None

    for s in kept_sensors:
        mean, std = stats[s]
        train_cols.append(_zscore(train_df[s].to_numpy(), mean, std))

    train_cols.append(train_rul)
    train_arr = np.stack(train_cols, axis=1)
    assert train_arr.shape[0] == n_train

    # Test query: last observed cycle per test engine, RUL=NaN
    test_df_with_idx = test_df.with_row_index("_row_idx")
    last_rows = _last_cycle_rows(test_df_with_idx)
    last_idx = last_rows["_row_idx"].to_numpy()
    n_test = last_rows.height
    if n_test != len(rul_truth):
        raise RuntimeError(
            f"{fd}: expected {len(rul_truth)} test engines, got {n_test} last-cycle rows"
        )

    test_cols: list[np.ndarray] = [
        _zscore(last_rows["time_in_cycles"].to_numpy(), cycle_mean, cycle_std)
    ]
    if fd in REGIME_SETS and test_op_full is not None:
        test_op_last = test_op_full[last_idx]
        for _ in OPSET_COLS:
            test_cols.append(test_op_last.copy())
    else:
        for s in OPSET_COLS:
            test_cols.append(last_rows[s].cast(pl.Float32).to_numpy())

    for s in kept_sensors:
        mean, std = stats[s]
        test_cols.append(_zscore(last_rows[s].to_numpy(), mean, std))

    test_cols.append(np.full(n_test, np.nan, dtype=np.float32))
    test_arr = np.stack(test_cols, axis=1)

    rul_truth_capped = np.minimum(rul_truth.astype(np.float32), RUL_CAP)

    cols = _build_columns(fd, kept_sensors)
    col_info = {
        "fd": fd,
        "rul_cap": RUL_CAP,
        "columns": cols,
        "sensor_stats": {s: {"mean": float(m), "std": float(sd)} for s, (m, sd) in stats.items()},
        "time_in_cycles_stats": {"mean": cycle_mean, "std": cycle_std},
        "kept_sensors": kept_sensors,
        "dropped_sensors": dropped,
        "n_train_rows": int(n_train),
        "n_test_engines": int(n_test),
        "regimes": regimes,
    }

    out_dir = OUT_ROOT / fd
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_data.npy", train_arr)
    np.save(out_dir / "test_query.npy", test_arr)
    np.save(out_dir / "test_rul_truth.npy", rul_truth_capped)
    (out_dir / "column_info.json").write_text(json.dumps(col_info, indent=2))

    rul_col = train_arr.shape[1] - 1
    print(
        f"  Saved to {out_dir}/:\n"
        f"    train_data.npy      {train_arr.shape}\n"
        f"    test_query.npy      {test_arr.shape}  (RUL col NaN)\n"
        f"    test_rul_truth.npy  {rul_truth_capped.shape}\n"
        f"    RUL train range: [{train_rul.min():.1f}, {train_rul.max():.1f}], "
        f"truth range: [{rul_truth_capped.min():.1f}, {rul_truth_capped.max():.1f}]\n"
        f"    Columns ({train_arr.shape[1]}): "
        f"{[c['name'] for c in cols]}\n"
        f"    RUL column index: {rul_col}"
    )


def main(argv: list[str]) -> int:
    targets = argv[1:] if len(argv) > 1 else [f"FD00{i}" for i in range(1, 5)]
    for fd in targets:
        if fd not in {"FD001", "FD002", "FD003", "FD004"}:
            print(f"Skipping unknown sub-dataset: {fd}")
            continue
        process_subdataset(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
