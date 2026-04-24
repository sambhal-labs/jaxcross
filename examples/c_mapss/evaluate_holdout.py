#!/usr/bin/env python3
"""Evaluate RUL prediction on the held-out training rows.

When ``run_inference.py`` is called with ``--subsample N``, the script trains
on N uniformly random rows and leaves (total - N) rows unused. This evaluator
uses those held-out rows as a large local validation set:

  - Load the preprocessed training matrix (full 20 631 rows for FD001)
  - Mask the RUL column to NaN for the holdout rows
  - Insert them into each trained chain's packed state
  - Impute RUL via ``batch_impute_column`` + compute 90/95/99 % CIs
  - Compare imputed vs ground-truth (which we have since these came from
    the training file)

The holdout set is typically 10-15x larger than the 100-engine official
test set, giving much tighter statistical power on MAE / RMSE / coverage.

Outputs: examples/c_mapss/results/evaluation/<fd>/holdout_metrics.json
         examples/c_mapss/results/evaluation/<fd>/holdout_predictions.csv

Usage:
    uv run python examples/c_mapss/evaluate_holdout.py [FD001] [--samples 500]
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from crosscat import batch_credible_interval, batch_impute_column, packed_insert_rows
from crosscat.serialization import load_packed_state

PREP_ROOT = Path("examples/c_mapss/results/preprocessed")
INF_ROOT = Path("examples/c_mapss/results/inference")
EVAL_ROOT = Path("examples/c_mapss/results/evaluation")

CI_LEVELS = (0.90, 0.95, 0.99)


def _metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = pred - truth
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(err)),
    }


def _evaluate_chain(
    chain_path: Path,
    train_used: np.ndarray,
    holdout_query: np.ndarray,
    rul_col: int,
    n_samples: int,
    seed: int,
    batch_size: int,
) -> dict:
    """Insert holdout rows into one chain, impute RUL, return per-row outputs."""
    packed, _ = load_packed_state(str(chain_path))
    train_j = jnp.array(train_used)
    query_j = jnp.array(holdout_query)

    t0 = time.time()
    key = jax.random.key(seed)
    key, sub = jax.random.split(key)
    packed_ext, data_ext = packed_insert_rows(sub, packed, train_j, query_j)
    jax.block_until_ready(packed_ext.view_row_assignments)
    t_ins = time.time() - t0

    n_train = train_used.shape[0]
    n_holdout = holdout_query.shape[0]

    # Impute RUL in batches so we don't hit memory limits on large holdout sets.
    preds = np.empty(n_holdout, dtype=np.float32)
    ci_los = {level: np.empty(n_holdout, dtype=np.float32) for level in CI_LEVELS}
    ci_his = {level: np.empty(n_holdout, dtype=np.float32) for level in CI_LEVELS}

    t0 = time.time()
    for start in range(0, n_holdout, batch_size):
        stop = min(start + batch_size, n_holdout)
        ids = jnp.arange(n_train + start, n_train + stop)

        key, sub = jax.random.split(key)
        pred_batch, _ = batch_impute_column(
            sub,
            packed_ext,
            data_ext,
            query_col=rul_col,
            row_ids=ids,
            n_samples=n_samples,
        )
        preds[start:stop] = np.array(pred_batch)

        for level in CI_LEVELS:
            key, sub = jax.random.split(key)
            _, lo, hi = batch_credible_interval(
                sub,
                packed_ext,
                data_ext,
                query_col=rul_col,
                row_ids=ids,
                n_samples=n_samples,
                ci_level=level,
            )
            ci_los[level][start:stop] = np.array(lo)
            ci_his[level][start:stop] = np.array(hi)
    t_imp = time.time() - t0

    del packed_ext, data_ext
    gc.collect()

    return {
        "pred_mean": preds,
        "ci": {level: (ci_los[level], ci_his[level]) for level in CI_LEVELS},
        "timing": {"insert_s": round(t_ins, 1), "impute_s": round(t_imp, 1)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fd", nargs="?", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"]
    )
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of holdout rows per batch_impute_column call. "
        "Lower = lower memory; higher = fewer JIT dispatches.",
    )
    parser.add_argument(
        "--max-holdout",
        type=int,
        default=0,
        help="If >0, cap the number of holdout rows evaluated (random subsample). "
        "Useful when the full holdout is too big to run quickly.",
    )
    args = parser.parse_args()
    fd = args.fd

    prep = PREP_ROOT / fd
    inf = INF_ROOT / fd

    if not (inf / "train_indices.npy").exists():
        raise FileNotFoundError(
            f"Missing {inf}/train_indices.npy — must run run_inference.py with --subsample "
            "to produce a holdout set."
        )

    full_train = np.load(prep / "train_data.npy")
    train_indices = np.load(inf / "train_indices.npy")
    meta = json.loads((inf / "inference_meta.json").read_text())
    info = json.loads((prep / "column_info.json").read_text())
    rul_col = len(info["columns"]) - 1

    all_idx = np.arange(full_train.shape[0], dtype=np.int64)
    holdout_idx = np.setdiff1d(all_idx, train_indices, assume_unique=False)
    if args.max_holdout and holdout_idx.size > args.max_holdout:
        rng = np.random.default_rng(args.seed)
        holdout_idx = np.sort(rng.choice(holdout_idx, size=args.max_holdout, replace=False))
        print(f"Capped holdout to {args.max_holdout} rows (random subsample)")

    holdout_full = full_train[holdout_idx].astype(np.float32)
    holdout_truth = holdout_full[:, rul_col].copy()
    holdout_query = holdout_full.copy()
    holdout_query[:, rul_col] = np.nan  # mask RUL for imputation

    train_used = np.load(inf / "train_used.npy")
    print(
        f"{fd}: trained on {train_used.shape[0]} rows, holdout has {holdout_idx.size} rows "
        f"(of {full_train.shape[0]} preprocessed total)"
    )
    print(f"  RUL col: {rul_col}   inference mode: {meta['mode']}")
    print(f"  Batch size for impute: {args.batch_size}")

    per_chain: list[dict] = []
    for ci in range(meta["n_chains"]):
        print(f"\n--- Chain {ci} ---")
        per_chain.append(
            _evaluate_chain(
                inf / f"chain_{ci}.jxc",
                train_used,
                holdout_query,
                rul_col,
                n_samples=args.samples,
                seed=args.seed + ci * 7919,
                batch_size=args.batch_size,
            )
        )
        t = per_chain[-1]["timing"]
        print(f"  insert={t['insert_s']}s  impute={t['impute_s']}s")

    # ── BMA aggregation ──────────────────────────────────────────
    stack_pred = np.stack([c["pred_mean"] for c in per_chain])
    bma_mean = stack_pred.mean(axis=0)
    bma_std = stack_pred.std(axis=0)

    combined_ci: dict = {}
    for level in CI_LEVELS:
        los = np.stack([c["ci"][level][0] for c in per_chain]).mean(axis=0)
        his = np.stack([c["ci"][level][1] for c in per_chain]).mean(axis=0)
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[level]
        lo = np.minimum(los, bma_mean - z * bma_std)
        hi = np.maximum(his, bma_mean + z * bma_std)
        combined_ci[level] = (lo, hi)

    metrics: dict = {
        "fd": fd,
        "n_holdout_rows": int(holdout_idx.size),
        "n_train_rows_used": int(train_used.shape[0]),
        "bma": _metrics(bma_mean, holdout_truth),
    }
    for level, (lo, hi) in combined_ci.items():
        coverage = float(np.mean((holdout_truth >= lo) & (holdout_truth <= hi)))
        width = float(np.mean(hi - lo))
        metrics[f"ci_{int(level * 100)}"] = {"coverage": coverage, "avg_width": width}

    metrics["per_chain"] = []
    for ci, c in enumerate(per_chain):
        m = _metrics(c["pred_mean"], holdout_truth)
        m["chain"] = ci
        m.update(c["timing"])
        metrics["per_chain"].append(m)

    # ── Report ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}\n{fd} — HOLDOUT METRICS ({holdout_idx.size} rows)\n{'=' * 70}")
    print(f"BMA across {meta['n_chains']} chains:")
    b = metrics["bma"]
    print(f"  MAE  = {b['mae']:.2f} cycles")
    print(f"  RMSE = {b['rmse']:.2f} cycles")
    print(f"  R^2  = {b['r2']:.4f}")
    print(f"  Bias = {b['bias']:+.2f} cycles")
    print("\nCalibration (CI coverage vs nominal):")
    for level in CI_LEVELS:
        c = metrics[f"ci_{int(level * 100)}"]
        print(
            f"  {int(level * 100)}% CI: coverage={c['coverage']:.1%}  "
            f"(target {level:.0%})  avg width={c['avg_width']:.1f} cycles"
        )

    # ── Save ─────────────────────────────────────────────────────
    out_dir = EVAL_ROOT / fd
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = {
        "holdout_row_idx": holdout_idx.astype(np.int64),
        "rul_truth": holdout_truth,
        "rul_bma_mean": bma_mean.astype(np.float32),
        "rul_bma_std": bma_std.astype(np.float32),
    }
    for level, (lo, hi) in combined_ci.items():
        pct = int(level * 100)
        cols[f"ci{pct}_lo"] = lo.astype(np.float32)
        cols[f"ci{pct}_hi"] = hi.astype(np.float32)
        cols[f"ci{pct}_covers"] = (holdout_truth >= lo) & (holdout_truth <= hi)
    for ci, c in enumerate(per_chain):
        cols[f"chain{ci}_pred"] = c["pred_mean"]

    df = pl.DataFrame(cols)
    df.write_csv(out_dir / "holdout_predictions.csv")
    df.write_ipc(out_dir / "holdout_predictions.arrow", compression="zstd")
    (out_dir / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {out_dir / 'holdout_metrics.json'} + holdout_predictions.{{csv,arrow}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
