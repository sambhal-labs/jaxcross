#!/usr/bin/env python3
"""Compute RUL metrics for the best-log_joint chain only (no BMA).

Intended for comparison against the BMA result in evaluate_rul.py —
demonstrates why the best-chain-by-log_joint choice is not necessarily
the best-chain-by-predictive-accuracy for a given query column.

Outputs: examples/c_mapss/results/evaluation/<fd>/best_chain_metrics.json

Usage:
    uv run python examples/c_mapss/evaluate_best_chain.py [FD001] [--samples 500]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from crosscat import batch_credible_interval, batch_impute_column, packed_insert_rows
from crosscat.serialization import load_packed_state

PREP_ROOT = Path("examples/c_mapss/results/preprocessed")
INF_ROOT = Path("examples/c_mapss/results/inference")
EVAL_ROOT = Path("examples/c_mapss/results/evaluation")
CI_LEVELS = (0.90, 0.95, 0.99)


def _metrics(pred, truth):
    err = pred - truth
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(err)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("fd", nargs="?", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=99)
    args = p.parse_args()

    prep = PREP_ROOT / args.fd
    inf = INF_ROOT / args.fd

    test_query = np.load(prep / "test_query.npy")
    truth = np.load(prep / "test_rul_truth.npy")
    info = json.loads((prep / "column_info.json").read_text())
    rul_col = len(info["columns"]) - 1

    meta = json.loads((inf / "inference_meta.json").read_text())
    best_idx = meta["best_chain_idx"]
    best_lj = meta["final_log_joints"][best_idx]
    print(
        f"Best chain by log_joint: chain {best_idx} "
        f"(log_joint={best_lj:,.1f} of {len(meta['final_log_joints'])} chains)"
    )

    train = np.load(inf / "train_used.npy")
    packed, _ = load_packed_state(str(inf / "best_chain.jxc"))

    key = jax.random.key(args.seed)
    k1, k2 = jax.random.split(key)
    packed_ext, data_ext = packed_insert_rows(k1, packed, jnp.array(train), jnp.array(test_query))
    new_ids = jnp.arange(train.shape[0], train.shape[0] + test_query.shape[0])

    pred, _ = batch_impute_column(
        k2, packed_ext, data_ext, query_col=rul_col, row_ids=new_ids, n_samples=args.samples
    )
    pred = np.array(pred)

    metrics = {
        "fd": args.fd,
        "chain": best_idx,
        "log_joint": best_lj,
        "n_test_engines": int(truth.shape[0]),
    }
    metrics["point"] = _metrics(pred, truth)

    for level in CI_LEVELS:
        kk, key = jax.random.split(key)
        _, lo, hi = batch_credible_interval(
            kk,
            packed_ext,
            data_ext,
            query_col=rul_col,
            row_ids=new_ids,
            n_samples=args.samples,
            ci_level=level,
        )
        lo = np.array(lo)
        hi = np.array(hi)
        coverage = float(np.mean((truth >= lo) & (truth <= hi)))
        width = float(np.mean(hi - lo))
        metrics[f"ci_{int(level * 100)}"] = {"coverage": coverage, "avg_width": width}

    # ── Report ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}\n{args.fd} — BEST-CHAIN-ONLY METRICS (chain {best_idx})\n{'=' * 70}")
    m = metrics["point"]
    print(
        f"Point prediction:  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  "
        f"R2={m['r2']:.3f}  Bias={m['bias']:+.2f}"
    )
    print("\nCalibration (single chain, no BMA widening):")
    for level in CI_LEVELS:
        c = metrics[f"ci_{int(level * 100)}"]
        print(
            f"  {int(level * 100)}% CI: coverage={c['coverage']:.1%}  "
            f"(target {level:.0%})  avg width={c['avg_width']:.1f} cycles"
        )

    out = EVAL_ROOT / args.fd
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_chain_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {out / 'best_chain_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
