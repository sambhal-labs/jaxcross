#!/usr/bin/env python3
"""Evaluate RUL prediction quality on a C-MAPSS test set.

For each trained chain:
  1. Insert the test-query rows (RUL column = NaN) into the packed state
  2. Impute RUL with batch_impute_column (posterior mean under that chain)
  3. Compute 90/95/99% credible intervals with batch_credible_interval

Then aggregate across chains (Bayesian Model Averaging) and report:
  - MAE / RMSE / R^2 against the published RUL ground truth
  - CI coverage at 90/95/99% (fraction of true RULs inside the CI)
  - Headline comparison vs published LSTM / Transformer baselines

Outputs (examples/c_mapss/results/evaluation/<fd>/):
  rul_predictions.csv      per-engine predictions + CIs + truth
  metrics.json             aggregate metrics
  metrics_per_chain.json   per-chain breakdown

Usage:
    uv run python examples/c_mapss/evaluate_rul.py [FD001] [--samples 500]
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

# Published baseline RUL MAE on FD001 (held-out test set) for context.
# Sources cited in examples/c_mapss/README.md.
PUBLISHED_BASELINES = {
    "FD001": {
        "LSTM (Zheng 2017)": 13.52,
        "CNN-LSTM (Li 2018)": 12.61,
        "Transformer (2024-2025)": 11.9,
    },
    "FD002": {"CNN-LSTM (Li 2018)": 19.61, "Transformer (2024-2025)": 17.2},
    "FD003": {"LSTM (Zheng 2017)": 12.64, "Transformer (2024-2025)": 11.4},
    "FD004": {"CNN-LSTM (Li 2018)": 23.57, "Transformer (2024-2025)": 19.8},
}


def _metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = pred - truth
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    bias = float(np.mean(err))
    return {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias}


def _evaluate_chain(
    chain_path: Path,
    train_data: np.ndarray,
    test_query: np.ndarray,
    rul_col: int,
    n_samples: int,
    seed: int,
) -> dict:
    packed, _ = load_packed_state(str(chain_path))
    train_j = jnp.array(train_data)
    test_j = jnp.array(test_query)

    print(f"  Inserting {len(test_query)} test-engine rows into chain...")
    t0 = time.time()
    key = jax.random.key(seed)
    key, sub = jax.random.split(key)
    packed_ext, data_ext = packed_insert_rows(sub, packed, train_j, test_j)
    jax.block_until_ready(packed_ext.view_row_assignments)
    t_ins = time.time() - t0

    n_train = train_data.shape[0]
    new_ids = jnp.arange(n_train, n_train + test_query.shape[0])

    print(f"  Imputing RUL (n_samples={n_samples})...")
    t0 = time.time()
    key, sub = jax.random.split(key)
    mean_pred, conf = batch_impute_column(
        sub, packed_ext, data_ext, query_col=rul_col, row_ids=new_ids, n_samples=n_samples
    )
    mean_pred = np.array(mean_pred)
    conf = np.array(conf)
    t_imp = time.time() - t0

    ci_out = {}
    for level in CI_LEVELS:
        key, sub = jax.random.split(key)
        _, lo, hi = batch_credible_interval(
            sub,
            packed_ext,
            data_ext,
            query_col=rul_col,
            row_ids=new_ids,
            n_samples=n_samples,
            ci_level=level,
        )
        ci_out[level] = (np.array(lo), np.array(hi))

    # Gather a sample draw matrix for BMA (reuse credible_interval draws
    # indirectly by drawing additional samples keyed per-chain).
    key, sub = jax.random.split(key)
    _, lo50, hi50 = batch_credible_interval(
        sub,
        packed_ext,
        data_ext,
        query_col=rul_col,
        row_ids=new_ids,
        n_samples=n_samples,
        ci_level=0.50,
    )

    del packed_ext, data_ext
    gc.collect()

    return {
        "pred_mean": mean_pred,
        "confidence": conf,
        "ci": ci_out,
        "iqr": (np.array(lo50), np.array(hi50)),
        "timing": {"insert_s": round(t_ins, 1), "impute_s": round(t_imp, 1)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fd", nargs="?", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"]
    )
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    fd = args.fd
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"\nEvaluating {fd} (n_samples={args.samples})")

    # Load preprocessed data + column info
    prep_dir = PREP_ROOT / fd
    test_query = np.load(prep_dir / "test_query.npy")
    truth = np.load(prep_dir / "test_rul_truth.npy")
    info = json.loads((prep_dir / "column_info.json").read_text())
    rul_col = len(info["columns"]) - 1
    assert info["columns"][rul_col]["name"] == "rul"

    inf_dir = INF_ROOT / fd
    if not inf_dir.exists():
        raise FileNotFoundError(f"{inf_dir} missing — run run_inference.py {fd} first")
    meta = json.loads((inf_dir / "inference_meta.json").read_text())
    n_chains = meta["n_chains"]
    # Prefer the training array that inference actually saw (may have been subsampled).
    train_used_path = inf_dir / "train_used.npy"
    if train_used_path.exists():
        train_data = np.load(train_used_path)
    else:
        train_data = np.load(prep_dir / "train_data.npy")
    print(
        f"  Train (trained on): {train_data.shape},  "
        f"Test engines: {test_query.shape[0]},  RUL col: {rul_col}"
    )
    print(f"  Inference: {n_chains} chains, {meta['n_sweeps']} sweeps, mode={meta['mode']}")

    # Per-chain evaluation
    per_chain: list[dict] = []
    for ci in range(n_chains):
        print(f"\n--- Chain {ci} ---")
        per_chain.append(
            _evaluate_chain(
                inf_dir / f"chain_{ci}.jxc",
                train_data,
                test_query,
                rul_col,
                n_samples=args.samples,
                seed=args.seed + ci * 7919,
            )
        )

    # ── BMA aggregation ───────────────────────────────────────────
    stack_pred = np.stack([c["pred_mean"] for c in per_chain])  # (n_chains, n_test)
    bma_mean = stack_pred.mean(axis=0)
    bma_std = stack_pred.std(axis=0)

    # Combined-chain CI: widen per-chain CI by the across-chain spread so
    # the reported interval reflects both within-chain posterior uncertainty
    # and between-chain model uncertainty.
    combined_ci = {}
    for level in CI_LEVELS:
        los = np.stack([c["ci"][level][0] for c in per_chain]).mean(axis=0)
        his = np.stack([c["ci"][level][1] for c in per_chain]).mean(axis=0)
        # Inflate symmetrically by the BMA std scaled by the level's normal quantile
        # (matches the convention in examples/materials_project/impute_dielectric_bma.py).
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[level]
        lo = np.minimum(los, bma_mean - z * bma_std)
        hi = np.maximum(his, bma_mean + z * bma_std)
        combined_ci[level] = (lo, hi)

    # ── Metrics ───────────────────────────────────────────────────
    metrics: dict = {"fd": fd, "n_test_engines": int(truth.shape[0])}
    metrics["bma"] = _metrics(bma_mean, truth)
    for level, (lo, hi) in combined_ci.items():
        coverage = float(np.mean((truth >= lo) & (truth <= hi)))
        width = float(np.mean(hi - lo))
        metrics[f"ci_{int(level * 100)}"] = {"coverage": coverage, "avg_width": width}

    metrics["per_chain"] = []
    for ci, c in enumerate(per_chain):
        m = _metrics(c["pred_mean"], truth)
        m["chain"] = ci
        m.update(c["timing"])
        metrics["per_chain"].append(m)

    # ── Report ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}\n{fd} — RUL PREDICTION METRICS\n{'=' * 70}")
    print(f"Test engines: {truth.shape[0]}")
    print(f"\nBMA across {n_chains} chains:")
    print(f"  MAE  = {metrics['bma']['mae']:.2f} cycles")
    print(f"  RMSE = {metrics['bma']['rmse']:.2f} cycles")
    print(f"  R^2  = {metrics['bma']['r2']:.4f}")
    print(f"  Bias = {metrics['bma']['bias']:+.2f} cycles")

    print("\nCalibration (CI coverage vs nominal):")
    for level in CI_LEVELS:
        c = metrics[f"ci_{int(level * 100)}"]
        print(
            f"  {int(level * 100)}% CI: coverage={c['coverage']:.1%}  "
            f"(target {level:.0%})  avg width={c['avg_width']:.1f} cycles"
        )

    if fd in PUBLISHED_BASELINES:
        print(f"\nPublished baselines on {fd} test set (MAE in cycles):")
        for name, mae in PUBLISHED_BASELINES[fd].items():
            print(f"  {name:35s} {mae:.2f}")
        print(f"  {'jaxcross BMA (this run)':35s} {metrics['bma']['mae']:.2f}")

    # ── Save ──────────────────────────────────────────────────────
    out_dir = EVAL_ROOT / fd
    out_dir.mkdir(parents=True, exist_ok=True)

    cols: dict[str, np.ndarray] = {
        "engine_idx": np.arange(truth.shape[0]),
        "rul_truth": truth,
        "rul_bma_mean": bma_mean,
        "rul_bma_std": bma_std,
    }
    for level, (lo, hi) in combined_ci.items():
        pct = int(level * 100)
        cols[f"ci{pct}_lo"] = lo
        cols[f"ci{pct}_hi"] = hi
        cols[f"ci{pct}_covers"] = (truth >= lo) & (truth <= hi)
    for ci, c in enumerate(per_chain):
        cols[f"chain{ci}_pred"] = c["pred_mean"]

    df = pl.DataFrame(cols)
    df.write_csv(out_dir / "rul_predictions.csv")
    df.write_ipc(out_dir / "rul_predictions.arrow", compression="zstd")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {out_dir / 'rul_predictions.csv'} and metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
