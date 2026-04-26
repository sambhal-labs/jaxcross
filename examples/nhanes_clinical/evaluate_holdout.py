#!/usr/bin/env python3
"""Held-out evaluation of the NHANES jaxcross discovery model.

Loads the Phase-3 best chain (trained on 7,403 train rows with 5 % of biomarker
cells masked), inserts the 1,851 held-out test rows via `packed_insert_rows`
(test rows had DIQ010 set to NaN before inference), and reports:

  1. Held-out diabetes classification (DIQ010): AUC, Brier, log-loss, ECE, with
     bootstrap 95 % CIs across 1,000 resamples — apples-to-apples vs the
     literature (Mehrabkhani 2025: 0.817 AUC; 3-cycle 2013-2018: 0.903 AUC).
  2. Held-out 90 / 50 / 95 % credible-interval coverage on the masked
     biomarker cells (1,432 cells across LBXGH, LBXSGL, BMXBMI, BPXSY1,
     LBXTC, LBDLDL) — the regulator-friendly story under the strictest
     held-out evaluation (model did not see these values during training).
  3. Decile calibration curves and per-column coverage tables.

Outputs (results/discovery_holdout/):
    holdout_classification.csv
    holdout_classification_bootstrap.json
    holdout_calibration.png
    holdout_ci_coverage.csv
    holdout_summary.json

Usage:
    uv run python examples/nhanes_clinical/evaluate_holdout.py \\
        [--inference-dir examples/nhanes_clinical/results/inference_holdout] \\
        [--prep-dir      examples/nhanes_clinical/results/preprocessed_holdout]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from crosscat import (
    batch_classify_column,
    batch_credible_interval,
)
from crosscat.packed.kernels import packed_insert_rows
from crosscat.serialization import load_packed_state

DEFAULT_INF_DIR = Path("examples/nhanes_clinical/results/inference_holdout")
DEFAULT_PREP_DIR = Path("examples/nhanes_clinical/results/preprocessed_holdout")
N_BOOTSTRAP = 1000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", type=str, default=str(DEFAULT_INF_DIR))
    parser.add_argument("--prep-dir", type=str, default=str(DEFAULT_PREP_DIR))
    args = parser.parse_args()
    inf_dir = Path(args.inference_dir)
    prep_dir = Path(args.prep_dir)

    out_dir = inf_dir.parent / inf_dir.name.replace("inference", "discovery", 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Inference dir: {inf_dir}")
    print(f"Prep dir:      {prep_dir}")
    print(f"Output dir:    {out_dir}")

    info = json.loads((prep_dir / "column_info.json").read_text())
    column_names = [c["name"] for c in info["columns"]]
    name_to_idx = {n: i for i, n in enumerate(column_names)}
    diq_idx = name_to_idx["DIQ010"]

    train_data = np.load(prep_dir / "train_data.npy")
    test_data = np.load(prep_dir / "test_data.npy")
    holdout_meta = json.loads((prep_dir / "holdout_meta.json").read_text())
    test_diq_truth = np.array(
        [np.nan if v is None else v for v in holdout_meta["test_diq010_ground_truth"]],
        dtype=np.float32,
    )
    masked_cells = holdout_meta["masked_cells"]

    print(
        f"Train: {train_data.shape}  Test: {test_data.shape}  "
        f"Masked cells: {len(masked_cells)}  Test DIQ010 observed: "
        f"{int((~np.isnan(test_diq_truth)).sum())}"
    )

    best_packed, _ = load_packed_state(str(inf_dir / "best_chain.jxc"))

    train_jax = jnp.array(train_data)
    test_jax = jnp.array(test_data)

    # ── (1) Insert test rows into the best chain ──────────────────────────
    print(f"\n{'=' * 70}\nINSERTING {test_data.shape[0]:,} TEST ROWS\n{'=' * 70}")
    rng = jax.random.key(101)
    extended_packed, extended_data = packed_insert_rows(rng, best_packed, train_jax, test_jax)
    n_train = train_data.shape[0]
    n_test = test_data.shape[0]
    test_row_ids = jnp.arange(n_train, n_train + n_test, dtype=jnp.int64)
    print(f"Extended state: {extended_data.shape[0]:,} rows total")

    # ── (2) Held-out diabetes classification ──────────────────────────────
    print(f"\n{'=' * 70}\nHELD-OUT DIABETES CLASSIFICATION\n{'=' * 70}")
    candidates = jnp.array([0.0, 1.0])
    log_p = np.asarray(
        batch_classify_column(
            extended_packed,
            extended_data,
            target_col=diq_idx,
            candidate_vals=candidates,
            row_ids=test_row_ids,
        )
    )  # (n_test, 2)
    log_p1 = log_p[:, 1] - np.logaddexp(log_p[:, 0], log_p[:, 1])
    p1 = np.exp(log_p1)
    observed_mask = ~np.isnan(test_diq_truth)
    truths = test_diq_truth[observed_mask].astype(np.int64)
    preds = p1[observed_mask]
    n_obs = int(observed_mask.sum())
    print(f"Held-out test n with observed DIQ010: {n_obs}")
    print(f"  prevalence: {truths.mean():.3f}")
    print(
        f"  predicted P(diabetes) range: [{preds.min():.3f}, {preds.max():.3f}], "
        f"mean: {preds.mean():.3f}"
    )

    auc_point = float(roc_auc_score(truths, preds))
    brier_point = float(brier_score_loss(truths, preds))
    ll_point = float(log_loss(truths, preds, labels=[0, 1]))

    # ECE (10-bin)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = (preds >= bin_edges[b]) & (
            preds < bin_edges[b + 1] + (1 if b == n_bins - 1 else 0)
        )
        if in_bin.sum() == 0:
            continue
        bin_acc = float(truths[in_bin].mean())
        bin_conf = float(preds[in_bin].mean())
        ece += (in_bin.sum() / n_obs) * abs(bin_acc - bin_conf)
    ece = float(ece)

    print("\nHeld-out point metrics:")
    print(f"  AUC      = {auc_point:.4f}")
    print(f"  Brier    = {brier_point:.4f}")
    print(f"  log-loss = {ll_point:.4f}")
    print(f"  ECE      = {ece:.4f}  (lower is better; 0 = perfectly calibrated)")

    # ── Bootstrap 95 % CIs on AUC, Brier, log-loss ────────────────────────
    print(f"\nBootstrapping {N_BOOTSTRAP} resamples for 95 % CIs ...")
    rng_np = np.random.default_rng(202)
    auc_b: list[float] = []
    brier_b: list[float] = []
    ll_b: list[float] = []
    n_resampled = len(truths)
    for _ in range(N_BOOTSTRAP):
        idx = rng_np.choice(n_resampled, size=n_resampled, replace=True)
        if len(set(truths[idx].tolist())) < 2:
            continue  # skip degenerate resamples
        auc_b.append(float(roc_auc_score(truths[idx], preds[idx])))
        brier_b.append(float(brier_score_loss(truths[idx], preds[idx])))
        ll_b.append(float(log_loss(truths[idx], preds[idx], labels=[0, 1])))
    auc_lo, auc_hi = float(np.percentile(auc_b, 2.5)), float(np.percentile(auc_b, 97.5))
    brier_lo, brier_hi = float(np.percentile(brier_b, 2.5)), float(np.percentile(brier_b, 97.5))
    ll_lo, ll_hi = float(np.percentile(ll_b, 2.5)), float(np.percentile(ll_b, 97.5))
    print(f"  AUC      = {auc_point:.4f}  95 % CI [{auc_lo:.4f}, {auc_hi:.4f}]")
    print(f"  Brier    = {brier_point:.4f}  95 % CI [{brier_lo:.4f}, {brier_hi:.4f}]")
    print(f"  log-loss = {ll_point:.4f}  95 % CI [{ll_lo:.4f}, {ll_hi:.4f}]")

    classification_summary = {
        "n_observed": n_obs,
        "prevalence": float(truths.mean()),
        "auc_point": auc_point,
        "auc_95ci": [auc_lo, auc_hi],
        "brier_point": brier_point,
        "brier_95ci": [brier_lo, brier_hi],
        "log_loss_point": ll_point,
        "log_loss_95ci": [ll_lo, ll_hi],
        "ece_10bin": ece,
        "n_bootstrap": N_BOOTSTRAP,
    }
    pl.DataFrame(
        [
            {
                "metric": "AUC",
                "point": auc_point,
                "ci_lo": auc_lo,
                "ci_hi": auc_hi,
            },
            {"metric": "Brier", "point": brier_point, "ci_lo": brier_lo, "ci_hi": brier_hi},
            {"metric": "log_loss", "point": ll_point, "ci_lo": ll_lo, "ci_hi": ll_hi},
            {"metric": "ECE_10bin", "point": ece, "ci_lo": float("nan"), "ci_hi": float("nan")},
        ]
    ).write_csv(out_dir / "holdout_classification.csv")
    (out_dir / "holdout_classification_bootstrap.json").write_text(
        json.dumps(classification_summary, indent=2)
    )

    # Decile calibration curve
    try:
        import matplotlib.pyplot as plt

        order = np.argsort(preds)
        edges = np.linspace(0, len(preds), 11, dtype=np.int64)
        binned_p, binned_obs = [], []
        for b in range(10):
            sl = order[edges[b] : edges[b + 1]]
            if len(sl) == 0:
                continue
            binned_p.append(float(preds[sl].mean()))
            binned_obs.append(float(truths[sl].mean()))
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="ideal")
        ax.plot(binned_p, binned_obs, "o-", color="firebrick", label="empirical (decile bins)")
        ax.set_xlabel("Predicted P(DIQ010 = 1)")
        ax.set_ylabel("Observed fraction")
        ax.set_title(
            f"DIQ010 held-out calibration\nAUC={auc_point:.3f}  "
            f"95% CI [{auc_lo:.3f}, {auc_hi:.3f}]  n={n_obs}",
            fontsize=10,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "holdout_calibration.png", dpi=120)
        plt.close()
    except ImportError:
        pass

    # ── (3) Held-out CI coverage on masked biomarker cells ────────────────
    print(f"\n{'=' * 70}\nHELD-OUT CI COVERAGE\n{'=' * 70}")
    by_col: dict[str, list[dict]] = {}
    for c in masked_cells:
        by_col.setdefault(c["col_name"], []).append(c)

    coverage_rows: list[dict] = []
    rng_key = jax.random.key(303)
    chunk_size = 500
    print(f"\n  {'Column':10s} {'n_cells':>8s} {'50%':>8s} {'90%':>8s} {'95%':>8s} {'MAE':>8s}")
    print("  " + "-" * 60)
    for col_name, cells in by_col.items():
        col_idx = name_to_idx[col_name]
        local_rows = jnp.array([c["train_row_idx"] for c in cells], dtype=jnp.int64)
        truth_vals = np.array([c["ground_truth"] for c in cells], dtype=np.float32)

        widths_50 = []
        coverages = {0.50: [], 0.90: [], 0.95: []}
        meds_all = []
        for level in [0.50, 0.90, 0.95]:
            rng_key, sub = jax.random.split(rng_key)
            meds_chunks = []
            los_chunks = []
            his_chunks = []
            for start in range(0, len(local_rows), chunk_size):
                end = min(start + chunk_size, len(local_rows))
                chunk = local_rows[start:end]
                key = jax.random.fold_in(sub, start)
                m, lo, hi = batch_credible_interval(
                    key,
                    best_packed,
                    train_jax,
                    query_col=col_idx,
                    row_ids=chunk,
                    n_samples=200,
                    ci_level=level,
                )
                meds_chunks.append(np.asarray(m))
                los_chunks.append(np.asarray(lo))
                his_chunks.append(np.asarray(hi))
            meds = np.concatenate(meds_chunks)
            los = np.concatenate(los_chunks)
            his = np.concatenate(his_chunks)
            cov = float(((truth_vals >= los) & (truth_vals <= his)).mean())
            mean_w = float((his - los).mean())
            coverages[level] = cov
            if level == 0.50:
                widths_50.append(mean_w)
            if level == 0.90:
                meds_all.append(meds)
            coverage_rows.append(
                {
                    "column": col_name,
                    "ci_level": level,
                    "empirical_coverage": cov,
                    "mean_width": mean_w,
                    "n_cells": len(cells),
                }
            )
        meds_arr = meds_all[0]
        mae = float(np.abs(meds_arr - truth_vals).mean())
        coverage_rows.append(
            {
                "column": col_name,
                "ci_level": "MAE",
                "empirical_coverage": mae,
                "mean_width": float("nan"),
                "n_cells": len(cells),
            }
        )
        print(
            f"  {col_name:10s} {len(cells):>8d} "
            f"{coverages[0.50]:>7.1%} "
            f"{coverages[0.90]:>7.1%} "
            f"{coverages[0.95]:>7.1%} "
            f"{mae:>8.3f}"
        )

    pl.DataFrame(coverage_rows).write_csv(out_dir / "holdout_ci_coverage.csv")

    # Compute aggregate (cell-weighted) coverage
    agg_cov: dict[float, float] = {}
    for level in [0.50, 0.90, 0.95]:
        rows = [r for r in coverage_rows if r["ci_level"] == level]
        weights = np.array([r["n_cells"] for r in rows], dtype=np.float32)
        covs = np.array([r["empirical_coverage"] for r in rows], dtype=np.float32)
        if weights.sum() > 0:
            agg_cov[level] = float((covs * weights).sum() / weights.sum())
    print(f"\n  Cell-weighted aggregate coverage across all {len(masked_cells)} cells:")
    for level, cov in agg_cov.items():
        print(f"    {int(level * 100)}% CI: {cov:.1%}  (target: {int(level * 100)}%)")

    # ── (4) Final summary ────────────────────────────────────────────────
    summary = {
        "inference_dir": str(inf_dir),
        "prep_dir": str(prep_dir),
        "n_train": int(train_data.shape[0]),
        "n_test": int(test_data.shape[0]),
        "n_test_diq010_observed": n_obs,
        "n_masked_cells": len(masked_cells),
        "classification": classification_summary,
        "ci_coverage_aggregate": agg_cov,
        "ci_coverage_per_column": coverage_rows,
        "literature_comparison_AUC": {
            "Mehrabkhani_2025_NHANES_2007_2018": 0.817,
            "three_cycle_2013_2018": 0.903,
            "Dinh_2019_NHANES_1999_2014": 0.86,
            "CATBoost_NHANES_2017_2020": 0.83,
        },
    }
    (out_dir / "holdout_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 70}\nDONE — held-out evaluation in {out_dir}/\n{'=' * 70}")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name:40s} ({f.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
