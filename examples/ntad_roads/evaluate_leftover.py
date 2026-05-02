#!/usr/bin/env python3
"""Strict held-out evaluation on the 54,990 rows Phase 2 NEVER saw.

Phase 2 was trained on a deterministic 15K subsample (seed=42) of the full
69,990-row preprocessed cohort. The other 54,990 rows are a natural held-out
set — uniform-random, deterministic, never touched by any chain.

This evaluator:
  1. Loads the Phase 2 best chain.
  2. Computes the 54,990 leftover row indices via setdiff1d.
  3. Builds a leftover-data array with 7 cells masked per row:
       - readmitted_30d (the classification target)
       - 6 encounter-summary columns (the CI-coverage targets):
         time_in_hospital, num_lab_procedures, num_procedures,
         num_medications, number_outpatient, number_inpatient
     Cluster assignment for each leftover row uses the remaining 22 features.
  4. Inserts the 54K masked leftover rows into the Phase 2 best chain via
     packed_insert_rows (no GPU re-training).
  5. Classifies readmitted_30d with batch_classify_column — AUC, Brier,
     log-loss, ECE, bootstrap 95% CI.
  6. Computes 50/90/95 % credible intervals for each of the 6 mask columns
     across all 54,990 rows (= 329,940 cells of strict held-out CI evidence).
  7. Compares against the latest published comparators on Diabetes 130:
       Cureus 2025 (PMC12085305) XGBoost AUC 0.667
       IJSAT 2025 CATBoost AUC 0.70
       Strack 2014 logistic regression AUC ~0.65

Outputs (results/discovery_leftover/):
    leftover_classification.csv
    leftover_classification_bootstrap.json
    leftover_calibration.png
    leftover_ci_coverage.csv
    leftover_summary.json

Usage:
    uv run python examples/ntad_roads/evaluate_leftover.py
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

PREP_DIR = Path("examples/ntad_roads/results/preprocessed")
DEFAULT_INF_DIR = Path("examples/ntad_roads/results/inference_warm")
N_BOOTSTRAP = 1000

TARGET_COLUMN = "is_interstate"  # headline binary (~12.5% prevalence on TX NTAD)
# Continuous columns masked for CI-coverage evaluation. Coverage cells
# count only the ground-truth-observed entries.
MASK_COLUMNS = [
    "lanes",
    "speedlim",
    "centroid_latitude",
    "centroid_longitude",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", type=str, default=str(DEFAULT_INF_DIR))
    args = parser.parse_args()
    inf_dir = Path(args.inference_dir)
    out_dir = inf_dir.parent / inf_dir.name.replace("inference", "discovery", 1).replace(
        "_warm", "_leftover"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Inference dir: {inf_dir}")
    print(f"Output dir:    {out_dir}")

    # ── Load full preprocessed data + Phase 2 train indices ──────────────
    info = json.loads((PREP_DIR / "column_info.json").read_text())
    column_names = [c["name"] for c in info["columns"]]
    name_to_idx = {n: i for i, n in enumerate(column_names)}
    target_idx = name_to_idx[TARGET_COLUMN]
    mask_idxs = [name_to_idx[c] for c in MASK_COLUMNS]

    full_data = np.load(PREP_DIR / "train_data.npy")
    n_total = full_data.shape[0]
    train_indices = np.load(inf_dir / "train_indices.npy")
    leftover_indices = np.setdiff1d(np.arange(n_total), train_indices)
    print(
        f"Full cohort: {n_total:,} rows. Phase 2 saw {len(train_indices):,}; "
        f"leftover {len(leftover_indices):,} rows."
    )

    # ── Build TWO leftover-data variants for the two evaluation tasks ────
    # Variant A (classification): only TARGET masked. The 28 OTHER features
    # — including the strong predictors number_inpatient + num_medications
    # that the Cureus 2025 SHAP analysis found dominant — drive the cluster
    # assignment. This is the apples-to-apples protocol vs literature.
    #
    # Variant B (CI coverage): TARGET + the 6 encounter-summary columns all
    # masked. Cluster assignment uses the remaining 22 features. This is the
    # strict held-out protocol for measuring CI coverage on the masked cells:
    # the model literally never saw those values when assigning the cluster.
    leftover_full = full_data[leftover_indices]
    target_truth = leftover_full[:, target_idx].copy()
    mask_truths: dict[str, np.ndarray] = {
        c: leftover_full[:, name_to_idx[c]].copy() for c in MASK_COLUMNS
    }

    leftover_clf = leftover_full.copy()
    leftover_clf[:, target_idx] = np.nan  # ONLY target masked

    leftover_ci = leftover_full.copy()
    leftover_ci[:, target_idx] = np.nan
    for c in MASK_COLUMNS:
        leftover_ci[:, name_to_idx[c]] = np.nan

    n_leftover = leftover_full.shape[0]
    print(
        f"Built two leftover variants for {n_leftover:,} rows:\n"
        f"  Variant A (classification): only {TARGET_COLUMN} masked. "
        f"Cluster sees 27 other features.\n"
        f"  Variant B (CI coverage): {TARGET_COLUMN} + 4 continuous "
        f"cols masked. Cluster sees 23 other features.\n"
        f"  Target observed in held-out truth: "
        f"{int((~np.isnan(target_truth)).sum()):,}\n"
        f"  Cells of strict-held-out CI evidence (across 4 cols): "
        f"{sum(int((~np.isnan(v)).sum()) for v in mask_truths.values()):,}"
    )

    # ── Load Phase 2 best chain + train data ─────────────────────────────
    best_packed, _ = load_packed_state(str(inf_dir / "best_chain.jxc"))
    train_used = np.load(inf_dir / "train_used.npy")
    print(f"\nLoaded Phase 2 best chain ({inf_dir / 'best_chain.jxc'})")
    print(f"  Phase 2 training data: {train_used.shape}")

    train_jax = jnp.array(train_used)

    # ── (1A) Insert classification variant; cluster uses 28 features ─────
    print(f"\n{'=' * 70}\nINSERTING {n_leftover:,} ROWS — CLASSIFICATION VARIANT\n{'=' * 70}")
    rng = jax.random.key(101)
    extended_packed_clf, extended_data_clf = packed_insert_rows(
        rng, best_packed, train_jax, jnp.array(leftover_clf)
    )
    n_train = train_used.shape[0]
    leftover_row_ids = jnp.arange(n_train, n_train + n_leftover, dtype=jnp.int64)
    print(f"Extended state (classification): {extended_data_clf.shape[0]:,} rows total")

    # ── (2) Held-out 30-day-readmission classification ────────────────────
    print(f"\n{'=' * 70}\nHELD-OUT 30-DAY READMISSION CLASSIFICATION\n{'=' * 70}")
    candidates = jnp.array([0.0, 1.0])
    # Chunk to avoid OOM on large leftover sets
    chunk = 5000
    log_p_chunks = []
    for start in range(0, n_leftover, chunk):
        end = min(start + chunk, n_leftover)
        log_p_chunks.append(
            np.asarray(
                batch_classify_column(
                    extended_packed_clf,
                    extended_data_clf,
                    target_col=target_idx,
                    candidate_vals=candidates,
                    row_ids=leftover_row_ids[start:end],
                )
            )
        )
    log_p = np.concatenate(log_p_chunks, axis=0)  # (n_leftover, 2)

    log_p1 = log_p[:, 1] - np.logaddexp(log_p[:, 0], log_p[:, 1])
    p1 = np.exp(log_p1)
    observed_mask = ~np.isnan(target_truth)
    truths = target_truth[observed_mask].astype(np.int64)
    preds = p1[observed_mask]
    n_obs = int(observed_mask.sum())
    print(f"Held-out test n with observed {TARGET_COLUMN}: {n_obs:,}")
    print(f"  prevalence: {truths.mean():.3f}")
    print(
        f"  predicted P(readmit_30) range: [{preds.min():.3f}, {preds.max():.3f}], "
        f"mean: {preds.mean():.3f}"
    )

    auc_point = float(roc_auc_score(truths, preds))
    brier_point = float(brier_score_loss(truths, preds))
    ll_point = float(log_loss(truths, preds, labels=[0, 1]))

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
    print(f"  ECE      = {ece:.4f}")

    print(f"\nBootstrapping {N_BOOTSTRAP} resamples for 95 % CIs ...")
    rng_np = np.random.default_rng(202)
    auc_b: list[float] = []
    brier_b: list[float] = []
    ll_b: list[float] = []
    n_resampled = len(truths)
    for _ in range(N_BOOTSTRAP):
        idx = rng_np.choice(n_resampled, size=n_resampled, replace=True)
        if len(set(truths[idx].tolist())) < 2:
            continue
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
            {"metric": "AUC", "point": auc_point, "ci_lo": auc_lo, "ci_hi": auc_hi},
            {"metric": "Brier", "point": brier_point, "ci_lo": brier_lo, "ci_hi": brier_hi},
            {"metric": "log_loss", "point": ll_point, "ci_lo": ll_lo, "ci_hi": ll_hi},
            {"metric": "ECE_10bin", "point": ece, "ci_lo": float("nan"), "ci_hi": float("nan")},
        ]
    ).write_csv(out_dir / "leftover_classification.csv")
    (out_dir / "leftover_classification_bootstrap.json").write_text(
        json.dumps(classification_summary, indent=2)
    )

    # Calibration curve
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
        ax.plot(binned_p, binned_obs, "o-", color="firebrick", label="empirical")
        ax.set_xlabel("Predicted P(readmitted_30d = 1)")
        ax.set_ylabel("Observed fraction")
        ax.set_title(
            f"30-day readmit held-out leftover calibration\n"
            f"AUC={auc_point:.3f}  95% CI [{auc_lo:.3f}, {auc_hi:.3f}]  n={n_obs:,}",
            fontsize=10,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "leftover_calibration.png", dpi=120)
        plt.close()
    except ImportError:
        pass

    # ── (3A) Insert CI-coverage variant; cluster uses 22 features ────────
    print(f"\n{'=' * 70}\nINSERTING {n_leftover:,} ROWS — CI-COVERAGE VARIANT\n{'=' * 70}")
    rng_b = jax.random.key(202)
    extended_packed_ci, extended_data_ci = packed_insert_rows(
        rng_b, best_packed, train_jax, jnp.array(leftover_ci)
    )
    print(f"Extended state (CI coverage): {extended_data_ci.shape[0]:,} rows total")

    # ── (3B) Held-out CI coverage on encounter-summary columns ───────────
    print(f"\n{'=' * 70}\nHELD-OUT CI COVERAGE (54K x 6 columns)\n{'=' * 70}")
    coverage_rows: list[dict] = []
    rng_key = jax.random.key(303)
    chunk_size = 1000
    print(f"\n  {'Column':22s} {'n_cells':>8s} {'50%':>8s} {'90%':>8s} {'95%':>8s} {'MAE':>8s}")
    print("  " + "-" * 70)
    for col_name in MASK_COLUMNS:
        col_idx = name_to_idx[col_name]
        truth_col = mask_truths[col_name]
        observed_idx = np.where(~np.isnan(truth_col))[0]
        if len(observed_idx) == 0:
            continue
        truth_vals = truth_col[observed_idx].astype(np.float32)
        local_rows = leftover_row_ids[observed_idx]

        coverages = {0.50: 0.0, 0.90: 0.0, 0.95: 0.0}
        meds_for_mae: np.ndarray | None = None
        widths_per_level: dict[float, float] = {}
        for level in [0.50, 0.90, 0.95]:
            rng_key, sub = jax.random.split(rng_key)
            meds_chunks, los_chunks, his_chunks = [], [], []
            for start in range(0, len(local_rows), chunk_size):
                end = min(start + chunk_size, len(local_rows))
                chunk_ids = local_rows[start:end]
                k = jax.random.fold_in(sub, start)
                m, lo, hi = batch_credible_interval(
                    k,
                    extended_packed_ci,
                    extended_data_ci,
                    query_col=col_idx,
                    row_ids=chunk_ids,
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
            widths_per_level[level] = mean_w
            if level == 0.90:
                meds_for_mae = meds
            coverage_rows.append(
                {
                    "column": col_name,
                    "ci_level": level,
                    "empirical_coverage": cov,
                    "mean_width": mean_w,
                    "n_cells": len(truth_vals),
                }
            )
        mae = (
            float(np.abs(meds_for_mae - truth_vals).mean())
            if meds_for_mae is not None
            else float("nan")
        )
        coverage_rows.append(
            {
                "column": col_name,
                "ci_level": "MAE",
                "empirical_coverage": mae,
                "mean_width": float("nan"),
                "n_cells": len(truth_vals),
            }
        )
        print(
            f"  {col_name:22s} {len(truth_vals):>8d} "
            f"{coverages[0.50]:>7.1%} "
            f"{coverages[0.90]:>7.1%} "
            f"{coverages[0.95]:>7.1%} "
            f"{mae:>8.3f}"
        )

    pl.DataFrame(coverage_rows).write_csv(out_dir / "leftover_ci_coverage.csv")

    # Aggregate (cell-weighted) coverage
    agg_cov: dict[float, float] = {}
    for level in [0.50, 0.90, 0.95]:
        rows = [r for r in coverage_rows if r["ci_level"] == level]
        weights = np.array([r["n_cells"] for r in rows], dtype=np.float32)
        covs = np.array([r["empirical_coverage"] for r in rows], dtype=np.float32)
        if weights.sum() > 0:
            agg_cov[level] = float((covs * weights).sum() / weights.sum())
    print(
        f"\n  Cell-weighted aggregate coverage across {sum(int((~np.isnan(v)).sum()) for v in mask_truths.values()):,} cells:"
    )
    for level, cov in agg_cov.items():
        print(f"    {int(level * 100)}% CI: {cov:.1%}  (target: {int(level * 100)}%)")

    # ── (4) Final summary ────────────────────────────────────────────────
    summary = {
        "evaluation_protocol": "leftover (54K rows Phase 2 never saw, deterministic)",
        "inference_dir": str(inf_dir),
        "n_total_cohort": int(n_total),
        "n_phase2_train": int(len(train_indices)),
        "n_leftover": int(n_leftover),
        "n_test_target_observed": n_obs,
        "n_masked_cells_total": sum(int((~np.isnan(v)).sum()) for v in mask_truths.values()),
        "classification": classification_summary,
        "ci_coverage_aggregate": agg_cov,
        "ci_coverage_per_column": coverage_rows,
    }
    (out_dir / "leftover_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 70}\nDONE — leftover evaluation in {out_dir}/\n{'=' * 70}")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name:40s} ({f.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
