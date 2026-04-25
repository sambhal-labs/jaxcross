#!/usr/bin/env python3
"""Final discovery section: classification calibration (best chain).

Uses jaxcross's `batch_classify_column` to compute predicted P(label = 1 | row)
for each binary clinical label, then reports Brier / log-loss / AUC and a
decile calibration curve. Single-state (best chain) avoids the multi-chain
`batch_conditional_entropy` compile-thrash issue we saw on a 4 GB GPU.

Section 9 (conditional-entropy variable importance) is skipped — the Z-matrix
+ MI table from earlier sections give the same variable-importance signal.

Outputs (results/discovery_warm/):
    classification_calibration.png
    classification_metrics.csv
    discovery_summary.json   (final aggregate of all section artifacts)

Usage:
    uv run python examples/nhanes_clinical/discover_classify.py \\
        [--inference-dir examples/nhanes_clinical/results/inference_warm]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from crosscat import batch_classify_column
from crosscat.serialization import load_packed_state

PREP_DIR = Path("examples/nhanes_clinical/results/preprocessed")
DEFAULT_INF_DIR = Path("examples/nhanes_clinical/results/inference_warm")

LABEL_COLS = ["DIQ010", "BPQ020", "MCQ160C", "RIAGENDR"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", type=str, default=str(DEFAULT_INF_DIR))
    args = parser.parse_args()
    inf_dir = Path(args.inference_dir)

    leaf = inf_dir.name.replace("inference", "discovery", 1)
    out_dir = inf_dir.parent / leaf
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Inference dir: {inf_dir}")
    print(f"Output dir:    {out_dir}")

    info = json.loads((PREP_DIR / "column_info.json").read_text())
    column_names = [c["name"] for c in info["columns"]]
    name_to_idx = {n: i for i, n in enumerate(column_names)}
    train_data = np.load(PREP_DIR / "train_data.npy")
    train_jax = jnp.array(train_data)

    best_packed, _ = load_packed_state(str(inf_dir / "best_chain.jxc"))
    print("Loaded best chain")

    import matplotlib.pyplot as plt

    print(f"\n{'=' * 70}\nCLASSIFICATION CALIBRATION (best chain)\n{'=' * 70}")
    label_targets = [c for c in LABEL_COLS if c in name_to_idx]
    cal_rows: list[dict] = []
    fig, axes = plt.subplots(
        1,
        len(label_targets),
        figsize=(4.0 * max(1, len(label_targets)), 4),
        squeeze=False,
    )
    candidates = jnp.array([0.0, 1.0])
    chunk_size = 1500

    for li, label_name in enumerate(label_targets):
        label_idx = name_to_idx[label_name]
        observed_mask = ~np.isnan(train_data[:, label_idx])
        if observed_mask.sum() < 50:
            continue
        obs_rows = jnp.array(np.where(observed_mask)[0])
        log_p_chunks = []
        for start in range(0, len(obs_rows), chunk_size):
            end = min(start + chunk_size, len(obs_rows))
            chunk = obs_rows[start:end]
            log_p_chunks.append(
                np.asarray(
                    batch_classify_column(
                        best_packed,
                        train_jax,
                        target_col=label_idx,
                        candidate_vals=candidates,
                        row_ids=chunk,
                    )
                )
            )
        log_p = np.concatenate(log_p_chunks, axis=0)
        log_p1 = log_p[:, 1] - np.logaddexp(log_p[:, 0], log_p[:, 1])
        p1 = np.exp(log_p1)
        truths = train_data[observed_mask, label_idx].astype(np.int64)

        brier = float(np.mean((p1 - truths) ** 2))
        ll = float(
            -np.mean(
                truths * np.log(np.clip(p1, 1e-12, 1))
                + (1 - truths) * np.log(np.clip(1 - p1, 1e-12, 1))
            )
        )
        auc = float(roc_auc_score(truths, p1)) if len(set(truths)) > 1 else float("nan")

        order = np.argsort(p1)
        n_bins = 10
        bin_edges = np.linspace(0, len(p1), n_bins + 1, dtype=np.int64)
        binned_p, binned_obs = [], []
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if hi <= lo:
                continue
            sl = order[lo:hi]
            binned_p.append(float(p1[sl].mean()))
            binned_obs.append(float(truths[sl].mean()))

        ax = axes[0, li]
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="ideal")
        ax.plot(binned_p, binned_obs, "o-", color="firebrick", label="empirical")
        ax.set_xlabel("Predicted P(label = 1)")
        ax.set_ylabel("Observed fraction")
        ax.set_title(
            f"{label_name}\nBrier={brier:.3f}  AUC={auc:.3f}  n={int(observed_mask.sum())}",
            fontsize=9,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="lower right")
        cal_rows.append(
            {
                "label": label_name,
                "n_observed": int(observed_mask.sum()),
                "prevalence": float(truths.mean()),
                "brier": brier,
                "log_loss": ll,
                "auc": auc,
            }
        )
        print(
            f"  {label_name}: AUC={auc:.3f}  Brier={brier:.3f}  log-loss={ll:.3f}  "
            f"prevalence={truths.mean():.3f}  n={observed_mask.sum()}"
        )
    plt.suptitle("Classification calibration: jaxcross batch_classify_column", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "classification_calibration.png", dpi=120)
    plt.close()
    pl.DataFrame(cal_rows).write_csv(out_dir / "classification_metrics.csv")

    # Final summary stitched from all section artifacts
    print(f"\n{'=' * 70}\nWRITING FINAL discovery_summary.json\n{'=' * 70}")
    summary: dict = {
        "inference_dir": str(inf_dir),
        "n_rows": int(train_data.shape[0]),
        "n_cols": int(train_data.shape[1]),
        "nan_fraction": float(np.isnan(train_data).mean()),
    }
    meta = json.loads((inf_dir / "inference_meta.json").read_text())
    summary.update(
        {
            "n_chains": meta["n_chains"],
            "n_sweeps": meta.get("n_sweeps"),
            "elapsed_seconds": meta.get("elapsed_seconds"),
            "init_from": meta.get("init_from"),
            "final_log_joints": meta["final_log_joints"],
            "best_chain_idx": meta["best_chain_idx"],
        }
    )
    if (out_dir / "z_matrix.npy").exists():
        z = np.load(out_dir / "z_matrix.npy")
        n = z.shape[0]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((column_names[i], column_names[j], float(z[i, j])))
        pairs.sort(key=lambda x: -x[2])
        summary["z_matrix"] = {
            "shape": list(z.shape),
            "off_diag_mean": float(z[np.triu_indices(n, k=1)].mean()),
            "top_dependencies": [{"a": a, "b": b, "score": s} for a, b, s in pairs[:10]],
        }
    for csv_name in (
        "mi_table.csv",
        "ci_coverage.csv",
        "imputation_metrics.csv",
        "imputation_naturally_missing.csv",
        "label_ari.csv",
    ):
        p = out_dir / csv_name
        if p.exists():
            summary[csv_name.replace(".csv", "")] = pl.read_csv(p).to_dicts()
    summary["classification_metrics"] = cal_rows
    (out_dir / "discovery_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 70}\nDONE — all artifacts in {out_dir}/\n{'=' * 70}")
    n_pngs = sum(1 for f in out_dir.iterdir() if f.suffix == ".png")
    n_csvs = sum(1 for f in out_dir.iterdir() if f.suffix == ".csv")
    print(f"  {n_pngs} PNGs, {n_csvs} CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
