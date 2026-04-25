#!/usr/bin/env python3
"""Discover clinical structure in NHANES via jaxcross.

Loads the trained packed states and runs the FULL set of structure-discovery
analyses jaxcross was built for. Produces publication-quality figures of the
discovered views (column groups) and clusters (row groups within each view).

Sections:
  1. View structure        — which columns cluster together (column views)
  2. Dependence matrix     — pairwise Z-matrix (BMA across chains)
  3. Mutual information    — for curated clinical-pair questions
  4. Row typicality        — atypical/typical scores per participant
  5. Anomaly scores        — likelihood-based anomaly per participant
  6. Patient similarity    — find the most similar patients to anchors
  7. Publication figures   — sorted Z-matrix, per-view cluster heatmap,
                             cluster mean profiles, cluster size bars,
                             view consistency across chains, label ARI

Outputs (results/<discovery-dir>/):
    views_per_chain.json     view assignments per chain (column lists)
    z_matrix.npy             29x29 dependency matrix (chains BMA)
    z_matrix.csv             same, with column-name headers
    z_matrix.png             heatmap visualization
    z_matrix_sorted.png      same, columns permuted by best-chain views
    view_overview.png        column membership + cluster count per view
    view_consistency.png     between-chain ARI of column partitions
    label_ari.csv            ARI of each view vs binary clinical labels
    cluster_profile_v{vi}.png standardized cluster mean per view (heatmap)
    cluster_sizes_v{vi}.png  per-view cluster size bar chart
    mi_table.csv             curated clinical-pair MI table
    typicality.csv           per-participant typicality + seqn
    anomaly.csv              per-participant anomaly score + seqn
    similarity_anchors.csv   anchor similarity sub-matrix
    nearest_neighbours.csv   top-5 nearest neighbours per anchor (full cohort)
    discovery_summary.json   aggregate stats + headline findings

Usage (Phase 1 cold-start chains):
    uv run python examples/nhanes_clinical/discover_structure.py

Usage (Phase 2 warm-start chains):
    uv run python examples/nhanes_clinical/discover_structure.py \\
        --inference-dir examples/nhanes_clinical/results/inference_warm \\
        --out-subdir discovery_warm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from sklearn.metrics import adjusted_rand_score

from crosscat import (
    batch_anomaly_score,
    batch_row_similarity,
    batch_row_typicality,
    packed_dependence_matrix,
    packed_mutual_information,
)
from crosscat.packed import unpack_state
from crosscat.serialization import load_packed_state
from crosscat.types import ColumnType

PREP_DIR = Path("examples/nhanes_clinical/results/preprocessed")
DEFAULT_INF_DIR = Path("examples/nhanes_clinical/results/inference")
RESULTS_ROOT = Path("examples/nhanes_clinical/results")

_TYPE_MAP = {
    "CONTINUOUS": ColumnType.CONTINUOUS,
    "CATEGORICAL": ColumnType.CATEGORICAL,
    "ORDINAL": ColumnType.ORDINAL,
    "BINARY": ColumnType.BINARY,
    "CYCLIC": ColumnType.CYCLIC,
}

# Curated clinical pairs to compute mutual information for. These are
# pairs where domain knowledge tells us the answer; high MI confirms
# jaxcross learned the correlation, low MI means our model missed it.
MI_PAIRS = [
    ("LBXGH", "LBXSGL", "Glycohemoglobin (HbA1c) ↔ glucose"),
    ("LBXGH", "DIQ010", "HbA1c ↔ diabetes self-report"),
    ("LBXSGL", "DIQ010", "Glucose ↔ diabetes self-report"),
    ("LBXTC", "LBDLDL", "Total chol ↔ LDL"),
    ("LBXTC", "LBDHDD", "Total chol ↔ HDL"),
    ("LBXTR", "LBDLDL", "Triglycerides ↔ LDL"),
    ("LBXSASSI", "LBXSATSI", "AST ↔ ALT (liver enzymes)"),
    ("LBXRBCSI", "LBXHGB", "RBC count ↔ hemoglobin"),
    ("LBXHGB", "LBXMCVSI", "Hemoglobin ↔ mean cell volume"),
    ("BPXSY1", "BPXDI1", "Systolic ↔ diastolic BP"),
    ("BPXSY1", "BPQ020", "Systolic BP ↔ hypertension self-report"),
    ("BMXBMI", "BMXWAIST", "BMI ↔ waist circumference"),
    ("BMXBMI", "DIQ010", "BMI ↔ diabetes"),
    ("RIDAGEYR", "MCQ160C", "Age ↔ coronary heart disease"),
    ("RIDAGEYR", "BPQ020", "Age ↔ hypertension"),
    # Negative controls — should be near zero
    ("LBXMCVSI", "RIDRETH3", "MCV ↔ race (negative control)"),
    ("BPXPLS", "DMDEDUC2", "Pulse ↔ education (negative control)"),
]

# Binary clinical labels we'll evaluate per-view row clustering against
# (with ARI). These exist as columns in the matrix so they're already part
# of the joint model; ARI tells us which view captures each label best.
LABEL_COLS = ["DIQ010", "BPQ020", "MCQ160C", "RIAGENDR"]


# ---------------------------------------------------------------------------
# Publication figures
# ---------------------------------------------------------------------------


def _make_publication_figures(
    chains: list,
    best_packed,
    column_types: list[ColumnType],
    column_names: list[str],
    train_data: np.ndarray,
    z: np.ndarray,
    name_to_idx: dict[str, int],
    out_dir: Path,
) -> dict:
    """Build all view + cluster visualizations. Returns extra summary dict."""
    import matplotlib.pyplot as plt

    n_cols = len(column_names)

    # Use the best chain (already the model's MAP-ish posterior sample)
    # for per-view cluster visualizations and the Z-matrix sort order.
    best_state = unpack_state(best_packed, column_types)
    views_sorted = sorted(best_state.views, key=lambda v: -len(v.column_indices))
    n_views = len(views_sorted)

    # ── (a) View-sorted Z-matrix ─────────────────────────────────────────
    perm: list[int] = []
    view_block_starts: list[int] = []
    for v in views_sorted:
        view_block_starts.append(len(perm))
        perm.extend(int(c) for c in v.column_indices)
    perm_arr = np.asarray(perm, dtype=np.int64)
    z_sorted = z[perm_arr][:, perm_arr]

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(z_sorted, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_cols))
    perm_names = [column_names[c] for c in perm_arr]
    ax.set_xticklabels(perm_names, rotation=90, fontsize=7)
    ax.set_yticklabels(perm_names, fontsize=7)
    # Block boundary lines (white) marking view partitions
    for s in view_block_starts[1:]:
        ax.axhline(s - 0.5, color="white", linewidth=1.5)
        ax.axvline(s - 0.5, color="white", linewidth=1.5)
    plt.colorbar(im, ax=ax, label="P(same view, BMA)")
    ax.set_title(f"NHANES dependency matrix sorted by best-chain views ({n_views} views)")
    plt.tight_layout()
    plt.savefig(out_dir / "z_matrix_sorted.png", dpi=120)
    plt.close()

    # ── (b) View overview ────────────────────────────────────────────────
    # Bar of view sizes + textbox of cluster counts
    fig, ax = plt.subplots(figsize=(10, max(3, n_views * 0.55)))
    sizes = [len(v.column_indices) for v in views_sorted]
    cluster_counts = [
        len(set(int(x) for x in np.asarray(v.row_assignments))) for v in views_sorted
    ]
    y = np.arange(n_views)
    ax.barh(y, sizes, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"View {i}\n({cc} clusters)" for i, cc in enumerate(cluster_counts)])
    ax.invert_yaxis()
    ax.set_xlabel("Number of columns")
    ax.set_title("Best-chain view structure (columns per view + cluster count)")
    for i, v in enumerate(views_sorted):
        cols = ", ".join(column_names[c] for c in v.column_indices)
        ax.text(sizes[i] + 0.3, i, cols, va="center", fontsize=7)
    ax.set_xlim(
        0,
        max(sizes)
        + max(
            8,
            max(len(", ".join(column_names[c] for c in v.column_indices)) for v in views_sorted)
            * 0.18,
        ),
    )
    plt.tight_layout()
    plt.savefig(out_dir / "view_overview.png", dpi=120)
    plt.close()

    # ── (c) Per-view cluster mean profile + cluster sizes ────────────────
    # For each view we standardize each column over observed rows, compute
    # the per-cluster mean (NaN-skip), and plot a (n_clusters x n_view_cols)
    # heatmap where colour = standardized cluster mean.
    # Sister figure: cluster size bar chart.
    cluster_profile_summary = []
    for vi, v in enumerate(views_sorted):
        col_idx = np.asarray(v.column_indices, dtype=np.int64)
        row_assign = np.asarray(v.row_assignments)
        cluster_ids = sorted(set(int(c) for c in row_assign))
        n_clusters = len(cluster_ids)
        view_cols = [column_names[c] for c in col_idx]

        # Standardize each column (over observed values)
        view_data = train_data[:, col_idx]
        col_mean = np.nanmean(view_data, axis=0)
        col_std = np.nanstd(view_data, axis=0)
        col_std[col_std < 1e-9] = 1.0
        view_z = (view_data - col_mean) / col_std

        # Cluster means (NaN-skip)
        means = np.full((n_clusters, len(col_idx)), np.nan, dtype=np.float32)
        sizes_per_cluster = np.zeros(n_clusters, dtype=np.int64)
        for ki, k in enumerate(cluster_ids):
            mask = row_assign == k
            sizes_per_cluster[ki] = int(mask.sum())
            if mask.any():
                means[ki] = np.nanmean(view_z[mask], axis=0)

        cluster_profile_summary.append(
            {
                "view_idx": vi,
                "view_columns": view_cols,
                "n_clusters": n_clusters,
                "cluster_sizes": sizes_per_cluster.tolist(),
            }
        )

        # Profile heatmap
        h = max(2.5, 0.32 * n_clusters)
        w = max(5.5, 0.6 * len(col_idx))
        fig, ax = plt.subplots(figsize=(w, h))
        # Symmetric colour scale around 0 so departure from population mean
        # reads at a glance
        vmax = float(np.nanmax(np.abs(means))) if np.isfinite(np.nanmax(np.abs(means))) else 1.0
        vmax = max(vmax, 0.5)
        im = ax.imshow(means, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(col_idx)))
        ax.set_xticklabels(view_cols, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(n_clusters))
        ax.set_yticklabels([f"C{ki} (n={sizes_per_cluster[ki]})" for ki in range(n_clusters)])
        plt.colorbar(im, ax=ax, label="Standardized cluster mean (z-score)")
        ax.set_title(f"View {vi} cluster profile — {n_clusters} clusters x {len(col_idx)} cols")
        plt.tight_layout()
        plt.savefig(out_dir / f"cluster_profile_v{vi:02d}.png", dpi=120)
        plt.close()

        # Cluster sizes bar
        fig, ax = plt.subplots(figsize=(max(5, 0.45 * n_clusters), 3))
        bars = ax.bar(
            np.arange(n_clusters),
            sizes_per_cluster,
            color="seagreen",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(np.arange(n_clusters))
        ax.set_xticklabels([f"C{ki}" for ki in range(n_clusters)])
        ax.set_ylabel("Number of participants")
        ax.set_title(f"View {vi} cluster sizes (total n={int(sizes_per_cluster.sum())})")
        for bar, sz in zip(bars, sizes_per_cluster, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(sz),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.tight_layout()
        plt.savefig(out_dir / f"cluster_sizes_v{vi:02d}.png", dpi=120)
        plt.close()

    # ── (d) View consistency across chains (column-partition ARI) ────────
    n_chains = len(chains)
    chain_partitions: list[np.ndarray] = []
    for packed in chains:
        st = unpack_state(packed, column_types)
        labels = np.full(n_cols, -1, dtype=np.int64)
        for vi, v in enumerate(st.views):
            for c in v.column_indices:
                labels[int(c)] = vi
        chain_partitions.append(labels)
    consistency = np.zeros((n_chains, n_chains), dtype=np.float32)
    for i in range(n_chains):
        for j in range(n_chains):
            consistency[i, j] = adjusted_rand_score(chain_partitions[i], chain_partitions[j])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(consistency, cmap="cividis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(n_chains))
    ax.set_yticks(np.arange(n_chains))
    ax.set_xticklabels([f"Chain {i}" for i in range(n_chains)])
    ax.set_yticklabels([f"Chain {i}" for i in range(n_chains)])
    for i in range(n_chains):
        for j in range(n_chains):
            ax.text(
                j,
                i,
                f"{consistency[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if consistency[i, j] < 0.6 else "black",
                fontsize=9,
            )
    plt.colorbar(im, ax=ax, label="Adjusted Rand Index (column partitions)")
    ax.set_title("Between-chain view consistency (ARI of column partitions)")
    plt.tight_layout()
    plt.savefig(out_dir / "view_consistency.png", dpi=120)
    plt.close()

    # ── (e) Label ARI: which view captures which binary label? ───────────
    label_rows = []
    for label_name in LABEL_COLS:
        if label_name not in name_to_idx:
            continue
        label_col_idx = name_to_idx[label_name]
        label_vals = train_data[:, label_col_idx]
        observed_mask = ~np.isnan(label_vals)
        if observed_mask.sum() < 50:
            continue
        label_obs = label_vals[observed_mask].astype(np.int64)
        for vi, v in enumerate(views_sorted):
            row_assign = np.asarray(v.row_assignments)
            ari = float(adjusted_rand_score(label_obs, row_assign[observed_mask]))
            label_rows.append(
                {
                    "label": label_name,
                    "n_observed": int(observed_mask.sum()),
                    "view_idx": vi,
                    "view_n_cols": len(v.column_indices),
                    "ari": ari,
                    "view_contains_label_col": bool(
                        label_col_idx in [int(c) for c in v.column_indices]
                    ),
                }
            )
    if label_rows:
        pl.DataFrame(label_rows).write_csv(out_dir / "label_ari.csv")

    return {
        "view_consistency_offdiag_mean": float(consistency[np.triu_indices(n_chains, k=1)].mean())
        if n_chains > 1
        else 1.0,
        "cluster_profile_summary": cluster_profile_summary,
        "label_ari": label_rows,
    }


# ---------------------------------------------------------------------------
# Advanced figures: imputation, conditional entropy, classification
# ---------------------------------------------------------------------------


def _make_advanced_figures(
    chains: list,
    best_packed,
    column_names: list[str],
    train_data: np.ndarray,
    name_to_idx: dict[str, int],
    out_dir: Path,
) -> dict:
    """Imputation calibration + conditional entropy + classification calibration.

    Uses jaxcross's batch_impute_column, batch_credible_interval,
    batch_conditional_entropy, and batch_classify_column to produce the
    quantitative figures that anchor the publication's commercial story
    (calibrated uncertainty on clinical estimates).
    """
    import matplotlib.pyplot as plt

    from crosscat import (
        batch_classify_column,
        batch_conditional_entropy,
        batch_credible_interval,
    )

    out: dict = {}
    train_jax = jnp.array(train_data)
    n_rows, n_cols = train_data.shape

    # Helper: chunked batch_credible_interval to fit on small GPUs.
    # n_obs ~ 9k * n_samples * n_clusters_max would OOM on a 4 GB card,
    # so we run in 1000-row chunks and concat.
    def _chunked_credible_interval(rng_key_seed, query_col, row_ids, n_samples, ci_level):
        chunk_size = 1000
        meds, los, his = [], [], []
        for start in range(0, len(row_ids), chunk_size):
            end = min(start + chunk_size, len(row_ids))
            chunk = row_ids[start:end]
            rkey = jax.random.fold_in(rng_key_seed, start)
            m, lo, hi = batch_credible_interval(
                rkey,
                best_packed,
                train_jax,
                query_col=query_col,
                row_ids=chunk,
                n_samples=n_samples,
                ci_level=ci_level,
            )
            meds.append(np.asarray(m))
            los.append(np.asarray(lo))
            his.append(np.asarray(hi))
        return np.concatenate(meds), np.concatenate(los), np.concatenate(his)

    def _chunked_classify(query_col, candidate_vals, row_ids):
        chunk_size = 1500
        log_ps = []
        for start in range(0, len(row_ids), chunk_size):
            end = min(start + chunk_size, len(row_ids))
            chunk = row_ids[start:end]
            log_ps.append(
                np.asarray(
                    batch_classify_column(
                        best_packed,
                        train_jax,
                        target_col=query_col,
                        candidate_vals=candidate_vals,
                        row_ids=chunk,
                    )
                )
            )
        return np.concatenate(log_ps, axis=0)

    # ── (8) Imputation + credible-interval calibration ───────────────────
    # For each clinical column, compute 50/80/90/95% credible intervals
    # for ALL observed rows using best chain. Then check empirical coverage
    # of the actual observed value. This is an in-sample diagnostic, but it
    # tells us whether the per-cluster predictive variance is realistic
    # (over-confident -> coverage << nominal).
    imputation_targets = ["LBXGH", "LBXSGL", "BMXBMI", "BPXSY1", "LBXTC", "LBDLDL"]
    imputation_targets = [c for c in imputation_targets if c in name_to_idx]
    rng_key = jax.random.key(11)
    coverage_rows: list[dict] = []
    impute_rows: list[dict] = []
    fig, axes = plt.subplots(
        2,
        max(1, len(imputation_targets) // 2 + len(imputation_targets) % 2),
        figsize=(4.5 * (len(imputation_targets) // 2 + len(imputation_targets) % 2), 7),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ti, col_name in enumerate(imputation_targets):
        col_idx = name_to_idx[col_name]
        observed_mask = ~np.isnan(train_data[:, col_idx])
        n_obs = int(observed_mask.sum())
        if n_obs < 50:
            continue
        obs_rows = jnp.array(np.where(observed_mask)[0])
        truths = train_data[observed_mask, col_idx]

        # 90/50/95% CIs (chunked for low-VRAM GPUs)
        rng_key, sub = jax.random.split(rng_key)
        med, lo90, hi90 = _chunked_credible_interval(
            sub,
            query_col=col_idx,
            row_ids=obs_rows,
            n_samples=200,
            ci_level=0.90,
        )
        rng_key, sub = jax.random.split(rng_key)
        _, lo50, hi50 = _chunked_credible_interval(
            sub,
            query_col=col_idx,
            row_ids=obs_rows,
            n_samples=200,
            ci_level=0.50,
        )
        rng_key, sub = jax.random.split(rng_key)
        _, lo95, hi95 = _chunked_credible_interval(
            sub,
            query_col=col_idx,
            row_ids=obs_rows,
            n_samples=200,
            ci_level=0.95,
        )
        med_np = med
        for level, lo, hi in [(0.50, lo50, hi50), (0.90, lo90, hi90), (0.95, lo95, hi95)]:
            lo_np, hi_np = lo, hi
            cov = float(((truths >= lo_np) & (truths <= hi_np)).mean())
            mean_width = float((hi_np - lo_np).mean())
            coverage_rows.append(
                {
                    "column": col_name,
                    "ci_level": level,
                    "empirical_coverage": cov,
                    "mean_width": mean_width,
                    "n_observed": n_obs,
                }
            )
        mae = float(np.abs(med_np - truths).mean())
        rmse = float(np.sqrt(np.mean((med_np - truths) ** 2)))
        impute_rows.append(
            {"column": col_name, "n_observed": n_obs, "mae_median": mae, "rmse_median": rmse}
        )

        ax = axes_flat[ti]
        # Sub-sample for scatter readability
        sample_idx = np.random.default_rng(42).choice(n_obs, size=min(800, n_obs), replace=False)
        ax.errorbar(
            truths[sample_idx],
            med_np[sample_idx],
            yerr=[
                med_np[sample_idx] - lo90[sample_idx],
                hi90[sample_idx] - med_np[sample_idx],
            ],
            fmt="o",
            markersize=2,
            alpha=0.25,
            ecolor="grey",
            elinewidth=0.4,
            capsize=0,
            color="steelblue",
            label="90% CI",
        )
        lim = (
            min(truths.min(), med_np.min()) * 0.95,
            max(truths.max(), med_np.max()) * 1.05,
        )
        ax.plot(lim, lim, "k--", linewidth=0.8, label="y = x")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"Observed {col_name}")
        ax.set_ylabel("Posterior median (best chain)")
        cov90 = next(
            r["empirical_coverage"]
            for r in coverage_rows
            if r["column"] == col_name and r["ci_level"] == 0.90
        )
        ax.set_title(f"{col_name}\nMAE={mae:.2f}  90%-CI cov={cov90:.0%}", fontsize=9)
        ax.legend(loc="upper left", fontsize=7)

    for ax in axes_flat[len(imputation_targets) :]:
        ax.set_visible(False)
    plt.suptitle(
        "Imputation calibration: posterior median vs observed (with 90% CI bars)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "imputation_calibration.png", dpi=120)
    plt.close()
    pl.DataFrame(coverage_rows).write_csv(out_dir / "ci_coverage.csv")
    pl.DataFrame(impute_rows).write_csv(out_dir / "imputation_metrics.csv")

    # Also impute NATURALLY MISSING values to demonstrate practical use
    naturally_missing_rows: list[dict] = []
    for col_name in imputation_targets:
        col_idx = name_to_idx[col_name]
        miss_mask = np.isnan(train_data[:, col_idx])
        n_miss = int(miss_mask.sum())
        if n_miss == 0:
            continue
        miss_rows = jnp.array(np.where(miss_mask)[0])
        rng_key, sub = jax.random.split(rng_key)
        med_miss_np, _, _ = _chunked_credible_interval(
            sub,
            query_col=col_idx,
            row_ids=miss_rows,
            n_samples=200,
            ci_level=0.90,
        )
        observed_vals = train_data[~miss_mask, col_idx]
        naturally_missing_rows.append(
            {
                "column": col_name,
                "n_naturally_missing": n_miss,
                "imputed_median_mean": float(med_miss_np.mean()),
                "imputed_median_std": float(med_miss_np.std()),
                "observed_mean": float(observed_vals.mean()),
                "observed_std": float(observed_vals.std()),
            }
        )
    if naturally_missing_rows:
        pl.DataFrame(naturally_missing_rows).write_csv(
            out_dir / "imputation_naturally_missing.csv"
        )

    # ── (9) Conditional entropy: variable importance for each binary label ──
    # H(label) - H(label | feature) per single feature, ranked.
    label_targets = [c for c in LABEL_COLS if c in name_to_idx]
    feature_candidates = [
        "LBXGH",
        "LBXSGL",
        "BMXBMI",
        "BMXWAIST",
        "BPXSY1",
        "BPXDI1",
        "RIDAGEYR",
        "LBXTC",
        "LBDHDD",
        "LBDLDL",
        "LBXTR",
        "LBXSCR",
        "LBXSASSI",
        "LBXSATSI",
        "LBXHGB",
        "BPXPLS",
    ]
    feature_candidates = [c for c in feature_candidates if c in name_to_idx]

    info_gain_rows: list[dict] = []
    fig, axes = plt.subplots(
        len(label_targets),
        1,
        figsize=(9, max(2.5, 2.2 * len(label_targets))),
        squeeze=False,
    )
    for li, label_name in enumerate(label_targets):
        label_idx = name_to_idx[label_name]
        target_cols = [label_idx]
        h_marginals: dict[str, float] = {}
        h_conditionals: dict[str, float] = {}
        # H(label | feature) for each feature (1 at a time keeps cost reasonable)
        for feat in feature_candidates:
            feat_idx = name_to_idx[feat]
            if feat_idx == label_idx:
                continue
            rng_key, sub = jax.random.split(rng_key)
            h_cond = float(
                np.asarray(
                    batch_conditional_entropy(
                        sub,
                        chains,
                        train_jax,
                        target_cols=target_cols,
                        given_cols=[feat_idx],
                        n_samples=300,
                    )
                )[0]
            )
            h_conditionals[feat] = h_cond
        # Marginal: condition on empty set is the marginal — approximate with
        # H(label | itself excluded). Use conditional with given=[label] gives 0,
        # so use entropy of label values directly.
        label_vals = train_data[:, label_idx]
        observed = label_vals[~np.isnan(label_vals)].astype(np.int64)
        if observed.size > 0:
            counts = np.bincount(observed)
            probs = counts / counts.sum()
            h_marg = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        else:
            h_marg = float("nan")
        h_marginals[label_name] = h_marg

        # Build ranked bar
        feats_sorted = sorted(h_conditionals.items(), key=lambda kv: kv[1])
        names_plot = [f for f, _ in feats_sorted]
        gains = [max(0.0, h_marg - h) for _, h in feats_sorted]
        for f, h in feats_sorted:
            info_gain_rows.append(
                {
                    "label": label_name,
                    "feature": f,
                    "h_label_marginal_nat": h_marg,
                    "h_label_given_feature_nat": h,
                    "info_gain_nat": max(0.0, h_marg - h),
                }
            )

        ax = axes[li, 0]
        ax.barh(
            np.arange(len(names_plot)), gains, color="darkorange", edgecolor="black", linewidth=0.4
        )
        ax.set_yticks(np.arange(len(names_plot)))
        ax.set_yticklabels(names_plot, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Information gain (nats)")
        ax.set_title(
            f"{label_name}: H(label)={h_marg:.3f} → H(label | feature) reduction",
            fontsize=10,
        )
    plt.suptitle("Conditional entropy: jaxcross variable-importance ranking", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "conditional_entropy.png", dpi=120)
    plt.close()
    pl.DataFrame(info_gain_rows).write_csv(out_dir / "conditional_entropy.csv")

    # ── (10) Classification calibration for binary labels ────────────────
    # batch_classify_column gives log P(label=v | row) for each row.
    # Compare predicted P(label=1) vs empirical fraction in deciles.
    cal_rows: list[dict] = []
    fig, axes = plt.subplots(
        1,
        len(label_targets),
        figsize=(4.0 * max(1, len(label_targets)), 4),
        squeeze=False,
    )
    for li, label_name in enumerate(label_targets):
        label_idx = name_to_idx[label_name]
        observed_mask = ~np.isnan(train_data[:, label_idx])
        if observed_mask.sum() < 50:
            continue
        obs_rows = jnp.array(np.where(observed_mask)[0])
        candidates = jnp.array([0.0, 1.0])
        log_p = _chunked_classify(
            query_col=label_idx,
            candidate_vals=candidates,
            row_ids=obs_rows,
        )
        # Convert to P(label=1) via log-softmax
        log_p1 = log_p[:, 1] - np.logaddexp(log_p[:, 0], log_p[:, 1])
        p1 = np.exp(log_p1)
        truths = train_data[observed_mask, label_idx].astype(np.int64)

        # Brier + log-loss + AUC + accuracy
        brier = float(np.mean((p1 - truths) ** 2))
        ll = float(
            -np.mean(
                truths * np.log(np.clip(p1, 1e-12, 1))
                + (1 - truths) * np.log(np.clip(1 - p1, 1e-12, 1))
            )
        )
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(truths, p1)) if len(set(truths)) > 1 else float("nan")
        except Exception:
            auc = float("nan")

        # Decile calibration curve
        order = np.argsort(p1)
        binned_p = []
        binned_obs = []
        n_bins = 10
        bin_edges = np.linspace(0, len(p1), n_bins + 1, dtype=np.int64)
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
    plt.suptitle("Classification calibration: jaxcross batch_classify_column", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "classification_calibration.png", dpi=120)
    plt.close()
    if cal_rows:
        pl.DataFrame(cal_rows).write_csv(out_dir / "classification_metrics.csv")

    out["coverage"] = coverage_rows
    out["imputation_metrics"] = impute_rows
    out["naturally_missing"] = naturally_missing_rows
    out["info_gain"] = info_gain_rows
    out["classification_metrics"] = cal_rows
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference-dir",
        type=str,
        default=str(DEFAULT_INF_DIR),
        help="Directory containing chain_*.jxc + best_chain.jxc + inference_meta.json",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default=None,
        help="Output subdirectory under results/. Default mirrors inference dir name "
        "with 'inference' -> 'discovery'.",
    )
    args = parser.parse_args()

    inf_dir = Path(args.inference_dir)
    if args.out_subdir is None:
        # Derive: results/inference -> results/discovery;
        # results/inference_warm -> results/discovery_warm
        leaf = (
            inf_dir.name.replace("inference", "discovery", 1)
            if "inference" in inf_dir.name
            else f"discovery_{inf_dir.name}"
        )
        out_dir = inf_dir.parent / leaf
    else:
        out_dir = RESULTS_ROOT / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Inference dir: {inf_dir}")
    print(f"Output dir:    {out_dir}")

    # ── Load preprocessed metadata ───────────────────────────────────────
    info = json.loads((PREP_DIR / "column_info.json").read_text())
    column_types = [_TYPE_MAP[c["type"]] for c in info["columns"]]
    column_names = [c["name"] for c in info["columns"]]
    seqn = np.load(PREP_DIR / "seqn.npy")
    train_data = np.load(PREP_DIR / "train_data.npy")
    n_rows, n_cols = train_data.shape
    name_to_idx = {n: i for i, n in enumerate(column_names)}

    meta = json.loads((inf_dir / "inference_meta.json").read_text())
    n_chains = meta["n_chains"]
    print(f"Loading {n_chains} chains from {inf_dir}")
    chains = [load_packed_state(str(inf_dir / f"chain_{i}.jxc"))[0] for i in range(n_chains)]
    best_packed = load_packed_state(str(inf_dir / "best_chain.jxc"))[0]

    # ── 1. View structure per chain ──────────────────────────────────────
    print(f"\n{'=' * 70}\n1. VIEW STRUCTURE\n{'=' * 70}")
    views_per_chain: list[list[list[str]]] = []
    for ci, packed in enumerate(chains):
        state = unpack_state(packed, column_types)
        views_sorted = sorted(state.views, key=lambda v: -len(v.column_indices))
        chain_views = []
        lj = meta["final_log_joints"][ci]
        print(f"\nChain {ci} ({len(state.views)} views, log_joint={lj:,.1f}):")
        for vi, v in enumerate(views_sorted):
            cols = [column_names[c] for c in v.column_indices]
            chain_views.append(cols)
            n_clusters = len(set(int(x) for x in np.asarray(v.row_assignments)))
            print(
                f"  View {vi}: {len(cols):2d} cols, {n_clusters:3d} clusters → {', '.join(cols)}"
            )
        views_per_chain.append(chain_views)
    (out_dir / "views_per_chain.json").write_text(json.dumps(views_per_chain, indent=2))

    # ── 2. Dependence matrix (Z-matrix, BMA) ─────────────────────────────
    print(f"\n{'=' * 70}\n2. DEPENDENCE MATRIX (Z-matrix, BMA)\n{'=' * 70}")
    z = np.asarray(packed_dependence_matrix(chains))
    print(f"  Shape: {z.shape}, off-diagonal mean={z[np.triu_indices(n_cols, k=1)].mean():.3f}")
    np.save(out_dir / "z_matrix.npy", z)
    pl.DataFrame(
        {"_": column_names, **{column_names[j]: z[:, j] for j in range(n_cols)}}
    ).write_csv(out_dir / "z_matrix.csv")

    pairs = []
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            pairs.append((column_names[i], column_names[j], float(z[i, j])))
    pairs.sort(key=lambda x: -x[2])
    print("  Top 10 dependencies:")
    for a, b, score in pairs[:10]:
        print(f"    {score:.3f}  {a:10s} <-> {b}")
    print("  Bottom 10 dependencies:")
    for a, b, score in pairs[-10:]:
        print(f"    {score:.3f}  {a:10s} <-> {b}")

    # Original (unsorted) Z heatmap
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 10))
        im = ax.imshow(z, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_cols))
        ax.set_xticklabels(column_names, rotation=90, fontsize=7)
        ax.set_yticklabels(column_names, fontsize=7)
        plt.colorbar(im, ax=ax, label="P(same view, BMA)")
        ax.set_title(f"NHANES jaxcross dependency matrix ({n_chains}-chain BMA)")
        plt.tight_layout()
        plt.savefig(out_dir / "z_matrix.png", dpi=120)
        plt.close()
        print(f"  Saved heatmap → {out_dir / 'z_matrix.png'}")
    except ImportError:
        print("  matplotlib unavailable; skipping heatmap PNG")

    # ── 3. Mutual information for curated clinical pairs ─────────────────
    print(f"\n{'=' * 70}\n3. MUTUAL INFORMATION (curated clinical pairs)\n{'=' * 70}")
    print(f"  {'Pair':50s} {'MI':>7s} {'Linfoot':>8s}")
    print(f"  {'-' * 50} {'-' * 7} {'-' * 8}")
    mi_rows: list[dict] = []
    rng_key = jax.random.key(99)
    for col_a, col_b, label in MI_PAIRS:
        if col_a not in name_to_idx or col_b not in name_to_idx:
            continue
        ia = name_to_idx[col_a]
        ib = name_to_idx[col_b]
        rng_key, sub = jax.random.split(rng_key)
        mi_val, _ = packed_mutual_information(
            chains, column_types, col_i=ia, col_j=ib, rng_key=sub
        )
        mi = float(mi_val)
        linfoot = float(np.sqrt(1 - np.exp(-2 * mi))) if mi > 0 else 0.0
        print(f"  {label:50s} {mi:>7.3f} {linfoot:>8.3f}")
        mi_rows.append(
            {"col_a": col_a, "col_b": col_b, "label": label, "mi": mi, "linfoot": linfoot}
        )
    pl.DataFrame(mi_rows).write_csv(out_dir / "mi_table.csv")

    # ── 4. Row typicality (BMA) ──────────────────────────────────────────
    print(f"\n{'=' * 70}\n4. ROW TYPICALITY\n{'=' * 70}")
    all_ids = jnp.arange(n_rows)
    typicality = np.asarray(batch_row_typicality(chains, all_ids))
    print(f"  Range: [{typicality.min():.4f}, {typicality.max():.4f}]")
    print(f"  Mean: {typicality.mean():.4f}, std: {typicality.std():.4f}")
    pl.DataFrame({"seqn": seqn, "typicality": typicality.astype(np.float32)}).write_csv(
        out_dir / "typicality.csv"
    )
    top_typical_idx = np.argsort(-typicality)[:10]
    bot_typical_idx = np.argsort(typicality)[:10]
    print("  Top 5 most typical participants:")
    for i in top_typical_idx[:5]:
        print(f"    SEQN={int(seqn[i])}  typicality={typicality[i]:.4f}")
    print("  Top 5 least typical participants:")
    for i in bot_typical_idx[:5]:
        print(f"    SEQN={int(seqn[i])}  typicality={typicality[i]:.4f}")

    # ── 5. Anomaly score (best chain) ────────────────────────────────────
    print(f"\n{'=' * 70}\n5. ANOMALY SCORE (best chain)\n{'=' * 70}")
    train_jax = jnp.array(train_data)
    anomaly = np.asarray(batch_anomaly_score(best_packed, train_jax, all_ids))
    print(f"  Range: [{anomaly.min():.3f}, {anomaly.max():.3f}]  mean={anomaly.mean():.3f}")
    pl.DataFrame({"seqn": seqn, "anomaly": anomaly.astype(np.float32)}).write_csv(
        out_dir / "anomaly.csv"
    )
    top_anom_idx = np.argsort(anomaly)[:10]
    print("  Top 5 most anomalous participants:")
    for i in top_anom_idx[:5]:
        print(f"    SEQN={int(seqn[i])}  anomaly={anomaly[i]:.3f}  typicality={typicality[i]:.4f}")

    # ── 6. Patient similarity ────────────────────────────────────────────
    print(f"\n{'=' * 70}\n6. PATIENT SIMILARITY (anchor cohort)\n{'=' * 70}")
    rng = np.random.default_rng(99)
    random_anchors = rng.choice(n_rows, size=4, replace=False)
    anchor_idx = np.concatenate([top_typical_idx[:3], top_anom_idx[:3], random_anchors])
    anchor_ids = jnp.array(anchor_idx)
    sim = np.asarray(batch_row_similarity(chains, anchor_ids))
    print(f"  10x10 similarity matrix (anchor seqns: {seqn[anchor_idx].tolist()}):")
    for i, ai in enumerate(anchor_idx):
        row = sim[i]
        print(
            f"    SEQN={int(seqn[ai])}  min={row.min():.3f}  "
            f"mean={row.mean():.3f}  max={row.max():.3f}"
        )
    sim_cols: dict = {"anchor_seqn": seqn[anchor_idx].astype(np.int64)}
    for j, aj in enumerate(anchor_idx):
        sim_cols[f"sim_to_seqn_{int(seqn[aj])}"] = sim[:, j].astype(np.float32)
    pl.DataFrame(sim_cols).write_csv(out_dir / "similarity_anchors.csv")

    print("\n  Top-5 nearest neighbours per anchor (across full cohort):")
    # Chunked: per-anchor similarity to all 9254 rows in 500-row chunks.
    # Avoids the 9264x9264 ~ 5 GB allocation that OOMs a 4 GB GPU.
    chunk_size = 500
    sim_anchor_to_all = np.zeros((len(anchor_idx), n_rows), dtype=np.float32)
    for j, ai in enumerate(anchor_idx):
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            chunk_ids = jnp.concatenate(
                [jnp.array([int(ai)], dtype=jnp.int64), jnp.arange(start, end, dtype=jnp.int64)]
            )
            chunk_sim = np.asarray(batch_row_similarity(chains, chunk_ids))
            # Row 0 is the anchor; columns 1: are the chunk
            sim_anchor_to_all[j, start:end] = chunk_sim[0, 1:]
    nearest_rows = []
    for j, ai in enumerate(anchor_idx):
        row = sim_anchor_to_all[j].copy()
        row[ai] = -np.inf
        nn = np.argsort(-row)[:5]
        nearest_rows.append(
            {
                "anchor_seqn": int(seqn[ai]),
                "nn1_seqn": int(seqn[nn[0]]),
                "nn1_sim": float(row[nn[0]]),
                "nn2_seqn": int(seqn[nn[1]]),
                "nn2_sim": float(row[nn[1]]),
                "nn3_seqn": int(seqn[nn[2]]),
                "nn3_sim": float(row[nn[2]]),
                "nn4_seqn": int(seqn[nn[3]]),
                "nn4_sim": float(row[nn[3]]),
                "nn5_seqn": int(seqn[nn[4]]),
                "nn5_sim": float(row[nn[4]]),
            }
        )
        print(
            f"    Anchor SEQN={int(seqn[ai])}: "
            f"NN={[int(seqn[k]) for k in nn]}  sims={[round(float(row[k]), 3) for k in nn]}"
        )
    pl.DataFrame(nearest_rows).write_csv(out_dir / "nearest_neighbours.csv")

    # ── 7. Publication figures (views, clusters, label-ARI) ──────────────
    print(f"\n{'=' * 70}\n7. PUBLICATION FIGURES (views + clusters + ARI)\n{'=' * 70}")
    try:
        pub_summary = _make_publication_figures(
            chains=chains,
            best_packed=best_packed,
            column_types=column_types,
            column_names=column_names,
            train_data=train_data,
            z=z,
            name_to_idx=name_to_idx,
            out_dir=out_dir,
        )
        n_pubpngs = sum(1 for f in out_dir.iterdir() if f.suffix == ".png")
        print(f"  {n_pubpngs} PNG(s) written under {out_dir}/")
        if pub_summary.get("label_ari"):
            best_ari = max(pub_summary["label_ari"], key=lambda r: r["ari"])
            print(
                f"  Best label-view ARI: {best_ari['label']} ↔ View {best_ari['view_idx']} "
                f"(ARI={best_ari['ari']:.3f}, n={best_ari['n_observed']})"
            )
        print(
            f"  Between-chain view consistency (ARI off-diag mean): "
            f"{pub_summary['view_consistency_offdiag_mean']:.3f}"
        )
    except ImportError:
        print("  matplotlib unavailable; skipping publication figures")
        pub_summary = {}

    # ── 8-10. Imputation / conditional entropy / classification ──────────
    print(f"\n{'=' * 70}\n8-10. IMPUTATION + COND. ENTROPY + CLASSIFICATION\n{'=' * 70}")
    try:
        adv_summary = _make_advanced_figures(
            chains=chains,
            best_packed=best_packed,
            column_names=column_names,
            train_data=train_data,
            name_to_idx=name_to_idx,
            out_dir=out_dir,
        )
        if adv_summary.get("classification_metrics"):
            for r in adv_summary["classification_metrics"]:
                print(
                    f"  Classification {r['label']}: "
                    f"AUC={r['auc']:.3f}  Brier={r['brier']:.3f}  "
                    f"prevalence={r['prevalence']:.3f}"
                )
        if adv_summary.get("coverage"):
            for r in adv_summary["coverage"]:
                if r["ci_level"] == 0.90:
                    print(
                        f"  CI calibration {r['column']}: "
                        f"90%-CI cov={r['empirical_coverage']:.0%}  "
                        f"width={r['mean_width']:.2f}"
                    )
    except ImportError:
        print("  matplotlib/sklearn unavailable; skipping advanced figures")
        adv_summary = {}

    # ── 11. Discovery summary ─────────────────────────────────────────────
    summary = {
        "inference_dir": str(inf_dir),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "nan_fraction": float(np.isnan(train_data).mean()),
        "n_chains": n_chains,
        "final_log_joints": meta["final_log_joints"],
        "best_chain_idx": meta["best_chain_idx"],
        "elapsed_seconds": meta.get("elapsed_seconds"),
        "init_from": meta.get("init_from"),
        "views": {
            f"chain_{ci}": [{"size": len(v), "columns": v} for v in views_per_chain[ci]]
            for ci in range(n_chains)
        },
        "z_matrix": {
            "shape": list(z.shape),
            "off_diag_mean": float(z[np.triu_indices(n_cols, k=1)].mean()),
            "top_dependencies": [{"a": a, "b": b, "score": s} for a, b, s in pairs[:10]],
        },
        "mutual_information": mi_rows,
        "typicality": {
            "min": float(typicality.min()),
            "max": float(typicality.max()),
            "mean": float(typicality.mean()),
            "std": float(typicality.std()),
        },
        "anomaly": {
            "min": float(anomaly.min()),
            "max": float(anomaly.max()),
            "mean": float(anomaly.mean()),
        },
        "publication": pub_summary,
        "advanced": adv_summary,
    }
    (out_dir / "discovery_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 70}\nSAVED to {out_dir}/\n{'=' * 70}")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name:35s} ({f.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
