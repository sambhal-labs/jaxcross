#!/usr/bin/env python3
"""Classical-baseline comparators for the NHANES jaxcross discovery demo.

Runs three orthogonal off-the-shelf approaches on the same 29-column
preprocessed matrix and saves their outputs side-by-side with jaxcross
results so the structure-discovery story is honestly framed:

  - **Hierarchical clustering** (Ward linkage on observed-pair correlations)
    — comparator for the column-views story. Produces a column-dendrogram
    that we contrast with jaxcross's view assignment.
  - **Pearson correlation matrix** — comparator for the Z-matrix.
    Captures linear pairwise dependence only (no mixed-type semantics,
    no NaN-aware joint mixing). Useful contrast.
  - **PCA + KMeans** — comparator for row clustering. PCA imputes via
    column-mean to handle NaN, then KMeans on the top components.
    A weak baseline that ignores variable types entirely.

Note: UMAP intentionally NOT included — it requires the `umap-learn` extra
which we don't want to add for one comparator. PCA + KMeans is a fairer
baseline since it doesn't need extra dependencies.

Outputs (results/baselines/):
    pearson_corr.npy         (29, 29) NaN-aware Pearson correlation
    pearson_corr.csv         same with column headers
    pearson_corr.png         heatmap
    column_dendrogram.png    Ward hierarchical clustering of columns
    pca_kmeans_clusters.npy  (n_rows,) KMeans cluster id from PCA-10 features
    baseline_summary.json    aggregate metrics + cluster sizes

Usage:
    uv run python examples/nhanes_clinical/baseline_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

PREP_DIR = Path("examples/nhanes_clinical/results/preprocessed")
OUT_DIR = Path("examples/nhanes_clinical/results/baselines")


def main() -> int:
    info = json.loads((PREP_DIR / "column_info.json").read_text())
    column_names = [c["name"] for c in info["columns"]]
    train = np.load(PREP_DIR / "train_data.npy")
    n_rows, n_cols = train.shape
    print(f"Loaded {n_rows:,} x {n_cols} matrix, NaN fraction = {np.isnan(train).mean():.1%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pearson correlation matrix (NaN-aware via pairwise complete obs) ─
    print("\n=== Pearson correlation matrix (NaN-aware) ===")
    corr = np.full((n_cols, n_cols), np.nan, dtype=np.float32)
    for i in range(n_cols):
        for j in range(i, n_cols):
            ci = train[:, i]
            cj = train[:, j]
            mask = ~(np.isnan(ci) | np.isnan(cj))
            if mask.sum() < 30:
                corr[i, j] = corr[j, i] = np.nan
                continue
            c = float(np.corrcoef(ci[mask], cj[mask])[0, 1])
            corr[i, j] = corr[j, i] = c
    np.save(OUT_DIR / "pearson_corr.npy", corr)
    pl.DataFrame(
        {"_": column_names, **{column_names[j]: corr[:, j] for j in range(n_cols)}}
    ).write_csv(OUT_DIR / "pearson_corr.csv")

    # Top |Pearson| pairs
    pairs = []
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            if np.isfinite(corr[i, j]):
                pairs.append((column_names[i], column_names[j], float(corr[i, j])))
    pairs.sort(key=lambda x: -abs(x[2]))
    print("Top-10 |Pearson| pairs:")
    for a, b, v in pairs[:10]:
        print(f"  {v:+.3f}  {a:10s} <-> {b}")

    # ── Hierarchical clustering of columns (Ward on |1 - corr|) ──────────
    print("\n=== Hierarchical clustering of columns ===")
    # Distance matrix from |corr|; replace NaNs with 0 (no relationship signal)
    dist_full = 1.0 - np.nan_to_num(np.abs(corr), nan=0.0)
    np.fill_diagonal(dist_full, 0.0)
    # Convert to condensed form for scipy.linkage
    iu = np.triu_indices(n_cols, k=1)
    condensed = dist_full[iu]
    Z = linkage(condensed, method="average")  # 'ward' requires Euclidean inputs
    print(f"Linkage matrix shape: {Z.shape}")

    # ── PCA + KMeans on imputed matrix ───────────────────────────────────
    print("\n=== PCA(10) + KMeans(8) on column-mean-imputed matrix ===")
    imputer = SimpleImputer(strategy="mean")
    train_imputed = imputer.fit_transform(train)
    n_components = min(10, n_cols)
    pca = PCA(n_components=n_components, random_state=42)
    embedded = pca.fit_transform(train_imputed)
    print(
        f"PCA explained variance ratio (top {n_components}): {pca.explained_variance_ratio_.round(3).tolist()}"
    )
    print(f"Cumulative: {pca.explained_variance_ratio_.cumsum().round(3).tolist()}")

    km = KMeans(n_clusters=8, n_init=10, random_state=42)
    cluster_ids = km.fit_predict(embedded)
    np.save(OUT_DIR / "pca_kmeans_clusters.npy", cluster_ids.astype(np.int32))
    sizes = np.bincount(cluster_ids)
    print(f"PCA-KMeans cluster sizes: {sizes.tolist()}")

    # ── Visualizations ───────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        # Pearson heatmap
        fig, ax = plt.subplots(figsize=(11, 10))
        plot_corr = np.nan_to_num(corr, nan=0.0)
        im = ax.imshow(plot_corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_cols))
        ax.set_xticklabels(column_names, rotation=90, fontsize=7)
        ax.set_yticklabels(column_names, fontsize=7)
        plt.colorbar(im, ax=ax, label="Pearson correlation")
        ax.set_title("NHANES Pearson correlation (baseline comparator)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "pearson_corr.png", dpi=120)
        plt.close()

        # Column dendrogram
        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(Z, labels=column_names, leaf_rotation=90, leaf_font_size=8)
        ax.set_title("NHANES column hierarchical clustering (Ward on |1-corr|)")
        ax.set_ylabel("Distance")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "column_dendrogram.png", dpi=120)
        plt.close()
        print(f"Saved heatmaps + dendrogram → {OUT_DIR}")
    except ImportError:
        print("matplotlib unavailable; skipping plots")

    # ── Summary ──────────────────────────────────────────────────────────
    summary = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "nan_fraction": float(np.isnan(train).mean()),
        "pearson_top10": [{"a": a, "b": b, "corr": v} for a, b, v in pairs[:10]],
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_cumulative_variance": pca.explained_variance_ratio_.cumsum().tolist(),
        "kmeans_n_clusters": 8,
        "kmeans_cluster_sizes": sizes.tolist(),
    }
    (OUT_DIR / "baseline_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nSaved {OUT_DIR}/")
    for f in sorted(OUT_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name:30s} ({f.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
