"""Off-the-shelf classical baselines for the NTAD Texas road-safety cohort.

  (a) NaN-aware Pearson correlation matrix + top-k pair table
  (b) Ward hierarchical clustering on |1 - corr| (column dendrogram)
  (c) PCA(10) + KMeans(8) on column-mean-imputed rows
  (d) Random Forest classification of `is_interstate` (road_system == I).
      Bootstrap AUC + Brier + 10-bin ECE. Matches the literature comparator
      standard road-classification literature.
  (e) XGBoost regression on `speedlim` (kph).
      Bootstrap MAE + R² on segment posted speed limit given segment-
      attribute features.

The supervised RF / XGBoost baselines are head-to-head with jaxcross's
target-free posterior (paper §5.x). Per saved-memory framing rule, jaxcross
is NOT trying to beat their AUC / MAE — calibration is the differentiator.

Outputs (results/baselines/):
    pearson_corr.csv / .npy / .png
    column_dendrogram.png
    pca_kmeans_clusters.npy
    rf_is_interstate_metrics.json     RF AUC / Brier / ECE bootstrap
    xgb_speedlim_metrics.json  XGBoost MAE / R² bootstrap
    baseline_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

PREP_DIR = Path("examples/ntad_roads/results/preprocessed")
OUT_DIR = Path("examples/ntad_roads/results/baselines")
N_BOOTSTRAP = 1000
RNG_SEED = 42


def _nan_pearson(X: np.ndarray) -> np.ndarray:
    """NaN-aware Pearson correlation matrix (per-pair masking)."""
    n_rows, n_cols = X.shape
    R = np.full((n_cols, n_cols), np.nan, dtype=np.float64)
    for i in range(n_cols):
        for j in range(i, n_cols):
            xi = X[:, i]
            xj = X[:, j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() < 3:
                continue
            xi_m = xi[mask] - xi[mask].mean()
            xj_m = xj[mask] - xj[mask].mean()
            denom = np.sqrt((xi_m**2).sum() * (xj_m**2).sum())
            if denom < 1e-12:
                continue
            r = float((xi_m * xj_m).sum() / denom)
            R[i, j] = R[j, i] = r
    return R


def _top_pairs(R: np.ndarray, names: list[str], k: int = 10) -> list[dict]:
    """Top-k absolute-value off-diagonal Pearson pairs."""
    n = R.shape[0]
    rows: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            r = R[i, j]
            if np.isfinite(r):
                rows.append((abs(r), i, j))
    rows.sort(reverse=True)
    return [{"a": names[i], "b": names[j], "corr": float(R[i, j])} for _, i, j in rows[:k]]


def _ece_10bin(probs: np.ndarray, truths: np.ndarray) -> float:
    """10-bin expected calibration error."""
    bin_edges = np.linspace(0.0, 1.0, 11)
    n = probs.shape[0]
    ece = 0.0
    for k in range(10):
        lo, hi = bin_edges[k], bin_edges[k + 1]
        mask = (probs >= lo) & (probs < hi if k < 9 else probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(truths[mask].mean())
        bin_conf = float(probs[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _bootstrap_metric(
    rng: np.random.Generator,
    fn,
    truths: np.ndarray,
    preds: np.ndarray,
    n: int = N_BOOTSTRAP,
) -> tuple[float, float, float]:
    """Returns (point, ci_low_2.5%, ci_hi_97.5%)."""
    point = float(fn(truths, preds))
    n_rows = truths.shape[0]
    samples: list[float] = []
    for _ in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        try:
            samples.append(float(fn(truths[idx], preds[idx])))
        except ValueError:
            continue
    arr = np.asarray(samples)
    return point, float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def _column_mean_impute(X: np.ndarray) -> np.ndarray:
    out = X.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        m = ~np.isnan(col)
        if m.sum() == 0:
            out[:, j] = 0.0
        else:
            out[~m, j] = col[m].mean()
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    info = json.loads((PREP_DIR / "column_info.json").read_text())
    col_names = [c["name"] for c in info["columns"]]
    name_to_idx = {n: i for i, n in enumerate(col_names)}
    X = np.load(PREP_DIR / "train_data.npy")
    n_rows, n_cols = X.shape
    print(f"Matrix: {X.shape}, NaN frac {np.isnan(X).mean():.4f}")

    # ── (a) Pearson ──────────────────────────────────────────────────────
    print("Computing NaN-aware Pearson correlation…")
    R = _nan_pearson(X)
    np.save(OUT_DIR / "pearson_corr.npy", R)
    pl.DataFrame(R, schema=col_names).write_csv(OUT_DIR / "pearson_corr.csv")

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_cols))
    ax.set_xticklabels(col_names, rotation=80, fontsize=6)
    ax.set_yticklabels(col_names, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pearson correlation (NaN-pair masked)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pearson_corr.png", dpi=120)
    plt.close()

    top10 = _top_pairs(R, col_names, k=10)
    print("Top-10 |Pearson|:")
    for p in top10:
        print(f"  {p['a']:30s} ↔ {p['b']:30s}  r = {p['corr']:+.3f}")

    # ── (b) Ward dendrogram ──────────────────────────────────────────────
    R_finite = np.where(np.isnan(R), 0.0, R)
    dist = 1.0 - np.abs(R_finite)
    np.fill_diagonal(dist, 0.0)
    cond = sch.distance.squareform(dist, checks=False)
    Z = sch.linkage(cond, method="ward")
    fig, ax = plt.subplots(figsize=(12, 6))
    sch.dendrogram(Z, labels=col_names, ax=ax, leaf_rotation=80, leaf_font_size=7)
    ax.set_title("Ward hierarchical clustering on |1 − Pearson r|")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "column_dendrogram.png", dpi=120)
    plt.close()

    # ── (c) PCA(10) + KMeans(8) ──────────────────────────────────────────
    X_imp = _column_mean_impute(X)
    pca = PCA(n_components=10, random_state=RNG_SEED).fit(X_imp)
    var_ratio = [float(v) for v in pca.explained_variance_ratio_]
    cum_var = [float(v) for v in np.cumsum(pca.explained_variance_ratio_)]
    Xp = pca.transform(X_imp)
    km = KMeans(n_clusters=8, random_state=RNG_SEED, n_init="auto").fit(Xp)
    np.save(OUT_DIR / "pca_kmeans_clusters.npy", km.labels_)
    cluster_sizes = [int(np.sum(km.labels_ == k)) for k in range(8)]
    print("\nPCA cum-var (PC1–PC10): " + " → ".join(f"{v:.1%}" for v in cum_var))
    print(f"KMeans cluster sizes: {cluster_sizes}")

    # ── (d) Random Forest classifier on is_interstate ───────────────────
    # is_interstate is derived from road_system == 'I'. We exclude
    # road_system from the feature set so the classifier has to predict
    # Interstate-segment status from segment-attribute features (length,
    # lanes, speedlim, class, NHS, etc.) rather than re-derive the rule.
    print("\nFitting Random Forest classifier on is_interstate…")
    EXCLUDE_FROM_FEATURES = {
        "is_interstate",
        "road_system",
    }
    target_idx = name_to_idx["is_interstate"]
    feature_idxs = [name_to_idx[c] for c in col_names if c not in EXCLUDE_FROM_FEATURES]
    y = X[:, target_idx]
    obs = np.isfinite(y)
    X_features = _column_mean_impute(X[obs][:, feature_idxs])
    y_obs = y[obs].astype(int)
    print(
        f"  train pool: {y_obs.shape[0]:,} rows, prevalence {y_obs.mean():.4%}, "
        f"{len(feature_idxs)} non-condition features"
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_features, y_obs, test_size=0.20, stratify=y_obs, random_state=RNG_SEED
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RNG_SEED,
        n_jobs=-1,
    ).fit(X_tr, y_tr)
    rf_probs = rf.predict_proba(X_te)[:, 1]
    rng = np.random.default_rng(RNG_SEED)

    rf_metrics = {
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "test_prevalence": float(y_te.mean()),
        "auc": _bootstrap_metric(rng, roc_auc_score, y_te, rf_probs),
        "brier": _bootstrap_metric(rng, brier_score_loss, y_te, rf_probs),
        "log_loss": _bootstrap_metric(
            rng,
            lambda yt, yp: log_loss(yt, np.clip(yp, 1e-6, 1 - 1e-6), labels=[0, 1]),
            y_te,
            rf_probs,
        ),
        "ece_10bin": _ece_10bin(rf_probs, y_te.astype(float)),
    }
    (OUT_DIR / "rf_is_interstate_metrics.json").write_text(json.dumps(rf_metrics, indent=2))
    print(
        f"  RF: AUC={rf_metrics['auc'][0]:.4f} [{rf_metrics['auc'][1]:.4f}, {rf_metrics['auc'][2]:.4f}]"
        f"  Brier={rf_metrics['brier'][0]:.4f}  ECE={rf_metrics['ece_10bin']:.4f}"
    )

    # ── (e) XGBoost regression on speedlim ──────────────────────────────
    # Predict the segment's posted speed limit from segment-attribute
    # features. Exclude road_system (highly predictive of speed via
    # Interstate / FM coupling) for a meaningful regression challenge.
    if _HAS_XGB:
        print("\nFitting XGBoost regressor on speedlim (kph)…")
        EXCLUDE_FOR_DECK = {
            "speedlim",
            "road_system",
            "is_interstate",
        }
        deck_idx = name_to_idx["speedlim"]
        feature_idxs = [name_to_idx[c] for c in col_names if c not in EXCLUDE_FOR_DECK]
        y_deck = X[:, deck_idx]
        obs = np.isfinite(y_deck)
        X_features = _column_mean_impute(X[obs][:, feature_idxs])
        y_obs = y_deck[obs].astype(np.float32)
        print(
            f"  train pool: {y_obs.shape[0]:,} segments (with observed speedlim), "
            f"{len(feature_idxs)} non-system features"
        )

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_features, y_obs, test_size=0.20, random_state=RNG_SEED
        )
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            random_state=RNG_SEED,
            n_jobs=-1,
            tree_method="hist",
        ).fit(X_tr, y_tr)
        deck_pred = xgb_model.predict(X_te)
        rng2 = np.random.default_rng(RNG_SEED + 1)
        xgb_metrics = {
            "n_train": int(X_tr.shape[0]),
            "n_test": int(X_te.shape[0]),
            "test_mean": float(y_te.mean()),
            "test_std": float(y_te.std()),
            "mae": _bootstrap_metric(rng2, mean_absolute_error, y_te, deck_pred),
            "r2": _bootstrap_metric(rng2, r2_score, y_te, deck_pred),
        }
        (OUT_DIR / "xgb_speedlim_metrics.json").write_text(json.dumps(xgb_metrics, indent=2))
        print(
            f"  XGB: MAE={xgb_metrics['mae'][0]:.4f} [{xgb_metrics['mae'][1]:.4f}, "
            f"{xgb_metrics['mae'][2]:.4f}]  R²={xgb_metrics['r2'][0]:.4f}"
        )
    else:
        print("\nxgboost not installed; skipping (e). pip install xgboost to enable.")
        xgb_metrics = None

    summary: dict = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "nan_fraction": float(np.isnan(X).mean()),
        "pearson_top10": top10,
        "pca_explained_variance_ratio": var_ratio,
        "pca_cumulative_variance": cum_var,
        "kmeans_n_clusters": 8,
        "kmeans_cluster_sizes": cluster_sizes,
        "rf_is_interstate": rf_metrics,
        "xgb_speedlim": xgb_metrics,
    }
    (OUT_DIR / "baseline_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\nDone — artifacts in {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
