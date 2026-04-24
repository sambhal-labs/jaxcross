#!/usr/bin/env python3
"""Non-jaxcross RUL baselines on the same C-MAPSS preprocessed split.

Trains two baselines for apples-to-apples comparison with jaxcross:
  - Ridge regression        (classical linear baseline)
  - Random Forest regressor (strong tabular baseline)

Uses the same training array that jaxcross was fitted on (inference/<fd>/
train_used.npy if present, else preprocessed/<fd>/train_data.npy) and the
same test query rows (last observed cycle per engine). Reports MAE / RMSE / R^2
vs the published RUL ground truth.

Usage:
    uv run python examples/c_mapss/baseline_rul.py [FD001]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

PREP_ROOT = Path("examples/c_mapss/results/preprocessed")
INF_ROOT = Path("examples/c_mapss/results/inference")
OUT_ROOT = Path("examples/c_mapss/results/baselines")


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fd", nargs="?", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"]
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    fd = args.fd

    prep_dir = PREP_ROOT / fd
    test_query = np.load(prep_dir / "test_query.npy")
    truth = np.load(prep_dir / "test_rul_truth.npy")
    info = json.loads((prep_dir / "column_info.json").read_text())
    rul_col = len(info["columns"]) - 1
    assert info["columns"][rul_col]["name"] == "rul"

    # Use the training array jaxcross was fitted on, if available.
    train_used = INF_ROOT / fd / "train_used.npy"
    train = np.load(train_used) if train_used.exists() else np.load(prep_dir / "train_data.npy")

    feat_idx = [i for i in range(train.shape[1]) if i != rul_col]
    X_train = train[:, feat_idx]
    y_train = train[:, rul_col]
    X_test = test_query[:, feat_idx]
    # Test query has no NaN in feature columns; sanity check.
    assert not np.isnan(X_train).any(), "training features contain NaN"
    assert not np.isnan(X_test).any(), "test features contain NaN"

    print(f"{fd}: train {X_train.shape}, test {X_test.shape}")

    # ── Ridge ────────────────────────────────────────────────────
    ridge = Ridge(alpha=1.0, random_state=args.seed)
    ridge.fit(X_train, y_train)
    ridge_pred = np.clip(ridge.predict(X_test), 0.0, info["rul_cap"])
    ridge_m = _metrics(ridge_pred, truth)
    print(
        f"  Ridge:         MAE={ridge_m['mae']:6.2f}  RMSE={ridge_m['rmse']:6.2f}  "
        f"R2={ridge_m['r2']:.3f}"
    )

    # ── RandomForest ─────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=args.seed,
    )
    rf.fit(X_train, y_train)
    rf_pred = np.clip(rf.predict(X_test), 0.0, info["rul_cap"])
    rf_m = _metrics(rf_pred, truth)
    print(
        f"  RandomForest:  MAE={rf_m['mae']:6.2f}  RMSE={rf_m['rmse']:6.2f}  R2={rf_m['r2']:.3f}"
    )

    out_dir = OUT_ROOT / fd
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_metrics.json").write_text(
        json.dumps(
            {"ridge": ridge_m, "random_forest": rf_m, "n_train": int(X_train.shape[0])}, indent=2
        )
    )
    print(f"\nSaved {out_dir / 'baseline_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
