"""Shared utilities for benchmark result persistence and chart generation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend — must be before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def create_results_dir(benchmark_name: str, base_dir: str = "results") -> Path:
    """Create timestamped results directory.

    Returns path like results/synthetic/2026-03-20_143022/
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = Path(base_dir) / benchmark_name / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_metrics(metrics: dict, results_dir: Path, filename: str = "metrics.json") -> Path:
    """Save metrics dict as JSON. Handles JAX/numpy array conversion."""
    path = results_dir / filename
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_serializable)
    print(f"  Saved metrics to {path}")
    return path


def _json_serializable(obj: Any) -> Any:
    """Convert JAX arrays, numpy arrays, and enums to JSON-safe types."""
    # JAX arrays
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    # Enums
    if hasattr(obj, "value"):
        return obj.value
    # Paths
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def plot_convergence(
    sweep_metrics: list[dict],
    results_dir: Path,
    filename: str = "convergence.png",
    *,
    ari_keys: list[str] | None = None,
    show_log_joint: bool = True,
) -> Path:
    """Convergence plot: ARI curves on left axis, log_joint on right axis.

    Args:
        sweep_metrics: List of per-sweep metric dicts. Must contain 'sweep' key.
        results_dir: Directory to save the plot.
        filename: Output filename.
        ari_keys: Keys to plot on left axis (e.g., ['col_ari', 'row_ari_v0']).
        show_log_joint: Whether to plot log_joint on right axis.

    Returns:
        Path to saved PNG.
    """
    sweeps = [m["sweep"] for m in sweep_metrics]
    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    if ari_keys:
        for i, key in enumerate(ari_keys):
            values = [m.get(key, 0.0) for m in sweep_metrics]
            color = colors[i % len(colors)]
            ax1.plot(sweeps, values, color=color, linewidth=2, label=key)
        ax1.set_ylabel("ARI / Score", fontsize=12)
        ax1.set_ylim(-0.1, 1.1)
        ax1.legend(loc="lower left", fontsize=10)

    if show_log_joint and "log_joint" in sweep_metrics[0]:
        ax2 = ax1.twinx() if ari_keys else ax1
        log_joints = [m["log_joint"] for m in sweep_metrics]
        ax2.plot(
            sweeps, log_joints, color="#F44336", linewidth=2, linestyle="--", label="log_joint"
        )
        ax2.set_ylabel("Log Joint", fontsize=12, color="#F44336")
        ax2.legend(loc="lower right", fontsize=10)

    if not ari_keys and not show_log_joint and "n_views" in sweep_metrics[0]:
        n_views = [m["n_views"] for m in sweep_metrics]
        ax1.plot(sweeps, n_views, color="#2196F3", linewidth=2, label="n_views")
        ax1.set_ylabel("Number of Views", fontsize=12)
        ax1.legend(loc="upper right", fontsize=10)

    ax1.set_xlabel("Sweep", fontsize=12)
    ax1.set_title("CrossCat Convergence", fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    path = results_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved convergence plot to {path}")
    return path


def plot_z_matrix(
    z_matrix: np.ndarray | Any,
    results_dir: Path,
    filename: str = "z_matrix.png",
    *,
    col_labels: list[str] | None = None,
) -> Path:
    """Dependence probability heatmap.

    Args:
        z_matrix: Square matrix of dependence probabilities.
        results_dir: Directory to save the plot.
        filename: Output filename.
        col_labels: Labels for columns/rows.

    Returns:
        Path to saved PNG.
    """
    z = np.array(z_matrix)
    n = z.shape[0]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(z, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, label="Dependence Probability")

    if col_labels:
        ax.set_xticks(range(n))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(col_labels, fontsize=9)

    ax.set_title("Dependence Probability Matrix (Z-matrix)", fontsize=14)
    plt.tight_layout()

    path = results_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Z-matrix to {path}")
    return path
