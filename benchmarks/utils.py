"""Shared utilities for benchmark result persistence and chart generation."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # headless backend — must be before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from crosscat import estimate_packed_memory, suggest_max_clusters  # noqa: E402
from crosscat.types import ColumnType  # noqa: E402

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


def detect_platform() -> dict:
    """Detect runtime platform, GPU count, and VRAM.

    Returns:
        Dict with keys: platform, backend, n_gpus, gpu_names, vram_gb.
    """
    if "COLAB_RELEASE_TAG" in os.environ:
        platform = "colab"
    elif Path("/kaggle/working").exists():
        platform = "kaggle"
    else:
        platform = "local"

    backend = str(jax.default_backend())
    try:
        gpu_devices = jax.devices("gpu")
        n_gpus = len(gpu_devices)
        gpu_names = [d.device_kind for d in gpu_devices]
    except RuntimeError:
        n_gpus = 0
        gpu_names = []

    vram_gb = _query_vram_gb()

    return {
        "platform": platform,
        "backend": backend,
        "n_gpus": n_gpus,
        "gpu_names": gpu_names,
        "vram_gb": vram_gb,
    }


def _query_vram_gb() -> float:
    """Query total GPU VRAM in GB via nvidia-smi. Returns 0.0 on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Sum all GPUs; nvidia-smi reports in MiB
            total_mib = sum(float(line.strip()) for line in result.stdout.strip().split("\n"))
            return round(total_mib / 1024, 1)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Auto-configuration
# ---------------------------------------------------------------------------


def auto_config(
    n_rows: int,
    n_cols: int,
    *,
    vram_gb: float | None = None,
) -> dict:
    """Recommend benchmark parameters based on data size and available VRAM.

    Returns:
        Dict with n_chains, n_sweeps, max_clusters, max_views,
        diag_interval, ckpt_interval.
    """
    if vram_gb is None:
        vram_gb = _query_vram_gb()
    if vram_gb <= 0:
        vram_gb = 4.0  # conservative CPU fallback

    max_clusters = suggest_max_clusters(n_rows)
    max_views = min(16, max(4, n_cols // 4))

    # Estimate single-chain memory
    mem = estimate_packed_memory(n_rows, n_cols, max_clusters=max_clusters, max_views=max_views)
    chain_gb = mem["total_bytes"] / (1024**3)

    # Leave 30% VRAM headroom for JIT intermediates
    usable_gb = vram_gb * 0.7
    max_chains = max(1, int(usable_gb / chain_gb)) if chain_gb > 0 else 4

    # Tier-based defaults
    if vram_gb >= 40:  # A100
        n_chains = min(max_chains, 8)
        n_sweeps = 500
    elif vram_gb >= 12:  # T4 / P100
        n_chains = min(max_chains, 4)
        n_sweeps = 200
    else:  # Small local GPU
        n_chains = min(max_chains, 2)
        n_sweeps = 100

    # Scale down sweeps for very large datasets
    if n_rows >= 100_000:
        n_sweeps = min(n_sweeps, 50)
    elif n_rows >= 10_000:
        n_sweeps = min(n_sweeps, 100)

    diag_interval = max(1, n_sweeps // 10)
    ckpt_interval = max(1, n_sweeps // 5)

    return {
        "n_chains": n_chains,
        "n_sweeps": n_sweeps,
        "max_clusters": max_clusters,
        "max_views": max_views,
        "diag_interval": diag_interval,
        "ckpt_interval": ckpt_interval,
        "estimated_chain_gb": round(chain_gb, 3),
    }


# ---------------------------------------------------------------------------
# Notebook setup
# ---------------------------------------------------------------------------

_KNOWN_REPO = "https://github.com/sambhal-labs/jaxcross.git"


def setup_notebook(branch: str = "main") -> None:
    """Install jaxcross in a notebook environment (Colab/Kaggle).

    Skips if crosscat is already importable. Preserves pre-installed
    JAX+CUDA stack by installing with --no-deps.
    """
    try:
        import crosscat  # noqa: F401

        print(f"crosscat {crosscat.__version__} already installed — skipping setup")
        return
    except ImportError:
        pass

    platform = "colab" if "COLAB_RELEASE_TAG" in os.environ else "kaggle"
    workdir = "/content/jaxcross" if platform == "colab" else "/kaggle/working/jaxcross"

    if not Path(workdir).exists():
        subprocess.run(["git", "clone", _KNOWN_REPO, workdir], check=True)

    subprocess.run(
        ["git", "fetch", "origin"],
        cwd=workdir,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", branch],
        cwd=workdir,
        check=True,
    )
    subprocess.run(
        ["git", "pull", "origin", branch],
        cwd=workdir,
        check=True,
    )
    subprocess.run(
        ["pip", "install", "-e", workdir, "--no-deps", "-q"],
        check=True,
    )

    print(f"Installed jaxcross from {workdir} (branch: {branch})")


# ---------------------------------------------------------------------------
# Shared benchmark data generation
# ---------------------------------------------------------------------------

_DEFAULT_TYPE_CYCLE = [
    ColumnType.CONTINUOUS,
    ColumnType.BINARY,
    ColumnType.CATEGORICAL,
    ColumnType.CONTINUOUS,
    ColumnType.BINARY,
]


def make_benchmark_data(
    key: jax.Array,
    n_rows: int,
    n_cols: int,
    col_types: list[ColumnType] | None = None,
) -> tuple[jax.Array, list[ColumnType]]:
    """Generate random mixed-type data for benchmarking.

    Args:
        key: JAX PRNG key.
        n_rows: Number of rows.
        n_cols: Number of columns.
        col_types: Column types. If None, cycles through
            CONTINUOUS, BINARY, CATEGORICAL.

    Returns:
        (data, col_types) tuple.
    """
    if col_types is None:
        col_types = [_DEFAULT_TYPE_CYCLE[i % len(_DEFAULT_TYPE_CYCLE)] for i in range(n_cols)]

    parts = []
    for j in range(n_cols):
        ct = col_types[j]
        kj = jax.random.fold_in(key, j)
        if ct == ColumnType.CONTINUOUS:
            col = jax.random.normal(kj, shape=(n_rows,)) * 3.0
        elif ct == ColumnType.BINARY:
            col = jax.random.bernoulli(kj, 0.5, shape=(n_rows,)).astype(jnp.float32)
        elif ct == ColumnType.CATEGORICAL:
            col = jax.random.randint(kj, shape=(n_rows,), minval=0, maxval=5).astype(jnp.float32)
        else:
            col = jax.random.normal(kj, shape=(n_rows,))
        parts.append(col)

    return jnp.stack(parts, axis=1), col_types


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------


def create_results_dir(benchmark_name: str, base_dir: str = "benchmarks/results") -> Path:
    """Create timestamped results directory.

    Returns path like benchmarks/results/synthetic/2026-03-20_143022/
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


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


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
