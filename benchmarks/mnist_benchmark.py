"""MNIST digit clustering benchmark for JAX-CrossCat.

Downloads MNIST, reduces to PCA components, runs CrossCat inference,
and evaluates structure discovery and held-out imputation quality.

Outputs convergence plot, Z-matrix heatmap, digit-cluster contingency
table, and JSON metrics to results/mnist/<timestamp>/.

Reference: Mansinghka et al. (2016) Section 5.2 — MNIST experiments.

Usage:
    uv run python benchmarks/mnist_benchmark.py

Requires benchmark extras:
    uv sync --extra benchmark
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.utils import create_results_dir, plot_convergence, plot_z_matrix, save_metrics
from crosscat.diagnostics import (
    collect_diagnostics,
    evaluate_imputation,
    random_holdout_mask,
)
from crosscat.inference import dependence_matrix
from crosscat.model import initialize
from crosscat.packed.kernels import packed_gibbs_sweep
from crosscat.packed.state import pack_state, unpack_state
from crosscat.types import ColumnType


def fetch_mnist(
    n_samples: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch MNIST and subsample with stratified sampling.

    Args:
        n_samples: Number of samples to keep.
        seed: Random seed for subsampling.

    Returns:
        Tuple of (images, labels) as numpy arrays.
    """
    from sklearn.datasets import fetch_openml

    print("Fetching MNIST dataset (cached after first download)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    images = mnist.data.astype(np.float32)
    labels = mnist.target.astype(np.int32)

    # Stratified subsample: equal number per digit
    rng = np.random.default_rng(seed)
    per_digit = n_samples // 10
    indices = []
    for digit in range(10):
        digit_idx = np.where(labels == digit)[0]
        chosen = rng.choice(digit_idx, size=min(per_digit, len(digit_idx)), replace=False)
        indices.append(chosen)
    indices = np.concatenate(indices)
    rng.shuffle(indices)

    return images[indices], labels[indices]


def pca_reduce(
    data: np.ndarray,
    n_components: int = 20,
) -> np.ndarray:
    """PCA dimensionality reduction.

    Args:
        data: Input array of shape (n_samples, 784).
        n_components: Number of PCA components.

    Returns:
        Reduced array of shape (n_samples, n_components).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print(f"Reducing {data.shape[1]} dims to {n_components} PCA components...")
    scaled = StandardScaler().fit_transform(data)
    reduced = PCA(n_components=n_components).fit_transform(scaled)
    return reduced.astype(np.float32)


def run_benchmark(
    n_samples: int = 2000,
    n_components: int = 20,
    n_sweeps: int = 30,
    n_chains: int = 2,
    holdout_fraction: float = 0.05,
    seed: int = 42,
) -> dict:
    """Run the MNIST digit clustering benchmark.

    Args:
        n_samples: Number of MNIST samples to use.
        n_components: Number of PCA components.
        n_sweeps: Number of Gibbs sweeps per chain.
        n_chains: Number of independent chains.
        holdout_fraction: Fraction of cells to hold out for imputation eval.
        seed: Random seed.

    Returns:
        Dictionary with benchmark metrics.
    """
    rng_key = jax.random.key(seed)
    t_start = time.time()

    # 1. Fetch and reduce data
    images, labels = fetch_mnist(n_samples, seed)
    data_np = pca_reduce(images, n_components)
    data_jax = jnp.array(data_np)
    col_types = [ColumnType.CONTINUOUS] * n_components

    print(f"\nData shape: {data_jax.shape}")
    unique_digits, digit_counts = np.unique(labels, return_counts=True)
    print(f"Digit distribution: {dict(zip(unique_digits, digit_counts, strict=True))}")

    # 2. Create holdout mask
    k_mask, k_init = jax.random.split(rng_key)
    mask = random_holdout_mask(k_mask, n_samples, n_components, holdout_fraction)
    n_held_out = int(mask.sum())
    print(f"Held out {n_held_out} cells ({holdout_fraction * 100:.0f}%)")

    # Mask data for inference (NaN out held-out cells)
    data_masked = jnp.where(mask, jnp.nan, data_jax)

    # 3. Run inference
    init_keys = jax.random.split(k_init, n_chains)
    states = []
    all_chain_metrics: list[list[dict]] = []

    diag_interval = 10  # Unpack for diagnostics every N sweeps

    for chain_idx in range(n_chains):
        print(f"\n--- Chain {chain_idx + 1}/{n_chains} ---")
        k_i, k_sweep = jax.random.split(init_keys[chain_idx])
        state = initialize(k_i, data_masked, col_types)
        packed = pack_state(state)

        chain_metrics: list[dict] = []
        t0 = time.time()
        for sweep in range(n_sweeps):
            k_sweep, subkey = jax.random.split(k_sweep)
            packed = packed_gibbs_sweep(subkey, packed, data_masked, n_sweeps=1)

            # Collect metrics periodically (unpacking is expensive)
            if (sweep + 1) % diag_interval == 0 or sweep == n_sweeps - 1:
                state = unpack_state(packed, col_types, data=data_masked)
                diag = collect_diagnostics(state, data_masked)
                chain_metrics.append({"sweep": sweep + 1, **diag})

                if (sweep + 1) % 10 == 0:
                    print(
                        f"  Sweep {sweep + 1:3d}/{n_sweeps}: "
                        f"n_views={state.n_views}, log_joint={diag['log_joint']:.0f}"
                    )

        state = unpack_state(packed, col_types, data=data_masked)
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s ({elapsed / n_sweeps:.2f}s/sweep)")
        states.append(state)
        all_chain_metrics.append(chain_metrics)

    # 4. Evaluate
    print("\n" + "=" * 60)
    print("MNIST BENCHMARK RESULTS")
    print("=" * 60)

    results: dict = {}
    final_state = states[-1]

    # Structure discovery
    results["n_views"] = final_state.n_views
    n_clusters_per_view = []
    for view in final_state.views:
        n_c = int(jnp.max(view.row_assignments)) + 1
        n_clusters_per_view.append(n_c)
    results["n_clusters_per_view"] = n_clusters_per_view
    print(f"\n  Views discovered: {final_state.n_views}")
    print(f"  Clusters per view: {n_clusters_per_view}")

    # Held-out imputation evaluation
    print("\n  Evaluating held-out imputation...")
    k_eval = jax.random.key(seed + 1)
    imputation = evaluate_imputation(final_state, data_jax, mask, col_types, rng_key=k_eval)
    results["imputation_mae"] = imputation["mae"]
    results["imputation_mean_log_lik"] = imputation["mean_log_lik"]
    results["imputation_n_held_out"] = imputation["n_held_out"]
    print(f"  Imputation MAE: {imputation['mae']:.4f}")
    print(f"  Mean log-lik: {imputation['mean_log_lik']:.4f}")

    # Digit-cluster contingency table
    contingency = _digit_cluster_contingency(labels, final_state)
    results["digit_cluster_contingency"] = contingency.tolist()

    total_elapsed = time.time() - t_start
    results["elapsed_seconds"] = total_elapsed
    print(f"\n  Total time: {total_elapsed:.1f}s")

    # 5. Generate charts and save
    print("\n--- Saving results ---")
    results_dir = create_results_dir("mnist")

    avg_metrics = _average_chain_metrics(all_chain_metrics)
    plot_convergence(avg_metrics, results_dir, show_log_joint=True)

    z_matrix = dependence_matrix(states)
    plot_z_matrix(
        z_matrix,
        results_dir,
        col_labels=[f"PC{i + 1}" for i in range(n_components)],
    )

    _plot_digit_clusters(contingency, results_dir)

    save_metrics(
        {
            **results,
            "config": {
                "n_samples": n_samples,
                "n_components": n_components,
                "n_sweeps": n_sweeps,
                "n_chains": n_chains,
                "holdout_fraction": holdout_fraction,
                "seed": seed,
            },
            "per_sweep": avg_metrics,
        },
        results_dir,
    )

    print(f"\nResults saved to {results_dir}/")
    results["results_dir"] = str(results_dir)
    return results


def _digit_cluster_contingency(
    labels: np.ndarray,
    state,
) -> np.ndarray:
    """Build digit-cluster contingency table for the largest view.

    Returns array of shape (10, n_clusters) with row counts.
    """
    # Use the view with the most columns
    view_sizes = [len(v.column_indices) for v in state.views]
    main_view_idx = int(np.argmax(view_sizes))
    main_view = state.views[main_view_idx]

    row_assign = np.array(main_view.row_assignments)
    n_clusters = int(row_assign.max()) + 1
    contingency = np.zeros((10, n_clusters), dtype=np.int32)

    for digit in range(10):
        digit_mask = labels == digit
        cluster_ids = row_assign[digit_mask]
        for c in range(n_clusters):
            contingency[digit, c] = int(np.sum(cluster_ids == c))

    return contingency


def _plot_digit_clusters(
    contingency: np.ndarray,
    results_dir: Path,
) -> Path:
    """Digit-cluster contingency heatmap.

    Rows = digits 0-9, columns = clusters. Normalized per digit (row).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Normalize per digit
    row_sums = contingency.sum(axis=1, keepdims=True)
    normalized = contingency / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(max(8, contingency.shape[1] * 0.8 + 2), 6))
    im = ax.imshow(normalized, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Fraction of digit in cluster")

    ax.set_xticks(range(contingency.shape[1]))
    ax.set_xticklabels([f"C{i}" for i in range(contingency.shape[1])])
    ax.set_yticks(range(10))
    ax.set_yticklabels([str(d) for d in range(10)])
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Digit", fontsize=12)
    ax.set_title("Digit-Cluster Correspondence (Main View)", fontsize=14)

    # Annotate cells with counts
    for i in range(10):
        for j in range(contingency.shape[1]):
            count = contingency[i, j]
            if count > 0:
                ax.text(
                    j,
                    i,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if normalized[i, j] > 0.5 else "black",
                )

    plt.tight_layout()
    path = results_dir / "digit_clusters.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved digit-cluster plot to {path}")
    return path


def _average_chain_metrics(all_chain_metrics: list[list[dict]]) -> list[dict]:
    """Average per-sweep metrics across chains."""
    n_sweeps = len(all_chain_metrics[0])
    avg = []
    for sweep_idx in range(n_sweeps):
        combined: dict = {"sweep": sweep_idx + 1}
        keys = [k for k in all_chain_metrics[0][sweep_idx] if k != "sweep"]
        for key in keys:
            vals = []
            for chain in all_chain_metrics:
                v = chain[sweep_idx].get(key)
                if isinstance(v, (int, float, list)):
                    vals.append(v)
            if vals and isinstance(vals[0], (int, float)):
                combined[key] = sum(vals) / len(vals)
            elif vals:
                combined[key] = vals[0]
        avg.append(combined)
    return avg


if __name__ == "__main__":
    run_benchmark()
