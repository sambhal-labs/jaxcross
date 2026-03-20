"""Synthetic recovery benchmark — reproduces the paper's Figure 7 experiment.

Generates data with known structure (2 views, 3 clusters each), runs CrossCat
inference, and measures recovery quality via ARI and dependence-matrix metrics.

Outputs convergence plots, Z-matrix heatmaps, cluster recovery scatters, and
JSON metrics to results/synthetic/<timestamp>/.

Reference: Mansinghka et al. (2016) "CrossCat: A Fully Bayesian Nonparametric
Method for Analyzing Heterogeneous, High Dimensional Data", JMLR 17(138):1-49.

Usage:
    uv run python benchmarks/paper_synthetic_benchmark.py
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.utils import create_results_dir, plot_convergence, plot_z_matrix, save_metrics
from crosscat.diagnostics import (
    adjusted_rand_index,
    collect_diagnostics,
    column_partition_ari,
)
from crosscat.gibbs import gibbs_sweep
from crosscat.inference import dependence_matrix
from crosscat.model import initialize
from crosscat.types import ColumnType


def generate_synthetic_data(
    rng_key: jax.Array,
    n_rows: int = 200,
) -> tuple[jax.Array, list[ColumnType], jax.Array, list[jax.Array]]:
    """Generate synthetic data with known CrossCat structure.

    Structure:
        View 0: columns 0-3 (continuous) — 3 well-separated Gaussian clusters
        View 1: columns 4-7 (continuous) — 3 well-separated Gaussian clusters
        Each view has independent row clustering.

    Returns:
        data: (n_rows, 8) observation matrix
        col_types: column type list
        true_col_assignments: ground truth column-to-view assignments
        true_row_assignments: list of ground truth row assignments per view
    """
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)

    # View 0: 3 clusters with well-separated means
    cluster_assignments_v0 = jax.random.categorical(
        k1, jnp.log(jnp.array([1.0 / 3, 1.0 / 3, 1.0 / 3])), shape=(n_rows,)
    )
    means_v0 = jnp.array([[-3.0, -3.0, -3.0, -3.0], [0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0]])
    noise_v0 = jax.random.normal(k2, shape=(n_rows, 4)) * 0.5
    data_v0 = means_v0[cluster_assignments_v0] + noise_v0

    # View 1: 3 clusters with different means (independent of View 0)
    cluster_assignments_v1 = jax.random.categorical(
        k3, jnp.log(jnp.array([1.0 / 3, 1.0 / 3, 1.0 / 3])), shape=(n_rows,)
    )
    means_v1 = jnp.array([[-4.0, -4.0, -4.0, -4.0], [0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 4.0, 4.0]])
    noise_v1 = jax.random.normal(k4, shape=(n_rows, 4)) * 0.5
    data_v1 = means_v1[cluster_assignments_v1] + noise_v1

    data = jnp.concatenate([data_v0, data_v1], axis=1)
    col_types = [ColumnType.CONTINUOUS] * 8

    true_col_assignments = jnp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32)
    true_row_assignments = [cluster_assignments_v0, cluster_assignments_v1]

    return data, col_types, true_col_assignments, true_row_assignments


def run_benchmark(
    n_rows: int = 200,
    n_sweeps: int = 30,
    n_chains: int = 4,
    seed: int = 42,
) -> dict:
    """Run the synthetic recovery benchmark.

    Args:
        n_rows: Number of data rows.
        n_sweeps: Number of Gibbs sweeps per chain.
        n_chains: Number of independent chains.
        seed: Random seed.

    Returns:
        Dictionary with recovery metrics.
    """
    rng_key = jax.random.key(seed)
    k_data, k_init = jax.random.split(rng_key)

    print(f"Generating synthetic data ({n_rows} rows, 8 columns, 2 views, 3 clusters each)...")
    data, col_types, true_col_assign, true_row_assign = generate_synthetic_data(k_data, n_rows)

    # Initialize and run chains, collecting per-sweep metrics
    init_keys = jax.random.split(k_init, n_chains)
    states = []
    all_chain_metrics: list[list[dict]] = []

    for chain_idx in range(n_chains):
        print(f"\n--- Chain {chain_idx + 1}/{n_chains} ---")
        k_i, k_sweep = jax.random.split(init_keys[chain_idx])
        state = initialize(k_i, data, col_types)

        chain_metrics: list[dict] = []
        t0 = time.time()
        for sweep in range(n_sweeps):
            k_sweep, subkey = jax.random.split(k_sweep)
            state = gibbs_sweep(subkey, state, data, n_sweeps=1)

            # Collect metrics every sweep
            diag = collect_diagnostics(state, data)
            col_ari = float(column_partition_ari(state, true_col_assign))
            row_ari_v0, row_ari_v1 = _best_view_match(state, true_row_assign)
            chain_metrics.append(
                {
                    "sweep": sweep + 1,
                    "col_ari": col_ari,
                    "row_ari_v0": row_ari_v0,
                    "row_ari_v1": row_ari_v1,
                    **diag,
                }
            )

            if (sweep + 1) % 10 == 0:
                print(
                    f"  Sweep {sweep + 1:3d}/{n_sweeps}: "
                    f"n_views={state.n_views}, col_ARI={col_ari:.3f}"
                )

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s ({elapsed / n_sweeps:.2f}s/sweep)")
        states.append(state)
        all_chain_metrics.append(chain_metrics)

    # Evaluate recovery
    print("\n" + "=" * 60)
    print("RECOVERY METRICS")
    print("=" * 60)

    results: dict = {}

    # 1. Column partition ARI
    col_aris = [float(column_partition_ari(s, true_col_assign)) for s in states]
    mean_col_ari = sum(col_aris) / len(col_aris)
    results["col_ari_mean"] = mean_col_ari
    results["col_ari_per_chain"] = col_aris

    # 2. Row clustering ARI per view (best-matching view)
    row_aris_v0 = []
    row_aris_v1 = []
    for s in states:
        best_v0, best_v1 = _best_view_match(s, true_row_assign)
        row_aris_v0.append(best_v0)
        row_aris_v1.append(best_v1)

    mean_row_ari_v0 = sum(row_aris_v0) / len(row_aris_v0)
    mean_row_ari_v1 = sum(row_aris_v1) / len(row_aris_v1)
    results["row_ari_v0_mean"] = mean_row_ari_v0
    results["row_ari_v1_mean"] = mean_row_ari_v1

    # 3. Dependence matrix (Z-matrix)
    z_matrix = dependence_matrix(states)
    within_view_prob = float(
        (z_matrix[:4, :4].sum() + z_matrix[4:, 4:].sum() - 8.0) / (2 * (4 * 3))
    )
    between_view_prob = float(z_matrix[:4, 4:].mean())
    results["within_view_dep_prob"] = within_view_prob
    results["between_view_dep_prob"] = between_view_prob

    # Print results
    PASS = "PASS"
    FAIL = "FAIL"

    def check(name: str, value: float, threshold: float, higher_is_better: bool = True) -> str:
        passed = value >= threshold if higher_is_better else value <= threshold
        status = PASS if passed else FAIL
        direction = ">=" if higher_is_better else "<="
        print(f"  [{status}] {name}: {value:.3f} (threshold {direction} {threshold:.2f})")
        return status

    print()
    s1 = check("Column partition ARI (mean)", mean_col_ari, 0.80)
    s2 = check("Row clustering ARI view 0 (mean)", mean_row_ari_v0, 0.70)
    s3 = check("Row clustering ARI view 1 (mean)", mean_row_ari_v1, 0.70)
    s4 = check("Within-view dependence prob", within_view_prob, 0.80)
    s5 = check("Between-view dependence prob", between_view_prob, 0.20, higher_is_better=False)

    all_passed = all(s == PASS for s in [s1, s2, s3, s4, s5])
    results["all_passed"] = all_passed

    print()
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — review results above")

    # Generate charts and save results
    print("\n--- Saving results ---")
    results_dir = create_results_dir("synthetic")

    avg_metrics = _average_chain_metrics(all_chain_metrics)

    plot_convergence(
        avg_metrics,
        results_dir,
        ari_keys=["col_ari", "row_ari_v0", "row_ari_v1"],
    )
    plot_z_matrix(
        z_matrix,
        results_dir,
        col_labels=[f"col_{i}" for i in range(8)],
    )
    _plot_cluster_recovery(data, states[-1], true_row_assign, results_dir)

    save_metrics(
        {
            **results,
            "config": {
                "n_rows": n_rows,
                "n_sweeps": n_sweeps,
                "n_chains": n_chains,
                "seed": seed,
            },
            "per_sweep": avg_metrics,
        },
        results_dir,
    )

    print(f"\nResults saved to {results_dir}/")
    results["results_dir"] = str(results_dir)
    return results


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
                combined[key] = vals[0]  # keep first chain's list values
        avg.append(combined)
    return avg


def _best_view_match(
    state,
    true_row_assignments: list[jax.Array],
) -> tuple[float, float]:
    """Find best ARI match between inferred and true views."""
    best_aris = []
    for true_assign in true_row_assignments:
        best_ari = -1.0
        for view in state.views:
            ari = float(adjusted_rand_index(true_assign, view.row_assignments))
            if ari > best_ari:
                best_ari = ari
        best_aris.append(best_ari)
    return best_aris[0], best_aris[1]


def _plot_cluster_recovery(
    data: jax.Array,
    state,
    true_row_assignments: list[jax.Array],
    results_dir: Path,
) -> Path:
    """2x2 scatter: (true vs inferred) x (view 0 vs view 1).

    Uses first 2 columns of each view for x/y axes.
    Top row: colored by true cluster. Bottom row: colored by inferred cluster.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_np = np.array(data)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors_map = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    # Find best-matching views
    best_view_idx = []
    for true_assign in true_row_assignments:
        best_ari = -1.0
        best_idx = 0
        for v_idx, view in enumerate(state.views):
            ari = float(adjusted_rand_index(true_assign, view.row_assignments))
            if ari > best_ari:
                best_ari = ari
                best_idx = v_idx
        best_view_idx.append(best_idx)

    view_col_pairs = [(0, 1), (4, 5)]  # columns to plot for each view
    view_labels = ["View 0 (cols 0-3)", "View 1 (cols 4-7)"]

    for v, (cx, cy) in enumerate(view_col_pairs):
        true_assign = np.array(true_row_assignments[v])
        inferred_assign = np.array(state.views[best_view_idx[v]].row_assignments)

        for cluster_id in range(int(true_assign.max()) + 1):
            mask = true_assign == cluster_id
            color = colors_map[cluster_id % len(colors_map)]
            axes[0, v].scatter(
                data_np[mask, cx],
                data_np[mask, cy],
                c=color,
                s=15,
                alpha=0.6,
                label=f"Cluster {cluster_id}",
            )

        for cluster_id in range(int(inferred_assign.max()) + 1):
            mask = inferred_assign == cluster_id
            color = colors_map[cluster_id % len(colors_map)]
            axes[1, v].scatter(
                data_np[mask, cx],
                data_np[mask, cy],
                c=color,
                s=15,
                alpha=0.6,
                label=f"Cluster {cluster_id}",
            )

        axes[0, v].set_title(f"True — {view_labels[v]}", fontsize=12)
        axes[1, v].set_title(f"Inferred — {view_labels[v]}", fontsize=12)
        for row in range(2):
            axes[row, v].set_xlabel(f"Column {cx}")
            axes[row, v].set_ylabel(f"Column {cy}")
            axes[row, v].legend(fontsize=8, markerscale=2)

    fig.suptitle("Cluster Recovery: True vs Inferred", fontsize=14, y=1.02)
    plt.tight_layout()

    path = results_dir / "cluster_recovery.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved cluster recovery plot to {path}")
    return path


if __name__ == "__main__":
    run_benchmark()
