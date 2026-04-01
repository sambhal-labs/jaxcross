"""Scalability benchmark — measures runtime vs data dimensions.

Generates scalability curves for the paper:
  1. Runtime vs N_rows (fixed N_cols=10)
  2. Runtime vs N_cols (fixed N_rows=200)
  3. Runtime vs N_sweeps (fixed 200x10)
  4. JIT compilation time vs problem size

All measurements use the packed Gibbs sweep on GPU.

Usage:
    uv run python benchmarks/scalability_benchmark.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from crosscat.model import initialize
from crosscat.packed.kernels import packed_gibbs_sweep
from crosscat.packed.state import pack_state
from crosscat.types import ColumnType


def _make_data(key, n_rows, n_cols):
    """Generate random mixed-type data for benchmarking."""
    # Cycle through types
    type_cycle = [
        ColumnType.CONTINUOUS,
        ColumnType.BINARY,
        ColumnType.CATEGORICAL,
        ColumnType.CONTINUOUS,
        ColumnType.BINARY,
    ]
    col_types = [type_cycle[i % len(type_cycle)] for i in range(n_cols)]

    k1, k2, k3 = jax.random.split(key, 3)
    data_parts = []
    for j in range(n_cols):
        ct = col_types[j]
        kj = jax.random.fold_in(k1, j)
        if ct == ColumnType.CONTINUOUS:
            col = jax.random.normal(kj, shape=(n_rows,)) * 3.0
        elif ct == ColumnType.BINARY:
            col = jax.random.bernoulli(kj, 0.5, shape=(n_rows,)).astype(jnp.float32)
        elif ct == ColumnType.CATEGORICAL:
            col = jax.random.randint(kj, shape=(n_rows,), minval=0, maxval=5).astype(jnp.float32)
        else:
            col = jax.random.normal(kj, shape=(n_rows,))
        data_parts.append(col)

    data = jnp.stack(data_parts, axis=1)
    return data, col_types


def time_sweep(key, data, col_types, n_sweeps=5, n_warmup=1):
    """Time packed_gibbs_sweep, returning (compile_time, sweep_time_per_iter).

    First call includes JIT compilation. Subsequent calls measure steady-state.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    state = initialize(k1, data, col_types)
    packed = pack_state(state)

    # Warmup / compile
    t0 = time.perf_counter()
    packed_w = packed_gibbs_sweep(k2, packed, data, n_sweeps=n_warmup)
    packed_w.column_assignments.block_until_ready()
    compile_time = time.perf_counter() - t0

    # Timed runs
    t0 = time.perf_counter()
    packed_out = packed_gibbs_sweep(k3, packed_w, data, n_sweeps=n_sweeps)
    packed_out.column_assignments.block_until_ready()
    total_time = time.perf_counter() - t0
    per_sweep = total_time / n_sweeps

    return compile_time, per_sweep


def benchmark_rows(base_key):
    """Scalability vs number of rows."""
    row_counts = [50, 100, 200, 500, 1000, 2000]
    n_cols = 10
    results = []

    print("\n=== Scalability vs N_rows (N_cols=10) ===")
    for n_rows in row_counts:
        key = jax.random.fold_in(base_key, n_rows)
        data, col_types = _make_data(key, n_rows, n_cols)
        k_time = jax.random.fold_in(key, 999)
        compile_t, sweep_t = time_sweep(k_time, data, col_types, n_sweeps=5)
        print(f"  N_rows={n_rows:5d}: compile={compile_t:.2f}s, sweep={sweep_t:.4f}s")
        results.append(
            {
                "n_rows": n_rows,
                "n_cols": n_cols,
                "compile_time": compile_t,
                "per_sweep_time": sweep_t,
            }
        )
    return results


def benchmark_cols(base_key):
    """Scalability vs number of columns."""
    col_counts = [5, 10, 20, 50, 100, 200]
    n_rows = 200
    results = []

    print("\n=== Scalability vs N_cols (N_rows=200) ===")
    for n_cols in col_counts:
        key = jax.random.fold_in(base_key, n_cols + 10000)
        data, col_types = _make_data(key, n_rows, n_cols)
        k_time = jax.random.fold_in(key, 999)
        compile_t, sweep_t = time_sweep(k_time, data, col_types, n_sweeps=5)
        print(f"  N_cols={n_cols:5d}: compile={compile_t:.2f}s, sweep={sweep_t:.4f}s")
        results.append(
            {
                "n_rows": n_rows,
                "n_cols": n_cols,
                "compile_time": compile_t,
                "per_sweep_time": sweep_t,
            }
        )
    return results


def benchmark_sweeps(base_key):
    """Measure per-sweep amortization as sweep count increases."""
    sweep_counts = [1, 5, 10, 50, 100]
    n_rows, n_cols = 200, 10
    results = []

    print("\n=== Per-sweep time vs N_sweeps (200x10) ===")
    key = jax.random.fold_in(base_key, 77777)
    data, col_types = _make_data(key, n_rows, n_cols)
    k_init = jax.random.fold_in(key, 0)
    state = initialize(k_init, data, col_types)
    packed = pack_state(state)

    # Pre-compile
    k_warmup = jax.random.fold_in(key, 1)
    packed = packed_gibbs_sweep(k_warmup, packed, data, n_sweeps=1)
    packed.column_assignments.block_until_ready()

    for n_sweeps in sweep_counts:
        k_run = jax.random.fold_in(key, n_sweeps)
        t0 = time.perf_counter()
        out = packed_gibbs_sweep(k_run, packed, data, n_sweeps=n_sweeps)
        out.column_assignments.block_until_ready()
        total = time.perf_counter() - t0
        per_sweep = total / n_sweeps
        print(f"  N_sweeps={n_sweeps:5d}: total={total:.3f}s, per_sweep={per_sweep:.4f}s")
        results.append(
            {
                "n_sweeps": n_sweeps,
                "total_time": total,
                "per_sweep_time": per_sweep,
            }
        )
    return results


def save_results(all_results, results_dir):
    """Save all benchmark results to JSON."""

    def convert(obj):
        if isinstance(obj, (np.integer, jnp.integer)):
            return int(obj)
        if isinstance(obj, (np.floating, jnp.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    path = results_dir / "scalability_results.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {path}")


def plot_scalability(all_results, results_dir):
    """Generate scalability plots for the paper."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Plot 1: Runtime vs N_rows
    ax = axes[0]
    rows_data = all_results["vs_rows"]
    xs = [r["n_rows"] for r in rows_data]
    ys = [r["per_sweep_time"] for r in rows_data]
    ax.plot(xs, ys, "o-", color="#2196F3", linewidth=2, markersize=6)
    ax.set_xlabel("Number of rows", fontsize=11)
    ax.set_ylabel("Time per sweep (s)", fontsize=11)
    ax.set_title("(a) Scaling with rows", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Plot 2: Runtime vs N_cols
    ax = axes[1]
    cols_data = all_results["vs_cols"]
    xs = [r["n_cols"] for r in cols_data]
    ys = [r["per_sweep_time"] for r in cols_data]
    ax.plot(xs, ys, "s-", color="#4CAF50", linewidth=2, markersize=6)
    ax.set_xlabel("Number of columns", fontsize=11)
    ax.set_ylabel("Time per sweep (s)", fontsize=11)
    ax.set_title("(b) Scaling with columns", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Plot 3: Compile time vs problem size
    ax = axes[2]
    # Combine rows and cols data
    sizes_r = [r["n_rows"] * r["n_cols"] for r in rows_data]
    compile_r = [r["compile_time"] for r in rows_data]
    sizes_c = [r["n_rows"] * r["n_cols"] for r in cols_data]
    compile_c = [r["compile_time"] for r in cols_data]
    ax.scatter(sizes_r, compile_r, marker="o", color="#2196F3", label="Vary rows", s=40)
    ax.scatter(sizes_c, compile_c, marker="s", color="#4CAF50", label="Vary cols", s=40)
    ax.set_xlabel("Table size (rows x cols)", fontsize=11)
    ax.set_ylabel("JIT compile time (s)", fontsize=11)
    ax.set_title("(c) Compilation overhead", fontsize=12)
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = results_dir / "scalability.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Scalability plot saved to {path}")

    # Also save individual high-res versions
    for name, data_key, xlabel, marker, color in [
        ("scalability_rows", "vs_rows", "Number of rows", "o-", "#2196F3"),
        ("scalability_cols", "vs_cols", "Number of columns", "s-", "#4CAF50"),
    ]:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        d = all_results[data_key]
        x_key = "n_rows" if "rows" in name else "n_cols"
        xs = [r[x_key] for r in d]
        ys = [r["per_sweep_time"] for r in d]
        ax2.plot(xs, ys, marker, color=color, linewidth=2, markersize=8)
        ax2.set_xlabel(xlabel, fontsize=12)
        ax2.set_ylabel("Time per Gibbs sweep (s)", fontsize=12)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(results_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)


def main():
    print("=" * 60)
    print("JAX-CrossCat Scalability Benchmark")
    print("=" * 60)

    backend = jax.default_backend()
    print(f"Backend: {backend}")
    print(f"Devices: {jax.devices()}")

    base_key = jax.random.key(42)

    results = {}
    results["backend"] = backend
    results["device"] = str(jax.devices()[0])

    results["vs_rows"] = benchmark_rows(jax.random.fold_in(base_key, 1))
    results["vs_cols"] = benchmark_cols(jax.random.fold_in(base_key, 2))
    results["vs_sweeps"] = benchmark_sweeps(jax.random.fold_in(base_key, 3))

    # Save
    results_dir = Path("benchmarks/results/scalability")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, results_dir)
    plot_scalability(results, results_dir)

    print("\n" + "=" * 60)
    print("Scalability benchmark complete.")


if __name__ == "__main__":
    main()
