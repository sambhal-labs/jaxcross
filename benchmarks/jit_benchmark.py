"""Performance benchmark comparing original vs packed CrossCat kernels.

Usage:
    python benchmarks/jit_benchmark.py

Compares wall-clock time for each kernel on a standard dataset:
200 rows, 10 columns, 5 types.
"""

from __future__ import annotations

import time

import jax

from crosscat.gibbs import (
    gibbs_sweep,
    transition_column_hypers,
    transition_crp_alphas,
    transition_row_assignments,
)
from crosscat.model import initialize, log_joint
from crosscat.packed_state import (
    pack_state,
    packed_gibbs_sweep,
    packed_transition_column_hypers,
    packed_transition_crp_alphas,
    packed_transition_row_assignments,
)
from crosscat.synthetic import generate_crosscat_data
from crosscat.types import ColumnType


def generate_benchmark_data():
    """Generate standard benchmark dataset: 200 rows, 10 cols, all 5 types."""
    key = jax.random.key(42)
    column_types = [
        ColumnType.CONTINUOUS,
        ColumnType.CONTINUOUS,
        ColumnType.CATEGORICAL,
        ColumnType.CATEGORICAL,
        ColumnType.BINARY,
        ColumnType.BINARY,
        ColumnType.ORDINAL,
        ColumnType.CYCLIC,
        ColumnType.CYCLIC,
        ColumnType.CONTINUOUS,
    ]
    return generate_crosscat_data(
        key, 200, column_types, n_views=3, n_clusters=3, cluster_separation=5.0
    )


def time_fn(fn, *args, n_runs=3, **kwargs):
    """Time a function, returning (mean_seconds, result)."""
    # Warmup
    result = fn(*args, **kwargs)
    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        # Block until JAX computation completes
        if hasattr(result, "column_assignments"):
            result.column_assignments.block_until_ready()
        elif hasattr(result, "view_row_assignments"):
            result.view_row_assignments.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times), result


def benchmark_kernel(name, orig_fn, packed_fn, state, packed, data, extra_args=None):
    """Benchmark one kernel: original vs packed."""
    if extra_args is None:
        extra_args = {}

    key = jax.random.key(123)

    # Original
    if name == "crp_alphas":
        t_orig, _ = time_fn(orig_fn, key, state)
    else:
        t_orig, _ = time_fn(orig_fn, key, state, data)

    # Packed
    if name == "crp_alphas":
        t_packed, _ = time_fn(packed_fn, key, packed)
    else:
        t_packed, _ = time_fn(packed_fn, key, packed, data)

    speedup = t_orig / max(t_packed, 1e-9)
    print(f"  {name:25s}  orig: {t_orig:.4f}s  packed: {t_packed:.4f}s  {speedup:.1f}x")
    return t_orig, t_packed


def main():
    print("=" * 70)
    print("JAX-CrossCat Performance Benchmark")
    print("=" * 70)
    print()

    # Generate data
    print("Generating benchmark data (200 rows, 10 cols, 5 types)...")
    result = generate_benchmark_data()
    data = result["data"]
    column_types = result["column_types"]

    # Initialize state
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    state = initialize(k1, data, column_types)
    # Warm up with a few sweeps
    state = gibbs_sweep(k2, state, data, n_sweeps=3)
    packed = pack_state(state)

    print(f"State: {state.n_views} views, {state.n_rows} rows, {state.n_cols} cols")
    print(f"Log joint: {float(log_joint(state, data)):.2f}")
    print()

    # Benchmark individual kernels
    print("Kernel benchmarks (mean of 3 runs):")
    print("-" * 70)

    total_orig = 0.0
    total_packed = 0.0

    t_o, t_p = benchmark_kernel(
        "row_assignments",
        transition_row_assignments,
        packed_transition_row_assignments,
        state, packed, data,
    )
    total_orig += t_o
    total_packed += t_p

    t_o, t_p = benchmark_kernel(
        "column_hypers",
        transition_column_hypers,
        packed_transition_column_hypers,
        state, packed, data,
    )
    total_orig += t_o
    total_packed += t_p

    t_o, t_p = benchmark_kernel(
        "crp_alphas",
        transition_crp_alphas,
        packed_transition_crp_alphas,
        state, packed, data,
    )
    total_orig += t_o
    total_packed += t_p

    print("-" * 70)
    print(f"  {'TOTAL':30s}  original: {total_orig:.4f}s  packed: {total_packed:.4f}s  "
          f"speedup: {total_orig / max(total_packed, 1e-9):.1f}x")
    print()

    # Full sweep benchmark
    print("Full sweep benchmark (3 sweeps, mean of 3 runs):")
    print("-" * 70)
    key = jax.random.key(456)
    t_orig_sweep, _ = time_fn(
        gibbs_sweep, key, state, data, n_sweeps=3,
        kernels=("row_assignments", "column_hypers", "crp_alphas"),
    )
    t_packed_sweep, _ = time_fn(
        packed_gibbs_sweep, key, packed, data, n_sweeps=3,
    )
    speedup = t_orig_sweep / max(t_packed_sweep, 1e-9)
    print(f"  {'full sweep (3 iters)':30s}  original: {t_orig_sweep:.4f}s  "
          f"packed: {t_packed_sweep:.4f}s  speedup: {speedup:.1f}x")

    print()
    print("=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
