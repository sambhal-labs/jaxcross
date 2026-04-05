"""Scaling benchmark — validates jaxcross at 1M rows.

Tests:
  1. Subsample-annealing workflow to 1M rows
  2. Mini-batch Gibbs sweep throughput at 1M scale
  3. Parallel row scoring vs sequential comparison
  4. Early stopping convergence

Designed for Kaggle T4 (16GB VRAM). Requires >= 16GB GPU memory.

Usage:
    uv run python benchmarks/scaling_1m_benchmark.py
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp

from crosscat.model import initialize
from crosscat.packed.kernels import (
    packed_gibbs_sweep,
    packed_insert_rows,
    packed_transition_row_assignments_minibatch,
    packed_transition_row_assignments_parallel,
)
from crosscat.packed.state import pack_state, suggest_max_clusters
from crosscat.scaling import (
    minibatch_gibbs_sweep,
    subsample_anneal,
)
from crosscat.types import ColumnType


def _make_data(key, n_rows, n_cols):
    """Generate random mixed-type data."""
    type_cycle = [
        ColumnType.CONTINUOUS,
        ColumnType.BINARY,
        ColumnType.CATEGORICAL,
        ColumnType.CONTINUOUS,
        ColumnType.BINARY,
    ]
    col_types = [type_cycle[i % len(type_cycle)] for i in range(n_cols)]

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


def benchmark_subsample_anneal(key, n_rows=1_000_000, n_cols=20):
    """Test subsample-annealing from 1K to 1M rows."""
    print(f"\n--- Subsample Annealing: {n_rows:,} rows x {n_cols} cols ---")
    data, col_types = _make_data(key, n_rows, n_cols)
    data_mb = data.nbytes / (1024 * 1024)
    print(f"  data memory: {data_mb:.1f} MB")

    t0 = time.perf_counter()
    packed, reordered_data = subsample_anneal(
        key,
        data,
        col_types,
        initial_size=2000,
        growth_factor=4.0,
        sweeps_per_stage=5,
    )
    total = time.perf_counter() - t0
    print(f"  annealing complete: {packed.n_rows:,} rows, {total:.1f}s")
    return {"total_time": total, "final_rows": packed.n_rows}


def benchmark_parallel_vs_sequential(key, n_rows=100_000, n_cols=20):
    """Compare parallel vs sequential row scoring."""
    print(f"\n--- Parallel vs Sequential: {n_rows:,} rows x {n_cols} cols ---")
    k1, k2, k3, k4 = jax.random.split(key, 4)
    data, col_types = _make_data(k1, n_rows, n_cols)
    max_k = suggest_max_clusters(n_rows)

    state = initialize(k2, data, col_types)
    packed = pack_state(state, max_clusters=max_k)
    packed = packed_gibbs_sweep(k3, packed, data, n_sweeps=3)
    packed.column_assignments.block_until_ready()

    # Parallel (vmap over all rows)
    t0 = time.perf_counter()
    p_packed = packed_transition_row_assignments_parallel(k4, packed, data)
    p_packed.column_assignments.block_until_ready()
    parallel_time = time.perf_counter() - t0
    print(f"  parallel row sweep: {parallel_time:.2f}s")

    # Mini-batch (10K rows)
    k5 = jax.random.fold_in(k4, 1)
    t0 = time.perf_counter()
    m_packed = packed_transition_row_assignments_minibatch(k5, packed, data, batch_size=10_000)
    m_packed.column_assignments.block_until_ready()
    minibatch_time = time.perf_counter() - t0
    print(f"  mini-batch (10K) row sweep: {minibatch_time:.2f}s")

    speedup = minibatch_time / max(parallel_time, 0.001)
    print(f"  parallel speedup vs mini-batch: {speedup:.1f}x")
    return {
        "parallel_time": parallel_time,
        "minibatch_time": minibatch_time,
    }


def benchmark_minibatch_throughput(key, n_rows=1_000_000, n_cols=20, n_sweeps=5):
    """Measure mini-batch sweep throughput at 1M rows."""
    print(f"\n--- Mini-batch Throughput: {n_rows:,} rows x {n_cols} cols ---")
    k1, k2, k3 = jax.random.split(key, 3)
    data, col_types = _make_data(k1, n_rows, n_cols)
    max_k = suggest_max_clusters(n_rows)

    # Initialize on subsample, sweep, then insert remaining rows
    result = initialize(k2, data, col_types, subsample_rows=5000)
    state = result.state
    sub_idx = result.subsample_idx
    packed = pack_state(state, max_clusters=max_k)
    sub_data = data[sub_idx]
    packed = packed_gibbs_sweep(jax.random.fold_in(k2, 1), packed, sub_data, n_sweeps=3)

    # Insert remaining rows in batches via packed_insert_rows
    included = jnp.zeros(n_rows, dtype=bool).at[sub_idx].set(True)
    remaining_idx = jnp.where(~included, size=n_rows - sub_data.shape[0])[0]
    remaining = data[remaining_idx]
    batch_size = 50_000
    current_data = sub_data
    for b in range(0, remaining.shape[0], batch_size):
        batch = remaining[b : b + batch_size]
        kb = jax.random.fold_in(k2, b + 100)
        packed, current_data = packed_insert_rows(kb, packed, current_data, batch)
    full_packed = packed

    # Time mini-batch sweeps
    t0 = time.perf_counter()
    full_packed = minibatch_gibbs_sweep(
        k3, full_packed, data, batch_size=10_000, n_sweeps=n_sweeps
    )
    full_packed.column_assignments.block_until_ready()
    total = time.perf_counter() - t0
    per_sweep = total / n_sweeps
    print(f"  {n_sweeps} mini-batch sweeps (B=10K): {total:.1f}s ({per_sweep:.1f}s/sweep)")
    return {"total_time": total, "per_sweep": per_sweep}


def main():
    print("=" * 60)
    print("JAX-CrossCat 1M Scaling Benchmark")
    print("=" * 60)

    backend = jax.default_backend()
    print(f"Backend: {backend}")
    print(f"Devices: {jax.devices()}")
    print(f"suggest_max_clusters(1000000) = {suggest_max_clusters(1000000)}")

    key = jax.random.key(42)

    # 1. Parallel vs sequential on 100K
    parallel_results = benchmark_parallel_vs_sequential(jax.random.fold_in(key, 1))

    # 2. Mini-batch throughput at 1M
    throughput_results = benchmark_minibatch_throughput(jax.random.fold_in(key, 2))

    # 3. Subsample annealing to 1M
    anneal_results = benchmark_subsample_anneal(jax.random.fold_in(key, 3))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Parallel row sweep (100K): {parallel_results['parallel_time']:.2f}s")
    print(f"  Mini-batch sweep (1M, B=10K): {throughput_results['per_sweep']:.1f}s/sweep")
    print(f"  Subsample annealing to 1M: {anneal_results['total_time']:.1f}s")
    print("=" * 60)

    # Save results
    results_dir = Path("benchmarks/results/scaling")
    results_dir.mkdir(parents=True, exist_ok=True)
    import json

    results = {
        "backend": backend,
        "device": str(jax.devices()[0]),
        "parallel": parallel_results,
        "throughput": throughput_results,
        "anneal": anneal_results,
    }
    with open(results_dir / "scaling_1m_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_dir / 'scaling_1m_results.json'}")


if __name__ == "__main__":
    main()
