"""Scaling benchmark — validates jaxcross at 10K+ rows.

Tests:
  1. Full Gibbs sweep on 10K rows x 20 cols (mixed types)
  2. Subsample initialization + streaming insertion workflow
  3. packed_insert_rows batch insertion throughput
  4. Memory footprint estimation

Designed for Kaggle 2xT4 (32GB VRAM) but also runnable on smaller GPUs.

Usage:
    uv run python benchmarks/scaling_10k_benchmark.py
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp

from crosscat.model import initialize
from crosscat.packed.kernels import packed_gibbs_sweep, packed_insert_rows
from crosscat.packed.state import pack_state, suggest_max_clusters
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


def _mem_mb(arr):
    """Estimate memory in MB for a JAX array."""
    return arr.nbytes / (1024 * 1024)


def benchmark_full_sweep(key, n_rows=10_000, n_cols=20, n_sweeps=5):
    """Benchmark full Gibbs sweep on 10K rows."""
    print(f"\n--- Full Gibbs Sweep: {n_rows} rows x {n_cols} cols ---")
    k1, k2, k3 = jax.random.split(key, 3)
    data, col_types = _make_data(k1, n_rows, n_cols)
    max_k = suggest_max_clusters(n_rows)
    print(f"  suggested max_clusters: {max_k}")
    print(f"  data memory: {_mem_mb(data):.1f} MB")

    t0 = time.perf_counter()
    state = initialize(k2, data, col_types)
    packed = pack_state(state, max_clusters=max_k)
    init_time = time.perf_counter() - t0
    print(f"  init + pack: {init_time:.2f}s")

    # JIT compile (first sweep)
    t0 = time.perf_counter()
    packed = packed_gibbs_sweep(k3, packed, data, n_sweeps=1)
    packed.column_assignments.block_until_ready()
    compile_time = time.perf_counter() - t0
    print(f"  JIT compile (1st sweep): {compile_time:.2f}s")

    # Timed sweeps
    k4 = jax.random.fold_in(k3, 1)
    t0 = time.perf_counter()
    packed = packed_gibbs_sweep(k4, packed, data, n_sweeps=n_sweeps)
    packed.column_assignments.block_until_ready()
    sweep_time = time.perf_counter() - t0
    per_sweep = sweep_time / n_sweeps
    print(f"  {n_sweeps} sweeps: {sweep_time:.2f}s ({per_sweep:.2f}s/sweep)")

    return per_sweep


def benchmark_subsample_init(key, n_rows=10_000, n_cols=20, subsample_size=1000):
    """Benchmark subsample init + batch insert workflow."""
    print(f"\n--- Subsample Init: {subsample_size} init, {n_rows} total ---")
    k1, k2, k3, k4 = jax.random.split(key, 4)
    data, col_types = _make_data(k1, n_rows, n_cols)
    max_k = suggest_max_clusters(n_rows)

    # Subsample init
    t0 = time.perf_counter()
    result = initialize(k2, data, col_types, subsample_rows=subsample_size)
    state, sub_idx = result
    sub_data = data[sub_idx]
    packed = pack_state(state, max_clusters=max_k)
    init_time = time.perf_counter() - t0
    print(f"  subsample init + pack: {init_time:.2f}s ({packed.n_rows} rows)")

    # Quick sweeps on subsample
    t0 = time.perf_counter()
    packed = packed_gibbs_sweep(k3, packed, sub_data, n_sweeps=5)
    packed.column_assignments.block_until_ready()
    pre_sweep_time = time.perf_counter() - t0
    print(f"  5 pre-sweeps on subsample: {pre_sweep_time:.2f}s")

    # Insert remaining rows in batches
    remaining_mask = jnp.ones(n_rows, dtype=bool).at[sub_idx].set(False)
    remaining_idx = jnp.where(remaining_mask, size=n_rows - subsample_size)[0]
    remaining_data = data[remaining_idx]
    batch_size = min(1000, remaining_data.shape[0])
    n_batches = (remaining_data.shape[0] + batch_size - 1) // batch_size

    t0 = time.perf_counter()
    current_data = sub_data
    for b in range(n_batches):
        batch = remaining_data[b * batch_size : (b + 1) * batch_size]
        kb = jax.random.fold_in(k4, b)
        packed, current_data = packed_insert_rows(kb, packed, current_data, batch)
    insert_time = time.perf_counter() - t0
    print(f"  inserted {remaining_data.shape[0]} rows in {n_batches} batches: {insert_time:.2f}s")
    print(f"  final n_rows: {packed.n_rows}")

    # Post-insertion sweeps
    k5 = jax.random.fold_in(k4, 999)
    t0 = time.perf_counter()
    packed = packed_gibbs_sweep(k5, packed, current_data, n_sweeps=3)
    packed.column_assignments.block_until_ready()
    post_sweep_time = time.perf_counter() - t0
    print(f"  3 post-sweeps on full data: {post_sweep_time:.2f}s")

    total = init_time + pre_sweep_time + insert_time + post_sweep_time
    print(f"  TOTAL workflow: {total:.2f}s")
    return total


def benchmark_insert_throughput(key, n_insert=5000, n_cols=20):
    """Measure batch insertion throughput."""
    print(f"\n--- Insert Throughput: {n_insert} rows into 1000-row base ---")
    k1, k2, k3, k4 = jax.random.split(key, 4)
    base_data, col_types = _make_data(k1, 1000, n_cols)
    new_rows, _ = _make_data(k2, n_insert, n_cols)
    max_k = suggest_max_clusters(1000 + n_insert)

    state = initialize(k3, base_data, col_types)
    packed = pack_state(state, max_clusters=max_k)
    packed = packed_gibbs_sweep(jax.random.fold_in(k3, 1), packed, base_data, n_sweeps=5)
    packed.column_assignments.block_until_ready()

    # Batch insert
    batch_size = 500
    n_batches = (n_insert + batch_size - 1) // batch_size
    current_data = base_data

    t0 = time.perf_counter()
    for b in range(n_batches):
        batch = new_rows[b * batch_size : (b + 1) * batch_size]
        kb = jax.random.fold_in(k4, b)
        packed, current_data = packed_insert_rows(kb, packed, current_data, batch)
    total = time.perf_counter() - t0
    rows_per_sec = n_insert / total
    print(f"  {n_insert} rows in {total:.2f}s ({rows_per_sec:.0f} rows/s)")
    print(f"  final n_rows: {packed.n_rows}")
    return rows_per_sec


def main():
    print("=" * 60)
    print("JAX-CrossCat 10K Scaling Benchmark")
    print("=" * 60)

    backend = jax.default_backend()
    print(f"Backend: {backend}")
    print(f"Devices: {jax.devices()}")
    print(f"suggest_max_clusters(10000) = {suggest_max_clusters(10000)}")
    print(f"suggest_max_clusters(100000) = {suggest_max_clusters(100000)}")
    print(f"suggest_max_clusters(1000000) = {suggest_max_clusters(1000000)}")

    key = jax.random.key(42)

    # 1. Full sweep on 10K
    per_sweep = benchmark_full_sweep(jax.random.fold_in(key, 1))

    # 2. Subsample init workflow
    workflow_time = benchmark_subsample_init(jax.random.fold_in(key, 2))

    # 3. Insert throughput
    throughput = benchmark_insert_throughput(jax.random.fold_in(key, 3))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Per-sweep (10K x 20): {per_sweep:.2f}s")
    print(f"  Subsample workflow total: {workflow_time:.2f}s")
    print(f"  Insert throughput: {throughput:.0f} rows/s")
    print("=" * 60)

    # Save results
    results_dir = Path("benchmarks/results/scaling")
    results_dir.mkdir(parents=True, exist_ok=True)
    import json

    results = {
        "backend": backend,
        "device": str(jax.devices()[0]),
        "per_sweep_10k": per_sweep,
        "workflow_time": workflow_time,
        "insert_throughput": throughput,
    }
    with open(results_dir / "scaling_10k_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_dir / 'scaling_10k_results.json'}")


if __name__ == "__main__":
    main()
