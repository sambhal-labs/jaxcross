"""Scaling benchmark — validates jaxcross at 100K rows.

Tests:
  1. Subsample init (5K) + batch insert (95K) + Gibbs sweeps
  2. Data connector roundtrip (save_npy / load_npy_mmap)

Designed for Kaggle T4 (16GB VRAM). Not runnable on small GPUs.

Usage:
    uv run python benchmarks/scaling_100k_benchmark.py
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from crosscat.data_utils import load_npy_mmap, save_npy
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


def benchmark_subsample_workflow(key, n_rows=100_000, n_cols=20, subsample_size=5000):
    """Full workflow: subsample init -> batch insert -> Gibbs sweeps."""
    print(f"\n--- Subsample Workflow: {n_rows} rows x {n_cols} cols ---")
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    data, col_types = _make_data(k1, n_rows, n_cols)
    max_k = suggest_max_clusters(n_rows)
    data_mb = data.nbytes / (1024 * 1024)
    print(f"  data: {data_mb:.1f} MB, max_clusters: {max_k}")

    # Step 1: Subsample init
    t0 = time.perf_counter()
    result = initialize(k2, data, col_types, subsample_rows=subsample_size)
    state = result.state
    sub_idx = result.subsample_idx
    sub_data = data[sub_idx]
    packed = pack_state(state, max_clusters=max_k)
    init_time = time.perf_counter() - t0
    print(f"  1. subsample init ({subsample_size} rows): {init_time:.2f}s")

    # Step 2: Pre-sweeps on subsample
    t0 = time.perf_counter()
    packed = packed_gibbs_sweep(k3, packed, sub_data, n_sweeps=5)
    packed.column_assignments.block_until_ready()
    pre_sweep_time = time.perf_counter() - t0
    print(f"  2. 5 pre-sweeps on subsample: {pre_sweep_time:.2f}s")

    # Step 3: Batch insert remaining rows
    remaining_mask = jnp.ones(n_rows, dtype=bool).at[sub_idx].set(False)
    remaining_idx = jnp.where(remaining_mask, size=n_rows - subsample_size)[0]
    remaining_data = data[remaining_idx]
    batch_size = 5000
    n_batches = (remaining_data.shape[0] + batch_size - 1) // batch_size

    t0 = time.perf_counter()
    current_data = sub_data
    for b in range(n_batches):
        batch = remaining_data[b * batch_size : (b + 1) * batch_size]
        kb = jax.random.fold_in(k4, b)
        packed, current_data = packed_insert_rows(kb, packed, current_data, batch)
        if (b + 1) % 5 == 0 or b == n_batches - 1:
            elapsed = time.perf_counter() - t0
            print(f"     batch {b + 1}/{n_batches}: {packed.n_rows} rows, {elapsed:.1f}s elapsed")
    insert_time = time.perf_counter() - t0
    print(f"  3. inserted {remaining_data.shape[0]} rows: {insert_time:.2f}s")

    # Step 4: Post-insertion sweeps on full data
    t0 = time.perf_counter()
    packed = packed_gibbs_sweep(k5, packed, current_data, n_sweeps=3)
    packed.column_assignments.block_until_ready()
    post_sweep_time = time.perf_counter() - t0
    per_sweep = post_sweep_time / 3
    print(f"  4. 3 post-sweeps on full {packed.n_rows} rows: {post_sweep_time:.2f}s")
    print(f"     per sweep: {per_sweep:.2f}s")

    total = init_time + pre_sweep_time + insert_time + post_sweep_time
    print(f"  TOTAL: {total:.2f}s")
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "init_time": init_time,
        "pre_sweep_time": pre_sweep_time,
        "insert_time": insert_time,
        "post_sweep_time": post_sweep_time,
        "per_sweep_full": per_sweep,
        "total": total,
    }


def benchmark_data_connectors(key, n_rows=100_000, n_cols=20):
    """Test save_npy / load_npy_mmap roundtrip at 100K scale."""
    print(f"\n--- Data Connector Roundtrip: {n_rows} x {n_cols} ---")
    data, col_types = _make_data(key, n_rows, n_cols)
    col_names = [f"col_{j}" for j in range(n_cols)]

    tmp_dir = Path("/tmp/jaxcross_bench")
    tmp_dir.mkdir(exist_ok=True)
    npy_path = tmp_dir / "test_100k.npy"

    # Save
    t0 = time.perf_counter()
    save_npy(npy_path, data, column_names=col_names)
    save_time = time.perf_counter() - t0
    file_size_mb = npy_path.stat().st_size / (1024 * 1024)
    print(f"  save_npy: {save_time:.2f}s ({file_size_mb:.1f} MB)")

    # Load with mmap
    t0 = time.perf_counter()
    loaded_data, loaded_names = load_npy_mmap(npy_path)
    load_time = time.perf_counter() - t0
    print(f"  load_npy_mmap: {load_time:.2f}s")

    # Verify (use np.allclose to avoid defeating mmap with JAX conversion)
    assert loaded_data.shape == data.shape, f"Shape mismatch: {loaded_data.shape} vs {data.shape}"
    assert np.allclose(loaded_data, np.asarray(data), equal_nan=True), "Data mismatch"
    assert loaded_names == col_names, "Column names mismatch"
    print("  roundtrip verified OK")

    # Cleanup
    npy_path.unlink(missing_ok=True)
    npy_path.with_suffix(".json").unlink(missing_ok=True)

    return {"save_time": save_time, "load_time": load_time, "file_size_mb": file_size_mb}


def main():
    print("=" * 60)
    print("JAX-CrossCat 100K Scaling Benchmark")
    print("=" * 60)

    backend = jax.default_backend()
    print(f"Backend: {backend}")
    print(f"Devices: {jax.devices()}")

    key = jax.random.key(42)

    # 1. Data connector roundtrip
    connector_results = benchmark_data_connectors(jax.random.fold_in(key, 1))

    # 2. Full subsample workflow
    workflow_results = benchmark_subsample_workflow(jax.random.fold_in(key, 2))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(
        f"  Data roundtrip: save={connector_results['save_time']:.2f}s, "
        f"load={connector_results['load_time']:.2f}s"
    )
    print(f"  Per-sweep (100K x 20): {workflow_results['per_sweep_full']:.2f}s")
    print(f"  Total workflow: {workflow_results['total']:.2f}s")
    print("=" * 60)

    # Save results
    results_dir = Path("benchmarks/results/scaling")
    results_dir.mkdir(parents=True, exist_ok=True)
    import json

    results = {
        "backend": backend,
        "device": str(jax.devices()[0]),
        "connectors": connector_results,
        "workflow": workflow_results,
    }
    with open(results_dir / "scaling_100k_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_dir / 'scaling_100k_results.json'}")


if __name__ == "__main__":
    main()
