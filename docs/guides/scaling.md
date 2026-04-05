# Scaling to Large Datasets

## When to Use

When your dataset has **10K+ rows**, standard `packed_gibbs_sweep` may be slow due to O(N\*K\*C) per-sweep cost. The `crosscat.scaling` module provides four strategies to make inference practical at scale.

## Step 0: Estimate Memory

Before running inference, check whether your dataset fits in GPU memory:

```python
from crosscat import estimate_packed_memory

mem = estimate_packed_memory(100_000, 20, max_clusters=16)
print(f"Estimated GPU memory: {mem['total'] / 1e6:.1f} MB")
```

Use `suggest_max_clusters()` for a data-driven `max_clusters` setting:

```python
from crosscat.packed import suggest_max_clusters

k = suggest_max_clusters(100_000)  # returns min(32, max(4, sqrt(N)))
```

## Strategy 1: Subsample Annealing

Best for **initial structure discovery** on large datasets. Starts small, grows progressively:

```python
from crosscat import subsample_anneal

packed, data_reordered = subsample_anneal(
    key, data, col_types,
    initial_size=2000,       # start with 2K rows
    growth_factor=2.0,       # double each stage
    sweeps_per_stage=20,     # refine between growth stages
)
```

**How it works:**

1. Initialize on `initial_size` rows (fast)
2. Run `sweeps_per_stage` Gibbs sweeps
3. Insert next batch of rows (2x growth)
4. Repeat until all rows are included

## Strategy 2: Mini-Batch Gibbs

Best for **ongoing inference** when full sweeps are too expensive:

```python
from crosscat import minibatch_gibbs_sweep

packed = minibatch_gibbs_sweep(
    key, packed, data,
    batch_size=10_000,  # update 10K rows per sweep
    n_sweeps=50,
)
```

Each sweep updates only `batch_size` randomly sampled rows, then runs full column/hyper/CRP transitions. Row kernel cost drops from O(N) to O(B).

## Strategy 3: Parallel Row Scoring

Best for **GPU-rich environments** where you want maximum parallelism:

```python
from crosscat import parallel_gibbs_sweep

packed = parallel_gibbs_sweep(
    key, packed, data,
    n_sweeps=50,
)
```

Uses `jax.vmap` to score all rows simultaneously with leave-one-out suffstat correction.

!!! warning
    The parallel kernel cannot create new clusters. Alternate with sequential or minibatch sweeps periodically for cluster birth/death.

## Strategy 4: Early Stopping

Best for **avoiding wasted compute** when convergence is reached:

```python
from crosscat import gibbs_sweep_early_stopping

packed, log_joints = gibbs_sweep_early_stopping(
    key, packed, data,
    max_sweeps=200,
    check_interval=10,    # check every 10 sweeps
    patience=3,           # stop after 3 stale checks
    min_improvement=0.001,
)
print(f"Converged after {len(log_joints) * 10} sweeps")
```

Combine with mini-batch for large datasets:

```python
packed, log_joints = gibbs_sweep_early_stopping(
    key, packed, data,
    max_sweeps=500,
    batch_size=10_000,  # use mini-batch row transitions
)
```

## Decision Table

| Dataset Size | Strategy | Notes |
|-------------|----------|-------|
| < 10K rows | `packed_gibbs_sweep` | Standard path, no scaling needed |
| 10K–100K rows | Early stopping + full sweeps | Let convergence decide sweep count |
| 100K+ rows | Subsample anneal → minibatch | Anneal for structure, then refine |
| 100K+ rows, GPU-rich | Parallel + periodic sequential | Max GPU utilization |

## Large-File I/O

For large datasets, avoid loading everything into memory at once:

```python
# Chunked CSV loading (bounded memory)
from crosscat import read_csv_chunked
data, names = read_csv_chunked("large.csv", chunk_size=50_000)

# Memory-mapped NumPy (OS pages on demand)
from crosscat import save_npy, load_npy_mmap
save_npy("data.npy", data, col_names)
data_np, names = load_npy_mmap("data.npy")
batch = jnp.array(data_np[0:10_000])  # only this slice hits RAM

# Parquet (columnar, compressed)
from crosscat import read_parquet, write_parquet
data, names = read_parquet("data.parquet", columns=["col1", "col2"])
```

## Full Example: 100K Rows

```python
import jax
from crosscat import (
    subsample_anneal, gibbs_sweep_early_stopping,
    estimate_packed_memory,
    read_csv_chunked, guess_column_types,
)
from crosscat.packed import suggest_max_clusters

# Load data
data, col_names = read_csv_chunked("large_dataset.csv")
col_types = guess_column_types(data)

# Check memory
k = suggest_max_clusters(data.shape[0])
mem = estimate_packed_memory(data.shape[0], data.shape[1], max_clusters=k)
print(f"Estimated memory: {mem['total'] / 1e6:.1f} MB")

# Phase 1: Subsample annealing for structure discovery
key = jax.random.key(42)
packed, data_reord = subsample_anneal(
    key, data, col_types,
    initial_size=2000, sweeps_per_stage=20, max_clusters=k,
)

# Phase 2: Refine with early stopping
key = jax.random.key(99)
packed, log_joints = gibbs_sweep_early_stopping(
    key, packed, data_reord,
    max_sweeps=200, patience=3,
)
```

## API Reference

- [`subsample_anneal`](../api/scaling.md#subsample_anneal)
- [`minibatch_gibbs_sweep`](../api/scaling.md#minibatch_gibbs_sweep)
- [`gibbs_sweep_early_stopping`](../api/scaling.md#gibbs_sweep_early_stopping)
- [`parallel_gibbs_sweep`](../api/scaling.md#parallel_gibbs_sweep)
- [`estimate_packed_memory`](../api/packed-state.md#estimate_packed_memory)
- [`suggest_max_clusters`](../api/packed-state.md#suggest_max_clusters)
