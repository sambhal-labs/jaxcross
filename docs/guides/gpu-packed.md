# GPU Acceleration (Packed Path)

## What

The packed state representation converts variable-length Python lists into fixed-size padded JAX arrays, enabling full JIT compilation and GPU acceleration. This gives 10-100x speedup over the unpacked path.

## When to Use

- Datasets with 50+ rows or 10+ columns
- GPU/TPU execution
- Production inference (many sweeps)
- Multi-chain workflows

## Workflow

```python
from crosscat import initialize, log_joint
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state

# 1. Initialize (unpacked)
state = initialize(key, data, col_types)

# 2. Pack for JIT compilation
packed = pack_state(state, max_views=16, max_clusters=32)

# 3. Run inference (JIT-compiled)
packed = packed_gibbs_sweep(key, packed, data, n_sweeps=100)

# 4. Unpack for queries
state = unpack_state(packed, col_types, data=data)
print(f"Log joint: {log_joint(state, data):.2f}")
```

## Sizing Padding Dimensions

The padding dimensions determine memory usage and must be large enough for your data:

| Parameter | Default | Description | Guidance |
|-----------|---------|-------------|----------|
| `max_views` | 16 | Max number of column groups | 8 for most data, 16 for 50+ columns |
| `max_clusters` | 32 | Max clusters per view | 16 for <500 rows, 32 for larger |
| `max_categories` | 16 | Max categories per column | Increase if categorical columns have 16+ values |

```python
# Memory-constrained setup
packed = pack_state(state, max_views=8, max_clusters=16)

# High-dimensional data
packed = pack_state(state, max_views=16, max_clusters=32, max_categories=64)
```

!!! warning "Out of memory"
    If you get GPU OOM errors, reduce `max_views` and `max_clusters`. These control the size of all padded arrays.

## Querying Packed State Directly

You can run inference queries directly on packed state without unpacking:

```python
from crosscat import (
    packed_predictive_sample,
    packed_anomaly_score,
    packed_dependence_matrix,
)

# Sample on packed state
samples = packed_predictive_sample(key, packed, data, query_cols=[0], n_samples=1000)

# Anomaly detection
score = packed_anomaly_score(key, packed, data, query_row=42)

# Dependence matrix
z = packed_dependence_matrix([packed])
```

## Batch Queries for Production

For querying many rows at once, use the `batch_*` functions instead of Python loops. These are `vmap`-vectorized and run in a single JIT call:

| Task | Loop (slow) | Batch (fast) |
|------|-------------|--------------|
| Anomaly scan | `for row: packed_anomaly_score(...)` | `batch_anomaly_score(packed, data, row_ids)` |
| Impute column | `for row: packed_impute_and_confidence(...)` | `batch_impute_column(key, packed, data, col, row_ids)` |
| Row typicality | `for row: packed_row_typicality(...)` | `batch_row_typicality(packed_states, row_ids)` |
| Similarity matrix | `for i,j: packed_row_similarity(...)` | `batch_row_similarity(packed_states, row_ids)` |
| Credible intervals | `for row: packed_credible_interval(...)` | `batch_credible_interval(key, packed, data, col, row_ids)` |

```python
from crosscat import batch_anomaly_score, batch_row_typicality
import jax.numpy as jnp

# Score all rows at once
scores = batch_anomaly_score(packed, data, jnp.arange(data.shape[0]))
typicality = batch_row_typicality([packed], jnp.arange(data.shape[0]))
```

See [Packed Inference API](../api/packed-inference.md#batch-queries-vectorized-over-rows) for the full list.

## JIT Compilation Timing

The first call triggers JAX compilation:

| Dataset Size | Compilation | Per Sweep (after) |
|-------------|-------------|-------------------|
| 50 x 11 | ~10s | 4.5s |
| 100 x 65 | ~15s | 4.8s |
| 1000 x 257 | ~23s | 12s |

Use [XLA Compilation Caching](xla-cache.md) to skip recompilation on subsequent runs.

## Tips

- Always pass `data=data` to `unpack_state` for exact sufficient statistics
- Changing data shape triggers recompilation — keep shapes consistent
- Use `packed_gibbs_step` instead of `packed_gibbs_sweep` for interactive/constraint workflows

## API Reference

- [`pack_state`](../api/packed-state.md#pack_state)
- [`unpack_state`](../api/packed-state.md#unpack_state)
- [`packed_gibbs_sweep`](../api/packed-kernels.md#packed_gibbs_sweep)
