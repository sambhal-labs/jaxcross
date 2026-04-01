# XLA Compilation Caching

Persist compiled XLA kernels to disk so JIT compilation is skipped on subsequent Python sessions.

## Why It Matters

JAX compiles Python functions to optimized machine code on first call. For CrossCat's packed kernels, this takes 20–60 seconds. The XLA persistent cache stores these compiled artifacts so subsequent runs start instantly.

| Scenario | First Sweep Time | Notes |
|----------|------------------|-------|
| No cache | 20–60s | Full JIT compilation |
| Cache exists (same shape) | ~2s | Loads compiled kernels from disk |
| Cache exists (different shape) | 20–60s | Recompiles for new shape |

## Auto-Enabled

XLA caching is **automatically enabled** when you import `crosscat.packed`:

```python
from crosscat.packed import pack_state, packed_gibbs_sweep
# Cache is already active — no extra code needed
```

You only need to call `enable_xla_cache()` manually if you want a custom cache directory.

## Custom Cache Directory

```python
from crosscat.packed.aot_cache import enable_xla_cache

enable_xla_cache(cache_dir="/tmp/my_xla_cache")
```

Default location: `~/.cache/jax/`

## Pre-Compile Kernels

Trigger compilation of all 4 Gibbs sub-kernels for a specific data shape before your main workload:

```python
from crosscat.packed.aot_cache import compile_kernels

compile_kernels(packed, data)
# All sub-kernels are now compiled and cached:
#   - packed_transition_row_assignments
#   - packed_transition_column_assignments
#   - packed_transition_column_hypers
#   - packed_transition_crp_alphas
```

This is useful for:

- **Benchmarking** — ensure you measure inference time, not compilation time
- **Production deployment** — pre-warm the cache during initialization
- **Interactive sessions** — avoid the compilation pause during analysis

## Cache Management

### Clearing the Cache

```python
from crosscat.packed.aot_cache import clear_cache

clear_cache()
```

Clear the cache when:

- Upgrading JAX versions (compiled kernels may be incompatible)
- Changing JAX configuration (e.g., `jax.config.update`)
- Compiled kernels seem stale or produce unexpected results

### Cache Size

The cache stores one compiled artifact per unique (computation graph, input shape) pair. Typical sizes:

- Single dataset shape: ~50–100 MB
- Multiple shapes: scales linearly

### Cache Invalidation

Recompilation is triggered automatically when:

- **Data shape changes** — different `n_rows` or `n_cols`
- **Padding dimensions change** — different `max_views`, `max_clusters`, `max_categories`
- **JAX version changes** — XLA compiler output may differ

The cache does **not** invalidate when:

- Data values change (same shape)
- Hyperparameters change
- Random keys change

## How It Works

1. JAX's XLA compiler converts Python + JAX operations into optimized GPU/CPU machine code
2. The computation graph is hashed based on the function and input shapes
3. Compiled artifacts are written to the cache directory
4. On subsequent calls with matching shapes, the cached artifact is loaded directly — skipping compilation entirely

This is particularly impactful for CrossCat because `packed_gibbs_sweep` compiles a large computation graph (all 4 kernel types composed via `lax.scan`).

## API Reference

- [`enable_xla_cache`](../api/aot-cache.md#enable_xla_cache)
- [`compile_kernels`](../api/aot-cache.md#compile_kernels)
- [`clear_cache`](../api/aot-cache.md#clear_cache)
