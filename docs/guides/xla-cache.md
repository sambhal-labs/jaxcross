# XLA Compilation Caching

## What

Persist compiled XLA kernels to disk so JIT compilation is skipped on subsequent Python sessions. Saves 20-60 seconds on startup.

## When to Use

- Production workflows where startup time matters
- Iterative development (re-running the same analysis)
- Benchmarking (avoid measuring compilation time)

## Enable Caching

```python
from crosscat.packed.aot_cache import enable_xla_cache

enable_xla_cache()  # call once at startup, before any JIT calls
```

!!! info
    XLA caching is **auto-enabled** when `crosscat.packed` is imported. You only need to call this manually if you want a custom cache directory.

## Custom Cache Directory

```python
enable_xla_cache(cache_dir="/tmp/my_xla_cache")
```

Default location: `~/.cache/jax/`

## Pre-Compile Kernels

Trigger compilation of all sub-kernels for a specific data shape:

```python
from crosscat.packed.aot_cache import compile_kernels

compile_kernels(packed, data)
# All 4 Gibbs kernels are now compiled and cached
```

This is useful before benchmarking or production deployment.

## Clear Cache

```python
from crosscat.packed.aot_cache import clear_cache

clear_cache()
```

Clear the cache when:

- Upgrading JAX versions
- Changing JAX configuration
- Cached kernels seem stale or produce errors

## How It Works

JAX's XLA compiler produces optimized machine code for each unique computation graph + input shape. The persistent cache stores these compiled artifacts so they can be reloaded without recompilation.

Recompilation is triggered when:

- Data shape changes (different n_rows, n_cols)
- Padding dimensions change (different max_views, max_clusters)
- JAX version changes

## API Reference

- [`enable_xla_cache`](../api/aot-cache.md#enable_xla_cache)
- [`compile_kernels`](../api/aot-cache.md#compile_kernels)
- [`clear_cache`](../api/aot-cache.md#clear_cache)
