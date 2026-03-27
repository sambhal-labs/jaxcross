# AOT Cache

::: crosscat.packed.aot_cache
    options:
      show_source: false

## Overview

XLA persistent compilation cache for skipping JIT recompilation across Python sessions.

## `enable_xla_cache`

```python
enable_xla_cache(cache_dir=None) -> None
```

Enable persistent XLA compilation cache. Saves compiled kernels to `~/.cache/jax/` (or specified directory) so subsequent runs skip the 20-60s JIT compilation step.

!!! info
    This is auto-enabled when `crosscat.packed` is imported. Call manually only if you need a custom cache directory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cache_dir` | `str \| Path \| None` | Custom cache directory (default: `~/.cache/jax/`) |

## `compile_kernels`

```python
compile_kernels(packed, data) -> None
```

Pre-compile all Gibbs sub-kernels for a given state shape. Triggers JIT compilation of all 4 kernels so that subsequent calls are instant.

Use this for warm-up before benchmarking or production inference.

## `clear_cache`

```python
clear_cache(cache_dir=None) -> None
```

Clear the XLA compilation cache. Use when upgrading JAX versions or if cached kernels become stale.
