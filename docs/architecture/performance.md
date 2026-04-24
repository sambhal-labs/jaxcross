# Performance Optimizations

The packed kernels were optimized for **12x speedup** in v0.9.0 via three techniques.

## 1. Vectorized Column Scoring

**Before**: `_score_row_one_cluster` used `lax.scan` to sequentially iterate over columns, computing one `unified_posterior_predictive_logp` per step. For 257 columns, this created a massive sequential XLA graph.

**After**: Replaced with `jax.vmap(unified_posterior_predictive_logp)` over all columns simultaneously. Column data, types, sufficient statistics, and hyperparameters are gathered into stacked arrays and scored in parallel.

## 2. Type-Specialized Scoring

**Before**: Every column score computed all 5 type results (NormalGamma, DirichletCategorical, BetaBernoulli, OrderedLogistic, VonMises) via nested `jnp.where`, wasting ~80% of computation for homogeneous-type views.

**After**: `_compute_dominant_type()` detects when all columns in a view share one type. When detected, `_score_row_one_cluster_typed` calls type-specific batch functions (`batch_bb_posterior_predictive_logp`, `batch_ng_posterior_predictive_logp`, `batch_dc_posterior_predictive_logp`) that skip the `jnp.where` dispatch entirely.

## 3. Batched Suffstat Updates

**Before**: `_add_row_to_suffstats` / `_remove_row_from_suffstats` used `lax.scan` to update one column at a time.

**After**: Uses batched `.at[cluster_id, li_range].add()` scatter operations over all columns simultaneously.

## Combined Effect

| Dataset | Before | After | Speedup |
|---------|--------|-------|---------|
| 50 x 11 | 25s/sweep | 4.5s/sweep | 5.5x |
| 100 x 65 | 38s/sweep | 4.8s/sweep | 7.9x |
| 1000 x 257 (MNIST) | 238s/sweep | 20s/sweep | **12x** |

JIT compilation time also dropped from 20+ minutes to ~23 seconds for 257 columns because the XLA graph is much smaller (single vmap vs 257-step unrolled scan).

## Benchmark Results (P100 GPU, v0.10.0)

| Dataset | Rows x Cols | Per Sweep | 100 Sweeps |
|---------|-------------|-----------|------------|
| Small (mixed types) | 50 x 11 | 4.5s | 7.5 min |
| Medium (binary+cat) | 100 x 65 | 4.8s | 8 min |
| MNIST 16x16 | 1000 x 257 | 12s | 20 min |

!!! note
    The Combined Effect table above shows v0.9.0 per-sweep times (20s for MNIST). The v0.10.0 kernel splitting and XLA persistent cache further reduced per-sweep time to ~12s.

## Profiling JAX Kernels

When a sweep is slower than expected, profile before optimizing.

### `jax.profiler.trace` (recommended for kernel analysis)

```python
import jax
import jax.numpy as jnp
from crosscat.packed import pack_state, packed_gibbs_sweep

jax.profiler.start_trace("/tmp/jaxcross-trace")
packed = packed_gibbs_sweep(key, packed, data, n_sweeps=10)
packed.column_assignments.block_until_ready()  # ensure completion
jax.profiler.stop_trace()
```

Open `/tmp/jaxcross-trace` in [Perfetto UI](https://ui.perfetto.dev/) (drag-and-drop the trace file) to see HLO-level timings per kernel.

Typical findings:

- **Long XLA compile step** → enable the [XLA persistent cache](../guides/xla-cache.md) (auto-enabled on import, but check your cache dir).
- **Repeated small kernels** → one of your tensor shapes is changing per sweep, causing recompilation. Check that `max_views` / `max_clusters` / `max_categories` are held constant.
- **Long memcpy** → you're transferring data between host and device every sweep. Move data to GPU once with `jax.device_put(data)`.

### TensorBoard during inference

The library ships `crosscat.tb_logger.TBLogger` (context manager) that logs per-sweep diagnostics (`log_joint`, Rhat, ESS, cluster counts) to TensorBoard via `tensorboardX`. See the [TB Logger guide](../guides/tb-logger.md) for the integration pattern; the WDI benchmark notebook has a worked example.

### Ruling out hardware

Before optimizing kernel math, confirm the device is not the bottleneck:

```python
print(jax.devices())                     # is GPU listed?
print(jax.default_backend())             # "gpu" or "cpu"?
# Force a blocking sync and measure one sweep in isolation
import time
t0 = time.perf_counter()
packed = packed_gibbs_sweep(key, packed, data, n_sweeps=1)
packed.column_assignments.block_until_ready()
print(f"One sweep: {time.perf_counter() - t0:.2f}s")
```

On GTX 1650 (4GB VRAM), the 257-column MNIST sweep runs at ~30–40 s/sweep — slower than P100 but functional. If you see orders-of-magnitude regressions, JAX has likely silently fallen back to CPU.
