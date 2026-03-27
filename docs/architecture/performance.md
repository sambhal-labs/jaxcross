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

## Benchmark Results (P100 GPU)

| Dataset | Rows x Cols | Per Sweep | 100 Sweeps |
|---------|-------------|-----------|------------|
| Small (mixed types) | 50 x 11 | 4.5s | 7.5 min |
| Medium (binary+cat) | 100 x 65 | 4.8s | 8 min |
| MNIST 16x16 | 1000 x 257 | 12s | 20 min |
