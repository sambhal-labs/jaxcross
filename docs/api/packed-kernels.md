# Packed Kernels

::: crosscat.packed.kernels
    options:
      members:
        - packed_gibbs_sweep
        - packed_gibbs_step
        - packed_transition_row_assignments
        - packed_transition_row_assignments_minibatch
        - packed_transition_row_assignments_parallel
        - packed_transition_column_assignments
        - packed_transition_column_hypers
        - packed_transition_crp_alphas
        - packed_log_joint
        - packed_insert_rows
        - multi_chain_packed_gibbs_sweep
      show_source: false

## Overview

All Gibbs kernels are fully JIT-compiled using `jax.lax.scan` and `jax.vmap`. Sub-kernels have independent `@jax.jit` decorators for separate compilation.

## `packed_gibbs_sweep`

```python
packed_gibbs_sweep(
    rng_key, packed, data, *,
    n_sweeps=1
) -> PackedCrossCatState
```

Run full Gibbs sweeps on packed state using `lax.scan` for maximum throughput. Each sweep runs all 4 kernels (row assignments, column assignments, column hypers, CRP alphas). To run individual kernels selectively, call the sub-kernel functions directly.

**Returns**: Updated `PackedCrossCatState`.

## `packed_gibbs_step`

```python
packed_gibbs_step(rng_key, packed, data) -> PackedCrossCatState
```

Single Gibbs step calling `@jax.jit` sub-kernels independently (4 smaller compilations). Used by constraint enforcement and interactive workflows.

**Returns**: Updated `PackedCrossCatState`.

## `packed_transition_row_assignments`

```python
packed_transition_row_assignments(rng_key, packed, data, *, recompute_suffstats=True) -> PackedCrossCatState
```

Resample row cluster assignments. Uses nested `lax.scan` (outer over views, inner over rows) with `vmap` over clusters for scoring.

!!! tip "recompute_suffstats"
    The `recompute_suffstats` parameter controls whether sufficient statistics are recomputed from scratch after the sweep. Set to `False` when a subsequent kernel (e.g., `packed_transition_column_assignments`) will recompute them anyway. Both `packed_gibbs_sweep` and `packed_gibbs_step` pass `recompute_suffstats=False` internally since column assignments always recomputes suffstats.

## `packed_transition_row_assignments_minibatch`

```python
packed_transition_row_assignments_minibatch(rng_key, packed, data, *, batch_size=10_000) -> PackedCrossCatState
```

Mini-batch row assignment kernel. Randomly selects `batch_size` rows and updates only their cluster assignments. Cost is O(B*K*C) instead of O(N*K*C). Useful for datasets with 10K+ rows.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rng_key` | `Array` | JAX PRNG key |
| `packed` | `PackedCrossCatState` | Current packed state |
| `data` | `Array (n_rows, n_cols)` | Data matrix |
| `batch_size` | `int` | Number of rows to update per call |

**Returns**: Updated `PackedCrossCatState`.

## `packed_transition_row_assignments_parallel`

```python
packed_transition_row_assignments_parallel(rng_key, packed, data) -> PackedCrossCatState
```

Parallel row assignment kernel. Uses `vmap` over all rows simultaneously with leave-one-out suffstat correction. Faster than sequential on wide datasets but cannot create new clusters.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rng_key` | `Array` | JAX PRNG key |
| `packed` | `PackedCrossCatState` | Current packed state |
| `data` | `Array (n_rows, n_cols)` | Data matrix |

**Returns**: Updated `PackedCrossCatState`.

!!! warning
    The parallel kernel cannot create new clusters — it only reassigns rows to existing clusters. Alternate with sequential or minibatch sweeps periodically for cluster birth/death.

## `packed_transition_column_assignments`

```python
packed_transition_column_assignments(rng_key, packed, data) -> PackedCrossCatState
```

Resample column-to-view assignments (outer DP Gibbs). Uses `lax.scan` over columns, `vmap` over views for scoring, bounded CRP sampling for new-view proposals, and automatic view compaction.

## `packed_transition_column_hypers`

```python
packed_transition_column_hypers(rng_key, packed, data) -> PackedCrossCatState
```

Grid-based Gibbs sampling for column hyperparameters, `vmap`-ed over all columns with unified type dispatch.

## `packed_transition_crp_alphas`

```python
packed_transition_crp_alphas(rng_key, packed) -> PackedCrossCatState
```

Sample CRP concentration parameters (row and column) using `vmap` over a log-spaced grid.

## `packed_log_joint`

```python
packed_log_joint(packed, data) -> Array
```

JIT-compatible log-joint probability on packed state. Computes CRP priors + data log-marginal likelihoods + Exp(1) alpha priors.

**Returns**: Scalar log probability.

## `packed_insert_rows`

```python
packed_insert_rows(rng_key, packed, data, new_rows) -> tuple[PackedCrossCatState, Array]
```

Insert new rows into packed state via CRP predictive assignment.

**Returns**: `(updated_packed, updated_data)`.

---

## Multi-Chain

### `multi_chain_packed_gibbs_sweep`

```python
multi_chain_packed_gibbs_sweep(rng_key, packed_list, data, *, n_sweeps=1) -> tuple[PackedCrossCatState, Array]
```

Run `packed_gibbs_sweep` across N chains in parallel via `jax.vmap`. Batches the states, vmaps the sweep, and scores each chain.

**Returns**: `(batched_result, log_joint_scores)` where scores is `(n_chains,)`.

**Example:**

```python
from crosscat import initialize, pack_state, multi_chain_packed_gibbs_sweep, select_best_chain

key = jax.random.key(42)
result = initialize(key, data, column_types, n_chains=4)
states = result.state
packed_list = [pack_state(s) for s in states]

key, subkey = jax.random.split(key)
batched, scores = multi_chain_packed_gibbs_sweep(subkey, packed_list, data, n_sweeps=50)
best = select_best_chain(batched, scores)
```
