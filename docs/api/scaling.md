# Scaling

::: crosscat.scaling
    options:
      show_source: false

## Overview

Higher-level workflows that combine subsample initialization, batch insertion, and mini-batch Gibbs sweeps for datasets with 10K+ rows. See the [Scaling Guide](../guides/scaling.md) for usage patterns.

---

## `subsample_anneal`

```python
subsample_anneal(
    rng_key, data, column_types, *,
    initial_size=1000, growth_factor=2.0,
    sweeps_per_stage=10, max_clusters=None,
    max_views=16, insert_batch_size=5000,
) -> tuple[PackedCrossCatState, Array]
```

Gradually grow the dataset during inference. Starts with a small subsample, runs Gibbs sweeps to find structure, then progressively inserts more rows and refines.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rng_key` | `Array` | JAX PRNG key |
| `data` | `Array (n_rows, n_cols)` | Full data matrix |
| `column_types` | `list[ColumnType]` | Type per column |
| `initial_size` | `int` | Number of rows for initial subsample (default 1000) |
| `growth_factor` | `float` | Multiplicative growth per stage (default 2x) |
| `sweeps_per_stage` | `int` | Gibbs sweeps per annealing stage |
| `max_clusters` | `int \| None` | Max clusters for `pack_state`. If None, uses `suggest_max_clusters` heuristic |
| `max_views` | `int` | Max views for `pack_state` |
| `insert_batch_size` | `int` | Batch size for `packed_insert_rows` calls |

**Returns**: `(packed_state, reordered_data)` — subsample rows first, then remaining rows in insertion order.

**Stages:**

1. Initialize on `initial_size` rows, run sweeps
2. Grow active rows by `growth_factor`, insert new batch, run sweeps
3. Repeat step 2 until all rows are included

---

## `minibatch_gibbs_sweep`

```python
minibatch_gibbs_sweep(
    rng_key, packed, data, *,
    batch_size=10_000, n_sweeps=1,
) -> PackedCrossCatState
```

Mini-batch Gibbs sweeps. Each sweep updates `batch_size` randomly sampled rows, then runs full column assignment, column hyper, and CRP alpha transitions. Row kernel cost is O(B) instead of O(N).

| Parameter | Type | Description |
|-----------|------|-------------|
| `rng_key` | `Array` | JAX PRNG key |
| `packed` | `PackedCrossCatState` | Current packed state |
| `data` | `Array (n_rows, n_cols)` | Data matrix |
| `batch_size` | `int` | Number of rows to update per sweep |
| `n_sweeps` | `int` | Number of sweeps to run |

**Returns**: Updated `PackedCrossCatState`.

---

## `gibbs_sweep_early_stopping`

```python
gibbs_sweep_early_stopping(
    rng_key, packed, data, *,
    max_sweeps=200, check_interval=10,
    patience=3, min_improvement=0.001,
    batch_size=None,
) -> tuple[PackedCrossCatState, list[float]]
```

Run Gibbs sweeps with convergence-based early stopping. Monitors log-joint probability every `check_interval` sweeps. Stops when the relative improvement falls below `min_improvement` for `patience` consecutive checks.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rng_key` | `Array` | JAX PRNG key |
| `packed` | `PackedCrossCatState` | Current packed state |
| `data` | `Array (n_rows, n_cols)` | Data matrix |
| `max_sweeps` | `int` | Maximum number of sweeps |
| `check_interval` | `int` | Sweeps between convergence checks |
| `patience` | `int` | Checks with insufficient improvement before stopping |
| `min_improvement` | `float` | Minimum relative improvement threshold |
| `batch_size` | `int \| None` | If set, use mini-batch row transitions. If None, use full sweeps |

**Returns**: `(final_packed_state, log_joint_history)`.

!!! warning
    If the log-joint becomes NaN or infinite, the loop stops immediately with a warning. The state may be degenerate.

---

## `parallel_gibbs_sweep`

```python
parallel_gibbs_sweep(
    rng_key, packed, data, *,
    n_sweeps=1,
) -> PackedCrossCatState
```

Gibbs sweeps using parallel row scoring (single device). Uses `packed_transition_row_assignments_parallel` for row assignments (vmap over all rows with leave-one-out suffstat correction) and standard kernels for column/hyper/CRP transitions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rng_key` | `Array` | JAX PRNG key |
| `packed` | `PackedCrossCatState` | Current packed state |
| `data` | `Array (n_rows, n_cols)` | Data matrix |
| `n_sweeps` | `int` | Number of sweeps to run |

**Returns**: Updated `PackedCrossCatState`.

!!! warning
    The parallel row kernel cannot create new clusters. Alternate with a sequential or minibatch sweep periodically for cluster birth/death.
