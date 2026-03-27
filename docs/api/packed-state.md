# Packed State

::: crosscat.packed.state
    options:
      show_source: false

## Overview

JIT-compatible padded state representation. All arrays use fixed dimensions, enabling `jax.jit`, `jax.vmap`, and `jax.lax.scan`.

## `PackedCrossCatState`

Dataclass with all fields as JAX arrays with fixed padding dimensions. Registered as a JAX pytree for automatic differentiation and JIT compilation.

Key fields include `column_assignments`, `view_row_assignments`, `view_n_clusters`, sufficient statistics arrays (`ss_counts`, `ss_sum_x`, `ss_sum_x_sq`, etc.), hyperparameters, and metadata (`n_rows`, `n_cols`, `max_views`, `max_clusters`).

## `pack_state`

```python
pack_state(state, *, max_views=16, max_clusters=32, max_categories=16) -> PackedCrossCatState
```

Convert `CrossCatState` to JIT-compatible `PackedCrossCatState` with padded arrays.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `CrossCatState` | Unpacked state to convert |
| `max_views` | `int` | Maximum number of views (padding dimension) |
| `max_clusters` | `int` | Maximum clusters per view (padding dimension) |
| `max_categories` | `int` | Maximum categories per categorical column |

!!! tip "Sizing padding dimensions"
    Set `max_views` and `max_clusters` large enough for the data but not excessively — they affect memory. For most datasets, `max_views=8, max_clusters=16` is sufficient. Use larger values for high-dimensional data.

## `unpack_state`

```python
unpack_state(packed, column_types, data=None) -> CrossCatState
```

Convert back to `CrossCatState`. When `data` is provided, sufficient statistics are recomputed from raw data for exact fidelity (recommended).

## `batch_packed_states`

```python
batch_packed_states(packed_list) -> PackedCrossCatState
```

Stack N `PackedCrossCatState` objects into a single batched pytree with leading `(n_chains,)` dimension. Used for multi-chain parallel inference.

## `unbatch_packed_states`

```python
unbatch_packed_states(batched, n_chains) -> list[PackedCrossCatState]
```

Reverse of `batch_packed_states` — extract N individual states from a batched state.

## `select_best_chain`

```python
select_best_chain(batched, scores) -> PackedCrossCatState
```

Pick the chain with the highest score from a batched state.

| Parameter | Type | Description |
|-----------|------|-------------|
| `batched` | `PackedCrossCatState` | Batched state with `(n_chains,)` leading dim |
| `scores` | `Array (n_chains,)` | Score per chain (e.g., from `packed_log_joint`) |

## Type ID Constants

| Constant | Value | Column Type |
|----------|-------|-------------|
| `CONTINUOUS_ID` | 0 | `ColumnType.CONTINUOUS` |
| `CATEGORICAL_ID` | 1 | `ColumnType.CATEGORICAL` |
| `ORDINAL_ID` | 2 | `ColumnType.ORDINAL` |
| `BINARY_ID` | 3 | `ColumnType.BINARY` |
| `CYCLIC_ID` | 4 | `ColumnType.CYCLIC` |
