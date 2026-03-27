# Constraints

::: crosscat.constraints
    options:
      show_source: false

## Overview

Enforce column and row dependency constraints during inference via rejection sampling.

## `check_column_dep_constraint`

```python
check_column_dep_constraint(state, col_a, col_b, dependent) -> bool
```

Check if two columns satisfy a dependency constraint.

- `dependent=True`: columns must be in the same view
- `dependent=False`: columns must be in different views

**Returns**: `True` if constraint is satisfied.

## `ensure_col_dep_constraints`

```python
ensure_col_dep_constraints(
    rng_key, state, data, constraints, *,
    max_rejections=100,
    n_sweeps_per_attempt=5
) -> CrossCatState | None
```

Find a state satisfying all column constraints via rejection sampling. Runs Gibbs sweeps and checks constraints until all are satisfied or `max_rejections` is exceeded.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rng_key` | `Array` | JAX PRNG key |
| `state` | `CrossCatState` | Starting state |
| `data` | `Array` | Observation matrix |
| `constraints` | `list[tuple[int, int, bool]]` | `(col_a, col_b, dependent)` tuples |
| `max_rejections` | `int` | Maximum attempts before returning `None` |
| `n_sweeps_per_attempt` | `int` | Gibbs sweeps per attempt |

**Returns**: `CrossCatState | None` (`None` if max_rejections exceeded).

## `ensure_row_dep_constraint`

```python
ensure_row_dep_constraint(
    rng_key, state, data, row_a, row_b, dependent, *,
    view_idx=None, max_iterations=100,
    n_sweeps_per_attempt=5
) -> CrossCatState | None
```

Find a state where two rows are in the same or different cluster.

| Parameter | Type | Description |
|-----------|------|-------------|
| `row_a`, `row_b` | `int` | Row indices |
| `dependent` | `bool` | `True` = same cluster, `False` = different |
| `view_idx` | `int \| None` | Restrict to specific view (default: all views) |

**Returns**: `CrossCatState | None`.
