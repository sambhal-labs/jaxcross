# Model

::: crosscat.model
    options:
      members:
        - initialize
        - log_joint
        - insert_rows
      show_source: false

## Overview

State initialization, scoring, and row insertion.

## `initialize`

```python
initialize(
    rng_key, data, column_types, *,
    n_chains=1,
    column_crp_alpha=1.0,
    row_crp_alpha=1.0,
    initialization="from_the_prior",
    subsample_rows=None,
) -> InitResult
```

Create initial CrossCat state(s).

| Parameter | Type | Description |
|-----------|------|-------------|
| `rng_key` | `Array` | JAX PRNG key |
| `data` | `Array (n_rows, n_cols)` | Observation matrix |
| `column_types` | `list[ColumnType]` | Type per column |
| `n_chains` | `int` | Number of independent initializations |
| `column_crp_alpha` | `float` | Outer DP concentration parameter |
| `row_crp_alpha` | `float` | Inner DP concentration parameter |
| `initialization` | `str` | `"from_the_prior"`, `"together"`, or `"apart"` |
| `subsample_rows` | `int \| None` | If set, CRP-sample this many rows for fast initialization on large datasets. The selected indices are stored in `InitResult.subsample_idx`. |

**Returns**: [`InitResult`](types.md#initresult) — access the state via `result.state` (`CrossCatState` if `n_chains=1`, else `list[CrossCatState]`). When `subsample_rows` is set, `result.subsample_idx` contains the selected row indices.

**Initialization modes:**

- `"from_the_prior"` — sample column and row assignments from CRP priors
- `"together"` — all columns in one view (conservative start)
- `"apart"` — each column in its own view (exploratory start)

## `log_joint`

```python
log_joint(state, data) -> Array
```

Compute joint log probability of state and data. Includes CRP priors for column and row partitions, data log-marginal likelihoods, and Exp(1) priors on CRP alpha parameters.

**Returns**: Scalar log probability.

## `insert_rows`

```python
insert_rows(rng_key, state, data, new_rows) -> tuple[CrossCatState, Array]
```

Insert new rows via CRP predictive (no re-inference on existing rows). Each new row is assigned to clusters using the posterior predictive of existing clusters.

**Returns**: `(updated_state, updated_data)` — the state with new rows incorporated and the data matrix with new rows appended.
