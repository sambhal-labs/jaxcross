# Types & State

::: crosscat.types
    options:
      show_source: false

## Overview

Core dataclasses and types for all CrossCat state.

## `ColumnType`

Enum specifying column data types.

| Value | Description | Component Model |
|-------|-------------|-----------------|
| `CONTINUOUS` | Real-valued data | NormalGamma (Normal-Inverse-Gamma) |
| `CATEGORICAL` | Unordered integer labels | DirichletCategorical |
| `BINARY` | 0 or 1 | BetaBernoulli |
| `ORDINAL` | Ordered integer levels | OrderedLogistic (cumulative link) |
| `CYCLIC` | Angles in [0, 2*pi) | VonMises |

## `CrossCatState`

Full model state containing column partition, row clusterings, hyperparameters, and sufficient statistics.

| Field | Type | Description |
|-------|------|-------------|
| `column_assignments` | `Array (n_cols,)` | Column-to-view mapping |
| `column_crp_alpha` | `Array (scalar)` | Outer DP concentration |
| `column_hypers` | `list[ColumnHypers]` | Per-column hyperparameters |
| `column_types` | `list[ColumnType]` | Per-column type |
| `views` | `list[ViewState]` | View states |
| `n_rows` | `int` | Number of data rows |
| `n_cols` | `int` | Number of data columns |
| `n_views` | `int` (property) | Number of active views |

## `ViewState`

State for a single view (column group).

| Field | Type | Description |
|-------|------|-------------|
| `column_indices` | `Array (n_cols_in_view,)` | Which columns belong to this view |
| `row_assignments` | `Array (n_rows,)` | Row-to-cluster mapping |
| `row_crp_alpha` | `Array (scalar)` | Inner DP concentration |
| `suffstats` | `list[list[SufficientStats]]` | `suffstats[cluster][col_in_view]` |

## `ColumnHypers`

Per-column hyperparameters. Fields vary by column type.

| Field | Type | Used by |
|-------|------|---------|
| `column_type` | `ColumnType` | All |
| `mu`, `r`, `s`, `nu` | `Array \| None` | CONTINUOUS (Normal-Gamma) |
| `dirichlet_alpha` | `Array \| None` | CATEGORICAL |
| `alpha`, `beta` | `Array \| None` | BINARY (Beta-Bernoulli) |
| `cutpoints` | `Array \| None` | ORDINAL (Ordered Logistic) |
| `kappa`, `vm_a`, `vm_mu` | `Array \| None` | CYCLIC (Von Mises) |

## `SufficientStats`

Sufficient statistics for a (cluster, column) pair.

| Field | Type | Used by |
|-------|------|---------|
| `column_type` | `ColumnType` | All |
| `count` | `Array` | All |
| `sum_x` | `Array \| None` | CONTINUOUS, BINARY |
| `sum_x_sq` | `Array \| None` | CONTINUOUS |
| `category_counts` | `Array \| None` | CATEGORICAL, ORDINAL |
| `sum_sin`, `sum_cos` | `Array \| None` | CYCLIC |

## `InitResult`

Frozen dataclass returned by [`initialize()`](model.md#initialize). Wraps state(s) with optional subsample info.

| Field | Type | Description |
|-------|------|-------------|
| `state` | `CrossCatState \| list[CrossCatState]` | Single state (`n_chains=1`) or list of states (`n_chains>1`) |
| `subsample_idx` | `Array \| None` | Row indices used for subsampling, shape `(subsample_rows,)`. `None` if full data was used. |

```python
result = initialize(key, data, col_types)
state = result.state  # CrossCatState

result = initialize(key, data, col_types, n_chains=4)
states = result.state  # list[CrossCatState]

result = initialize(key, data, col_types, subsample_rows=5000)
state = result.state
sub_idx = result.subsample_idx  # Array (5000,)
```

## Constants

- `LOG_EPS = 1e-30` — Numerical stability floor. Used throughout as a lower clamp on likelihoods/probabilities before taking logs or dividing (prevents `-inf` / NaN propagation in JIT-traced code where both branches of `jnp.where` execute).
- `LOGISTIC_INF = 1e6` — Saturation cap for the ordinal-logistic location parameter. Ordinal cutpoints are padded with `+LOGISTIC_INF` beyond the real cutpoint count; the kernel masks these to only update real entries (see the [ordered logistic grid algorithm page](../architecture/algorithms/ordered-logistic-grid.md)).
- `ORDINAL_N_GRID = 31` — Grid points for ordinal logistic location parameter integration. Override to trade accuracy for speed (captured at import time).

## Related

- [`PackedCrossCatState`](packed-state.md) — the JIT-friendly flattened variant of `CrossCatState` used by the packed kernels and `packed_inference` queries.
- [`ValidationError`](validation.md) — raised by `assert_valid_state` when state invariants (shape consistency, cluster continuity, suffstat alignment) are violated.
