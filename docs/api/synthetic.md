# Synthetic Data

::: crosscat.synthetic
    options:
      show_source: false

## Overview

Generate synthetic data from a known CrossCat generative model for testing and benchmarking.

## `generate_crosscat_data`

```python
generate_crosscat_data(
    rng_key, n_rows, column_types, *,
    n_views=2, n_clusters=2, cluster_separation=5.0
) -> dict
```

Generate data from a known CrossCat generative model with ground-truth assignments.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rng_key` | `Array` | JAX PRNG key |
| `n_rows` | `int` | Number of rows to generate |
| `column_types` | `list[ColumnType]` | Type per column |
| `n_views` | `int` | Number of column groups (views) |
| `n_clusters` | `int` | Number of row clusters per view |
| `cluster_separation` | `float` | Controls how well-separated clusters are |

**Returns**: Dict with keys:

- `data` — `Array (n_rows, n_cols)`
- `column_types` — `list[ColumnType]`
- `true_column_assignments` — `Array (n_cols,)`
- `true_row_assignments` — `list[Array]` (one per view)
- `n_rows`, `n_cols`

## `add_missing_data`

```python
add_missing_data(rng_key, data, missing_fraction=0.1) -> Array
```

Inject random NaN values into data for testing missing data handling.

**Returns**: `Array` with `missing_fraction` of entries replaced with NaN.
