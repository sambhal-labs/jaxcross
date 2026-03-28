# Packed Suffstats

::: crosscat.packed.suffstats
    options:
      members:
        - compute_suffstats_vectorized
        - recompute_all_suffstats
      show_source: false

## Overview

Vectorized sufficient statistics computation for the packed state path.

## `compute_suffstats_vectorized`

```python
compute_suffstats_vectorized(
    data, column_indices, col_type_ids, row_assignments,
    n_clusters, max_clusters, max_categories
) -> tuple[Array, Array, Array, Array, Array, Array]
```

Compute sufficient statistics for all (cluster, column) pairs in a view using matrix operations and batched scatter-add.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `Array` | Full data matrix (n_rows, n_cols) |
| `column_indices` | `Array` | Column indices in this view, -1 for padding |
| `col_type_ids` | `Array` | Column type IDs for each column |
| `row_assignments` | `Array` | Row-to-cluster assignments (n_rows,) |
| `n_clusters` | `int` | Actual number of clusters |
| `max_clusters` | `int` | Padding dimension for clusters |
| `max_categories` | `int` | Padding dimension for category counts |

**Returns**: Tuple of arrays (counts, sum_x, sum_x_sq, category_counts, sum_sin, sum_cos).

## `recompute_all_suffstats`

```python
recompute_all_suffstats(packed, data) -> PackedCrossCatState
```

Recompute all sufficient statistics in a packed state from raw data and current cluster assignments. Useful after deserialization or manual state modification.

**Returns**: Updated `PackedCrossCatState` with fresh sufficient statistics.
