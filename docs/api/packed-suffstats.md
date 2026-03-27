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
compute_suffstats_vectorized(data, col_type_ids, n_categories=None) -> tuple
```

Compute sufficient statistics for all columns simultaneously using matrix operations and batched scatter-add.

**Returns**: Tuple of arrays (counts, sum_x, sum_x_sq, category_counts, sum_sin, sum_cos).

## `recompute_all_suffstats`

```python
recompute_all_suffstats(packed, data) -> PackedCrossCatState
```

Recompute all sufficient statistics in a packed state from raw data and current cluster assignments. Useful after deserialization or manual state modification.

**Returns**: Updated `PackedCrossCatState` with fresh sufficient statistics.
