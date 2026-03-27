# Diagnostics

::: crosscat.diagnostics
    options:
      show_source: false

## Overview

Convergence metrics, partition comparison, and held-out evaluation.

## `adjusted_rand_index`

```python
adjusted_rand_index(assignments_true, assignments_pred) -> Array
```

ARI between two partitions. 1 = perfect, 0 = random, <0 = anti-correlated.

## `column_partition_ari`

```python
column_partition_ari(state, true_assignments) -> Array
```

ARI of inferred column partition vs ground truth.

## `row_partition_ari`

```python
row_partition_ari(state, view_idx, true_assignments) -> Array
```

ARI of row partition in a specific view vs ground truth.

## `collect_diagnostics`

```python
collect_diagnostics(state, data) -> dict
```

Per-sweep diagnostic metrics.

**Returns**: Dict with keys: `log_joint`, `n_views`, `column_crp_alpha`, `row_crp_alphas`, `n_clusters_per_view`.

## `mean_test_log_likelihood`

```python
mean_test_log_likelihood(state, data, test_rows) -> Array
```

Mean held-out log-likelihood on specified test rows.

## `random_holdout_mask`

```python
random_holdout_mask(rng_key, n_rows, n_cols, holdout_fraction=0.1) -> Array
```

Create a random boolean mask for held-out evaluation.

**Returns**: `Array (n_rows, n_cols)` boolean mask.

## `evaluate_imputation`

```python
evaluate_imputation(state, data, mask, col_types, *, rng_key=None) -> dict
```

Evaluate imputation accuracy on held-out cells. Computes MAE (continuous), accuracy (discrete), and log-likelihood.

**Returns**: Dict with per-type metrics.
