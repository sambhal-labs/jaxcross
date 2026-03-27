# Packed Inference

::: crosscat.packed_inference
    options:
      show_source: false

## Overview

Vectorized inference queries on `PackedCrossCatState`. These are the packed equivalents of [`crosscat.inference`](inference.md) functions, plus multi-chain wrappers that average results across posterior samples.

---

## Single-State Queries

### `packed_predictive_probability`

```python
packed_predictive_probability(packed, data, query_cols, query_vals, *, row_id=None) -> Array
```

Conditional predictive log-probability on packed state.

### `packed_predictive_sample`

```python
packed_predictive_sample(rng_key, packed, data, query_cols, *, n_samples=1000, row_id=None) -> Array
```

Posterior predictive samples on packed state.

**Returns**: `Array (n_samples, len(query_cols))`.

### `packed_predictive_cdf`

```python
packed_predictive_cdf(rng_key, packed, data, query_col, query_val, *, n_samples=10000) -> Array
```

Posterior predictive CDF on packed state.

**Returns**: Scalar in [0, 1].

### `packed_credible_interval`

```python
packed_credible_interval(rng_key, packed, data, query_col, *, n_samples=1000, ci_level=0.90) -> tuple[Array, Array, Array]
```

Credible interval on packed state.

**Returns**: `(median, lower, upper)`.

### `packed_impute_and_confidence`

```python
packed_impute_and_confidence(rng_key, packed, data, query_col, *, n_samples=1000) -> tuple[Array, Array]
```

Imputation with confidence on packed state.

**Returns**: `(point_estimate, confidence_score)`.

### `packed_anomaly_score`

```python
packed_anomaly_score(rng_key, packed, data, query_row, *, n_samples=500) -> Array
```

Anomaly score for a row on packed state.

**Returns**: Scalar in [0, 1].

### `packed_joint_predictive_probability`

```python
packed_joint_predictive_probability(packed, data, query_cols, query_vals) -> Array
```

Joint predictive probability on packed state.

### `packed_sample_and_insert`

```python
packed_sample_and_insert(rng_key, packed, data, partial_row) -> tuple[PackedCrossCatState, Array, Array]
```

Sample missing values and insert completed row into packed state.

**Returns**: `(updated_packed, updated_data, completed_row)`.

---

## Structure Queries (Accept Lists of States)

### `packed_dependence_probability`

```python
packed_dependence_probability(packed_states, col_a, col_b) -> Array
```

Posterior probability that two columns are dependent.

### `packed_dependence_matrix`

```python
packed_dependence_matrix(packed_states) -> Array
```

Full Z-matrix from packed states.

**Returns**: `Array (n_cols, n_cols)`.

### `packed_mutual_information`

```python
packed_mutual_information(packed_states, column_types, col_i, col_j, *, n_samples=1000, rng_key=None) -> tuple[Array, Array]
```

MI and Linfoot correlation from packed states.

**Returns**: `(mi, linfoot_correlation)`.

### `packed_row_similarity`

```python
packed_row_similarity(packed_states, row_a, row_b, *, target_columns=None) -> Array
```

Row co-clustering probability on packed states.

### `packed_row_typicality`

```python
packed_row_typicality(packed_states, row_id) -> Array
```

Row structural typicality on packed states.

### `packed_column_typicality`

```python
packed_column_typicality(packed_states, col_id) -> Array
```

Column structural typicality on packed states.

### `packed_conditional_entropy`

```python
packed_conditional_entropy(rng_key, packed_states, data, target_col, given_cols, *, n_samples=500) -> Array
```

Conditional entropy H(target | given) on packed states.

---

## Multi-Chain Wrappers

These functions average query results across multiple chains for more robust estimates.

### `multi_chain_predictive_probability`

```python
multi_chain_predictive_probability(packed_states, data, query_cols, query_vals, *, row_id=None) -> Array
```

Average predictive probability across chains.

### `multi_chain_predictive_sample`

```python
multi_chain_predictive_sample(rng_key, packed_states, data, query_cols, *, n_samples=1000, row_id=None) -> Array
```

Sample from posterior averaging across chains.

### `multi_chain_anomaly_score`

```python
multi_chain_anomaly_score(rng_key, packed_states, data, query_row) -> Array
```

Average anomaly score across chains.

### `multi_chain_impute_and_confidence`

```python
multi_chain_impute_and_confidence(rng_key, packed_states, data, query_col, *, n_samples=1000) -> tuple[Array, Array]
```

Imputation with confidence averaged across chains.

### `multi_chain_predictive_cdf`

```python
multi_chain_predictive_cdf(rng_key, packed_states, data, query_col, query_val, *, n_samples=10000) -> Array
```

Predictive CDF averaged across chains.
