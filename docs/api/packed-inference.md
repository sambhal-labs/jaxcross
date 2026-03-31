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
packed_anomaly_score(rng_key, packed, data, query_row) -> Array
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

### `packed_classify_column`

```python
packed_classify_column(packed, data, target_col, candidate_vals, row_id) -> Array
```

Compute log P(target_col=v | row) for each candidate value v. Useful for classification where `target_col` is categorical and `candidate_vals` are the possible classes.

**Returns**: `Array (len(candidate_vals),)` with log probabilities.

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
packed_row_similarity(packed_states, column_types, row_a, row_b, *, target_columns=None) -> Array
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

---

## Batch Queries (Vectorized over Rows)

!!! tip "Production path"
    Batch functions are the recommended way to run queries at scale. They use `jax.vmap` to vectorize over rows in a single JIT call — 10-100x faster than Python loops over single-row functions.

### `batch_anomaly_score`

```python
batch_anomaly_score(packed, data, row_ids) -> Array
```

Anomaly scores for multiple rows in one JIT call. Evaluates average log predictive probability across all observed columns per row, then applies a sigmoid transform.

**Returns**: `Array (len(row_ids),)` with anomaly scores in [0, 1]. Higher = more anomalous.

### `batch_row_typicality`

```python
batch_row_typicality(packed_states, row_ids) -> Array
```

Structural typicality scores for multiple rows. Measures how well each row fits its assigned cluster(s), averaged over views and posterior states.

**Args**: `packed_states` is a list of `PackedCrossCatState` (MCMC samples).

**Returns**: `Array (len(row_ids),)` with typicality in [0, 1]. Lower = more atypical.

### `batch_impute_column`

```python
batch_impute_column(rng_key, packed, data, query_col, row_ids, *, n_samples=100) -> tuple[Array, Array]
```

Impute a column for multiple rows in one JIT call. Draws posterior predictive samples per row using that row's cluster assignment.

**Returns**: `(point_estimates, confidences)`, each shape `(len(row_ids),)`.

### `batch_classify_column`

```python
batch_classify_column(packed, data, target_col, candidate_vals, row_ids) -> Array
```

Batch classification: log P(target_col=v | row) for all rows and candidate values. Double-vmapped over rows and values.

**Returns**: `Array (len(row_ids), len(candidate_vals))` with log probabilities.

### `batch_score_columns_binary`

```python
batch_score_columns_binary(packed, data, col_indices, row_id) -> Array
```

Compute P(col=1 | row) for multiple binary columns in one JIT call. Designed for inpainting: score all missing pixels at once.

**Returns**: `Array (len(col_indices),)` with P(col=1 | row) in [0, 1].

### `batch_row_similarity`

```python
batch_row_similarity(packed_states, row_ids) -> Array
```

Pairwise similarity matrix for multiple rows. Similarity is the probability that two rows share the same cluster, averaged over views and posterior states.

**Args**: `packed_states` is a list of `PackedCrossCatState`.

**Returns**: Symmetric `Array (N, N)` with similarity in [0, 1]. Diagonal is 1.0.

### `batch_predictive_cdf`

```python
batch_predictive_cdf(rng_key, packed, data, query_col, query_val, row_ids, *, n_samples=1000) -> Array
```

Posterior predictive CDF P(X <= query_val | row) for multiple rows in one JIT call.

**Returns**: `Array (len(row_ids),)` with CDF values in [0, 1].

### `batch_credible_interval`

```python
batch_credible_interval(rng_key, packed, data, query_col, row_ids, *, n_samples=1000, ci_level=0.90) -> tuple[Array, Array, Array]
```

Credible intervals for multiple rows in one JIT call. Draws posterior predictive samples per row and computes percentile-based CI.

**Returns**: `(medians, lower_bounds, upper_bounds)`, each shape `(len(row_ids),)`.
