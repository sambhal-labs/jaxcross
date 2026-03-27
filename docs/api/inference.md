# Inference Queries

::: crosscat.inference
    options:
      show_source: false

## Overview

Posterior predictive queries on unpacked `CrossCatState`. For packed state equivalents, see [Packed Inference](packed-inference.md).

---

## `predictive_probability`

```python
predictive_probability(
    state, data, query_cols, query_vals, *,
    condition_cols=None, condition_vals=None, row_id=None
) -> Array
```

Conditional predictive probability: p(query | conditions, state).

**Returns**: Scalar log probability.

## `predictive_sample`

```python
predictive_sample(
    rng_key, state, data, query_cols, *,
    condition_cols=None, condition_vals=None,
    n_samples=1000, row_id=None
) -> Array
```

Draw samples from the posterior predictive.

**Returns**: `Array (n_samples, len(query_cols))`.

## `credible_interval`

```python
credible_interval(
    rng_key, state, data, query_col, *,
    condition_cols=None, condition_vals=None,
    n_samples=1000, ci_level=0.90
) -> tuple[Array, Array, Array]
```

Compute a credible interval for a column via posterior predictive sampling.

**Returns**: `(median, lower, upper)`.

## `joint_predictive_probability`

```python
joint_predictive_probability(
    state, data, query_cols, query_vals, *,
    condition_cols=None, condition_vals=None
) -> Array
```

Joint predictive probability over multiple query columns.

**Returns**: Scalar log probability.

## `predictive_cdf`

```python
predictive_cdf(
    rng_key, state, data, query_col, query_val, *,
    condition_cols=None, condition_vals=None,
    row_id=None, n_samples=10000
) -> Array
```

Posterior predictive CDF: P(X <= value). Analytic for discrete types, Monte Carlo for continuous/cyclic.

**Returns**: Scalar in [0, 1].

---

## Dependence & Structure

### `dependence_probability`

```python
dependence_probability(states, col_i, col_j) -> Array
```

Posterior probability that two columns are dependent (share a view). Fraction of posterior samples where columns share a view. This is the paper's primary exploratory statistic (Mansinghka et al. 2016, Section 2.5.2).

**Returns**: Scalar in [0, 1].

### `dependence_matrix`

```python
dependence_matrix(states, columns=None) -> Array
```

Full dependence probability matrix (Z-matrix). `Z[i,j]` = fraction of posterior samples where columns i and j share a view. Diagonal is always 1.0. Symmetric.

**Returns**: `Array (n_cols, n_cols)` with values in [0, 1].

### `mutual_information`

```python
mutual_information(states, col_i, col_j, *, n_samples=1000, rng_key=None) -> tuple[Array, Array]
```

Estimate mutual information between two columns, averaged over posterior samples.

**Returns**: `(mi, linfoot_correlation)`.

### `conditional_entropy`

```python
conditional_entropy(rng_key, states, data, target_col, given_cols, *, n_samples=500) -> Array
```

Estimate H(target | given) via Monte Carlo.

**Returns**: Scalar conditional entropy (nats).

---

## Row & Column Analysis

### `row_similarity`

```python
row_similarity(states, row_a, row_b, *, target_columns=None) -> Array
```

Probability that two rows are in the same cluster, averaged over views and posterior samples. Optionally restrict to views containing specific columns.

**Returns**: Scalar in [0, 1].

### `row_typicality`

```python
row_typicality(states, row_id) -> Array
```

Structural typicality score for a row (low = anomalous).

**Returns**: Scalar in [0, 1].

### `column_typicality`

```python
column_typicality(states, col_id) -> Array
```

Structural typicality score for a column (consistency of view assignment across samples).

**Returns**: Scalar in [0, 1].

### `predictive_anomalousness`

```python
predictive_anomalousness(rng_key, state, data, query_row, *, n_samples=1000) -> Array
```

Predictive anomaly score for a row (high = anomalous). Compares predictive probability of the row's values against Monte Carlo samples.

**Returns**: Scalar in [0, 1].

---

## Imputation

### `impute_and_confidence`

```python
impute_and_confidence(
    rng_key, state, data, query_col, *,
    condition_cols=None, condition_vals=None,
    row_id=None, n_samples=1000
) -> tuple[Array, Array]
```

Impute a missing value with confidence. Continuous: median + IQR-based confidence. Discrete: mode + mode frequency.

**Returns**: `(point_estimate, confidence_score)`.

### `sample_and_insert`

```python
sample_and_insert(rng_key, state, data, partial_row) -> tuple[CrossCatState, Array, Array]
```

Fill NaN entries via predictive sampling, then insert the completed row into the state.

**Returns**: `(updated_state, updated_data, completed_row)`.
