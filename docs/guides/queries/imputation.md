# Imputation

## What

Predict missing values with confidence scores using the posterior predictive distribution. CrossCat provides principled Bayesian imputation that accounts for uncertainty.

## When to Use

- Filling in missing data points
- Completing partial observations
- Evaluating data quality (confidence of predictions)

## Basic Imputation

```python
from crosscat import impute_and_confidence
import jax.numpy as jnp

key, subkey = jax.random.split(key)
value, confidence = impute_and_confidence(
    subkey, state, data, query_col=0,  # impute column 0
)
print(f"Imputed value: {value:.2f} (confidence: {confidence:.2f})")
```

## Conditional Imputation

Condition on known values for better predictions:

```python
value, confidence = impute_and_confidence(
    subkey, state, data, query_col=0,
    condition_cols=[1, 2],
    condition_vals=jnp.array([5.0, 0.0]),  # years=5, dept=eng
)
```

## How Confidence Works

- **Continuous columns**: Median of samples; confidence = `1 / (1 + std)`
- **Discrete columns**: Mode of samples; confidence = mode frequency

## Sample and Insert

Complete a partial row and insert it into the state in one step:

```python
from crosscat import sample_and_insert

# Partial row with NaN for missing values
partial_row = jnp.array([jnp.nan, 5.0, 0.0, jnp.nan])

key, subkey = jax.random.split(key)
updated_state, updated_data, completed_row = sample_and_insert(
    subkey, state, data, partial_row
)
print(f"Completed row: {completed_row}")
```

## Evaluate Imputation Accuracy

Test how well the model imputes by holding out known values:

```python
from crosscat.diagnostics import random_holdout_mask, evaluate_imputation

# Create a random holdout mask
key, subkey = jax.random.split(key)
mask = random_holdout_mask(subkey, data.shape[0], data.shape[1], holdout_fraction=0.1)

# Evaluate imputation quality
metrics = evaluate_imputation(state, data, mask, col_types, rng_key=key)
# Returns: MAE for continuous, accuracy for discrete, log-likelihood
```

## Packed Versions

```python
from crosscat import packed_impute_and_confidence, packed_sample_and_insert

value, conf = packed_impute_and_confidence(key, packed, data, query_col=0)
packed_new, data_new, row = packed_sample_and_insert(key, packed, data, partial_row)
```

## Packed Imputation Evaluation

Use `packed_evaluate_imputation` for significantly faster held-out evaluation on packed state:

```python
from crosscat.diagnostics import random_holdout_mask, packed_evaluate_imputation

mask = random_holdout_mask(key, data.shape[0], data.shape[1], holdout_fraction=0.1)
metrics = packed_evaluate_imputation(packed, data, mask, col_types, rng_key=key)
```

## API Reference

- [`impute_and_confidence`](../../api/inference.md#impute_and_confidence)
- [`sample_and_insert`](../../api/inference.md#sample_and_insert)
- [`evaluate_imputation`](../../api/diagnostics.md#evaluate_imputation)
- [`packed_evaluate_imputation`](../../api/diagnostics.md#packed_evaluate_imputation)
