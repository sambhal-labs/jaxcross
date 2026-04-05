# Missing Data Imputation

Fill missing values with Bayesian posterior predictive sampling, complete with confidence scores. No separate imputation pipeline needed.

## The Challenge

Real-world datasets are rarely complete. Standard approaches either drop incomplete rows (losing data) or use simple strategies like mean/mode imputation (ignoring structure). CrossCat imputes missing values by learning the full joint distribution and sampling from the posterior predictive.

## Why CrossCat Fits

- **Joint modeling** — imputation uses the full data structure (cross-column relationships and row clusters), not just per-column statistics
- **Uncertainty quantification** — each imputed value comes with a confidence score
- **Mixed types** — impute continuous, categorical, binary, ordinal, and cyclic values with the appropriate model
- **NaN transparency** — missing values (NaN) are handled automatically during inference — no preprocessing

## Workflow

```python
import jax
import jax.numpy as jnp
from crosscat import initialize, impute_and_confidence
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state

# Data with missing values (NaN)
# CrossCat handles NaN transparently during inference
result = initialize(jax.random.key(42), data, col_types)
state = result.state
packed = pack_state(state)
packed = packed_gibbs_sweep(jax.random.key(1), packed, data, n_sweeps=100)
state = unpack_state(packed, col_types, data=data)

# Impute missing values for a specific row and column
row_id = 5    # Row with missing data
query_col = 2 # Column to impute

value, confidence = impute_and_confidence(
    jax.random.key(2), state, data, query_col=query_col, row_id=row_id
)
print(f"Imputed value: {value:.2f}, Confidence: {confidence:.3f}")
```

## Packed Version (Faster)

```python
from crosscat import packed_impute_and_confidence

value, confidence = packed_impute_and_confidence(
    jax.random.key(2), packed, data, query_col=query_col, row_id=row_id
)

# Multi-chain for more robust imputation
from crosscat import multi_chain_impute_and_confidence
value, confidence = multi_chain_impute_and_confidence(
    jax.random.key(2), packed_states, data, query_col=query_col, row_id=row_id
)
```

## Credible Intervals

For continuous columns, get a Bayesian credible interval instead of a point estimate:

```python
from crosscat import credible_interval

median, lower, upper = credible_interval(
    jax.random.key(3), state, data,
    query_col=query_col,
    ci_level=0.95  # 95% credible interval
)
print(f"95% CI: [{lower:.2f}, {upper:.2f}], Median: {median:.2f}")
```

## Batch Imputation

Impute all missing values in the dataset:

```python
import numpy as np

imputed_data = np.array(data)
confidences = np.zeros_like(data)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if np.isnan(data[i, j]):
            val, conf = impute_and_confidence(
                jax.random.key(i * data.shape[1] + j),
                state, data, query_col=j, row_id=i
            )
            imputed_data[i, j] = val
            confidences[i, j] = conf
```

## Evaluating Imputation Quality

Use the built-in holdout evaluation to assess imputation accuracy:

```python
from crosscat import random_holdout_mask, evaluate_imputation

# Create a random holdout mask (10% of observed values)
mask = random_holdout_mask(jax.random.key(99), data.shape[0], data.shape[1], holdout_fraction=0.1)

# Evaluate imputation quality
metrics = evaluate_imputation(state, data, mask, col_types)
# Returns: MAE (continuous), accuracy (categorical/binary), log-likelihood
```

## Confidence Scores

The confidence score is an inverse-variance measure in [0, 1]:

| Confidence | Interpretation |
|------------|----------------|
| > 0.8 | High — strong cluster structure supports this imputation |
| 0.5 – 0.8 | Medium — reasonable estimate with some uncertainty |
| < 0.5 | Low — multiple plausible values; consider the credible interval |

Low confidence doesn't mean the imputation is wrong — it means the model sees multiple plausible values. Use `credible_interval` to see the range.

## Tips

- **NaN is native** — just load your data as-is. No need to drop rows, fill with means, or preprocess
- **Run enough sweeps** — imputation quality depends on model quality. 100+ sweeps recommended
- **Use multi-chain** — averaging across chains gives more robust imputation
- **Check confidence** — low-confidence imputations may benefit from collecting more data
- **Holdout evaluation** — use `evaluate_imputation` to validate before trusting imputations in production
