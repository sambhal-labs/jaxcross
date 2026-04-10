---
name: jxc-predict
description: Make predictions using a trained jaxcross model. Classify column values, compute predictive probabilities, generate credible intervals, sample from the posterior predictive, and answer conditional queries ("given X, what is Y?"). Use after /jxc-model.
version: "1.0.0"
license: Apache-2.0
---

# Predictive Queries

Answer "what will this value be?" using the CrossCat posterior predictive distribution.

Usage: `/jxc-predict [--model model.jxc] [--target column_name]`

Examples:
- `/jxc-predict --target income`
- `/jxc-predict --target churn --given "age=35, tenure=2"`

## Step 1: Load model

```python
import jax
import jax.numpy as jnp
from crosscat import load_packed_state
from crosscat.data_utils import load_data

packed, col_types = load_packed_state("model.jxc")
data, col_names, _ = load_data("data/prepared.arrow")

target_col = col_names.index("<target_column_name>")
print(f"Target: {col_names[target_col]} (type: {col_types[target_col].name})")
```

## Step 2: Classification (most likely value)

For categorical/binary columns — predict the most likely category:

```python
from crosscat import batch_classify_column

key = jax.random.key(42)
row_ids = jnp.arange(data.shape[0])

predictions = batch_classify_column(
    key, packed, data,
    query_col=target_col,
    row_ids=row_ids,
)

print(f"Classification results for '{col_names[target_col]}':")
print(f"  Predictions shape: {predictions.shape}")
print(f"  Unique predicted values: {jnp.unique(predictions).tolist()}")
```

For multi-chain classification:
```python
from crosscat import multi_chain_classify_column

key, subkey = jax.random.split(key)
pred = multi_chain_classify_column(
    subkey, all_chains, data,
    query_col=target_col,
    query_row=42,
)
print(f"Row 42 predicted class: {float(pred)}")
```

## Step 3: Predictive probability (how likely is a specific value?)

```python
from crosscat import batch_predictive_probability

# Score how likely specific values are
query_vals = jnp.array([0.5])  # The value to evaluate
log_probs = batch_predictive_probability(
    packed, data,
    query_cols=jnp.array([target_col]),
    query_vals=query_vals,
    row_ids=row_ids,
)

print(f"Log P({col_names[target_col]}=0.5) per row:")
print(f"  Mean: {float(jnp.mean(log_probs)):.3f}")
print(f"  Most likely row: {int(jnp.argmax(log_probs))}")
```

## Step 4: Credible intervals (continuous columns)

Get Bayesian credible intervals — ranges where the true value likely falls:

```python
from crosscat import batch_credible_interval

key, subkey = jax.random.split(key)
lower, upper = batch_credible_interval(
    subkey, packed, data,
    query_col=target_col,
    row_ids=row_ids,
    alpha=0.05,  # 95% credible interval
)

print(f"95% credible intervals for '{col_names[target_col]}':")
for i in range(min(10, len(row_ids))):
    actual = float(data[i, target_col])
    lo, hi = float(lower[i]), float(upper[i])
    in_ci = "yes" if lo <= actual <= hi else "NO"
    print(f"  Row {i}: [{lo:.2f}, {hi:.2f}] actual={actual:.2f} in_CI={in_ci}")
```

For multi-chain credible intervals:
```python
from crosscat import multi_chain_credible_interval
key, subkey = jax.random.split(key)
lo, hi = multi_chain_credible_interval(
    subkey, all_chains, data,
    query_col=target_col, query_row=42, alpha=0.05
)
```

See [credible-intervals.md](references/credible-intervals.md) for interpretation.

## Step 5: Posterior predictive sampling

Draw samples from the model's predicted distribution:

```python
from crosscat import batch_predictive_sample

key, subkey = jax.random.split(key)
samples = batch_predictive_sample(
    subkey, packed, data,
    query_cols=jnp.array([target_col]),
    row_ids=row_ids,
)

print(f"Posterior predictive samples for '{col_names[target_col]}':")
print(f"  Shape: {samples.shape}")
print(f"  Mean: {float(jnp.mean(samples)):.3f}")
print(f"  Std: {float(jnp.std(samples)):.3f}")
```

## Step 6: Conditional queries ("given X, predict Y")

The most powerful query type — predict a target given specific values of other columns:

```python
from crosscat import packed_predictive_probability, packed_predictive_sample

# "Given age=35 and tenure=2, what is the probability of income=50000?"
condition_cols = jnp.array([
    col_names.index("age"),
    col_names.index("tenure"),
])
condition_vals = jnp.array([35.0, 2.0])

# Probability of a specific value
log_p = packed_predictive_probability(
    packed, data,
    query_cols=jnp.array([target_col]),
    query_vals=jnp.array([50000.0]),
    condition_cols=condition_cols,
    condition_vals=condition_vals,
)
print(f"P(income=50000 | age=35, tenure=2) = exp({float(log_p):.3f})")

# Sample from conditional distribution
key, subkey = jax.random.split(key)
sample = packed_predictive_sample(
    subkey, packed, data,
    query_cols=jnp.array([target_col]),
    condition_cols=condition_cols,
    condition_vals=condition_vals,
)
print(f"Sample from P(income | age=35, tenure=2) = {float(sample):.0f}")
```

See [conditional-queries.md](references/conditional-queries.md) for building complex conditions.

## Step 7: Joint predictions

Predict multiple columns simultaneously:

```python
from crosscat import batch_joint_predictive_probability

# Joint probability of multiple values
query_cols = jnp.array([
    col_names.index("income"),
    col_names.index("education"),
])
query_vals = jnp.array([50000.0, 2.0])  # income=50k, education=bachelors

log_p_joint = batch_joint_predictive_probability(
    packed, data,
    query_cols=query_cols,
    query_vals=query_vals,
    row_ids=row_ids,
)
```

## Common Pitfalls

- **Log probabilities**: All probability functions return log-scale values. Use `jnp.exp(log_p)` for probabilities, but beware of underflow for very unlikely values.
- **Conditional queries need valid columns**: `condition_cols` must not overlap with `query_cols`.
- **Categorical predictions are integers**: The predicted value is the integer category code, not the original string label. Use `encodings.json` for reverse lookup.
- **Credible intervals are for continuous columns only**: For categorical columns, use classification or predictive probability instead.
