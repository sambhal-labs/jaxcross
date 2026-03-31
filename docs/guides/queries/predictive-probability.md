# Predictive CDF & Probability

## What

Evaluate the posterior predictive probability of specific values and compute cumulative distribution functions.

## When to Use

- Scoring how likely a specific observation is
- Computing p-values and percentiles
- Model checking and calibration

## Predictive Probability

*"How likely is salary=85000 given 5 years of experience?"*

```python
from crosscat import predictive_probability
import jax.numpy as jnp

log_p = predictive_probability(
    state, data,
    query_cols=[0],
    query_vals=jnp.array([85000.0]),
    condition_cols=[1],
    condition_vals=jnp.array([5.0]),
)
print(f"Log p(salary=85000 | exp=5): {log_p:.3f}")
```

## Predictive CDF

*"What fraction of salaries are below 100k?"*

```python
from crosscat import predictive_cdf

cdf = predictive_cdf(
    key, state, data,
    query_col=0,
    query_val=jnp.array(100000.0),
)
print(f"P(salary <= 100k): {cdf:.3f}")
```

- Analytic for discrete types (BINARY, CATEGORICAL)
- Monte Carlo estimation for continuous and cyclic types

## Joint Predictive Probability

Score multiple columns jointly:

```python
from crosscat import joint_predictive_probability

log_p = joint_predictive_probability(
    state, data,
    query_cols=[0, 2],
    query_vals=jnp.array([85000.0, 0.0]),  # salary=85k, dept=eng
)
```

## Batch CDF & Credible Intervals

Vectorized over rows for production use:

```python
from crosscat import batch_predictive_cdf, batch_credible_interval
import jax.numpy as jnp

packed = pack_state(best)
row_ids = jnp.arange(data.shape[0])

# CDF for all rows at a threshold
key, subkey = jax.random.split(key)
cdfs = batch_predictive_cdf(
    subkey, packed, data,
    query_col=0,
    query_val=jnp.array(100000.0),
    row_ids=row_ids,
)
# Shape: (n_rows,) — P(salary <= 100k | row) for each row

# Credible intervals for all rows
key, subkey = jax.random.split(key)
medians, lowers, uppers = batch_credible_interval(
    subkey, packed, data, query_col=0, row_ids=row_ids
)
# Each shape: (n_rows,)
```

## Packed Versions

```python
from crosscat import (
    packed_predictive_probability,
    packed_predictive_cdf,
    packed_joint_predictive_probability,
)

log_p = packed_predictive_probability(packed, data, [0], jnp.array([85000.0]))
cdf = packed_predictive_cdf(key, packed, data, query_col=0, query_val=jnp.array(100000.0))
```

## API Reference

- [`predictive_probability`](../../api/inference.md#predictive_probability)
- [`predictive_cdf`](../../api/inference.md#predictive_cdf)
- [`joint_predictive_probability`](../../api/inference.md#joint_predictive_probability)
- [`batch_predictive_cdf`](../../api/packed-inference.md#batch_predictive_cdf)
- [`batch_credible_interval`](../../api/packed-inference.md#batch_credible_interval)
