# Conditional Sampling

## What

Draw samples from the posterior predictive distribution, optionally conditioned on observed values. Also compute credible intervals and joint probabilities.

## When to Use

- Predicting values for unobserved features
- Generating "what-if" scenarios
- Quantifying uncertainty via credible intervals

## Conditional Sampling

*"What salary would we expect given 5 years of experience?"*

```python
from crosscat import predictive_sample
import jax.numpy as jnp

key, subkey = jax.random.split(key)
samples = predictive_sample(
    subkey, state, data,
    query_cols=[0],                      # salary (column index)
    condition_cols=[1],                   # years_experience
    condition_vals=jnp.array([5.0]),
    n_samples=1000,
)

print(f"Expected salary: {jnp.median(samples[:, 0]):.0f}")
print(f"90% CI: [{jnp.percentile(samples[:, 0], 5):.0f}, "
      f"{jnp.percentile(samples[:, 0], 95):.0f}]")
```

## Unconditional Sampling

```python
# Sample from the marginal distribution of column 0
samples = predictive_sample(key, state, data, query_cols=[0], n_samples=1000)
```

## Multi-Column Sampling

```python
# Sample salary AND department jointly
samples = predictive_sample(
    key, state, data,
    query_cols=[0, 2],  # salary and department
    condition_cols=[1],
    condition_vals=jnp.array([5.0]),
    n_samples=1000,
)
# samples.shape == (1000, 2)
```

## Credible Intervals

```python
from crosscat import credible_interval

median, lower, upper = credible_interval(
    key, state, data,
    query_col=0,                         # salary
    condition_cols=[1],
    condition_vals=jnp.array([5.0]),
    ci_level=0.90,                       # 90% interval
)
print(f"Salary: {median:.0f} [{lower:.0f}, {upper:.0f}]")
```

## Predictive Probability

*"How likely is this specific value?"*

```python
from crosscat import predictive_probability

log_p = predictive_probability(
    state, data,
    query_cols=[0],
    query_vals=jnp.array([85000.0]),
    condition_cols=[1],
    condition_vals=jnp.array([5.0]),
)
print(f"Log p(salary=85000 | exp=5): {log_p:.3f}")
```

## Joint Predictive Probability

```python
from crosscat import joint_predictive_probability

log_p = joint_predictive_probability(
    state, data,
    query_cols=[0, 2],
    query_vals=jnp.array([85000.0, 0.0]),  # salary=85k, dept=eng
)
```

## Packed State Versions

All functions have packed equivalents:

```python
from crosscat import packed_predictive_sample, packed_credible_interval

samples = packed_predictive_sample(key, packed, data, query_cols=[0], n_samples=1000)
median, lo, hi = packed_credible_interval(key, packed, data, query_col=0)
```

## API Reference

- [`predictive_sample`](../../api/inference.md#predictive_sample)
- [`credible_interval`](../../api/inference.md#credible_interval)
- [`predictive_probability`](../../api/inference.md#predictive_probability)
- [`joint_predictive_probability`](../../api/inference.md#joint_predictive_probability)
