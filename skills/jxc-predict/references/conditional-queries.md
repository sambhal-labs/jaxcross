# Conditional Queries Guide

## Building conditions

Conditions are specified as arrays of column indices and values:

```python
condition_cols = jnp.array([col_idx_1, col_idx_2, ...])
condition_vals = jnp.array([value_1, value_2, ...])
```

The model computes P(query | conditions) by finding which clusters are consistent with the conditions.

## Examples

### "What income do 35-year-old engineers typically have?"
```python
condition_cols = jnp.array([col_names.index("age"), col_names.index("job")])
condition_vals = jnp.array([35.0, encodings["job"]["engineer"]])

# Sample multiple values
samples = []
for _ in range(100):
    key, subkey = jax.random.split(key)
    s = packed_predictive_sample(
        subkey, packed, data,
        query_cols=jnp.array([col_names.index("income")]),
        condition_cols=condition_cols,
        condition_vals=condition_vals,
    )
    samples.append(float(s))

print(f"Income | age=35, job=engineer:")
print(f"  Median: {np.median(samples):.0f}")
print(f"  90% range: [{np.percentile(samples, 5):.0f}, {np.percentile(samples, 95):.0f}]")
```

### "Is this transaction fraudulent given amount=5000 and country=3?"
```python
condition_cols = jnp.array([col_names.index("amount"), col_names.index("country")])
condition_vals = jnp.array([5000.0, 3.0])

# P(fraud=1 | amount=5000, country=3)
log_p = packed_predictive_probability(
    packed, data,
    query_cols=jnp.array([col_names.index("fraud")]),
    query_vals=jnp.array([1.0]),
    condition_cols=condition_cols,
    condition_vals=condition_vals,
)
print(f"P(fraud | amount=5000, country=3) = {float(jnp.exp(log_p)):.3f}")
```

## Row-conditioned queries

Instead of specifying conditions manually, use an existing row as context:

```python
# "What would column 5 be for a row similar to row 42?"
log_p = packed_predictive_probability(
    packed, data,
    query_cols=jnp.array([5]),
    query_vals=jnp.array([some_value]),
    condition_row=42,  # Use all non-query columns from row 42 as conditions
)
```

## Multiple target columns

Query multiple columns jointly:
```python
# P(income=50k AND education=bachelors | age=35)
log_p_joint = packed_predictive_probability(
    packed, data,
    query_cols=jnp.array([income_col, edu_col]),
    query_vals=jnp.array([50000.0, 2.0]),
    condition_cols=jnp.array([age_col]),
    condition_vals=jnp.array([35.0]),
)
```
