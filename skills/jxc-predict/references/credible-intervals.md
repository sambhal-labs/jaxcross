# Credible Intervals Guide

## What are Bayesian credible intervals?

A 95% credible interval [L, U] means: given the data and model, there is a 95% posterior probability that the true value lies in [L, U].

This is different from a frequentist confidence interval:
- **Credible interval**: "95% probability the value is in this range"
- **Confidence interval**: "If we repeated the experiment, 95% of intervals would contain the true value"

## Using credible intervals

```python
from crosscat import batch_credible_interval

key = jax.random.key(42)
lower, upper = batch_credible_interval(
    key, packed, data,
    query_col=target_col,
    row_ids=row_ids,
    alpha=0.05,  # 95% CI (1 - alpha)
)
```

### Common alpha values
| alpha | Credible interval | Use case |
|-------|-------------------|----------|
| 0.01 | 99% | Conservative, wide intervals |
| 0.05 | 95% | Standard, most common |
| 0.10 | 90% | Moderate uncertainty |
| 0.32 | 68% | Roughly ±1 sigma equivalent |

## Interpreting width

- **Narrow interval**: Model is confident about the prediction
- **Wide interval**: High uncertainty — could mean:
  - The column has high variance
  - The row's cluster is small (less data to learn from)
  - The column is weakly related to other columns

## Coverage check

Verify that credible intervals are well-calibrated:
```python
# For rows with known values
actual = data[:, target_col]
coverage = jnp.mean((actual >= lower) & (actual <= upper))
print(f"Empirical coverage: {float(coverage):.1%}")
# Should be close to 1-alpha (e.g., ~95% for alpha=0.05)
```

## Only for continuous columns

Credible intervals are meaningful for CONTINUOUS and CYCLIC columns. For CATEGORICAL, ORDINAL, and BINARY columns, use:
- `batch_classify_column()` for the most likely value
- `batch_predictive_probability()` for the probability of each category
