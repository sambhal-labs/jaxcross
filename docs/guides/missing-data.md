# Missing Data Handling

## What

CrossCat handles missing data transparently. NaN values are skipped during sufficient statistic computation, and all inference queries work seamlessly with incomplete data.

## When to Use

You don't need to do anything special — missing data just works. This page explains how.

## Representing Missing Data

Missing values are represented as `NaN`:

```python
import jax.numpy as jnp

# NaN in CSV files is loaded automatically
data, col_names = read_csv("data_with_gaps.csv")

# Manual NaN insertion
data = data.at[5, 2].set(jnp.nan)

# Inject random missing data for testing
from crosscat.synthetic import add_missing_data
data_missing = add_missing_data(key, data, missing_fraction=0.15)
```

## How NaN Flows Through the Pipeline

1. **Sufficient statistics**: NaN values are filtered before accumulation. A cluster with 10 rows but 2 NaN values for a column has `count=8` for that column's statistics.

2. **Row scoring**: When scoring a row for cluster assignment, NaN columns contribute 0 to the log-likelihood (they're skipped).

3. **Posterior queries**: NaN conditioning values are skipped. Queries on NaN columns use the full marginal predictive.

4. **Imputation**: `impute_and_confidence` naturally fills in NaN values using the posterior predictive.

## No Preprocessing Needed

Unlike many ML methods, you do **not** need to:

- Drop rows with missing values
- Fill NaN with means/medians before inference
- Use special missing-data indicators

Just pass your data as-is.

## Tips

- If an entire column is NaN, the model still works but that column provides no information
- Missing data fractions up to ~30% work well; higher fractions degrade inference quality
- Use `evaluate_imputation` to measure how well the model handles missing data

## API Reference

- [`add_missing_data`](../api/synthetic.md#add_missing_data)
- [`impute_and_confidence`](../api/inference.md#impute_and_confidence)
