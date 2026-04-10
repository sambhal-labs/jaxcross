# Imputation Evaluation Guide

## Holdout evaluation

The gold standard: hide known values, impute them, compare to truth.

```python
from crosscat.diagnostics import random_holdout_mask, packed_evaluate_imputation

# Hide 10% of known values
holdout_mask = random_holdout_mask(key, data, fraction=0.1)
data_masked = jnp.where(holdout_mask, jnp.nan, data)

# Evaluate
results = packed_evaluate_imputation(key, packed, data, data_masked, holdout_mask, col_types)
```

## Metrics by column type

### Continuous columns
- **MAE** (Mean Absolute Error): average |imputed - true|. Lower is better.
- **RMSE**: sqrt(mean((imputed - true)^2)). More sensitive to large errors.
- **Coverage**: fraction of true values within the credible interval.

### Categorical columns
- **Accuracy**: fraction of correctly imputed categories.
- **Top-k accuracy**: fraction where true category is in top-k imputed candidates.

### Binary columns
- **Accuracy**: fraction of correctly imputed 0/1 values.
- **F1 score**: harmonic mean of precision and recall.

## Interpreting results

| Metric | Good | Acceptable | Poor |
|--------|------|-----------|------|
| Continuous MAE | < 0.5 * std(column) | < 1.0 * std(column) | > 1.0 * std(column) |
| Categorical accuracy | > 80% | > 60% | < 60% |
| Binary accuracy | > 90% | > 75% | < 75% |
| Credible interval coverage | 90-95% (for 95% CI) | 85-100% | < 85% |

## Factors affecting quality

1. **Missing rate**: More missing data → worse imputation quality
2. **Column dependencies**: Strongly correlated columns help imputation
3. **Model convergence**: Unconverged models give poor imputations
4. **Missingness mechanism**: MCAR is easiest; MNAR can bias imputations
