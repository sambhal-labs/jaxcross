# Imputation Strategies

## Single imputation (default)
- One imputed value per missing cell
- Fast, simple
- Underestimates uncertainty

## Multiple imputation
Generate M imputed datasets, analyze each, combine results:
```python
M = 10
imputed_datasets = []
for m in range(M):
    key_m = jax.random.key(m)
    imputed = jnp.array(data)
    for col_j in columns_with_missing:
        vals, _ = batch_impute_column(key_m, packed, data, col_j, missing_rows)
        imputed = imputed.at[missing_rows, col_j].set(vals)
    imputed_datasets.append(imputed)
```

## Multi-chain imputation
Average across chains for Bayesian model averaging:
```python
value, confidence = multi_chain_impute_and_confidence(
    key, all_chains, data, query_row=r, query_col=c
)
```
Better uncertainty quantification than single-chain.

## Credible interval imputation
Get not just a point estimate but a range:
```python
from crosscat import batch_credible_interval
lower, upper = batch_credible_interval(
    key, packed, data, query_col=c, row_ids=missing_rows,
    alpha=0.05  # 95% credible interval
)
```
