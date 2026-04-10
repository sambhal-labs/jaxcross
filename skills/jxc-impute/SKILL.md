---
name: jxc-impute
description: Impute missing values in your dataset using a trained jaxcross model with Bayesian confidence scores. Supports batch imputation across all columns, quality evaluation via held-out masking, and multi-chain robustness. Use after /jxc-model when your data has NaN values.
version: "1.0.0"
license: Apache-2.0
---

# Missing Data Imputation

Fill missing values using the CrossCat posterior predictive, with per-value confidence scores.

Usage: `/jxc-impute [--model model.jxc] [--data data.arrow] [--evaluate]`

Examples:
- `/jxc-impute`
- `/jxc-impute --model model.jxc --evaluate`

## Step 1: Load model and identify missing values

```python
import jax
import jax.numpy as jnp
from crosscat import load_packed_state
from crosscat.data_utils import load_data

packed, col_types = load_packed_state("model.jxc")
data, col_names, _ = load_data("data/prepared.arrow")

# Identify missing values
nan_mask = jnp.isnan(data)
missing_per_col = nan_mask.sum(axis=0)

print("Missing data summary:")
for j, name in enumerate(col_names):
    count = int(missing_per_col[j])
    if count > 0:
        pct = 100 * count / data.shape[0]
        print(f"  {name}: {count} missing ({pct:.1f}%)")

total_missing = int(nan_mask.sum())
print(f"\nTotal missing values: {total_missing}")
```

## Step 2: Batch impute per column

```python
from crosscat import batch_impute_column

key = jax.random.key(42)
imputed_data = jnp.array(data)  # Copy to fill

for col_j in range(data.shape[1]):
    missing_rows = jnp.where(nan_mask[:, col_j])[0]
    if len(missing_rows) == 0:
        continue
    
    key, subkey = jax.random.split(key)
    values, confidences = batch_impute_column(
        subkey, packed, data,
        query_col=col_j,
        row_ids=missing_rows,
    )
    
    # Fill imputed values
    imputed_data = imputed_data.at[missing_rows, col_j].set(values)
    
    # Report
    mean_conf = float(jnp.mean(confidences))
    low_conf = int((confidences < 0.5).sum())
    print(f"  {col_names[col_j]}: imputed {len(missing_rows)} values "
          f"(mean confidence={mean_conf:.2f}, {low_conf} low-confidence)")
```

**How confidence works:** Confidence is derived from the posterior predictive distribution. High confidence means the model is fairly certain about the imputed value (the distribution is concentrated). Low confidence means the distribution is spread out — the true value could be far from the imputation.

## Step 3: Quality evaluation (optional but recommended)

Evaluate imputation quality by hiding known values and checking recovery:

```python
from crosscat.diagnostics import random_holdout_mask, packed_evaluate_imputation

# Create a holdout mask (hide 10% of known values)
key, subkey = jax.random.split(key)
holdout_mask = random_holdout_mask(subkey, data, fraction=0.1)

# Mask the holdout values
data_masked = jnp.where(holdout_mask, jnp.nan, data)

# Impute the masked values
key, subkey = jax.random.split(key)
eval_results = packed_evaluate_imputation(
    subkey, packed, data, data_masked, holdout_mask, col_types
)

print("\nImputation quality evaluation (10% holdout):")
print(f"  MAE (continuous): {eval_results.get('mae', 'N/A')}")
print(f"  Accuracy (categorical): {eval_results.get('accuracy', 'N/A')}")
```

See [evaluation-guide.md](references/evaluation-guide.md) for interpreting evaluation results.

## Step 4: Multi-chain imputation (optional)

For more robust imputations, average across multiple chains:

```python
from crosscat import multi_chain_impute_and_confidence

# If all_chains available from multi-chain training:
key, subkey = jax.random.split(key)
value, confidence = multi_chain_impute_and_confidence(
    subkey, all_chains, data,
    query_row=42,
    query_col=3,
)
print(f"Multi-chain imputation: value={float(value):.3f}, "
      f"confidence={float(confidence):.3f}")
```

Multi-chain imputation averages across posterior samples from different chains, providing better uncertainty estimates.

## Step 5: Handle low-confidence imputations

```python
# Identify low-confidence imputations
LOW_CONF_THRESHOLD = 0.3

for col_j in range(data.shape[1]):
    missing_rows = jnp.where(nan_mask[:, col_j])[0]
    if len(missing_rows) == 0:
        continue
    
    key, subkey = jax.random.split(key)
    values, confidences = batch_impute_column(
        subkey, packed, data, query_col=col_j, row_ids=missing_rows,
    )
    
    low_conf_rows = missing_rows[confidences < LOW_CONF_THRESHOLD]
    if len(low_conf_rows) > 0:
        print(f"WARNING: {col_names[col_j]} has {len(low_conf_rows)} "
              f"low-confidence imputations (conf < {LOW_CONF_THRESHOLD})")
        print(f"  Row IDs: {[int(r) for r in low_conf_rows[:10]]}")
        print(f"  Consider: keeping as NaN, collecting more data, or manual review")
```

## Step 6: Export completed dataset

```python
import pandas as pd
from crosscat.data_utils import save_data

# Save imputed data as Arrow
save_data(imputed_data, "data/imputed.arrow",
          column_names=col_names, column_types=col_types)

# Also export as CSV for human review
df = pd.DataFrame(
    [[float(imputed_data[i, j]) for j in range(data.shape[1])]
     for i in range(data.shape[0])],
    columns=col_names,
)
df.to_csv("data/imputed.csv", index=False)

# Export imputation metadata
import json
meta = {
    "total_imputed": total_missing,
    "per_column": {
        col_names[j]: int(missing_per_col[j])
        for j in range(data.shape[1])
        if int(missing_per_col[j]) > 0
    },
}
with open("data/imputation_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nExported:")
print(f"  Imputed data: data/imputed.arrow")
print(f"  CSV: data/imputed.csv")
print(f"  Metadata: data/imputation_metadata.json")
```

## Common Pitfalls

- **Don't fill NaN before training**: jaxcross handles missing data natively during Gibbs sampling. Impute AFTER training, not before.
- **Confidence depends on model quality**: An unconverged model gives unreliable confidence scores. Use `/jxc-model` with convergence monitoring first.
- **Categorical imputations are integers**: For categorical columns, the imputed value is the integer category code. Use `encodings.json` from `/data-transform` for reverse lookup.
- **Multiple imputation**: For statistical analyses that need proper uncertainty quantification, generate multiple imputations by calling `batch_impute_column` with different random keys.

See [imputation-strategies.md](references/imputation-strategies.md) for advanced patterns.
