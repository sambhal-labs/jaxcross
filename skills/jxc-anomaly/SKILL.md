---
name: jxc-anomaly
description: Detect and rank anomalous rows in your dataset using a trained jaxcross model. Scores all rows by anomalousness, applies threshold selection, drills down to identify which columns drive each anomaly, and exports flagged records. Use after /jxc-model or /jxc-quickstart.
version: "1.0.0"
license: Apache-2.0
---

# Anomaly Detection

Score, rank, and explain anomalous rows using a trained CrossCat model.

Usage: `/jxc-anomaly [--model model.jxc] [--data data.arrow] [--threshold top1pct]`

Examples:
- `/jxc-anomaly` (uses model.jxc and data from current session)
- `/jxc-anomaly --model checkpoints/sweep_400.jxc --data data/prepared.arrow`

## Step 1: Load model and data

```python
import jax
import jax.numpy as jnp
from crosscat import load_packed_state
from crosscat.data_utils import load_data

# Load trained model
packed, col_types = load_packed_state("model.jxc")

# Load data
data, col_names, _ = load_data("data/prepared.arrow")
# Or use the data array from the current session

n_rows, n_cols = data.shape
print(f"Model loaded. Data: {n_rows} rows x {n_cols} columns")
```

## Step 2: Score all rows

```python
from crosscat import batch_anomaly_score

row_ids = jnp.arange(n_rows)
scores = batch_anomaly_score(packed, data, row_ids)

print(f"Anomaly score statistics:")
print(f"  Mean: {float(jnp.mean(scores)):.3f}")
print(f"  Std:  {float(jnp.std(scores)):.3f}")
print(f"  Min:  {float(jnp.min(scores)):.3f}")
print(f"  Max:  {float(jnp.max(scores)):.3f}")
```

**How anomaly scores work:** Each row is scored by how unlikely it is under the model. Higher scores = more anomalous. The score is the negative log-probability of the row under the posterior predictive distribution, averaged across views.

## Step 3: Select threshold

Choose one approach based on the use case:

### Percentile threshold (recommended for exploration)
```python
# Flag top 1% most anomalous
threshold = float(jnp.percentile(scores, 99))
flagged_mask = scores > threshold
n_flagged = int(flagged_mask.sum())
print(f"Threshold (top 1%): {threshold:.3f}")
print(f"Flagged rows: {n_flagged}")
```

### Statistical threshold (recommended for production)
```python
# Flag rows > 2 standard deviations above mean
mean_score = float(jnp.mean(scores))
std_score = float(jnp.std(scores))
threshold = mean_score + 2 * std_score
flagged_mask = scores > threshold
n_flagged = int(flagged_mask.sum())
print(f"Threshold (mean + 2*std): {threshold:.3f}")
print(f"Flagged rows: {n_flagged}")
```

### Fixed count
```python
# Flag top N rows
N = 50
top_indices = jnp.argsort(-scores)[:N]
print(f"Top {N} anomalous rows: {[int(i) for i in top_indices]}")
```

See [threshold-selection.md](references/threshold-selection.md) for advanced methods.

## Step 4: Per-column drilldown

For each flagged row, identify which columns contribute most to its anomalousness:

```python
from crosscat import packed_column_typicality, packed_predictive_probability

flagged_indices = jnp.where(flagged_mask)[0]

print("\nPer-column anomaly attribution for flagged rows:")
for idx in flagged_indices[:10]:  # Top 10
    idx_int = int(idx)
    row_score = float(scores[idx])
    
    # Score each column individually
    col_scores = []
    for col_j in range(n_cols):
        val = data[idx_int, col_j]
        if jnp.isnan(val):
            continue
        
        # Predictive probability of this value given the row's cluster
        log_p = packed_predictive_probability(
            packed, data,
            query_cols=jnp.array([col_j]),
            query_vals=jnp.array([val]),
            condition_row=idx_int,
        )
        col_scores.append((col_names[col_j], float(val), float(log_p)))
    
    # Sort by most surprising (lowest log_p)
    col_scores.sort(key=lambda x: x[2])
    
    print(f"\nRow {idx_int} (score={row_score:.3f}):")
    for name, val, lp in col_scores[:5]:
        print(f"  {name}={val:.2f} (log_p={lp:.2f})")
```

See [drilldown-guide.md](references/drilldown-guide.md) for efficient batch drilldown.

## Step 5: Multi-chain robustness (optional)

If multiple chains are available, average anomaly scores across chains for more robust estimates:

```python
from crosscat import multi_chain_anomaly_score, unbatch_packed_states

# If you have all_chains from multi-chain training:
# all_chains = unbatch_packed_states(batched, N_CHAINS)

key = jax.random.key(99)
for idx in flagged_indices[:5]:
    mc_score = multi_chain_anomaly_score(
        key, all_chains, data, query_row=int(idx)
    )
    print(f"Row {int(idx)}: single-chain={float(scores[idx]):.3f}, "
          f"multi-chain={float(mc_score):.3f}")
```

## Step 6: Export results

```python
import pandas as pd

# Create results DataFrame
results = pd.DataFrame({
    "row_id": range(n_rows),
    "anomaly_score": [float(s) for s in scores],
    "is_flagged": [bool(flagged_mask[i]) for i in range(n_rows)],
})

# Add original data columns
for j, name in enumerate(col_names):
    results[name] = [float(data[i, j]) for i in range(n_rows)]

# Sort by score
results = results.sort_values("anomaly_score", ascending=False)

# Export
results.to_csv("anomaly_results.csv", index=False)
flagged_only = results[results.is_flagged]
flagged_only.to_csv("flagged_rows.csv", index=False)

print(f"\nExported:")
print(f"  All rows: anomaly_results.csv ({len(results)} rows)")
print(f"  Flagged only: flagged_rows.csv ({len(flagged_only)} rows)")
```

## Common Pitfalls

- **Anomaly scores are relative**: They depend on the model and data distribution. Compare within the same dataset, not across datasets.
- **Missing values aren't anomalous**: NaN values are simply excluded from scoring. A row with many NaN values may have a misleadingly low anomaly score because fewer columns contribute.
- **Model quality matters**: Anomaly scores from an unconverged model (Rhat > 1.2) are unreliable. Train with `/jxc-model` first.
- **Batch scoring is much faster**: Use `batch_anomaly_score()` for all rows at once, not a loop over `packed_anomaly_score()`.
