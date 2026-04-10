# Per-Column Anomaly Drilldown Guide

## Efficient batch drilldown

For flagged rows, identify which columns drive the anomaly by computing per-column predictive probabilities:

```python
from crosscat import batch_predictive_probability

flagged_indices = jnp.where(flagged_mask)[0]

# For each flagged row, score each column
drilldown_results = {}
for idx in flagged_indices:
    idx_int = int(idx)
    col_contributions = []
    
    for col_j in range(n_cols):
        val = data[idx_int, col_j]
        if jnp.isnan(val):
            col_contributions.append((col_names[col_j], float("nan"), float("nan")))
            continue
        
        log_p = packed_predictive_probability(
            packed, data,
            query_cols=jnp.array([col_j]),
            query_vals=jnp.array([val]),
            condition_row=idx_int,
        )
        col_contributions.append((col_names[col_j], float(val), float(log_p)))
    
    # Sort: most surprising columns first
    col_contributions.sort(key=lambda x: x[2] if not np.isnan(x[2]) else float("inf"))
    drilldown_results[idx_int] = col_contributions
```

## Interpreting results

Each column gets a log predictive probability. The columns with the **most negative** log_p are the primary anomaly drivers:

```
Row 42 (anomaly_score=3.2):
  salary=250000    (log_p=-8.5)  ← Very surprising given the cluster
  age=22           (log_p=-3.2)  ← Moderately surprising
  department=sales (log_p=-0.8)  ← Normal
  tenure=1         (log_p=-0.3)  ← Normal
```

Interpretation: Row 42 is anomalous primarily because the salary is unusually high for someone aged 22 in the sales department.

## Column typicality alternative

For a faster (but coarser) view, use column typicality:

```python
from crosscat import batch_column_typicality

# This scores how typical each column's value is across the entire dataset
typ_scores = batch_column_typicality(packed)
# Shape: (n_cols,) — lower = less typical column behavior overall
```

This gives per-column scores for the dataset as a whole, not per-row. Use it to identify globally unusual columns.
