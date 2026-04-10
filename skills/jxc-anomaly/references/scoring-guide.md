# Anomaly Scoring Guide

## Available scoring functions

| Function | Input | Best for |
|----------|-------|----------|
| `batch_anomaly_score(packed, data, row_ids)` | Single packed state, batch of rows | Production batch scoring |
| `multi_chain_anomaly_score(key, chains, data, query_row)` | List of chains, single row | Robust single-row scoring |
| `batch_row_typicality(packed, data, row_ids)` | Single packed state, batch of rows | Row clustering typicality |

## batch_anomaly_score vs batch_row_typicality

**`batch_anomaly_score`**: Measures how surprising the row's values are under the posterior predictive. Considers the actual data values against the model's predicted distribution. A row with unusual values for its cluster gets a high score.

**`batch_row_typicality`**: Measures how well a row fits its assigned cluster. A row at the boundary between clusters or in a very small cluster gets low typicality. Does not consider the data values themselves, only cluster membership.

**Recommendation**: Use `batch_anomaly_score` for data-driven anomaly detection. Use `batch_row_typicality` for structural anomaly detection (rows that don't fit any cluster well).

## Score interpretation

Anomaly scores are log-scale. A difference of 1.0 means the row is ~e times more unlikely.

Typical distribution of scores:
- Most rows cluster around the mean
- Anomalies are in the right tail
- The distribution is often approximately Gaussian in the center with a heavy right tail

## Per-column attribution

To understand why a row is anomalous, compare each column's predictive probability:

```python
# For each column, compute how surprising the value is
for col_j in range(n_cols):
    val = data[row_idx, col_j]
    log_p = packed_predictive_probability(
        packed, data,
        query_cols=jnp.array([col_j]),
        query_vals=jnp.array([val]),
        condition_row=row_idx,
    )
    # Lower log_p = more surprising = drives anomaly score
```

The columns with the lowest log_p are the primary drivers of the anomaly.
