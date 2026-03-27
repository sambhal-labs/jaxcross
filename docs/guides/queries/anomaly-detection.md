# Anomaly Detection

## What

Identify unusual rows using two complementary approaches: predictive anomaly scores (data-driven) and structural typicality (cluster-membership-driven).

## When to Use

- Fraud detection, outlier identification
- Data quality auditing
- Finding interesting/unusual observations

## Predictive Anomaly Score

Compares each row's predictive probability against Monte Carlo samples. High score = anomalous.

```python
from crosscat import predictive_anomalousness

key, subkey = jax.random.split(key)
score = predictive_anomalousness(subkey, state, data, query_row=42)
print(f"Anomaly score: {score:.3f}")  # 0=normal, 1=anomalous
```

### Scan all rows

```python
scores = []
for row_id in range(data.shape[0]):
    key, subkey = jax.random.split(key)
    s = predictive_anomalousness(subkey, state, data, query_row=row_id)
    scores.append(float(s))

# Top 5 most anomalous rows
import numpy as np
top5 = np.argsort(scores)[-5:][::-1]
for idx in top5:
    print(f"Row {idx}: score={scores[idx]:.3f}")
```

## Row Typicality

Structural typicality measures how well a row fits its assigned cluster(s). Low = anomalous.

```python
from crosscat import row_typicality

typ = row_typicality([state], row_id=42)
print(f"Typicality: {typ:.3f}")  # 0=atypical, 1=typical
```

## Column Typicality

How consistently a column is assigned to the same view across posterior samples.

```python
from crosscat import column_typicality

ct = column_typicality([state], col_id=0)
print(f"Column 0 typicality: {ct:.3f}")
```

## Multi-Chain Anomaly Scores

Average anomaly scores across chains for more robust detection:

```python
from crosscat import multi_chain_anomaly_score

score = multi_chain_anomaly_score(key, packed_states, data, query_row=42)
```

## Packed Versions

```python
from crosscat import packed_anomaly_score, packed_row_typicality

score = packed_anomaly_score(key, packed, data, query_row=42)
typ = packed_row_typicality([packed], row_id=42)
```

## Tips

- Use `predictive_anomalousness` for data-level anomalies (unusual values)
- Use `row_typicality` for structural anomalies (doesn't fit any cluster well)
- Always use multi-chain for reliable anomaly scores

## API Reference

- [`predictive_anomalousness`](../../api/inference.md#predictive_anomalousness)
- [`row_typicality`](../../api/inference.md#row_typicality)
- [`column_typicality`](../../api/inference.md#column_typicality)
