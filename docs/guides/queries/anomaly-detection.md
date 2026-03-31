# Anomaly Detection

## What

Identify unusual rows using two complementary approaches: predictive anomaly scores (data-driven) and structural typicality (cluster-membership-driven).

## When to Use

- Fraud detection, outlier identification
- Data quality auditing
- Finding interesting/unusual observations

## Predictive Anomaly Score

Computes the average log predictive probability of each column value in the row under the posterior, then transforms to a [0, 1] anomaly scale via sigmoid. High score = anomalous.

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

## Batch Anomaly Scoring (Vectorized)

For production use, batch functions score all rows in a single JIT call — 10-100x faster than the Python loop above:

```python
from crosscat import batch_anomaly_score, batch_row_typicality
import jax.numpy as jnp

packed = pack_state(best)

# Anomaly scores for all rows in one call
key, subkey = jax.random.split(key)
scores = batch_anomaly_score(packed, data, jnp.arange(data.shape[0]))

import numpy as np
top5 = np.argsort(np.array(scores))[-5:][::-1]
for idx in top5:
    print(f"Row {idx}: score={float(scores[idx]):.3f}")
```

### Batch Row Typicality

Structural typicality for all rows at once — no RNG key needed:

```python
typicality = batch_row_typicality([packed], jnp.arange(data.shape[0]))

# Most atypical rows (lowest typicality)
bottom5 = np.argsort(np.array(typicality))[:5]
for idx in bottom5:
    print(f"Row {idx}: typicality={float(typicality[idx]):.3f}")
```

## Tips

- Use `predictive_anomalousness` / `batch_anomaly_score` for data-level anomalies (unusual values)
- Use `row_typicality` / `batch_row_typicality` for structural anomalies (doesn't fit any cluster well)
- Always use multi-chain for reliable anomaly scores
- **Prefer batch functions** for datasets with more than a few rows

## API Reference

- [`predictive_anomalousness`](../../api/inference.md#predictive_anomalousness)
- [`row_typicality`](../../api/inference.md#row_typicality)
- [`column_typicality`](../../api/inference.md#column_typicality)
- [`batch_anomaly_score`](../../api/packed-inference.md#batch_anomaly_score)
- [`batch_row_typicality`](../../api/packed-inference.md#batch_row_typicality)
