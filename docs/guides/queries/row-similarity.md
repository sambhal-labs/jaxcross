# Row Similarity

## What

Measure the probability that two rows belong to the same cluster, averaged across views and posterior samples. High similarity means the rows are behaviorally alike.

## When to Use

- Finding similar entities (customers, patients, products)
- Record linkage and deduplication
- Building similarity-based recommendations

## Basic Usage

```python
from crosscat import row_similarity

sim = row_similarity([state], row_a=10, row_b=20)
print(f"Similarity: {sim:.3f}")  # 0=different clusters, 1=same clusters
```

## Targeted Similarity

Restrict to views containing specific columns:

```python
# Only compare based on compensation-related columns
sim = row_similarity([state], row_a=10, row_b=20, target_columns=[0, 1])
```

## Similarity Matrix

```python
import numpy as np

n = data.shape[0]
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        s = float(row_similarity([state], row_a=i, row_b=j))
        sim_matrix[i, j] = s
        sim_matrix[j, i] = s
```

## Packed Version

```python
from crosscat import packed_row_similarity

sim = packed_row_similarity([packed], row_a=10, row_b=20)
```

## API Reference

- [`row_similarity`](../../api/inference.md#row_similarity)
