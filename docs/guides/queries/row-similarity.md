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

## Batch Similarity Matrix (Vectorized)

Compute the full pairwise similarity matrix in one call — replaces the O(n²) Python loop:

```python
from crosscat import batch_row_similarity
import jax.numpy as jnp

packed_states = [pack_state(s) for s in final_states]

# Full similarity matrix for all rows
sim_matrix = batch_row_similarity(packed_states, jnp.arange(data.shape[0]))
# Shape: (n_rows, n_rows), values in [0, 1]

# Or for a subset of rows
subset = jnp.array([0, 5, 10, 20, 50])
sim_sub = batch_row_similarity(packed_states, subset)
```

## Packed Version

```python
from crosscat import packed_row_similarity

sim = packed_row_similarity([packed], col_types, row_a=10, row_b=20)
```

## API Reference

- [`row_similarity`](../../api/inference.md#row_similarity)
- [`batch_row_similarity`](../../api/packed-inference.md#batch_row_similarity)
