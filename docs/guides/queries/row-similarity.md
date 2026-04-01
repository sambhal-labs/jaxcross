# Row Similarity

Measure the probability that two rows belong to the same cluster, averaged across views and posterior samples. High similarity means the rows are behaviorally alike across the model's learned structure.

## When to Use

- **Finding similar entities** — customers, patients, products with similar profiles
- **Record linkage and deduplication** — detecting near-duplicate records in messy data
- **Cohort analysis** — grouping rows into behavioral cohorts based on learned structure
- **Recommendation** — "users like this one" based on co-clustering patterns

## How It Works

Row similarity is defined as:

$$\text{sim}(i, j) = \frac{1}{V} \sum_{v=1}^{V} \mathbf{1}[z_{v,i} = z_{v,j}]$$

where $V$ is the number of views and $z_{v,i}$ is the cluster assignment of row $i$ in view $v$. When using multiple posterior samples (states), the result is averaged across samples.

**Interpretation:**

| Value | Meaning |
|-------|---------|
| 1.0 | Rows are in the same cluster in every view |
| 0.5 | Rows co-cluster in half the views |
| 0.0 | Rows are in different clusters in every view |

Values above 0.5 indicate the rows are more similar than different. Values near 0.0 mean the rows are behaviorally distinct across all discovered structures.

## Basic Usage

```python
from crosscat import row_similarity

sim = row_similarity([state], row_a=10, row_b=20)
print(f"Similarity: {sim:.3f}")  # 0=different clusters, 1=same clusters
```

## Targeted Similarity

Restrict similarity to views containing specific columns. This lets you ask "are these rows similar *with respect to* these features?"

```python
# Compare rows based only on compensation-related columns
sim = row_similarity([state], row_a=10, row_b=20, target_columns=[0, 1])

# Compare based on geographic columns
geo_sim = row_similarity([state], row_a=10, row_b=20, target_columns=[5, 6])
```

## Similarity Matrix

Build a full pairwise similarity matrix for all rows:

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

Compute the full pairwise similarity matrix in one call — replaces the O(n^2) Python loop:

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

## Practical Tips

- **Use multiple states** for more robust estimates — `row_similarity([state1, state2, state3], ...)`
- **Visualize the matrix** as a heatmap and look for block structure — bright blocks indicate groups of similar rows
- **Combine with anomaly detection** — rows with low similarity to all others are likely anomalies
- **Use targeted similarity** when you care about similarity along specific dimensions (e.g., "similar spending behavior" vs "similar demographics")

## Connection to Other Queries

- **Dependence matrix** measures column-column relationships; **row similarity** measures row-row relationships
- **Row typicality** (`row_typicality`) measures how typical a row is overall; **row similarity** compares specific pairs
- **Anomaly detection** (`predictive_anomalousness`) scores individual rows; row similarity helps understand *which other rows* an anomaly is unlike

## API Reference

- [`row_similarity`](../../api/inference.md#row_similarity)
- [`batch_row_similarity`](../../api/packed-inference.md#batch_row_similarity)
