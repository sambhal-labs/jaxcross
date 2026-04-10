---
name: jxc-discover
description: Discover variable dependencies and data structure using a trained jaxcross model. Generates dependence matrix, mutual information rankings, view decomposition (which variables are independent groups), column typicality, and conditional entropy. Use after /jxc-model to answer "which variables are related and how is my data structured?"
version: "1.0.0"
license: Apache-2.0
---

# Structure & Dependency Discovery

Discover which variables are related, how they cluster into independent groups, and the information-theoretic structure of your data.

Usage: `/jxc-discover [--model model.jxc] [--data data.arrow]`

## Step 1: Load model

```python
import jax
import jax.numpy as jnp
from crosscat import load_packed_state, unbatch_packed_states
from crosscat.data_utils import load_data

packed, col_types = load_packed_state("model.jxc")
data, col_names, _ = load_data("data/prepared.arrow")

# If multi-chain model available:
# all_chains = unbatch_packed_states(batched, N_CHAINS)
# For single chain, wrap in list:
all_chains = [packed]

n_cols = len(col_names)
```

## Step 2: Dependence matrix

The dependence matrix shows how strongly each pair of columns is related (probability they belong to the same view across posterior samples):

```python
from crosscat import packed_dependence_matrix

z_matrix = packed_dependence_matrix(all_chains)

# Extract and rank all pairs
pairs = []
for i in range(n_cols):
    for j in range(i + 1, n_cols):
        pairs.append((col_names[i], col_names[j], float(z_matrix[i, j])))

pairs.sort(key=lambda x: -x[2])

print("Top 20 variable dependencies:")
print(f"{'Column A':<25} {'Column B':<25} {'Dependence':>10}")
print("-" * 62)
for a, b, score in pairs[:20]:
    print(f"{a:<25} {b:<25} {score:>10.3f}")

print(f"\nWeakest dependencies:")
for a, b, score in pairs[-5:]:
    print(f"{a:<25} {b:<25} {score:>10.3f}")
```

**Interpretation:** Dependence probability close to 1.0 means the columns almost always end up in the same view (strongly related). Close to 0.0 means they're almost always in different views (independent).

See [dependence-matrix-guide.md](references/dependence-matrix-guide.md) for visualization and interpretation.

## Step 3: Mutual information (top pairs)

For the strongest dependencies, compute mutual information to quantify how much knowing one variable tells you about another:

```python
from crosscat import packed_mutual_information

print("\nMutual information for top dependencies:")
for a, b, dep_score in pairs[:10]:
    col_i = col_names.index(a)
    col_j = col_names.index(b)
    
    mi = packed_mutual_information(all_chains, col_i=col_i, col_j=col_j)
    
    # Linfoot correlation (normalized MI, 0-1 scale)
    linfoot = float(jnp.sqrt(1 - jnp.exp(-2 * mi)))
    
    print(f"  {a} <-> {b}: MI={float(mi):.3f}, Linfoot={linfoot:.3f}")
```

**Interpretation:** MI = 0 means fully independent. Higher MI means more shared information. Linfoot correlation normalizes to [0, 1] like Pearson correlation but captures nonlinear relationships.

## Step 4: View decomposition

CrossCat partitions columns into views — groups of columns that share a common row clustering. Columns in different views are modeled as independent given the clustering.

```python
from crosscat import unpack_state

state = unpack_state(packed, col_types, data=data)

print(f"\nView decomposition ({len(state.views)} views):")
for i, view in enumerate(state.views):
    cols = [col_names[j] for j in view.column_indices]
    n_clusters = len(set(int(a) for a in view.row_assignments))
    print(f"\n  View {i} ({n_clusters} clusters):")
    print(f"    Columns: {cols}")
    
    # Show cluster sizes
    from collections import Counter
    cluster_counts = Counter(int(a) for a in view.row_assignments)
    sizes = sorted(cluster_counts.values(), reverse=True)
    print(f"    Cluster sizes: {sizes}")
```

**Interpretation:** Columns in the same view are statistically dependent — they share a common latent structure (row clustering). Columns in different views are independent: knowing the cluster assignment in one view tells you nothing about the other.

See [view-decomposition.md](references/view-decomposition.md) for deeper interpretation.

## Step 5: Column typicality

Score how "typical" each column is within the model structure:

```python
from crosscat import batch_column_typicality

col_typ = batch_column_typicality(packed)

print("\nColumn typicality (lower = more unusual):")
for j in jnp.argsort(col_typ):
    print(f"  {col_names[int(j)]}: {float(col_typ[j]):.3f}")
```

**Interpretation:** Low typicality means the column doesn't fit well into the model structure — it may need a different column type, or it may genuinely be unusual (e.g., a noisy or uninformative feature).

## Step 6: Conditional entropy

Measure how much uncertainty remains in each column given the others:

```python
from crosscat import batch_conditional_entropy

key = jax.random.key(99)

print("\nConditional entropy (lower = more predictable from other columns):")
for target_col in range(n_cols):
    key, subkey = jax.random.split(key)
    
    # Entropy of target given all other columns
    condition_cols = [j for j in range(n_cols) if j != target_col]
    
    h = batch_conditional_entropy(
        subkey, packed, data,
        target_col=target_col,
        row_ids=jnp.arange(min(100, data.shape[0])),  # Sample rows for speed
    )
    mean_h = float(jnp.mean(h))
    print(f"  {col_names[target_col]}: H = {mean_h:.3f}")
```

**Interpretation:** Lower conditional entropy means the column is more predictable from the others. Columns with high conditional entropy are either independent or have complex relationships not captured by the model.

## Summary report

Print a structured summary:

```
# Dependency Discovery Report

## Data
- File: <path>
- Shape: N rows x M columns
- Views: V

## Variable Groups (Views)
- View 0: [col_a, col_b, col_c] (K clusters)
- View 1: [col_d, col_e] (K clusters)
- View 2: [col_f] (K clusters)

## Strongest Dependencies
| Column A | Column B | Dependence | MI | Linfoot |
...

## Independent Column Groups
Columns in different views are statistically independent:
- {col_a, col_b, col_c} ⊥ {col_d, col_e} ⊥ {col_f}

## Most Predictable Columns
(lowest conditional entropy — best candidates for imputation/prediction)

## Most Unusual Columns
(lowest typicality — may need attention)
```

## Common Pitfalls

- **Single chain gives noisy dependence**: Use multi-chain (`all_chains = unbatch_packed_states(...)`) for more stable dependence probabilities.
- **MI requires multiple chains**: `packed_mutual_information` accepts a list for posterior averaging. With a single chain, MI estimates may be noisy.
- **View count depends on convergence**: An unconverged model may have too many or too few views. Check Rhat first.
- **Dependence != causation**: High dependence probability means statistical association, not causal relationship.
