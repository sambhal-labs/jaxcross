---
name: jxc-segment
description: Discover and profile natural data segments (clusters) using a trained jaxcross model. Extracts cluster assignments per view, computes per-segment statistics (mean, mode, proportions), identifies segment-defining features, and measures row similarity. Use after /jxc-model for customer segmentation, cohort analysis, or entity grouping.
version: "1.0.0"
license: Apache-2.0
---

# Segmentation & Clustering

Discover natural segments in your data and profile them.

Usage: `/jxc-segment [--model model.jxc] [--data data.arrow]`

## Step 1: Extract segment assignments

```python
import jax
import jax.numpy as jnp
import numpy as np
from collections import Counter
from crosscat import load_packed_state, unpack_state
from crosscat.data_utils import load_data

packed, col_types = load_packed_state("model.jxc")
data, col_names, _ = load_data("data/prepared.arrow")

state = unpack_state(packed, col_types, data=data)

print(f"Model has {len(state.views)} views (independent segmentation structures)")
```

**Key insight:** CrossCat discovers **multiple overlapping segmentations**. Each view provides a different way to segment the data based on different subsets of columns.

## Step 2: Profile segments per view

```python
from crosscat.types import ColumnType

for v_idx, view in enumerate(state.views):
    col_idx = view.column_indices
    view_cols = [col_names[j] for j in col_idx]
    assignments = np.array(view.row_assignments)
    cluster_ids = sorted(set(assignments))
    
    print(f"\n{'='*60}")
    print(f"View {v_idx}: Segments based on {view_cols}")
    print(f"{'='*60}")
    
    for cluster_id in cluster_ids:
        mask = assignments == cluster_id
        n_members = int(mask.sum())
        pct = 100 * n_members / len(assignments)
        
        print(f"\n  Segment {cluster_id} ({n_members} rows, {pct:.1f}%):")
        
        for j in col_idx:
            col_data = np.array(data[mask, j])
            col_data = col_data[~np.isnan(col_data)]
            
            if len(col_data) == 0:
                continue
            
            ct = col_types[j]
            if ct == ColumnType.CONTINUOUS:
                print(f"    {col_names[j]}: mean={np.mean(col_data):.2f}, "
                      f"std={np.std(col_data):.2f}")
            elif ct in (ColumnType.CATEGORICAL, ColumnType.ORDINAL):
                mode_val = int(Counter(col_data.astype(int)).most_common(1)[0][0])
                print(f"    {col_names[j]}: mode={mode_val}, "
                      f"n_unique={len(set(col_data.astype(int)))}")
            elif ct == ColumnType.BINARY:
                rate = np.mean(col_data)
                print(f"    {col_names[j]}: positive_rate={rate:.2f}")
            elif ct == ColumnType.CYCLIC:
                # Circular mean
                sin_mean = np.mean(np.sin(col_data))
                cos_mean = np.mean(np.cos(col_data))
                circ_mean = np.arctan2(sin_mean, cos_mean) % (2 * np.pi)
                print(f"    {col_names[j]}: circular_mean={circ_mean:.2f} rad")
```

## Step 3: Segment sizing and distribution

```python
print("\nSegment size distribution:")
for v_idx, view in enumerate(state.views):
    assignments = np.array(view.row_assignments)
    counts = Counter(assignments)
    sizes = sorted(counts.values(), reverse=True)
    
    print(f"\n  View {v_idx}:")
    for cluster_id, count in counts.most_common():
        bar = "█" * int(50 * count / len(assignments))
        print(f"    Segment {cluster_id}: {count:5d} ({100*count/len(assignments):5.1f}%) {bar}")
```

## Step 4: Segment-defining features

Columns in the same view are the features that define that segmentation:

```python
from crosscat import packed_dependence_matrix

z_matrix = packed_dependence_matrix([packed])

print("\nSegment-defining features per view:")
for v_idx, view in enumerate(state.views):
    view_cols = [col_names[j] for j in view.column_indices]
    print(f"\n  View {v_idx} segments are defined by: {view_cols}")
    
    # Show within-view dependencies
    for i, ci in enumerate(view.column_indices):
        for j, cj in enumerate(view.column_indices):
            if i < j:
                dep = float(z_matrix[ci, cj])
                print(f"    {col_names[ci]} <-> {col_names[cj]}: {dep:.2f}")
```

## Step 5: Row similarity

Find which rows are most similar (belong to the same clusters across views):

```python
from crosscat import batch_row_similarity

# Compare specific rows
row_pairs = jnp.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
similarities = batch_row_similarity(packed, data, row_pairs)

print("\nRow similarities:")
for pair_idx in range(len(row_pairs)):
    i, j = int(row_pairs[pair_idx, 0]), int(row_pairs[pair_idx, 1])
    sim = float(similarities[pair_idx])
    print(f"  Row {i} <-> Row {j}: {sim:.3f}")
```

For finding nearest neighbors within a segment:
```python
# Find the 5 most similar rows to row 42
target_row = 42
pair_ids = jnp.stack([
    jnp.full(data.shape[0], target_row),
    jnp.arange(data.shape[0])
], axis=1)
sims = batch_row_similarity(packed, data, pair_ids)
sims = sims.at[target_row].set(-jnp.inf)  # Exclude self
top5 = jnp.argsort(-sims)[:5]
print(f"\nMost similar rows to row {target_row}:")
for idx in top5:
    print(f"  Row {int(idx)}: similarity={float(sims[idx]):.3f}")
```

## Step 6: Export results

```python
import pandas as pd

# Export segment assignments
seg_df = pd.DataFrame({"row_id": range(data.shape[0])})
for v_idx, view in enumerate(state.views):
    seg_df[f"view_{v_idx}_segment"] = np.array(view.row_assignments).astype(int)

# Add original data
for j, name in enumerate(col_names):
    seg_df[name] = [float(data[i, j]) for i in range(data.shape[0])]

seg_df.to_csv("segment_assignments.csv", index=False)

# Export segment profiles
profiles = []
for v_idx, view in enumerate(state.views):
    assignments = np.array(view.row_assignments)
    for cluster_id in sorted(set(assignments)):
        mask = assignments == cluster_id
        profile = {
            "view": v_idx,
            "segment": int(cluster_id),
            "size": int(mask.sum()),
            "pct": 100 * mask.sum() / len(assignments),
        }
        for j in view.column_indices:
            col_data = np.array(data[mask, j])
            col_data = col_data[~np.isnan(col_data)]
            if len(col_data) > 0:
                profile[f"{col_names[j]}_mean"] = float(np.mean(col_data))
                profile[f"{col_names[j]}_std"] = float(np.std(col_data))
        profiles.append(profile)

pd.DataFrame(profiles).to_csv("segment_profiles.csv", index=False)

print("Exported: segment_assignments.csv, segment_profiles.csv")
```

## Common Pitfalls

- **Multiple segmentations**: Unlike k-means, CrossCat gives you MULTIPLE segmentations (one per view). This is a feature, not a bug — different aspects of the data may have different cluster structures.
- **Segment labels are arbitrary**: Segment 0 in one chain may correspond to segment 2 in another. Compare by profile, not by label.
- **Unpack for assignments**: Cluster assignments are in the unpacked state (`state.views[i].row_assignments`), not directly in the packed state.
- **CRP concentration**: The number of segments is determined automatically by the Chinese Restaurant Process. More data → potentially more segments.

See [segment-profiling.md](references/segment-profiling.md) for advanced profiling techniques.
