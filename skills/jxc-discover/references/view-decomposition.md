# View Decomposition Guide

## What are views?

In CrossCat, a **view** is a group of columns that share a common row clustering. The model discovers these groups automatically via Gibbs sampling.

- Columns in the **same view** are statistically dependent — they share a common latent structure
- Columns in **different views** are conditionally independent — knowing about one group tells you nothing about the other

## Interpretation examples

### Customer data
```
View 0: [age, income, education]     (3 clusters)
View 1: [color_preference, brand]    (5 clusters)
View 2: [zip_code]                   (2 clusters)
```
Interpretation: Demographics (age/income/education) form one independent structure. Product preferences (color/brand) form another. Geography is independent of both.

### Medical data
```
View 0: [blood_pressure, cholesterol, bmi]  (4 clusters)
View 1: [smoking, alcohol, exercise]         (3 clusters)
```
Interpretation: Physiological measures cluster independently from lifestyle behaviors.

## Multiple segmentations

Unlike k-means (one clustering), CrossCat discovers **multiple overlapping segmentations**:

- A customer might be in cluster 2 of the demographics view (high-income, older) AND cluster 1 of the preferences view (likes blue, prefers brand X)
- These are independent facts about the customer

## View stability

Check view stability across chains:
```python
# Compare view structures across chains
for i, chain in enumerate(all_chains):
    state = unpack_state(chain, col_types, data=data)
    print(f"Chain {i}: {len(state.views)} views")
    for v, view in enumerate(state.views):
        cols = [col_names[j] for j in view.column_indices]
        print(f"  View {v}: {cols}")
```

If all chains agree on the same column groupings, the structure is robust. If they disagree, the dependence between those columns is ambiguous.

## Single-column views

A column alone in its own view means it's independent of all other columns. This could mean:
- The column is genuinely unrelated to the rest
- The column is noise or an ID field
- The model needs more sweeps to discover its relationships
