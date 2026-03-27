# Dependence Discovery

## What

Discover which columns are statistically dependent by examining the posterior probability that columns share the same view. The dependence matrix (Z-matrix) is CrossCat's primary exploratory output.

## When to Use

- Exploring which features are related
- Variable selection and feature grouping
- Understanding dataset structure

## Pairwise Dependence

```python
from crosscat import dependence_probability

dp = dependence_probability([state], col_i=0, col_j=1)
print(f"P(salary ~ experience): {dp:.3f}")  # ~1.0 if related
```

## Z-Matrix (Full Dependence Matrix)

```python
from crosscat import dependence_matrix

z = dependence_matrix([state])
# z.shape == (n_cols, n_cols)
# z[i,j] = probability columns i and j share a view
# Diagonal = 1.0, symmetric
```

### Visualizing the Z-matrix

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(z, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(col_names)))
ax.set_xticklabels(col_names, rotation=45, ha="right")
ax.set_yticks(range(len(col_names)))
ax.set_yticklabels(col_names)
plt.colorbar(im, label="Dependence probability")
plt.title("Column Dependence Matrix (Z-matrix)")
plt.tight_layout()
plt.show()
```

## Multi-Chain Z-Matrix

For robust estimates, pass multiple posterior states:

```python
# After running multiple chains
z = dependence_matrix(final_states)  # averages across chains
```

## Column Assignments

You can also directly inspect which view each column was assigned to:

```python
print(f"Column assignments: {state.column_assignments}")
# e.g., [0, 0, 1, 1] means cols 0,1 in View 0 and cols 2,3 in View 1
```

## Packed Versions

```python
from crosscat import packed_dependence_probability, packed_dependence_matrix

dp = packed_dependence_probability([packed], col_a=0, col_b=1)
z = packed_dependence_matrix([packed])
```

## Tips

- The Z-matrix should show clear block structure if there are distinct groups
- Use multiple chains — a single chain gives a binary matrix (0 or 1)
- Block structure in the Z-matrix reveals groups of related features

## API Reference

- [`dependence_probability`](../../api/inference.md#dependence_probability)
- [`dependence_matrix`](../../api/inference.md#dependence_matrix)
