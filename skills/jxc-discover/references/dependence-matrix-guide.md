# Dependence Matrix Guide

## What is the dependence matrix?

The dependence matrix (Z-matrix) is an N_cols x N_cols matrix where entry Z[i,j] is the posterior probability that columns i and j are in the same view.

- Z[i,j] = 1.0: columns always co-occur in the same view across posterior samples
- Z[i,j] = 0.0: columns never co-occur in the same view
- Z[i,i] = 1.0: diagonal is always 1 (a column is always with itself)

## Reading the matrix

Clusters of high values (close to 1.0) indicate groups of related variables. Block-diagonal structure means clean separation into independent groups.

Example:
```
          GDP   Pop   Area  Color  Size
GDP      1.00  0.95  0.85  0.05  0.10
Pop      0.95  1.00  0.90  0.03  0.08
Area     0.85  0.90  1.00  0.07  0.12
Color    0.05  0.03  0.07  1.00  0.92
Size     0.10  0.08  0.12  0.92  1.00
```

Two clear blocks: {GDP, Pop, Area} and {Color, Size} — these are independent variable groups.

## Visualization

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(z_matrix, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(n_cols))
ax.set_yticks(range(n_cols))
ax.set_xticklabels(col_names, rotation=45, ha="right")
ax.set_yticklabels(col_names)
plt.colorbar(im, label="Dependence probability")
plt.title("Column Dependence Matrix")
plt.tight_layout()
plt.savefig("dependence_matrix.png", dpi=150)
```

## Reordering for clarity

Sort columns by view assignment for cleaner block structure:
```python
from crosscat import unpack_state
state = unpack_state(packed, col_types, data=data)

# Get column order by view
order = []
for view in state.views:
    order.extend(view.column_indices)

z_ordered = z_matrix[jnp.array(order)][:, jnp.array(order)]
names_ordered = [col_names[i] for i in order]
```
