# Packed State Design

## Why Padding?

The unpacked `CrossCatState` uses Python lists of variable length — `views` is a list, each view has variable-length `column_indices`, each cluster has variable-length sufficient statistics. This prevents JAX's `jit` compiler from tracing the computation graph.

The packed representation solves this by padding everything to fixed maximum dimensions:

## Original vs Packed

### Original (Python lists)

```python
CrossCatState(
    column_assignments=jnp.array([0, 0, 1, 1]),
    views=[
        ViewState(
            column_indices=jnp.array([0, 1]),
            row_assignments=jnp.array([0, 0, 1, ...]),
            suffstats=[[ss_c0_col0, ss_c0_col1], [ss_c1_col0, ss_c1_col1]],
        ),
        ViewState(...),
    ],
)
```

### Packed (padded arrays)

```python
PackedCrossCatState(
    column_assignments=jnp.array([0, 0, 1, 1]),
    view_row_assignments=jnp.zeros((16, 200)),     # (max_views, n_rows)
    view_n_clusters=jnp.array([2, 3, 0, ...]),     # (max_views,)
    ss_counts=jnp.zeros((16, 32, 8)),              # (max_views, max_clusters, max_cols)
    ss_sum_x=jnp.zeros((16, 32, 8)),
    hyper_mu=jnp.zeros(4),                         # (n_cols,) per-column hyperparams
    hyper_cutpoints=jnp.full((4, 15), jnp.inf),   # (n_cols, max_categories-1) ordinal cutpoints
    ...
)
```

## What Padding Enables

All views, clusters, and columns padded to fixed dimensions. Invalid entries masked. This enables:

- **`jax.jit`** — compile entire kernels into optimized XLA programs
- **`jax.vmap`** — vectorize over cluster or column dimensions
- **`jnp.where`** — column type dispatch without Python branching
- **`jax.lax.scan`** — efficient sequential loops with compiled loop bodies

## Pytree Registration

`PackedCrossCatState` is registered as a JAX pytree, which means JAX can automatically:

- Trace through the state structure for JIT compilation
- Apply `vmap` transformations to all state arrays simultaneously
- Differentiate through state operations (if needed)

## Masking

Invalid entries (beyond actual view/cluster counts) are masked:

- `view_mask` — which views are active
- `n_views` — actual number of views
- `view_n_clusters` — actual clusters per view

Kernels use these masks to skip computation on padding entries.

## Memory Impact

Padding dimensions directly control memory usage:

| Setting | Array Size Example |
|---------|-------------------|
| `max_views=8, max_clusters=16` | `ss_counts`: 8 x 16 x n_cols |
| `max_views=16, max_clusters=32` | `ss_counts`: 16 x 32 x n_cols |

For 257 columns (MNIST), the difference is ~4x memory. Choose the smallest padding that exceeds your data's actual complexity.

### `max_cols_per_view`

By default, `pack_state()` sets `max_cols_per_view = n_cols` — the safe worst case where all columns merge into a single view. For wide datasets (>100 columns), this creates significant padding overhead in the suffstat arrays, which have shape `(max_views, max_clusters, max_cols_per_view)`.

You can override this to reduce memory:

```python
packed = pack_state(state, max_cols_per_view=max(32, n_cols // 4))
```

!!! warning "Risk of column loss"
    If a view accumulates more columns than `max_cols_per_view` during Gibbs sweeps, excess columns are silently dropped. Only reduce this if you have domain knowledge about your data's view structure.
