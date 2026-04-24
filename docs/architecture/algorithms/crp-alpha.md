# CRP Concentration Update

**Source**: [`crosscat/packed/kernels.py`](https://github.com/sambhal-labs/jaxcross/blob/main/crosscat/packed/kernels.py) — `packed_transition_crp_alphas`.

CrossCat has two Chinese-Restaurant-Process concentration parameters, both resampled every sweep:

- **`alpha_col`** — outer DP; governs how many views are preferred.
- **`alpha_v` (one per view)** — inner DP; governs how many row clusters per view.

Both are updated via **grid Gibbs** — the same pattern as [hyper transitions](hyper-transitions.md).

## Math

For a CRP with concentration `alpha`, `n` items, and `k` active tables:

```
log P(partition | alpha)  =  k · log(alpha) + log Γ(alpha) - log Γ(alpha + n) + const
```

The posterior over `alpha` given the current partition is:

```
P(alpha | k, n)  ∝  P(alpha) · alpha^k · Γ(alpha) / Γ(alpha + n)
```

where `P(alpha)` is a weakly informative prior (typically Gamma). The kernel:

1. Builds a 31-point log-spaced grid of candidate `alpha` values spanning `[1/N, N]`, where `N = n_cols` for the outer (column) CRP and `N = n_rows` for each inner (per-view row) CRP. The grid adapts to the dataset size — it is not a fixed range.
2. Evaluates the log posterior on each grid point.
3. Draws `alpha` from the categorical induced by softmax.

## Algorithm

```
# Outer DP (single alpha for column-to-view assignments)
k_col = n_active_views
alpha_col_new = grid_gibbs_update(alpha_grid, k_col, n_cols)

# Inner DP (one alpha per view)
for view v in 0..n_active_views:
    k_v = n_active_clusters_in_view_v
    alpha_v_new = grid_gibbs_update(alpha_grid, k_v, n_rows)
```

Because each alpha only depends on *its own* partition's `(k, n)`, the update is embarrassingly parallel across views and is `vmap`'d in the packed kernel.

## Key Observations

- **`k`, not the full partition, is sufficient.** The conditional posterior depends only on the number of active tables and the total items — not on the cluster sizes. This is why the update is cheap.
- **`log Γ` terms.** `jax.lax.lgamma` is used for numerical stability; subtracting before exponentiating prevents overflow at large `n`.
- **Grid padding.** For padded (inactive) views, the kernel masks the alpha update with `jnp.where(view_active, alpha_new, alpha_old)` so ghost views don't drift.

## Hyperparameter Guidance

- **Grid bounds.** The grid spans `[1/N, N]` log-spaced on 31 points, so it automatically scales with the dataset: for 100 columns the outer grid is `[0.01, 100]`; for 10 000 rows each per-view row grid is `[1e-4, 1e4]`. The grid is built per-sweep from the current `n_cols` / `n_rows`, so no manual tuning is needed for typical datasets.
- **Adaptation.** Because the grid is recomputed each transition from `n_cols` / `n_rows`, it tracks dataset size, but the endpoints are hard-coded (the `1/N .. N` convention matches probcomp/crosscat). If you observe the posterior concentrating at a grid endpoint, that is a sign the prior/data strongly prefer extreme `alpha` — investigate the partition before trying to widen the grid.

## Gotchas

- **`alpha_v` is per-view.** Don't assume a single `alpha` governs all views — `PackedCrossCatState.view_row_crp_alpha` is shape `(max_views,)`.
- **Alpha affects model complexity, not fit.** A higher `alpha` means more groups are a-priori likely, but the data likelihood still drives assignment. Tuning `alpha` manually is almost never necessary — let the Gibbs update do its job.
- **Resetting between runs.** When reloading a checkpointed state (`load_packed_state`), the alphas are restored — do not reinitialize them to a prior mean.

## Related

- [Row-Assignment Gibbs](row-gibbs.md) — uses `alpha_v` in its CRP prior term.
- [Column-Assignment Gibbs](column-gibbs.md) — uses `alpha_col`.
- [Hyperparameter Transitions](hyper-transitions.md) — same grid-Gibbs machinery.
