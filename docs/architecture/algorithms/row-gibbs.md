# Row-Assignment Gibbs

**Source**: [`crosscat/packed/kernels.py`](https://github.com/sambhal-labs/jaxcross/blob/main/crosscat/packed/kernels.py) — `packed_transition_row_assignments`, `_score_row_all_clusters`, `_score_row_one_cluster`, `_score_row_one_cluster_typed`.

The row-assignment kernel is the inner loop of a CrossCat sweep: given the current column-to-view partition, it resamples the cluster assignment of every row *in every view*, one row at a time.

## Math

For each view `v`, each row `n`, draw a new cluster assignment `z_{n,v}` from its collapsed full conditional:

```
P(z_{n,v} = k | rest)  ∝  CRP(k | z_{-n,v}, alpha_v) · ∏_{c ∈ view v} P(x_{n,c} | suffstats_{k, c, -n})
```

- **CRP prior** — existing clusters get weight proportional to their current size; a fresh cluster gets weight `alpha_v`. Concentration `alpha_v` is the per-view inner-DP concentration.
- **Likelihood** — for each column `c` in the view, the collapsed posterior predictive of the conjugate component, *evaluated against the sufficient statistics of cluster `k` excluding row `n`*.

Because all component parameters are integrated out (collapsed sampling), the sampler state is just the discrete assignment arrays plus each cluster's sufficient statistics — no means, variances, or Dirichlet parameters are explicitly represented.

## Algorithm

```
for view v in 0..n_views:
    for row n in 0..n_rows:
        1. Remove row n's contribution from its current cluster's suffstats
           (batched scatter via .at[].add() over all columns in the view)
        2. Compute CRP prior log-weights for each candidate cluster (and one auxiliary)
        3. For each candidate cluster k, compute
               log L(row n | cluster k) = Σ_c log P(x_{n,c} | suffstats_{k,c})
           vectorized over columns with vmap, vectorized over clusters with vmap
        4. Sample k ~ softmax(log_prior + log_likelihood)
        5. Add row n's contribution back into cluster k's suffstats
        6. If k is the auxiliary (a new cluster), grow the cluster budget
```

Steps 1–6 are expressed as an inner `lax.scan` body (`scan_one_row`) over rows, nested inside an outer `lax.scan` body (`scan_one_view`) over views. Both closures live inside `packed_transition_row_assignments`.

## Key Optimizations

### Vectorized column scoring

Prior to v0.9, scoring a row against a cluster was a `lax.scan` over columns — sequential, one type-dispatch per column. The current path (`_score_row_one_cluster`) uses `jax.vmap(unified_posterior_predictive_logp)` over every column in the view at once. This gave the 12× speedup headlined in the README.

### Type-specialized fast path

When all columns in a view share the same type (common at init, and the norm for MNIST / binary-encoded datasets), `_compute_dominant_type` detects this and `_score_row_one_cluster_typed` dispatches to a type-specific batched function (`batch_bb_posterior_predictive_logp`, `batch_ng_posterior_predictive_logp`, `batch_dc_posterior_predictive_logp`, `batch_vm_posterior_predictive_logp`) — skipping the 5-way `jnp.where` that the heterogeneous path must pay.

### Batched suffstat updates

`_add_row_to_suffstats` / `_remove_row_from_suffstats` scatter-add over *all columns at once* (`.at[].add()`) rather than iterating column-by-column. This is critical: without it, every row update would be O(n_cols) serial operations on the GPU.

### NaN masking

Missing values propagate as `NaN` through the data matrix. `_is_observed` masks likelihood contributions to real values only; the suffstat updaters similarly guard the scatter with an observed-mask. This is the "missing-data transparency" promise — no preprocessing needed.

## Hyperparameter Guidance

- **`alpha_v` (inner-DP concentration)** — Sampled automatically by [`packed_transition_crp_alphas`](crp-alpha.md). Rarely needs manual tuning; the data-driven default from `_default_hypers` is adequate for most cases.
- **`max_clusters` in `pack_state`** — The cluster-budget padding. If inference saturates it (you'll see a `jax.debug.callback` warning), rerun with `max_clusters = suggest_max_clusters(n_rows) * 1.5`.

## Gotchas

- **Cluster compaction.** `_compact_clusters` reindexes clusters after every row sweep so that cluster ids are dense (`0..n_active-1`). Without this, the budget fills up with ghost empty clusters.
- **Auxiliary cluster scoring.** The "new cluster" branch uses an *empty* sufficient-statistics block with the column's prior hypers — not a zero-filled real cluster. Confusing these leads to off-by-one in the effective CRP weights.
- **`jnp.where` evaluates both branches.** Row scoring must keep padded cluster slots numerically well-behaved (finite `-inf` prior weight, finite likelihood) because JAX evaluates both sides before selecting. The pattern: clamp, then `where`, never the other way around.

## Related

- [Column-Assignment Gibbs](column-gibbs.md) — the outer-DP counterpart that resamples view membership of each column.
- [Hyperparameter Transitions](hyper-transitions.md) — grid-based updates for component hyperparameters after each row/column sweep.
- [Packed Kernels API](../../api/packed-kernels.md)
