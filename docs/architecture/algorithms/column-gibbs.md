# Column-Assignment Gibbs

**Source**: [`crosscat/packed/kernels.py`](https://github.com/sambhal-labs/jaxcross/blob/main/crosscat/packed/kernels.py) — `packed_transition_column_assignments`, `_score_column_in_view`.

The column-assignment kernel is the *outer* DP of CrossCat. For each column `c`, it resamples which view that column belongs to, conditional on all current row-in-view assignments.

## Math

```
P(view(c) = v | rest)  ∝  CRP(v | view(-c), alpha_col) · P(x_{:,c} | row_assignments_v, hypers_c)
```

- **CRP prior** — favors views that already own more columns; allows spawning a fresh view with weight `alpha_col` (outer-DP concentration).
- **Data likelihood** — for candidate view `v`, compute the collapsed log marginal of column `c` given view `v`'s current row clustering and column `c`'s hyperparameters. This is `log P(x_{:,c} | z_v)` summed over clusters of view `v`.

Creating a new view requires drawing a *fresh row partition* for that view (an auxiliary parameter, since the likelihood requires a clustering). This is the "sampling an auxiliary view" step.

## Algorithm

```
for column c in 0..n_cols:
    1. Remove column c from its current view (update view size)
    2. For each candidate view v (existing + auxiliary):
         a. If auxiliary, draw a fresh row clustering via CRP from alpha_col
         b. Compute collapsed log marginal of column c under view v's clustering
    3. Sample v ~ softmax(log_prior + log_likelihood)
    4. Assign column c to view v; if auxiliary, register the new view's row clustering
    5. If a view was emptied, compact view indices
```

Executed as a `lax.scan` over columns inside `packed_transition_column_assignments`, which is a top-level `@jax.jit` entry point (so it recompiles only when the data shape changes).

## Auxiliary View Creation

When column `c` lands in a *new* view, that view needs its own row partition. The kernel draws one on the fly: it samples a fresh row-CRP concentration `alpha_v ~ Gamma(1, 1)` (via `jax.random.gamma(k_alpha, 1.0)`) and then draws row assignments from the CRP prior under that `alpha_v`. If `c` is moving out of a singleton view, the existing view's `alpha_v` is reused instead. The outer-DP concentration `alpha_col` governs column-to-view membership only — it is never used as a row-level concentration. The new view's suffstats are then computed from the drawn clustering and the observed values of column `c`.

After the column transition, the **row-assignment kernel** ([row-gibbs.md](row-gibbs.md)) will typically run immediately, so the drawn clustering quickly relaxes toward its posterior under the new column.

## Max-Columns-Per-View Overflow

`PackedCrossCatState` allocates a fixed-size `(max_views, max_cols_per_view)` assignment buffer at pack time. If a Gibbs step assigns more than `max_cols_per_view` columns to a single view, the kernel:

1. Emits a warning via `jax.debug.callback` (`_warn_column_overflow`); under `set_overflow_policy("raise")` or `JAXCROSS_OVERFLOW_POLICY=raise` this becomes a `RuntimeError`.
2. **Silently drops** the columns beyond the budget — they do not land in the view's assignment buffer, so they are missing from the model's sufficient statistics until the next transition that can re-assign them. This is data corruption, not a rejected move.

To avoid this, set `max_cols_per_view = n_cols` (the default) at `pack_state` time — which is always safe, just a bit more memory — or run production pipelines with `set_overflow_policy("raise")` so any overflow surfaces loudly.

## Hyperparameter Guidance

- **`alpha_col` (outer-DP concentration)** — Updated automatically by [`packed_transition_crp_alphas`](crp-alpha.md). Higher values → more views; lower → one-view-fits-all.
- **`max_views` in `pack_state`** — Cap on the number of views. If saturated, the kernel silently keeps the previous assignment; increase `max_views` and re-run.

## Gotchas

- **View compaction.** After a view is emptied, the kernel must reindex views to keep ids dense. Skipping compaction causes the view budget to fill with ghost empties and appear saturated.
- **Auxiliary-view likelihood.** The collapsed log marginal for an empty view is *not* zero — it's the prior predictive log marginal of column `c` under a fresh CRP clustering. Mistaking these drives columns into new views artificially.
- **Hypers live with the column, not the view.** When a column moves between views, its `ColumnHypers` move with it unchanged. Only the *view's row clustering* changes the likelihood.

## Related

- [Row-Assignment Gibbs](row-gibbs.md) — the inner-DP counterpart.
- [CRP Alpha Update](crp-alpha.md) — how `alpha_col` itself is sampled.
- [Packed State](../packed-state.md) — the `max_cols_per_view` and `max_views` budgets.
