# Hyperparameter Transitions

**Source**: [`crosscat/packed/kernels.py`](https://github.com/sambhal-labs/jaxcross/blob/main/crosscat/packed/kernels.py) — `packed_transition_column_hypers`.

After each row/column assignment sweep, the model resamples each column's component-model hyperparameters. Unlike the assignment kernels (exact discrete Gibbs), the hyper update uses a **grid Gibbs** — the continuous hyper is discretized onto a fixed grid and sampled categorically.

## Math

For each column `c`, for each of its component-type-specific hyperparameters `θ`:

```
P(θ | all data, assignments)  ∝  P(θ) · ∏_{views containing c} ∏_{clusters k} P(data_{cluster k, col c} | θ)
```

The data term factorises into the cluster-level collapsed log marginal under `θ`. The prior `P(θ)` is weakly informative (typically a wide log-uniform or broad Gamma).

Instead of sampling `θ` continuously, the kernel:

1. Evaluates `log P(θ_i | data)` on a fixed grid `{θ_0, θ_1, ...}`.
2. Softmaxes to a categorical distribution.
3. Draws the new `θ` from that categorical.

## Per-Component Hypers

| Component | Hypers updated | Grid description |
|-----------|----------------|------------------|
| **NormalGamma** (continuous) | `mu` (mean prior), `r` (prior strength), `s` (scale), `nu` (dof) | Four independent grid updates, each Gibbs-conditional on the other three |
| **DirichletCategorical** | `dirichlet_alpha` (symmetric concentration) | Log-spaced grid over [0.001, 10] |
| **BetaBernoulli** (binary) | `alpha`, `beta` | Two independent grid updates |
| **OrderedLogistic** (ordinal) | `cutpoints[0..K-1]` | See [ordered-logistic-grid.md](ordered-logistic-grid.md); grid integration over the latent location parameter |
| **VonMises** (cyclic) | `kappa` (concentration), `vm_a` (prior strength), `vm_mu` (prior mean) | Three independent grid updates |

Each update is a Gibbs step conditional on the other hypers of the same column, so they must be done serially within a column (the kernel does this in a fixed order).

## Algorithm

```
for column c in 0..n_cols:
    t = column_type[c]
    for each hyper θ of component type t (in fixed order):
        grid = generate_grid(θ, data_range[c])
        log_posts = vmap(lambda θ_i: log_prior(θ_i) + log_data(θ_i, c, assignments))(grid)
        θ_new = grid[jax.random.categorical(softmax(log_posts))]
        write θ_new back into column_hypers[c]
```

Type dispatch is implemented with chained `jnp.where` selectors on `column_type[c]` (JAX evaluates every branch, which is acceptable because each type-specific path is cheap relative to the grid-score vmap). The kernel is a `jax.vmap(process_one_column)(jnp.arange(n_cols))`, not a `lax.scan` — columns are processed in parallel.

## Key Optimizations

- **Data-driven grid range.** `generate_grid` uses per-column data statistics (min/max/std) to center the grid — essential for continuous columns with wildly different scales. Without this, most grid points land in the prior tail and sampling quality collapses.
- **Log-sum-exp for collapsed likelihood.** The collapsed log marginal per cluster/view uses `jax.scipy.special.logsumexp` for numerical stability.
- **Vectorized over grid points.** The whole grid is scored in a single `vmap`, one JIT call per hyper per column.

## Hyperparameter Guidance

- **Grid size** — 31 points (see `ORDINAL_N_GRID` in [types.py](../../api/types.md)) is the library-wide default. Larger grids improve accuracy at roughly linear cost; smaller grids trade accuracy for speed.
- **Warm-start on subsample → full data.** When using `subsample_anneal`, hypers fit on the subsample are *reused* on the full data — so they should be near-correct already. A few extra hyper sweeps post-annealing refines them.

## Gotchas

- **Grid must span the posterior mode.** If the data pushes the posterior far from the grid's support, the sampler effectively clamps to a grid endpoint — silently biasing inference. The data-driven grid defaults in `generate_grid` handle common cases but can fail for extreme-scale columns; check your column preprocessing.
- **Hyper updates run *after* assignment updates.** Don't try to reorder — the hypers are conditioned on current assignments, not vice versa.
- **Padded columns.** `PackedCrossCatState` may pad beyond `n_cols` up to the allocation budget; the kernel masks these with `jnp.where(col_idx < n_cols, ..., no-op)` to keep them untouched.

## Related

- [Ordered Logistic Grid](ordered-logistic-grid.md) — the most intricate case (latent location + cutpoints).
- [CRP Alpha Update](crp-alpha.md) — the other continuous-parameter update; also grid-based.
- [Components API](../../api/components.md)
