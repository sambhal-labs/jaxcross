# Gibbs Kernels in Detail

!!! info "Per-kernel deep-dives"
    This page is the overview. Each kernel also has its own detailed page:

    - [Row-Assignment Gibbs](algorithms/row-gibbs.md) — CRP prior, vectorized column scoring, type-specialized fast path.
    - [Column-Assignment Gibbs](algorithms/column-gibbs.md) — view membership, auxiliary view creation, `max_cols_per_view` overflow.
    - [Hyperparameter Transitions](algorithms/hyper-transitions.md) — grid-based updates per component type.
    - [CRP Alpha Update](algorithms/crp-alpha.md) — outer and inner DP concentrations.
    - [Ordered Logistic Grid Integration](algorithms/ordered-logistic-grid.md) — the only non-conjugate component.

## Row Assignment (Critical Path)

For each view, for each row *i*:

1. Remove row *i* from its current cluster *c_old*
2. For each existing cluster *c* and the "new cluster" option:
    - **CRP prior**: log(count_c) for existing, log(alpha) for new
    - **Likelihood**: product over columns of posterior_predictive_logp(x_i | suffstats_c)
3. Sample new assignment from categorical(log_probs)
4. Compact cluster indices (remove empty clusters)

This is O(n_rows x n_clusters x n_cols_per_view) per view per sweep — the computational bottleneck.

## Column Assignment

For each column *j*:

1. Remove column *j* from its current view
2. For each existing view *v* and a proposed new view:
    - **CRP prior**: log(count_v) for existing, log(alpha) for new
    - **Likelihood**: total log_marginal_likelihood of column *j*'s data under view *v*'s row clustering
3. Sample new assignment
4. If new view selected, sample row clustering from CRP
5. Remove empty views, reindex

## Hyperparameter Sampling

Grid-based Gibbs following original CrossCat, with data-dependent ranges and N_GRID=31:

| Type | Parameters | Grid |
|------|-----------|------|
| Continuous | s, mu, nu, r | s: log-spaced [SSD/100, SSD]; mu: linear [min, max]; nu: log-spaced [1, N]; r: log-spaced [1/N, N] |
| Categorical | dirichlet_alpha | log-spaced [1/N, N] |
| Binary | alpha, beta | 8x8 log-spaced [1/N, N] |
| Cyclic | kappa, vm_a, vm_mu | kappa: linspace [kappa_est, N*kappa_est]; vm_a: 31 pts; vm_mu: 31 pts |
| CRP alphas | column/row alpha | log-spaced [1/N, N] scaled to row/column count |

Each grid point is scored by the log marginal likelihood of all data in each cluster, and a new value is sampled proportional to these scores.

## Packed Implementation

In the packed path, these kernels are implemented with JAX primitives:

- **Row assignments**: Nested `lax.scan` (outer over views, inner over rows) with `vmap` over clusters for scoring
- **Column assignments**: `lax.scan` over columns, `vmap` over views, bounded CRP for new-view proposals
- **Hyperparameters**: `vmap` over all columns with unified type dispatch via `jnp.where`
- **CRP alphas**: `vmap` over log-spaced grid

All packed kernels are decorated with `@jax.jit` for independent compilation.

## Synchronous (Parallel) Row Kernel — Approximation Tradeoff

`packed_transition_row_assignments_parallel` is an alternate row kernel
available through [scaling.parallel_gibbs_sweep](../guides/scaling.md). Every
row evaluates its leave-one-out cluster scores against the **same shared
baseline** of counts and sufficient statistics, then all rows are reassigned
in one vectorized step. This is dramatically faster than the sequential
kernel because the inner loop is a pure `vmap` with no row-to-row dependency.

It is **not exactly collapsed Gibbs**, however. The target of the parallel
kernel is the product of per-row conditional posteriors rather than the joint
posterior over assignments. In practice convergence to the correct stationary
distribution is observed, but mixing is slowed for views with heavily
overlapping clusters (each row "sees" a stale baseline that still contains
the other movers).

**When to use the parallel kernel**:

- Very wide datasets (many rows) where the sequential kernel is the bottleneck.
- Large minibatch training where approximate steps are already acceptable.

**Recommended cadence**:

- Alternate with one full pass of `packed_transition_row_assignments`
  (sequential) every 3–5 parallel sweeps. The sequential pass cleans up
  inconsistencies the parallel baseline introduces and is cheap relative to
  the full workload.
- Always finish a run with at least a few sequential sweeps before reading
  off the final posterior. Inference queries are insensitive to this, but
  chain comparisons (Rhat, ESS) become more reliable when the final
  assignments are exact-conditional samples.
- The parallel kernel **cannot create new clusters** — combine with either
  the sequential kernel or `packed_transition_row_assignments_minibatch` if
  cluster birth is required during the run.

The docstring at `crosscat.packed.kernels.packed_transition_row_assignments_parallel`
also describes this tradeoff; the `jnp.maximum(counts - 1, 0)` clamp inside
that kernel is load-bearing specifically because of the shared-baseline semantics.
