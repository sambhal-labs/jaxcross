# Gibbs Kernels in Detail

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
