# Architecture

This document explains the internal architecture of jax-crosscat: how the CrossCat model works, how it maps to JAX, and how the modules fit together.

## The CrossCat Model

CrossCat is a **two-level Dirichlet Process mixture model** for heterogeneous tabular data:

1. **Outer DP** (column partition): A Chinese Restaurant Process assigns each column to a "view." Columns in the same view are modeled as dependent; columns in different views are independent.

2. **Inner DP** (row clustering): Within each view, another CRP independently clusters rows. Each cluster maintains sufficient statistics for its columns.

3. **Component models**: Each (cluster, column) pair uses a conjugate Bayesian model. Parameters are analytically integrated out — only cluster assignments and hyperparameters are stored.

<p align="center">
  <img src="diagrams/crosscat-state.svg" alt="CrossCatState Data Structure" width="650" />
</p>

### Why Two Levels?

Consider a dataset with salary, experience, zip code, and commute columns. A single clustering would try to find groups where all four columns are correlated. But salary/experience correlate differently than zip/commute — they have different clustering structures.

CrossCat solves this by:
- Grouping `{salary, experience}` into View 0 with its own row clustering
- Grouping `{zip, commute}` into View 1 with a different row clustering
- Row 42 can be "Senior" in View 0 and "Urban" in View 1 independently

### Collapsed Inference

All component model parameters are integrated out analytically. The sufficient statistics (count, sum, sum-of-squares, etc.) are enough to compute:

- **Log marginal likelihood**: p(data in cluster | hypers) — used for scoring
- **Posterior predictive**: p(new observation | data in cluster, hypers) — used for queries

This means the state only stores:
- Cluster assignments (integers)
- Hyperparameters (a few scalars per column)
- Sufficient statistics (derived from data + assignments)

No per-observation parameters. This is why CrossCat scales well.

## Module Architecture

<p align="center">
  <img src="diagrams/module-architecture.svg" alt="Module Architecture" width="800" />
</p>

### Data Flow

1. **`initialize()`** creates a `CrossCatState` by:
   - Sampling column-to-view assignments from CRP
   - Sampling row-to-cluster assignments per view from CRP
   - Computing data-driven hyperparameter defaults
   - Computing sufficient statistics for each (cluster, column)

2. **`gibbs_sweep()`** iteratively improves the state:
   - **Row assignment kernel**: For each view, for each row, remove from cluster → score all clusters → sample new assignment
   - **Column assignment kernel**: For each column, remove from view → score all views + new view → sample
   - **Hyperparameter kernel**: Grid-based Gibbs over hyper grids per column type
   - **CRP alpha kernel**: Grid-based sampling for concentration parameters

3. **Queries** read the posterior state to answer questions:
   - Mixture over clusters weighted by CRP counts
   - Posterior predictive from each component model
   - Cross-view independence for mutual information

## State Representation

### Original (Python lists)

```python
CrossCatState(
    column_assignments=jnp.array([0, 0, 1, 1]),  # 4 cols, 2 views
    column_crp_alpha=1.0,
    column_hypers=[ColumnHypers(...), ...],       # list of 4
    column_types=[CONTINUOUS, CONTINUOUS, ...],    # list of 4
    views=[
        ViewState(
            column_indices=jnp.array([0, 1]),
            row_assignments=jnp.array([0, 0, 1, ...]),  # 200 rows
            row_crp_alpha=1.0,
            suffstats=[[ss_c0_col0, ss_c0_col1], [ss_c1_col0, ss_c1_col1]],
        ),
        ViewState(...),
    ],
    n_rows=200, n_cols=4,
)
```

This uses Python lists — easy to work with but prevents `jax.jit`.

### Packed (padded arrays)

```python
PackedCrossCatState(
    column_assignments=jnp.array([0, 0, 1, 1]),
    view_row_assignments=jnp.zeros((16, 200)),    # (max_views, n_rows)
    view_n_clusters=jnp.array([2, 3, 0, ...]),    # (max_views,)
    ss_counts=jnp.zeros((16, 32, 8)),             # (max_views, max_clusters, max_cols_per_view)
    ss_sum_x=jnp.zeros((16, 32, 8)),
    ...
)
```

All views/clusters/columns padded to fixed dimensions. Invalid entries masked. This enables:
- `jax.jit` compilation of entire kernels
- `jax.vmap` over cluster or column dimensions
- `jnp.where` for column type dispatch (no Python branching)

## Gibbs Kernels in Detail

### Row Assignment (Critical Path)

For each view, for each row *i*:

1. Remove row *i* from its current cluster *c_old*
2. For each existing cluster *c* and the "new cluster" option:
   - **CRP prior**: log(count_c) for existing, log(alpha) for new
   - **Likelihood**: product over columns of posterior_predictive_logp(x_i | suffstats_c)
3. Sample new assignment from categorical(log_probs)
4. Compact cluster indices (remove empty clusters)

This is O(n_rows × n_clusters × n_cols_per_view) per view per sweep — the computational bottleneck.

### Column Assignment

For each column *j*:

1. Remove column *j* from its current view
2. For each existing view *v* and a proposed new view:
   - **CRP prior**: log(count_v) for existing, log(alpha) for new
   - **Likelihood**: total log_marginal_likelihood of column *j*'s data under view *v*'s row clustering
3. Sample new assignment
4. If new view selected, sample row clustering from CRP
5. Remove empty views, reindex

### Hyperparameter Sampling

Grid-based Gibbs following original CrossCat, with data-dependent ranges and N_GRID=31:
- **Continuous**: grids over s (log-spaced [SSD/100, SSD]), mu (linear [min, max]), nu (log-spaced [1, N]), r (log-spaced [1/N, N])
- **Categorical**: grid over dirichlet_alpha (log-spaced [1/N, N])
- **Binary**: grid over (alpha, beta) — 8x8 log-spaced [1/N, N]
- **Cyclic**: grids over kappa (linspace [kappa_est, N*kappa_est, 31]), vm_a (31 pts), vm_mu (31 pts)
- **CRP alphas**: separate inner/outer grids, log-spaced [1/N, N] scaled to row/column count

## Performance Optimizations (v0.9.0)

The packed kernels were optimized for 12x speedup via three techniques:

### 1. Vectorized Column Scoring

**Before**: `_score_row_one_cluster` used `lax.scan` to sequentially iterate over columns, computing one `unified_posterior_predictive_logp` per step. For 257 columns, this created a massive sequential XLA graph.

**After**: Replaced with `jax.vmap(unified_posterior_predictive_logp)` over all columns simultaneously. Column data, types, sufficient statistics, and hyperparameters are gathered into stacked arrays and scored in parallel.

### 2. Type-Specialized Scoring

**Before**: Every column score computed all 5 type results (NormalGamma, DirichletCategorical, BetaBernoulli, OrderedLogistic, VonMises) via nested `jnp.where`, wasting ~80% of computation for homogeneous-type views.

**After**: `_compute_dominant_type()` detects when all columns in a view share one type. When detected, `_score_row_one_cluster_typed` calls type-specific batch functions (`batch_bb_posterior_predictive_logp`, `batch_ng_posterior_predictive_logp`, `batch_dc_posterior_predictive_logp`) that skip the `jnp.where` dispatch entirely.

### 3. Batched Suffstat Updates

**Before**: `_add_row_to_suffstats` / `_remove_row_from_suffstats` used `lax.scan` to update one column at a time.

**After**: Uses batched `.at[cluster_id, li_range].add()` scatter operations over all columns simultaneously.

### Combined Effect

| Dataset | Before | After | Speedup |
|---------|--------|-------|---------|
| 50 × 11 | 25s/sweep | 4.5s/sweep | 5.5x |
| 100 × 65 | 38s/sweep | 4.8s/sweep | 7.9x |
| 1000 × 257 (MNIST) | 238s/sweep | 20s/sweep | 12x |

JIT compilation time also dropped from 20+ minutes to ~23 seconds for 257 columns because the XLA graph is much smaller (single vmap vs 257-step unrolled scan).

## JAX Design Patterns

### Deterministic RNG

All randomness flows through `jax.random.key()` and `jax.random.split()`. Every function receives a key and splits it for sub-operations. This ensures exact reproducibility.

### NaN Transparency

Missing data is represented as `NaN`. All sufficient statistic computations filter NaN before accumulation:

```python
clean = data[~jnp.isnan(data)]
count = clean.shape[0]
sum_x = jnp.sum(clean)
```

Posterior predictive queries skip NaN conditioning values. This means inference works seamlessly with incomplete data.

### Immutable State

All JAX arrays are immutable. Operations return new arrays:

```python
# This creates a new array — doesn't modify in-place
row_assignments = row_assignments.at[i].set(new_cluster)
```

`CrossCatState` fields are replaced wholesale when updated.
