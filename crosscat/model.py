"""CrossCat model initialization and joint probability.

Maps to the original State.h / State.cpp from probcomp/crosscat.
The original State class managed initialization from prior, transition
dispatching, and state serialization. Here we separate concerns:
- model.py: initialization and scoring
- gibbs.py: transition kernels
- types.py: state representation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln

from crosscat.components import BetaBernoulli, DirichletCategorical, NormalGamma, OrderedLogistic
from crosscat.types import ColumnHypers, ColumnType, CrossCatState, SufficientStats, ViewState


def _crp_sample(rng_key: Array, alpha: float, n: int) -> Array:
    """Sample assignments from a Chinese Restaurant Process.

    Sequential CRP: each item sits at an existing table with probability
    proportional to occupancy, or starts a new table with probability
    proportional to alpha.

    Args:
        rng_key: JAX PRNG key.
        alpha: CRP concentration parameter.
        n: Number of items to assign.

    Returns:
        Integer assignments array of shape (n,).
    """

    def _step(carry, key):
        assignments, counts, n_tables = carry
        # Probabilities: existing tables proportional to counts, new table proportional to alpha
        max_tables = n  # upper bound
        table_probs = jnp.where(jnp.arange(max_tables) < n_tables, counts[:max_tables], 0.0)
        # Probability of new table
        new_table_probs = table_probs.at[n_tables].set(alpha)
        log_probs = jnp.log(new_table_probs + 1e-30)
        chosen = jax.random.categorical(key, log_probs)

        # Update
        is_new = chosen == n_tables
        new_n_tables = jnp.where(is_new, n_tables + 1, n_tables)
        new_counts = counts.at[chosen].add(1.0)
        return (assignments, new_counts, new_n_tables), chosen

    keys = jax.random.split(rng_key, n)
    init_counts = jnp.zeros(n, dtype=jnp.float32)
    init_state = (jnp.zeros(n, dtype=jnp.int32), init_counts, jnp.array(0, dtype=jnp.int32))

    _, assignments = jax.lax.scan(_step, init_state, keys)
    return assignments


def _default_hypers(column_type: ColumnType, col_data: Array) -> ColumnHypers:
    """Initialize column hyperparameters from data-driven defaults.

    Following the original CrossCat initialization strategy:
    - Continuous: set mu to data mean, s to data variance, r=1, nu=2
    - Categorical: alpha = 1 (symmetric)
    - Binary: alpha=1, beta=1 (uniform prior)
    - Ordinal: cutpoints as linspace (but using Dirichlet internally)
    """
    if column_type == ColumnType.CONTINUOUS:
        mean = jnp.mean(col_data)
        var = jnp.var(col_data) + 1e-6
        return ColumnHypers(
            column_type=column_type,
            mu=mean,
            r=jnp.array(1.0),
            s=var,
            nu=jnp.array(2.0),
        )
    elif column_type == ColumnType.CATEGORICAL:
        return ColumnHypers(
            column_type=column_type,
            dirichlet_alpha=jnp.array(1.0),
        )
    elif column_type == ColumnType.BINARY:
        return ColumnHypers(
            column_type=column_type,
            alpha=jnp.array(1.0),
            beta=jnp.array(1.0),
        )
    elif column_type == ColumnType.ORDINAL:
        n_levels = int(jnp.max(col_data)) + 1
        return ColumnHypers(
            column_type=column_type,
            cutpoints=jnp.linspace(0.0, 1.0, n_levels - 1) if n_levels > 1 else None,
        )
    else:
        raise ValueError(f"Unknown column type: {column_type}")


def _compute_suffstats_for_view(
    data: Array,
    column_indices: Array,
    column_types: list[ColumnType],
    row_assignments: Array,
    n_clusters: int,
) -> list[list[SufficientStats]]:
    """Compute sufficient statistics for each (cluster, column) in a view.

    Returns:
        Nested list: suffstats[cluster_idx][col_idx_in_view]
    """
    all_suffstats = []
    for c in range(n_clusters):
        cluster_mask = row_assignments == c
        cluster_stats = []
        for local_idx in range(len(column_indices)):
            col_idx = int(column_indices[local_idx])
            col_type = column_types[col_idx]
            col_data = data[cluster_mask, col_idx]

            if col_type == ColumnType.CONTINUOUS:
                ss = NormalGamma.sufficient_statistics(col_data)
            elif col_type == ColumnType.CATEGORICAL:
                n_cats = int(jnp.max(data[:, col_idx])) + 1
                ss = DirichletCategorical.sufficient_statistics(col_data, n_cats)
            elif col_type == ColumnType.BINARY:
                ss = BetaBernoulli.sufficient_statistics(col_data)
            elif col_type == ColumnType.ORDINAL:
                n_levels = int(jnp.max(data[:, col_idx])) + 1
                ss = OrderedLogistic.sufficient_statistics(col_data, n_levels)
            else:
                raise ValueError(f"Unknown column type: {col_type}")
            cluster_stats.append(ss)
        all_suffstats.append(cluster_stats)
    return all_suffstats


def initialize(
    rng_key: Array,
    data: Array,
    column_types: list[ColumnType],
    *,
    n_chains: int = 1,
    column_crp_alpha: float = 1.0,
    row_crp_alpha: float = 1.0,
) -> CrossCatState | list[CrossCatState]:
    """Initialize CrossCat state(s) from the prior.

    Maps to original LocalEngine.initialize() which calls State.__init__
    with initialization='from_the_prior'.

    Procedure (following original CrossCat):
    1. Sample column-to-view assignments from CRP(column_crp_alpha)
    2. For each view, sample row-to-cluster assignments from CRP(row_crp_alpha)
    3. Compute sufficient statistics for each (cluster, column) pair
    4. Initialize column hyperparameters from data-driven defaults

    Args:
        rng_key: JAX PRNG key.
        data: Observation matrix, shape (n_rows, n_cols).
        column_types: Type specification per column.
        n_chains: Number of independent chains to initialize.
        column_crp_alpha: Initial outer DP concentration.
        row_crp_alpha: Initial inner DP concentration per view.

    Returns:
        Single CrossCatState if n_chains=1, else list of states.
    """
    n_rows, n_cols = data.shape

    def _init_one(key):
        k1, k2 = jax.random.split(key)

        # Step 1: Sample column-to-view assignments from CRP
        col_assignments = _crp_sample(k1, column_crp_alpha, n_cols)
        n_views = int(jnp.max(col_assignments)) + 1

        # Step 2: Initialize column hyperparameters
        col_hypers = []
        for j in range(n_cols):
            col_hypers.append(_default_hypers(column_types[j], data[:, j]))

        # Step 3: For each view, sample row assignments and compute suffstats
        view_keys = jax.random.split(k2, n_views)
        views = []
        for v in range(n_views):
            col_indices = jnp.where(col_assignments == v, size=n_cols)[0]
            # Filter to actual columns in this view
            col_mask = col_assignments == v
            col_indices = jnp.arange(n_cols)[col_mask]

            # Sample row-to-cluster assignments for this view
            row_assigns = _crp_sample(view_keys[v], row_crp_alpha, n_rows)
            n_clusters = int(jnp.max(row_assigns)) + 1

            # Compute sufficient statistics
            suffstats = _compute_suffstats_for_view(
                data, col_indices, column_types, row_assigns, n_clusters
            )

            views.append(
                ViewState(
                    column_indices=col_indices,
                    row_assignments=row_assigns,
                    row_crp_alpha=jnp.array(row_crp_alpha),
                    suffstats=suffstats,
                )
            )

        return CrossCatState(
            column_assignments=col_assignments,
            column_crp_alpha=jnp.array(column_crp_alpha),
            column_hypers=col_hypers,
            column_types=column_types,
            views=views,
            n_rows=n_rows,
            n_cols=n_cols,
        )

    if n_chains == 1:
        return _init_one(rng_key)

    keys = jax.random.split(rng_key, n_chains)
    return [_init_one(keys[i]) for i in range(n_chains)]


def _log_crp(assignments: Array, alpha: Array) -> Array:
    """Log probability of a partition under a CRP.

    log p(z | alpha) = K * log(alpha) + sum_k gammaln(n_k) - gammaln(N + alpha) + gammaln(alpha)

    where K = number of clusters, n_k = size of cluster k, N = total items.
    """
    n = assignments.shape[0]
    n_clusters = int(jnp.max(assignments)) + 1
    counts = jnp.bincount(assignments, length=n_clusters).astype(jnp.float32)

    log_p = (
        n_clusters * jnp.log(alpha)
        + jnp.sum(gammaln(counts))
        - gammaln(n + alpha)
        + gammaln(alpha)
    )
    return log_p


def log_joint(state: CrossCatState, data: Array) -> Array:
    """Compute the joint log probability of state and data.

    log p(state, data) = log p(column_partition | alpha_outer)
                       + sum_v log p(row_partition_v | alpha_v)
                       + sum_v sum_c sum_col log p(data[rows_c, col] | hypers_col)
                       + log p(alpha_outer) + sum_v log p(alpha_v)
                       + sum_col log p(hypers_col)

    Maps to scoring logic spread across original State.cpp, View.cpp,
    and numerics.cpp (calc_crp_logp, calc_continuous_logp, etc.).

    Args:
        state: Current CrossCat state.
        data: Observation matrix, shape (n_rows, n_cols).

    Returns:
        Scalar log joint probability.
    """
    log_p = jnp.array(0.0)

    # 1. Log CRP probability for column partition
    log_p = log_p + _log_crp(state.column_assignments, state.column_crp_alpha)

    # 2. For each view: log CRP for row partition + data likelihood
    for view in state.views:
        # Row partition CRP
        log_p = log_p + _log_crp(view.row_assignments, view.row_crp_alpha)

        # Data likelihood for each (cluster, column) pair
        if view.suffstats is not None:
            n_clusters = len(view.suffstats)
            for c in range(n_clusters):
                for local_idx, col_idx in enumerate(view.column_indices.tolist()):
                    col_idx = int(col_idx)
                    hypers = state.column_hypers[col_idx]
                    ss = view.suffstats[c][local_idx]
                    col_type = state.column_types[col_idx]

                    if col_type == ColumnType.CONTINUOUS:
                        log_p = log_p + NormalGamma.log_marginal_likelihood(ss, hypers)
                    elif col_type == ColumnType.CATEGORICAL:
                        log_p = log_p + DirichletCategorical.log_marginal_likelihood(ss, hypers)
                    elif col_type == ColumnType.BINARY:
                        log_p = log_p + BetaBernoulli.log_marginal_likelihood(ss, hypers)
                    elif col_type == ColumnType.ORDINAL:
                        log_p = log_p + OrderedLogistic.log_marginal_likelihood(ss, hypers)

    # 3. Gamma(1,1) prior on CRP alphas
    log_p = log_p - state.column_crp_alpha  # Exp(1) prior
    for view in state.views:
        log_p = log_p - view.row_crp_alpha

    return log_p
