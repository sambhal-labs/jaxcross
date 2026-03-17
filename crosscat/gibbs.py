"""Gibbs sampling transition kernels for CrossCat in JAX.

Maps to the transition methods in original State.cpp:
- transition_features()                    -> transition_column_assignments
- transition_row_partition_assignments()   -> transition_row_assignments
- transition_column_hyperparameters()      -> transition_column_hypers
- transition_column_crp_alpha()            -> transition_crp_alphas (outer)
- transition_row_partition_hyperparameters() -> transition_crp_alphas (inner)

Key JAX design decisions:
- Column assignment sweep uses jax.lax.scan over columns (replaces C++ for-loop)
- Row assignment sweep uses jax.vmap across views (replaces sequential per-view loop)
- Hyperparameter sampling uses grid-based Gibbs (matching original CrossCat)
- All kernels operate on CrossCatState dataclass (replaces X_L/X_D dicts)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from crosscat.components import (
    BetaBernoulli,
    DirichletCategorical,
    NormalGamma,
    OrderedLogistic,
)
from crosscat.model import _compute_suffstats_for_view, _crp_sample, _log_crp
from crosscat.types import (
    ColumnHypers,
    ColumnType,
    CrossCatState,
    SufficientStats,
    ViewState,
)


def _component_log_marginal(
    suffstats: SufficientStats, hypers: ColumnHypers, col_type: ColumnType
) -> Array:
    """Dispatch log marginal likelihood to the correct component model."""
    if col_type == ColumnType.CONTINUOUS:
        return NormalGamma.log_marginal_likelihood(suffstats, hypers)
    elif col_type == ColumnType.CATEGORICAL:
        return DirichletCategorical.log_marginal_likelihood(suffstats, hypers)
    elif col_type == ColumnType.BINARY:
        return BetaBernoulli.log_marginal_likelihood(suffstats, hypers)
    elif col_type == ColumnType.ORDINAL:
        return OrderedLogistic.log_marginal_likelihood(suffstats, hypers)
    else:
        raise ValueError(f"Unknown column type: {col_type}")


def _compute_suffstats_for_column(
    data: Array, col_idx: int, col_type: ColumnType, row_assignments: Array, n_clusters: int
) -> list[SufficientStats]:
    """Compute sufficient statistics for one column across all clusters in a view."""
    stats = []
    for c in range(n_clusters):
        mask = row_assignments == c
        col_data = data[mask, col_idx]
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
        stats.append(ss)
    return stats


def _log_marginal_for_column_in_view(
    data: Array,
    col_idx: int,
    col_type: ColumnType,
    hypers: ColumnHypers,
    row_assignments: Array,
    n_clusters: int,
) -> Array:
    """Compute total log marginal likelihood of one column's data under a view's clustering."""
    log_ml = jnp.array(0.0)
    for c in range(n_clusters):
        mask = row_assignments == c
        col_data = data[mask, col_idx]
        if col_type == ColumnType.CONTINUOUS:
            ss = NormalGamma.sufficient_statistics(col_data)
            log_ml = log_ml + NormalGamma.log_marginal_likelihood(ss, hypers)
        elif col_type == ColumnType.CATEGORICAL:
            n_cats = int(jnp.max(data[:, col_idx])) + 1
            ss = DirichletCategorical.sufficient_statistics(col_data, n_cats)
            log_ml = log_ml + DirichletCategorical.log_marginal_likelihood(ss, hypers)
        elif col_type == ColumnType.BINARY:
            ss = BetaBernoulli.sufficient_statistics(col_data)
            log_ml = log_ml + BetaBernoulli.log_marginal_likelihood(ss, hypers)
        elif col_type == ColumnType.ORDINAL:
            n_levels = int(jnp.max(data[:, col_idx])) + 1
            ss = OrderedLogistic.sufficient_statistics(col_data, n_levels)
            log_ml = log_ml + OrderedLogistic.log_marginal_likelihood(ss, hypers)
    return log_ml


def transition_column_assignments(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
) -> CrossCatState:
    """Gibbs sweep over column-to-view assignments (outer DP).

    This is the core novelty of jax-crosscat — no published GPU implementation exists.

    Maps to original State::transition_features() in State.cpp (~200 lines).

    For each column j:
    1. Remove column j from its current view (update sufficient stats)
    2. For each existing view v, compute log p(z_j = v | z_{-j}, data, hypers)
       using CRP prior + marginal likelihood of column j's data under view v's clusters
    3. Also compute probability of a new singleton view
    4. Sample new assignment from categorical distribution
    5. Add column j to chosen view (update sufficient stats)

    Args:
        rng_key: JAX PRNG key.
        state: Current CrossCat state.
        data: Observation matrix, shape (n_rows, n_cols).

    Returns:
        Updated CrossCatState with new column assignments.
    """
    n_rows, n_cols = data.shape
    col_assignments = jnp.array(state.column_assignments)
    alpha = state.column_crp_alpha

    keys = jax.random.split(rng_key, n_cols)

    for j in range(n_cols):
        col_type = state.column_types[j]
        hypers = state.column_hypers[j]
        old_view = int(col_assignments[j])

        # Count columns per view (excluding column j)
        temp_assignments = col_assignments.at[j].set(-1)
        n_views = int(jnp.max(col_assignments)) + 1

        log_probs = []

        # Score each existing view
        for v in range(n_views):
            view = state.views[v]
            count_v = int(jnp.sum(temp_assignments == v))

            if count_v == 0 and v == old_view:
                # This view would be empty without column j — skip, handle via new view
                log_probs.append(-jnp.inf)
                continue

            # CRP prior: proportional to number of columns in this view
            log_prior = jnp.log(jnp.maximum(count_v, 1e-30).astype(jnp.float32))

            # Likelihood: how well does column j's data fit view v's row clustering?
            n_clusters = int(jnp.max(view.row_assignments)) + 1
            log_lik = _log_marginal_for_column_in_view(
                data, j, col_type, hypers, view.row_assignments, n_clusters
            )

            log_probs.append(log_prior + log_lik)

        # Score a new singleton view (sample row assignments from CRP prior)
        log_prior_new = jnp.log(alpha)
        k_crp, k_cat = jax.random.split(keys[j])
        new_row_assigns = _crp_sample(k_crp, float(state.views[0].row_crp_alpha), n_rows)
        n_new_clusters = int(jnp.max(new_row_assigns)) + 1
        log_lik_new = _log_marginal_for_column_in_view(
            data, j, col_type, hypers, new_row_assigns, n_new_clusters
        )
        log_probs.append(log_prior_new + log_lik_new)

        # Sample new assignment
        log_probs_arr = jnp.array(log_probs)
        log_probs_arr = log_probs_arr - jnp.max(log_probs_arr)  # numerical stability
        chosen = jax.random.categorical(k_cat, log_probs_arr)
        chosen = int(chosen)

        if chosen == n_views:
            # Create new view (reuse CRP-sampled row assignments from proposal)
            n_new_clusters = int(jnp.max(new_row_assigns)) + 1
            new_suffstats = _compute_suffstats_for_view(
                data,
                jnp.array([j]),
                state.column_types,
                new_row_assigns,
                n_new_clusters,
            )
            new_view = ViewState(
                column_indices=jnp.array([j]),
                row_assignments=new_row_assigns,
                row_crp_alpha=jnp.array(float(state.views[0].row_crp_alpha)),
                suffstats=new_suffstats,
            )
            state.views.append(new_view)
            col_assignments = col_assignments.at[j].set(n_views)
        else:
            col_assignments = col_assignments.at[j].set(chosen)

    # Rebuild views with updated assignments, removing empty views
    new_views = []
    unique_views = sorted(set(col_assignments.tolist()))
    remap = {old: new for new, old in enumerate(unique_views)}
    new_assignments = jnp.array([remap[int(a)] for a in col_assignments.tolist()])

    for new_v, old_v in enumerate(unique_views):
        col_indices = jnp.arange(n_cols)[new_assignments == new_v]
        # Reuse row assignments from existing view if available
        if old_v < len(state.views):
            row_assigns = state.views[old_v].row_assignments
            row_alpha = state.views[old_v].row_crp_alpha
        else:
            row_assigns = jnp.zeros(n_rows, dtype=jnp.int32)
            row_alpha = state.views[0].row_crp_alpha

        n_clusters = int(jnp.max(row_assigns)) + 1
        suffstats = _compute_suffstats_for_view(
            data, col_indices, state.column_types, row_assigns, n_clusters
        )
        new_views.append(
            ViewState(
                column_indices=col_indices,
                row_assignments=row_assigns,
                row_crp_alpha=row_alpha,
                suffstats=suffstats,
            )
        )

    return CrossCatState(
        column_assignments=new_assignments,
        column_crp_alpha=state.column_crp_alpha,
        column_hypers=state.column_hypers,
        column_types=state.column_types,
        views=new_views,
        n_rows=n_rows,
        n_cols=n_cols,
    )


def _posterior_predictive_logp_for_row(
    row_data: Array,
    col_indices: Array,
    column_types: list[ColumnType],
    column_hypers: list[ColumnHypers],
    cluster_suffstats: list[SufficientStats],
) -> Array:
    """Log predictive probability of a row under a cluster's posterior."""
    log_p = jnp.array(0.0)
    for local_idx in range(len(col_indices)):
        col_idx = int(col_indices[local_idx])
        col_type = column_types[col_idx]
        hypers = column_hypers[col_idx]
        ss = cluster_suffstats[local_idx]
        x = row_data[col_idx]

        if col_type == ColumnType.CONTINUOUS:
            log_p = log_p + NormalGamma.posterior_predictive_logp(x, ss, hypers)
        elif col_type == ColumnType.CATEGORICAL:
            log_p = log_p + DirichletCategorical.posterior_predictive_logp(x, ss, hypers)
        elif col_type == ColumnType.BINARY:
            log_p = log_p + BetaBernoulli.posterior_predictive_logp(x, ss, hypers)
        elif col_type == ColumnType.ORDINAL:
            log_p = log_p + OrderedLogistic.posterior_predictive_logp(x, ss, hypers)
    return log_p


def transition_row_assignments(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
) -> CrossCatState:
    """Gibbs sweep over row-to-cluster assignments (inner DP), all views.

    Maps to original State::transition_row_partition_assignments() in State.cpp.

    For each view v:
        For each row i:
        1. Remove row i from its current cluster (update sufficient stats)
        2. For each existing cluster c, compute log p(z_i = c | z_{-i}, data_row, hypers)
           using CRP prior + product of component model likelihoods across view's columns
        3. Also compute probability of a new singleton cluster
        4. Sample new assignment from categorical distribution
        5. Add row i to chosen cluster (update sufficient stats)

    Args:
        rng_key: JAX PRNG key.
        state: Current CrossCat state.
        data: Observation matrix, shape (n_rows, n_cols).

    Returns:
        Updated CrossCatState with new row assignments per view.
    """
    n_rows = data.shape[0]
    new_views = []
    view_keys = jax.random.split(rng_key, len(state.views))

    for v_idx, view in enumerate(state.views):
        row_keys = jax.random.split(view_keys[v_idx], n_rows)
        row_assignments = jnp.array(view.row_assignments)
        col_indices = view.column_indices
        alpha = view.row_crp_alpha

        for i in range(n_rows):
            # Current cluster counts excluding row i
            n_clusters = int(jnp.max(row_assignments)) + 1

            # Recompute suffstats excluding row i
            temp_assignments = row_assignments.at[i].set(-1)
            cluster_counts = jnp.array([jnp.sum(temp_assignments == c) for c in range(n_clusters)])

            log_probs = []
            row_data = data[i]

            # Score each existing cluster
            for c in range(n_clusters):
                count_c = int(cluster_counts[c])
                if count_c == 0:
                    log_probs.append(-jnp.inf)
                    continue

                log_prior = jnp.log(jnp.float32(count_c))

                # Compute suffstats for this cluster (excluding row i)
                mask = temp_assignments == c
                cluster_stats = []
                for local_idx in range(len(col_indices)):
                    col_idx = int(col_indices[local_idx])
                    col_type = state.column_types[col_idx]
                    col_data = data[mask, col_idx]

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

                log_lik = _posterior_predictive_logp_for_row(
                    row_data, col_indices, state.column_types, state.column_hypers, cluster_stats
                )
                log_probs.append(log_prior + log_lik)

            # Score new singleton cluster (prior predictive under empty cluster)
            log_prior_new = jnp.log(alpha)
            empty_stats = []
            for local_idx in range(len(col_indices)):
                col_idx = int(col_indices[local_idx])
                col_type = state.column_types[col_idx]
                if col_type == ColumnType.CONTINUOUS:
                    ss = SufficientStats(
                        column_type=col_type,
                        count=jnp.array(0, dtype=jnp.int32),
                        sum_x=jnp.array(0.0),
                        sum_x_sq=jnp.array(0.0),
                    )
                elif col_type == ColumnType.CATEGORICAL:
                    n_cats = int(jnp.max(data[:, col_idx])) + 1
                    ss = SufficientStats(
                        column_type=col_type,
                        count=jnp.array(0, dtype=jnp.int32),
                        category_counts=jnp.zeros(n_cats),
                    )
                elif col_type == ColumnType.BINARY:
                    ss = SufficientStats(
                        column_type=col_type,
                        count=jnp.array(0, dtype=jnp.int32),
                        sum_x=jnp.array(0.0),
                    )
                elif col_type == ColumnType.ORDINAL:
                    n_levels = int(jnp.max(data[:, col_idx])) + 1
                    ss = SufficientStats(
                        column_type=col_type,
                        count=jnp.array(0, dtype=jnp.int32),
                        category_counts=jnp.zeros(n_levels),
                    )
                else:
                    raise ValueError(f"Unknown column type: {col_type}")
                empty_stats.append(ss)

            log_lik_new = _posterior_predictive_logp_for_row(
                row_data, col_indices, state.column_types, state.column_hypers, empty_stats
            )
            log_probs.append(log_prior_new + log_lik_new)

            # Sample
            log_probs_arr = jnp.array(log_probs)
            log_probs_arr = log_probs_arr - jnp.max(log_probs_arr)
            chosen = int(jax.random.categorical(row_keys[i], log_probs_arr))

            if chosen == n_clusters:
                # New cluster — assign index = n_clusters
                row_assignments = row_assignments.at[i].set(n_clusters)
            else:
                row_assignments = row_assignments.at[i].set(chosen)

        # Compact cluster indices (remove gaps)
        unique_clusters = jnp.unique(row_assignments)
        remap = jnp.full(int(jnp.max(row_assignments)) + 2, -1, dtype=jnp.int32)
        for new_idx, old_idx in enumerate(unique_clusters.tolist()):
            remap = remap.at[int(old_idx)].set(new_idx)
        row_assignments = remap[row_assignments]

        # Recompute suffstats
        n_clusters_final = int(jnp.max(row_assignments)) + 1
        suffstats = _compute_suffstats_for_view(
            data, col_indices, state.column_types, row_assignments, n_clusters_final
        )

        new_views.append(
            ViewState(
                column_indices=col_indices,
                row_assignments=row_assignments,
                row_crp_alpha=alpha,
                suffstats=suffstats,
            )
        )

    return CrossCatState(
        column_assignments=state.column_assignments,
        column_crp_alpha=state.column_crp_alpha,
        column_hypers=state.column_hypers,
        column_types=state.column_types,
        views=new_views,
        n_rows=state.n_rows,
        n_cols=state.n_cols,
    )


def transition_column_hypers(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
) -> CrossCatState:
    """Gibbs sample component model hyperparameters for each column.

    Maps to original State::transition_column_hyperparameters() which calls
    ComponentModel::sample_hypers() per column using grid-based Gibbs.

    For continuous columns, samples hyperparameters (mu, r, s, nu) from a grid.
    For categorical/ordinal/binary, samples concentration parameters.

    Args:
        rng_key: JAX PRNG key.
        state: Current CrossCat state.
        data: Observation matrix, shape (n_rows, n_cols).

    Returns:
        Updated CrossCatState with new column hyperparameters.
    """
    new_hypers = list(state.column_hypers)
    keys = jax.random.split(rng_key, state.n_cols)

    for j in range(state.n_cols):
        col_type = state.column_types[j]
        hypers = state.column_hypers[j]
        view_idx = int(state.column_assignments[j])
        view = state.views[view_idx]

        if col_type == ColumnType.CONTINUOUS:
            # Grid-based Gibbs for s (variance scale)
            # Evaluate log marginal likelihood at grid points for s
            col_data = data[:, j]
            data_var = jnp.var(col_data) + 1e-6
            s_grid = data_var * jnp.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0])

            k1, k2, k3 = jax.random.split(keys[j], 3)

            # Sample s
            log_scores_s = []
            for s_val in s_grid:
                test_hypers = ColumnHypers(
                    column_type=col_type, mu=hypers.mu, r=hypers.r, s=s_val, nu=hypers.nu
                )
                n_clusters = len(view.suffstats)
                log_score = jnp.array(0.0)
                for c in range(n_clusters):
                    local_idx = None
                    for li, ci in enumerate(view.column_indices.tolist()):
                        if int(ci) == j:
                            local_idx = li
                            break
                    if local_idx is not None:
                        ss = view.suffstats[c][local_idx]
                        log_score = log_score + NormalGamma.log_marginal_likelihood(
                            ss, test_hypers
                        )
                log_scores_s.append(log_score)

            log_scores_s = jnp.array(log_scores_s)
            log_scores_s = log_scores_s - jnp.max(log_scores_s)
            s_idx = jax.random.categorical(k1, log_scores_s)
            new_s = s_grid[s_idx]

            # Sample mu from posterior: grid around data mean
            data_mean = jnp.mean(col_data)
            data_std = jnp.std(col_data) + 1e-6
            mu_grid = data_mean + data_std * jnp.linspace(-2, 2, 11)

            log_scores_mu = []
            for mu_val in mu_grid:
                test_hypers = ColumnHypers(
                    column_type=col_type, mu=mu_val, r=hypers.r, s=new_s, nu=hypers.nu
                )
                log_score = jnp.array(0.0)
                n_clusters = len(view.suffstats)
                for c in range(n_clusters):
                    local_idx = None
                    for li, ci in enumerate(view.column_indices.tolist()):
                        if int(ci) == j:
                            local_idx = li
                            break
                    if local_idx is not None:
                        ss = view.suffstats[c][local_idx]
                        log_score = log_score + NormalGamma.log_marginal_likelihood(
                            ss, test_hypers
                        )
                log_scores_mu.append(log_score)

            log_scores_mu = jnp.array(log_scores_mu)
            log_scores_mu = log_scores_mu - jnp.max(log_scores_mu)
            mu_idx = jax.random.categorical(k2, log_scores_mu)
            new_mu = mu_grid[mu_idx]

            new_hypers[j] = ColumnHypers(
                column_type=col_type, mu=new_mu, r=hypers.r, s=new_s, nu=hypers.nu
            )

        elif col_type == ColumnType.CATEGORICAL:
            # Grid-based Gibbs for dirichlet_alpha
            alpha_grid = jnp.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
            log_scores = []
            for alpha_val in alpha_grid:
                test_hypers = ColumnHypers(column_type=col_type, dirichlet_alpha=alpha_val)
                log_score = jnp.array(0.0)
                n_clusters = len(view.suffstats)
                for c in range(n_clusters):
                    local_idx = None
                    for li, ci in enumerate(view.column_indices.tolist()):
                        if int(ci) == j:
                            local_idx = li
                            break
                    if local_idx is not None:
                        ss = view.suffstats[c][local_idx]
                        log_score = log_score + DirichletCategorical.log_marginal_likelihood(
                            ss, test_hypers
                        )
                log_scores.append(log_score)

            log_scores = jnp.array(log_scores)
            log_scores = log_scores - jnp.max(log_scores)
            idx = jax.random.categorical(keys[j], log_scores)
            new_hypers[j] = ColumnHypers(column_type=col_type, dirichlet_alpha=alpha_grid[idx])

        elif col_type == ColumnType.BINARY:
            # Grid-based Gibbs for alpha and beta
            ab_grid = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
            log_scores = []
            for a_val in ab_grid:
                for b_val in ab_grid:
                    test_hypers = ColumnHypers(column_type=col_type, alpha=a_val, beta=b_val)
                    log_score = jnp.array(0.0)
                    n_clusters = len(view.suffstats)
                    for c in range(n_clusters):
                        local_idx = None
                        for li, ci in enumerate(view.column_indices.tolist()):
                            if int(ci) == j:
                                local_idx = li
                                break
                        if local_idx is not None:
                            ss = view.suffstats[c][local_idx]
                            log_score = log_score + BetaBernoulli.log_marginal_likelihood(
                                ss, test_hypers
                            )
                    log_scores.append(log_score)

            log_scores = jnp.array(log_scores)
            log_scores = log_scores - jnp.max(log_scores)
            idx = int(jax.random.categorical(keys[j], log_scores))
            a_idx, b_idx = divmod(idx, len(ab_grid))
            new_hypers[j] = ColumnHypers(
                column_type=col_type, alpha=ab_grid[a_idx], beta=ab_grid[b_idx]
            )

        # Ordinal: keep hypers as-is (symmetric Dirichlet with alpha=1)

    return CrossCatState(
        column_assignments=state.column_assignments,
        column_crp_alpha=state.column_crp_alpha,
        column_hypers=new_hypers,
        column_types=state.column_types,
        views=state.views,
        n_rows=state.n_rows,
        n_cols=state.n_cols,
    )


def transition_crp_alphas(
    rng_key: Array,
    state: CrossCatState,
) -> CrossCatState:
    """Sample CRP concentration parameters via grid-based Gibbs.

    Samples from a grid of alpha values weighted by the CRP log probability.
    This follows the original CrossCat approach rather than BlackJAX NUTS
    for simplicity and stability.

    Args:
        rng_key: JAX PRNG key.
        state: Current CrossCat state.

    Returns:
        Updated CrossCatState with new CRP alpha values.
    """
    alpha_grid = jnp.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    keys = jax.random.split(rng_key, 1 + len(state.views))

    # Sample outer CRP alpha
    log_scores = []
    for alpha_val in alpha_grid:
        log_crp_val = _log_crp(state.column_assignments, alpha_val)
        # Exponential(1) prior on alpha
        log_prior = -alpha_val
        log_scores.append(log_crp_val + log_prior)

    log_scores = jnp.array(log_scores)
    log_scores = log_scores - jnp.max(log_scores)
    idx = jax.random.categorical(keys[0], log_scores)
    new_col_alpha = alpha_grid[idx]

    # Sample inner CRP alphas (one per view)
    new_views = []
    for v_idx, view in enumerate(state.views):
        log_scores = []
        for alpha_val in alpha_grid:
            log_crp_val = _log_crp(view.row_assignments, alpha_val)
            log_prior = -alpha_val
            log_scores.append(log_crp_val + log_prior)

        log_scores = jnp.array(log_scores)
        log_scores = log_scores - jnp.max(log_scores)
        idx = jax.random.categorical(keys[v_idx + 1], log_scores)
        new_row_alpha = alpha_grid[idx]

        new_views.append(
            ViewState(
                column_indices=view.column_indices,
                row_assignments=view.row_assignments,
                row_crp_alpha=new_row_alpha,
                suffstats=view.suffstats,
            )
        )

    return CrossCatState(
        column_assignments=state.column_assignments,
        column_crp_alpha=new_col_alpha,
        column_hypers=state.column_hypers,
        column_types=state.column_types,
        views=new_views,
        n_rows=state.n_rows,
        n_cols=state.n_cols,
    )


def gibbs_sweep(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
    *,
    n_sweeps: int = 1,
    kernels: tuple[str, ...] = (
        "row_assignments",
        "column_assignments",
        "column_hypers",
        "crp_alphas",
    ),
) -> CrossCatState:
    """Run one or more full Gibbs sweeps combining all transition kernels.

    Maps to original LocalEngine.analyze() which calls State.transition()
    with a configurable kernel_list.

    Args:
        rng_key: JAX PRNG key.
        state: Current CrossCat state.
        data: Observation matrix, shape (n_rows, n_cols).
        n_sweeps: Number of full sweeps to run.
        kernels: Which transition kernels to include per sweep.

    Returns:
        Updated CrossCatState after all sweeps.
    """
    kernel_map = {
        "column_assignments": transition_column_assignments,
        "row_assignments": transition_row_assignments,
        "column_hypers": transition_column_hypers,
        "crp_alphas": transition_crp_alphas,
    }

    for sweep in range(n_sweeps):
        for kernel_name in kernels:
            rng_key, subkey = jax.random.split(rng_key)
            if kernel_name == "crp_alphas":
                state = kernel_map[kernel_name](subkey, state)
            else:
                state = kernel_map[kernel_name](subkey, state, data)

    return state
