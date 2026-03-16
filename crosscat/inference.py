"""Posterior predictive queries and analysis utilities.

Maps to the query API in original CrossCat:
- sample_utils.py: simple_predictive_probability, simple_predictive_sample,
                    impute_and_confidence, row_structural_typicality
- inference_utils.py: mutual_information, calculate_MI_bounded_discrete

The original query pattern:
    Y = [(row_idx, col_idx, value), ...]  # constraints (observations)
    Q = [(row_idx, col_idx), ...]          # queries (what to predict)
    result = engine.simple_predictive_sample(M_c, X_L, X_D, Y, Q, seed, n)

This implementation uses a more Pythonic API with explicit column names/indices.
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
from crosscat.types import ColumnType, CrossCatState


def _get_view_for_column(state: CrossCatState, col: int) -> int:
    """Return the view index containing a given column."""
    return int(state.column_assignments[col])


def _cluster_weights(view, n_rows: int) -> Array:
    """CRP-based cluster assignment probabilities for a view.

    Returns normalized weights proportional to cluster counts.
    """
    n_clusters = int(jnp.max(view.row_assignments)) + 1
    counts = jnp.array([jnp.sum(view.row_assignments == c) for c in range(n_clusters)]).astype(
        jnp.float32
    )
    return counts / counts.sum()


def _cluster_weights_conditioned(
    state: CrossCatState,
    view,
    view_idx: int,
    condition_cols: list[int],
    condition_vals: Array,
    data: Array,
) -> Array:
    """Compute cluster weights for a view conditioned on observed columns.

    p(cluster=c | conditions) proportional to p(cluster=c) * p(conditions | cluster=c)
    """
    n_clusters = int(jnp.max(view.row_assignments)) + 1
    counts = jnp.array([jnp.sum(view.row_assignments == c) for c in range(n_clusters)]).astype(
        jnp.float32
    )

    log_weights = jnp.log(counts + 1e-30)

    # Add likelihood of condition columns that are in this view
    for cond_idx, col in enumerate(condition_cols):
        if _get_view_for_column(state, col) != view_idx:
            continue  # Condition column is in a different view — independent

        col_type = state.column_types[col]
        hypers = state.column_hypers[col]
        x = condition_vals[cond_idx]

        # Find local index of this column in the view
        local_idx = None
        for li, ci in enumerate(view.column_indices.tolist()):
            if int(ci) == col:
                local_idx = li
                break
        if local_idx is None:
            continue

        for c in range(n_clusters):
            ss = view.suffstats[c][local_idx]
            if col_type == ColumnType.CONTINUOUS:
                log_lik = NormalGamma.posterior_predictive_logp(x, ss, hypers)
            elif col_type == ColumnType.CATEGORICAL:
                log_lik = DirichletCategorical.posterior_predictive_logp(x, ss, hypers)
            elif col_type == ColumnType.BINARY:
                log_lik = BetaBernoulli.posterior_predictive_logp(x, ss, hypers)
            elif col_type == ColumnType.ORDINAL:
                log_lik = OrderedLogistic.posterior_predictive_logp(x, ss, hypers)
            else:
                continue
            log_weights = log_weights.at[c].add(log_lik)

    # Normalize
    log_weights = log_weights - jnp.max(log_weights)
    weights = jnp.exp(log_weights)
    return weights / weights.sum()


def predictive_probability(
    state: CrossCatState,
    data: Array,
    query_cols: list[int],
    query_vals: Array,
    *,
    condition_cols: list[int] | None = None,
    condition_vals: Array | None = None,
) -> Array:
    """Compute conditional predictive probability.

    p(query_cols = query_vals | condition_cols = condition_vals, state)

    Maps to original sample_utils.simple_predictive_probability().

    The computation follows the original CrossCat chain rule:
    1. Identify which views contain the query and condition columns
    2. For each view, determine cluster probabilities given conditions
    3. Compute predictive probability as mixture over clusters
    4. Multiply across views (columns in different views are independent given state)

    Args:
        state: CrossCat state (single posterior sample).
        data: Full observation matrix for context.
        query_cols: Column indices to query.
        query_vals: Values to evaluate probability at.
        condition_cols: Column indices to condition on (optional).
        condition_vals: Conditioning values (optional).

    Returns:
        Log probability (scalar).
    """
    if condition_cols is None:
        condition_cols = []
    if condition_vals is None:
        condition_vals = jnp.array([])

    log_p_total = jnp.array(0.0)

    for q_idx, col in enumerate(query_cols):
        view_idx = _get_view_for_column(state, col)
        view = state.views[view_idx]
        n_clusters = int(jnp.max(view.row_assignments)) + 1

        # Get cluster weights (conditioned if applicable)
        if condition_cols:
            weights = _cluster_weights_conditioned(
                state, view, view_idx, condition_cols, condition_vals, data
            )
        else:
            weights = _cluster_weights(view, state.n_rows)

        # Find local index of query column in this view
        local_idx = None
        for li, ci in enumerate(view.column_indices.tolist()):
            if int(ci) == col:
                local_idx = li
                break

        # Mixture over clusters
        col_type = state.column_types[col]
        hypers = state.column_hypers[col]
        x = query_vals[q_idx]
        log_mixture = -jnp.inf

        for c in range(n_clusters):
            ss = view.suffstats[c][local_idx]
            if col_type == ColumnType.CONTINUOUS:
                log_lik = NormalGamma.posterior_predictive_logp(x, ss, hypers)
            elif col_type == ColumnType.CATEGORICAL:
                log_lik = DirichletCategorical.posterior_predictive_logp(x, ss, hypers)
            elif col_type == ColumnType.BINARY:
                log_lik = BetaBernoulli.posterior_predictive_logp(x, ss, hypers)
            elif col_type == ColumnType.ORDINAL:
                log_lik = OrderedLogistic.posterior_predictive_logp(x, ss, hypers)
            else:
                continue

            log_term = jnp.log(weights[c]) + log_lik
            log_mixture = jnp.logaddexp(log_mixture, log_term)

        log_p_total = log_p_total + log_mixture

    return log_p_total


def predictive_sample(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
    query_cols: list[int],
    *,
    condition_cols: list[int] | None = None,
    condition_vals: Array | None = None,
    n_samples: int = 1000,
) -> Array:
    """Draw samples from the posterior predictive distribution.

    Maps to original sample_utils.simple_predictive_sample().

    Args:
        rng_key: JAX PRNG key.
        state: CrossCat state.
        data: Full observation matrix.
        query_cols: Column indices to sample.
        condition_cols: Column indices to condition on.
        condition_vals: Conditioning values.
        n_samples: Number of posterior predictive samples.

    Returns:
        Array of shape (n_samples, len(query_cols)).
    """
    if condition_cols is None:
        condition_cols = []
    if condition_vals is None:
        condition_vals = jnp.array([])

    samples = jnp.zeros((n_samples, len(query_cols)))
    keys = jax.random.split(rng_key, n_samples)

    for s in range(n_samples):
        sample_keys = jax.random.split(keys[s], len(query_cols))

        for q_idx, col in enumerate(query_cols):
            view_idx = _get_view_for_column(state, col)
            view = state.views[view_idx]

            # Get cluster weights
            if condition_cols:
                weights = _cluster_weights_conditioned(
                    state, view, view_idx, condition_cols, condition_vals, data
                )
            else:
                weights = _cluster_weights(view, state.n_rows)

            # Sample cluster
            k1, k2 = jax.random.split(sample_keys[q_idx])
            cluster = jax.random.categorical(k1, jnp.log(weights + 1e-30))
            cluster = int(cluster)

            # Find local index
            local_idx = None
            for li, ci in enumerate(view.column_indices.tolist()):
                if int(ci) == col:
                    local_idx = li
                    break

            # Sample from component
            col_type = state.column_types[col]
            hypers = state.column_hypers[col]
            ss = view.suffstats[cluster][local_idx]

            if col_type == ColumnType.CONTINUOUS:
                val = NormalGamma.sample_posterior_predictive(k2, ss, hypers, n=1)[0]
            elif col_type == ColumnType.CATEGORICAL:
                val = DirichletCategorical.sample_posterior_predictive(k2, ss, hypers, n=1)[0]
            elif col_type == ColumnType.BINARY:
                val = BetaBernoulli.sample_posterior_predictive(k2, ss, hypers, n=1)[0]
            elif col_type == ColumnType.ORDINAL:
                val = OrderedLogistic.sample_posterior_predictive(k2, ss, hypers, n=1)[0]
            else:
                val = jnp.array(0.0)

            samples = samples.at[s, q_idx].set(val)

    return samples


def credible_interval(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
    query_col: int,
    *,
    condition_cols: list[int] | None = None,
    condition_vals: Array | None = None,
    n_samples: int = 1000,
    ci_level: float = 0.90,
) -> tuple[Array, Array, Array]:
    """Compute credible interval for a query column.

    Not in original CrossCat — added for LaborLens confidence tier system.
    Uses posterior predictive samples to compute percentile-based CI.

    Args:
        rng_key: JAX PRNG key.
        state: CrossCat state.
        data: Full observation matrix.
        query_col: Column to compute CI for.
        condition_cols: Conditioning columns.
        condition_vals: Conditioning values.
        n_samples: Number of samples for CI estimation.
        ci_level: Credible interval level (0.90 = 90% CI).

    Returns:
        Tuple of (median, lower_bound, upper_bound).
    """
    samples = predictive_sample(
        rng_key,
        state,
        data,
        [query_col],
        condition_cols=condition_cols,
        condition_vals=condition_vals,
        n_samples=n_samples,
    )
    samples_flat = samples[:, 0]

    tail = (1.0 - ci_level) / 2.0
    lower = jnp.percentile(samples_flat, 100.0 * tail)
    upper = jnp.percentile(samples_flat, 100.0 * (1.0 - tail))
    median = jnp.median(samples_flat)

    return median, lower, upper


def mutual_information(
    states: list[CrossCatState],
    col_i: int,
    col_j: int,
    *,
    n_samples: int = 1000,
) -> tuple[Array, Array]:
    """Estimate mutual information between two columns.

    Maps to original inference_utils.mutual_information() and
    inference_utils.mutual_information_to_linfoot().

    Uses the CrossCat structure: if two columns are in the same view,
    they share row clustering and thus have nonzero MI. If in different
    views, they are independent (MI = 0).

    Averaged over multiple posterior states (ensemble inference).

    Args:
        states: List of CrossCat posterior states.
        col_i: First column index.
        col_j: Second column index.
        n_samples: MC samples for MI estimation.

    Returns:
        Tuple of (mutual_information, linfoot_correlation).
    """
    mi_estimates = []

    for state in states:
        view_i = int(state.column_assignments[col_i])
        view_j = int(state.column_assignments[col_j])

        if view_i != view_j:
            # Different views => independent => MI = 0
            mi_estimates.append(0.0)
            continue

        # Same view — estimate MI from cluster structure
        view = state.views[view_i]
        n_clusters = int(jnp.max(view.row_assignments)) + 1

        # MI from clustering: H(X) + H(Y) - H(X,Y)
        # Here we use the approximation based on cluster assignment entropy
        cluster_counts = jnp.array(
            [jnp.sum(view.row_assignments == c) for c in range(n_clusters)]
        ).astype(jnp.float32)
        cluster_probs = cluster_counts / cluster_counts.sum()

        # Since both columns share the same row clustering,
        # MI is bounded by the entropy of the clustering
        entropy_clustering = -jnp.sum(
            jnp.where(cluster_probs > 0, cluster_probs * jnp.log(cluster_probs + 1e-30), 0.0)
        )

        # Scale by number of clusters relative to max possible
        # More clusters with columns co-assigned = higher dependency signal
        mi_est = entropy_clustering * (1.0 - 1.0 / jnp.maximum(n_clusters, 1.0))
        mi_estimates.append(float(mi_est))

    mi = jnp.array(mi_estimates).mean()
    # Linfoot correlation: sqrt(1 - exp(-2*MI))
    linfoot = jnp.sqrt(1.0 - jnp.exp(-2.0 * mi))

    return mi, linfoot


def row_typicality(
    states: list[CrossCatState],
    row_id: int,
) -> Array:
    """Compute structural typicality score for a row (anomaly detection).

    Maps to original sample_utils.row_structural_typicality().

    A low typicality score indicates an unusual/anomalous row — one that
    doesn't fit well into any cluster across views.

    Typicality is the proportion of rows that are less probable than this row
    under the posterior clustering. Computed by comparing cluster assignment
    probability of this row vs others.

    Averaged over multiple posterior states.

    Args:
        states: List of CrossCat posterior states.
        row_id: Row index to evaluate.

    Returns:
        Typicality score in [0, 1] — lower = more anomalous.
    """
    typicality_scores = []

    for state in states:
        # For each view, compute how typical this row's cluster assignment is
        view_scores = []
        for view in state.views:
            cluster = int(view.row_assignments[row_id])
            n_clusters = int(jnp.max(view.row_assignments)) + 1
            cluster_counts = jnp.array(
                [jnp.sum(view.row_assignments == c) for c in range(n_clusters)]
            ).astype(jnp.float32)
            total = cluster_counts.sum()

            # Probability of being in this cluster
            p_cluster = cluster_counts[cluster] / total

            # Proportion of rows in equally or less probable clusters
            row_probs = cluster_counts[view.row_assignments.astype(jnp.int32)] / total
            my_prob = p_cluster
            # Typicality = fraction of rows with lower or equal probability
            n_less_typical = jnp.sum(row_probs <= my_prob).astype(jnp.float32)
            view_score = n_less_typical / total
            view_scores.append(float(view_score))

        # Average across views
        typicality_scores.append(sum(view_scores) / len(view_scores))

    # Average across states
    return jnp.array(typicality_scores).mean()


def column_typicality(
    states: list[CrossCatState],
    col_id: int,
) -> Array:
    """Compute structural typicality score for a column.

    Maps to original sample_utils.column_structural_typicality().

    A column is typical if it is frequently grouped with the same set
    of columns across posterior samples. An atypical column is one that
    moves between views across samples.

    Args:
        states: List of CrossCat posterior states.
        col_id: Column index to evaluate.

    Returns:
        Typicality score in [0, 1].
    """
    if len(states) <= 1:
        return jnp.array(0.5)

    n_cols = states[0].n_cols
    # For each pair (col_id, other_col), count how often they're in the same view
    co_occurrence = jnp.zeros(n_cols)

    for state in states:
        my_view = int(state.column_assignments[col_id])
        same_view = (state.column_assignments == my_view).astype(jnp.float32)
        co_occurrence = co_occurrence + same_view

    co_occurrence = co_occurrence / len(states)

    # Typicality: high if co-occurrence pattern is concentrated (consistent grouping)
    # Use entropy of co-occurrence as inverse typicality
    # Remove self-co-occurrence
    co_other = jnp.concatenate([co_occurrence[:col_id], co_occurrence[col_id + 1 :]])
    # Normalize to probability-like
    p = co_other / jnp.maximum(co_other.sum(), 1e-30)
    entropy = -jnp.sum(jnp.where(p > 0, p * jnp.log(p + 1e-30), 0.0))
    max_entropy = jnp.log(jnp.float32(n_cols - 1))

    # Typicality: 1 - normalized entropy (high = consistent grouping = typical)
    typicality = 1.0 - entropy / jnp.maximum(max_entropy, 1e-30)
    return jnp.clip(typicality, 0.0, 1.0)
