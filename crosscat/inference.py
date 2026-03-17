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

from crosscat.components import get_component
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

        # Skip NaN conditioning values
        if jnp.isnan(x):
            continue

        # Find local index of this column in the view
        local_idx = None
        for li, ci in enumerate(view.column_indices.tolist()):
            if int(ci) == col:
                local_idx = li
                break
        if local_idx is None:
            continue

        comp = get_component(col_type)
        for c in range(n_clusters):
            ss = view.suffstats[c][local_idx]
            log_lik = comp.posterior_predictive_logp(x, ss, hypers)
            log_weights = log_weights.at[c].add(log_lik)

    # Normalize
    log_weights = log_weights - jnp.max(log_weights)
    weights = jnp.exp(log_weights)
    return weights / weights.sum()


def _cluster_weights_for_observed_row(view, row_id: int) -> Array:
    """Get cluster weights for an observed row based on its actual assignment.

    For observed rows, the original CrossCat uses the actual cluster assignment
    rather than marginalizing over clusters. This returns a one-hot weight
    vector for the row's assigned cluster.
    """
    cluster = int(view.row_assignments[row_id])
    n_clusters = int(jnp.max(view.row_assignments)) + 1
    weights = jnp.zeros(n_clusters)
    return weights.at[cluster].set(1.0)


def predictive_probability(
    state: CrossCatState,
    data: Array,
    query_cols: list[int],
    query_vals: Array,
    *,
    condition_cols: list[int] | None = None,
    condition_vals: Array | None = None,
    row_id: int | None = None,
) -> Array:
    """Compute conditional predictive probability.

    p(query_cols = query_vals | condition_cols = condition_vals, state)

    Maps to original sample_utils.simple_predictive_probability().

    Args:
        state: CrossCat state (single posterior sample).
        data: Full observation matrix for context.
        query_cols: Column indices to query.
        query_vals: Values to evaluate probability at.
        condition_cols: Column indices to condition on (optional).
        condition_vals: Conditioning values (optional).
        row_id: If provided, use the observed row's actual cluster assignment
            rather than marginalizing over clusters (observed row distinction).

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

        # Get cluster weights
        if row_id is not None:
            weights = _cluster_weights_for_observed_row(view, row_id)
        elif condition_cols:
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

        comp = get_component(col_type)
        for c in range(n_clusters):
            ss = view.suffstats[c][local_idx]
            log_lik = comp.posterior_predictive_logp(x, ss, hypers)
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
    row_id: int | None = None,
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
        row_id: If provided, use observed row's cluster assignment.

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
            if row_id is not None:
                weights = _cluster_weights_for_observed_row(view, row_id)
            elif condition_cols:
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

            comp = get_component(col_type)
            val = comp.sample_posterior_predictive(k2, ss, hypers, n=1)[0]

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


def row_similarity(
    states: list[CrossCatState],
    row_a: int,
    row_b: int,
    *,
    target_columns: list[int] | None = None,
) -> Array:
    """Compute similarity between two rows.

    Maps to original LocalEngine.similarity().

    Similarity is the probability that two rows are in the same cluster,
    averaged over views and posterior samples. If target_columns is given,
    only views containing those columns contribute.

    Args:
        states: List of CrossCat posterior states.
        row_a: First row index.
        row_b: Second row index.
        target_columns: Restrict to views containing these columns (optional).

    Returns:
        Similarity score in [0, 1].
    """
    sim_scores = []

    for state in states:
        view_scores = []
        for v_idx, view in enumerate(state.views):
            # If target_columns specified, skip views not containing them
            if target_columns is not None:
                view_cols = set(view.column_indices.tolist())
                if not any(c in view_cols for c in target_columns):
                    continue

            # Same cluster = similar
            same_cluster = float(view.row_assignments[row_a] == view.row_assignments[row_b])
            view_scores.append(same_cluster)

        if view_scores:
            sim_scores.append(sum(view_scores) / len(view_scores))

    if not sim_scores:
        return jnp.array(0.0)
    return jnp.array(sum(sim_scores) / len(sim_scores))


def impute_and_confidence(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
    query_col: int,
    *,
    condition_cols: list[int] | None = None,
    condition_vals: Array | None = None,
    row_id: int | None = None,
    n_samples: int = 1000,
) -> tuple[Array, Array]:
    """Impute a value with confidence score.

    Maps to original sample_utils.impute_and_confidence().

    For continuous columns: returns median as point estimate, confidence from
    mixture weight concentration.
    For categorical/binary/ordinal: returns mode, confidence = mode probability.

    Args:
        rng_key: JAX PRNG key.
        state: CrossCat state.
        data: Full observation matrix.
        query_col: Column to impute.
        condition_cols: Conditioning columns.
        condition_vals: Conditioning values.
        row_id: Observed row index (uses actual cluster assignment).
        n_samples: Number of samples for estimation.

    Returns:
        Tuple of (point_estimate, confidence_score).
    """
    col_type = state.column_types[query_col]
    samples = predictive_sample(
        rng_key,
        state,
        data,
        [query_col],
        condition_cols=condition_cols,
        condition_vals=condition_vals,
        n_samples=n_samples,
        row_id=row_id,
    )
    s = samples[:, 0]

    if col_type == ColumnType.CONTINUOUS:
        point_est = jnp.median(s)
        # Confidence: inverse of normalized IQR
        iqr = jnp.percentile(s, 75) - jnp.percentile(s, 25)
        std = jnp.std(s) + 1e-30
        confidence = jnp.exp(-iqr / std)
    else:
        # Categorical, ordinal, binary: mode and mode frequency
        s_int = s.astype(jnp.int32)
        max_val = int(jnp.max(s_int)) + 1
        counts = jnp.bincount(s_int, length=max_val)
        point_est = jnp.argmax(counts).astype(jnp.float32)
        confidence = counts[jnp.argmax(counts)] / jnp.float32(n_samples)

    return point_est, confidence


def predictive_anomalousness(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
    query_row: int,
    *,
    n_samples: int = 1000,
) -> Array:
    """Compute predictive anomalousness of a row.

    Maps to original LocalEngine.predictive_anomalousness().

    Measures how surprising each column value is under the posterior predictive,
    then aggregates into an overall anomaly score.

    Args:
        rng_key: JAX PRNG key.
        state: CrossCat state.
        data: Full observation matrix.
        query_row: Row index to score.
        n_samples: Number of MC samples for scoring.

    Returns:
        Anomaly score in [0, 1] — higher = more anomalous.
    """
    log_p_total = jnp.array(0.0)
    n_scored = 0

    for col in range(state.n_cols):
        x = data[query_row, col]
        if jnp.isnan(x):
            continue

        log_p = predictive_probability(
            state,
            data,
            [col],
            jnp.array([x]),
            row_id=query_row,
        )
        log_p_total = log_p_total + log_p
        n_scored += 1

    if n_scored == 0:
        return jnp.array(0.5)

    # Convert to anomaly score: compare against samples
    avg_log_p = log_p_total / n_scored
    # Use sigmoid-like transform: more negative log_p = more anomalous
    anomaly = 1.0 / (1.0 + jnp.exp(avg_log_p + 2.0))
    return jnp.clip(anomaly, 0.0, 1.0)


def joint_predictive_probability(
    state: CrossCatState,
    data: Array,
    query_cols: list[int],
    query_vals: Array,
    *,
    condition_cols: list[int] | None = None,
    condition_vals: Array | None = None,
) -> Array:
    """Compute joint predictive probability via chain rule.

    Maps to original LocalEngine.predictive_probability() which uses
    the chain rule to decompose joint probabilities.

    p(q1, q2, ..., qn | conditions) = p(q1 | conditions) *
        p(q2 | conditions, q1) * p(q3 | conditions, q1, q2) * ...

    Args:
        state: CrossCat state.
        data: Full observation matrix.
        query_cols: Column indices to query.
        query_vals: Values to evaluate joint probability at.
        condition_cols: Conditioning columns.
        condition_vals: Conditioning values.

    Returns:
        Log joint probability (scalar).
    """
    if condition_cols is None:
        condition_cols = []
    if condition_vals is None:
        condition_vals = jnp.array([])

    log_p_joint = jnp.array(0.0)
    running_cond_cols = list(condition_cols)
    running_cond_vals = list(condition_vals.tolist()) if condition_vals.size > 0 else []

    for q_idx, col in enumerate(query_cols):
        # p(q_idx | conditions so far)
        cond_vals_arr = jnp.array(running_cond_vals) if running_cond_vals else jnp.array([])
        cond_cols_list = running_cond_cols if running_cond_cols else None
        cond_vals_list = cond_vals_arr if running_cond_cols else None

        log_p = predictive_probability(
            state,
            data,
            [col],
            jnp.array([query_vals[q_idx]]),
            condition_cols=cond_cols_list,
            condition_vals=cond_vals_list,
        )
        log_p_joint = log_p_joint + log_p

        # Add this query to the running conditions
        running_cond_cols.append(col)
        running_cond_vals.append(float(query_vals[q_idx]))

    return log_p_joint


def predictive_cdf(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
    query_col: int,
    query_val: Array,
    *,
    condition_cols: list[int] | None = None,
    condition_vals: Array | None = None,
    row_id: int | None = None,
    n_samples: int = 10000,
) -> Array:
    """Compute the posterior predictive CDF: P(X <= query_val | conditions).

    For continuous and cyclic columns, estimated via Monte Carlo sampling.
    For categorical, ordinal, and binary columns, computed analytically
    by summing probabilities of categories <= query_val.

    Args:
        rng_key: JAX PRNG key.
        state: CrossCat state.
        data: Full observation matrix.
        query_col: Column index to evaluate CDF for.
        query_val: Value at which to evaluate CDF.
        condition_cols: Conditioning columns.
        condition_vals: Conditioning values.
        row_id: If provided, use observed row's cluster assignment.
        n_samples: Number of MC samples (for continuous/cyclic columns).

    Returns:
        CDF value P(X <= query_val) in [0, 1].
    """
    col_type = state.column_types[query_col]

    if col_type in (
        ColumnType.CATEGORICAL,
        ColumnType.ORDINAL,
        ColumnType.BINARY,
    ):
        # Analytic CDF: sum p(x=k) for k <= query_val
        view_idx = _get_view_for_column(state, query_col)
        view = state.views[view_idx]
        n_clusters = int(jnp.max(view.row_assignments)) + 1

        # Get cluster weights
        if row_id is not None:
            weights = _cluster_weights_for_observed_row(view, row_id)
        elif condition_cols:
            weights = _cluster_weights_conditioned(
                state, view, view_idx,
                condition_cols or [], condition_vals, data
            )
        else:
            weights = _cluster_weights(view, state.n_rows)

        # Find local index
        local_idx = None
        for li, ci in enumerate(view.column_indices.tolist()):
            if int(ci) == query_col:
                local_idx = li
                break

        # Determine max category
        if col_type == ColumnType.BINARY:
            max_cat = 2
        else:
            # Get from suffstats of first cluster
            max_cat = int(view.suffstats[0][local_idx].category_counts.shape[0])

        # Sum p(x=k) for k <= query_val
        comp = get_component(col_type)
        hypers = state.column_hypers[query_col]
        cdf_val = jnp.array(0.0)
        for k in range(max_cat):
            if k > int(query_val):
                break
            log_p_k = jnp.array(-jnp.inf)
            for c in range(n_clusters):
                ss = view.suffstats[c][local_idx]
                log_lik = comp.posterior_predictive_logp(
                    jnp.array(float(k)), ss, hypers
                )
                log_term = jnp.log(weights[c]) + log_lik
                log_p_k = jnp.logaddexp(log_p_k, log_term)
            cdf_val = cdf_val + jnp.exp(log_p_k)

        return jnp.clip(cdf_val, 0.0, 1.0)

    else:
        # Continuous / Cyclic: MC estimate
        samples = predictive_sample(
            rng_key,
            state,
            data,
            [query_col],
            condition_cols=condition_cols,
            condition_vals=condition_vals,
            n_samples=n_samples,
            row_id=row_id,
        )
        return jnp.mean(samples[:, 0] <= query_val)


def sample_and_insert(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
    partial_row: Array,
) -> tuple[CrossCatState, Array, Array]:
    """Sample missing values for a partial row and insert into the state.

    Combines predictive_sample and insert_rows: for each NaN entry in the
    partial row, draw a sample from the posterior predictive conditioned on
    the observed entries, then insert the completed row.

    Useful for data augmentation, active learning, and what-if analysis.

    Args:
        rng_key: JAX PRNG key.
        state: CrossCat state.
        data: Full observation matrix, shape (n_rows, n_cols).
        partial_row: 1D array of shape (n_cols,) with NaN for missing entries.

    Returns:
        Tuple of (updated_state, updated_data, completed_row) where
        completed_row has NaN entries filled with posterior predictive samples.
    """
    from crosscat.model import insert_rows

    observed_mask = ~jnp.isnan(partial_row)
    observed_cols = [int(i) for i in range(len(partial_row)) if observed_mask[i]]
    missing_cols = [int(i) for i in range(len(partial_row)) if not observed_mask[i]]

    completed = jnp.array(partial_row, copy=True)

    if missing_cols:
        observed_vals = (
            jnp.array([float(partial_row[c]) for c in observed_cols])
            if observed_cols
            else None
        )
        cond_cols = observed_cols if observed_cols else None

        k1, k2 = jax.random.split(rng_key)
        samples = predictive_sample(
            k1,
            state,
            data,
            missing_cols,
            condition_cols=cond_cols,
            condition_vals=observed_vals,
            n_samples=1,
        )
        for idx, col in enumerate(missing_cols):
            completed = completed.at[col].set(samples[0, idx])
    else:
        k2 = rng_key

    new_row = completed.reshape(1, -1)
    new_state, new_data = insert_rows(k2, state, data, new_row)

    return new_state, new_data, completed


def conditional_entropy(
    rng_key: Array,
    states: list[CrossCatState],
    data: Array,
    target_col: int,
    given_cols: list[int],
    *,
    n_samples: int = 500,
) -> Array:
    """Estimate conditional entropy H(target | given).

    Maps to original LocalEngine.conditional_entropy().

    Args:
        rng_key: JAX PRNG key.
        states: List of posterior states.
        data: Observation matrix.
        target_col: Column whose entropy to compute.
        given_cols: Conditioning columns.
        n_samples: Number of MC samples.

    Returns:
        Conditional entropy estimate (nats).
    """
    entropy_estimates = []
    keys = jax.random.split(rng_key, len(states))

    for s_idx, state in enumerate(states):
        s_keys = jax.random.split(keys[s_idx], n_samples)
        log_ps = []
        for i in range(n_samples):
            # Sample conditioning values from marginal
            cond_samples = predictive_sample(
                s_keys[i], state, data, given_cols, n_samples=1
            )
            cond_vals = cond_samples[0]

            # Sample target given conditions
            k1, k2 = jax.random.split(jax.random.fold_in(s_keys[i], 999))
            target_samples = predictive_sample(
                k1, state, data, [target_col],
                condition_cols=given_cols, condition_vals=cond_vals, n_samples=1
            )
            target_val = target_samples[0, 0]

            # Evaluate log prob
            log_p = predictive_probability(
                state, data, [target_col], jnp.array([target_val]),
                condition_cols=given_cols, condition_vals=cond_vals
            )
            log_ps.append(float(log_p))

        entropy_estimates.append(-jnp.mean(jnp.array(log_ps)))

    return jnp.mean(jnp.array(entropy_estimates))
