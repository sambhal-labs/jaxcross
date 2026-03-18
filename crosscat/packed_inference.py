"""Vectorized inference functions operating on PackedCrossCatState.

Parallel counterparts of crosscat.inference functions, using JAX vmap/lax.scan
instead of Python for-loops in the core computation paths.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from crosscat.packed import (
    PackedCrossCatState,
    unified_posterior_predictive_logp,
    unified_sample_posterior_predictive,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _find_local_col_index(packed: PackedCrossCatState, view_idx: int, col_idx: int) -> Array:
    """Find local index of col_idx within view_idx's column list."""
    col_list = packed.view_column_indices[view_idx]
    return jnp.argmax(col_list == col_idx)


def _cluster_weights_packed(packed: PackedCrossCatState, view_idx: int) -> Array:
    """CRP-based cluster weights for a view.

    Returns normalized weights of shape (max_clusters,).
    """
    assigns = packed.view_row_assignments[view_idx]
    max_k = packed.max_clusters
    # bincount via one-hot reduction (JIT-safe)
    one_hot = jax.nn.one_hot(assigns, max_k)  # (n_rows, max_k)
    counts = one_hot.sum(axis=0).astype(jnp.float32)
    return counts / jnp.maximum(counts.sum(), 1e-30)


def _cluster_weights_for_row(packed: PackedCrossCatState, view_idx: int, row_id: int) -> Array:
    """One-hot cluster weights for an observed row."""
    cluster = packed.view_row_assignments[view_idx, row_id]
    return jax.nn.one_hot(cluster, packed.max_clusters).astype(jnp.float32)


def _logp_one_column_mixture(
    packed: PackedCrossCatState,
    view_idx: int,
    col_idx: int,
    x: Array,
    weights: Array,
) -> Array:
    """Log probability of x under the cluster mixture for col_idx in view_idx.

    Uses vmap over clusters to avoid Python loops.
    """
    local_idx = _find_local_col_index(packed, view_idx, col_idx)
    max_k = packed.max_clusters

    # Gather per-cluster sufficient statistics for this local column
    ss_counts_col = packed.ss_counts[view_idx, :, local_idx].astype(jnp.float32)
    ss_sum_x_col = packed.ss_sum_x[view_idx, :, local_idx]
    ss_sum_x_sq_col = packed.ss_sum_x_sq[view_idx, :, local_idx]
    ss_cat_counts_col = packed.ss_cat_counts[view_idx, :, local_idx]
    ss_sum_sin_col = packed.ss_sum_sin[view_idx, :, local_idx]
    ss_sum_cos_col = packed.ss_sum_cos[view_idx, :, local_idx]

    type_id = packed.col_type_ids[col_idx]

    # vmap unified_posterior_predictive_logp over the cluster axis
    def _score_cluster(c_idx):
        return unified_posterior_predictive_logp(
            x,
            type_id,
            ss_counts_col[c_idx],
            ss_sum_x_col[c_idx],
            ss_sum_x_sq_col[c_idx],
            ss_cat_counts_col[c_idx],
            ss_sum_sin_col[c_idx],
            ss_sum_cos_col[c_idx],
            packed.hyper_mu[col_idx],
            packed.hyper_r[col_idx],
            packed.hyper_s[col_idx],
            packed.hyper_nu[col_idx],
            packed.hyper_dirichlet_alpha[col_idx],
            packed.hyper_alpha[col_idx],
            packed.hyper_beta[col_idx],
            packed.hyper_kappa[col_idx],
            packed.hyper_vm_mu[col_idx],
        )

    cluster_indices = jnp.arange(max_k)
    log_liks = jax.vmap(_score_cluster)(cluster_indices)  # (max_k,)

    # Mask inactive clusters (weight == 0 -> log_weight = -inf)
    log_weights = jnp.log(jnp.maximum(weights, 1e-30))
    # Where weights are exactly 0, force -inf
    log_weights = jnp.where(weights > 0, log_weights, -jnp.inf)
    log_terms = log_weights + log_liks

    return jax.scipy.special.logsumexp(log_terms)


def _sample_one_column(
    rng_key: Array,
    packed: PackedCrossCatState,
    view_idx: int,
    col_idx: int,
    weights: Array,
) -> Array:
    """Sample a single value from the cluster mixture for col_idx in view_idx."""
    local_idx = _find_local_col_index(packed, view_idx, col_idx)
    k1, k2 = jax.random.split(rng_key)

    # Sample cluster from weights
    log_w = jnp.log(jnp.maximum(weights, 1e-30))
    cluster = jax.random.categorical(k1, log_w)

    # Gather suffstats for the sampled cluster
    return unified_sample_posterior_predictive(
        k2,
        packed.col_type_ids[col_idx],
        packed.ss_counts[view_idx, cluster, local_idx].astype(jnp.float32),
        packed.ss_sum_x[view_idx, cluster, local_idx],
        packed.ss_sum_x_sq[view_idx, cluster, local_idx],
        packed.ss_cat_counts[view_idx, cluster, local_idx],
        packed.ss_sum_sin[view_idx, cluster, local_idx],
        packed.ss_sum_cos[view_idx, cluster, local_idx],
        packed.hyper_mu[col_idx],
        packed.hyper_r[col_idx],
        packed.hyper_s[col_idx],
        packed.hyper_nu[col_idx],
        packed.hyper_dirichlet_alpha[col_idx],
        packed.hyper_alpha[col_idx],
        packed.hyper_beta[col_idx],
        packed.hyper_kappa[col_idx],
        packed.hyper_vm_mu[col_idx],
    )


# ---------------------------------------------------------------------------
# Public inference functions
# ---------------------------------------------------------------------------


def packed_predictive_probability(
    packed: PackedCrossCatState,
    data: Array,
    query_cols: list[int],
    query_vals: Array,
    *,
    row_id: int | None = None,
) -> Array:
    """Compute predictive log probability on PackedCrossCatState.

    For each query column: find view, get cluster weights, vmap over clusters
    for posterior predictive logp, logsumexp for mixture, sum across columns.

    Args:
        packed: Packed CrossCat state.
        data: Observation matrix (n_rows, n_cols).
        query_cols: Column indices to query.
        query_vals: Values to evaluate probability at.
        row_id: If provided, use observed row's cluster assignment (one-hot weights).

    Returns:
        Scalar log probability.
    """
    log_p_total = jnp.array(0.0)

    for q_idx, col in enumerate(query_cols):
        view_idx = int(packed.column_assignments[col])

        if row_id is not None:
            weights = _cluster_weights_for_row(packed, view_idx, row_id)
        else:
            weights = _cluster_weights_packed(packed, view_idx)

        log_mixture = _logp_one_column_mixture(packed, view_idx, col, query_vals[q_idx], weights)
        log_p_total = log_p_total + log_mixture

    return log_p_total


def packed_predictive_sample(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    query_cols: list[int],
    *,
    n_samples: int = 1000,
    row_id: int | None = None,
) -> Array:
    """Draw samples from the posterior predictive on PackedCrossCatState.

    Uses vmap over n_samples. For each sample, each query column: sample cluster
    from weights, then sample value from that cluster.

    Args:
        rng_key: JAX PRNG key.
        packed: Packed CrossCat state.
        data: Observation matrix.
        query_cols: Column indices to sample.
        n_samples: Number of posterior predictive samples.
        row_id: If provided, use observed row's cluster assignment.

    Returns:
        Array of shape (n_samples, len(query_cols)).
    """
    n_q = len(query_cols)

    # Pre-compute view indices and weights per query column
    view_indices = [int(packed.column_assignments[col]) for col in query_cols]
    weights_list = []
    for q_idx, _col in enumerate(query_cols):
        v = view_indices[q_idx]
        if row_id is not None:
            w = _cluster_weights_for_row(packed, v, row_id)
        else:
            w = _cluster_weights_packed(packed, v)
        weights_list.append(w)
    weights_arr = jnp.stack(weights_list)  # (n_q, max_clusters)

    # Pre-compute local indices for each query column
    local_indices = jnp.array(
        [int(_find_local_col_index(packed, view_indices[q], query_cols[q])) for q in range(n_q)],
        dtype=jnp.int32,
    )
    view_idx_arr = jnp.array(view_indices, dtype=jnp.int32)
    col_arr = jnp.array(query_cols, dtype=jnp.int32)

    def _sample_one(rng_key_s):
        """Sample one row of (n_q,) values."""
        col_keys = jax.random.split(rng_key_s, n_q)

        def _sample_q(q_idx):
            """Sample one query column."""
            k = col_keys[q_idx]
            k1, k2 = jax.random.split(k)
            v = view_idx_arr[q_idx]
            c = col_arr[q_idx]
            li = local_indices[q_idx]
            w = weights_arr[q_idx]

            # Sample cluster
            log_w = jnp.log(jnp.maximum(w, 1e-30))
            cluster = jax.random.categorical(k1, log_w)

            return unified_sample_posterior_predictive(
                k2,
                packed.col_type_ids[c],
                packed.ss_counts[v, cluster, li].astype(jnp.float32),
                packed.ss_sum_x[v, cluster, li],
                packed.ss_sum_x_sq[v, cluster, li],
                packed.ss_cat_counts[v, cluster, li],
                packed.ss_sum_sin[v, cluster, li],
                packed.ss_sum_cos[v, cluster, li],
                packed.hyper_mu[c],
                packed.hyper_r[c],
                packed.hyper_s[c],
                packed.hyper_nu[c],
                packed.hyper_dirichlet_alpha[c],
                packed.hyper_alpha[c],
                packed.hyper_beta[c],
                packed.hyper_kappa[c],
                packed.hyper_vm_mu[c],
            )

        q_indices = jnp.arange(n_q)
        return jax.vmap(_sample_q)(q_indices)  # (n_q,)

    sample_keys = jax.random.split(rng_key, n_samples)
    return jax.vmap(_sample_one)(sample_keys)  # (n_samples, n_q)


def packed_mutual_information(
    packed_states: list[PackedCrossCatState],
    column_types: list,
    col_i: int,
    col_j: int,
) -> tuple[Array, Array]:
    """Estimate mutual information between two columns from packed states.

    Uses CrossCat structure: columns in the same view share clustering and
    have nonzero MI; columns in different views are independent (MI=0).

    Averaged over multiple packed states (ensemble inference).
    Python loop over the (small) list of states is acceptable.

    Args:
        packed_states: List of PackedCrossCatState (4-8 chains typically).
        column_types: Column type list.
        col_i: First column index.
        col_j: Second column index.

    Returns:
        Tuple of (mutual_information, linfoot_correlation).
    """
    mi_estimates = []

    for packed in packed_states:
        view_i = packed.column_assignments[col_i]
        view_j = packed.column_assignments[col_j]

        same_view = view_i == view_j

        # If different views, MI = 0
        if not same_view:
            mi_estimates.append(0.0)
            continue

        # Same view: estimate MI from clustering entropy
        v = int(view_i)
        assigns = packed.view_row_assignments[v]
        max_k = packed.max_clusters
        one_hot = jax.nn.one_hot(assigns, max_k)
        counts = one_hot.sum(axis=0).astype(jnp.float32)
        total = counts.sum()
        probs = counts / jnp.maximum(total, 1e-30)

        entropy_clustering = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs + 1e-30), 0.0))

        n_clusters = int(packed.view_n_clusters[v])
        mi_est = entropy_clustering * (1.0 - 1.0 / jnp.maximum(float(n_clusters), 1.0))
        mi_estimates.append(float(mi_est))

    mi = jnp.array(mi_estimates).mean()
    linfoot = jnp.sqrt(1.0 - jnp.exp(-2.0 * mi))
    return mi, linfoot


def packed_row_similarity(
    packed_states: list[PackedCrossCatState],
    column_types: list,
    row_a: int,
    row_b: int,
    *,
    target_columns: list[int] | None = None,
) -> Array:
    """Compute similarity between two rows from packed states.

    Similarity is the probability that two rows are in the same cluster,
    averaged over views and posterior states.

    Python loop over the (small) list of states is acceptable.

    Args:
        packed_states: List of PackedCrossCatState.
        column_types: Column type list.
        row_a: First row index.
        row_b: Second row index.
        target_columns: Restrict to views containing these columns (optional).

    Returns:
        Similarity score in [0, 1].
    """
    sim_scores = []

    for packed in packed_states:
        n_views = int(packed.n_views)
        view_scores = []

        for v in range(n_views):
            # If target_columns specified, check if this view contains any
            if target_columns is not None:
                n_cols_v = int(packed.view_n_columns[v])
                view_col_set = set(
                    int(packed.view_column_indices[v, li]) for li in range(n_cols_v)
                )
                if not any(c in view_col_set for c in target_columns):
                    continue

            same_cluster = float(
                packed.view_row_assignments[v, row_a] == packed.view_row_assignments[v, row_b]
            )
            view_scores.append(same_cluster)

        if view_scores:
            sim_scores.append(sum(view_scores) / len(view_scores))

    if not sim_scores:
        return jnp.array(0.0)
    return jnp.array(sum(sim_scores) / len(sim_scores))


def packed_impute_and_confidence(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    query_col: int,
    *,
    n_samples: int = 1000,
) -> tuple[Array, Array]:
    """Impute a value with confidence score using packed state.

    For continuous columns: returns median, confidence from exp(-IQR/std).
    For discrete columns: returns mode, confidence = mode frequency.

    Args:
        rng_key: JAX PRNG key.
        packed: Packed CrossCat state.
        data: Observation matrix.
        query_col: Column to impute.
        n_samples: Number of samples for estimation.

    Returns:
        Tuple of (point_estimate, confidence_score).
    """
    from crosscat.packed import CONTINUOUS_ID

    samples = packed_predictive_sample(rng_key, packed, data, [query_col], n_samples=n_samples)
    s = samples[:, 0]

    type_id = int(packed.col_type_ids[query_col])

    if type_id == CONTINUOUS_ID:
        point_est = jnp.median(s)
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


def packed_anomaly_score(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    query_row: int,
) -> Array:
    """Compute anomaly score for a row using packed state.

    Evaluates predictive probability for each non-NaN column of the query row,
    averages log probs, and applies sigmoid transform to [0, 1].

    Args:
        rng_key: JAX PRNG key.
        packed: Packed CrossCat state.
        data: Observation matrix.
        query_row: Row index to score.

    Returns:
        Anomaly score in [0, 1] -- higher = more anomalous.
    """
    n_cols = packed.n_cols
    row_data = data[query_row]

    # Score all columns at once: build list of non-NaN columns
    valid_cols = []
    valid_vals = []
    for col in range(n_cols):
        x = row_data[col]
        if not jnp.isnan(x):
            valid_cols.append(col)
            valid_vals.append(float(x))

    if not valid_cols:
        return jnp.array(0.5)

    # Compute log probability using row_id (observed row's cluster assignment)
    log_p = packed_predictive_probability(
        packed, data, valid_cols, jnp.array(valid_vals), row_id=query_row
    )

    n_scored = len(valid_cols)
    avg_log_p = log_p / n_scored

    # Sigmoid transform: more negative log_p = more anomalous
    anomaly = 1.0 / (1.0 + jnp.exp(avg_log_p + 2.0))
    return jnp.clip(anomaly, 0.0, 1.0)


def packed_predictive_cdf(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    query_col: int,
    query_val: Array,
    *,
    n_samples: int = 10000,
) -> Array:
    """Compute posterior predictive CDF: P(X <= query_val) via MC sampling.

    Args:
        rng_key: JAX PRNG key.
        packed: Packed CrossCat state.
        data: Observation matrix.
        query_col: Column index.
        query_val: Value at which to evaluate CDF.
        n_samples: Number of MC samples.

    Returns:
        CDF value P(X <= query_val) in [0, 1].
    """
    samples = packed_predictive_sample(rng_key, packed, data, [query_col], n_samples=n_samples)
    return jnp.mean(samples[:, 0] <= query_val)
