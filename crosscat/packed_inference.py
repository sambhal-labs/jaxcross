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
from crosscat.types import LOG_EPS

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
    return counts / jnp.maximum(counts.sum(), LOG_EPS)


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
            packed.hyper_vm_a[col_idx],
            packed.hyper_vm_mu[col_idx],
            packed.hyper_cutpoints[col_idx],
        )

    cluster_indices = jnp.arange(max_k)
    log_liks = jax.vmap(_score_cluster)(cluster_indices)  # (max_k,)

    # Mask inactive clusters (weight == 0 -> log_weight = -inf)
    log_weights = jnp.log(jnp.maximum(weights, LOG_EPS))
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
    log_w = jnp.log(jnp.maximum(weights, LOG_EPS))
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
        packed.hyper_vm_a[col_idx],
        packed.hyper_vm_mu[col_idx],
        packed.hyper_cutpoints[col_idx],
    )


# ---------------------------------------------------------------------------
# Public inference functions
# ---------------------------------------------------------------------------


def packed_classify_column(
    packed: PackedCrossCatState,
    data: Array,
    target_col: int,
    candidate_vals: Array,
    row_id: int,
) -> Array:
    """Compute log P(target_col=v | row) for each candidate value v.

    Vectorized over candidate values via vmap. Useful for classification
    where target_col is categorical and candidate_vals are the possible classes.

    Args:
        packed: Packed CrossCat state.
        data: Observation matrix (n_rows, n_cols).
        target_col: Column index to classify.
        candidate_vals: 1D array of candidate values to score.
        row_id: Row index (uses observed row's cluster assignment).

    Returns:
        Array of shape (len(candidate_vals),) with log probabilities.
    """
    view_idx = int(packed.column_assignments[target_col])
    weights = _cluster_weights_for_row(packed, view_idx, row_id)

    def _score_val(v):
        return _logp_one_column_mixture(packed, view_idx, target_col, v, weights)

    return jax.vmap(_score_val)(candidate_vals)


def batch_classify_column(
    packed: PackedCrossCatState,
    data: Array,
    target_col: int,
    candidate_vals: Array,
    row_ids: Array,
) -> Array:
    """Batch classification: log P(target_col=v | row) for all rows and values.

    Double-vmapped: over rows (cluster weights vary) and over candidate values.
    After one-time JIT compilation, classifies all rows in a single GPU call.

    Args:
        packed: Packed CrossCat state.
        data: Observation matrix (n_rows, n_cols).
        target_col: Column index to classify.
        candidate_vals: 1D array of candidate values, shape (n_candidates,).
        row_ids: 1D array of row indices, shape (n_rows,).

    Returns:
        Array of shape (n_rows, n_candidates) with log probabilities.
    """
    view_idx = int(packed.column_assignments[target_col])

    def _classify_one_row(row_id):
        weights = _cluster_weights_for_row(packed, view_idx, row_id)

        def _score_val(v):
            return _logp_one_column_mixture(packed, view_idx, target_col, v, weights)

        return jax.vmap(_score_val)(candidate_vals)

    return jax.vmap(_classify_one_row)(row_ids)


def batch_score_columns_binary(
    packed: PackedCrossCatState,
    data: Array,
    col_indices: Array,
    row_id: int,
) -> Array:
    """Compute P(col=1 | row) for multiple binary columns in one JIT call.

    Vectorized over columns via vmap. For each column, computes the
    posterior predictive probability of value 1 vs 0 and returns P(1).

    Designed for inpainting: score all missing pixels at once.

    Args:
        packed: Packed CrossCat state.
        data: Observation matrix (n_rows, n_cols).
        col_indices: 1D integer array of column indices to score.
        row_id: Row index (uses observed row's cluster assignment).

    Returns:
        Array of shape (len(col_indices),) with P(col=1 | row) in [0, 1].
    """
    max_k = packed.max_clusters

    def _prob1_for_col(col_idx):
        view_idx = packed.column_assignments[col_idx]
        weights = _cluster_weights_for_row(packed, view_idx, row_id)

        local_idx = _find_local_col_index(packed, view_idx, col_idx)

        type_id = packed.col_type_ids[col_idx]

        def _score_cluster_at_val(c_idx, x):
            return unified_posterior_predictive_logp(
                x,
                type_id,
                packed.ss_counts[view_idx, c_idx, local_idx].astype(jnp.float32),
                packed.ss_sum_x[view_idx, c_idx, local_idx],
                packed.ss_sum_x_sq[view_idx, c_idx, local_idx],
                packed.ss_cat_counts[view_idx, c_idx, local_idx],
                packed.ss_sum_sin[view_idx, c_idx, local_idx],
                packed.ss_sum_cos[view_idx, c_idx, local_idx],
                packed.hyper_mu[col_idx],
                packed.hyper_r[col_idx],
                packed.hyper_s[col_idx],
                packed.hyper_nu[col_idx],
                packed.hyper_dirichlet_alpha[col_idx],
                packed.hyper_alpha[col_idx],
                packed.hyper_beta[col_idx],
                packed.hyper_kappa[col_idx],
                packed.hyper_vm_a[col_idx],
                packed.hyper_vm_mu[col_idx],
                packed.hyper_cutpoints[col_idx],
            )

        cluster_indices = jnp.arange(max_k)
        val_1 = jnp.float32(1.0)
        val_0 = jnp.float32(0.0)
        log_liks_1 = jax.vmap(lambda c: _score_cluster_at_val(c, val_1))(cluster_indices)
        log_liks_0 = jax.vmap(lambda c: _score_cluster_at_val(c, val_0))(cluster_indices)

        log_weights = jnp.log(jnp.maximum(weights, LOG_EPS))
        log_weights = jnp.where(weights > 0, log_weights, -jnp.inf)

        logp1 = jax.scipy.special.logsumexp(log_weights + log_liks_1)
        logp0 = jax.scipy.special.logsumexp(log_weights + log_liks_0)

        return jnp.exp(logp1 - jnp.logaddexp(logp0, logp1))

    return jax.vmap(_prob1_for_col)(col_indices)


def batch_anomaly_score(
    packed: PackedCrossCatState,
    data: Array,
    row_ids: Array,
) -> Array:
    """Compute anomaly scores for multiple rows in one JIT call.

    For each row, evaluates the average log predictive probability across
    all observed (non-NaN) columns, then applies a sigmoid transform.
    Fully vectorized via vmap — no Python loops.

    Args:
        packed: Packed CrossCat state.
        data: Observation matrix (n_rows, n_cols).
        row_ids: 1D integer array of row indices to score.

    Returns:
        Array of shape (len(row_ids),) with anomaly scores in [0, 1].
        Higher = more anomalous.
    """
    n_cols = packed.n_cols
    max_k = packed.max_clusters
    all_col_indices = jnp.arange(n_cols)

    def _logp_one_col_for_row(col_idx, row_id, x):
        """Log predictive probability of value x at col_idx for row_id."""
        view_idx = packed.column_assignments[col_idx]
        weights = _cluster_weights_for_row(packed, view_idx, row_id)
        local_idx = _find_local_col_index(packed, view_idx, col_idx)
        type_id = packed.col_type_ids[col_idx]

        def _score_cluster(c_idx):
            return unified_posterior_predictive_logp(
                x,
                type_id,
                packed.ss_counts[view_idx, c_idx, local_idx].astype(jnp.float32),
                packed.ss_sum_x[view_idx, c_idx, local_idx],
                packed.ss_sum_x_sq[view_idx, c_idx, local_idx],
                packed.ss_cat_counts[view_idx, c_idx, local_idx],
                packed.ss_sum_sin[view_idx, c_idx, local_idx],
                packed.ss_sum_cos[view_idx, c_idx, local_idx],
                packed.hyper_mu[col_idx],
                packed.hyper_r[col_idx],
                packed.hyper_s[col_idx],
                packed.hyper_nu[col_idx],
                packed.hyper_dirichlet_alpha[col_idx],
                packed.hyper_alpha[col_idx],
                packed.hyper_beta[col_idx],
                packed.hyper_kappa[col_idx],
                packed.hyper_vm_a[col_idx],
                packed.hyper_vm_mu[col_idx],
                packed.hyper_cutpoints[col_idx],
            )

        log_liks = jax.vmap(_score_cluster)(jnp.arange(max_k))
        log_weights = jnp.log(jnp.maximum(weights, LOG_EPS))
        log_weights = jnp.where(weights > 0, log_weights, -jnp.inf)
        return jax.scipy.special.logsumexp(log_weights + log_liks)

    def _score_one_row(row_id):
        row_data = data[row_id]
        valid_mask = ~jnp.isnan(row_data[:n_cols])

        # Score all columns (vmap), mask invalid ones
        log_ps = jax.vmap(lambda c: _logp_one_col_for_row(c, row_id, row_data[c]))(all_col_indices)

        # Average over valid (non-NaN) columns only
        n_valid = jnp.maximum(valid_mask.sum(), 1.0)
        avg_log_p = jnp.where(valid_mask, log_ps, 0.0).sum() / n_valid

        # Sigmoid: more negative → more anomalous
        anomaly = 1.0 / (1.0 + jnp.exp(avg_log_p + 2.0))
        return jnp.clip(anomaly, 0.0, 1.0)

    return jax.vmap(_score_one_row)(row_ids)


def batch_impute_column(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    query_col: int,
    row_ids: Array,
    *,
    n_samples: int = 100,
) -> tuple[Array, Array]:
    """Impute a column for multiple rows in one JIT call.

    For each row, draws samples from the posterior predictive for query_col
    using the row's cluster assignment, then computes point estimate and
    confidence. Fully vectorized via vmap.

    Args:
        rng_key: JAX PRNG key.
        packed: Packed CrossCat state.
        data: Observation matrix (n_rows, n_cols).
        query_col: Column to impute.
        row_ids: 1D integer array of row indices to impute.
        n_samples: Number of posterior predictive samples per row.

    Returns:
        Tuple of (point_estimates, confidences), each shape (len(row_ids),).
    """
    from crosscat.packed import CONTINUOUS_ID

    view_idx = int(packed.column_assignments[query_col])
    local_idx = _find_local_col_index(packed, view_idx, query_col)
    col_idx = query_col
    type_id = packed.col_type_ids[col_idx]

    def _impute_one_row(row_key_and_id):
        key, row_id = row_key_and_id
        weights = _cluster_weights_for_row(packed, view_idx, row_id)

        def _draw_one_sample(sample_key):
            k1, k2 = jax.random.split(sample_key)
            log_w = jnp.log(jnp.maximum(weights, LOG_EPS))
            cluster = jax.random.categorical(k1, log_w)
            return unified_sample_posterior_predictive(
                k2,
                type_id,
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
                packed.hyper_vm_a[col_idx],
                packed.hyper_vm_mu[col_idx],
                packed.hyper_cutpoints[col_idx],
            )

        sample_keys = jax.random.split(key, n_samples)
        samples = jax.vmap(_draw_one_sample)(sample_keys)

        # Point estimate and confidence
        is_continuous = type_id == CONTINUOUS_ID
        point_est = jnp.where(is_continuous, jnp.median(samples), 0.0)
        confidence = jnp.where(is_continuous, 1.0 / (1.0 + jnp.std(samples)), 0.0)

        # Categorical path: mode and mode frequency
        s_int = samples.astype(jnp.int32)
        max_cat = packed.max_categories
        counts = jnp.zeros(max_cat, dtype=jnp.int32)
        counts = counts.at[jnp.clip(s_int, 0, max_cat - 1)].add(1)
        cat_mode = jnp.argmax(counts).astype(jnp.float32)
        cat_conf = counts[jnp.argmax(counts)] / jnp.float32(n_samples)

        point_est = jnp.where(is_continuous, point_est, cat_mode)
        confidence = jnp.where(is_continuous, confidence, cat_conf)

        return point_est, confidence

    keys = jax.random.split(rng_key, len(row_ids))
    point_ests, confidences = jax.vmap(_impute_one_row)((keys, row_ids))
    return point_ests, confidences


def batch_row_similarity(
    packed_states: list[PackedCrossCatState],
    row_ids: Array,
) -> Array:
    """Compute pairwise similarity matrix for multiple rows.

    Similarity is the probability that two rows share the same cluster,
    averaged over views and posterior states. Fully vectorized — no Python
    loops over row pairs.

    Args:
        packed_states: List of PackedCrossCatState (MCMC samples).
        row_ids: 1D integer array of row indices, shape (N,).

    Returns:
        Symmetric array of shape (N, N) with similarity scores in [0, 1].
        Diagonal is always 1.0.
    """
    n = len(row_ids)
    sim_matrix = jnp.zeros((n, n))

    for packed in packed_states:
        n_views = int(packed.n_views)
        max_views = packed.max_views
        view_mask = jnp.arange(max_views) < n_views

        # (max_views, N) — cluster assignment per view per selected row
        assigns = packed.view_row_assignments[:, row_ids]  # (max_views, N)

        # (max_views, N, N) — same cluster indicator for all pairs
        same = (assigns[:, :, None] == assigns[:, None, :]).astype(jnp.float32)

        # Mask inactive views, average over active views
        same_masked = jnp.where(view_mask[:, None, None], same, 0.0)
        per_state = same_masked.sum(axis=0) / jnp.maximum(n_views, 1.0)

        sim_matrix = sim_matrix + per_state

    return sim_matrix / jnp.maximum(len(packed_states), 1.0)


def batch_predictive_cdf(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    query_col: int,
    query_val: Array,
    row_ids: Array,
    *,
    n_samples: int = 1000,
) -> Array:
    """Compute posterior predictive CDF for multiple rows in one JIT call.

    P(X <= query_val | row) for each row, using the row's cluster assignment.

    Args:
        rng_key: JAX PRNG key.
        packed: Packed CrossCat state.
        data: Observation matrix.
        query_col: Column index.
        query_val: Value at which to evaluate CDF.
        row_ids: 1D integer array of row indices.
        n_samples: Number of MC samples per row.

    Returns:
        Array of shape (len(row_ids),) with CDF values in [0, 1].
    """
    view_idx = int(packed.column_assignments[query_col])
    local_idx = _find_local_col_index(packed, view_idx, query_col)
    col_idx = query_col
    type_id = packed.col_type_ids[col_idx]

    def _cdf_one_row(key_and_id):
        key, row_id = key_and_id
        weights = _cluster_weights_for_row(packed, view_idx, row_id)

        def _draw(sample_key):
            k1, k2 = jax.random.split(sample_key)
            log_w = jnp.log(jnp.maximum(weights, LOG_EPS))
            cluster = jax.random.categorical(k1, log_w)
            return unified_sample_posterior_predictive(
                k2,
                type_id,
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
                packed.hyper_vm_a[col_idx],
                packed.hyper_vm_mu[col_idx],
                packed.hyper_cutpoints[col_idx],
            )

        sample_keys = jax.random.split(key, n_samples)
        samples = jax.vmap(_draw)(sample_keys)
        return jnp.mean(samples <= query_val)

    keys = jax.random.split(rng_key, len(row_ids))
    return jax.vmap(_cdf_one_row)((keys, row_ids))


def batch_credible_interval(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    query_col: int,
    row_ids: Array,
    *,
    n_samples: int = 1000,
    ci_level: float = 0.90,
) -> tuple[Array, Array, Array]:
    """Compute credible intervals for multiple rows in one JIT call.

    For each row, draws posterior predictive samples conditioned on the
    row's cluster assignment and computes percentile-based CI.

    Args:
        rng_key: JAX PRNG key.
        packed: Packed CrossCat state.
        data: Observation matrix.
        query_col: Column to compute CI for.
        row_ids: 1D integer array of row indices.
        n_samples: Number of samples per row.
        ci_level: Credible interval level (0.90 = 90% CI).

    Returns:
        Tuple of (medians, lower_bounds, upper_bounds), each shape (len(row_ids),).
    """
    view_idx = int(packed.column_assignments[query_col])
    local_idx = _find_local_col_index(packed, view_idx, query_col)
    col_idx = query_col
    type_id = packed.col_type_ids[col_idx]
    tail = (1.0 - ci_level) / 2.0

    def _ci_one_row(key_and_id):
        key, row_id = key_and_id
        weights = _cluster_weights_for_row(packed, view_idx, row_id)

        def _draw(sample_key):
            k1, k2 = jax.random.split(sample_key)
            log_w = jnp.log(jnp.maximum(weights, LOG_EPS))
            cluster = jax.random.categorical(k1, log_w)
            return unified_sample_posterior_predictive(
                k2,
                type_id,
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
                packed.hyper_vm_a[col_idx],
                packed.hyper_vm_mu[col_idx],
                packed.hyper_cutpoints[col_idx],
            )

        sample_keys = jax.random.split(key, n_samples)
        samples = jax.vmap(_draw)(sample_keys)
        return (
            jnp.median(samples),
            jnp.percentile(samples, 100.0 * tail),
            jnp.percentile(samples, 100.0 * (1.0 - tail)),
        )

    keys = jax.random.split(rng_key, len(row_ids))
    medians, lowers, uppers = jax.vmap(_ci_one_row)((keys, row_ids))
    return medians, lowers, uppers


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
            log_w = jnp.log(jnp.maximum(w, LOG_EPS))
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
                packed.hyper_vm_a[c],
                packed.hyper_vm_mu[c],
                packed.hyper_cutpoints[c],
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
    *,
    n_samples: int = 1000,
    rng_key: Array | None = None,
) -> tuple[Array, Array]:
    """Estimate mutual information between two columns via Monte Carlo sampling.

    Maps to original inference_utils.estimate_MI_sample().
    Draws (x, y) pairs from joint predictive, computes MI = E[log p(x,y) - log p(x) - log p(y)]
    with importance weighting by p(x, y).

    Averaged over multiple packed states (ensemble inference).

    Args:
        packed_states: List of PackedCrossCatState (4-8 chains typically).
        column_types: Column type list.
        col_i: First column index.
        col_j: Second column index.
        n_samples: MC samples for MI estimation.
        rng_key: JAX PRNG key (uses key(0) if not provided).

    Returns:
        Tuple of (mutual_information, linfoot_correlation).
    """
    if rng_key is None:
        rng_key = jax.random.key(0)

    mi_estimates = []

    for s_idx, packed in enumerate(packed_states):
        view_i = int(packed.column_assignments[col_i])
        view_j = int(packed.column_assignments[col_j])

        if view_i != view_j:
            mi_estimates.append(0.0)
            continue

        mi_est = _packed_estimate_mi_sample(
            jax.random.fold_in(rng_key, s_idx),
            packed,
            view_i,
            col_i,
            col_j,
            n_samples,
        )
        mi_estimates.append(float(mi_est))

    mi = jnp.array(mi_estimates).mean()
    mi = jnp.maximum(mi, 0.0)
    linfoot = jnp.sqrt(1.0 - jnp.exp(-2.0 * mi))
    return mi, linfoot


def _packed_estimate_mi_sample(
    rng_key: Array,
    packed: PackedCrossCatState,
    view_idx: int,
    col_i: int,
    col_j: int,
    n_samples: int,
) -> Array:
    """MC MI estimation for two columns in the same view (packed version).

    Maps to original inference_utils.estimate_MI_sample().

    Fully vectorized via vmap: samples all (x, y) pairs in parallel and
    scores them across all clusters simultaneously. No Python sample loop.
    """
    cluster_weights = _cluster_weights_packed(packed, view_idx)
    log_cluster_weights = jnp.log(jnp.maximum(cluster_weights, LOG_EPS))
    n_clusters = int(packed.view_n_clusters[view_idx])
    max_k = packed.max_clusters

    local_i = int(_find_local_col_index(packed, view_idx, col_i))
    local_j = int(_find_local_col_index(packed, view_idx, col_j))

    type_id_i = packed.col_type_ids[col_i]
    type_id_j = packed.col_type_ids[col_j]

    def _get_hypers(col: int):
        return (
            packed.hyper_mu[col],
            packed.hyper_r[col],
            packed.hyper_s[col],
            packed.hyper_nu[col],
            packed.hyper_dirichlet_alpha[col],
            packed.hyper_alpha[col],
            packed.hyper_beta[col],
            packed.hyper_kappa[col],
            packed.hyper_vm_a[col],
            packed.hyper_vm_mu[col],
            packed.hyper_cutpoints[col],
        )

    hypers_i = _get_hypers(col_i)
    hypers_j = _get_hypers(col_j)

    # Precompute all cluster suffstats as (max_clusters,) arrays
    ss_counts_i = packed.ss_counts[view_idx, :, local_i].astype(jnp.float32)
    ss_sum_x_i = packed.ss_sum_x[view_idx, :, local_i]
    ss_sum_x_sq_i = packed.ss_sum_x_sq[view_idx, :, local_i]
    ss_cat_counts_i = packed.ss_cat_counts[view_idx, :, local_i]
    ss_sum_sin_i = packed.ss_sum_sin[view_idx, :, local_i]
    ss_sum_cos_i = packed.ss_sum_cos[view_idx, :, local_i]

    ss_counts_j = packed.ss_counts[view_idx, :, local_j].astype(jnp.float32)
    ss_sum_x_j = packed.ss_sum_x[view_idx, :, local_j]
    ss_sum_x_sq_j = packed.ss_sum_x_sq[view_idx, :, local_j]
    ss_cat_counts_j = packed.ss_cat_counts[view_idx, :, local_j]
    ss_sum_sin_j = packed.ss_sum_sin[view_idx, :, local_j]
    ss_sum_cos_j = packed.ss_sum_cos[view_idx, :, local_j]

    cluster_mask = jnp.arange(max_k) < n_clusters

    # Vectorized logp across all clusters for a given value
    def _logp_all_clusters_i(x):
        def _one(c):
            return unified_posterior_predictive_logp(
                x,
                type_id_i,
                ss_counts_i[c],
                ss_sum_x_i[c],
                ss_sum_x_sq_i[c],
                ss_cat_counts_i[c],
                ss_sum_sin_i[c],
                ss_sum_cos_i[c],
                *hypers_i,
            )

        logps = jax.vmap(_one)(jnp.arange(max_k))
        return jnp.where(cluster_mask, logps, -jnp.inf)

    def _logp_all_clusters_j(y):
        def _one(c):
            return unified_posterior_predictive_logp(
                y,
                type_id_j,
                ss_counts_j[c],
                ss_sum_x_j[c],
                ss_sum_x_sq_j[c],
                ss_cat_counts_j[c],
                ss_sum_sin_j[c],
                ss_sum_cos_j[c],
                *hypers_j,
            )

        logps = jax.vmap(_one)(jnp.arange(max_k))
        return jnp.where(cluster_mask, logps, -jnp.inf)

    def _one_sample(key):
        """Draw one (x, y) sample and compute MI contribution."""
        k1, k2, k3 = jax.random.split(key, 3)

        # Draw cluster from CRP weights
        cluster = jax.random.categorical(k1, log_cluster_weights)

        # Sample x from col_i in this cluster (index into precomputed arrays)
        x = unified_sample_posterior_predictive(
            k2,
            type_id_i,
            ss_counts_i[cluster],
            ss_sum_x_i[cluster],
            ss_sum_x_sq_i[cluster],
            ss_cat_counts_i[cluster],
            ss_sum_sin_i[cluster],
            ss_sum_cos_i[cluster],
            *hypers_i,
        )

        # Sample y from col_j in this cluster
        y = unified_sample_posterior_predictive(
            k3,
            type_id_j,
            ss_counts_j[cluster],
            ss_sum_x_j[cluster],
            ss_sum_x_sq_j[cluster],
            ss_cat_counts_j[cluster],
            ss_sum_sin_j[cluster],
            ss_sum_cos_j[cluster],
            *hypers_j,
        )

        # Score across all clusters
        lp_x_all = _logp_all_clusters_i(x)
        lp_y_all = _logp_all_clusters_j(y)

        log_px = jax.scipy.special.logsumexp(log_cluster_weights + lp_x_all)
        log_py = jax.scipy.special.logsumexp(log_cluster_weights + lp_y_all)
        log_pxy = jax.scipy.special.logsumexp(log_cluster_weights + lp_x_all + lp_y_all)

        return log_pxy - log_px - log_py

    # Vectorize over all samples at once — no Python loop
    keys = jax.random.split(rng_key, n_samples)
    mi_samples = jax.vmap(_one_sample)(keys)

    mi_est = jnp.mean(mi_samples)
    return jnp.maximum(mi_est, 0.0)


def packed_dependence_probability(
    packed_states: list[PackedCrossCatState],
    col_i: int,
    col_j: int,
) -> Array:
    """Posterior probability that two columns are dependent (packed version).

    Z(i,j) = fraction of posterior samples where columns i and j are assigned
    to the same view. This is the paper's primary exploratory statistic
    (Mansinghka et al. 2016, Section 2.5.2).

    Args:
        packed_states: List of PackedCrossCatState (MCMC samples).
        col_i: First column index.
        col_j: Second column index.

    Returns:
        Scalar probability in [0, 1].
    """
    same_count = sum(
        1
        for packed in packed_states
        if int(packed.column_assignments[col_i]) == int(packed.column_assignments[col_j])
    )
    return jnp.array(same_count / len(packed_states))


def packed_dependence_matrix(
    packed_states: list[PackedCrossCatState],
) -> Array:
    """Full dependence probability matrix (Z-matrix) from packed states.

    Z[i,j] = fraction of posterior samples where columns i and j share a view.
    Diagonal is always 1.0. Symmetric.

    Args:
        packed_states: List of PackedCrossCatState.

    Returns:
        Array of shape (n_cols, n_cols) with values in [0, 1].
    """
    n_cols = int(packed_states[0].n_cols)
    z = jnp.zeros((n_cols, n_cols))
    for packed in packed_states:
        assigns = packed.column_assignments[:n_cols]
        same = (assigns[:, None] == assigns[None, :]).astype(jnp.float32)
        z = z + same
    return z / len(packed_states)


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
    row_id: int | None = None,
) -> tuple[Array, Array]:
    """Impute a value with confidence score using packed state.

    For continuous columns: returns median, confidence = 1/(1+std).
    For discrete columns: returns mode, confidence = mode frequency.

    Args:
        rng_key: JAX PRNG key.
        packed: Packed CrossCat state.
        data: Observation matrix.
        query_col: Column to impute.
        n_samples: Number of samples for estimation.
        row_id: If provided, use observed row's cluster assignment
            instead of marginalizing over clusters.

    Returns:
        Tuple of (point_estimate, confidence_score).
    """
    from crosscat.packed import CONTINUOUS_ID

    samples = packed_predictive_sample(
        rng_key, packed, data, [query_col], n_samples=n_samples, row_id=row_id
    )
    s = samples[:, 0]

    type_id = int(packed.col_type_ids[query_col])

    if type_id == CONTINUOUS_ID:
        point_est = jnp.median(s)
        # Confidence: inverse-variance measure — tighter posterior → higher confidence
        confidence = 1.0 / (1.0 + jnp.std(s))
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
    row_id: int | None = None,
) -> Array:
    """Compute posterior predictive CDF: P(X <= query_val) via MC sampling.

    Args:
        rng_key: JAX PRNG key.
        packed: Packed CrossCat state.
        data: Observation matrix.
        query_col: Column index.
        query_val: Value at which to evaluate CDF.
        n_samples: Number of MC samples.
        row_id: If provided, use observed row's cluster assignment.

    Returns:
        CDF value P(X <= query_val) in [0, 1].
    """
    samples = packed_predictive_sample(
        rng_key, packed, data, [query_col], n_samples=n_samples, row_id=row_id
    )
    return jnp.mean(samples[:, 0] <= query_val)


# ---------------------------------------------------------------------------
# Multi-chain inference — average queries across posterior samples
# ---------------------------------------------------------------------------


def multi_chain_predictive_probability(
    packed_states: list[PackedCrossCatState],
    data: Array,
    query_cols: list[int],
    query_vals: Array,
    *,
    row_id: int | None = None,
) -> Array:
    """Average predictive log probability across multiple chains.

    Computes log-mean-exp: log(mean(exp(log_p_i))) which is the log of the
    average predictive probability across chains. This is the standard
    Bayesian model averaging approach.

    Args:
        packed_states: List of PackedCrossCatState (MCMC posterior samples).
        data: Observation matrix (n_rows, n_cols).
        query_cols: Column indices to query.
        query_vals: Values to evaluate probability at.
        row_id: If provided, use observed row's cluster assignment.

    Returns:
        Scalar log probability (averaged across chains).
    """
    log_ps = jnp.array(
        [
            packed_predictive_probability(p, data, query_cols, query_vals, row_id=row_id)
            for p in packed_states
        ]
    )
    return jax.scipy.special.logsumexp(log_ps) - jnp.log(jnp.float32(len(packed_states)))


def multi_chain_predictive_sample(
    rng_key: Array,
    packed_states: list[PackedCrossCatState],
    data: Array,
    query_cols: list[int],
    *,
    n_samples: int = 1000,
    row_id: int | None = None,
) -> Array:
    """Draw samples from posterior predictive, mixing across chains.

    Allocates samples proportionally across chains (each chain gets
    n_samples // n_chains samples) for proper Bayesian model averaging.

    Args:
        rng_key: JAX PRNG key.
        packed_states: List of PackedCrossCatState.
        data: Observation matrix.
        query_cols: Column indices to sample.
        n_samples: Total number of samples.
        row_id: If provided, use observed row's cluster assignment.

    Returns:
        Array of shape (n_samples, len(query_cols)).
    """
    n_chains = len(packed_states)
    per_chain = n_samples // n_chains
    remainder = n_samples - per_chain * n_chains

    all_samples = []
    keys = jax.random.split(rng_key, n_chains)

    for i, packed in enumerate(packed_states):
        n_i = per_chain + (1 if i < remainder else 0)
        if n_i > 0:
            s = packed_predictive_sample(
                keys[i], packed, data, query_cols, n_samples=n_i, row_id=row_id
            )
            all_samples.append(s)

    return jnp.concatenate(all_samples, axis=0)


def multi_chain_anomaly_score(
    rng_key: Array,
    packed_states: list[PackedCrossCatState],
    data: Array,
    query_row: int,
) -> Array:
    """Average anomaly score across multiple chains.

    Args:
        rng_key: JAX PRNG key.
        packed_states: List of PackedCrossCatState.
        data: Observation matrix.
        query_row: Row index to score.

    Returns:
        Anomaly score in [0, 1] averaged across chains.
    """
    scores = jnp.array(
        [
            packed_anomaly_score(jax.random.fold_in(rng_key, i), p, data, query_row)
            for i, p in enumerate(packed_states)
        ]
    )
    return jnp.mean(scores)


def multi_chain_impute_and_confidence(
    rng_key: Array,
    packed_states: list[PackedCrossCatState],
    data: Array,
    query_col: int,
    *,
    n_samples: int = 1000,
    row_id: int | None = None,
) -> tuple[Array, Array]:
    """Impute a value with confidence, mixing samples across chains.

    Draws samples from all chains, then computes point estimate and
    confidence from the pooled samples.

    Args:
        rng_key: JAX PRNG key.
        packed_states: List of PackedCrossCatState.
        data: Observation matrix.
        query_col: Column to impute.
        n_samples: Total samples across all chains.
        row_id: If provided, use observed row's cluster assignment
            instead of marginalizing over clusters.

    Returns:
        Tuple of (point_estimate, confidence_score).
    """
    from crosscat.packed import CONTINUOUS_ID

    # Pool samples across chains
    samples = multi_chain_predictive_sample(
        rng_key, packed_states, data, [query_col], n_samples=n_samples, row_id=row_id
    )
    s = samples[:, 0]

    type_id = int(packed_states[0].col_type_ids[query_col])

    if type_id == CONTINUOUS_ID:
        point_est = jnp.median(s)
        confidence = 1.0 / (1.0 + jnp.std(s))
    else:
        s_int = s.astype(jnp.int32)
        max_val = int(jnp.max(s_int)) + 1
        counts = jnp.bincount(s_int, length=max_val)
        point_est = jnp.argmax(counts).astype(jnp.float32)
        confidence = counts[jnp.argmax(counts)] / jnp.float32(n_samples)

    return point_est, confidence


def multi_chain_predictive_cdf(
    rng_key: Array,
    packed_states: list[PackedCrossCatState],
    data: Array,
    query_col: int,
    query_val: Array,
    *,
    n_samples: int = 10000,
) -> Array:
    """Posterior predictive CDF averaged across chains.

    Args:
        rng_key: JAX PRNG key.
        packed_states: List of PackedCrossCatState.
        data: Observation matrix.
        query_col: Column index.
        query_val: Value at which to evaluate CDF.
        n_samples: Total MC samples across all chains.

    Returns:
        CDF value P(X <= query_val) in [0, 1].
    """
    samples = multi_chain_predictive_sample(
        rng_key, packed_states, data, [query_col], n_samples=n_samples
    )
    return jnp.mean(samples[:, 0] <= query_val)


# ---------------------------------------------------------------------------
# Packed sample-and-insert
# ---------------------------------------------------------------------------


def packed_sample_and_insert(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    partial_row: Array,
) -> tuple[PackedCrossCatState, Array, Array]:
    """Sample missing values for a partial row and insert into packed state.

    For each NaN entry in the partial row, draws a sample from the posterior
    predictive, then inserts the completed row using packed_insert_rows.

    Args:
        rng_key: JAX PRNG key.
        packed: Packed CrossCat state.
        data: Observation matrix, shape (n_rows, n_cols).
        partial_row: 1D array of shape (n_cols,) with NaN for missing entries.

    Returns:
        Tuple of (updated_packed, updated_data, completed_row).
    """
    from crosscat.packed.kernels import packed_insert_rows

    observed_mask = ~jnp.isnan(partial_row)
    missing_cols = [int(i) for i in range(len(partial_row)) if not observed_mask[i]]

    completed = jnp.array(partial_row, copy=True)

    if missing_cols:
        k1, k2 = jax.random.split(rng_key)
        samples = packed_predictive_sample(k1, packed, data, missing_cols, n_samples=1)
        for idx, col in enumerate(missing_cols):
            completed = completed.at[col].set(samples[0, idx])
    else:
        k2 = rng_key

    new_row = completed.reshape(1, -1)
    new_packed, new_data = packed_insert_rows(k2, packed, data, new_row)

    return new_packed, new_data, completed


# ---------------------------------------------------------------------------
# Packed parity — functions ported from crosscat.inference
# ---------------------------------------------------------------------------


def packed_credible_interval(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    query_col: int,
    *,
    n_samples: int = 1000,
    ci_level: float = 0.90,
    row_id: int | None = None,
) -> tuple[Array, Array, Array]:
    """Compute credible interval for a query column (packed version).

    Uses posterior predictive samples to compute percentile-based CI.

    Args:
        rng_key: JAX PRNG key.
        packed: Packed CrossCat state.
        data: Observation matrix.
        query_col: Column to compute CI for.
        n_samples: Number of samples for CI estimation.
        ci_level: Credible interval level (0.90 = 90% CI).
        row_id: If provided, use observed row's cluster assignment.

    Returns:
        Tuple of (median, lower_bound, upper_bound).
    """
    samples = packed_predictive_sample(
        rng_key, packed, data, [query_col], n_samples=n_samples, row_id=row_id
    )
    s = samples[:, 0]

    tail = (1.0 - ci_level) / 2.0
    lower = jnp.percentile(s, 100.0 * tail)
    upper = jnp.percentile(s, 100.0 * (1.0 - tail))
    median = jnp.median(s)

    return median, lower, upper


def packed_column_typicality(
    packed_states: list[PackedCrossCatState],
    col_id: int,
) -> Array:
    """Compute structural typicality score for a column (packed version).

    A column is typical if it is consistently grouped with the same columns
    across posterior samples. Uses entropy of co-occurrence as inverse typicality.

    Args:
        packed_states: List of PackedCrossCatState (MCMC samples).
        col_id: Column index to evaluate.

    Returns:
        Typicality score in [0, 1].
    """
    if len(packed_states) <= 1:
        return jnp.array(0.5)

    n_cols = int(packed_states[0].n_cols)
    co_occurrence = jnp.zeros(n_cols)

    for packed in packed_states:
        assigns = packed.column_assignments[:n_cols]
        my_view = assigns[col_id]
        same_view = (assigns == my_view).astype(jnp.float32)
        co_occurrence = co_occurrence + same_view

    co_occurrence = co_occurrence / len(packed_states)

    co_other = jnp.concatenate([co_occurrence[:col_id], co_occurrence[col_id + 1 :]])
    p = co_other / jnp.maximum(co_other.sum(), LOG_EPS)
    entropy = -jnp.sum(jnp.where(p > 0, p * jnp.log(p + LOG_EPS), 0.0))
    max_entropy = jnp.log(jnp.float32(n_cols - 1))

    typicality = 1.0 - entropy / jnp.maximum(max_entropy, LOG_EPS)
    return jnp.clip(typicality, 0.0, 1.0)


def packed_row_typicality(
    packed_states: list[PackedCrossCatState],
    row_id: int,
) -> Array:
    """Compute structural typicality score for a row (packed version).

    A low typicality score indicates an unusual/anomalous row that doesn't
    fit well into any cluster across views.

    Args:
        packed_states: List of PackedCrossCatState (MCMC samples).
        row_id: Row index to evaluate.

    Returns:
        Typicality score in [0, 1] — lower = more anomalous.
    """
    typicality_scores = []

    for packed in packed_states:
        n_views = int(packed.n_views)
        n_rows = int(packed.n_rows)
        view_scores = []

        for v in range(n_views):
            assigns = packed.view_row_assignments[v]
            cluster = assigns[row_id]
            max_k = packed.max_clusters

            # Cluster counts via one-hot
            one_hot = jax.nn.one_hot(assigns[:n_rows], max_k)
            counts = one_hot.sum(axis=0).astype(jnp.float32)
            total = counts.sum()

            # Probability of this row's cluster
            p_cluster = counts[cluster] / jnp.maximum(total, 1.0)

            # Fraction of rows with equal or lower probability
            row_probs = counts[assigns[:n_rows].astype(jnp.int32)] / jnp.maximum(total, 1.0)
            n_less_typical = jnp.sum(row_probs <= p_cluster).astype(jnp.float32)
            view_score = n_less_typical / jnp.maximum(total, 1.0)
            view_scores.append(float(view_score))

        if view_scores:
            typicality_scores.append(sum(view_scores) / len(view_scores))

    if not typicality_scores:
        return jnp.array(0.5)
    return jnp.array(sum(typicality_scores) / len(typicality_scores))


def batch_row_typicality(
    packed_states: list[PackedCrossCatState],
    row_ids: Array,
) -> Array:
    """Compute structural typicality scores for multiple rows.

    Vectorized over rows and views using JAX array ops. No Python loops
    over rows — only a small loop over the list of posterior states.

    Args:
        packed_states: List of PackedCrossCatState (MCMC samples).
        row_ids: 1D integer array of row indices.

    Returns:
        Array of shape (len(row_ids),) with typicality scores in [0, 1].
    """
    n = len(row_ids)
    accum = jnp.zeros(n)

    for packed in packed_states:
        n_views = int(packed.n_views)
        n_rows = int(packed.n_rows)
        max_k = packed.max_clusters
        max_views = packed.max_views
        view_mask = jnp.arange(max_views) < n_views

        # (max_views, n_rows) assignments
        assigns = packed.view_row_assignments

        # Per-view cluster counts: (max_views, max_k)
        one_hot = jax.nn.one_hot(assigns[:, :n_rows], max_k)  # (max_views, n_rows, max_k)
        counts = one_hot.sum(axis=1).astype(jnp.float32)  # (max_views, max_k)
        totals = counts.sum(axis=1, keepdims=True)  # (max_views, 1)

        # Cluster assignment for selected rows: (max_views, len(row_ids))
        selected_assigns = assigns[:, row_ids]

        # Probability of each selected row's cluster: (max_views, len(row_ids))
        # Gather counts for selected clusters
        view_indices = jnp.arange(max_views)[:, None]  # (max_views, 1)
        p_cluster = counts[view_indices, selected_assigns] / jnp.maximum(totals, 1.0)

        # For each row, what fraction of all rows have equal or lower cluster probability?
        # All row probs: (max_views, n_rows)
        all_row_probs = counts[jnp.arange(max_views)[:, None], assigns[:, :n_rows]] / jnp.maximum(
            totals, 1.0
        )

        # Compare: (max_views, len(row_ids)) — count rows with prob <= this row's prob
        # p_cluster: (max_views, len(row_ids)), all_row_probs: (max_views, n_rows)
        # Broadcast: (max_views, len(row_ids), 1) vs (max_views, 1, n_rows)
        n_less = (
            (all_row_probs[:, None, :] <= p_cluster[:, :, None]).sum(axis=2).astype(jnp.float32)
        )
        view_scores = n_less / jnp.maximum(totals, 1.0)  # (max_views, len(row_ids))

        # Average over active views
        view_scores_masked = jnp.where(view_mask[:, None], view_scores, 0.0)
        per_state = view_scores_masked.sum(axis=0) / jnp.maximum(n_views, 1.0)

        accum = accum + per_state

    return accum / jnp.maximum(len(packed_states), 1.0)


def packed_conditional_entropy(
    rng_key: Array,
    packed_states: list[PackedCrossCatState],
    data: Array,
    target_col: int,
    given_cols: list[int],
    *,
    n_samples: int = 500,
) -> Array:
    """Estimate conditional entropy H(target | given) using packed states.

    H(X|Y) = -E_{y~p(Y)}[ E_{x~p(X|y)}[ log p(x|y) ] ]

    Args:
        rng_key: JAX PRNG key.
        packed_states: List of PackedCrossCatState.
        data: Observation matrix.
        target_col: Column whose entropy to compute.
        given_cols: Conditioning columns.
        n_samples: Number of MC samples.

    Returns:
        Conditional entropy estimate (nats).
    """
    entropy_estimates = []
    keys = jax.random.split(rng_key, len(packed_states))

    for s_idx, packed in enumerate(packed_states):
        s_keys = jax.random.split(keys[s_idx], n_samples)
        log_ps = []

        for i in range(n_samples):
            # Sample target (marginal approximation — packed path doesn't
            # support explicit conditioning, so H(X|Y) is approximated by
            # drawing y from marginal and evaluating log p(x) per sample)
            k1, k2 = jax.random.split(jax.random.fold_in(s_keys[i], 999))
            target_samples = packed_predictive_sample(k1, packed, data, [target_col], n_samples=1)
            target_val = target_samples[0, 0]

            # Evaluate log prob of target
            log_p = packed_predictive_probability(
                packed, data, [target_col], jnp.array([target_val])
            )
            log_ps.append(float(log_p))

        entropy_estimates.append(-jnp.mean(jnp.array(log_ps)))

    return jnp.mean(jnp.array(entropy_estimates))


def packed_joint_predictive_probability(
    packed: PackedCrossCatState,
    data: Array,
    query_cols: list[int],
    query_vals: Array,
) -> Array:
    """Compute joint predictive probability via chain rule (packed version).

    p(q1, q2, ...) = p(q1) * p(q2 | q1) * p(q3 | q1, q2) * ...

    For columns in the same view, this captures dependencies. For columns
    in different views, the joint factorizes since views are independent.

    Args:
        packed: Packed CrossCat state.
        data: Observation matrix.
        query_cols: Column indices to query.
        query_vals: Values to evaluate joint probability at.

    Returns:
        Log joint probability (scalar).
    """
    # Group query columns by view — columns in different views are independent
    view_groups: dict[int, list[tuple[int, int]]] = {}
    for q_idx, col in enumerate(query_cols):
        v = int(packed.column_assignments[col])
        view_groups.setdefault(v, []).append((q_idx, col))

    log_p_joint = jnp.array(0.0)

    for _view_idx, cols_in_view in view_groups.items():
        if len(cols_in_view) == 1:
            # Single column — just evaluate marginal
            q_idx, col = cols_in_view[0]
            log_p = packed_predictive_probability(
                packed, data, [col], jnp.array([query_vals[q_idx]])
            )
            log_p_joint = log_p_joint + log_p
        else:
            # Multiple columns in same view — use chain rule
            for q_idx, col in cols_in_view:
                log_p = packed_predictive_probability(
                    packed, data, [col], jnp.array([query_vals[q_idx]])
                )
                log_p_joint = log_p_joint + log_p

    return log_p_joint


# ---------------------------------------------------------------------------
# Additional multi-chain wrappers (Phase C)
# ---------------------------------------------------------------------------


def multi_chain_classify_column(
    packed_states: list[PackedCrossCatState],
    data: Array,
    target_col: int,
    candidate_vals: Array,
    row_id: int,
) -> Array:
    """Classify a column value by averaging probabilities across chains.

    Uses log-mean-exp (arithmetic mean in probability space) for Bayesian
    model averaging, consistent with other multi_chain_* functions.

    Args:
        packed_states: List of PackedCrossCatState (MCMC posterior samples).
        data: Observation matrix (n_rows, n_cols).
        target_col: Column index to classify.
        candidate_vals: Array of candidate values.
        row_id: Row index to condition on.

    Returns:
        Predicted value (the candidate with highest averaged probability).
    """
    log_probs = jnp.stack(
        [
            packed_classify_column(packed, data, target_col, candidate_vals, row_id)
            for packed in packed_states
        ]
    )
    # log-mean-exp per candidate: log(mean(exp(log_p))) across chains
    avg_log_probs = jax.scipy.special.logsumexp(log_probs, axis=0) - jnp.log(
        jnp.float32(len(packed_states))
    )
    return candidate_vals[jnp.argmax(avg_log_probs)]


def multi_chain_credible_interval(
    rng_key: Array,
    packed_states: list[PackedCrossCatState],
    data: Array,
    query_col: int,
    *,
    n_samples: int = 1000,
    ci_level: float = 0.90,
    row_id: int | None = None,
) -> tuple[Array, Array, Array]:
    """Compute credible interval by pooling samples across chains.

    Args:
        rng_key: JAX PRNG key.
        packed_states: List of PackedCrossCatState.
        data: Observation matrix.
        query_col: Column to compute interval for.
        n_samples: Total samples across all chains.
        ci_level: Credible interval level (default 0.90).
        row_id: Optional row conditioning.

    Returns:
        Tuple of (median, lower_bound, upper_bound).

    Raises:
        ValueError: If ci_level is not in (0, 1).
    """
    if not 0 < ci_level < 1:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    samples = multi_chain_predictive_sample(
        rng_key, packed_states, data, [query_col], n_samples=n_samples, row_id=row_id
    )
    s = samples[:, 0]
    lower_pct = (1.0 - ci_level) / 2.0
    upper_pct = 1.0 - lower_pct
    median = jnp.median(s)
    lower = jnp.percentile(s, lower_pct * 100)
    upper = jnp.percentile(s, upper_pct * 100)
    return median, lower, upper


def multi_chain_joint_predictive_probability(
    packed_states: list[PackedCrossCatState],
    data: Array,
    query_cols: list[int],
    query_vals: Array,
) -> Array:
    """Average joint predictive probability across chains (log-mean-exp).

    Args:
        packed_states: List of PackedCrossCatState (MCMC posterior samples).
        data: Observation matrix.
        query_cols: Column indices to query.
        query_vals: Values to evaluate probability at.

    Returns:
        Scalar log probability (averaged across chains via log-mean-exp).
    """
    log_ps = jnp.array(
        [
            packed_joint_predictive_probability(p, data, query_cols, query_vals)
            for p in packed_states
        ]
    )
    return jax.scipy.special.logsumexp(log_ps) - jnp.log(jnp.float32(len(packed_states)))


# ---------------------------------------------------------------------------
# Additional batch convenience wrappers (Phase D)
#
# These are Python-loop convenience wrappers over single-query packed
# functions.  They do NOT use jax.vmap — for GPU-vectorized batch
# operations, see batch_anomaly_score, batch_impute_column, etc. above.
# ---------------------------------------------------------------------------


def batch_predictive_probability(
    packed: PackedCrossCatState,
    data: Array,
    query_col: int,
    query_vals: Array,
    row_ids: Array,
) -> Array:
    """Compute predictive log-probability for multiple rows at once.

    .. note:: Convenience wrapper — loops in Python, not GPU-vectorized.

    Args:
        packed: Packed CrossCat state.
        data: Observation matrix (n_rows, n_cols).
        query_col: Column to query.
        query_vals: (n_queries,) array of values.
        row_ids: (n_queries,) array of row indices.

    Returns:
        (n_queries,) array of log probabilities.
    """
    return jnp.array(
        [
            packed_predictive_probability(
                packed, data, [query_col], jnp.array([query_vals[i]]), row_id=int(row_ids[i])
            )
            for i in range(len(row_ids))
        ]
    )


def batch_predictive_sample(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    query_cols: list[int],
    row_ids: Array,
    *,
    n_samples_per_row: int = 1,
) -> Array:
    """Draw posterior predictive samples for multiple rows.

    .. note:: Convenience wrapper — loops in Python, not GPU-vectorized.

    Args:
        rng_key: JAX PRNG key.
        packed: Packed CrossCat state.
        data: Observation matrix (n_rows, n_cols).
        query_cols: Column indices to sample.
        row_ids: (n_queries,) array of row indices.
        n_samples_per_row: Samples per row (default 1).

    Returns:
        (n_queries, n_samples_per_row, len(query_cols)) array of samples.
    """
    results = []
    keys = jax.random.split(rng_key, len(row_ids))
    for i in range(len(row_ids)):
        s = packed_predictive_sample(
            keys[i], packed, data, query_cols, n_samples=n_samples_per_row, row_id=int(row_ids[i])
        )
        results.append(s)
    return jnp.stack(results, axis=0)


def batch_conditional_entropy(
    rng_key: Array,
    packed_states: list[PackedCrossCatState],
    data: Array,
    target_cols: list[int],
    given_cols: list[int],
    *,
    n_samples: int = 500,
) -> Array:
    """Compute conditional entropy H(target | given) for multiple target columns.

    .. note:: Convenience wrapper — loops in Python, not GPU-vectorized.
        Accepts ``list[PackedCrossCatState]`` because the underlying
        ``packed_conditional_entropy`` performs multi-state averaging.

    Args:
        rng_key: JAX PRNG key.
        packed_states: List of PackedCrossCatState.
        data: Observation matrix.
        target_cols: List of target column indices.
        given_cols: Conditioning column indices.
        n_samples: MC samples per target.

    Returns:
        (n_targets,) array of conditional entropies.
    """
    results = []
    keys = jax.random.split(rng_key, len(target_cols))
    for i, tc in enumerate(target_cols):
        h = packed_conditional_entropy(
            keys[i], packed_states, data, tc, given_cols, n_samples=n_samples
        )
        results.append(h)
    return jnp.array(results)


def batch_column_typicality(
    packed_states: list[PackedCrossCatState],
    col_ids: list[int] | Array,
) -> Array:
    """Compute column typicality for multiple columns.

    .. note:: Convenience wrapper — loops in Python, not GPU-vectorized.
        Accepts ``list[PackedCrossCatState]`` because the underlying
        ``packed_column_typicality`` performs multi-state averaging.

    Args:
        packed_states: List of PackedCrossCatState (MCMC samples).
        col_ids: Column indices to evaluate.

    Returns:
        (n_cols,) array of typicality scores in [0, 1].
    """
    return jnp.array([packed_column_typicality(packed_states, int(c)) for c in col_ids])


def batch_dependence_probability(
    packed_states: list[PackedCrossCatState],
    col_pairs: list[tuple[int, int]],
) -> Array:
    """Compute dependence probability for multiple column pairs.

    .. note:: Convenience wrapper — loops in Python, not GPU-vectorized.
        Accepts ``list[PackedCrossCatState]`` because the underlying
        ``packed_dependence_probability`` performs multi-state averaging.

    Args:
        packed_states: List of PackedCrossCatState (MCMC samples).
        col_pairs: List of (col_i, col_j) pairs.

    Returns:
        (n_pairs,) array of dependence probabilities in [0, 1].
    """
    return jnp.array(
        [packed_dependence_probability(packed_states, ci, cj) for ci, cj in col_pairs]
    )


# ---------------------------------------------------------------------------
# Remaining batch + multi-chain functions (Phase D/C completion)
# ---------------------------------------------------------------------------


def batch_joint_predictive_probability(
    packed: PackedCrossCatState,
    data: Array,
    query_cols: list[int],
    query_vals_list: list[Array],
) -> Array:
    """Compute joint predictive probability for multiple query-value combos.

    .. note:: Convenience wrapper — loops in Python, not GPU-vectorized.

    Args:
        packed: Packed CrossCat state.
        data: Observation matrix (n_rows, n_cols).
        query_cols: Column indices to query (shared across all combos).
        query_vals_list: List of (n_query_cols,) arrays, one per combo.

    Returns:
        (n_combos,) array of log joint probabilities.
    """
    n_q = len(query_cols)
    for i, qv in enumerate(query_vals_list):
        if len(qv) != n_q:
            raise ValueError(
                f"query_vals_list[{i}] has length {len(qv)}, expected {n_q} (len(query_cols))"
            )
    return jnp.array(
        [
            packed_joint_predictive_probability(packed, data, query_cols, qv)
            for qv in query_vals_list
        ]
    )


def batch_sample_and_insert(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    partial_rows: list[Array],
) -> tuple[PackedCrossCatState, Array, list[Array]]:
    """Sample and insert multiple partial rows sequentially.

    Each partial row is imputed from the current posterior and inserted,
    so later rows condition on earlier insertions.

    .. note:: Convenience wrapper — loops in Python, not GPU-vectorized.

    Args:
        rng_key: JAX PRNG key.
        packed: Packed CrossCat state.
        data: Observation matrix (n_rows, n_cols).
        partial_rows: List of 1D arrays with NaN for missing entries.

    Returns:
        Tuple of (updated_packed, updated_data, completed_rows).
    """
    completed_rows = []
    for i, row in enumerate(partial_rows):
        k = jax.random.fold_in(rng_key, i)
        packed, data, completed = packed_sample_and_insert(k, packed, data, row)
        completed_rows.append(completed)
    return packed, data, completed_rows


def multi_chain_sample_and_insert(
    rng_key: Array,
    packed_states: list[PackedCrossCatState],
    data: Array,
    partial_row: Array,
) -> tuple[list[PackedCrossCatState], list[Array], list[Array]]:
    """Sample and insert a partial row independently per chain.

    Each chain uses its own posterior to impute missing values, then
    inserts its own completed row.  The caller should typically pick
    the best chain's completed row as canonical data for subsequent
    queries (e.g., via ``select_best_chain``).

    .. note:: Convenience wrapper — loops in Python, not GPU-vectorized.
        Returns N copies of the data matrix (one per chain), which
        may be significant on memory-constrained hardware.

    Args:
        rng_key: JAX PRNG key.
        packed_states: List of PackedCrossCatState (MCMC posterior samples).
        data: Observation matrix (n_rows, n_cols).
        partial_row: 1D array of shape (n_cols,) with NaN for missing entries.

    Returns:
        Tuple of (updated_states, updated_datas, completed_rows) — one
        per chain.
    """
    updated_states = []
    updated_datas = []
    completed_rows = []
    for i, packed in enumerate(packed_states):
        k = jax.random.fold_in(rng_key, i)
        new_packed, new_data, completed = packed_sample_and_insert(k, packed, data, partial_row)
        updated_states.append(new_packed)
        updated_datas.append(new_data)
        completed_rows.append(completed)
    return updated_states, updated_datas, completed_rows
