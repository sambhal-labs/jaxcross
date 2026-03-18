"""MCMC convergence diagnostics and inference quality metrics.

Maps to original CrossCat:
- diagnostic_utils.py: get_logscore, get_num_views, get_column_crp_alpha
- convergence_test_utils.py: calc_ari, get_column_ARI, multi_chain_ARI
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from crosscat.model import log_joint
from crosscat.types import CrossCatState


def adjusted_rand_index(assignments_true: Array, assignments_pred: Array) -> Array:
    """Compute Adjusted Rand Index between two partitions.

    Maps to original convergence_test_utils.calc_ari().

    ARI = (RI - Expected_RI) / (max_RI - Expected_RI)

    Args:
        assignments_true: Ground truth cluster assignments, shape (n,).
        assignments_pred: Predicted cluster assignments, shape (n,).

    Returns:
        ARI score in [-1, 1]. 1 = perfect, 0 = random, <0 = worse than random.
    """
    n = assignments_true.shape[0]

    # Build contingency table
    n_true = int(jnp.max(assignments_true)) + 1
    n_pred = int(jnp.max(assignments_pred)) + 1

    contingency = jnp.zeros((n_true, n_pred), dtype=jnp.float32)
    for i in range(n):
        t = int(assignments_true[i])
        p = int(assignments_pred[i])
        contingency = contingency.at[t, p].add(1.0)

    # Row sums and column sums
    a = contingency.sum(axis=1)  # shape (n_true,)
    b = contingency.sum(axis=0)  # shape (n_pred,)

    # Combinatorial terms: C(x, 2) = x * (x-1) / 2
    def comb2(x):
        return x * (x - 1.0) / 2.0

    sum_comb_nij = jnp.sum(comb2(contingency))
    sum_comb_a = jnp.sum(comb2(a))
    sum_comb_b = jnp.sum(comb2(b))
    comb_n = comb2(jnp.float32(n))

    expected = sum_comb_a * sum_comb_b / jnp.maximum(comb_n, 1e-30)
    max_index = 0.5 * (sum_comb_a + sum_comb_b)

    denominator = max_index - expected
    # Handle edge case where denominator is 0 (all in one cluster)
    ari = jnp.where(
        jnp.abs(denominator) < 1e-10,
        jnp.where(jnp.abs(sum_comb_nij - expected) < 1e-10, 1.0, 0.0),
        (sum_comb_nij - expected) / denominator,
    )
    return ari


def column_partition_ari(state: CrossCatState, true_assignments: Array) -> Array:
    """Compute ARI of column partition vs ground truth.

    Maps to original convergence_test_utils.get_column_ARI().

    Args:
        state: CrossCat state.
        true_assignments: Ground truth column-to-view assignments.

    Returns:
        ARI score for column partition.
    """
    return adjusted_rand_index(true_assignments, state.column_assignments)


def row_partition_ari(state: CrossCatState, view_idx: int, true_assignments: Array) -> Array:
    """Compute ARI of row partition in a view vs ground truth.

    Args:
        state: CrossCat state.
        view_idx: View index.
        true_assignments: Ground truth row cluster assignments.

    Returns:
        ARI score for row partition in the specified view.
    """
    return adjusted_rand_index(true_assignments, state.views[view_idx].row_assignments)


def collect_diagnostics(state: CrossCatState, data: Array) -> dict:
    """Collect per-sweep diagnostic metrics.

    Maps to original diagnostic_utils.py functions.

    Args:
        state: Current CrossCat state.
        data: Observation matrix.

    Returns:
        Dictionary with diagnostic metrics.
    """
    n_clusters_per_view = []
    row_crp_alphas = []
    for view in state.views:
        n_c = int(jnp.max(view.row_assignments)) + 1
        n_clusters_per_view.append(n_c)
        row_crp_alphas.append(float(view.row_crp_alpha))

    return {
        "log_joint": float(log_joint(state, data)),
        "n_views": state.n_views,
        "column_crp_alpha": float(state.column_crp_alpha),
        "n_clusters_per_view": n_clusters_per_view,
        "row_crp_alphas": row_crp_alphas,
    }


def mean_test_log_likelihood(
    state: CrossCatState,
    data: Array,
    test_rows: Array,
) -> Array:
    """Compute mean test log-likelihood on held-out rows.

    Maps to original convergence_test_utils.calc_mean_test_log_likelihood().

    Args:
        state: CrossCat state.
        data: Full observation matrix (training + test).
        test_rows: Indices of test rows.

    Returns:
        Mean log-likelihood over test rows.
    """
    from crosscat.inference import predictive_probability

    total_ll = jnp.array(0.0)
    n_scored = 0

    for row_idx in test_rows.tolist():
        row_idx = int(row_idx)
        for col in range(state.n_cols):
            x = data[row_idx, col]
            if jnp.isnan(x):
                continue
            log_p = predictive_probability(state, data, [col], jnp.array([x]), row_id=row_idx)
            total_ll = total_ll + log_p
            n_scored += 1

    return total_ll / jnp.maximum(jnp.float32(n_scored), 1.0)
