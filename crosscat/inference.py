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

from jax import Array

from crosscat.types import CrossCatState


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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

    Averaged over multiple posterior states (ensemble inference).

    Args:
        states: List of CrossCat posterior states.
        col_i: First column index.
        col_j: Second column index.
        n_samples: MC samples for MI estimation.

    Returns:
        Tuple of (mutual_information, linfoot_correlation).
    """
    raise NotImplementedError


def row_typicality(
    states: list[CrossCatState],
    row_id: int,
) -> Array:
    """Compute structural typicality score for a row (anomaly detection).

    Maps to original sample_utils.row_structural_typicality().

    A low typicality score indicates an unusual/anomalous row — one that
    doesn't fit well into any cluster across views.

    Averaged over multiple posterior states.

    Args:
        states: List of CrossCat posterior states.
        row_id: Row index to evaluate.

    Returns:
        Typicality score in [0, 1] — lower = more anomalous.
    """
    raise NotImplementedError


def column_typicality(
    states: list[CrossCatState],
    col_id: int,
) -> Array:
    """Compute structural typicality score for a column.

    Maps to original sample_utils.column_structural_typicality().

    Args:
        states: List of CrossCat posterior states.
        col_id: Column index to evaluate.

    Returns:
        Typicality score in [0, 1].
    """
    raise NotImplementedError
