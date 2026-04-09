"""MCMC convergence diagnostics and inference quality metrics.

Maps to original CrossCat:
- diagnostic_utils.py: get_logscore, get_num_views, get_column_crp_alpha
- convergence_test_utils.py: calc_ari, get_column_ARI, multi_chain_ARI
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from crosscat.model import log_joint
from crosscat.packed.state import PackedCrossCatState
from crosscat.types import LOG_EPS, ColumnType, CrossCatState


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

    contingency = jax.nn.one_hot(assignments_true, n_true).T @ jax.nn.one_hot(
        assignments_pred, n_pred
    )

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

    expected = sum_comb_a * sum_comb_b / jnp.maximum(comb_n, LOG_EPS)
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


def random_holdout_mask(
    rng_key: Array,
    n_rows: int,
    n_cols: int,
    holdout_fraction: float = 0.1,
) -> Array:
    """Generate random boolean mask for held-out evaluation.

    Args:
        rng_key: JAX PRNG key.
        n_rows: Number of rows.
        n_cols: Number of columns.
        holdout_fraction: Fraction of cells to hold out.

    Returns:
        Boolean array of shape (n_rows, n_cols). True = held-out.
    """
    return jax.random.bernoulli(rng_key, holdout_fraction, shape=(n_rows, n_cols))


def evaluate_imputation(
    state: CrossCatState,
    data: Array,
    mask: Array,
    col_types: list[ColumnType],
    *,
    rng_key: Array | None = None,
) -> dict:
    """Evaluate imputation accuracy on held-out cells.

    For each held-out cell (mask == True), computes:
    - Predictive log-likelihood of the true value
    - Point estimate error (MAE for continuous, accuracy for discrete)

    Args:
        state: CrossCat state (single posterior sample).
        data: Full observation matrix, shape (n_rows, n_cols).
        mask: Boolean mask, shape (n_rows, n_cols). True = held-out.
        col_types: Column type per column.
        rng_key: JAX PRNG key (needed for imputation sampling).

    Returns:
        Dictionary with:
            'mae': mean absolute error (continuous columns only)
            'accuracy': fraction correct (categorical/binary/ordinal columns only)
            'mean_log_lik': mean predictive log-likelihood across all held-out cells
            'n_held_out': total number of held-out cells evaluated
            'per_column': dict mapping column index to per-column metrics
    """
    from crosscat.inference import impute_and_confidence, predictive_probability

    if rng_key is None:
        rng_key = jax.random.key(0)

    total_ll = 0.0
    n_scored = 0
    continuous_errors = []
    discrete_correct = 0
    discrete_total = 0
    per_column: dict[int, dict] = {}

    held_out_rows, held_out_cols = jnp.where(mask)

    for idx in range(len(held_out_rows)):
        row_idx = int(held_out_rows[idx])
        col_idx = int(held_out_cols[idx])
        true_val = data[row_idx, col_idx]

        if jnp.isnan(true_val):
            continue

        # Predictive log-likelihood
        log_p = predictive_probability(
            state, data, [col_idx], jnp.array([true_val]), row_id=row_idx
        )
        total_ll += float(log_p)
        n_scored += 1

        # Point estimate via imputation
        rng_key, subkey = jax.random.split(rng_key)
        point_est, _conf = impute_and_confidence(
            subkey, state, data, col_idx, row_id=row_idx, n_samples=200
        )

        col_type = col_types[col_idx]
        if col_idx not in per_column:
            per_column[col_idx] = {"errors": [], "correct": 0, "total": 0, "log_liks": []}

        per_column[col_idx]["log_liks"].append(float(log_p))

        if col_type == ColumnType.CONTINUOUS or col_type == ColumnType.CYCLIC:
            error = float(jnp.abs(true_val - point_est))
            continuous_errors.append(error)
            per_column[col_idx]["errors"].append(error)
        else:
            correct = int(true_val) == int(point_est)
            if correct:
                discrete_correct += 1
                per_column[col_idx]["correct"] += 1
            discrete_total += 1
            per_column[col_idx]["total"] += 1

    # Aggregate
    mae = sum(continuous_errors) / max(len(continuous_errors), 1)
    accuracy = discrete_correct / max(discrete_total, 1)
    mean_ll = total_ll / max(n_scored, 1)

    # Summarize per-column
    per_column_summary = {}
    for col_idx, metrics in per_column.items():
        col_summary: dict = {
            "mean_log_lik": sum(metrics["log_liks"]) / max(len(metrics["log_liks"]), 1),
            "n_held_out": len(metrics["log_liks"]),
        }
        if metrics["errors"]:
            col_summary["mae"] = sum(metrics["errors"]) / len(metrics["errors"])
        if metrics["total"] > 0:
            col_summary["accuracy"] = metrics["correct"] / metrics["total"]
        per_column_summary[col_idx] = col_summary

    return {
        "mae": mae,
        "accuracy": accuracy,
        "mean_log_lik": mean_ll,
        "n_held_out": n_scored,
        "per_column": per_column_summary,
    }


def packed_evaluate_imputation(
    packed: PackedCrossCatState,
    data: Array,
    mask: Array,
    col_types: list[ColumnType],
    *,
    rng_key: Array | None = None,
    n_samples: int = 200,
) -> dict:
    """Evaluate imputation accuracy on held-out cells using packed inference.

    Faster alternative to ``evaluate_imputation`` that uses the packed
    inference path (``packed_predictive_probability``,
    ``packed_impute_and_confidence``).

    .. note::
        Log-likelihood (``mean_log_lik``) uses ``row_id`` conditioning and
        matches the unpacked path exactly. However, point-estimate metrics
        (``mae``, ``accuracy``) may differ slightly because
        ``packed_impute_and_confidence`` uses marginal cluster weights
        rather than row-conditioned weights.

    Args:
        packed: Packed CrossCat state.
        data: Full observation matrix, shape (n_rows, n_cols).
        mask: Boolean mask, shape (n_rows, n_cols). True = held-out.
        col_types: Column type per column. Accepted for API parity with
            ``evaluate_imputation`` but unused — type IDs are read from
            ``packed.col_type_ids`` instead.
        rng_key: JAX PRNG key (needed for imputation sampling).
        n_samples: Number of posterior predictive samples per held-out cell.

    Returns:
        Dictionary with:
            'mae': mean absolute error (continuous columns only)
            'accuracy': fraction correct (categorical/binary/ordinal columns only)
            'mean_log_lik': mean predictive log-likelihood across all held-out cells
            'n_held_out': total number of held-out cells evaluated
            'per_column': dict mapping column index to per-column metrics
    """
    from crosscat.packed.state import CONTINUOUS_ID, CYCLIC_ID
    from crosscat.packed_inference import (
        packed_impute_and_confidence,
        packed_predictive_probability,
    )

    if rng_key is None:
        rng_key = jax.random.key(0)

    total_ll = 0.0
    n_scored = 0
    continuous_errors: list[float] = []
    discrete_correct = 0
    discrete_total = 0
    per_column: dict[int, dict] = {}

    held_out_rows, held_out_cols = jnp.where(mask)

    for idx in range(len(held_out_rows)):
        row_idx = int(held_out_rows[idx])
        col_idx = int(held_out_cols[idx])
        true_val = data[row_idx, col_idx]

        if jnp.isnan(true_val):
            continue

        # Predictive log-likelihood via packed path
        log_p = packed_predictive_probability(
            packed, data, [col_idx], jnp.array([true_val]), row_id=row_idx
        )
        total_ll += float(log_p)
        n_scored += 1

        # Point estimate via packed imputation
        rng_key, subkey = jax.random.split(rng_key)
        point_est, _conf = packed_impute_and_confidence(
            subkey, packed, data, col_idx, n_samples=n_samples
        )

        type_id = int(packed.col_type_ids[col_idx])
        if col_idx not in per_column:
            per_column[col_idx] = {"errors": [], "correct": 0, "total": 0, "log_liks": []}

        per_column[col_idx]["log_liks"].append(float(log_p))

        if type_id in (CONTINUOUS_ID, CYCLIC_ID):
            error = float(jnp.abs(true_val - point_est))
            continuous_errors.append(error)
            per_column[col_idx]["errors"].append(error)
        else:
            correct = int(true_val) == int(point_est)
            if correct:
                discrete_correct += 1
                per_column[col_idx]["correct"] += 1
            discrete_total += 1
            per_column[col_idx]["total"] += 1

    # Aggregate
    mae = sum(continuous_errors) / max(len(continuous_errors), 1)
    accuracy = discrete_correct / max(discrete_total, 1)
    mean_ll = total_ll / max(n_scored, 1)

    # Summarize per-column
    per_column_summary = {}
    for col_idx, metrics in per_column.items():
        col_summary: dict = {
            "mean_log_lik": sum(metrics["log_liks"]) / max(len(metrics["log_liks"]), 1),
            "n_held_out": len(metrics["log_liks"]),
        }
        if metrics["errors"]:
            col_summary["mae"] = sum(metrics["errors"]) / len(metrics["errors"])
        if metrics["total"] > 0:
            col_summary["accuracy"] = metrics["correct"] / metrics["total"]
        per_column_summary[col_idx] = col_summary

    return {
        "mae": mae,
        "accuracy": accuracy,
        "mean_log_lik": mean_ll,
        "n_held_out": n_scored,
        "per_column": per_column_summary,
    }


# ---------------------------------------------------------------------------
# Standard MCMC convergence diagnostics
# ---------------------------------------------------------------------------


def _fft_next_pow2(n: int) -> int:
    """Next power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def _autocorrelation(x: Array) -> Array:
    """Compute autocorrelation of a 1-D trace using FFT.

    Returns autocorrelation at lags 0..n-1, normalized so lag-0 = 1.
    For constant traces (zero variance), returns all ones — this correctly
    yields ESS ≈ 1 since all samples are perfectly correlated.
    """
    n = x.shape[0]
    x_centered = x - jnp.mean(x)
    var = jnp.sum(x_centered**2)
    is_constant = var < LOG_EPS

    fft_len = _fft_next_pow2(2 * n)
    fft_x = jnp.fft.rfft(x_centered, n=fft_len)
    acf = jnp.fft.irfft(fft_x * jnp.conj(fft_x), n=fft_len)[:n]
    acf_normalized = acf / jnp.maximum(acf[0], LOG_EPS)

    return jnp.where(is_constant, jnp.ones(n), acf_normalized)


def effective_sample_size(traces: Array) -> Array:
    """Estimate effective sample size (ESS) from MCMC traces.

    Uses the initial positive sequence estimator (Geyer 1992) applied to
    the pooled autocorrelation across chains.

    .. note::
        This function uses Python control flow (``if pair_sum < 0: break``)
        on JAX traced values and **cannot be JIT-compiled**.  Call it from
        Python after collecting trace arrays.

    Args:
        traces: Array of shape (n_chains, n_samples) containing a scalar
            diagnostic (e.g. log_joint) per sweep per chain.  A 1-D array
            of shape (n_samples,) is treated as a single chain.
            Requires at least 2 samples per chain.

    Returns:
        Scalar ESS estimate (float array).

    Raises:
        ValueError: If fewer than 2 samples per chain.
    """
    if traces.ndim == 1:
        traces = traces[jnp.newaxis, :]

    n_chains, n_samples = traces.shape

    if n_samples < 2:
        raise ValueError(f"ESS requires at least 2 samples per chain, got {n_samples}.")

    # Pool within-chain autocorrelations (vectorized over chains)
    acf_mean = jnp.mean(
        jnp.stack([_autocorrelation(traces[c]) for c in range(n_chains)]),
        axis=0,
    )

    # Initial positive sequence estimator: sum consecutive pairs until
    # a pair sum becomes negative (Geyer 1992).
    tau = 0.0
    for lag in range(1, n_samples, 2):
        pair_sum = acf_mean[lag] + (acf_mean[lag + 1] if lag + 1 < n_samples else 0.0)
        if pair_sum < 0:
            break
        tau += pair_sum
    # τ_int = 1 + 2 * Σ ρ(k), where tau accumulated the pair sums
    tau = 1.0 + 2.0 * tau
    tau = jnp.maximum(tau, 1.0 / n_samples)  # floor at 1 sample

    return jnp.array(n_chains * n_samples / tau)


def gelman_rubin_rhat(traces: Array) -> Array:
    """Compute the Gelman-Rubin R-hat convergence diagnostic.

    R-hat compares between-chain and within-chain variance.  Values close
    to 1.0 indicate convergence; R-hat > 1.1 suggests the chains have
    not mixed.

    Uses the split-R-hat formulation (Vehtari et al. 2021): each chain is
    split in half, doubling the number of "chains" for a stricter test.

    .. note::
        With very short chains (< 20 samples), the variance estimates are
        noisy and R-hat may be unreliable.  Use at least 50-100 samples
        per chain for meaningful results.

    Args:
        traces: Array of shape (n_chains, n_samples) containing a scalar
            diagnostic (e.g. log_joint) per sweep per chain.  Requires
            n_chains >= 2.

    Returns:
        Scalar R-hat estimate (float array), always >= 1.0.  Values near
        1.0 indicate convergence.

    Raises:
        ValueError: If fewer than 2 chains or fewer than 4 samples.
    """
    if traces.ndim == 1:
        raise ValueError("R-hat requires at least 2 chains; got a 1-D trace array.")
    if traces.shape[0] < 2:
        raise ValueError(f"R-hat requires at least 2 chains, got {traces.shape[0]}.")

    # Split each chain in half for split-R-hat
    n_samples = traces.shape[1]
    half = n_samples // 2
    if half < 2:
        raise ValueError(f"Each chain needs at least 4 samples for split-R-hat, got {n_samples}.")
    first_half = traces[:, :half]
    second_half = traces[:, half : 2 * half]
    split_chains = jnp.concatenate([first_half, second_half], axis=0)  # (2*m, half)

    m = split_chains.shape[0]  # number of split chains
    n = split_chains.shape[1]  # samples per split chain

    chain_means = jnp.mean(split_chains, axis=1)
    grand_mean = jnp.mean(chain_means)

    # Between-chain variance
    B = n * jnp.sum((chain_means - grand_mean) ** 2) / (m - 1)

    # Within-chain variance
    chain_vars = jnp.var(split_chains, axis=1, ddof=1)
    W = jnp.mean(chain_vars)

    # Pooled posterior variance estimate
    var_hat = (n - 1) / n * W + B / n

    rhat = jnp.sqrt(var_hat / jnp.maximum(W, LOG_EPS))
    rhat = jnp.maximum(rhat, 1.0)  # R-hat is theoretically >= 1.0
    return rhat
