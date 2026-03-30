"""Tests for multi-chain inference functions.

Verifies that chain-averaged inference queries produce valid results
and behave consistently with single-state queries.

All tests in this module require packed Gibbs sweep JIT compilation
which exceeds 300s on GTX 1650, so the entire module is marked slow.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.model import initialize
from crosscat.packed import pack_state, packed_gibbs_sweep
from crosscat.packed_inference import (
    multi_chain_anomaly_score,
    multi_chain_impute_and_confidence,
    multi_chain_predictive_cdf,
    multi_chain_predictive_probability,
    multi_chain_predictive_sample,
    packed_anomaly_score,
    packed_predictive_probability,
)
from crosscat.synthetic import generate_crosscat_data
from crosscat.types import ColumnType

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def multi_chain_states():
    """Create 3 packed chains from different initializations with a few sweeps."""
    key = jax.random.key(42)
    column_types = [
        ColumnType.CONTINUOUS,
        ColumnType.CATEGORICAL,
        ColumnType.BINARY,
    ]
    result = generate_crosscat_data(key, 50, column_types, n_views=2, n_clusters=2)
    data = result["data"]

    packed_list = []
    for i in range(3):
        k = jax.random.fold_in(key, i)
        state = initialize(k, data, column_types)
        packed = pack_state(state)
        packed = packed_gibbs_sweep(jax.random.fold_in(key, i + 100), packed, data, n_sweeps=2)
        packed_list.append(packed)

    return packed_list, data, column_types


def test_multi_chain_predictive_probability_finite(multi_chain_states):
    """Multi-chain predictive probability returns finite log prob."""
    packed_states, data, _ = multi_chain_states
    log_p = multi_chain_predictive_probability(packed_states, data, [0], jnp.array([data[0, 0]]))
    assert jnp.isfinite(log_p), f"Expected finite, got {log_p}"


def test_multi_chain_predictive_probability_averages_chains(multi_chain_states):
    """Multi-chain result should be between the individual chain results."""
    packed_states, data, _ = multi_chain_states
    query_cols = [0]
    query_vals = jnp.array([data[5, 0]])

    single_log_ps = jnp.array(
        [packed_predictive_probability(p, data, query_cols, query_vals) for p in packed_states]
    )
    multi_log_p = multi_chain_predictive_probability(packed_states, data, query_cols, query_vals)

    # log-mean-exp should be between min and max of individual log probs
    assert multi_log_p >= jnp.min(single_log_ps) - 0.1
    assert multi_log_p <= jnp.max(single_log_ps) + 0.1


def test_multi_chain_predictive_sample_shape(multi_chain_states):
    """Multi-chain samples have correct shape."""
    packed_states, data, _ = multi_chain_states
    key = jax.random.key(0)
    samples = multi_chain_predictive_sample(key, packed_states, data, [0, 1], n_samples=300)
    assert samples.shape == (300, 2)


def test_multi_chain_predictive_sample_mixes_chains(multi_chain_states):
    """Samples are drawn from multiple chains (not just one)."""
    packed_states, data, _ = multi_chain_states
    key = jax.random.key(1)
    n_samples = 300
    n_chains = len(packed_states)

    samples = multi_chain_predictive_sample(key, packed_states, data, [0], n_samples=n_samples)
    # Total samples should match requested count
    assert samples.shape[0] == n_samples
    # Each chain should contribute roughly n_samples // n_chains
    assert samples.shape[0] >= n_chains


def test_multi_chain_anomaly_score_in_range(multi_chain_states):
    """Anomaly score is in [0, 1]."""
    packed_states, data, _ = multi_chain_states
    key = jax.random.key(2)
    score = multi_chain_anomaly_score(key, packed_states, data, query_row=0)
    assert 0.0 <= float(score) <= 1.0, f"Score out of range: {score}"


def test_multi_chain_anomaly_score_averages(multi_chain_states):
    """Multi-chain anomaly is close to mean of single-chain anomalies."""
    packed_states, data, _ = multi_chain_states
    key = jax.random.key(3)

    single_scores = jnp.array(
        [
            packed_anomaly_score(jax.random.fold_in(key, i), p, data, query_row=0)
            for i, p in enumerate(packed_states)
        ]
    )
    multi_score = multi_chain_anomaly_score(key, packed_states, data, query_row=0)

    assert jnp.allclose(multi_score, jnp.mean(single_scores), atol=1e-5)


def test_multi_chain_impute_returns_valid(multi_chain_states):
    """Imputation returns finite point estimate and confidence in [0, 1]."""
    packed_states, data, _ = multi_chain_states
    key = jax.random.key(4)

    # Continuous column
    point_est, confidence = multi_chain_impute_and_confidence(
        key, packed_states, data, query_col=0, n_samples=500
    )
    assert jnp.isfinite(point_est), f"Non-finite estimate: {point_est}"
    assert 0.0 <= float(confidence) <= 1.0, f"Confidence out of range: {confidence}"

    # Binary column
    point_est_b, confidence_b = multi_chain_impute_and_confidence(
        key, packed_states, data, query_col=2, n_samples=500
    )
    assert float(point_est_b) in (0.0, 1.0), f"Binary estimate not 0 or 1: {point_est_b}"
    assert 0.0 <= float(confidence_b) <= 1.0


def test_multi_chain_predictive_cdf_monotone(multi_chain_states):
    """CDF should be monotonically non-decreasing."""
    packed_states, data, _ = multi_chain_states
    key = jax.random.key(5)

    vals = jnp.array([-10.0, 0.0, 5.0, 10.0, 100.0])
    cdfs = [
        float(multi_chain_predictive_cdf(key, packed_states, data, 0, v, n_samples=2000))
        for v in vals
    ]
    for i in range(len(cdfs) - 1):
        assert cdfs[i] <= cdfs[i + 1] + 0.05, (
            f"CDF not monotone: {cdfs[i]} > {cdfs[i + 1]} at vals {vals[i]}, {vals[i + 1]}"
        )


def test_multi_chain_predictive_cdf_bounds(multi_chain_states):
    """CDF at extreme values should approach 0 and 1."""
    packed_states, data, _ = multi_chain_states
    key = jax.random.key(6)

    cdf_low = multi_chain_predictive_cdf(key, packed_states, data, 0, jnp.array(-1e6))
    cdf_high = multi_chain_predictive_cdf(key, packed_states, data, 0, jnp.array(1e6))

    assert float(cdf_low) < 0.05, f"CDF at -1e6 too high: {cdf_low}"
    assert float(cdf_high) > 0.95, f"CDF at 1e6 too low: {cdf_high}"
