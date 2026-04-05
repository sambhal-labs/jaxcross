"""Tests for packed inference parity functions.

Verifies that packed versions of credible_interval, column_typicality,
row_typicality, conditional_entropy, and joint_predictive_probability
produce valid and consistent results.

All tests require packed Gibbs sweep JIT compilation in the fixture,
which exceeds 300s on GTX 1650, so the entire module is marked slow.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.inference import (
    column_typicality,
    credible_interval,
    impute_and_confidence,
    joint_predictive_probability,
    row_typicality,
)
from crosscat.model import initialize
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat.packed_inference import (
    multi_chain_impute_and_confidence,
    packed_column_typicality,
    packed_conditional_entropy,
    packed_credible_interval,
    packed_impute_and_confidence,
    packed_joint_predictive_probability,
    packed_row_typicality,
)
from crosscat.synthetic import generate_crosscat_data
from crosscat.types import ColumnType

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def inference_setup():
    """Create packed states with a few sweeps for stable inference."""
    key = jax.random.key(42)
    column_types = [
        ColumnType.CONTINUOUS,
        ColumnType.CATEGORICAL,
        ColumnType.BINARY,
    ]
    result = generate_crosscat_data(key, 50, column_types, n_views=2, n_clusters=2)
    data = result["data"]

    # Create 2 states with different initializations (packed path for speed)
    states = []
    packed_states = []
    for i in range(2):
        k = jax.random.fold_in(key, i)
        state = initialize(k, data, column_types).state
        packed = pack_state(state)
        packed = packed_gibbs_sweep(jax.random.fold_in(key, i + 100), packed, data, n_sweeps=3)
        packed_states.append(packed)
        states.append(unpack_state(packed, column_types, data=data))

    return states, packed_states, data, column_types


# ---------------------------------------------------------------------------
# Credible interval
# ---------------------------------------------------------------------------


def test_packed_credible_interval_returns_valid(inference_setup):
    """Packed credible interval returns finite median and ordered bounds."""
    _, packed_states, data, _ = inference_setup
    key = jax.random.key(10)
    median, lower, upper = packed_credible_interval(key, packed_states[0], data, 0)

    assert jnp.isfinite(median)
    assert jnp.isfinite(lower)
    assert jnp.isfinite(upper)
    assert lower <= median <= upper


def test_packed_credible_interval_level(inference_setup):
    """Wider CI level produces wider interval."""
    _, packed_states, data, _ = inference_setup
    key = jax.random.key(11)

    _, lo_90, hi_90 = packed_credible_interval(
        key, packed_states[0], data, 0, ci_level=0.90, n_samples=2000
    )
    _, lo_50, hi_50 = packed_credible_interval(
        key, packed_states[0], data, 0, ci_level=0.50, n_samples=2000
    )

    width_90 = float(hi_90 - lo_90)
    width_50 = float(hi_50 - lo_50)
    assert width_90 >= width_50 - 0.5, f"90% CI ({width_90}) should be >= 50% CI ({width_50})"


def test_packed_credible_interval_matches_original(inference_setup):
    """Packed credible interval roughly matches unpacked version."""
    states, packed_states, data, _ = inference_setup
    key = jax.random.key(12)

    med_p, lo_p, hi_p = packed_credible_interval(key, packed_states[0], data, 0, n_samples=3000)
    med_o, lo_o, hi_o = credible_interval(key, states[0], data, 0, n_samples=3000)

    # Should be roughly similar (MC noise)
    assert abs(float(med_p) - float(med_o)) < 2.0, f"Medians differ: packed={med_p}, orig={med_o}"


# ---------------------------------------------------------------------------
# Column typicality
# ---------------------------------------------------------------------------


def test_packed_column_typicality_in_range(inference_setup):
    """Column typicality is in [0, 1]."""
    _, packed_states, _, _ = inference_setup
    for col in range(3):
        score = packed_column_typicality(packed_states, col)
        assert 0.0 <= float(score) <= 1.0, f"Col {col} typicality out of range: {score}"


def test_packed_column_typicality_single_state(inference_setup):
    """Single state returns 0.5."""
    _, packed_states, _, _ = inference_setup
    score = packed_column_typicality([packed_states[0]], 0)
    assert float(score) == 0.5


def test_packed_column_typicality_matches_original(inference_setup):
    """Packed column typicality matches unpacked version."""
    states, packed_states, _, _ = inference_setup
    for col in range(3):
        score_p = packed_column_typicality(packed_states, col)
        score_o = column_typicality(states, col)
        assert abs(float(score_p) - float(score_o)) < 0.1, (
            f"Col {col}: packed={score_p}, orig={score_o}"
        )


# ---------------------------------------------------------------------------
# Row typicality
# ---------------------------------------------------------------------------


def test_packed_row_typicality_in_range(inference_setup):
    """Row typicality is in [0, 1]."""
    _, packed_states, _, _ = inference_setup
    for row in [0, 10, 25]:
        score = packed_row_typicality(packed_states, row)
        assert 0.0 <= float(score) <= 1.0, f"Row {row} typicality out of range: {score}"


def test_packed_row_typicality_matches_original(inference_setup):
    """Packed row typicality matches unpacked version."""
    states, packed_states, _, _ = inference_setup
    for row in [0, 5, 20]:
        score_p = packed_row_typicality(packed_states, row)
        score_o = row_typicality(states, row)
        assert abs(float(score_p) - float(score_o)) < 0.15, (
            f"Row {row}: packed={score_p}, orig={score_o}"
        )


# ---------------------------------------------------------------------------
# Conditional entropy
# ---------------------------------------------------------------------------


def test_packed_conditional_entropy_non_negative(inference_setup):
    """Conditional entropy H(X|Y) >= 0 (with MC tolerance)."""
    _, packed_states, data, _ = inference_setup
    key = jax.random.key(30)
    h = packed_conditional_entropy(
        key, packed_states, data, target_col=0, given_cols=[1], n_samples=200
    )
    assert float(h) >= -1.0, f"Entropy should be ~non-negative, got {h}"


def test_packed_conditional_entropy_finite(inference_setup):
    """Conditional entropy returns finite value."""
    _, packed_states, data, _ = inference_setup
    key = jax.random.key(31)
    h = packed_conditional_entropy(
        key, packed_states, data, target_col=0, given_cols=[2], n_samples=100
    )
    assert jnp.isfinite(h), f"Expected finite, got {h}"


# ---------------------------------------------------------------------------
# Joint predictive probability
# ---------------------------------------------------------------------------


def test_packed_joint_predictive_probability_finite(inference_setup):
    """Joint predictive probability returns finite log prob."""
    _, packed_states, data, _ = inference_setup
    log_p = packed_joint_predictive_probability(
        packed_states[0], data, [0, 2], jnp.array([data[0, 0], data[0, 2]])
    )
    assert jnp.isfinite(log_p), f"Expected finite, got {log_p}"


def test_packed_joint_single_col_equals_marginal(inference_setup):
    """Joint with single column equals marginal."""
    _, packed_states, data, _ = inference_setup
    from crosscat.packed_inference import packed_predictive_probability

    packed = packed_states[0]
    val = jnp.array([data[3, 0]])

    log_p_joint = packed_joint_predictive_probability(packed, data, [0], val)
    log_p_marginal = packed_predictive_probability(packed, data, [0], val)

    assert jnp.allclose(log_p_joint, log_p_marginal, atol=1e-5), (
        f"joint={log_p_joint}, marginal={log_p_marginal}"
    )


def test_packed_joint_predictive_matches_original(inference_setup):
    """Packed joint probability roughly matches unpacked version."""
    states, packed_states, data, _ = inference_setup
    query_cols = [0, 2]
    query_vals = jnp.array([data[0, 0], data[0, 2]])

    log_p_packed = packed_joint_predictive_probability(
        packed_states[0], data, query_cols, query_vals
    )
    log_p_orig = joint_predictive_probability(states[0], data, query_cols, query_vals)

    assert abs(float(log_p_packed) - float(log_p_orig)) < 1.0, (
        f"packed={log_p_packed}, orig={log_p_orig}"
    )


# ---------------------------------------------------------------------------
# Impute and confidence with row_id
# ---------------------------------------------------------------------------


def test_packed_impute_row_id_returns_valid(inference_setup):
    """Packed impute with row_id returns finite estimate and valid confidence."""
    _, packed_states, data, _ = inference_setup
    key = jax.random.key(99)

    point_est, confidence = packed_impute_and_confidence(
        key, packed_states[0], data, 0, row_id=0, n_samples=200
    )
    assert jnp.isfinite(point_est), f"point_est not finite: {point_est}"
    assert 0.0 <= float(confidence) <= 1.0, f"confidence out of range: {confidence}"


def test_packed_impute_row_id_matches_unpacked(inference_setup):
    """Packed impute with row_id roughly matches unpacked version."""
    states, packed_states, data, _ = inference_setup
    key = jax.random.key(101)

    # Use same key for both — sampling-based so allow tolerance
    point_packed, conf_packed = packed_impute_and_confidence(
        key, packed_states[0], data, 0, row_id=0, n_samples=500
    )
    point_orig, conf_orig = impute_and_confidence(key, states[0], data, 0, row_id=0, n_samples=500)

    assert abs(float(point_packed) - float(point_orig)) < 2.0, (
        f"packed={point_packed}, orig={point_orig}"
    )
    assert 0.0 <= float(conf_packed) <= 1.0
    assert 0.0 <= float(conf_orig) <= 1.0


def test_multi_chain_impute_row_id_returns_valid(inference_setup):
    """Multi-chain impute with row_id returns finite estimate and valid confidence."""
    _, packed_states, data, _ = inference_setup
    key = jax.random.key(102)

    point_est, confidence = multi_chain_impute_and_confidence(
        key, packed_states, data, 0, row_id=0, n_samples=200
    )
    assert jnp.isfinite(point_est), f"point_est not finite: {point_est}"
    assert 0.0 <= float(confidence) <= 1.0, f"confidence out of range: {confidence}"
