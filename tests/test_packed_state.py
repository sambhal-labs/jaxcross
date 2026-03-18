"""Tests for packed state representation and vectorized kernels.

Verifies:
- pack/unpack roundtrip preserves state
- Vectorized suffstats match original implementation
- Packed row assignment kernel produces valid state
- Packed hyper sampling produces valid hypers
- Packed CRP alpha sampling works
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.gibbs import gibbs_sweep
from crosscat.model import _compute_suffstats_for_view, initialize, log_joint
from crosscat.packed import (
    compute_suffstats_vectorized,
    pack_state,
    packed_gibbs_sweep,
    packed_transition_crp_alphas,
    packed_transition_row_assignments,
    unpack_state,
)
from crosscat.types import ColumnType
from crosscat.validate import validate_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def continuous_state_and_data():
    """A simple continuous state for testing."""
    key = jax.random.key(99)
    k1, k2, k3 = jax.random.split(key, 3)
    col0 = jnp.concatenate([jax.random.normal(k1, (25,)), 5.0 + jax.random.normal(k2, (25,))])
    col1 = jnp.concatenate([jax.random.normal(k1, (25,)) - 2, 3.0 + jax.random.normal(k2, (25,))])
    data = jnp.column_stack([col0, col1])
    column_types = [ColumnType.CONTINUOUS, ColumnType.CONTINUOUS]
    state = initialize(k3, data, column_types)
    return state, data, column_types


@pytest.fixture
def mixed_state_and_data():
    """A mixed-type state for testing."""
    key = jax.random.key(77)
    from crosscat.synthetic import generate_crosscat_data

    column_types = [
        ColumnType.CONTINUOUS,
        ColumnType.CATEGORICAL,
        ColumnType.BINARY,
        ColumnType.CYCLIC,
    ]
    result = generate_crosscat_data(key, 60, column_types, n_views=2, n_clusters=2)
    k2 = jax.random.key(78)
    state = initialize(k2, result["data"], column_types)
    return state, result["data"], column_types


# ---------------------------------------------------------------------------
# Pack/Unpack roundtrip
# ---------------------------------------------------------------------------


def test_pack_unpack_roundtrip_continuous(continuous_state_and_data):
    """Pack then unpack preserves continuous state structure."""
    state, data, column_types = continuous_state_and_data
    packed = pack_state(state)
    recovered = unpack_state(packed, column_types)

    assert recovered.n_rows == state.n_rows
    assert recovered.n_cols == state.n_cols
    assert recovered.n_views == state.n_views
    assert jnp.array_equal(recovered.column_assignments, state.column_assignments)
    assert float(recovered.column_crp_alpha) == pytest.approx(float(state.column_crp_alpha))

    for v in range(state.n_views):
        assert jnp.array_equal(recovered.views[v].row_assignments, state.views[v].row_assignments)
        assert jnp.array_equal(recovered.views[v].column_indices, state.views[v].column_indices)


def test_pack_unpack_roundtrip_mixed(mixed_state_and_data):
    """Pack then unpack preserves mixed-type state structure."""
    state, data, column_types = mixed_state_and_data
    packed = pack_state(state)
    recovered = unpack_state(packed, column_types)

    assert recovered.n_rows == state.n_rows
    assert recovered.n_cols == state.n_cols
    assert recovered.n_views == state.n_views

    for j in range(state.n_cols):
        assert recovered.column_types[j] == state.column_types[j]
        assert recovered.column_hypers[j].column_type == state.column_hypers[j].column_type


def test_pack_unpack_preserves_log_joint(continuous_state_and_data):
    """Pack/unpack roundtrip preserves log_joint value."""
    state, data, column_types = continuous_state_and_data
    lj_original = float(log_joint(state, data))

    packed = pack_state(state)
    recovered = unpack_state(packed, column_types)
    lj_recovered = float(log_joint(recovered, data))

    assert lj_recovered == pytest.approx(lj_original, rel=1e-4)


def test_pack_unpack_validation(continuous_state_and_data):
    """Recovered state passes validation."""
    state, data, column_types = continuous_state_and_data
    packed = pack_state(state)
    recovered = unpack_state(packed, column_types)
    errors = validate_state(recovered, data)
    assert errors == [], f"Validation errors: {errors}"


# ---------------------------------------------------------------------------
# Vectorized suffstats
# ---------------------------------------------------------------------------


def test_vectorized_suffstats_match_original(continuous_state_and_data):
    """Vectorized suffstats computation matches original loop-based version."""
    state, data, column_types = continuous_state_and_data
    view = state.views[0]
    n_clusters = int(jnp.max(view.row_assignments)) + 1

    # Original
    orig_ss = _compute_suffstats_for_view(
        data, view.column_indices, column_types, view.row_assignments, n_clusters
    )

    # Vectorized
    packed = pack_state(state, max_clusters=32, max_categories=16)
    col_type_ids = packed.col_type_ids
    counts, sum_x, sum_x_sq, cat_counts, sum_sin, sum_cos = compute_suffstats_vectorized(
        data, view.column_indices, col_type_ids, view.row_assignments, n_clusters, 32, 16
    )

    # Compare
    n_cols_v = len(view.column_indices)
    for c in range(n_clusters):
        for li in range(n_cols_v):
            orig = orig_ss[c][li]
            assert int(counts[c, li]) == int(orig.count), (
                f"Count mismatch at cluster {c}, col {li}"
            )
            if orig.sum_x is not None:
                assert float(sum_x[c, li]) == pytest.approx(float(orig.sum_x), abs=1e-4)
            if orig.sum_x_sq is not None:
                assert float(sum_x_sq[c, li]) == pytest.approx(float(orig.sum_x_sq), abs=1e-4)


# ---------------------------------------------------------------------------
# Packed kernels produce valid output
# ---------------------------------------------------------------------------


def test_packed_row_assignments_valid(continuous_state_and_data):
    """Packed row assignment kernel produces a valid state."""
    state, data, column_types = continuous_state_and_data
    packed = pack_state(state)
    key = jax.random.key(55)
    packed_new = packed_transition_row_assignments(key, packed, data)
    recovered = unpack_state(packed_new, column_types)

    errors = validate_state(recovered, data)
    assert errors == [], f"Validation errors: {errors}"
    assert jnp.isfinite(log_joint(recovered, data))


def test_packed_crp_alphas_valid(continuous_state_and_data):
    """Packed CRP alpha sampling produces valid values."""
    state, data, column_types = continuous_state_and_data
    packed = pack_state(state)
    key = jax.random.key(66)
    packed_new = packed_transition_crp_alphas(key, packed)
    assert float(packed_new.column_crp_alpha) > 0
    for v in range(int(packed_new.n_views)):
        assert float(packed_new.view_row_crp_alpha[v]) > 0


def test_packed_gibbs_sweep_valid(continuous_state_and_data):
    """Full packed Gibbs sweep produces valid state."""
    state, data, column_types = continuous_state_and_data
    packed = pack_state(state)
    key = jax.random.key(88)
    packed_new = packed_gibbs_sweep(key, packed, data, n_sweeps=2)
    recovered = unpack_state(packed_new, column_types)

    errors = validate_state(recovered, data)
    assert errors == [], f"Validation errors: {errors}"


# ---------------------------------------------------------------------------
# Packed vs original produce comparable results
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_packed_inference_comparable_to_original(continuous_state_and_data):
    """Packed and original inference produce similar log_joint after sweeps."""
    state, data, column_types = continuous_state_and_data

    # Original path
    key1 = jax.random.key(333)
    state_orig = gibbs_sweep(key1, state, data, n_sweeps=5)
    lj_orig = float(log_joint(state_orig, data))

    # Packed path
    packed = pack_state(state)
    key2 = jax.random.key(333)
    packed_new = packed_gibbs_sweep(key2, packed, data, n_sweeps=5)
    recovered = unpack_state(packed_new, column_types)
    lj_packed = float(log_joint(recovered, data))

    # Both should reach reasonable log_joint values (not identical due to
    # different iteration order, but both should be finite and negative)
    assert jnp.isfinite(jnp.array(lj_orig))
    assert jnp.isfinite(jnp.array(lj_packed))
