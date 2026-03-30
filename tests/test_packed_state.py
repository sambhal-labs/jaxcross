"""Tests for packed state representation: unique coverage not in other test files.

Covers:
- Pack/unpack log_joint numerical preservation
- unpack_state(data=...) exact suffstat recomputation
- Vectorized suffstats vs original loop-based implementation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.model import _compute_suffstats_for_view, initialize, log_joint
from crosscat.packed import (
    compute_suffstats_vectorized,
    pack_state,
    unpack_state,
)
from crosscat.types import ColumnType

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
# Pack/unpack log_joint preservation
# ---------------------------------------------------------------------------


def test_pack_unpack_preserves_log_joint(continuous_state_and_data):
    """Pack/unpack roundtrip preserves log_joint value."""
    state, data, column_types = continuous_state_and_data
    lj_original = float(log_joint(state, data))

    packed = pack_state(state)
    recovered = unpack_state(packed, column_types)
    lj_recovered = float(log_joint(recovered, data))

    assert lj_recovered == pytest.approx(lj_original, rel=1e-4)


def test_unpack_with_data_exact_log_joint(mixed_state_and_data):
    """unpack_state(data=...) recomputes suffstats for exact log_joint."""
    state, data, column_types = mixed_state_and_data
    lj_original = float(log_joint(state, data))

    packed = pack_state(state)

    # Without data: may have precision loss
    recovered_no_data = unpack_state(packed, column_types)
    lj_no_data = float(log_joint(recovered_no_data, data))

    # With data: suffstats recomputed from scratch
    recovered_with_data = unpack_state(packed, column_types, data=data)
    lj_with_data = float(log_joint(recovered_with_data, data))

    # With data should be exact
    assert lj_with_data == pytest.approx(lj_original, abs=1e-4)
    # And closer than without data
    assert abs(lj_with_data - lj_original) <= abs(lj_no_data - lj_original)


# ---------------------------------------------------------------------------
# Vectorized suffstats vs original
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
