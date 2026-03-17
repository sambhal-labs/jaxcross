"""Tests for vectorized (v2) packed kernels and packed inference.

Validates correctness by comparing against unpacked reference implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.model import initialize
from crosscat.packed_state import (
    _add_row_to_suffstats,
    _remove_row_from_suffstats,
    pack_state,
)
from crosscat.types import ColumnType


@pytest.fixture
def mixed_packed_state():
    """Mixed-type packed state for testing v2 kernels."""
    key = jax.random.key(42)
    from crosscat.synthetic import generate_crosscat_data

    column_types = [
        ColumnType.CONTINUOUS,
        ColumnType.CATEGORICAL,
        ColumnType.BINARY,
        ColumnType.CYCLIC,
    ]
    result = generate_crosscat_data(key, 50, column_types, n_views=2, n_clusters=2)
    k2 = jax.random.key(43)
    state = initialize(k2, result["data"], column_types)
    packed = pack_state(state, max_clusters=8, max_categories=8)
    return packed, result["data"], column_types


def test_incremental_suffstats_correctness(mixed_packed_state):
    """Remove row then add it back equals original suffstats."""
    packed, data, column_types = mixed_packed_state
    v = 0  # test first view
    row_idx = 5
    n_cols_v = int(packed.view_n_columns[v])
    col_indices = packed.view_column_indices[v, :n_cols_v]
    old_cluster = packed.view_row_assignments[v, row_idx]

    # Remove row
    ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos = _remove_row_from_suffstats(
        packed.ss_counts[v], packed.ss_sum_x[v], packed.ss_sum_x_sq[v],
        packed.ss_cat_counts[v], packed.ss_sum_sin[v], packed.ss_sum_cos[v],
        old_cluster, data[row_idx], col_indices, packed.col_type_ids,
        packed.max_categories,
    )

    # Add row back to same cluster
    ss_c2, ss_sx2, ss_sxsq2, ss_cat2, ss_sin2, ss_cos2 = _add_row_to_suffstats(
        ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos,
        old_cluster, data[row_idx], col_indices, packed.col_type_ids,
        packed.max_categories,
    )

    # Should match original
    assert jnp.allclose(ss_c2[:, :n_cols_v], packed.ss_counts[v, :, :n_cols_v], atol=1e-5)
    assert jnp.allclose(ss_sx2[:, :n_cols_v], packed.ss_sum_x[v, :, :n_cols_v], atol=1e-5)
    assert jnp.allclose(ss_sxsq2[:, :n_cols_v], packed.ss_sum_x_sq[v, :, :n_cols_v], atol=1e-5)
    assert jnp.allclose(ss_sin2[:, :n_cols_v], packed.ss_sum_sin[v, :, :n_cols_v], atol=1e-5)
    assert jnp.allclose(ss_cos2[:, :n_cols_v], packed.ss_sum_cos[v, :, :n_cols_v], atol=1e-5)
