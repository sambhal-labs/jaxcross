"""Tests for packed row insertion (streaming / online inference).

Verifies that packed_insert_rows and packed_sample_and_insert correctly
insert new rows, update cluster assignments and sufficient statistics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.model import initialize, log_joint
from crosscat.packed import pack_state, packed_gibbs_sweep, packed_insert_rows, unpack_state
from crosscat.packed_inference import packed_sample_and_insert
from crosscat.synthetic import generate_crosscat_data
from crosscat.types import ColumnType
from crosscat.validate import validate_state


@pytest.fixture(scope="module")
def trained_state():
    """Packed state with a few sweeps for stable insertion tests."""
    key = jax.random.key(42)
    column_types = [
        ColumnType.CONTINUOUS,
        ColumnType.CATEGORICAL,
        ColumnType.BINARY,
    ]
    result = generate_crosscat_data(key, 50, column_types, n_views=2, n_clusters=2)
    data = result["data"]

    k1, k2 = jax.random.split(key)
    state = initialize(k1, data, column_types)
    packed = pack_state(state)
    packed = packed_gibbs_sweep(k2, packed, data, n_sweeps=10)

    return packed, data, column_types


# ---------------------------------------------------------------------------
# packed_insert_rows tests
# ---------------------------------------------------------------------------


def test_insert_single_row_increases_n_rows(trained_state):
    """Inserting one row increases n_rows by 1."""
    packed, data, col_types = trained_state
    key = jax.random.key(100)
    new_row = data[0:1]  # duplicate first row
    new_packed, new_data = packed_insert_rows(key, packed, data, new_row)

    assert new_packed.n_rows == packed.n_rows + 1
    assert new_data.shape[0] == data.shape[0] + 1


def test_insert_multiple_rows(trained_state):
    """Inserting multiple rows works correctly."""
    packed, data, col_types = trained_state
    key = jax.random.key(101)
    new_rows = data[:5]  # 5 rows
    new_packed, new_data = packed_insert_rows(key, packed, data, new_rows)

    assert new_packed.n_rows == packed.n_rows + 5
    assert new_data.shape[0] == data.shape[0] + 5


def test_inserted_rows_have_valid_assignments(trained_state):
    """New rows are assigned to valid clusters."""
    packed, data, col_types = trained_state
    key = jax.random.key(102)
    new_rows = data[:3]
    new_packed, _ = packed_insert_rows(key, packed, data, new_rows)

    n_views = int(new_packed.n_views)
    for v in range(n_views):
        n_clusters = int(new_packed.view_n_clusters[v])
        for row_i in range(new_packed.n_rows):
            assign = int(new_packed.view_row_assignments[v, row_i])
            assert 0 <= assign < n_clusters, (
                f"View {v}, row {row_i}: assignment {assign} >= n_clusters {n_clusters}"
            )


def test_suffstats_updated_after_insertion(trained_state):
    """Sufficient statistics counts increase after insertion."""
    packed, data, col_types = trained_state
    key = jax.random.key(103)
    new_rows = data[:3]

    old_total_counts = int(jnp.sum(packed.ss_counts))
    new_packed, _ = packed_insert_rows(key, packed, data, new_rows)
    new_total_counts = int(jnp.sum(new_packed.ss_counts))

    # Each new row contributes to suffstats in each view (3 rows * n_views worth of counts)
    assert new_total_counts > old_total_counts, (
        f"Counts didn't increase: {old_total_counts} -> {new_total_counts}"
    )


def test_inserted_state_unpacks_validly(trained_state):
    """State after insertion can be unpacked and validates."""
    packed, data, col_types = trained_state
    key = jax.random.key(104)
    new_rows = data[:2]
    new_packed, new_data = packed_insert_rows(key, packed, data, new_rows)

    state = unpack_state(new_packed, col_types, data=new_data)
    errors = validate_state(state, new_data)
    assert not errors, f"Validation errors: {errors}"


def test_inserted_state_has_finite_log_joint(trained_state):
    """log_joint is finite after insertion."""
    packed, data, col_types = trained_state
    key = jax.random.key(105)
    new_rows = data[:2]
    new_packed, new_data = packed_insert_rows(key, packed, data, new_rows)

    state = unpack_state(new_packed, col_types, data=new_data)
    lj = log_joint(state, new_data)
    assert jnp.isfinite(lj), f"log_joint not finite: {lj}"


def test_insert_deterministic_with_same_key(trained_state):
    """Same key produces same assignments."""
    packed, data, col_types = trained_state
    key = jax.random.key(106)
    new_rows = data[:2]

    p1, _ = packed_insert_rows(key, packed, data, new_rows)
    p2, _ = packed_insert_rows(key, packed, data, new_rows)

    for v in range(int(packed.n_views)):
        assert jnp.array_equal(p1.view_row_assignments[v], p2.view_row_assignments[v]), (
            f"View {v}: assignments differ with same key"
        )


# ---------------------------------------------------------------------------
# packed_sample_and_insert tests
# ---------------------------------------------------------------------------


def test_sample_and_insert_completes_missing_values(trained_state):
    """NaN values in partial row are filled."""
    packed, data, col_types = trained_state
    key = jax.random.key(200)

    partial_row = jnp.array([1.5, jnp.nan, 0.0])
    _, _, completed = packed_sample_and_insert(key, packed, data, partial_row)

    assert jnp.all(jnp.isfinite(completed)), f"Non-finite values in completed row: {completed}"
    # Observed values should be preserved
    assert float(completed[0]) == 1.5
    assert float(completed[2]) == 0.0


def test_sample_and_insert_fully_observed(trained_state):
    """Fully observed row inserts without sampling."""
    packed, data, col_types = trained_state
    key = jax.random.key(201)

    full_row = jnp.array([2.0, 1.0, 1.0])
    new_packed, new_data, completed = packed_sample_and_insert(key, packed, data, full_row)

    assert new_packed.n_rows == packed.n_rows + 1
    assert jnp.allclose(completed, full_row)


def test_sample_and_insert_all_missing(trained_state):
    """All-NaN row gets fully sampled values."""
    packed, data, col_types = trained_state
    key = jax.random.key(202)

    all_nan = jnp.array([jnp.nan, jnp.nan, jnp.nan])
    new_packed, _, completed = packed_sample_and_insert(key, packed, data, all_nan)

    assert new_packed.n_rows == packed.n_rows + 1
    assert jnp.all(jnp.isfinite(completed)), f"Non-finite: {completed}"
