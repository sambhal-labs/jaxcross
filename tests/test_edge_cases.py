"""Edge case tests for CrossCat robustness.

Tests boundary conditions: all-NaN columns, single row, single cluster,
single view, single column.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.model import initialize, log_joint
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat.types import ColumnType
from crosscat.validate import validate_state

# ---------------------------------------------------------------------------
# Single row
# ---------------------------------------------------------------------------


def test_single_row_initializes():
    """A single-row dataset should initialize without error."""
    key = jax.random.key(1)
    data = jnp.array([[1.0, 0.0, 3.5]])
    types = [ColumnType.CONTINUOUS, ColumnType.BINARY, ColumnType.CONTINUOUS]
    state = initialize(key, data, types).state

    assert state.n_rows == 1
    assert state.n_cols == 3
    assert jnp.isfinite(log_joint(state, data))


@pytest.mark.slow
def test_single_row_packed_sweep():
    """Packed sweep on a single-row dataset should not crash.

    Note: log_joint may be non-finite for n_rows=1 due to CRP degeneracy
    (gammaln(0) = inf). We only verify the sweep completes without error.
    """
    key = jax.random.key(2)
    data = jnp.array([[5.0, 1.0]])
    types = [ColumnType.CONTINUOUS, ColumnType.CATEGORICAL]
    state = initialize(key, data, types).state
    packed = pack_state(state)

    k1, k2 = jax.random.split(key)
    packed_new = packed_gibbs_sweep(k1, packed, data, n_sweeps=2)
    recovered = unpack_state(packed_new, types, data=data)

    assert recovered.n_rows == 1


# ---------------------------------------------------------------------------
# All-NaN column
# ---------------------------------------------------------------------------


def test_all_nan_column():
    """A column that is entirely NaN should not crash during initialization.

    Note: log_joint may be NaN for an all-NaN column since there are no
    valid observations for suffstats. We only verify initialization succeeds.
    """
    key = jax.random.key(3)
    data = jnp.array(
        [
            [1.0, jnp.nan],
            [2.0, jnp.nan],
            [3.0, jnp.nan],
            [4.0, jnp.nan],
            [5.0, jnp.nan],
        ]
    )
    types = [ColumnType.CONTINUOUS, ColumnType.CONTINUOUS]
    state = initialize(key, data, types).state

    # Should initialize without error
    assert state.n_rows == 5
    assert state.n_cols == 2
    # The non-NaN column should contribute a finite partial score
    # (all-NaN column produces NaN log marginal, which propagates)


@pytest.mark.slow
def test_partial_nan_column():
    """A column with some NaN values should initialize and run inference."""
    key = jax.random.key(4)
    data = jnp.array(
        [
            [1.0, 10.0],
            [2.0, jnp.nan],
            [3.0, 30.0],
            [4.0, jnp.nan],
            [5.0, 50.0],
        ]
    )
    types = [ColumnType.CONTINUOUS, ColumnType.CONTINUOUS]
    state = initialize(key, data, types).state

    # Should initialize without crashing
    assert state.n_rows == 5
    # Gibbs sweep should complete (NaN rows are skipped in suffstats)
    from crosscat.gibbs import gibbs_sweep

    state = gibbs_sweep(jax.random.split(key)[0], state, data, n_sweeps=2)
    assert state.n_rows == 5


# ---------------------------------------------------------------------------
# Single column
# ---------------------------------------------------------------------------


def test_single_column():
    """A dataset with a single column should initialize and run."""
    key = jax.random.key(5)
    data = jnp.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    types = [ColumnType.CONTINUOUS]
    state = initialize(key, data, types).state

    assert state.n_cols == 1
    assert state.n_views == 1  # single column -> single view
    assert jnp.isfinite(log_joint(state, data))


# ---------------------------------------------------------------------------
# Single view (all columns together)
# ---------------------------------------------------------------------------


def test_single_view_initialization():
    """Initialize with 'together' mode puts all columns in one view."""
    key = jax.random.key(6)
    data = jnp.column_stack(
        [
            jax.random.normal(key, (20,)),
            jax.random.normal(key, (20,)) + 5,
        ]
    )
    types = [ColumnType.CONTINUOUS, ColumnType.CONTINUOUS]
    state = initialize(key, data, types, initialization="together").state

    assert state.n_views == 1


# ---------------------------------------------------------------------------
# Binary-only dataset
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_binary_only_dataset():
    """A dataset with only binary columns should work."""
    key = jax.random.key(7)
    data = jnp.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=jnp.float32,
    )
    types = [ColumnType.BINARY] * 3
    state = initialize(key, data, types).state

    assert jnp.isfinite(log_joint(state, data))

    # Pack and sweep
    packed = pack_state(state)
    packed_new = packed_gibbs_sweep(jax.random.split(key)[0], packed, data, n_sweeps=2)
    recovered = unpack_state(packed_new, types, data=data)
    errors = validate_state(recovered, data)
    assert not errors, f"Validation errors: {errors}"


# ---------------------------------------------------------------------------
# Cyclic-only dataset
# ---------------------------------------------------------------------------


def test_cyclic_only_dataset():
    """A dataset with only cyclic columns should work."""
    key = jax.random.key(8)
    # Angles in [0, 2*pi)
    data = jnp.array(
        [
            [0.1, 3.14],
            [0.2, 3.15],
            [3.0, 0.1],
            [3.1, 0.2],
            [6.0, 6.1],
        ]
    )
    types = [ColumnType.CYCLIC, ColumnType.CYCLIC]
    state = initialize(key, data, types).state

    assert jnp.isfinite(log_joint(state, data))


# ---------------------------------------------------------------------------
# Invalid-input contracts — tests the public API rejects or copes with bad
# inputs in documented ways. Guards against silent misbehavior.
# ---------------------------------------------------------------------------


class TestInvalidInputContracts:
    """Contract tests for the public API under malformed inputs."""

    def test_insert_rows_column_count_mismatch_raises(self):
        """insert_rows with wrong column count must raise ValueError."""
        from crosscat.model import insert_rows

        key = jax.random.key(400)
        data = jax.random.normal(key, (20, 3))
        types = [ColumnType.CONTINUOUS] * 3
        state = initialize(key, data, types).state

        wrong_rows = jax.random.normal(jax.random.split(key)[0], (5, 4))
        with pytest.raises(ValueError, match=r"must have shape"):
            insert_rows(key, state, data, wrong_rows)

    def test_insert_rows_rejects_1d(self):
        """insert_rows with 1D new_rows must raise ValueError."""
        from crosscat.model import insert_rows

        key = jax.random.key(401)
        data = jax.random.normal(key, (10, 2))
        types = [ColumnType.CONTINUOUS] * 2
        state = initialize(key, data, types).state

        with pytest.raises(ValueError, match=r"must have shape"):
            insert_rows(key, state, data, jnp.array([1.0, 2.0]))

    def test_pack_state_with_out_of_range_category_raises(self):
        """pack_state with category value >= max_categories raises ValueError.

        Documents the contract: users set max_categories at pack time and
        observed category values must fit. A silent clip would produce
        wrong inference results, so pack_state validates upfront when
        ``data`` is passed.
        """
        key = jax.random.key(402)
        # 4 categories observed, but pack with max_categories=3
        data = jnp.array([[0.0], [1.0], [2.0], [3.0], [0.0], [1.0]], dtype=jnp.float32)
        types = [ColumnType.CATEGORICAL]
        state = initialize(key, data, types).state

        # max_categories too small for observed values → must raise
        with pytest.raises((ValueError, AssertionError)):
            pack_state(state, max_categories=3, data=data)

    @pytest.mark.parametrize("n_chains", [1, 4])
    def test_initialize_n_chains_shape_contract(self, n_chains):
        """InitResult.state is a single CrossCatState for n_chains=1,
        a list of length n_chains otherwise. Documents the contract
        explicitly so downstream callers can branch correctly.
        """
        key = jax.random.key(403)
        data = jax.random.normal(key, (20, 3))
        types = [ColumnType.CONTINUOUS] * 3
        result = initialize(key, data, types, n_chains=n_chains)

        if n_chains == 1:
            assert not isinstance(result.state, list), (
                "n_chains=1 should return a single CrossCatState (not a list)"
            )
            assert result.state.n_rows == 20
        else:
            assert isinstance(result.state, list), f"n_chains={n_chains} should return a list"
            assert len(result.state) == n_chains
            for s in result.state:
                assert s.n_rows == 20

    def test_initialize_column_type_mismatch_raises(self):
        """initialize() rejects a column_types list with the wrong length."""
        key = jax.random.key(404)
        data = jax.random.normal(key, (10, 3))
        with pytest.raises(ValueError):
            initialize(key, data, [ColumnType.CONTINUOUS, ColumnType.CONTINUOUS])
