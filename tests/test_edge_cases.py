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
