"""Tests for model initialization, row insertion, and input validation.

Covers: initialization modes, insert_rows, input validation,
safe category counting, MH column transitions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.types import ColumnType

# --- Initialization Modes ---


class TestInitializationModes:
    @pytest.mark.parametrize(
        "mode,expected_views",
        [
            ("together", 1),
            ("apart", 4),
        ],
    )
    def test_deterministic_modes(self, rng_key, mode, expected_views):
        from crosscat.model import initialize

        data = jax.random.normal(rng_key, (50, 4))
        column_types = [ColumnType.CONTINUOUS] * 4
        state = initialize(rng_key, data, column_types, initialization=mode).state
        assert state.n_views == expected_views

    def test_from_the_prior(self, rng_key):
        from crosscat.model import initialize

        data = jax.random.normal(rng_key, (50, 4))
        column_types = [ColumnType.CONTINUOUS] * 4
        state = initialize(rng_key, data, column_types, initialization="from_the_prior").state
        assert state.n_views >= 1


# --- Row Insertion ---


class TestRowInsertion:
    def test_insert_rows(self, rng_key, simple_state):
        from crosscat.model import insert_rows

        state, data, _ = simple_state
        new_rows = jax.random.normal(rng_key, (5, 4))
        new_state, new_data = insert_rows(rng_key, state, data, new_rows)
        assert new_state.n_rows == state.n_rows + 5
        assert new_data.shape[0] == data.shape[0] + 5
        for view in new_state.views:
            assert view.row_assignments.shape[0] == new_state.n_rows


# --- Input Validation ---


class TestInputValidation:
    """Tests for input validation on public APIs."""

    def test_initialize_rejects_empty_data(self, rng_key):
        from crosscat.model import initialize

        with pytest.raises(ValueError, match="at least one row"):
            initialize(rng_key, jnp.zeros((0, 3)), [ColumnType.CONTINUOUS] * 3)

    def test_initialize_rejects_1d_data(self, rng_key):
        from crosscat.model import initialize

        with pytest.raises(ValueError, match="2-dimensional"):
            initialize(rng_key, jnp.zeros(10), [ColumnType.CONTINUOUS])

    def test_initialize_rejects_column_type_mismatch(self, rng_key):
        from crosscat.model import initialize

        with pytest.raises(ValueError, match="column_types length"):
            initialize(rng_key, jnp.zeros((5, 3)), [ColumnType.CONTINUOUS] * 2)

    def test_initialize_rejects_invalid_initialization(self, rng_key):
        from crosscat.model import initialize

        with pytest.raises(ValueError, match="Unknown initialization"):
            initialize(
                rng_key,
                jnp.zeros((5, 3)),
                [ColumnType.CONTINUOUS] * 3,
                initialization="invalid",
            )


# --- Safe N Categories ---


class TestSafeNCategories:
    """Tests for safe category counting with NaN handling."""

    def test_all_nan_column(self):
        from crosscat.model import _safe_n_categories

        result = _safe_n_categories(jnp.array([jnp.nan, jnp.nan, jnp.nan]))
        assert result == 2  # default minimum

    def test_normal_column(self):
        from crosscat.model import _safe_n_categories

        result = _safe_n_categories(jnp.array([0.0, 1.0, 2.0, 1.0]))
        assert result == 3

    def test_single_category(self):
        from crosscat.model import _safe_n_categories

        result = _safe_n_categories(jnp.array([0.0, 0.0, 0.0]))
        assert result == 1

    def test_with_nan_mixed(self):
        from crosscat.model import _safe_n_categories

        result = _safe_n_categories(jnp.array([0.0, jnp.nan, 2.0, jnp.nan]))
        assert result == 3


# --- MH Column Transitions ---


class TestMHTransition:
    def test_mh_column_sweep(self, rng_key, simple_state):
        from crosscat.gibbs import transition_column_assignments_mh

        state, data, _ = simple_state
        new_state = transition_column_assignments_mh(rng_key, state, data)
        assert new_state.n_rows == state.n_rows
        assert new_state.n_cols == state.n_cols
        assert new_state.n_views >= 1

    def test_mh_via_gibbs_sweep(self, rng_key, simple_state):
        from crosscat.gibbs import gibbs_sweep

        state, data, _ = simple_state
        new_state = gibbs_sweep(
            rng_key,
            state,
            data,
            n_sweeps=1,
            kernels=("row_assignments", "column_assignments_mh", "crp_alphas"),
        )
        assert new_state.n_views >= 1


# --- Initialize all column types ---


class TestInitializeColumnTypes:
    """Verify initialize works for each column type."""

    @pytest.mark.parametrize(
        "col_type,data_fn",
        [
            (ColumnType.CONTINUOUS, lambda k: jax.random.normal(k, (30, 2))),
            (ColumnType.BINARY, lambda k: jnp.where(jax.random.normal(k, (30, 2)) > 0, 1.0, 0.0)),
            (
                ColumnType.CATEGORICAL,
                lambda k: jax.random.randint(k, (30, 2), 0, 4).astype(jnp.float32),
            ),
            (
                ColumnType.ORDINAL,
                lambda k: jax.random.randint(k, (30, 2), 0, 3).astype(jnp.float32),
            ),
            (ColumnType.CYCLIC, lambda k: jax.random.uniform(k, (30, 2)) * 2 * jnp.pi),
        ],
    )
    def test_initialize_and_log_joint(self, rng_key, col_type, data_fn):
        from crosscat.model import initialize, log_joint

        data = data_fn(rng_key)
        state = initialize(rng_key, data, [col_type, col_type]).state
        assert state.n_rows == 30
        assert state.n_cols == 2
        lj = log_joint(state, data)
        assert jnp.isfinite(lj), f"Non-finite log_joint for {col_type}: {lj}"
