"""Tests for scaling features: subsample_rows, suggest_max_clusters, InitResult."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.model import initialize
from crosscat.packed import pack_state, packed_gibbs_sweep, packed_insert_rows
from crosscat.packed.state import suggest_max_clusters
from crosscat.types import ColumnType, InitResult
from crosscat.validate import validate_state

# ---------------------------------------------------------------------------
# InitResult return type tests
# ---------------------------------------------------------------------------


class TestInitResult:
    """Verify initialize() always returns InitResult."""

    def test_single_chain_returns_init_result(self):
        key = jax.random.key(1)
        data = jax.random.normal(key, (20, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        result = initialize(key, data, col_types)
        assert isinstance(result, InitResult)
        assert result.subsample_idx is None

    def test_multi_chain_returns_init_result(self):
        key = jax.random.key(2)
        data = jax.random.normal(key, (20, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        result = initialize(key, data, col_types, n_chains=3)
        assert isinstance(result, InitResult)
        assert isinstance(result.state, list)
        assert len(result.state) == 3
        assert result.subsample_idx is None

    def test_state_attribute_works(self):
        key = jax.random.key(3)
        data = jax.random.normal(key, (20, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        state = initialize(key, data, col_types).state
        assert state.n_rows == 20
        assert state.n_cols == 3


# ---------------------------------------------------------------------------
# subsample_rows tests
# ---------------------------------------------------------------------------


class TestSubsampleRows:
    """Verify subsample initialization."""

    def test_subsample_returns_indices(self):
        key = jax.random.key(10)
        data = jax.random.normal(key, (100, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        result = initialize(key, data, col_types, subsample_rows=20)
        assert result.subsample_idx is not None
        assert result.subsample_idx.shape == (20,)
        assert result.state.n_rows == 20

    def test_subsample_indices_sorted(self):
        key = jax.random.key(11)
        data = jax.random.normal(key, (100, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        result = initialize(key, data, col_types, subsample_rows=30)
        idx = result.subsample_idx
        assert jnp.all(idx[1:] >= idx[:-1]), "subsample_idx should be sorted"

    def test_subsample_indices_in_range(self):
        key = jax.random.key(12)
        data = jax.random.normal(key, (50, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        result = initialize(key, data, col_types, subsample_rows=15)
        idx = result.subsample_idx
        assert jnp.all(idx >= 0)
        assert jnp.all(idx < 50)

    def test_subsample_ge_nrows_uses_full_data(self):
        """subsample_rows >= n_rows should use full data (no subsampling)."""
        key = jax.random.key(13)
        data = jax.random.normal(key, (30, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        result = initialize(key, data, col_types, subsample_rows=30)
        assert result.subsample_idx is None
        assert result.state.n_rows == 30

    def test_subsample_greater_than_nrows_uses_full_data(self):
        key = jax.random.key(14)
        data = jax.random.normal(key, (20, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        result = initialize(key, data, col_types, subsample_rows=999)
        assert result.subsample_idx is None
        assert result.state.n_rows == 20

    def test_subsample_one_row(self):
        key = jax.random.key(15)
        data = jax.random.normal(key, (50, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        result = initialize(key, data, col_types, subsample_rows=1)
        assert result.state.n_rows == 1
        assert result.subsample_idx.shape == (1,)

    def test_subsample_less_than_one_raises(self):
        key = jax.random.key(16)
        data = jax.random.normal(key, (20, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        with pytest.raises(ValueError, match="subsample_rows must be >= 1"):
            initialize(key, data, col_types, subsample_rows=0)

    def test_subsample_negative_raises(self):
        key = jax.random.key(17)
        data = jax.random.normal(key, (20, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        with pytest.raises(ValueError, match="subsample_rows must be >= 1"):
            initialize(key, data, col_types, subsample_rows=-5)

    def test_subsample_multi_chain(self):
        key = jax.random.key(18)
        data = jax.random.normal(key, (100, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        result = initialize(key, data, col_types, n_chains=3, subsample_rows=25)
        assert isinstance(result.state, list)
        assert len(result.state) == 3
        for s in result.state:
            assert s.n_rows == 25
        assert result.subsample_idx.shape == (25,)

    def test_subsample_state_validates(self):
        """Subsampled state should pass validation against subsample data."""
        key = jax.random.key(19)
        data = jax.random.normal(key, (50, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        result = initialize(key, data, col_types, subsample_rows=15)
        sub_data = data[result.subsample_idx]
        errors = validate_state(result.state, sub_data)
        assert not errors, f"Validation errors: {errors}"

    def test_subsample_hypers_from_full_data(self):
        """Column hypers should be computed from full data, not subsample."""
        key = jax.random.key(20)
        # Create data with very different mean for subsample vs full
        data = jnp.concatenate(
            [
                jnp.ones((90, 1)) * 100.0,  # most rows = 100
                jnp.ones((10, 1)) * 0.0,  # few rows = 0
            ],
            axis=0,
        )
        col_types = [ColumnType.CONTINUOUS]
        result = initialize(key, data, col_types, subsample_rows=10)
        # Hyper mu should be close to full-data mean (~90), not subsample mean
        mu = float(result.state.column_hypers[0].mu)
        full_mean = float(jnp.mean(data[:, 0]))
        # Allow tolerance since it's a prior mean, but it should NOT be 0 or 100
        assert abs(mu - full_mean) < 1.0, f"Hyper mu={mu} should equal full-data mean={full_mean}"

    def test_subsample_e2e_workflow(self):
        """End-to-end: subsample init -> pack -> insert remaining -> validate."""
        key = jax.random.key(21)
        k1, k2, k3 = jax.random.split(key, 3)
        data = jax.random.normal(k1, (50, 3))
        col_types = [ColumnType.CONTINUOUS] * 3

        # Init on subsample
        result = initialize(k2, data, col_types, subsample_rows=15)
        sub_data = data[result.subsample_idx]
        packed = pack_state(result.state)

        # Run a few sweeps on subsample
        packed = packed_gibbs_sweep(k3, packed, sub_data, n_sweeps=3)

        # Insert remaining rows
        remaining_mask = jnp.ones(50, dtype=bool).at[result.subsample_idx].set(False)
        remaining_idx = jnp.where(remaining_mask, size=35)[0]
        remaining = data[remaining_idx]

        packed, full_data = packed_insert_rows(
            jax.random.fold_in(k3, 1), packed, sub_data, remaining
        )

        assert packed.n_rows == 50
        assert full_data.shape[0] == 50


# ---------------------------------------------------------------------------
# suggest_max_clusters tests
# ---------------------------------------------------------------------------


class TestSuggestMaxClusters:
    """Boundary value tests for suggest_max_clusters."""

    @pytest.mark.parametrize(
        "n_rows,expected",
        [
            (0, 4),  # sqrt(0)=0, clamped to 4
            (1, 4),  # sqrt(1)=1, clamped to 4
            (4, 4),  # sqrt(4)=2, clamped to 4
            (16, 4),  # sqrt(16)=4, exactly 4
            (25, 5),  # sqrt(25)=5
            (100, 10),  # sqrt(100)=10
            (1024, 32),  # sqrt(1024)=32, exactly 32
            (10000, 32),  # sqrt(10000)=100, capped to 32
            (1000000, 32),  # sqrt(1M)=1000, capped to 32
        ],
    )
    def test_boundary_values(self, n_rows, expected):
        assert suggest_max_clusters(n_rows) == expected

    def test_always_at_least_4(self):
        for n in range(20):
            assert suggest_max_clusters(n) >= 4

    def test_never_exceeds_32(self):
        for n in [100, 1000, 10000, 100000, 1000000]:
            assert suggest_max_clusters(n) <= 32

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="n_rows must be >= 0"):
            suggest_max_clusters(-1)

    def test_negative_large_raises(self):
        with pytest.raises(ValueError, match="n_rows must be >= 0"):
            suggest_max_clusters(-100)
