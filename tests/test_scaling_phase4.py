"""Tests for Phase 4 scaling features.

Covers:
- packed_transition_row_assignments_parallel validity and correctness
- save_arrow / load_arrow roundtrip
- parallel_gibbs_sweep composition
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crosscat.model import initialize
from crosscat.packed import (
    pack_state,
    packed_log_joint,
    packed_transition_row_assignments_parallel,
    unpack_state,
)
from crosscat.types import ColumnType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_packed():
    """Small packed state + data for fast tests."""
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key)
    data = jax.random.normal(k1, (40, 4))
    col_types = [ColumnType.CONTINUOUS] * 4
    state = initialize(k2, data, col_types).state
    packed = pack_state(state)
    return packed, data, col_types


@pytest.fixture
def multi_cluster_packed():
    """State initialized with multiple clusters for cluster-count tests."""
    from crosscat.synthetic import generate_crosscat_data

    key = jax.random.key(88)
    col_types = [ColumnType.CONTINUOUS] * 3
    result = generate_crosscat_data(key, 60, col_types, n_views=1, n_clusters=3)
    state = initialize(jax.random.key(89), result["data"], col_types).state
    packed = pack_state(state)
    return packed, result["data"], col_types


# ---------------------------------------------------------------------------
# 1. Parallel row assignment kernel
# ---------------------------------------------------------------------------


class TestParallelRowAssignments:
    def test_returns_finite_log_joint(self, simple_packed):
        """Parallel kernel returns a state with finite log-joint."""
        packed, data, _ = simple_packed
        key = jax.random.key(100)
        updated = packed_transition_row_assignments_parallel(key, packed, data)
        lj = float(packed_log_joint(updated, data))
        assert np.isfinite(lj), f"Log-joint is {lj}"

    def test_preserves_n_rows(self, simple_packed):
        """Parallel kernel does not change the number of rows."""
        packed, data, _ = simple_packed
        updated = packed_transition_row_assignments_parallel(jax.random.key(101), packed, data)
        assert updated.n_rows == packed.n_rows

    def test_assignments_in_range(self, simple_packed):
        """All row assignments must be in [0, max_clusters)."""
        packed, data, _ = simple_packed
        updated = packed_transition_row_assignments_parallel(jax.random.key(102), packed, data)
        ra = updated.view_row_assignments
        n_rows = updated.n_rows
        max_k = updated.max_clusters
        for v in range(int(updated.n_views)):
            assigns = ra[v, :n_rows]
            assert jnp.all(assigns >= 0), f"View {v}: negative assignment"
            assert jnp.all(assigns < max_k), f"View {v}: assignment >= max_clusters"

    def test_validate_state_after_parallel(self, simple_packed):
        """validate_state passes on parallel kernel output."""
        from crosscat.validate import validate_state

        packed, data, col_types = simple_packed
        updated = packed_transition_row_assignments_parallel(jax.random.key(103), packed, data)
        state = unpack_state(updated, col_types, data=data)
        errors = validate_state(state, data)
        assert not errors, f"Validation errors: {errors}"

    def test_does_not_collapse_to_single_cluster(self, multi_cluster_packed):
        """Multiple clusters should survive after several parallel sweeps."""
        packed, data, _ = multi_cluster_packed
        key = jax.random.key(104)
        updated = packed
        for i in range(5):
            k = jax.random.fold_in(key, i)
            updated = packed_transition_row_assignments_parallel(k, updated, data)

        # At least one view should have > 1 cluster
        has_multi = False
        for v in range(int(updated.n_views)):
            if int(updated.view_n_clusters[v]) > 1:
                has_multi = True
                break
        assert has_multi, (
            f"All views collapsed to 1 cluster after 5 parallel sweeps. "
            f"n_clusters: {[int(updated.view_n_clusters[v]) for v in range(int(updated.n_views))]}"
        )

    def test_assignments_change(self, simple_packed):
        """At least some row assignments should change after parallel sweep."""
        packed, data, _ = simple_packed
        key = jax.random.key(105)
        updated = packed
        for i in range(5):
            k = jax.random.fold_in(key, i)
            updated = packed_transition_row_assignments_parallel(k, updated, data)
        orig_ra = packed.view_row_assignments
        new_ra = updated.view_row_assignments
        assert not jnp.array_equal(orig_ra, new_ra)


# ---------------------------------------------------------------------------
# 2. Arrow IPC roundtrip
# ---------------------------------------------------------------------------


class TestArrowRoundtrip:
    def test_basic_roundtrip(self):
        """save_arrow -> load_arrow preserves data and column names."""
        pytest.importorskip("pyarrow")
        from crosscat.data_utils import load_arrow, save_arrow

        data = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        col_names = ["a", "b"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.arrow"
            save_arrow(path, data, col_names)
            loaded, loaded_names = load_arrow(path)
            assert loaded_names == col_names
            np.testing.assert_allclose(np.array(loaded), np.array(data), atol=1e-6)

    def test_nan_preservation(self):
        """NaN values survive Arrow roundtrip."""
        pytest.importorskip("pyarrow")
        from crosscat.data_utils import load_arrow, save_arrow

        data = jnp.array([[1.0, float("nan")], [float("nan"), 4.0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nan.arrow"
            save_arrow(path, data, ["a", "b"])
            loaded, _ = load_arrow(path)
            assert np.isnan(float(loaded[0, 1]))
            assert np.isnan(float(loaded[1, 0]))
            assert float(loaded[0, 0]) == pytest.approx(1.0)

    def test_default_column_names(self):
        """save_arrow without column names generates defaults."""
        pytest.importorskip("pyarrow")
        from crosscat.data_utils import load_arrow, save_arrow

        data = jnp.ones((3, 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonames.arrow"
            save_arrow(path, data)
            _, names = load_arrow(path)
            assert names == ["col_0", "col_1"]

    def test_column_subset(self):
        """load_arrow with columns parameter loads subset."""
        pytest.importorskip("pyarrow")
        from crosscat.data_utils import load_arrow, save_arrow

        data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subset.arrow"
            save_arrow(path, data, ["a", "b", "c"])
            loaded, names = load_arrow(path, columns=["a", "c"])
            assert names == ["a", "c"]
            assert loaded.shape == (2, 2)


# ---------------------------------------------------------------------------
# 3. parallel_gibbs_sweep composition
# ---------------------------------------------------------------------------


class TestParallelGibbsSweep:
    def test_single_sweep_valid(self, simple_packed):
        """Single parallel Gibbs sweep produces valid state."""
        from crosscat.scaling import parallel_gibbs_sweep

        packed, data, _ = simple_packed
        updated = parallel_gibbs_sweep(jax.random.key(300), packed, data, n_sweeps=1)
        lj = float(packed_log_joint(updated, data))
        assert np.isfinite(lj)

    def test_multi_sweep_valid(self, simple_packed):
        """Multiple parallel sweeps produce valid state."""
        from crosscat.scaling import parallel_gibbs_sweep

        packed, data, _ = simple_packed
        updated = parallel_gibbs_sweep(jax.random.key(301), packed, data, n_sweeps=3)
        lj = float(packed_log_joint(updated, data))
        assert np.isfinite(lj)

    def test_state_changes(self, simple_packed):
        """State should differ after parallel sweeps."""
        from crosscat.scaling import parallel_gibbs_sweep

        packed, data, _ = simple_packed
        updated = parallel_gibbs_sweep(jax.random.key(302), packed, data, n_sweeps=5)
        assert not jnp.array_equal(
            packed.view_row_assignments, updated.view_row_assignments
        ) or not jnp.array_equal(packed.ss_counts, updated.ss_counts)
