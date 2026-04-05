"""Tests for Phase 2+3 scaling features.

Covers:
- packed_transition_row_assignments_minibatch validity
- Bincount _score_column_in_view regression (via column assignment)
- save_npy / load_npy_mmap roundtrip
- read_csv_chunked parity + warning behavior
- subsample_anneal small-scale e2e
- minibatch_gibbs_sweep multi-sweep validity
- gibbs_sweep_early_stopping convergence + edge cases
"""

from __future__ import annotations

import csv
import tempfile
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crosscat.model import initialize
from crosscat.packed import (
    pack_state,
    packed_gibbs_sweep,
    packed_log_joint,
    packed_transition_column_assignments,
    packed_transition_row_assignments_minibatch,
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
def mixed_packed():
    """Mixed-type packed state for broader coverage."""
    key = jax.random.key(55)
    from crosscat.synthetic import generate_crosscat_data

    col_types = [
        ColumnType.CONTINUOUS,
        ColumnType.CATEGORICAL,
        ColumnType.BINARY,
    ]
    result = generate_crosscat_data(key, 50, col_types, n_views=1, n_clusters=2)
    state = initialize(jax.random.key(56), result["data"], col_types).state
    packed = pack_state(state)
    return packed, result["data"], col_types


# ---------------------------------------------------------------------------
# 1. packed_transition_row_assignments_minibatch
# ---------------------------------------------------------------------------


class TestMinibatchRowAssignments:
    def test_returns_valid_state(self, simple_packed):
        """Minibatch kernel returns a state with finite log-joint."""
        packed, data, _ = simple_packed
        key = jax.random.key(100)
        updated = packed_transition_row_assignments_minibatch(key, packed, data, batch_size=10)
        lj = float(packed_log_joint(updated, data))
        assert np.isfinite(lj), f"Log-joint is {lj} after minibatch sweep"

    def test_preserves_n_rows(self, simple_packed):
        """Minibatch kernel does not change the number of rows."""
        packed, data, _ = simple_packed
        key = jax.random.key(101)
        updated = packed_transition_row_assignments_minibatch(key, packed, data, batch_size=10)
        assert updated.n_rows == packed.n_rows

    def test_assignments_change(self, simple_packed):
        """At least some row assignments should change after minibatch sweep."""
        packed, data, _ = simple_packed
        # Run several sweeps to increase chance of changes
        key = jax.random.key(102)
        updated = packed
        for i in range(5):
            k = jax.random.fold_in(key, i)
            updated = packed_transition_row_assignments_minibatch(k, updated, data, batch_size=20)
        # Check at least one view has changed assignments
        orig_ra = packed.view_row_assignments
        new_ra = updated.view_row_assignments
        assert not jnp.array_equal(orig_ra, new_ra), (
            "Row assignments unchanged after 5 minibatch sweeps"
        )

    def test_batch_size_ge_nrows_equivalent_to_full(self, simple_packed):
        """batch_size >= n_rows should behave like a full sweep."""
        packed, data, _ = simple_packed
        key = jax.random.key(103)
        updated = packed_transition_row_assignments_minibatch(key, packed, data, batch_size=1000)
        lj = float(packed_log_joint(updated, data))
        assert np.isfinite(lj)

    def test_mixed_types(self, mixed_packed):
        """Minibatch kernel works with mixed column types."""
        packed, data, _ = mixed_packed
        key = jax.random.key(104)
        updated = packed_transition_row_assignments_minibatch(key, packed, data, batch_size=15)
        lj = float(packed_log_joint(updated, data))
        assert np.isfinite(lj)


# ---------------------------------------------------------------------------
# 2. Bincount _score_column_in_view regression
# ---------------------------------------------------------------------------


class TestBincountColumnScoring:
    def test_column_assignment_produces_finite_log_joint(self, simple_packed):
        """Column assignment transition (which uses bincount scoring) yields finite log-joint."""
        packed, data, _ = simple_packed
        key = jax.random.key(200)
        updated = packed_transition_column_assignments(key, packed, data)
        lj = float(packed_log_joint(updated, data))
        assert np.isfinite(lj), f"Log-joint is {lj} after column assignment"

    def test_column_assignment_preserves_column_count(self, simple_packed):
        """Column assignment does not lose or duplicate columns."""
        packed, data, _ = simple_packed
        key = jax.random.key(201)
        updated = packed_transition_column_assignments(key, packed, data)
        # Total columns assigned across views should equal n_cols
        total = int(jnp.sum(updated.view_n_columns))
        assert total == packed.n_cols

    def test_column_assignment_with_nan(self):
        """Column scoring handles NaN data correctly via bincount path."""
        key = jax.random.key(202)
        data = jax.random.normal(key, (30, 3))
        # Inject NaN
        data = data.at[0, 0].set(float("nan"))
        data = data.at[5, 1].set(float("nan"))
        data = data.at[10, 2].set(float("nan"))
        col_types = [ColumnType.CONTINUOUS] * 3
        state = initialize(jax.random.key(203), data, col_types).state
        packed = pack_state(state)
        updated = packed_transition_column_assignments(jax.random.key(204), packed, data)
        lj = float(packed_log_joint(updated, data))
        assert np.isfinite(lj)


# ---------------------------------------------------------------------------
# 3. save_npy / load_npy_mmap roundtrip
# ---------------------------------------------------------------------------


class TestNpyRoundtrip:
    def test_basic_roundtrip(self):
        """save_npy -> load_npy_mmap preserves data and column names."""
        from crosscat.data_utils import load_npy_mmap, save_npy

        data = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        col_names = ["a", "b"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            save_npy(path, data, col_names)
            loaded, loaded_names = load_npy_mmap(path)
            assert isinstance(loaded, np.ndarray), "Should return numpy array, not JAX"
            np.testing.assert_allclose(loaded, np.array(data), atol=1e-6)
            assert loaded_names == col_names

    def test_nan_preservation(self):
        """NaN values survive the roundtrip."""
        from crosscat.data_utils import load_npy_mmap, save_npy

        data = jnp.array([[1.0, float("nan")], [float("nan"), 4.0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nan.npy"
            save_npy(path, data)
            loaded, _ = load_npy_mmap(path)
            assert np.isnan(loaded[0, 1])
            assert np.isnan(loaded[1, 0])
            assert loaded[0, 0] == pytest.approx(1.0)
            assert loaded[1, 1] == pytest.approx(4.0)

    def test_missing_sidecar_warns(self):
        """load_npy_mmap warns when JSON sidecar is missing."""
        from crosscat.data_utils import load_npy_mmap, save_npy

        data = jnp.array([[1.0, 2.0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nosidecar.npy"
            save_npy(path, data, column_names=["a", "b"])
            # Delete the sidecar that was created
            sidecar = path.with_suffix(".json")
            assert sidecar.exists(), "Sidecar should have been created"
            sidecar.unlink()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                loaded, names = load_npy_mmap(path)
                assert names is None
                assert any("sidecar" in str(x.message).lower() for x in w)

    def test_returns_numpy_memmap(self):
        """load_npy_mmap returns a numpy array (memmap), not a JAX array."""
        from crosscat.data_utils import load_npy_mmap, save_npy

        data = jnp.ones((10, 3))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memmap.npy"
            save_npy(path, data)
            loaded, _ = load_npy_mmap(path)
            assert isinstance(loaded, np.ndarray)
            assert not hasattr(loaded, "devices")  # JAX arrays have .devices()


# ---------------------------------------------------------------------------
# 4. read_csv_chunked parity + warnings
# ---------------------------------------------------------------------------


class TestReadCsvChunked:
    def _write_csv(self, path: Path, rows: list[list[str]], header: list[str] | None = None):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(rows)

    def test_basic_numeric(self):
        """Reads a simple numeric CSV correctly."""
        from crosscat.data_utils import read_csv_chunked

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "basic.csv"
            self._write_csv(
                path,
                [["1.0", "2.0"], ["3.0", "4.0"], ["5.0", "6.0"]],
                header=["x", "y"],
            )
            data, names = read_csv_chunked(path, chunk_size=2)
            assert names == ["x", "y"]
            assert data.shape == (3, 2)
            assert float(data[0, 0]) == pytest.approx(1.0)
            assert float(data[2, 1]) == pytest.approx(6.0)

    def test_nan_values_parsed(self):
        """NaN sentinel values are correctly parsed as NaN."""
        from crosscat.data_utils import read_csv_chunked

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nans.csv"
            self._write_csv(
                path,
                [["1.0", "NA"], ["", "3.0"], ["NaN", "NULL"]],
                header=["a", "b"],
            )
            data, _ = read_csv_chunked(path)
            assert np.isnan(float(data[0, 1]))  # NA
            assert np.isnan(float(data[1, 0]))  # empty
            assert np.isnan(float(data[2, 0]))  # NaN
            assert np.isnan(float(data[2, 1]))  # NULL

    def test_unparseable_values_warn(self):
        """Non-numeric values trigger a warning."""
        from crosscat.data_utils import read_csv_chunked

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.csv"
            self._write_csv(
                path,
                [["1.0", "hello"], ["world", "3.0"]],
                header=["a", "b"],
            )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                data, _ = read_csv_chunked(path)
                assert any("Could not parse" in str(x.message) for x in w)
                # Unparseable values should be NaN
                assert np.isnan(float(data[0, 1]))
                assert np.isnan(float(data[1, 0]))

    def test_mismatched_row_length_warns(self):
        """Rows with wrong column count trigger a warning."""
        from crosscat.data_utils import read_csv_chunked

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mismatch.csv"
            self._write_csv(
                path,
                [["1.0", "2.0"], ["3.0"], ["4.0", "5.0", "6.0"]],
                header=["a", "b"],
            )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                data, _ = read_csv_chunked(path)
                assert any("mismatched column count" in str(x.message) for x in w)
                assert data.shape[1] == 2
                # Short row: second col should be NaN
                assert np.isnan(float(data[1, 1]))

    def test_chunking_boundary(self):
        """Data split across chunks produces same result as single chunk."""
        from crosscat.data_utils import read_csv_chunked

        rows = [[str(float(i)), str(float(i * 2))] for i in range(25)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chunks.csv"
            self._write_csv(path, rows, header=["x", "y"])
            data_small, _ = read_csv_chunked(path, chunk_size=3)
            data_big, _ = read_csv_chunked(path, chunk_size=10000)
            np.testing.assert_allclose(np.array(data_small), np.array(data_big), atol=1e-6)


# ---------------------------------------------------------------------------
# 5. subsample_anneal small-scale e2e
# ---------------------------------------------------------------------------


class TestSubsampleAnneal:
    def test_all_rows_included(self):
        """After annealing, the packed state should contain all rows."""
        from crosscat.scaling import subsample_anneal

        key = jax.random.key(300)
        data = jax.random.normal(key, (40, 3))
        col_types = [ColumnType.CONTINUOUS] * 3

        packed, reordered = subsample_anneal(
            jax.random.key(301),
            data,
            col_types,
            initial_size=10,
            sweeps_per_stage=2,
        )
        assert packed.n_rows == 40
        assert reordered.shape[0] == 40

    def test_valid_final_state(self):
        """Final state has a finite log-joint."""
        from crosscat.scaling import subsample_anneal

        key = jax.random.key(302)
        data = jax.random.normal(key, (30, 2))
        col_types = [ColumnType.CONTINUOUS] * 2

        packed, reordered = subsample_anneal(
            jax.random.key(303),
            data,
            col_types,
            initial_size=10,
            sweeps_per_stage=2,
        )
        lj = float(packed_log_joint(packed, reordered))
        assert np.isfinite(lj)


# ---------------------------------------------------------------------------
# 6. minibatch_gibbs_sweep multi-sweep
# ---------------------------------------------------------------------------


class TestMinibatchGibbsSweep:
    def test_multi_sweep_valid(self, simple_packed):
        """Multiple mini-batch sweeps produce a valid state."""
        from crosscat.scaling import minibatch_gibbs_sweep

        packed, data, _ = simple_packed
        updated = minibatch_gibbs_sweep(
            jax.random.key(400), packed, data, batch_size=10, n_sweeps=3
        )
        lj = float(packed_log_joint(updated, data))
        assert np.isfinite(lj)

    def test_state_changes_over_sweeps(self, simple_packed):
        """State should differ after multiple sweeps."""
        from crosscat.scaling import minibatch_gibbs_sweep

        packed, data, _ = simple_packed
        updated = minibatch_gibbs_sweep(
            jax.random.key(401), packed, data, batch_size=20, n_sweeps=5
        )
        # At least assignments or suffstats should differ
        assert not jnp.array_equal(
            packed.view_row_assignments, updated.view_row_assignments
        ) or not jnp.array_equal(packed.ss_counts, updated.ss_counts)


# ---------------------------------------------------------------------------
# 7. gibbs_sweep_early_stopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_returns_log_joint_history(self, simple_packed):
        """Returns a non-empty log-joint history list."""
        from crosscat.scaling import gibbs_sweep_early_stopping

        packed, data, _ = simple_packed
        _, history = gibbs_sweep_early_stopping(
            jax.random.key(500),
            packed,
            data,
            max_sweeps=20,
            check_interval=5,
        )
        assert len(history) > 0
        assert all(np.isfinite(lj) for lj in history)

    def test_stops_before_max(self, simple_packed):
        """With tight patience, should stop before max_sweeps on a converged state."""
        from crosscat.scaling import gibbs_sweep_early_stopping

        packed, data, _ = simple_packed
        # Pre-converge
        packed = packed_gibbs_sweep(jax.random.key(501), packed, data, n_sweeps=20)
        _, history = gibbs_sweep_early_stopping(
            jax.random.key(502),
            packed,
            data,
            max_sweeps=100,
            check_interval=5,
            patience=2,
            min_improvement=0.1,  # high threshold → stops quickly
        )
        # Should stop before using all 100/5 = 20 checks
        assert len(history) < 20

    def test_minibatch_mode(self, simple_packed):
        """Works with batch_size parameter for mini-batch mode."""
        from crosscat.scaling import gibbs_sweep_early_stopping

        packed, data, _ = simple_packed
        result, history = gibbs_sweep_early_stopping(
            jax.random.key(503),
            packed,
            data,
            max_sweeps=10,
            check_interval=5,
            batch_size=10,
        )
        assert len(history) > 0
        lj = float(packed_log_joint(result, data))
        assert np.isfinite(lj)


# ---------------------------------------------------------------------------
# 8. Additional must-have tests
# ---------------------------------------------------------------------------


class TestMinibatchAssignmentInvariants:
    """Validate structural invariants after minibatch kernel, not just log-joint."""

    def test_assignments_in_range(self, simple_packed):
        """All row assignments must be in [0, max_clusters)."""
        packed, data, _ = simple_packed
        key = jax.random.key(600)
        updated = packed_transition_row_assignments_minibatch(key, packed, data, batch_size=15)
        ra = updated.view_row_assignments
        n_rows = updated.n_rows
        max_k = updated.max_clusters
        for v in range(int(updated.n_views)):
            assigns = ra[v, :n_rows]
            assert jnp.all(assigns >= 0), f"View {v}: negative assignment"
            assert jnp.all(assigns < max_k), f"View {v}: assignment >= max_clusters"

    def test_validate_state_after_minibatch(self, simple_packed):
        """validate_state should pass on minibatch output."""
        from crosscat.packed import unpack_state
        from crosscat.validate import validate_state

        packed, data, col_types = simple_packed
        key = jax.random.key(601)
        updated = packed_transition_row_assignments_minibatch(key, packed, data, batch_size=15)
        state = unpack_state(updated, col_types, data=data)
        errors = validate_state(state, data)
        assert not errors, f"Validation errors: {errors}"


class TestEarlyStoppingEdgeCases:
    def test_nan_log_joint_stops_with_warning(self):
        """If log-joint is NaN, early stopping should break with a warning."""
        from unittest.mock import patch

        from crosscat.scaling import gibbs_sweep_early_stopping

        key = jax.random.key(700)
        data = jax.random.normal(key, (20, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        state = initialize(jax.random.key(701), data, col_types).state
        packed = pack_state(state)

        # Patch packed_log_joint to return NaN after first call
        real_plj = packed_log_joint
        call_count = [0]

        def fake_log_joint(p, d):
            call_count[0] += 1
            if call_count[0] == 1:
                return real_plj(p, d)
            return jnp.array(float("nan"))

        with (
            patch("crosscat.scaling.packed_log_joint", side_effect=fake_log_joint),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            _, history = gibbs_sweep_early_stopping(
                jax.random.key(702),
                packed,
                data,
                max_sweeps=50,
                check_interval=5,
            )
            # Should have stopped after 2 checks (first OK, second NaN)
            assert len(history) == 2
            assert np.isnan(history[-1])
            assert any("NaN" in str(x.message) or "nan" in str(x.message) for x in w)


class TestReadCsvEdgeCases:
    def test_empty_csv_raises(self):
        """Empty CSV file raises ValueError."""
        from crosscat.data_utils import read_csv_chunked

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.csv"
            path.write_text("")
            with pytest.raises(ValueError, match="empty"):
                read_csv_chunked(path)

    def test_empty_csv_read_csv_raises(self):
        """Empty CSV file raises ValueError in read_csv too."""
        from crosscat.data_utils import read_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.csv"
            path.write_text("")
            with pytest.raises(ValueError, match="empty"):
                read_csv(path)

    def test_has_header_false(self):
        """read_csv_chunked with has_header=False generates column names."""
        from crosscat.data_utils import read_csv_chunked

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "noheader.csv"
            path.write_text("1.0,2.0\n3.0,4.0\n")
            data, names = read_csv_chunked(path, has_header=False)
            assert names == ["col_0", "col_1"]
            assert data.shape == (2, 2)
            assert float(data[0, 0]) == pytest.approx(1.0)

    def test_read_csv_warns_on_unparseable(self):
        """read_csv (not chunked) now also warns on unparseable values."""
        from crosscat.data_utils import read_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.csv"
            path.write_text("a,b\n1.0,hello\nworld,3.0\n")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                data, _ = read_csv(path)
                assert any("Could not parse" in str(x.message) for x in w)
                assert np.isnan(float(data[0, 1]))


class TestNpyMmap:
    """Tests for the corrected save_npy/load_npy_mmap using .npy format."""

    def test_saves_as_npy(self):
        """save_npy creates a .npy file (not .npz)."""
        from crosscat.data_utils import save_npy

        data = jnp.array([[1.0, 2.0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                save_npy(Path(tmpdir) / "test.npz", data)
            # Should create .npy regardless of input extension
            assert (Path(tmpdir) / "test.npy").exists()

    def test_true_memmap(self):
        """load_npy_mmap returns a true numpy memmap on .npy files."""
        from crosscat.data_utils import load_npy_mmap, save_npy

        data = jnp.ones((100, 10))
        with tempfile.TemporaryDirectory() as tmpdir:
            save_npy(Path(tmpdir) / "test", data)
            loaded, _ = load_npy_mmap(Path(tmpdir) / "test")
            assert isinstance(loaded, np.memmap), f"Expected np.memmap, got {type(loaded)}"
