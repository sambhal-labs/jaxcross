"""Tests from PR #86/#87 review followup.

Covers:
- write_parquet / read_parquet roundtrip
- save_npy / load_npy_mmap (new preferred names)
- save_npz / load_npz_mmap deprecation warnings
- subsample_anneal growth stages and edge cases
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crosscat.types import ColumnType

# ---------------------------------------------------------------------------
# 1. Parquet I/O roundtrip
# ---------------------------------------------------------------------------


class TestParquetRoundtrip:
    def test_write_read_roundtrip(self):
        """write_parquet -> read_parquet preserves data and column names."""
        pytest.importorskip("pyarrow")
        from crosscat.data_utils import read_parquet, write_parquet

        data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        col_names = ["x", "y", "z"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            write_parquet(path, data, col_names)
            loaded, loaded_names = read_parquet(path)
            assert loaded_names == col_names
            assert loaded.shape == (3, 3)
            np.testing.assert_allclose(np.array(loaded), np.array(data), atol=1e-6)

    def test_nan_preservation(self):
        """NaN values survive Parquet roundtrip."""
        pytest.importorskip("pyarrow")
        from crosscat.data_utils import read_parquet, write_parquet

        data = jnp.array([[1.0, float("nan")], [float("nan"), 4.0]])
        col_names = ["a", "b"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nan.parquet"
            write_parquet(path, data, col_names)
            loaded, _ = read_parquet(path)
            assert np.isnan(float(loaded[0, 1]))
            assert np.isnan(float(loaded[1, 0]))
            assert float(loaded[0, 0]) == pytest.approx(1.0)
            assert float(loaded[1, 1]) == pytest.approx(4.0)

    def test_column_subset(self):
        """read_parquet with columns parameter loads subset."""
        pytest.importorskip("pyarrow")
        from crosscat.data_utils import read_parquet, write_parquet

        data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        col_names = ["a", "b", "c"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subset.parquet"
            write_parquet(path, data, col_names)
            loaded, names = read_parquet(path, columns=["a", "c"])
            assert names == ["a", "c"]
            assert loaded.shape == (2, 2)
            np.testing.assert_allclose(
                np.array(loaded), np.array([[1.0, 3.0], [4.0, 6.0]]), atol=1e-6
            )

    def test_single_column(self):
        """Parquet roundtrip works with a single column."""
        pytest.importorskip("pyarrow")
        from crosscat.data_utils import read_parquet, write_parquet

        data = jnp.array([[1.0], [2.0], [3.0]])
        col_names = ["only"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "single.parquet"
            write_parquet(path, data, col_names)
            loaded, names = read_parquet(path)
            assert names == ["only"]
            assert loaded.shape == (3, 1)

    def test_missing_pyarrow_raises(self):
        """read_parquet raises ImportError with clear message when pyarrow is missing."""
        # This test only validates the error path exists — skip if pyarrow is installed
        try:
            import pyarrow  # noqa: F401

            pytest.skip("pyarrow is installed, cannot test import error path")
        except ImportError:
            from crosscat.data_utils import read_parquet

            with pytest.raises(ImportError, match="pyarrow"):
                read_parquet("dummy.parquet")


# ---------------------------------------------------------------------------
# 2. save_npy / load_npy_mmap (preferred names)
# ---------------------------------------------------------------------------


class TestNpyPreferredNames:
    def test_save_load_roundtrip(self):
        """save_npy -> load_npy_mmap preserves data and column names."""
        from crosscat.data_utils import load_npy_mmap, save_npy

        data = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        col_names = ["a", "b"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            save_npy(path, data, col_names)
            loaded, loaded_names = load_npy_mmap(path)
            assert isinstance(loaded, np.ndarray)
            np.testing.assert_allclose(loaded, np.array(data), atol=1e-6)
            assert loaded_names == col_names

    def test_returns_memmap(self):
        """load_npy_mmap returns a true numpy memmap."""
        from crosscat.data_utils import load_npy_mmap, save_npy

        data = jnp.ones((100, 10))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memmap.npy"
            save_npy(path, data)
            loaded, _ = load_npy_mmap(path)
            assert isinstance(loaded, np.memmap), f"Expected np.memmap, got {type(loaded)}"

    def test_nan_preservation(self):
        """NaN values survive npy roundtrip."""
        from crosscat.data_utils import load_npy_mmap, save_npy

        data = jnp.array([[1.0, float("nan")], [float("nan"), 4.0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nan.npy"
            save_npy(path, data)
            loaded, _ = load_npy_mmap(path)
            assert np.isnan(loaded[0, 1])
            assert np.isnan(loaded[1, 0])


# ---------------------------------------------------------------------------
# 3. Deprecation warnings on old names
# ---------------------------------------------------------------------------


class TestDeprecationWarnings:
    def test_save_npz_warns(self):
        """save_npz emits DeprecationWarning."""
        from crosscat.data_utils import save_npz

        data = jnp.array([[1.0, 2.0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                save_npz(path, data)
                deprecation_msgs = [x for x in w if issubclass(x.category, DeprecationWarning)]
                assert len(deprecation_msgs) >= 1
                assert "save_npy" in str(deprecation_msgs[0].message)

    def test_load_npz_mmap_warns(self):
        """load_npz_mmap emits DeprecationWarning."""
        from crosscat.data_utils import load_npz_mmap, save_npy

        data = jnp.array([[1.0, 2.0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            save_npy(path, data)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                loaded, _ = load_npz_mmap(path)
                deprecation_msgs = [x for x in w if issubclass(x.category, DeprecationWarning)]
                assert len(deprecation_msgs) >= 1
                assert "load_npy_mmap" in str(deprecation_msgs[0].message)

    def test_deprecated_functions_still_work(self):
        """Deprecated save_npz/load_npz_mmap still produce correct results."""
        from crosscat.data_utils import load_npz_mmap, save_npz

        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        col_names = ["x", "y"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "compat.npy"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                save_npz(path, data, col_names)
                loaded, loaded_names = load_npz_mmap(path)
            np.testing.assert_allclose(loaded, np.array(data), atol=1e-6)
            assert loaded_names == col_names


# ---------------------------------------------------------------------------
# 4. subsample_anneal — growth stages and edge cases
# ---------------------------------------------------------------------------


class TestSubsampleAnnealExtended:
    def test_growth_factor_both_reach_full_size(self):
        """Both slow and fast growth factors produce correct final state with all rows."""
        from crosscat.scaling import subsample_anneal

        key = jax.random.key(400)
        data = jax.random.normal(key, (80, 3))
        col_types = [ColumnType.CONTINUOUS] * 3

        packed_slow, data_slow = subsample_anneal(
            jax.random.key(401),
            data,
            col_types,
            initial_size=10,
            growth_factor=2.0,
            sweeps_per_stage=1,
        )
        packed_fast, data_fast = subsample_anneal(
            jax.random.key(402),
            data,
            col_types,
            initial_size=10,
            growth_factor=8.0,
            sweeps_per_stage=1,
        )
        assert packed_slow.n_rows == 80
        assert packed_fast.n_rows == 80
        assert data_slow.shape[0] == 80
        assert data_fast.shape[0] == 80

    def test_initial_size_ge_nrows(self):
        """When initial_size >= n_rows, all rows are included immediately."""
        from crosscat.scaling import subsample_anneal

        key = jax.random.key(403)
        data = jax.random.normal(key, (20, 2))
        col_types = [ColumnType.CONTINUOUS] * 2

        packed, reordered = subsample_anneal(
            jax.random.key(404),
            data,
            col_types,
            initial_size=100,
            sweeps_per_stage=1,
        )
        assert packed.n_rows == 20
        assert reordered.shape[0] == 20

    def test_mixed_types(self):
        """subsample_anneal works with mixed column types."""
        from crosscat.scaling import subsample_anneal
        from crosscat.synthetic import generate_crosscat_data

        key = jax.random.key(405)
        col_types = [
            ColumnType.CONTINUOUS,
            ColumnType.CATEGORICAL,
            ColumnType.BINARY,
        ]
        result = generate_crosscat_data(key, 60, col_types, n_views=1, n_clusters=2)

        packed, reordered = subsample_anneal(
            jax.random.key(406),
            result["data"],
            col_types,
            initial_size=15,
            sweeps_per_stage=2,
        )
        assert packed.n_rows == 60
        assert reordered.shape[0] == 60

    def test_finite_log_joint_after_anneal(self):
        """Final state after annealing has a finite log-joint."""
        from crosscat.packed import packed_log_joint
        from crosscat.scaling import subsample_anneal

        key = jax.random.key(407)
        data = jax.random.normal(key, (50, 4))
        col_types = [ColumnType.CONTINUOUS] * 4

        packed, reordered = subsample_anneal(
            jax.random.key(408),
            data,
            col_types,
            initial_size=10,
            sweeps_per_stage=3,
        )
        lj = float(packed_log_joint(packed, reordered))
        assert np.isfinite(lj), f"Log-joint is {lj}"

    def test_small_insert_batch_size(self):
        """Works with insert_batch_size smaller than rows to insert per stage."""
        from crosscat.scaling import subsample_anneal

        key = jax.random.key(409)
        data = jax.random.normal(key, (50, 2))
        col_types = [ColumnType.CONTINUOUS] * 2

        packed, reordered = subsample_anneal(
            jax.random.key(410),
            data,
            col_types,
            initial_size=10,
            growth_factor=2.0,
            sweeps_per_stage=1,
            insert_batch_size=3,
        )
        assert packed.n_rows == 50
        assert reordered.shape[0] == 50
