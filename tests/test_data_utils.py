"""Tests for data_utils module: CSV I/O, column type guessing, metadata, discretization."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from crosscat.data_utils import (
    discretize_column,
    gen_column_metadata,
    guess_column_type,
    guess_column_types,
    read_csv,
    write_csv,
)
from crosscat.types import ColumnType

# ---------------------------------------------------------------------------
# guess_column_type
# ---------------------------------------------------------------------------


def test_guess_binary_01():
    """Columns with only 0s and 1s should be detected as BINARY."""
    col = jnp.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    assert guess_column_type(col) == ColumnType.BINARY


def test_guess_binary_single_value():
    """A column with only 0s is still BINARY."""
    col = jnp.array([0.0, 0.0, 0.0])
    assert guess_column_type(col) == ColumnType.BINARY


def test_guess_categorical_few_integers():
    """Integer column with few unique values should be CATEGORICAL."""
    col = jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0])
    assert guess_column_type(col) == ColumnType.CATEGORICAL


def test_guess_continuous_float():
    """A column with many distinct float values should be CONTINUOUS."""
    key = jax.random.key(0)
    col = jax.random.normal(key, (100,))
    assert guess_column_type(col) == ColumnType.CONTINUOUS


def test_guess_continuous_many_integers():
    """An integer column with many unique values should be CONTINUOUS."""
    col = jnp.arange(100, dtype=jnp.float32)
    assert guess_column_type(col) == ColumnType.CONTINUOUS


def test_guess_all_nan_returns_continuous():
    """An all-NaN column should default to CONTINUOUS."""
    col = jnp.array([float("nan"), float("nan"), float("nan")])
    assert guess_column_type(col) == ColumnType.CONTINUOUS


def test_guess_with_nan_mixed():
    """NaN values should be filtered before guessing type."""
    col = jnp.array([0.0, 1.0, float("nan"), 0.0, 1.0, float("nan")])
    assert guess_column_type(col) == ColumnType.BINARY


def test_guess_categorical_ratio_cutoff():
    """Integer column with low unique/total ratio should be CATEGORICAL."""
    # 5 unique values in 1000 rows => ratio=0.005 < default 0.02
    col = jnp.tile(jnp.arange(5, dtype=jnp.float32), 200)
    assert guess_column_type(col) == ColumnType.CATEGORICAL


def test_guess_custom_cutoffs():
    """Custom count_cutoff and ratio_cutoff should be respected."""
    # 10 unique integers in 50 rows; default cutoff=20 would say CATEGORICAL
    col = jnp.tile(jnp.arange(10, dtype=jnp.float32), 5)
    assert guess_column_type(col, count_cutoff=20) == ColumnType.CATEGORICAL
    # With count_cutoff=5 and high ratio, should be CONTINUOUS
    assert guess_column_type(col, count_cutoff=5, ratio_cutoff=0.001) == ColumnType.CONTINUOUS


# ---------------------------------------------------------------------------
# guess_column_types
# ---------------------------------------------------------------------------


def test_guess_column_types_mixed():
    """guess_column_types should return correct types for a mixed dataset."""
    key = jax.random.key(1)
    n = 100
    binary_col = jnp.array([0.0, 1.0] * (n // 2))
    cat_col = jnp.tile(jnp.arange(3, dtype=jnp.float32), n // 3 + 1)[:n]
    cont_col = jax.random.normal(key, (n,))
    data = jnp.column_stack([binary_col, cat_col, cont_col])

    types = guess_column_types(data)
    assert types[0] == ColumnType.BINARY
    assert types[1] == ColumnType.CATEGORICAL
    assert types[2] == ColumnType.CONTINUOUS


# ---------------------------------------------------------------------------
# CSV I/O round-trip
# ---------------------------------------------------------------------------


def test_csv_roundtrip(tmp_path):
    """write_csv then read_csv should recover original data."""
    data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    names = ["a", "b", "c"]
    path = tmp_path / "test.csv"

    write_csv(path, data, names)
    loaded_data, loaded_names = read_csv(path)

    assert loaded_names == names
    assert jnp.allclose(loaded_data, data)


def test_csv_roundtrip_with_nan(tmp_path):
    """NaN values should survive a CSV round-trip."""
    data = jnp.array([[1.0, float("nan"), 3.0], [float("nan"), 5.0, 6.0]])
    names = ["x", "y", "z"]
    path = tmp_path / "nan_test.csv"

    write_csv(path, data, names)
    loaded_data, loaded_names = read_csv(path)

    assert loaded_names == names
    # NaN != NaN, so check element-wise
    assert jnp.isnan(loaded_data[0, 1])
    assert jnp.isnan(loaded_data[1, 0])
    assert jnp.allclose(loaded_data[0, 0], 1.0)
    assert jnp.allclose(loaded_data[1, 1], 5.0)


def test_read_csv_no_header(tmp_path):
    """Reading without header should generate col_0, col_1, ... names."""
    path = tmp_path / "noheader.csv"
    path.write_text("1,2,3\n4,5,6\n")

    data, names = read_csv(path, has_header=False)
    assert names == ["col_0", "col_1", "col_2"]
    assert data.shape == (2, 3)
    assert jnp.allclose(data[0, 0], 1.0)


def test_read_csv_nan_values(tmp_path):
    """Various NaN string representations should be read as NaN."""
    path = tmp_path / "nanstrings.csv"
    path.write_text("a,b,c,d,e\nNA,NULL,.,N/A,\n1,2,3,4,5\n")

    data, _ = read_csv(path)
    # First row should be all NaN
    assert jnp.all(jnp.isnan(data[0]))
    # Second row should be all finite
    assert jnp.all(jnp.isfinite(data[1]))


def test_read_csv_non_numeric_becomes_nan(tmp_path):
    """Non-numeric strings should be read as NaN."""
    path = tmp_path / "mixed.csv"
    path.write_text("a,b\n1.0,hello\n3.0,4.0\n")

    data, _ = read_csv(path)
    assert jnp.isnan(data[0, 1])
    assert jnp.allclose(data[1, 1], 4.0)


def test_read_csv_ragged_rows(tmp_path):
    """Short rows should be padded with NaN."""
    path = tmp_path / "ragged.csv"
    path.write_text("a,b,c\n1,2,3\n4,5\n")

    data, names = read_csv(path)
    assert data.shape == (2, 3)
    assert jnp.isnan(data[1, 2])


# ---------------------------------------------------------------------------
# gen_column_metadata
# ---------------------------------------------------------------------------


def test_gen_column_metadata_basic():
    """Metadata should contain name mappings and column info."""
    data = jnp.array([[0.0, 1.0], [1.0, 2.0], [0.0, 3.0]])
    types = [ColumnType.BINARY, ColumnType.CATEGORICAL]
    names = ["flag", "category"]

    meta = gen_column_metadata(data, types, names)

    assert meta["name_to_idx"]["flag"] == 0
    assert meta["name_to_idx"]["category"] == 1
    assert meta["idx_to_name"][0] == "flag"
    assert len(meta["column_metadata"]) == 2
    assert meta["column_metadata"][0]["modeltype"] == ColumnType.BINARY.value
    assert meta["column_metadata"][1]["modeltype"] == ColumnType.CATEGORICAL.value


def test_gen_column_metadata_categorical_codes():
    """Categorical columns should have value_to_code and code_to_value."""
    data = jnp.array([[1.0], [3.0], [5.0], [1.0]])
    types = [ColumnType.CATEGORICAL]

    meta = gen_column_metadata(data, types)
    col_meta = meta["column_metadata"][0]

    assert "value_to_code" in col_meta
    assert "code_to_value" in col_meta
    # Should map 3 unique values: 1, 3, 5
    assert len(col_meta["value_to_code"]) == 3


def test_gen_column_metadata_default_names():
    """Without names, should use col_0, col_1, etc."""
    data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    types = [ColumnType.CONTINUOUS, ColumnType.CONTINUOUS]

    meta = gen_column_metadata(data, types)
    assert meta["idx_to_name"][0] == "col_0"
    assert meta["idx_to_name"][1] == "col_1"


def test_gen_column_metadata_with_nan():
    """NaN values should be excluded from categorical codes."""
    data = jnp.array([[1.0], [float("nan")], [2.0], [1.0]])
    types = [ColumnType.CATEGORICAL]

    meta = gen_column_metadata(data, types)
    col_meta = meta["column_metadata"][0]
    # Only 1 and 2 should appear, not NaN
    assert len(col_meta["value_to_code"]) == 2


# ---------------------------------------------------------------------------
# discretize_column
# ---------------------------------------------------------------------------


def test_discretize_basic():
    """Discretized values should be integer bin indices."""
    col = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
    disc, edges = discretize_column(col, n_bins=4)

    assert disc.shape == col.shape
    # Bin indices should be non-negative integers
    assert jnp.all(disc >= 0)
    assert edges.shape[0] == 5  # n_bins + 1


def test_discretize_nan_passthrough():
    """NaN values should produce valid bin index (digitize handles NaN)."""
    col = jnp.array([0.0, float("nan"), 1.0, 2.0])
    disc, edges = discretize_column(col, n_bins=3)
    # Should not crash; NaN bins are implementation-defined
    assert disc.shape == (4,)
    assert edges.shape[0] == 4  # n_bins + 1


def test_discretize_uniform_data():
    """Uniformly spaced data should spread across all bins."""
    col = jnp.linspace(0, 10, 100)
    disc, edges = discretize_column(col, n_bins=5)
    # Each bin should get roughly 20 values
    unique_bins = jnp.unique(disc)
    assert unique_bins.shape[0] >= 4  # at least most bins used
