"""Data I/O, metadata generation, and column type detection utilities.

Maps to original CrossCat data_utils.py:
- read_csv, write_csv, read_data_objects
- gen_M_c_from_T, gen_M_r_from_T
- guess_column_type, guess_column_types
- convert_columns_to_multinomial, convert_columns_to_continuous
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import jax.numpy as jnp
from jax import Array

from crosscat.types import ColumnType


def read_csv(
    filepath: str | Path,
    *,
    has_header: bool = True,
    nan_values: set[str] | None = None,
) -> tuple[Array, list[str]]:
    """Read CSV file into JAX array.

    Maps to original data_utils.read_csv().

    Args:
        filepath: Path to CSV file.
        has_header: Whether first row is a header.
        nan_values: Strings to interpret as NaN (default: {"", "NA", "nan", "NaN", "NULL"}).

    Returns:
        Tuple of (data_array, column_names).
    """
    if nan_values is None:
        nan_values = {"", "NA", "nan", "NaN", "NULL", "None", "null", "N/A", "."}

    filepath = Path(filepath)
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if has_header:
        col_names = rows[0]
        data_rows = rows[1:]
    else:
        data_rows = rows
        col_names = [f"col_{i}" for i in range(len(data_rows[0]))]

    # Convert to float array, replacing nan_values with NaN
    n_rows = len(data_rows)
    n_cols = len(col_names)
    values = []
    for row in data_rows:
        row_vals = []
        for val in row:
            if val.strip() in nan_values:
                row_vals.append(float("nan"))
            else:
                try:
                    row_vals.append(float(val))
                except ValueError:
                    row_vals.append(float("nan"))
        row_vals.extend([float("nan")] * (n_cols - len(row_vals)))
        values.append(row_vals[:n_cols])

    return jnp.array(values), col_names


def write_csv(
    filepath: str | Path,
    data: Array,
    column_names: list[str],
) -> None:
    """Write JAX array to CSV file.

    Args:
        filepath: Output path.
        data: Data array, shape (n_rows, n_cols).
        column_names: Column header names.
    """
    filepath = Path(filepath)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        for row in data.tolist():
            writer.writerow(row)


def guess_column_type(
    col_data: Array,
    *,
    count_cutoff: int = 20,
    ratio_cutoff: float = 0.02,
) -> ColumnType:
    """Guess column type from data, matching original CrossCat heuristic.

    Maps to original data_utils.guess_column_type().

    Decision logic:
    - If only 0s and 1s: BINARY
    - If integer-valued and n_unique <= count_cutoff: CATEGORICAL
    - If integer-valued and n_unique/n_rows < ratio_cutoff: CATEGORICAL
    - Otherwise: CONTINUOUS

    Args:
        col_data: 1D array of values (may contain NaN).
        count_cutoff: Max unique values for categorical.
        ratio_cutoff: Max unique/total ratio for categorical.

    Returns:
        Inferred ColumnType.
    """
    clean = col_data[~jnp.isnan(col_data)]
    if clean.shape[0] == 0:
        return ColumnType.CONTINUOUS

    unique_vals = jnp.unique(clean)
    n_unique = unique_vals.shape[0]
    n_total = clean.shape[0]

    # Check binary
    if n_unique <= 2:
        vals_set = set(unique_vals.tolist())
        if vals_set <= {0.0, 1.0}:
            return ColumnType.BINARY

    # Check if integer-valued
    is_integer = jnp.allclose(clean, jnp.round(clean))

    if is_integer and (n_unique <= count_cutoff or n_unique / n_total < ratio_cutoff):
        return ColumnType.CATEGORICAL

    return ColumnType.CONTINUOUS


def guess_column_types(
    data: Array,
    *,
    count_cutoff: int = 20,
    ratio_cutoff: float = 0.02,
) -> list[ColumnType]:
    """Guess column types for all columns in a data array.

    Maps to original data_utils.guess_column_types().

    Args:
        data: Data array, shape (n_rows, n_cols).
        count_cutoff: Max unique values for categorical.
        ratio_cutoff: Max unique/total ratio for categorical.

    Returns:
        List of ColumnType, one per column.
    """
    n_cols = data.shape[1]
    return [
        guess_column_type(data[:, j], count_cutoff=count_cutoff, ratio_cutoff=ratio_cutoff)
        for j in range(n_cols)
    ]


def gen_column_metadata(
    data: Array,
    column_types: list[ColumnType],
    column_names: list[str] | None = None,
) -> dict:
    """Generate column metadata dictionary (M_c equivalent).

    Maps to original data_utils.gen_M_c_from_T().

    Args:
        data: Data array, shape (n_rows, n_cols).
        column_types: Type per column.
        column_names: Column names (optional).

    Returns:
        Metadata dictionary with column info.
    """
    n_cols = data.shape[1]
    if column_names is None:
        column_names = [f"col_{j}" for j in range(n_cols)]

    metadata = {
        "name_to_idx": {name: j for j, name in enumerate(column_names)},
        "idx_to_name": {j: name for j, name in enumerate(column_names)},
        "column_metadata": [],
    }

    for j in range(n_cols):
        col_meta = {
            "modeltype": column_types[j].value,
            "name": column_names[j],
        }
        if column_types[j] in (ColumnType.CATEGORICAL, ColumnType.ORDINAL):
            clean = data[:, j][~jnp.isnan(data[:, j])]
            unique_vals = sorted(set(int(v) for v in clean.tolist()))
            col_meta["value_to_code"] = {str(v): i for i, v in enumerate(unique_vals)}
            col_meta["code_to_value"] = {i: str(v) for i, v in enumerate(unique_vals)}
        metadata["column_metadata"].append(col_meta)

    return metadata


def discretize_column(
    col_data: Array,
    n_bins: int = 10,
) -> tuple[Array, Array]:
    """Discretize a continuous column into bins.

    Maps to original data_utils.discretize_data().

    Args:
        col_data: 1D continuous data array.
        n_bins: Number of bins.

    Returns:
        Tuple of (discretized_data, bin_edges).
    """
    clean = col_data[~jnp.isnan(col_data)]
    bin_edges = jnp.linspace(float(jnp.min(clean)), float(jnp.max(clean)), n_bins + 1)
    discretized = jnp.digitize(col_data, bin_edges[1:-1])
    return discretized, bin_edges
