"""Data I/O, metadata generation, and column type detection utilities.

Maps to original CrossCat data_utils.py:
- read_csv, write_csv, read_data_objects
- gen_M_c_from_T, gen_M_r_from_T
- guess_column_type, guess_column_types
- convert_columns_to_multinomial, convert_columns_to_continuous

Scaling additions:
- read_csv_chunked: streaming CSV reader for large files
- load_npz_mmap: memory-mapped NPZ loading
- read_parquet: Apache Parquet/Arrow integration
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np
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
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if has_header:
        col_names = rows[0]
        data_rows = rows[1:]
    else:
        data_rows = rows
        col_names = [f"col_{i}" for i in range(len(data_rows[0]))]

    # Convert to float array, replacing nan_values with NaN
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


# ---------------------------------------------------------------------------
# Scaling: chunked / streaming data loading
# ---------------------------------------------------------------------------


def read_csv_chunked(
    filepath: str | Path,
    *,
    chunk_size: int = 10_000,
    has_header: bool = True,
    nan_values: set[str] | None = None,
) -> tuple[Array, list[str]]:
    """Read a large CSV file in chunks to limit peak memory.

    Reads ``chunk_size`` rows at a time into NumPy, then converts to a single
    JAX array at the end. This avoids holding the full Python list-of-lists
    in memory simultaneously with the final array.

    Args:
        filepath: Path to CSV file.
        chunk_size: Number of rows to read per chunk.
        has_header: Whether first row is a header.
        nan_values: Strings to interpret as NaN.

    Returns:
        Tuple of (data_array, column_names).
    """
    if nan_values is None:
        nan_values = {"", "NA", "nan", "NaN", "NULL", "None", "null", "N/A", "."}

    filepath = Path(filepath)
    chunks: list[np.ndarray] = []

    with open(filepath, newline="") as f:
        reader = csv.reader(f)

        if has_header:
            col_names = next(reader)
        else:
            first_row = next(reader)
            col_names = [f"col_{i}" for i in range(len(first_row))]
            # Process first row since we already consumed it
            chunks.append(_parse_rows([first_row], len(col_names), nan_values))

        n_cols = len(col_names)
        batch: list[list[str]] = []

        for row in reader:
            batch.append(row)
            if len(batch) >= chunk_size:
                chunks.append(_parse_rows(batch, n_cols, nan_values))
                batch = []

        if batch:
            chunks.append(_parse_rows(batch, n_cols, nan_values))

    if not chunks:
        return jnp.zeros((0, len(col_names))), col_names

    data_np = np.concatenate(chunks, axis=0)
    return jnp.array(data_np), col_names


def _parse_rows(rows: list[list[str]], n_cols: int, nan_values: set[str]) -> np.ndarray:
    """Parse a batch of CSV string rows into a float32 NumPy array."""
    import contextlib

    out = np.full((len(rows), n_cols), float("nan"), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, val in enumerate(row[:n_cols]):
            if val.strip() not in nan_values:
                with contextlib.suppress(ValueError):
                    out[i, j] = float(val)
    return out


def save_npz(
    filepath: str | Path,
    data: Array,
    column_names: list[str] | None = None,
) -> None:
    """Save data array to compressed NPZ for fast reloading.

    Column names are stored in a separate JSON sidecar file to avoid
    pickle serialization.

    Args:
        filepath: Output .npz path.
        data: Data array, shape (n_rows, n_cols).
        column_names: Optional column names (saved as JSON sidecar).
    """
    import json

    filepath = Path(filepath)
    np.savez_compressed(filepath, data=np.asarray(data, dtype=np.float32))
    if column_names is not None:
        meta_path = filepath.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump({"column_names": column_names}, f)


def load_npz_mmap(
    filepath: str | Path,
    *,
    mmap_mode: str = "r",
) -> tuple[Array, list[str] | None]:
    """Load data from NPZ with memory-mapping for large files.

    Memory-mapping lets the OS page data in on demand rather than loading
    the entire array into RAM. JAX will transfer slices to GPU as needed.

    Args:
        filepath: Path to .npz file (created by ``save_npz``).
        mmap_mode: NumPy mmap mode ('r' for read-only, 'r+' for read-write).

    Returns:
        Tuple of (data_array, column_names_or_None).
    """
    import json

    filepath = Path(filepath)
    npz = np.load(filepath, mmap_mode=mmap_mode)
    data_np = npz["data"]
    col_names = None
    meta_path = filepath.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        col_names = meta.get("column_names")
    return jnp.array(data_np), col_names


def read_parquet(
    filepath: str | Path,
    *,
    columns: list[str] | None = None,
) -> tuple[Array, list[str]]:
    """Read an Apache Parquet file into a JAX array.

    Requires ``pyarrow`` to be installed (``pip install pyarrow``).
    Parquet's columnar format is a natural fit for CrossCat since column
    reassignment reads one column at a time.

    Args:
        filepath: Path to .parquet file.
        columns: Optional subset of columns to read.

    Returns:
        Tuple of (data_array, column_names).

    Raises:
        ImportError: If pyarrow is not installed.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet support. Install with: pip install pyarrow"
        ) from None

    table = pq.read_table(filepath, columns=columns)
    col_names = table.column_names
    # Convert to pandas then numpy for reliable NaN handling
    df = table.to_pandas()
    data_np = df.to_numpy(dtype=np.float32, na_value=float("nan"))
    return jnp.array(data_np), col_names


def write_parquet(
    filepath: str | Path,
    data: Array,
    column_names: list[str],
) -> None:
    """Write JAX array to Apache Parquet file.

    Requires ``pyarrow`` to be installed.

    Args:
        filepath: Output .parquet path.
        data: Data array, shape (n_rows, n_cols).
        column_names: Column header names.

    Raises:
        ImportError: If pyarrow is not installed.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet support. Install with: pip install pyarrow"
        ) from None

    data_np = np.asarray(data, dtype=np.float32)
    table = pa.table({name: data_np[:, j] for j, name in enumerate(column_names)})
    pq.write_table(table, filepath)
