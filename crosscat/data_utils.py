"""Data I/O, metadata generation, and column type detection utilities.

Maps to original CrossCat data_utils.py:
- read_csv, write_csv, read_data_objects
- gen_M_c_from_T, gen_M_r_from_T
- guess_column_type, guess_column_types
- convert_columns_to_multinomial, convert_columns_to_continuous

Scaling additions:
- read_csv_chunked: streaming CSV reader for large files
- load_npy_mmap: memory-mapped NPY loading
- read_parquet: Apache Parquet/Arrow integration
"""

from __future__ import annotations

import csv
import warnings
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

    if not rows:
        raise ValueError(f"CSV file is empty: {filepath}")

    if has_header:
        col_names = rows[0]
        data_rows = rows[1:]
    else:
        data_rows = rows
        col_names = [f"col_{i}" for i in range(len(data_rows[0]))]

    if not data_rows:
        return jnp.zeros((0, len(col_names))), col_names

    n_cols = len(col_names)
    parsed, bad_examples, n_bad, n_mismatched = _parse_rows(data_rows, n_cols, nan_values)

    if n_bad > 0:
        warnings.warn(
            f"Could not parse {n_bad} values to float "
            f"(converted to NaN). Examples: {bad_examples[:5]}",
            stacklevel=2,
        )
    if n_mismatched > 0:
        warnings.warn(
            f"{n_mismatched} rows had mismatched column count "
            f"(expected {n_cols}). Short rows are NaN-padded, long rows truncated.",
            stacklevel=2,
        )

    return jnp.array(parsed), col_names


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
    all_bad_examples: list[str] = []
    total_bad = 0
    total_mismatched = 0

    with open(filepath, newline="") as f:
        reader = csv.reader(f)

        try:
            if has_header:
                col_names = next(reader)
            else:
                first_row = next(reader)
                col_names = [f"col_{i}" for i in range(len(first_row))]
        except StopIteration:
            raise ValueError(f"CSV file is empty: {filepath}") from None

        if not has_header:
            # Process first row since we already consumed it
            arr, bad, nb, mis = _parse_rows([first_row], len(col_names), nan_values)
            chunks.append(arr)
            all_bad_examples.extend(bad)
            total_bad += nb
            total_mismatched += mis

        n_cols = len(col_names)
        batch: list[list[str]] = []

        for row in reader:
            batch.append(row)
            if len(batch) >= chunk_size:
                arr, bad, nb, mis = _parse_rows(batch, n_cols, nan_values)
                chunks.append(arr)
                all_bad_examples.extend(bad[: 5 - len(all_bad_examples)])
                total_bad += nb
                total_mismatched += mis
                batch = []

        if batch:
            arr, bad, nb, mis = _parse_rows(batch, n_cols, nan_values)
            chunks.append(arr)
            all_bad_examples.extend(bad[: 5 - len(all_bad_examples)])
            total_bad += nb
            total_mismatched += mis

    if not chunks:
        return jnp.zeros((0, len(col_names))), col_names

    if total_bad > 0:
        warnings.warn(
            f"Could not parse {total_bad} values to float "
            f"(converted to NaN). Examples: {all_bad_examples[:5]}",
            stacklevel=2,
        )
    if total_mismatched > 0:
        warnings.warn(
            f"{total_mismatched} rows had mismatched column count "
            f"(expected {n_cols}). Short rows are NaN-padded, long rows truncated.",
            stacklevel=2,
        )

    data_np = np.concatenate(chunks, axis=0)
    return jnp.array(data_np), col_names


def _parse_rows(
    rows: list[list[str]],
    n_cols: int,
    nan_values: set[str],
) -> tuple[np.ndarray, list[str], int, int]:
    """Parse a batch of CSV string rows into a float32 NumPy array.

    Returns:
        Tuple of (array, unparseable_examples, n_bad, n_mismatched_rows)
        where unparseable_examples collects up to 5 sample values that
        could not be converted to float, n_bad is the total count of
        unparseable values, and n_mismatched_rows counts rows with wrong
        column count.
    """
    out = np.full((len(rows), n_cols), float("nan"), dtype=np.float32)
    bad_examples: list[str] = []
    n_bad = 0
    n_mismatched = 0
    for i, row in enumerate(rows):
        if len(row) != n_cols:
            n_mismatched += 1
        for j, val in enumerate(row[:n_cols]):
            stripped = val.strip()
            if stripped not in nan_values:
                try:
                    out[i, j] = float(stripped)
                except ValueError:
                    n_bad += 1
                    if len(bad_examples) < 5:
                        bad_examples.append(stripped)
    return out, bad_examples, n_bad, n_mismatched


def save_npy(
    filepath: str | Path,
    data: Array,
    column_names: list[str] | None = None,
) -> None:
    """Save data array to uncompressed ``.npy`` for fast memory-mapped reloading.

    Saves an uncompressed ``.npy`` file so that ``load_npy_mmap`` can
    truly memory-map the result. Column names are stored in a separate
    JSON sidecar file.

    Args:
        filepath: Output path. The ``.npy`` suffix is used regardless of
            what extension is provided.
        data: Data array, shape (n_rows, n_cols).
        column_names: Optional column names (saved as JSON sidecar).
    """
    import json

    filepath = Path(filepath)
    if filepath.suffix and filepath.suffix != ".npy":
        warnings.warn(
            f"save_npy writes .npy (not {filepath.suffix}). "
            f"Output file: {filepath.with_suffix('.npy')}",
            stacklevel=2,
        )
    filepath = filepath.with_suffix(".npy")
    np.save(filepath, np.asarray(data, dtype=np.float32))
    if column_names is not None:
        meta_path = filepath.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump({"column_names": column_names}, f)


def save_npz(
    filepath: str | Path,
    data: Array,
    column_names: list[str] | None = None,
) -> None:
    """Deprecated: use ``save_npy`` instead.

    This function saves ``.npy`` files despite its name. The ``save_npy``
    alias is preferred for clarity.
    """
    warnings.warn(
        "save_npz is deprecated — use save_npy instead (saves .npy files, not .npz).",
        DeprecationWarning,
        stacklevel=2,
    )
    save_npy(filepath, data, column_names)


def load_npy_mmap(
    filepath: str | Path,
    *,
    mmap_mode: str = "r",
) -> tuple[np.ndarray, list[str] | None]:
    """Load data from ``.npy`` file with memory-mapping for large files.

    Returns a **NumPy memmap**, not a JAX array.  The OS pages data in on
    demand, so peak RAM stays low for multi-GB files.  Convert slices to
    JAX when needed::

        data_np, names = load_npy_mmap("data.npy")
        batch = jnp.array(data_np[1000:2000])   # only this slice hits RAM/GPU

    Args:
        filepath: Path to ``.npy`` file (created by ``save_npy``).
        mmap_mode: NumPy mmap mode ('r' for read-only, 'r+' for read-write).

    Returns:
        Tuple of (numpy_memmap, column_names_or_None).

    Warns:
        If the JSON sidecar with column names is missing.
    """
    import json

    filepath = Path(filepath)
    if filepath.suffix and filepath.suffix != ".npy":
        warnings.warn(
            f"load_npy_mmap reads .npy (not {filepath.suffix}). "
            f"Loading: {filepath.with_suffix('.npy')}",
            stacklevel=2,
        )
    filepath = filepath.with_suffix(".npy")
    data_np = np.load(filepath, mmap_mode=mmap_mode)
    col_names = None
    meta_path = filepath.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        col_names = meta.get("column_names")
    else:
        warnings.warn(
            f"Column name sidecar {meta_path} not found. Column names will be None.",
            stacklevel=2,
        )
    return data_np, col_names


def load_npz_mmap(
    filepath: str | Path,
    *,
    mmap_mode: str = "r",
) -> tuple[np.ndarray, list[str] | None]:
    """Deprecated: use ``load_npy_mmap`` instead.

    This function loads ``.npy`` files despite its name. The ``load_npy_mmap``
    alias is preferred for clarity.
    """
    warnings.warn(
        "load_npz_mmap is deprecated — use load_npy_mmap instead (loads .npy files).",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_npy_mmap(filepath, mmap_mode=mmap_mode)


def read_parquet(
    filepath: str | Path,
    *,
    columns: list[str] | None = None,
) -> tuple[Array, list[str]]:
    """Read an Apache Parquet file into a JAX array.

    Requires ``pyarrow`` to be installed (``pip install pyarrow``).
    The full data is loaded into a dense JAX array in memory.

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
    return _arrow_table_to_jax(table)


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


# ---------------------------------------------------------------------------
# Arrow IPC (Feather v2) format
# ---------------------------------------------------------------------------


def _require_pyarrow():
    """Import and return pyarrow, raising a clear error if missing."""
    try:
        import pyarrow

        return pyarrow
    except ImportError:
        raise ImportError(
            "pyarrow is required for Arrow-based data loading. Install with: pip install pyarrow"
        ) from None


def _arrow_table_to_jax(table) -> tuple[Array, list[str]]:
    """Convert an Arrow Table to a JAX float32 array.

    Reads each column via ``to_numpy()`` into a pre-allocated float32
    matrix, then converts to JAX. This materializes the full dataset
    into memory (Arrow + NumPy + JAX copies).
    """
    col_names = table.column_names
    n_rows = table.num_rows
    n_cols = table.num_columns

    if n_rows == 0:
        return jnp.zeros((0, n_cols)), col_names

    out = np.empty((n_rows, n_cols), dtype=np.float32)
    for j in range(n_cols):
        chunked = table.column(j)
        arr = chunked.to_numpy(zero_copy_only=False).astype(np.float32)
        out[:, j] = arr

    return jnp.array(out), col_names


def save_arrow(
    filepath: str | Path,
    data: Array,
    column_names: list[str] | None = None,
    *,
    compression: str = "lz4",
) -> None:
    """Save data array in Arrow IPC (Feather v2) format.

    Arrow IPC is faster than NPZ/Parquet for read-heavy workflows.
    Note: LZ4-compressed files (the default) cannot be memory-mapped
    for random access — use ``compression="uncompressed"`` if you
    need true memory-mapped reads via ``load_arrow(memory_map=True)``.

    Args:
        filepath: Output path (conventionally .arrow or .feather).
        data: Data array, shape (n_rows, n_cols).
        column_names: Column names. Defaults to col_0, col_1, ...
        compression: Compression codec ("lz4", "zstd", "uncompressed").

    Raises:
        ImportError: If pyarrow is not installed.
    """
    pa = _require_pyarrow()

    filepath = Path(filepath)
    data_np = np.asarray(data, dtype=np.float32)
    n_cols = data_np.shape[1]

    if column_names is None:
        column_names = [f"col_{j}" for j in range(n_cols)]

    table = pa.table({name: data_np[:, j] for j, name in enumerate(column_names)})
    import pyarrow.feather as pf

    pf.write_feather(table, filepath, compression=compression)


def load_arrow(
    filepath: str | Path,
    *,
    memory_map: bool = True,
    columns: list[str] | None = None,
) -> tuple[Array, list[str]]:
    """Load data from Arrow IPC (Feather v2) format.

    When ``memory_map=True`` (default), pyarrow memory-maps the file.
    However, the data is still fully materialized into a JAX array,
    so peak RAM includes Arrow + NumPy + JAX copies.  For truly
    lazy loading, use ``load_npy_mmap`` which returns a NumPy memmap.

    Args:
        filepath: Path to .arrow/.feather file (created by ``save_arrow``).
        memory_map: If True, memory-map the file at the Arrow level.
        columns: Optional subset of columns to load.

    Returns:
        Tuple of (data_array, column_names).

    Raises:
        ImportError: If pyarrow is not installed.
    """
    _require_pyarrow()
    import pyarrow.feather as pf

    table = pf.read_table(filepath, memory_map=memory_map, columns=columns)
    return _arrow_table_to_jax(table)
