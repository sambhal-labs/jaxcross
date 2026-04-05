# Data Utilities

::: crosscat.data_utils
    options:
      show_source: false

## Overview

CSV/Parquet/Arrow/NPY I/O, column type detection, and discretization.

## `read_csv`

```python
read_csv(filepath, *, has_header=True, nan_values=None) -> tuple[Array, list[str]]
```

Read a CSV file into a JAX array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str \| Path` | Path to CSV file |
| `has_header` | `bool` | Whether first row is column names |
| `nan_values` | `set[str] \| None` | Additional strings to treat as NaN (e.g., `{"NA", "null"}`) |

**Returns**: `(data_array, column_names)`.

## `read_csv_chunked`

```python
read_csv_chunked(filepath, *, chunk_size=10_000, has_header=True, nan_values=None) -> Iterator[tuple[Array, list[str]]]
```

Stream a large CSV file in chunks. Each iteration yields `(chunk_array, column_names)`. Column names are read from the header once and reused for all chunks.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str \| Path` | Path to CSV file |
| `chunk_size` | `int` | Rows per chunk |
| `has_header` | `bool` | Whether first row is column names |
| `nan_values` | `set[str] \| None` | Additional strings to treat as NaN |

**Returns**: Iterator of `(data_chunk, column_names)`.

```python
from crosscat import read_csv_chunked

chunks = []
for chunk, col_names in read_csv_chunked("large_data.csv", chunk_size=50_000):
    chunks.append(chunk)
data = jnp.concatenate(chunks)
```

## `write_csv`

```python
write_csv(filepath, data, column_names) -> None
```

Write a JAX array to CSV with column headers.

## `guess_column_type`

```python
guess_column_type(col_data, *, count_cutoff=20, ratio_cutoff=0.02) -> ColumnType
```

Heuristically detect column type from data values.

**Decision logic:**

1. Only values 0 and 1 → `BINARY`
2. All integers and unique count <= `count_cutoff` → `CATEGORICAL`
3. All integers and unique/total < `ratio_cutoff` → `CATEGORICAL`
4. Otherwise → `CONTINUOUS`

**Returns**: `ColumnType`.

## `guess_column_types`

```python
guess_column_types(data, *, count_cutoff=20, ratio_cutoff=0.02) -> list[ColumnType]
```

Detect types for all columns in a data matrix.

**Returns**: `list[ColumnType]`.

## `gen_column_metadata`

```python
gen_column_metadata(data, column_types, column_names=None) -> dict
```

Generate metadata dict for a dataset including type info, category mappings, and value ranges.

**Returns**: Dict with `name_to_idx`, `idx_to_name`, and per-column metadata.

## `discretize_column`

```python
discretize_column(col_data, n_bins=10) -> tuple[Array, Array]
```

Bin a continuous column into discrete buckets.

**Returns**: `(binned_data, bin_edges)`.

---

## Parquet I/O

### `read_parquet`

```python
read_parquet(filepath, *, columns=None) -> tuple[Array, list[str]]
```

Read a Parquet file into a JAX array. Requires `pyarrow`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str \| Path` | Path to `.parquet` file |
| `columns` | `list[str] \| None` | Subset of columns to read (default: all) |

**Returns**: `(data_array, column_names)`.

### `write_parquet`

```python
write_parquet(filepath, data, column_names) -> None
```

Write a JAX array to Parquet format. Requires `pyarrow`.

---

## Arrow IPC I/O

### `read_arrow_ipc`

```python
read_arrow_ipc(filepath) -> tuple[Array, list[str]]
```

Read an Arrow IPC (Feather v2) file into a JAX array. Requires `pyarrow`.

**Returns**: `(data_array, column_names)`.

### `write_arrow_ipc`

```python
write_arrow_ipc(filepath, data, column_names) -> None
```

Write a JAX array to Arrow IPC format.

---

## NumPy I/O

### `read_npy`

```python
read_npy(filepath) -> Array
```

Load a `.npy` file as a JAX array via memory mapping (`mmap_mode='r'`). Efficient for large arrays — data is loaded on demand without copying into RAM.

**Returns**: JAX array.

### `write_npy`

```python
write_npy(filepath, data) -> None
```

Save a JAX array to `.npy` format.
