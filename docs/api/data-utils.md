# Data Utilities

::: crosscat.data_utils
    options:
      show_source: false

## Overview

CSV/Parquet/Arrow/NPY I/O, column type detection, and discretization. **Prefer `save_data` / `load_data` (Arrow IPC) for new code** — they preserve column type metadata inside the file itself so you do not need a sidecar schema. CSV / Parquet / NPY entry points remain for compatibility.

---

## High-Level I/O (Recommended)

### `save_data`

```python
save_data(filepath, data, *, column_names=None, column_types=None, compression="lz4") -> None
```

Save a data array in Arrow IPC format with column type metadata embedded in the Arrow schema.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str \| Path` | Output path (conventionally `.arrow`) |
| `data` | `Array (n_rows, n_cols)` | Data array (coerced to `float32`) |
| `column_names` | `list[str] \| None` | Column names. Defaults to `col_0`, `col_1`, ... |
| `column_types` | `list[ColumnType] \| None` | If provided, stored in Arrow schema metadata for recovery on load |
| `compression` | `str` | `"lz4"`, `"zstd"`, or `"uncompressed"` (must be `"uncompressed"` for true memory-mapped reads) |

Raises `ValueError` if `compression` is invalid or `column_names` / `column_types` length does not match `data.shape[1]`.

### `load_data`

```python
load_data(filepath, *, memory_map=True, columns=None) -> tuple[Array, list[str], list[ColumnType] | None]
```

Load a data array from Arrow IPC format, recovering column type metadata if present.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str \| Path` | Path to `.arrow` file (from `save_data` or `save_arrow`) |
| `memory_map` | `bool` | Memory-map at the Arrow level (requires uncompressed file for true mmap) |
| `columns` | `list[str] \| None` | Optional subset of columns to load |

**Returns**: `(data_array, column_names, column_types)`. `column_types` is `None` if the file has no embedded schema metadata.

```python
from crosscat import save_data, load_data, guess_column_types

data, names = read_csv("raw.csv")
col_types = guess_column_types(data)
save_data("dataset.arrow", data, column_names=names, column_types=col_types)

# Later — type info round-trips without a sidecar
data2, names2, types2 = load_data("dataset.arrow")
assert types2 == col_types
```

---

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
read_csv_chunked(filepath, *, chunk_size=10_000, has_header=True, nan_values=None) -> tuple[Array, list[str]]
```

Read a large CSV file in chunks to limit peak memory. Reads `chunk_size` rows at a time into NumPy, then converts to a single JAX array at the end. This avoids holding the full Python list-of-lists in memory simultaneously with the final array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str \| Path` | Path to CSV file |
| `chunk_size` | `int` | Number of rows to read per chunk (default 10,000) |
| `has_header` | `bool` | Whether first row is column names |
| `nan_values` | `set[str] \| None` | Additional strings to treat as NaN |

**Returns**: `(data_array, column_names)` — same as `read_csv`, but with bounded peak memory.

```python
from crosscat import read_csv_chunked

data, col_names = read_csv_chunked("large_data.csv", chunk_size=50_000)
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

## Arrow IPC (Feather v2)

Requires `pyarrow` (`pip install pyarrow`).

### `save_arrow`

```python
save_arrow(filepath, data, column_names=None, *, compression="lz4") -> None
```

Save data array in Arrow IPC format. Faster than Parquet for read-heavy workflows.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str \| Path` | Output path (conventionally .arrow or .feather) |
| `data` | `Array (n_rows, n_cols)` | Data array |
| `column_names` | `list[str] \| None` | Column names. Defaults to col_0, col_1, ... |
| `compression` | `str` | Compression codec: `"lz4"`, `"zstd"`, or `"uncompressed"` |

!!! note
    LZ4-compressed files cannot be memory-mapped for random access. Use `compression="uncompressed"` for true memory-mapped reads via `load_arrow(memory_map=True)`.

### `load_arrow`

```python
load_arrow(filepath, *, memory_map=True, columns=None) -> tuple[Array, list[str]]
```

Load data from Arrow IPC format.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str \| Path` | Path to .arrow/.feather file |
| `memory_map` | `bool` | If True, memory-map the file at the Arrow level |
| `columns` | `list[str] \| None` | Optional subset of columns to load |

**Returns**: `(data_array, column_names)`.

---

## NumPy Memory-Mapped I/O

### `save_npy`

```python
save_npy(filepath, data, column_names=None) -> None
```

Save data array to uncompressed `.npy` for fast memory-mapped reloading. Column names are stored in a separate JSON sidecar file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str \| Path` | Output path (`.npy` suffix is used regardless of extension) |
| `data` | `Array (n_rows, n_cols)` | Data array |
| `column_names` | `list[str] \| None` | Optional column names (saved as JSON sidecar) |

### `load_npy_mmap`

```python
load_npy_mmap(filepath, *, mmap_mode="r") -> tuple[np.ndarray, list[str] | None]
```

Load data from `.npy` file with memory-mapping for large files. Returns a **NumPy memmap**, not a JAX array. The OS pages data in on demand, so peak RAM stays low for multi-GB files.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str \| Path` | Path to `.npy` file |
| `mmap_mode` | `str` | NumPy mmap mode (`"r"` for read-only, `"r+"` for read-write) |

**Returns**: `(numpy_memmap, column_names_or_None)`.

```python
data_np, names = load_npy_mmap("data.npy")
batch = jnp.array(data_np[1000:2000])  # only this slice hits RAM/GPU
```

!!! warning "Deprecated aliases"
    `save_npz` and `load_npz_mmap` are deprecated aliases that emit `DeprecationWarning`. Use `save_npy` and `load_npy_mmap` instead.
