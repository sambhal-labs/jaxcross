# Data Utilities

::: crosscat.data_utils
    options:
      show_source: false

## Overview

CSV I/O, column type detection, and discretization.

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
