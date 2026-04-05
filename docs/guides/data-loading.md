# Data Loading & Column Types

## What

Load tabular data from CSV files into JAX arrays and assign the correct column type to each feature.

## When to Use

Every jax-crosscat workflow starts here. You need a 2D JAX array where rows are observations and columns are features, plus a list of column types.

## Loading CSV Data

```python
from crosscat import read_csv, guess_column_types

# Load from CSV
data, col_names = read_csv("employees.csv", has_header=True)
print(f"Shape: {data.shape}")
print(f"Columns: {col_names}")

# Handle custom missing value strings
data, col_names = read_csv("data.csv", nan_values={"NA", "null", "-999"})
```

## Column Types

Each column must be assigned one of 5 types:

| Type | Values | Statistical Model | Example |
|------|--------|-------------------|---------|
| `CONTINUOUS` | Any real number | Normal-Gamma | Salary, temperature |
| `CATEGORICAL` | Non-negative integers (0, 1, 2, ...) | Dirichlet-Categorical | Department ID, color |
| `BINARY` | 0 or 1 | Beta-Bernoulli | Yes/no flags |
| `ORDINAL` | Ordered integers (1, 2, 3, ...) | Ordered Logistic | Star ratings, education |
| `CYCLIC` | Floats in [0, 2*pi) | Von Mises | Wind direction, hour of day |

## Auto-Detection

```python
col_types = guess_column_types(data)
for name, ct in zip(col_names, col_types):
    print(f"  {name}: {ct.value}")
```

The heuristic logic:

1. Only values 0 and 1 → `BINARY`
2. All integers and ≤ 20 unique values → `CATEGORICAL`
3. All integers and unique/total < 2% → `CATEGORICAL`
4. Otherwise → `CONTINUOUS`

!!! warning "Auto-detection limitations"
    - `ORDINAL` and `CYCLIC` types are **never** auto-detected — you must specify these manually
    - Categorical columns must be encoded as non-negative integers (0, 1, 2, ...), not strings

## Manual Override

```python
from crosscat.types import ColumnType

# Start with auto-detection, then override specific columns
col_types = guess_column_types(data)
col_types[3] = ColumnType.ORDINAL    # education_level is ordinal, not categorical
col_types[7] = ColumnType.CYCLIC     # wind_direction is cyclic
```

Or specify all types manually:

```python
col_types = [
    ColumnType.CONTINUOUS,    # salary
    ColumnType.CONTINUOUS,    # years_experience
    ColumnType.CATEGORICAL,   # department (0=eng, 1=sales, 2=hr)
    ColumnType.BINARY,        # is_remote
    ColumnType.ORDINAL,       # performance_rating (1-5)
]
```

## Missing Data

Missing values are represented as `NaN`:

```python
import jax.numpy as jnp

# NaN in CSV files is loaded automatically
data, _ = read_csv("data_with_gaps.csv")

# Or inject manually
data = data.at[5, 2].set(jnp.nan)  # mark cell (5, 2) as missing
```

CrossCat handles NaN transparently — no preprocessing needed. See [Missing Data Handling](missing-data.md) for details.

## Discretizing Continuous Columns

If you want to treat a continuous column as categorical:

```python
from crosscat import discretize_column

binned, edges = discretize_column(data[:, 0], n_bins=10)
data = data.at[:, 0].set(binned)
col_types[0] = ColumnType.CATEGORICAL
```

## Column Metadata

Generate a metadata dictionary for documentation or persistence:

```python
from crosscat import gen_column_metadata

metadata = gen_column_metadata(data, col_types, col_names)
# Contains: name_to_idx, idx_to_name, per-column type and category info
```

## Large CSV Files

For CSV files too large to fit in memory at once, use chunked reading:

```python
from crosscat import read_csv_chunked
import jax.numpy as jnp

chunks = []
for chunk, col_names in read_csv_chunked("large_data.csv", chunk_size=50_000):
    chunks.append(chunk)
data = jnp.concatenate(chunks)
```

## Parquet Files

Parquet is the recommended format for large datasets — columnar storage with compression:

```python
from crosscat import read_parquet, write_parquet

# Read (requires pyarrow)
data, col_names = read_parquet("data.parquet")

# Read specific columns only
data, col_names = read_parquet("data.parquet", columns=["salary", "age"])

# Write
write_parquet("output.parquet", data, col_names)
```

## Arrow IPC Format

Arrow IPC (Feather v2) provides zero-copy reads for maximum I/O speed:

```python
from crosscat import read_arrow_ipc, write_arrow_ipc

data, col_names = read_arrow_ipc("data.arrow")
write_arrow_ipc("output.arrow", data, col_names)
```

## Memory-Mapped Loading

For very large numeric arrays, use NumPy memory mapping to avoid loading the full dataset into RAM:

```python
from crosscat import read_npy, write_npy

# Save once
write_npy("data.npy", data)

# Load on demand (memory-mapped, no full copy)
data = read_npy("data.npy")
```

## Format Comparison

| Format | Speed | Size | Random Access | Dependencies |
|--------|-------|------|--------------|--------------|
| CSV | Slow | Large | No | None |
| Parquet | Fast | Small (compressed) | Column-level | `pyarrow` |
| Arrow IPC | Fastest | Medium | Column-level | `pyarrow` |
| NPY | Fast | Medium | Memory-mapped | None |

## API Reference

- [`read_csv`](../api/data-utils.md#read_csv)
- [`read_csv_chunked`](../api/data-utils.md#read_csv_chunked)
- [`read_parquet`](../api/data-utils.md#read_parquet)
- [`read_arrow_ipc`](../api/data-utils.md#read_arrow_ipc)
- [`read_npy`](../api/data-utils.md#read_npy)
- [`guess_column_types`](../api/data-utils.md#guess_column_types)
- [`gen_column_metadata`](../api/data-utils.md#gen_column_metadata)
- [`discretize_column`](../api/data-utils.md#discretize_column)
