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

For datasets that don't fit in memory as Python lists, use chunked reading. It reads `chunk_size` rows at a time into NumPy internally, then returns a single concatenated JAX array:

```python
from crosscat import read_csv_chunked

data, col_names = read_csv_chunked("large_data.csv", chunk_size=50_000)
```

## Parquet Files

Parquet is the recommended format for large datasets — columnar storage with compression (requires `pip install pyarrow`):

```python
from crosscat import read_parquet, write_parquet

# Read
data, col_names = read_parquet("data.parquet")

# Read specific columns only
data, col_names = read_parquet("data.parquet", columns=["salary", "age"])

# Write
write_parquet("output.parquet", data, col_names)
```

## Arrow IPC Format

Arrow IPC (Feather v2) is faster than Parquet for read-heavy workflows (requires `pip install pyarrow`):

```python
from crosscat import save_arrow, load_arrow

save_arrow("data.arrow", data, col_names, compression="lz4")
data, col_names = load_arrow("data.arrow")
```

## Memory-Mapped Loading

For multi-GB datasets, use NumPy memory-mapping to avoid loading everything into RAM. Returns a **NumPy memmap**, not a JAX array — convert slices as needed:

```python
from crosscat import save_npy, load_npy_mmap
import jax.numpy as jnp

# Save once (uncompressed .npy)
save_npy("data.npy", data, col_names)

# Load as memory-mapped NumPy array (OS pages on demand)
data_np, names = load_npy_mmap("data.npy")
batch = jnp.array(data_np[0:10_000])  # only this slice hits RAM/GPU
```

## Format Comparison

| Format | Speed | Size | Random Access | Dependencies |
|--------|-------|------|--------------|--------------|
| CSV | Slow | Large | No | None |
| Parquet | Fast | Small (compressed) | Column-level | `pyarrow` |
| Arrow IPC | Fastest | Medium | Column-level | `pyarrow` |
| NPY | Fast | Medium | Memory-mapped | None |

!!! tip
    For production workflows with repeated reads, save your data as `.npy` (for memory-mapping) or `.arrow` (for fast columnar access). Use Parquet for interchange with other tools.

## API Reference

- [`read_csv`](../api/data-utils.md#read_csv) / [`write_csv`](../api/data-utils.md#write_csv)
- [`read_csv_chunked`](../api/data-utils.md#read_csv_chunked)
- [`read_parquet`](../api/data-utils.md#read_parquet) / [`write_parquet`](../api/data-utils.md#write_parquet)
- [`save_arrow`](../api/data-utils.md#save_arrow) / [`load_arrow`](../api/data-utils.md#load_arrow)
- [`save_npy`](../api/data-utils.md#save_npy) / [`load_npy_mmap`](../api/data-utils.md#load_npy_mmap)
- [`guess_column_types`](../api/data-utils.md#guess_column_types)
- [`gen_column_metadata`](../api/data-utils.md#gen_column_metadata)
- [`discretize_column`](../api/data-utils.md#discretize_column)
