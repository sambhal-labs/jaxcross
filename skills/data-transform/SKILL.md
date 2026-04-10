---
name: data-transform
description: Transform raw tabular data for jaxcross modeling. Encodes string categoricals as 0-indexed integers, converts cyclic features to radians, decomposes datetime columns, validates float32 compatibility, and outputs an Arrow IPC file with a column_types list. Use after /data-quality or when raw data needs encoding before modeling.
version: "1.0.0"
license: Apache-2.0
---

# Data Transform

Transform raw tabular data into jaxcross-ready format: all columns as float32, categoricals integer-encoded, cyclic features in radians.

Usage: `/data-transform <file_path> [--output output.arrow]`

Examples:
- `/data-transform data/raw_data.csv`
- `/data-transform data/raw_data.parquet --output data/prepared.arrow`

## Step 1: Load and assess

```python
import pandas as pd
import numpy as np
import json

df = pd.read_csv("<file_path>")  # or read_parquet, read_feather
print(f"Input: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Dtypes:\n{df.dtypes}")
```

Review the `/data-quality` report if available. Identify which columns need transformation.

## Step 2: Categorical encoding (string → integer)

For each string column, map unique values to 0-indexed integers:

```python
encodings = {}  # Track mappings for reverse lookup

for col in df.select_dtypes(include=["object", "category"]).columns:
    unique_vals = sorted(df[col].dropna().unique())
    mapping = {v: i for i, v in enumerate(unique_vals)}
    encodings[col] = {str(k): int(v) for k, v in mapping.items()}
    df[col] = df[col].map(mapping)  # NaN stays NaN
    print(f"Encoded '{col}': {len(mapping)} categories → 0..{len(mapping)-1}")
```

Save the encoding mappings for reverse lookup:
```python
with open("data/encodings.json", "w") as f:
    json.dump(encodings, f, indent=2)
```

**Important:** Values must be 0-indexed contiguous integers (0, 1, 2, ..., K-1). Gaps (0, 2, 5) will cause issues with `max_categories` sizing.

## Step 3: Binary encoding

For columns with exactly 2 unique values, encode as 0/1:

```python
for col in df.columns:
    unique = df[col].dropna().unique()
    if len(unique) == 2 and not set(unique).issubset({0, 1, 0.0, 1.0}):
        mapping = {unique[0]: 0, unique[1]: 1}
        encodings[col] = {str(k): int(v) for k, v in mapping.items()}
        df[col] = df[col].map(mapping)
        print(f"Binary-encoded '{col}': {mapping}")
```

## Step 4: Cyclic encoding (→ radians)

Convert cyclic features to radians in [0, 2*pi):

```python
cyclic_transforms = {
    # "column_name": (period, offset)
    # hour of day: period=24, offset=0
    # day of week: period=7, offset=0
    # month: period=12, offset=1 (if 1-indexed)
    # compass bearing: period=360, offset=0
}

for col, (period, offset) in cyclic_transforms.items():
    df[col] = ((df[col] - offset) / period) * 2 * np.pi
    print(f"Cyclic-encoded '{col}': period={period} → radians [0, 2pi)")
```

Common cyclic encodings:
| Feature | Formula | Period |
|---------|---------|--------|
| Hour (0-23) | `hour * 2*pi/24` | 24 |
| Day of week (0-6) | `dow * 2*pi/7` | 7 |
| Month (1-12) | `(month-1) * 2*pi/12` | 12 |
| Compass (0-360) | `deg * pi/180` | 360 |
| Wind direction | `deg * pi/180` | 360 |

## Step 5: Datetime decomposition

Break datetime columns into numeric components:

```python
for col in df.select_dtypes(include=["datetime64"]).columns:
    dt = df[col]
    # Create derived columns
    df[f"{col}_hour"] = dt.dt.hour  # → mark as CYCLIC
    df[f"{col}_dow"] = dt.dt.dayofweek  # → mark as CYCLIC
    df[f"{col}_month"] = dt.dt.month  # → mark as CYCLIC
    df[f"{col}_year"] = dt.dt.year  # → CONTINUOUS
    df[f"{col}_day"] = dt.dt.day  # → CONTINUOUS or CATEGORICAL
    
    # Drop original datetime column (not float-compatible)
    df = df.drop(columns=[col])
    print(f"Decomposed '{col}' into hour/dow/month/year/day")
```

Then apply cyclic encoding to the hour/dow/month columns (Step 4).

## Step 6: Ordinal encoding

For columns with a natural order, map to sequential integers:

```python
ordinal_mappings = {
    # "education": {"none": 0, "high_school": 1, "bachelors": 2, "masters": 3, "phd": 4},
    # "size": {"small": 0, "medium": 1, "large": 2, "xlarge": 3},
    # "satisfaction": {"very_unsatisfied": 0, "unsatisfied": 1, "neutral": 2, "satisfied": 3, "very_satisfied": 4},
}

for col, mapping in ordinal_mappings.items():
    if col in df.columns:
        encodings[col] = {str(k): int(v) for k, v in mapping.items()}
        df[col] = df[col].map(mapping)
        print(f"Ordinal-encoded '{col}': {mapping}")
```

## Step 7: Drop ID and non-informative columns

```python
# Drop columns that are pure identifiers
id_cols = [col for col in df.columns if df[col].nunique() == len(df)]
if id_cols:
    print(f"Dropping ID columns: {id_cols}")
    df = df.drop(columns=id_cols)

# Drop constant columns
const_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
if const_cols:
    print(f"Dropping constant columns: {const_cols}")
    df = df.drop(columns=const_cols)
```

## Step 8: Validate and convert to float32

```python
# Convert all columns to float32
for col in df.columns:
    try:
        df[col] = df[col].astype(np.float32)
    except (ValueError, TypeError) as e:
        print(f"ERROR: Cannot convert '{col}' to float32: {e}")
        print(f"  Unique values sample: {df[col].dropna().unique()[:10]}")

# Check for infinities
inf_cols = [col for col in df.columns if np.isinf(df[col].dropna().values).any()]
if inf_cols:
    print(f"WARNING: Infinite values in: {inf_cols}")
    for col in inf_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
```

## Step 9: Generate column_types list

```python
from crosscat.types import ColumnType

column_types = []
for col in df.columns:
    s = df[col].dropna()
    unique_vals = s.unique()
    
    if col in cyclic_transforms:
        column_types.append(ColumnType.CYCLIC)
    elif col in ordinal_mappings:
        column_types.append(ColumnType.ORDINAL)
    elif set(unique_vals).issubset({0.0, 1.0}):
        column_types.append(ColumnType.BINARY)
    elif s.dtype in [np.float32] and s.nunique() <= 20 and all(v == int(v) for v in unique_vals if not np.isnan(v)):
        column_types.append(ColumnType.CATEGORICAL)
    else:
        column_types.append(ColumnType.CONTINUOUS)

# Print assignments
for col, ct in zip(df.columns, column_types):
    print(f"  {col}: {ct.name}")
```

## Step 10: Save output

```python
import jax.numpy as jnp
from crosscat.data_utils import save_data

# Convert to JAX array
data = jnp.array(df.values, dtype=jnp.float32)

# Save as Arrow IPC (preferred)
save_data(data, "data/prepared.arrow", column_names=list(df.columns), column_types=column_types)
print(f"\nSaved: data/prepared.arrow ({data.shape[0]} rows x {data.shape[1]} cols)")

# Also save column_types as Python for easy reuse
col_types_str = [ct.name for ct in column_types]
with open("data/column_types.json", "w") as f:
    json.dump({"columns": list(df.columns), "types": col_types_str}, f, indent=2)

# Save encodings
with open("data/encodings.json", "w") as f:
    json.dump(encodings, f, indent=2)

print(f"Column types saved to: data/column_types.json")
print(f"Encodings saved to: data/encodings.json")
```

**Output files:**
1. `data/prepared.arrow` — jaxcross-ready data (Arrow IPC)
2. `data/column_types.json` — column names + ColumnType assignments
3. `data/encodings.json` — categorical encoding mappings (for reverse lookup)

## Common Pitfalls

- **Category values must be 0-indexed**: jaxcross expects 0, 1, 2, ..., K-1. If your categories start at 1, subtract 1.
- **ORDINAL/CYCLIC are never auto-detected**: `guess_column_types()` only returns CONTINUOUS, CATEGORICAL, BINARY. Always set ordinal and cyclic manually.
- **NaN stays NaN**: Don't fill NaN before modeling. jaxcross handles missing data natively via sufficient statistic filtering. Use `/jxc-impute` after modeling instead.
- **Category values must be < max_categories**: When calling `pack_state()`, pass `data=` for validation. Out-of-range values are silently clipped.

See [encoding-guide.md](references/encoding-guide.md) for advanced encoding strategies.
