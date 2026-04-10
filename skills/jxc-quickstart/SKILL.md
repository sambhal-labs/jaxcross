---
name: jxc-quickstart
description: End-to-end jaxcross workflow from a raw data file to first insights in under 5 minutes. Loads data, detects column types, initializes a model, runs 100 Gibbs sweeps, and produces a summary with top dependencies, anomalies, and cluster structure. Use for quick exploration or onboarding.
version: "1.0.0"
license: Apache-2.0
---

# jaxcross Quickstart

Go from a data file to first probabilistic insights in one command.

Usage: `/jxc-quickstart <file_path>`

Examples:
- `/jxc-quickstart data/prepared.arrow`
- `/jxc-quickstart data/customers.csv`

## Step 1: Load data

```python
import jax
import jax.numpy as jnp

# Try Arrow first (preserves column type metadata), fall back to CSV/Parquet
file_path = "<user_provided_path>"

try:
    from crosscat.data_utils import load_data
    data, col_names, col_types = load_data(file_path)
    print(f"Loaded Arrow data: {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"Column types from metadata: {[ct.name for ct in col_types]}")
except Exception:
    from crosscat import read_csv, read_parquet
    if file_path.endswith(".parquet"):
        data, col_names = read_parquet(file_path)
    else:
        data, col_names = read_csv(file_path)
    data = jnp.array(data, dtype=jnp.float32)
    col_types = None  # Will auto-detect in Step 2
    print(f"Loaded data: {data.shape[0]} rows x {data.shape[1]} columns")

print(f"Columns: {col_names}")
```

## Step 2: Detect column types

```python
from crosscat import guess_column_types
from crosscat.types import ColumnType

if col_types is None:
    col_types = guess_column_types(data)

print("\nColumn type assignments:")
for name, ct in zip(col_names, col_types):
    print(f"  {name}: {ct.name}")

# IMPORTANT: guess_column_types never returns ORDINAL or CYCLIC
# If you know columns are ordinal (ratings, education) or cyclic (hour, angle),
# override them manually:
# col_types[col_names.index("education_level")] = ColumnType.ORDINAL
# col_types[col_names.index("hour_of_day")] = ColumnType.CYCLIC
```

Ask the user if any column types need manual override before proceeding.

## Step 3: Initialize model

```python
from crosscat import initialize

key = jax.random.key(42)
result = initialize(key, data, col_types)
state = result.state  # CrossCatState

print(f"\nInitialized model:")
print(f"  Views: {len(state.views)}")
for i, view in enumerate(state.views):
    n_clusters = len(set(int(a) for a in view.row_assignments))
    print(f"  View {i}: {len(view.column_indices)} columns, {n_clusters} clusters")
```

**Note:** `initialize()` returns an `InitResult`, not a bare state. Access `.state` to get the `CrossCatState`.

## Step 4: Pack and run Gibbs sweeps

```python
from crosscat import pack_state, packed_gibbs_sweep

# Pack for JIT compilation
packed = pack_state(state, data=data)

# Run 100 sweeps (first call triggers JIT compilation — may take 30-60s)
key, subkey = jax.random.split(key)
print("\nRunning 100 Gibbs sweeps (first run compiles JIT — please wait)...")
packed = packed_gibbs_sweep(subkey, packed, data, n_sweeps=100)

# Check log-joint (higher is better)
from crosscat import packed_log_joint
lj = float(packed_log_joint(packed, data))
print(f"Log-joint after 100 sweeps: {lj:.1f}")
```

## Step 5: Quick insights

### Top dependencies (which columns are related?)

```python
from crosscat import packed_dependence_matrix, unbatch_packed_states

# Dependence matrix (accepts list of states for posterior averaging)
z_matrix = packed_dependence_matrix([packed])

# Find top 10 strongest dependencies
n_cols = len(col_names)
pairs = []
for i in range(n_cols):
    for j in range(i + 1, n_cols):
        pairs.append((col_names[i], col_names[j], float(z_matrix[i, j])))

pairs.sort(key=lambda x: -x[2])
print("\nTop 10 variable dependencies:")
for a, b, score in pairs[:10]:
    print(f"  {a} <-> {b}: {score:.2f}")
```

### Top anomalies (which rows are unusual?)

```python
from crosscat import batch_anomaly_score

row_ids = jnp.arange(data.shape[0])
scores = batch_anomaly_score(packed, data, row_ids)

# Top 5 most anomalous rows
top_idx = jnp.argsort(-scores)[:5]
print("\nTop 5 anomalous rows:")
for rank, idx in enumerate(top_idx):
    print(f"  Row {int(idx)}: anomaly score = {float(scores[idx]):.3f}")
    # Show the row values
    row_vals = {col_names[j]: float(data[idx, j]) for j in range(n_cols) if not jnp.isnan(data[idx, j])}
    print(f"    Values: {row_vals}")
```

### Cluster structure

```python
from crosscat import unpack_state

state_out = unpack_state(packed, col_types, data=data)

print(f"\nModel structure:")
print(f"  Total views: {len(state_out.views)}")
for i, view in enumerate(state_out.views):
    col_idx = view.column_indices
    view_cols = [col_names[j] for j in col_idx]
    n_clusters = len(set(int(a) for a in view.row_assignments))
    print(f"  View {i}: {view_cols} ({n_clusters} clusters)")
```

## Step 6: Save model

```python
from crosscat import save_packed_state

save_packed_state(packed, "model.jxc", column_types=col_types)
print(f"\nModel saved to: model.jxc")
print(f"Load later with: packed, col_types = load_packed_state('model.jxc')")
```

## Summary report

Print a final summary:

```
# jaxcross Quickstart Report

- Data: <file_path> (<N> rows x <M> columns)
- Column types: <type counts>
- Model: <V> views, <K> total clusters
- Log-joint: <value>

## Top Dependencies
<table>

## Top Anomalies
<table>

## Model Structure
<view assignments>

## Next Steps
- Run `/jxc-model` for production-grade training (multi-chain, convergence monitoring)
- Run `/jxc-anomaly` for full anomaly detection pipeline
- Run `/jxc-discover` for detailed dependency analysis
- Run `/jxc-impute` if your data has missing values
```

## Common Pitfalls

- **First JIT compilation is slow**: The first `packed_gibbs_sweep` call compiles the XLA kernel. Subsequent calls are fast. On a GTX 1650, expect 30-60s for compilation.
- **`initialize()` returns `InitResult`**: Access `.state` to get the `CrossCatState`. When `n_chains > 1`, `.state` is a list.
- **ORDINAL and CYCLIC are never auto-detected**: `guess_column_types()` only detects CONTINUOUS, CATEGORICAL, BINARY. Set ordinal/cyclic manually.
- **NaN values are fine**: jaxcross handles missing data natively. Don't fill NaN before modeling.

See [column-type-guide.md](references/column-type-guide.md) for the full ColumnType decision tree.
