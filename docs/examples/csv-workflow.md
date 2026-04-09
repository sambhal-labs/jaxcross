# CSV End-to-End Workflow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sambhal-labs/jaxcross/blob/main/notebooks/intro_tutorial.ipynb)

A complete walkthrough from raw CSV data to posterior queries, using all major features.

## 1. Load Data

```python
import jax
import jax.numpy as jnp
from crosscat import read_csv, guess_column_types
from crosscat.types import ColumnType

# Load CSV
data, col_names = read_csv("employees.csv")
print(f"Loaded: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"Columns: {col_names}")

# Auto-detect column types
col_types = guess_column_types(data)
for name, ct in zip(col_names, col_types):
    print(f"  {name}: {ct.value}")

# Override specific columns if needed
# col_types[4] = ColumnType.ORDINAL  # performance rating
```

## 2. Initialize Multi-Chain

```python
from crosscat import initialize

key = jax.random.key(42)
result = initialize(key, data, col_types, n_chains=4)
states = result.state
print(f"Initialized {len(states)} chains")
```

## 3. Run GPU-Accelerated Inference

```python
from crosscat.packed import (
    pack_state, multi_chain_packed_gibbs_sweep,
    unbatch_packed_states, select_best_chain, unpack_state,
)
from crosscat.packed.aot_cache import enable_xla_cache

# Enable compilation caching
enable_xla_cache()

# Run all chains in parallel (GPU-accelerated via vmap)
packed_list = [pack_state(s, max_views=8, max_clusters=16) for s in states]
batched, scores = multi_chain_packed_gibbs_sweep(key, packed_list, data, n_sweeps=100)
print(f"Log-joint scores: {[f'{s:.1f}' for s in scores]}")
```

## 4. Select Best Chain

```python
best = select_best_chain(batched, scores)
best_state = unpack_state(best, col_types, data=data)
print(f"Best chain: {best_state.n_views} views")
print(f"Column assignments: {best_state.column_assignments}")
```

## 5. Explore Dependencies

```python
from crosscat import dependence_matrix, mutual_information

# Z-matrix: which columns are related?
z = dependence_matrix(final_states)
print("Dependence matrix:")
for i, name_i in enumerate(col_names):
    for j, name_j in enumerate(col_names):
        if i < j and z[i, j] > 0.5:
            print(f"  {name_i} ~ {name_j}: {z[i, j]:.2f}")

# Mutual information for top pairs
for i in range(len(col_names)):
    for j in range(i + 1, len(col_names)):
        if z[i, j] > 0.5:
            mi, linfoot = mutual_information(final_states, col_i=i, col_j=j)
            print(f"  MI({col_names[i]}, {col_names[j]}): {mi:.3f}")
```

## 6. Conditional Predictions

```python
from crosscat import predictive_sample, credible_interval

key, subkey = jax.random.split(key)
samples = predictive_sample(
    subkey, best, data,
    query_cols=[0],          # predict salary
    condition_cols=[1],      # given experience
    condition_vals=jnp.array([5.0]),
    n_samples=1000,
)
print(f"Expected salary (5yr exp): {jnp.median(samples[:, 0]):.0f}")
print(f"Std: {jnp.std(samples[:, 0]):.0f}")
```

## 7. Anomaly Detection

```python
from crosscat import batch_anomaly_score
import numpy as np

packed = pack_state(best)

# Score all rows in one vectorized call (no Python loop)
key, subkey = jax.random.split(key)
scores = batch_anomaly_score(packed, data, jnp.arange(data.shape[0]))

print("\nTop anomalous rows:")
top5 = np.argsort(np.array(scores))[-5:][::-1]
for idx in top5:
    print(f"  Row {idx}: score={float(scores[idx]):.3f}")
```

## 8. Imputation

```python
from crosscat import batch_impute_column

# Impute missing values column-by-column (vectorized over rows)
for col in range(data.shape[1]):
    nan_rows = jnp.where(jnp.isnan(data[:, col]))[0]
    if len(nan_rows) > 0:
        key, subkey = jax.random.split(key)
        values, confs = batch_impute_column(
            subkey, packed, data, query_col=col, row_ids=nan_rows[:3]
        )
        for i, row in enumerate(nan_rows[:3]):
            print(f"  {col_names[col]}[row={int(row)}]: {float(values[i]):.2f} (conf={float(confs[i]):.2f})")
```

## 9. Save the Model

```python
from crosscat import save_packed_state

packed = pack_state(best)
save_packed_state(packed, "trained_model", column_types=col_types)
print("Model saved to trained_model/")
```

## 10. Load and Query Later

```python
from crosscat import load_packed_state

packed, col_types = load_packed_state("trained_model")
state = unpack_state(packed, col_types, data=data)

# Continue querying...
z = dependence_matrix([state])
```
