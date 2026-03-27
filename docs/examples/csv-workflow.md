# CSV End-to-End Workflow

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
states = initialize(key, data, col_types, n_chains=4)
print(f"Initialized {len(states)} chains")
```

## 3. Run GPU-Accelerated Inference

```python
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat import packed_log_joint
from crosscat.packed.aot_cache import enable_xla_cache

# Enable compilation caching
enable_xla_cache()

# Run each chain
final_states = []
for i, s in enumerate(states):
    packed = pack_state(s, max_views=8, max_clusters=16)
    k = jax.random.fold_in(key, i + 100)
    packed = packed_gibbs_sweep(k, packed, data, n_sweeps=100)
    s = unpack_state(packed, col_types, data=data)
    final_states.append(s)
    score = float(packed_log_joint(pack_state(s), data))
    print(f"Chain {i}: log_joint={score:.1f}, views={s.n_views}")
```

## 4. Select Best Chain

```python
from crosscat import log_joint

best = max(final_states, key=lambda s: float(log_joint(s, data)))
print(f"Best chain: {best.n_views} views")
print(f"Column assignments: {best.column_assignments}")
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
from crosscat import predictive_anomalousness

print("\nTop anomalous rows:")
scores = []
for row_id in range(data.shape[0]):
    key, subkey = jax.random.split(key)
    s = predictive_anomalousness(subkey, best, data, query_row=row_id)
    scores.append(float(s))

import numpy as np
top5 = np.argsort(scores)[-5:][::-1]
for idx in top5:
    print(f"  Row {idx}: score={scores[idx]:.3f}")
```

## 8. Imputation

```python
from crosscat import impute_and_confidence

# Find missing values and impute them
for col in range(data.shape[1]):
    nan_rows = jnp.where(jnp.isnan(data[:, col]))[0]
    for row in nan_rows[:3]:  # first 3 missing per column
        key, subkey = jax.random.split(key)
        value, conf = impute_and_confidence(subkey, best, data, query_col=col)
        print(f"  {col_names[col]}[row={int(row)}]: {value:.2f} (conf={conf:.2f})")
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
