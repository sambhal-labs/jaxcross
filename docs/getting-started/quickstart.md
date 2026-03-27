# Quick Start

This guide walks through a complete analysis: loading CSV data, running inference, and querying the posterior.

## 1. Load Your Data

```python
import jax
import jax.numpy as jnp
from crosscat import initialize, read_csv, guess_column_types
from crosscat.types import ColumnType

# Load a CSV file
data, col_names = read_csv("employees.csv")
print(f"Loaded {data.shape[0]} rows, {data.shape[1]} columns")
print(f"Columns: {col_names}")

# Auto-detect column types
col_types = guess_column_types(data)
for name, ct in zip(col_names, col_types):
    print(f"  {name}: {ct.value}")
```

You can also specify column types manually for full control:

```python
col_types = [
    ColumnType.CONTINUOUS,    # salary
    ColumnType.CONTINUOUS,    # years_experience
    ColumnType.CATEGORICAL,   # department (0=eng, 1=sales, 2=hr)
    ColumnType.BINARY,        # is_remote (0 or 1)
]
```

!!! tip "Column type guide"

    | Type | Values | Example |
    |------|--------|---------|
    | `CONTINUOUS` | Any float | Salary, temperature, height |
    | `CATEGORICAL` | Non-negative integers | Department ID, color code |
    | `BINARY` | 0 or 1 | Yes/no, true/false flags |
    | `ORDINAL` | Ordered integers | Rating (1-5), education level |
    | `CYCLIC` | Floats in [0, 2*pi) | Wind direction, time of day |

    Missing data: use `jnp.nan` — handled transparently.

## 2. Initialize the Model

```python
key = jax.random.key(42)

# Single chain
state = initialize(key, data, col_types)

# Multi-chain (recommended — pick best by log_joint)
states = initialize(key, data, col_types, n_chains=4)
```

**Initialization modes:**

- `"from_the_prior"` (default) — sample from CRP priors
- `"together"` — all columns in one view (conservative start)
- `"apart"` — each column in its own view (exploratory start)

## 3. Run Inference

For best performance, use the packed (GPU-accelerated) path:

```python
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat import log_joint

# Convert to JIT-compatible packed state
packed = pack_state(state, max_views=16, max_clusters=32)

# Run 100 Gibbs sweeps (JIT-compiled)
key, subkey = jax.random.split(key)
packed = packed_gibbs_sweep(subkey, packed, data, n_sweeps=100)

# Convert back to query-friendly state
state = unpack_state(packed, col_types, data=data)

print(f"Log joint: {log_joint(state, data):.2f}")
print(f"Discovered {state.n_views} views")
```

!!! info "First-run JIT compilation"
    The first call to `packed_gibbs_sweep` triggers JAX compilation (~20-60s depending on data size). Subsequent calls with the same data shape are fast. Enable [XLA caching](../guides/xla-cache.md) to skip recompilation across sessions.

For multi-chain inference, run each chain and pick the best:

```python
best_state = None
best_score = -float('inf')
for i, s in enumerate(states):
    packed = pack_state(s)
    k = jax.random.fold_in(key, i + 100)
    packed = packed_gibbs_sweep(k, packed, data, n_sweeps=100)
    s = unpack_state(packed, col_types, data=data)
    score = float(log_joint(s, data))
    if score > best_score:
        best_score = score
        best_state = s

state = best_state
```

## 4. Query the Posterior

### Conditional Sampling

*"What salary would we expect given 5 years of experience?"*

```python
from crosscat import predictive_sample

key, subkey = jax.random.split(key)
samples = predictive_sample(
    subkey, state, data,
    query_cols=[0],                      # salary
    condition_cols=[1],                   # years_experience
    condition_vals=jnp.array([5.0]),
    n_samples=1000,
)
print(f"Expected salary: {jnp.median(samples[:, 0]):.0f}")
print(f"90% CI: [{jnp.percentile(samples[:, 0], 5):.0f}, "
      f"{jnp.percentile(samples[:, 0], 95):.0f}]")
```

### Anomaly Detection

*"Is this employee unusual?"*

```python
from crosscat import predictive_anomalousness

key, subkey = jax.random.split(key)
score = predictive_anomalousness(subkey, state, data, query_row=42)
print(f"Anomaly score: {score:.3f}")  # 0=normal, 1=anomalous
```

### Dependence Discovery

*"Which columns are related?"*

```python
from crosscat import dependence_probability, dependence_matrix

# Pairwise probability that two columns share a view
dp = dependence_probability([state], col_i=0, col_j=1)
print(f"P(salary ~ experience): {dp:.3f}")

# Full Z-matrix: all pairwise dependencies
z = dependence_matrix([state])
print(z)  # (n_cols, n_cols) matrix, diagonal = 1.0
```

### Imputation

*"Fill in a missing value with confidence:"*

```python
from crosscat import impute_and_confidence

key, subkey = jax.random.split(key)
value, confidence = impute_and_confidence(
    subkey, state, data, query_col=0,
    condition_cols=[1, 2],
    condition_vals=jnp.array([5.0, 0.0]),
)
print(f"Imputed salary: {value:.0f} (confidence: {confidence:.2f})")
```

## 5. Save the Model

```python
from crosscat import save_packed_state, load_packed_state

# Save
packed = pack_state(state)
save_packed_state(packed, "my_model", column_types=col_types)

# Load later
packed, col_types = load_packed_state("my_model")
state = unpack_state(packed, col_types, data=data)
```

## Next Steps

- **[Feature Guides](../guides/index.md)** — deep dives into every feature
- **[API Reference](../api/index.md)** — complete function documentation
- **[Multi-Chain Inference](../guides/multi-chain.md)** — parallel chains for robust results
- **[GPU Acceleration](../guides/gpu-packed.md)** — packed state and JIT compilation
- **[Examples](../examples/csv-workflow.md)** — full end-to-end workflows

## Tips

- **More sweeps = better**: 50-200 sweeps is typical. Watch `log_joint` for convergence.
- **Multi-chain**: Always run 4+ chains and select best by `log_joint`.
- **Column types matter**: Misspecifying types (e.g., treating categorical as continuous) hurts inference.
- **Scale continuous data**: CrossCat uses data-driven hyper defaults, but extreme scales can cause issues.
- **Missing data is fine**: NaN values are handled — no imputation needed before inference.
