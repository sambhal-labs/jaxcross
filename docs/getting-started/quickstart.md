# Quick Start

This guide walks through a complete analysis: loading CSV data, running inference, and querying the posterior. Two paths — pick yours.

## 60-Second Path

The minimal pipeline on synthetic data, so you can verify your install and see a query work end-to-end.

```python
import jax
import jax.numpy as jnp
from crosscat import initialize, dependence_matrix
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat.types import ColumnType

key = jax.random.key(0)
data = jax.random.normal(key, (200, 6)).astype(jnp.float32)
col_types = [ColumnType.CONTINUOUS] * 6

result = initialize(key, data, col_types)
packed = pack_state(result.state)
packed = packed_gibbs_sweep(jax.random.key(1), packed, data, n_sweeps=50)
state = unpack_state(packed, col_types, data=data)

z = dependence_matrix([state])  # 6x6 probability of column dependence
print(z)
```

If `z` prints as a 6x6 array with values in `[0, 1]` — you're ready for the full walkthrough below.

## 10-Minute Path

The rest of this page is the full CSV-in → queries-out pipeline with multi-chain inference, convergence checks, and production-grade query patterns. Read it linearly for a complete mental model.

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
result = initialize(key, data, col_types)
state = result.state

# Multi-chain (recommended — pick best by log_joint)
result = initialize(key, data, col_types, n_chains=4)
states = result.state
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

For multi-chain inference, run all chains in parallel and pick the best:

```python
from crosscat.packed import multi_chain_packed_gibbs_sweep, select_best_chain, unbatch_packed_states

packed_list = [pack_state(s) for s in states]
batched, scores = multi_chain_packed_gibbs_sweep(key, packed_list, data, n_sweeps=100)
best = select_best_chain(batched, scores)
state = unpack_state(best, col_types, data=data)
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
print(f"Anomaly score: {score:.3f}")  # closer to 0=normal, closer to 1=anomalous
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
