# Quick Start Guide

This guide walks through a complete analysis using jax-crosscat: generating synthetic data, running inference, and querying the posterior.

## Setup

```bash
uv pip install jax-crosscat

# With GPU support (NVIDIA CUDA 13)
uv pip install "jax-crosscat[gpu]"
```

```python
import jax
import jax.numpy as jnp
from crosscat import initialize, gibbs_sweep, log_joint
from crosscat.types import ColumnType
```

## 1. Prepare Your Data

jax-crosscat expects a 2D JAX array where rows are observations and columns are features. Each column has a type:

```python
# Example: employee dataset
# Columns: salary (continuous), years_exp (continuous),
#           department (categorical), is_remote (binary)
column_types = [
    ColumnType.CONTINUOUS,    # salary
    ColumnType.CONTINUOUS,    # years_experience
    ColumnType.CATEGORICAL,   # department (0=eng, 1=sales, 2=hr)
    ColumnType.BINARY,        # is_remote (0 or 1)
]

# data shape: (n_employees, 4)
data = jnp.array([
    [75000, 3.0, 0, 1],
    [120000, 8.0, 0, 0],
    [65000, 1.0, 1, 1],
    # ... more rows
])
```

**Column type guide:**

| Type | Values | Example |
|------|--------|---------|
| `CONTINUOUS` | Any float | Salary, temperature, height |
| `CATEGORICAL` | Non-negative integers | Department ID, color code |
| `BINARY` | 0 or 1 | Yes/no, true/false flags |
| `ORDINAL` | Ordered integers | Rating (1-5), education level |
| `CYCLIC` | Floats in [0, 2*pi) | Wind direction, time of day |

**Missing data**: Use `jnp.nan` for missing values. They are handled transparently.

## 2. Generate Synthetic Data (for Testing)

```python
from crosscat.synthetic import generate_crosscat_data

key = jax.random.key(42)
result = generate_crosscat_data(
    key,
    n_rows=200,
    column_types=[
        ColumnType.CONTINUOUS, ColumnType.CONTINUOUS,
        ColumnType.CATEGORICAL, ColumnType.BINARY,
    ],
    n_views=2,       # ground truth: 2 column groups
    n_clusters=2,    # ground truth: 2 row clusters per view
)

data = result["data"]
true_col_assigns = result["true_column_assignments"]
true_row_assigns = result["true_row_assignments"]
```

## 3. Initialize the Model

```python
key = jax.random.key(0)

# Single chain
state = initialize(key, data, column_types)

# Multi-chain (recommended — pick best by log_joint)
states = initialize(key, data, column_types, n_chains=4)
```

**Initialization modes:**
- `"from_the_prior"` (default) — sample from CRP priors
- `"together"` — all columns in one view (conservative start)
- `"apart"` — each column in its own view (exploratory start)

## 4. Run Inference

```python
key, subkey = jax.random.split(key)
state = gibbs_sweep(subkey, state, data, n_sweeps=100)

print(f"Log joint: {log_joint(state, data):.2f}")
print(f"Views: {state.n_views}")
print(f"Column assignments: {state.column_assignments}")
```

For multi-chain, run each chain and pick the best:

```python
final_states = []
for i, s in enumerate(states):
    k = jax.random.fold_in(key, i)
    s = gibbs_sweep(k, s, data, n_sweeps=100)
    final_states.append(s)

# Select best chain
best = max(final_states, key=lambda s: float(log_joint(s, data)))
```

## 5. Query the Posterior

### Conditional Sampling

"What salary would we expect given 5 years of experience?"

```python
from crosscat.inference import predictive_sample

key, subkey = jax.random.split(key)
samples = predictive_sample(
    subkey, best, data,
    query_cols=[0],                          # salary
    condition_cols=[1],                       # years_experience
    condition_vals=jnp.array([5.0]),
    n_samples=1000,
)
print(f"Expected salary: {jnp.median(samples[:, 0]):.0f}")
print(f"90% CI: [{jnp.percentile(samples[:, 0], 5):.0f}, "
      f"{jnp.percentile(samples[:, 0], 95):.0f}]")
```

### Anomaly Detection

"Is this employee unusual?"

```python
from crosscat.inference import predictive_anomalousness

key, subkey = jax.random.split(key)
score = predictive_anomalousness(subkey, best, data, query_row=42)
print(f"Anomaly score: {score:.3f}")  # 0=normal, 1=anomalous
```

### Column Dependencies

"Which columns are related?"

```python
from crosscat.inference import dependence_probability, dependence_matrix

# Pairwise: probability that two columns share a view
dp = dependence_probability(final_states, col_i=0, col_j=1)
print(f"P(salary ~ experience): {dp:.3f}")  # likely ~1.0

# Full Z-matrix: all pairwise dependency probabilities
z = dependence_matrix(final_states)
print(z)  # (n_cols, n_cols) matrix, diagonal = 1.0
```

For a continuous measure of dependency strength, use mutual information:

```python
from crosscat.inference import mutual_information

mi, linfoot = mutual_information(final_states, col_i=0, col_j=1)
print(f"MI(salary, experience): {mi:.3f}")
print(f"Linfoot correlation: {linfoot:.3f}")

mi_cross, _ = mutual_information(final_states, col_i=0, col_j=3)
print(f"MI(salary, is_remote): {mi_cross:.3f}")  # likely ~0
```

### Imputation

"Fill in missing values with confidence:"

```python
from crosscat.inference import impute_and_confidence

key, subkey = jax.random.split(key)
value, confidence = impute_and_confidence(
    subkey, best, data, query_col=0,
    condition_cols=[1, 2],
    condition_vals=jnp.array([5.0, 0.0]),
)
print(f"Imputed salary: {value:.0f} (confidence: {confidence:.2f})")
```

## 6. Evaluate Recovery (with Synthetic Data)

```python
from crosscat.diagnostics import column_partition_ari, row_partition_ari

col_ari = column_partition_ari(best, true_col_assigns)
print(f"Column partition ARI: {col_ari:.3f}")  # 1.0 = perfect

for v in range(best.n_views):
    for true_assigns in true_row_assigns:
        ari = row_partition_ari(best, v, true_assigns)
        print(f"  View {v} row ARI: {ari:.3f}")
```

## 7. Enforce Constraints

```python
from crosscat.constraints import ensure_col_dep_constraints

# Force salary and experience into the same view
key, subkey = jax.random.split(key)
constrained = ensure_col_dep_constraints(
    subkey, best, data,
    constraints=[(0, 1, True)],   # dependent
    max_rejections=100,
)
```

## 8. GPU-Accelerated Workflow (Packed State)

For large datasets or GPU execution, use the packed representation which enables full JIT compilation:

```python
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state

# Convert to packed state (fixed-size padded arrays)
packed = pack_state(state, max_views=16, max_clusters=32)

# JIT-compiled inference — all 4 kernels per sweep
key, subkey = jax.random.split(key)
packed = packed_gibbs_sweep(subkey, packed, data, n_sweeps=100)

# Convert back to CrossCatState for queries
# Pass data= for exact suffstats fidelity (recommended)
state = unpack_state(packed, column_types, data=data)
```

You can also run packed inference queries directly without unpacking:

```python
from crosscat import (
    packed_predictive_sample,
    packed_anomaly_score,
    packed_mutual_information,
)

# Sample on packed state
key, subkey = jax.random.split(key)
samples = packed_predictive_sample(
    subkey, packed, data,
    query_cols=[0],
    n_samples=1000,
)

# Anomaly detection
key, subkey = jax.random.split(key)
score = packed_anomaly_score(subkey, packed, data, query_row=42)
```

**When to use packed state:**

- Datasets with 500+ rows or 20+ columns
- GPU/TPU execution (compile once, run many sweeps fast)
- Batch inference across many PRNG keys

**Tip**: The first call to `packed_gibbs_sweep` triggers JIT compilation (~30-60s). Subsequent calls with the same data shape are fast.

## Tips

- **More sweeps = better**: 50-200 sweeps is typical. Watch `log_joint` for convergence.
- **Multi-chain**: Always run 4+ chains and select best by `log_joint`.
- **Column types matter**: Misspecifying types (e.g., treating categorical as continuous) hurts inference.
- **Scale continuous data**: CrossCat uses data-driven hyper defaults, but extreme scales can cause issues.
- **Missing data is fine**: NaN values are handled — no imputation needed before inference.
