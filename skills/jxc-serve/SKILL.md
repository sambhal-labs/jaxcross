---
name: jxc-serve
description: Deploy a trained jaxcross model for online inference. Loads from checkpoint, pre-compiles XLA kernels, supports online row insertion with incremental sweeps, and exposes query functions for production serving. Use after /jxc-model to set up a serving pipeline.
version: "1.0.0"
license: Apache-2.0
---

# Model Serving

Deploy a trained CrossCat model for production inference.

Usage: `/jxc-serve [--model model.jxc] [--data data.arrow]`

## Step 1: Load model and verify

```python
import jax
import jax.numpy as jnp
import os

from crosscat import (
    load_packed_state, load_latest_checkpoint,
    packed_log_joint, estimate_packed_memory,
)
from crosscat.data_utils import load_data

# Load from file or latest checkpoint
model_path = "model.jxc"
if os.path.exists(model_path):
    packed, col_types = load_packed_state(model_path)
else:
    packed, col_types = load_latest_checkpoint("./checkpoints")

data, col_names, _ = load_data("data/prepared.arrow")

# Verify
lj = float(packed_log_joint(packed, data))
mem = estimate_packed_memory(packed)
devices = jax.devices()

print(f"Model loaded:")
print(f"  Log-joint: {lj:.1f}")
print(f"  Memory: {mem / 1e6:.1f} MB")
print(f"  Data: {data.shape[0]} rows x {data.shape[1]} cols")
print(f"  Device: {devices}")
```

## Step 2: Pre-compile XLA kernels

```python
from crosscat.packed.aot_cache import compile_kernels

# Pre-compile all sub-kernels for this data shape
# This avoids JIT compilation latency on first query
print("Pre-compiling XLA kernels (may take 30-60s)...")
compile_kernels(packed, data)
print("Compilation complete. Subsequent calls will be fast.")
```

XLA persistent cache is auto-enabled when `crosscat.packed` is imported. Compiled kernels persist across Python sessions for the same data shape.

## Step 3: Online inference (new data insertion)

When new data arrives, insert it into the model and run incremental sweeps:

```python
from crosscat import packed_insert_rows, packed_gibbs_sweep

def ingest_new_rows(key, packed, data, new_rows):
    """Insert new rows and update the model with incremental sweeps."""
    # Insert new rows (CRP predictive assignment)
    packed, data_updated = packed_insert_rows(key, packed, data, new_rows)
    
    # Run a few sweeps to incorporate new data
    key, subkey = jax.random.split(key)
    packed = packed_gibbs_sweep(subkey, packed, data_updated, n_sweeps=5)
    
    return packed, data_updated

# Example: insert 10 new rows
key = jax.random.key(42)
new_rows = jnp.array(new_data_batch, dtype=jnp.float32)  # Shape: (n_new, n_cols)
packed, data = ingest_new_rows(key, packed, data, new_rows)
print(f"Data now has {data.shape[0]} rows")
```

See [online-inference-guide.md](references/online-inference-guide.md) for batch insertion patterns.

## Step 4: Expose query functions

Set up callable functions for production use:

```python
from crosscat import (
    batch_anomaly_score,
    batch_impute_column,
    batch_classify_column,
    batch_predictive_probability,
    batch_credible_interval,
    packed_predictive_probability,
)

# ── Anomaly scoring ─────────────────────────────────
def score_anomalies(row_ids):
    """Score rows by anomalousness. Returns array of scores."""
    return batch_anomaly_score(packed, data, jnp.array(row_ids))

# ── Imputation ──────────────────────────────────────
def impute_missing(key, col_name, row_ids):
    """Impute missing values for a column. Returns (values, confidences)."""
    col_idx = col_names.index(col_name)
    return batch_impute_column(key, packed, data, query_col=col_idx, row_ids=jnp.array(row_ids))

# ── Classification ──────────────────────────────────
def classify(key, target_col_name, row_ids):
    """Classify rows for a target column. Returns predictions."""
    col_idx = col_names.index(target_col_name)
    return batch_classify_column(key, packed, data, query_col=col_idx, row_ids=jnp.array(row_ids))

# ── Conditional prediction ──────────────────────────
def predict_given(query_col_name, query_val, condition_dict):
    """P(query_col=val | conditions). Returns log probability."""
    query_col = jnp.array([col_names.index(query_col_name)])
    query_val = jnp.array([query_val])
    cond_cols = jnp.array([col_names.index(k) for k in condition_dict])
    cond_vals = jnp.array(list(condition_dict.values()))
    return packed_predictive_probability(
        packed, data, query_cols=query_col, query_vals=query_val,
        condition_cols=cond_cols, condition_vals=cond_vals,
    )
```

## Step 5: Health monitoring

```python
from crosscat import packed_log_joint

baseline_lj = float(packed_log_joint(packed, data))

def check_model_health(packed, data, baseline_lj):
    """Check if model needs retraining."""
    current_lj = float(packed_log_joint(packed, data))
    degradation = (baseline_lj - current_lj) / abs(baseline_lj)
    
    if degradation > 0.1:
        print(f"WARNING: Log-joint degraded by {degradation:.1%}. Consider retraining.")
        return False
    print(f"Model healthy. Log-joint: {current_lj:.1f} (baseline: {baseline_lj:.1f})")
    return True
```

## Step 6: Periodic checkpointing

```python
from crosscat import save_packed_state, save_checkpoint

# Save after ingesting new data
save_packed_state(packed, "model_serving.jxc", column_types=col_types)

# Or use timestamped checkpoints
import time
sweep_id = int(time.time())
save_checkpoint(packed, "./serving_checkpoints", sweep_id, column_types=col_types)
```

## Common Pitfalls

- **XLA cache**: First call after loading compiles kernels. Use `compile_kernels()` to pre-warm the cache on startup.
- **Data must match**: New rows must have the same number of columns and the same column types as training data.
- **Incremental sweeps are not full retraining**: After inserting many new rows (>20% of original data), consider full retraining with `/jxc-model`.
- **Memory growth**: `packed_insert_rows` grows the data array. Monitor memory with `estimate_packed_memory()`.
- **Thread safety**: JAX operations are not thread-safe by default. Use a lock or queue for concurrent requests.

See [serving-template.md](references/serving-template.md) for a complete production serving script.
