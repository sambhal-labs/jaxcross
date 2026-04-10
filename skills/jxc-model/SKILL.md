---
name: jxc-model
description: Train a production-grade jaxcross model with automatic strategy selection (direct/subsample-anneal/minibatch based on dataset size), multi-chain inference, Gelman-Rubin convergence monitoring, periodic checkpointing, and best-chain selection. Use for serious modeling after data preparation.
version: "1.0.0"
license: Apache-2.0
---

# jaxcross Model Training

Train a production-quality CrossCat model with convergence guarantees.

Usage: `/jxc-model <file_path> [--chains N] [--sweeps N]`

Examples:
- `/jxc-model data/prepared.arrow`
- `/jxc-model data/prepared.arrow --chains 4 --sweeps 400`
- `/jxc-model data/large_dataset.arrow` (auto-selects scaling strategy)

## Step 1: Load data and assess

```python
import jax
import jax.numpy as jnp
from crosscat import initialize, estimate_packed_memory, suggest_max_clusters
from crosscat.data_utils import load_data
from crosscat.types import ColumnType

# Load data
data, col_names, col_types = load_data("<file_path>")
# Or: data, col_names = read_csv(path); col_types = guess_column_types(data)

n_rows, n_cols = data.shape
print(f"Dataset: {n_rows:,} rows x {n_cols} columns")
print(f"Column types: {[ct.name for ct in col_types]}")

# Estimate resource requirements
max_clusters = suggest_max_clusters(n_rows)
print(f"Suggested max_clusters: {max_clusters}")

# Check GPU
devices = jax.devices()
print(f"JAX devices: {devices}")
```

## Step 2: Select training strategy

Based on dataset size, choose the optimal strategy:

| Rows | Strategy | Function | Why |
|------|----------|----------|-----|
| < 10,000 | Direct | `multi_chain_packed_gibbs_sweep` | Full data fits in GPU, JIT compiles fast |
| 10K–100K | Subsample + sweep | `subsample_anneal` + `packed_gibbs_sweep` | Progressive growth avoids poor initialization |
| > 100K | Subsample + early stopping | `subsample_anneal` + `gibbs_sweep_early_stopping` | Convergence-aware, saves compute |

```python
if n_rows < 10_000:
    strategy = "direct"
elif n_rows < 100_000:
    strategy = "subsample_anneal"
else:
    strategy = "subsample_anneal_early_stopping"

print(f"Selected strategy: {strategy}")
```

See [strategy-selection.md](references/strategy-selection.md) for the full decision tree.

## Step 3: Configure training

```python
# Training parameters
N_CHAINS = 4             # Number of independent chains
N_SWEEPS = 200           # Total sweeps (direct strategy)
DIAG_EVERY = 50          # Check convergence every N sweeps
CKPT_EVERY = 100         # Checkpoint every N sweeps
MAX_VIEWS = 16           # Max views (column groups)
MAX_CLUSTERS = max_clusters  # Max clusters per view

CHECKPOINT_DIR = "./checkpoints"
import os
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
```

## Step 4: Initialize chains

```python
key = jax.random.key(42)
result = initialize(key, data, col_types, n_chains=N_CHAINS)
states = result.state  # List of N_CHAINS CrossCatStates

print(f"Initialized {N_CHAINS} chains")
for i, s in enumerate(states):
    n_views = len(s.views)
    print(f"  Chain {i}: {n_views} views")
```

## Step 5: Pack chains

```python
from crosscat import pack_state

chains = [
    pack_state(s, max_views=MAX_VIEWS, max_clusters=MAX_CLUSTERS, data=data)
    for s in states
]

# Estimate memory
mem_bytes = estimate_packed_memory(chains[0])
print(f"Memory per chain: {mem_bytes / 1e6:.1f} MB")
print(f"Total memory ({N_CHAINS} chains): {mem_bytes * N_CHAINS / 1e6:.1f} MB")
```

## Step 6A: Direct training (< 10K rows)

```python
from crosscat import (
    multi_chain_packed_gibbs_sweep, unbatch_packed_states,
    select_best_chain, save_checkpoint,
)
from crosscat.diagnostics import gelman_rubin_rhat, effective_sample_size

log_joint_traces = [[] for _ in range(N_CHAINS)]

for sweep in range(0, N_SWEEPS, DIAG_EVERY):
    key, subkey = jax.random.split(key)
    batched, scores = multi_chain_packed_gibbs_sweep(
        subkey, chains, data, n_sweeps=DIAG_EVERY
    )
    chains = unbatch_packed_states(batched, N_CHAINS)
    
    # Track per-chain log_joint
    for i in range(N_CHAINS):
        log_joint_traces[i].append(float(scores[i]))
    
    # Convergence diagnostics
    current_sweep = sweep + DIAG_EVERY
    if len(log_joint_traces[0]) >= 4:
        traces = jnp.array(log_joint_traces)
        rhat = float(gelman_rubin_rhat(traces))
        ess = float(effective_sample_size(traces))
        print(f"Sweep {current_sweep}: Rhat={rhat:.3f} ESS={ess:.1f} "
              f"log_joints={[float(s) for s in scores]}")
        
        if rhat < 1.05 and ess > 100:
            print(f"  ✓ Converged! (Rhat < 1.05, ESS > 100)")
    else:
        print(f"Sweep {current_sweep}: log_joints={[float(s) for s in scores]}")
    
    # Checkpoint
    if current_sweep % CKPT_EVERY == 0:
        best = select_best_chain(batched, scores)
        save_checkpoint(best, CHECKPOINT_DIR, current_sweep, column_types=col_types)
        print(f"  Checkpoint saved: {CHECKPOINT_DIR}/sweep_{current_sweep}")
```

See [convergence-guide.md](references/convergence-guide.md) for interpreting Rhat and ESS values.

## Step 6B: Subsample annealing (10K–100K rows)

```python
from crosscat import subsample_anneal, packed_gibbs_sweep, pack_state
from crosscat.diagnostics import gelman_rubin_rhat

# Initialize on subsample, grow progressively
key, subkey = jax.random.split(key)
packed, data_reordered = subsample_anneal(
    subkey, data, col_types,
    initial_size=min(1000, n_rows // 10),
    growth_factor=2.0,
    sweeps_per_stage=20,
    max_views=MAX_VIEWS,
    max_clusters=MAX_CLUSTERS,
)

# Continue with full-data sweeps
print(f"Subsample annealing complete. Running full-data sweeps...")
key, subkey = jax.random.split(key)
packed = packed_gibbs_sweep(subkey, packed, data_reordered, n_sweeps=N_SWEEPS)

from crosscat import packed_log_joint
lj = float(packed_log_joint(packed, data_reordered))
print(f"Final log-joint: {lj:.1f}")

from crosscat import save_packed_state
save_packed_state(packed, "model.jxc", column_types=col_types)
```

## Step 6C: Early stopping (100K+ rows)

```python
from crosscat import subsample_anneal, gibbs_sweep_early_stopping

# Progressive initialization
key, subkey = jax.random.split(key)
packed, data_reordered = subsample_anneal(
    subkey, data, col_types,
    initial_size=1000,
    growth_factor=2.0,
    sweeps_per_stage=10,
    max_views=MAX_VIEWS,
    max_clusters=MAX_CLUSTERS,
)

# Convergence-aware sweep
key, subkey = jax.random.split(key)
packed, log_joints = gibbs_sweep_early_stopping(
    subkey, packed, data_reordered,
    max_sweeps=500,
    check_interval=20,
    patience=5,
    min_improvement=0.001,
)

print(f"Stopped after {len(log_joints)} checks")
print(f"Final log-joint: {float(log_joints[-1]):.1f}")

from crosscat import save_packed_state
save_packed_state(packed, "model.jxc", column_types=col_types)
```

## Step 7: Select best chain and save

```python
# For direct training (Step 6A):
best = select_best_chain(batched, scores)

# Save final model
from crosscat import save_packed_state
save_packed_state(best, "model.jxc", column_types=col_types)
print(f"Final model saved to: model.jxc")

# Also save all chains for multi-chain queries
all_chains = unbatch_packed_states(batched, N_CHAINS)
# These can be used with packed_dependence_matrix(all_chains),
# multi_chain_predictive_probability(all_chains, ...), etc.
```

## Step 8: Validate model

```python
from crosscat import validate_state, unpack_state

state_out = unpack_state(best, col_types, data=data)
errors = validate_state(state_out)
if errors:
    print(f"WARNING: Validation issues: {errors}")
else:
    print("Model validation passed")

# Print model summary
print(f"\nFinal model summary:")
print(f"  Views: {len(state_out.views)}")
for i, view in enumerate(state_out.views):
    cols = [col_names[j] for j in view.column_indices]
    n_clusters = len(set(int(a) for a in view.row_assignments))
    print(f"  View {i}: {cols} ({n_clusters} clusters)")
```

## Output

The trained model is saved as `model.jxc` (or in `checkpoints/`). Use it with:
- `/jxc-anomaly` for anomaly detection
- `/jxc-impute` for missing data imputation
- `/jxc-discover` for dependency analysis
- `/jxc-predict` for predictive queries
- `/jxc-segment` for customer segmentation

## Common Pitfalls

- **First compilation is slow**: `multi_chain_packed_gibbs_sweep` compiles on first call (30-90s depending on data shape). Subsequent calls are fast. XLA cache persists across sessions.
- **GPU OOM**: If you get out-of-memory, reduce `MAX_CLUSTERS` or `MAX_VIEWS`, or use subsample annealing with a smaller `initial_size`.
- **Rhat > 1.2**: Chains haven't converged. Run more sweeps or increase `N_CHAINS`.
- **`max_cols_per_view` overflow**: If a warning appears, set `max_cols_per_view=n_cols` when packing (this is the default).
- **`initialize()` returns `InitResult`**: Access `.state` for the state(s). With `n_chains > 1`, `.state` is a list.

See [gpu-memory-guide.md](references/gpu-memory-guide.md) for memory estimation and tuning.
