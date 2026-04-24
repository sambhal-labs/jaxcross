# Multi-Chain Inference

## What

Run multiple independent Gibbs chains from different initializations and combine results for more robust inference. This avoids getting stuck in local optima.

## When to Use

- Any serious analysis (always recommended)
- Complex datasets where structure discovery is uncertain
- When you need reliable dependence probabilities

## Initialize Multiple Chains

```python
import jax
from crosscat import initialize

key = jax.random.key(42)
result = initialize(key, data, col_types, n_chains=4)
states = result.state
# Returns a list of 4 CrossCatState objects
```

## Run Chains in Parallel (GPU)

```python
from crosscat import pack_state, multi_chain_packed_gibbs_sweep, select_best_chain

packed_list = [pack_state(s) for s in states]

key, subkey = jax.random.split(key)
batched, scores = multi_chain_packed_gibbs_sweep(
    subkey, packed_list, data, n_sweeps=50
)
print(f"Log-joint scores: {scores}")

# Select best chain
best = select_best_chain(batched, scores)
```

## Run Chains Sequentially (Legacy)

!!! note
    The parallel approach above is preferred. Use sequential only when debugging
    or when GPU memory is too limited for vmap-batched chains.

```python
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat import log_joint

final_states = []
for i, s in enumerate(states):
    packed = pack_state(s)
    k = jax.random.fold_in(key, i + 100)
    packed = packed_gibbs_sweep(k, packed, data, n_sweeps=100)
    s = unpack_state(packed, col_types, data=data)
    final_states.append(s)

# Select best by log-joint
best = max(final_states, key=lambda s: float(log_joint(s, data)))
```

## Multi-Chain Queries

Several queries accept lists of packed states and average across posterior samples:

```python
from crosscat import packed_dependence_matrix, packed_mutual_information

all_chains = unbatch_packed_states(batched, n_chains=4)

# Z-matrix averaged across chains (preferred — stays packed)
z = packed_dependence_matrix(all_chains)

# MI averaged across chains
mi, linfoot = packed_mutual_information(all_chains, col_types, 0, 1, rng_key=key)
```

The unpacked path (`dependence_matrix`, `mutual_information`) also accepts lists of `CrossCatState` but is slower.

## Multi-Chain Packed Wrappers

```python
from crosscat import (
    multi_chain_predictive_probability,
    multi_chain_predictive_sample,
    multi_chain_anomaly_score,
    multi_chain_impute_and_confidence,
    multi_chain_predictive_cdf,
)

# Average predictions across chains
samples = multi_chain_predictive_sample(key, packed_states, data, query_cols=[0])
score = multi_chain_anomaly_score(key, packed_states, data, query_row=42)
```

## Batching and Unbatching

```python
from crosscat import batch_packed_states, unbatch_packed_states

# Stack into batched state
batched = batch_packed_states(packed_list)

# Extract individual states
individuals = unbatch_packed_states(batched, n_chains=4)
```

## Tips

- **4 chains is a good minimum** for most analyses
- Use `initialization="apart"` for half the chains and `"together"` for the other half
- The Z-matrix is most useful with multi-chain results — single-chain gives binary values

## `multi_chain_packed_gibbs_sweep` vs `jax.pmap`

Two ways to actually run chains in parallel:

| Approach | What it does | When to use |
|----------|--------------|-------------|
| `multi_chain_packed_gibbs_sweep` | Stacks chains into a leading batch dim and `vmap`s the single-chain kernel on **one device**. | Single-GPU or multi-chain on CPU. Simplest to set up. |
| Explicit `jax.pmap` | Replicates the kernel across **multiple physical devices** (GPUs/TPU cores). One chain per device. | Multi-GPU (Kaggle 2×T4, 4×A100), or TPU v4-8. |

### Multi-GPU pmap pattern

For Kaggle's dual-T4 setup or any multi-GPU host, `jax.pmap` runs one chain per device in true parallel:

```python
import jax
from crosscat import initialize
from crosscat.packed import pack_state, packed_gibbs_sweep, select_best_chain

N_DEVICES = jax.device_count()  # 2 on Kaggle T4x2
print(f"Running {N_DEVICES} chains on {N_DEVICES} devices")

# 1. Initialize one chain per device
key = jax.random.key(42)
result = initialize(key, data, col_types, n_chains=N_DEVICES)
packed_per_chain = [pack_state(s, max_views=16, max_clusters=32) for s in result.state]

# 2. Stack into a device-sharded pytree
from crosscat.packed import batch_packed_states
batched = batch_packed_states(packed_per_chain)

# 3. Replicate data (broadcast) and split keys
keys = jax.random.split(key, N_DEVICES)
data_rep = jax.numpy.broadcast_to(data, (N_DEVICES, *data.shape))

# 4. pmap the sweep
pmap_sweep = jax.pmap(
    lambda k, p, d: packed_gibbs_sweep(k, p, d, n_sweeps=50),
    axis_name="chain",
)
batched = pmap_sweep(keys, batched, data_rep)

# 5. Score each chain and pick the best
from crosscat.packed import packed_log_joint
scores = jax.pmap(packed_log_joint)(batched, data_rep)
best = select_best_chain(batched, scores)
```

See [`benchmarks/wdi_macroeconomic_benchmark.ipynb`](https://github.com/sambhal-labs/jaxcross/blob/main/benchmarks/wdi_macroeconomic_benchmark.ipynb) for a full worked example including checkpointing and Rhat monitoring across pmap'd chains.

!!! tip "Hardware recommendation"
    Per jaxcross memory, **prefer Kaggle 2×T4 with pmap** over a single P100 for multi-chain workloads — the T4 cluster is faster on per-chain throughput *and* runs true-parallel chains.

### Gotchas

- `batch_packed_states` produces a pytree with a leading `(N_CHAINS,)` axis. `jax.pmap` requires the leading axis equal `jax.device_count()`.
- Data must be broadcast to the chains dim (`(N_DEVICES, n_rows, n_cols)`), *not* sharded — all chains see the full dataset.
- `key` must be split per-chain *before* entering the pmap.

## API Reference

- [`multi_chain_packed_gibbs_sweep`](../api/packed-kernels.md#multi_chain_packed_gibbs_sweep)
- [`batch_packed_states`](../api/packed-state.md#batch_packed_states)
- [`select_best_chain`](../api/packed-state.md#select_best_chain)
