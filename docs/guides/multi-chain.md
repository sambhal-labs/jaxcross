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
states = initialize(key, data, col_types, n_chains=4)
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

## Run Chains Sequentially

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

Several queries accept lists of states and average across them:

```python
from crosscat import dependence_matrix, mutual_information, row_similarity

# Z-matrix averaged across chains
z = dependence_matrix(final_states)

# MI averaged across chains
mi, linfoot = mutual_information(final_states, col_i=0, col_j=1)

# Row similarity averaged across chains
sim = row_similarity(final_states, row_a=10, row_b=20)
```

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

## API Reference

- [`multi_chain_packed_gibbs_sweep`](../api/packed-kernels.md#multi_chain_packed_gibbs_sweep)
- [`batch_packed_states`](../api/packed-state.md#batch_packed_states)
- [`select_best_chain`](../api/packed-state.md#select_best_chain)
