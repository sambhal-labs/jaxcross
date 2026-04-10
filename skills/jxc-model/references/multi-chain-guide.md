# Multi-Chain Inference Guide

## Why multiple chains?

CrossCat's Gibbs sampler can get stuck in local modes. Running multiple chains with different random initializations:
1. **Detects non-convergence**: if chains disagree (high Rhat), the model hasn't converged
2. **Finds better solutions**: more starting points means more chances to find good structure
3. **Enables Bayesian model averaging**: average predictions across chains for more robust estimates

## Initialization

```python
from crosscat import initialize

key = jax.random.key(42)
result = initialize(key, data, col_types, n_chains=4)
states = result.state  # List of 4 CrossCatStates
```

Each chain gets a different random initialization (different initial column-to-view and row-to-cluster assignments).

## Multi-chain sweep

```python
from crosscat import (
    pack_state, multi_chain_packed_gibbs_sweep,
    unbatch_packed_states, select_best_chain,
)

chains = [pack_state(s, max_views=16, max_clusters=32, data=data) for s in states]

key, subkey = jax.random.split(key)
batched, scores = multi_chain_packed_gibbs_sweep(
    subkey, chains, data, n_sweeps=200
)
```

`multi_chain_packed_gibbs_sweep` uses `vmap` to run all chains in parallel on the GPU.

## Best chain selection

```python
best = select_best_chain(batched, scores)
# best is the PackedCrossCatState with highest log-joint
```

Use the best chain for single-state queries (`batch_anomaly_score`, `batch_impute_column`, etc.).

## All-chains queries (Bayesian model averaging)

For queries that benefit from posterior averaging, pass all chains:

```python
all_chains = unbatch_packed_states(batched, n_chains=4)

# Structure queries — average over chains
z_matrix = packed_dependence_matrix(all_chains)  # Accepts list
mi = packed_mutual_information(all_chains, col_i=0, col_j=1)  # Accepts list

# Multi-chain wrappers — average predictions
from crosscat import multi_chain_predictive_probability
log_p = multi_chain_predictive_probability(
    all_chains, data, query_cols=[0], query_vals=jnp.array([0.5])
)
```

## When to use each

| Query type | Use best chain | Use all chains |
|-----------|----------------|----------------|
| Anomaly scores | `batch_anomaly_score(best, ...)` | `multi_chain_anomaly_score(key, all_chains, ...)` |
| Imputation | `batch_impute_column(key, best, ...)` | `multi_chain_impute_and_confidence(key, all_chains, ...)` |
| Dependence | — | `packed_dependence_matrix(all_chains)` |
| Mutual information | — | `packed_mutual_information(all_chains, ...)` |
| Classification | `batch_classify_column(key, best, ...)` | `multi_chain_classify_column(key, all_chains, ...)` |
| Prediction | `batch_predictive_probability(best, ...)` | `multi_chain_predictive_probability(all_chains, ...)` |

Rule of thumb: use all chains for posterior structure queries (dependence, MI). Use best chain for fast batch operations. Use multi-chain wrappers when robustness matters.
