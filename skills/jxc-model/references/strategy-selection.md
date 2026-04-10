# Training Strategy Selection Guide

## Decision Tree

```
How many rows?
├── < 10,000 rows
│   └── Use: multi_chain_packed_gibbs_sweep (direct)
│       ├── N_SWEEPS: 200-400
│       ├── N_CHAINS: 4
│       └── Full data every sweep, JIT compiles once
│
├── 10,000 – 100,000 rows
│   └── Use: subsample_anneal + packed_gibbs_sweep
│       ├── initial_size: min(1000, n_rows // 10)
│       ├── growth_factor: 2.0
│       ├── sweeps_per_stage: 10-20
│       └── Follow with 100-200 full-data sweeps
│
└── > 100,000 rows
    └── Use: subsample_anneal + gibbs_sweep_early_stopping
        ├── initial_size: 1000
        ├── growth_factor: 2.0
        ├── sweeps_per_stage: 5-10
        ├── max_sweeps: 500
        ├── patience: 5
        └── min_improvement: 0.001
```

## Why subsample annealing?

For large datasets, initializing on the full data creates many small clusters. Starting on a subsample finds the coarse structure first, then refines as data grows. This converges faster and finds better structures.

## Minibatch Gibbs

For very large datasets where even full-data sweeps are slow:

```python
from crosscat import minibatch_gibbs_sweep

packed, log_joints = minibatch_gibbs_sweep(
    key, packed, data,
    n_sweeps=100,
    batch_size=1000,  # Update 1000 rows per sweep instead of all
)
```

Row transitions update `batch_size` random rows per sweep (O(B) instead of O(N)). Column/hyper/CRP transitions still use all data.

## Parallel Gibbs

For maximizing GPU utilization:

```python
from crosscat import parallel_gibbs_sweep

packed = parallel_gibbs_sweep(key, packed, data, n_sweeps=50)
```

Uses vmap over all rows simultaneously. Cannot create new clusters, so use alongside sequential sweeps periodically.

## Multi-GPU (pmap)

For 2+ GPUs, replace `multi_chain_packed_gibbs_sweep` with explicit pmap:

```python
import jax

n_devices = jax.device_count()
chains_per_device = N_CHAINS // n_devices

# Batch chains and replicate data
batched = batch_packed_states(chains)
data_rep = jnp.broadcast_to(data, (n_devices, *data.shape))

@jax.pmap
def pmap_sweep(key, packed, data):
    return packed_gibbs_sweep(key, packed, data, n_sweeps=50)

keys = jax.random.split(key, n_devices)
batched = pmap_sweep(keys, batched, data_rep)
```

See `benchmarks/wdi_macroeconomic_benchmark.ipynb` for the full pmap pattern.
