# Gibbs Sampling

::: crosscat.gibbs
    options:
      members:
        - gibbs_sweep
      show_source: false

## Overview

Collapsed Gibbs MCMC kernels for the unpacked state path.

!!! warning "Prefer the packed path"
    The unpacked `gibbs_sweep` uses Python for-loops and is 10-100x slower than the packed equivalent. Use [`packed_gibbs_sweep`](packed-kernels.md) for production workloads.

## `gibbs_sweep`

```python
gibbs_sweep(
    rng_key, state, data, *,
    n_sweeps=1,
    kernels=("row_assignments", "column_assignments", "column_hypers", "crp_alphas")
) -> CrossCatState
```

Run full Gibbs sweeps combining all transition kernels.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rng_key` | `Array` | JAX PRNG key |
| `state` | `CrossCatState` | Current model state |
| `data` | `Array (n_rows, n_cols)` | Observation matrix |
| `n_sweeps` | `int` | Number of full iterations |
| `kernels` | `tuple[str]` | Which kernels to include per sweep |

**Available kernels:**

| Kernel | Description |
|--------|-------------|
| `"row_assignments"` | Resample row-to-cluster assignments per view |
| `"column_assignments"` | Resample column-to-view assignments (Gibbs) |
| `"column_assignments_mh"` | Resample column-to-view assignments (Metropolis-Hastings) |
| `"column_hypers"` | Grid-based Gibbs over column hyperparameters |
| `"crp_alphas"` | Grid-based sampling for CRP concentration parameters |

**Returns**: Updated `CrossCatState`.
