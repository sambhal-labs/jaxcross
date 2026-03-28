# Packed Components

::: crosscat.packed.components
    options:
      members:
        - unified_log_marginal
        - unified_posterior_predictive_logp
        - unified_sample_posterior_predictive
      show_source: false

## Overview

Unified scoring functions that dispatch across all 5 column types using `jnp.where`. These are the JIT-compatible equivalents of the per-type component model methods.

## `unified_log_marginal`

```python
unified_log_marginal(
    type_id, count, sum_x, sum_x_sq, cat_counts,
    sum_sin, sum_cos, mu, r, s, nu,
    dir_alpha, alpha, beta,
    kappa, vm_a, vm_mu, cutpoints
) -> Array
```

Compute log marginal likelihood for any column type. Uses `jnp.where` to select the correct formula based on `type_id`.

**Returns**: Scalar log probability.

## `unified_posterior_predictive_logp`

```python
unified_posterior_predictive_logp(
    x, type_id, count, sum_x, sum_x_sq, cat_counts,
    sum_sin, sum_cos, mu, r, s, nu,
    dir_alpha, alpha, beta,
    kappa, vm_a, vm_mu, cutpoints
) -> Array
```

Compute posterior predictive log probability for a new observation. Dispatches to the correct component model based on type ID.

**Returns**: Scalar log probability.

## `unified_sample_posterior_predictive`

```python
unified_sample_posterior_predictive(
    rng_key, type_id, count, sum_x, sum_x_sq, cat_counts,
    sum_sin, sum_cos, mu, r, s, nu,
    dir_alpha, alpha, beta,
    kappa, vm_a, vm_mu, cutpoints
) -> Array
```

Draw a sample from the posterior predictive distribution for any column type.

**Returns**: Scalar sample value.

## Type-Specialized Batch Functions

For homogeneous-type views, these batch functions skip the 5-way `jnp.where` dispatch:

- `batch_bb_posterior_predictive_logp` — BetaBernoulli batch scoring
- `batch_ng_posterior_predictive_logp` — NormalGamma batch scoring
- `batch_dc_posterior_predictive_logp` — DirichletCategorical batch scoring

These are used automatically by the row scoring kernel when all columns in a view share the same type.

!!! note "Ordinal and Cyclic types"
    There are no type-specialized batch functions for `OrderedLogistic` or `VonMises`. Ordinal columns always use the general `unified_posterior_predictive_logp` dispatch path (the grid integration makes batch specialization impractical). Cyclic columns also use the general path.
