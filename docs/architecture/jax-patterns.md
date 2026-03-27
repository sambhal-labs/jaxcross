# JAX Design Patterns

Key JAX idioms used throughout jax-crosscat.

## Deterministic RNG

All randomness flows through `jax.random.key()` and `jax.random.split()`. Every function receives a key and splits it for sub-operations. This ensures exact reproducibility.

```python
key = jax.random.key(42)
key, subkey = jax.random.split(key)
state = initialize(subkey, data, col_types)

key, subkey = jax.random.split(key)
state = gibbs_sweep(subkey, state, data)
```

## NaN Transparency

Missing data is represented as `NaN`. All sufficient statistic computations filter NaN before accumulation:

```python
clean = data[~jnp.isnan(data)]
count = clean.shape[0]
sum_x = jnp.sum(clean)
```

Posterior predictive queries skip NaN conditioning values. This means inference works seamlessly with incomplete data.

## Immutable State

All JAX arrays are immutable. Operations return new arrays:

```python
# This creates a new array — doesn't modify in-place
row_assignments = row_assignments.at[i].set(new_cluster)
```

`CrossCatState` fields are replaced wholesale when updated. The `PackedCrossCatState` dataclass follows the same pattern — all operations return new state objects.

## `lax.scan` for Sequential Loops

JAX cannot JIT Python `for` loops over traced values. We use `jax.lax.scan` instead:

```python
# Sequential loop over n_sweeps
def step(carry, _):
    key, packed = carry
    key, subkey = jax.random.split(key)
    packed = _one_sweep(subkey, packed, data)
    return (key, packed), None

(_, packed), _ = jax.lax.scan(step, (key, packed), None, length=n_sweeps)
```

## `vmap` for Parallel Operations

Vectorize over independent dimensions instead of looping:

```python
# Score all clusters in parallel
scores = jax.vmap(score_one_cluster, in_axes=(None, 0))(row_data, cluster_suffstats)

# Score all columns in parallel
col_scores = jax.vmap(unified_posterior_predictive_logp)(x_vals, col_type_ids, ...)
```

## `jnp.where` for Type Dispatch

Instead of Python `if/else` on column types (which breaks JIT), use `jnp.where`:

```python
result = jnp.where(
    col_type_id == CONTINUOUS_ID,
    normal_gamma_logp(x, ...),
    jnp.where(
        col_type_id == CATEGORICAL_ID,
        dirichlet_categorical_logp(x, ...),
        ...
    )
)
```

This computes all branches but selects the correct result — a necessary trade-off for JIT compatibility.

## Numerical Stability

The `LOG_EPS = 1e-30` constant from `crosscat.types` is used throughout for underflow protection:

```python
from crosscat.types import LOG_EPS

log_p = jnp.log(probability + LOG_EPS)
```
