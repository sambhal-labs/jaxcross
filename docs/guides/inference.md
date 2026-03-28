# Running Inference

## What

Run collapsed Gibbs sampling to discover structure in your data. Each sweep iterates through all four kernels: row assignments, column assignments, hyperparameters, and CRP concentrations.

## When to Use

After initializing a model, inference is how the model learns from data.

## Basic Usage

=== "Packed (recommended)"

    ```python
    from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state

    packed = pack_state(state)
    key, subkey = jax.random.split(key)
    packed = packed_gibbs_sweep(subkey, packed, data, n_sweeps=100)
    state = unpack_state(packed, col_types, data=data)
    ```

=== "Unpacked (simple, slow)"

    ```python
    from crosscat import gibbs_sweep

    key, subkey = jax.random.split(key)
    state = gibbs_sweep(subkey, state, data, n_sweeps=100)
    ```

!!! warning "Always prefer the packed path"
    The unpacked path uses Python for-loops and is 10-100x slower. See [GPU Acceleration](gpu-packed.md).

## How Many Sweeps?

- **50-100 sweeps**: Typical for small-medium datasets
- **100-200 sweeps**: For large or complex data
- **Watch `log_joint`** for convergence — it should plateau

```python
from crosscat import log_joint

for batch in range(10):
    key, subkey = jax.random.split(key)
    packed = packed_gibbs_sweep(subkey, packed, data, n_sweeps=10)
    state = unpack_state(packed, col_types, data=data)
    score = log_joint(state, data)
    print(f"Sweep {(batch+1)*10}: log_joint={score:.1f}, views={state.n_views}")
```

## Kernel Selection

The **unpacked** `gibbs_sweep` supports selecting which kernels to run:

```python
# Unpacked path: run a subset of kernels
state = gibbs_sweep(key, state, data, n_sweeps=10,
                    kernels=("row_assignments",))
```

The **packed** path always runs all 4 kernels per sweep. To run individual kernels selectively, call the sub-kernels directly:

```python
from crosscat.packed.kernels import (
    packed_transition_row_assignments,
    packed_transition_column_hypers,
    packed_transition_crp_alphas,
)

# Only resample row assignments
packed = packed_transition_row_assignments(key, packed, data)

# Or use packed_gibbs_step for all 4 kernels once (used by constraints)
from crosscat.packed.kernels import packed_gibbs_step
packed = packed_gibbs_step(key, packed, data)
```

| Kernel | What It Does | Cost |
|--------|-------------|------|
| `row_assignments` | Move rows between clusters | Highest (bottleneck) |
| `column_assignments` | Move columns between views | Medium |
| `column_hypers` | Update hyperparameters | Low |
| `crp_alphas` | Update CRP concentrations | Low |

## Two Sweep Modes

| Mode | Function | Use Case |
|------|----------|----------|
| `packed_gibbs_sweep` | Uses `lax.scan` — one large compiled kernel | Production: multi-sweep batch inference |
| `packed_gibbs_step` | Calls 4 independent `@jax.jit` sub-kernels | Interactive: constraint enforcement, debugging |

**When to choose which:** `packed_gibbs_sweep` compiles the entire multi-sweep loop into a single XLA program — maximum throughput, but the first compilation is slower. `packed_gibbs_step` compiles each sub-kernel independently (4 smaller compilations), so it starts faster and allows inspecting intermediate state between kernels. Use `packed_gibbs_step` when you need to check constraints after each sweep or want faster iteration during development.

## Tips

- **More sweeps = better** but with diminishing returns after convergence
- **Multi-chain**: Run 4+ chains from different initializations. See [Multi-Chain Inference](multi-chain.md)
- **Checkpointing**: Save progress during long runs. See [Serialization](serialization.md)
- **JIT warmup**: First sweep is slow due to compilation. Use [XLA caching](xla-cache.md) to skip on re-runs

## API Reference

- [`packed_gibbs_sweep`](../api/packed-kernels.md#packed_gibbs_sweep)
- [`packed_gibbs_step`](../api/packed-kernels.md#packed_gibbs_step)
- [`gibbs_sweep`](../api/gibbs.md#gibbs_sweep)
