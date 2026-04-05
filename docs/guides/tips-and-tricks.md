# Tips & Tricks

Practical advice for getting the most out of jax-crosscat.

---

## Choosing the Right Number of Sweeps

There's no universal answer, but these guidelines work well:

| Dataset | Sweeps | Reasoning |
|---------|--------|-----------|
| Quick exploration | 20–50 | Good enough to see rough structure |
| Standard analysis | 100–200 | Sufficient for most datasets |
| Publication-quality | 200–500 | Robust convergence, use multi-chain |
| Benchmarking | 100 per chain x 10 chains | Matches the original paper methodology |

**How to tell if you've run enough**: Monitor the log-joint trace. When it plateaus across multiple chains, you've converged.

```python
from crosscat import collect_diagnostics
diagnostics = collect_diagnostics(state, data)
# Plot diagnostics['log_joint'] over sweeps
```

## Multi-Chain vs. Single-Chain

**Use multi-chain for any serious analysis.** Single chains can get stuck in local modes.

```python
# Initialize 4 chains
result = initialize(key, data, col_types, n_chains=4)
states = result.state

# Run each chain independently
packed_states = []
for i, state in enumerate(states):
    packed = pack_state(state)
    packed = packed_gibbs_sweep(jax.random.key(i + 100), packed, data, n_sweeps=100)
    packed_states.append(packed)

# Select the best chain by log-joint
from crosscat.packed import batch_packed_states, select_best_chain
from crosscat.packed.kernels import packed_log_joint
import jax.numpy as jnp

batched = batch_packed_states(packed_states)
scores = jnp.array([packed_log_joint(p, data) for p in packed_states])
best = select_best_chain(batched, scores)
```

**When single-chain is fine**: Quick exploration, small datasets (<50 rows), or when you just need a rough answer.

## Memory-Efficient Inference

GPU memory is often the bottleneck. Reduce padding to fit larger datasets:

```python
packed = pack_state(state,
    max_views=5,         # Default: n_cols (wasteful for wide data)
    max_clusters=20,     # Default: n_rows (most data has <20 real clusters)
    max_categories=10,   # Default: auto (reduce if your max category is small)
)
```

**Rule of thumb**: Set `max_views` to `min(10, n_cols // 3)` and `max_clusters` to `min(30, n_rows // 5)` for moderate datasets. The model will warn if these are exceeded during inference.

## Interpreting the Z-Matrix

The dependence matrix (Z-matrix) is your most valuable output. Here's how to read it:

- **Bright blocks on the diagonal**: Groups of columns that are statistically related (they form a view)
- **Dark off-diagonal regions**: Columns that are statistically independent
- **Uniform mid-values (~0.5)**: Uncertainty — the model isn't sure whether these columns belong together. Run more sweeps or chains.

```python
z = dependence_matrix(states)

# Find the strongest dependencies
import numpy as np
for i in range(z.shape[0]):
    for j in range(i + 1, z.shape[1]):
        if z[i, j] > 0.8:
            print(f"Columns {i} and {j}: strong dependence ({z[i,j]:.2f})")
```

## Debugging Convergence Issues

If the model doesn't seem to converge:

1. **Check the log-joint trace** — is it still climbing? Run more sweeps.
2. **Use more chains** — single chains can get stuck. Try 10 chains.
3. **Check column types** — misspecified types (e.g., categorical treated as continuous) cause poor structure recovery.
4. **Reduce padding** — overly large `max_views` / `max_clusters` can slow convergence and waste memory.
5. **Scale your data** — extreme value ranges in continuous columns can cause numerical issues.

```python
# Debug: enable NaN detection
jax.config.update("jax_debug_nans", True)

# Debug: check state validity
from crosscat import validate_state
validate_state(state, data, col_types)
```

## Performance Optimization Checklist

1. **Always use the packed path** — `packed_gibbs_sweep`, not `gibbs_sweep`
2. **Enable XLA cache** — automatic on `import crosscat.packed`
3. **Pre-compile kernels** — call `compile_kernels(packed, data)` before benchmarking
4. **Right-size padding** — don't use defaults for wide datasets
5. **Use `packed_gibbs_sweep` not `packed_gibbs_step`** — the former uses `lax.scan` for maximum throughput
6. **Batch queries** — use `batch_row_similarity` instead of looping over `row_similarity`

## Column Type Selection Guide

When in doubt:

| Data Description | Type | Why |
|-----------------|------|-----|
| Money, temperature, weight | `CONTINUOUS` | Real-valued, no inherent ordering of categories |
| Country, department, color | `CATEGORICAL` | Unordered discrete values |
| Yes/no, true/false, 0/1 | `BINARY` | More efficient than `CATEGORICAL` for 2 values |
| 1-5 stars, education level | `ORDINAL` | Order matters but spacing doesn't |
| Compass direction, hour of day | `CYCLIC` | 359 degrees is close to 1 degree |

**Automatic detection** handles most cases:

```python
from crosscat import guess_column_types
col_types = guess_column_types(data)
```

Override specific columns when auto-detection gets it wrong:

```python
col_types[3] = ColumnType.ORDINAL  # Rating column misdetected as CATEGORICAL
```
