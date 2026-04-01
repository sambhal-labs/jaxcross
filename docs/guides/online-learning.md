# Online Learning (Row Insertion)

Add new observations to an existing model without re-running inference from scratch. New rows are assigned to clusters using the posterior predictive distribution.

## When to Use

- **Streaming data** — new records arriving incrementally (transactions, sensor readings, user events)
- **Scoring new observations** — classify or impute for a new row using an existing trained model
- **Growing datasets** — extend a model without the cost of full re-inference

## How It Works

Row insertion assigns each new row to clusters using the Chinese Restaurant Process posterior:

1. For each view, compute the posterior predictive probability of the new row under each existing cluster
2. Include the CRP prior probability of starting a new cluster
3. Sample a cluster assignment from these probabilities
4. Update sufficient statistics for the assigned cluster

This is **not** the same as running additional Gibbs sweeps — it's a single forward pass through the posterior predictive. Existing row assignments are not updated.

## Basic Usage (Unpacked)

```python
from crosscat import insert_rows
import jax
import jax.numpy as jnp

new_rows = jnp.array([
    [95000.0, 7.0, 0.0, 1.0],
    [55000.0, 1.0, 2.0, 0.0],
])

key, subkey = jax.random.split(key)
state, data = insert_rows(subkey, state, data, new_rows)
print(f"New data shape: {data.shape}")
```

## Packed Version (GPU-Accelerated)

```python
from crosscat.packed.kernels import packed_insert_rows

packed, data = packed_insert_rows(key, packed, data, new_rows)
```

The packed version triggers JIT compilation on first call, then runs at GPU speed for subsequent insertions.

## Sample and Insert

Complete a partial observation (fill missing values) and insert it in one step:

```python
from crosscat import sample_and_insert

# Row with some missing values (NaN)
partial_row = jnp.array([jnp.nan, 5.0, 0.0, jnp.nan])

state, data, completed_row = sample_and_insert(key, state, data, partial_row)
print(f"Completed: {completed_row}")
# Missing values are sampled from the posterior predictive before insertion
```

## Batch Insertion

Insert multiple rows at once. Each row is assigned independently:

```python
batch = jnp.array([
    [80000.0, 5.0, 1.0, 0.0],
    [120000.0, 12.0, 0.0, 1.0],
    [45000.0, 0.5, 3.0, 0.0],
])

packed, data = packed_insert_rows(key, packed, data, batch)
```

## When to Re-Run Full Inference

Row insertion is fast but doesn't update the global structure. Consider re-running Gibbs sweeps when:

| Scenario | Action |
|----------|--------|
| Inserted a few rows (<5% of dataset) | Insertion alone is fine |
| Inserted many rows (>10% of dataset) | Run 10-20 additional sweeps |
| Data distribution has shifted | Re-initialize and run full inference |
| Need updated column partition | Run full inference (insertion doesn't change views) |

```python
# After inserting many rows, refine the model
packed = packed_gibbs_sweep(key, packed, data, n_sweeps=20)
```

## Streaming Workflow Example

A typical streaming pattern:

```python
import jax
from crosscat import initialize
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat.packed.kernels import packed_insert_rows

# 1. Train on initial batch
state = initialize(jax.random.key(0), initial_data, col_types)
packed = pack_state(state)
packed = packed_gibbs_sweep(jax.random.key(1), packed, initial_data, n_sweeps=100)

data = initial_data

# 2. Process incoming rows
for i, new_batch in enumerate(data_stream):
    key = jax.random.key(i + 100)
    k1, k2 = jax.random.split(key)
    packed, data = packed_insert_rows(k1, packed, data, new_batch)

    # Optionally refine every N batches
    if (i + 1) % 10 == 0:
        packed = packed_gibbs_sweep(k2, packed, data, n_sweeps=10)
```

## Tips

- Row insertion uses the **current** cluster structure — it doesn't update existing row assignments
- The unpacked `insert_rows` is fast (pure Python, no JIT needed). The packed `packed_insert_rows` triggers JIT on first call but is faster for subsequent batch insertions
- New rows can contain NaN values — they are handled transparently during cluster assignment
- If the new data has different value ranges than the training data, consider re-running hyperparameter transitions

## API Reference

- [`insert_rows`](../api/model.md#insert_rows)
- [`packed_insert_rows`](../api/packed-kernels.md#packed_insert_rows)
- [`sample_and_insert`](../api/inference.md#sample_and_insert)
