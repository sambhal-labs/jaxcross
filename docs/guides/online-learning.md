# Online Learning (Row Insertion)

## What

Add new observations to an existing model without re-running inference from scratch. New rows are assigned to clusters using the posterior predictive.

## When to Use

- Streaming data that arrives incrementally
- Scoring new observations against an existing model
- Growing datasets without re-inference

## Basic Usage

```python
from crosscat import insert_rows
import jax.numpy as jnp

new_rows = jnp.array([
    [95000.0, 7.0, 0.0, 1.0],
    [55000.0, 1.0, 2.0, 0.0],
])

key, subkey = jax.random.split(key)
state, data = insert_rows(subkey, state, data, new_rows)
print(f"New data shape: {data.shape}")
```

## Packed Version

```python
from crosscat.packed.kernels import packed_insert_rows

packed, data = packed_insert_rows(key, packed, data, new_rows)
```

## Sample and Insert

Complete a partial observation and insert it:

```python
from crosscat import sample_and_insert

# Row with some missing values
partial_row = jnp.array([jnp.nan, 5.0, 0.0, jnp.nan])

state, data, completed_row = sample_and_insert(key, state, data, partial_row)
print(f"Completed: {completed_row}")
```

## Tips

- Row insertion uses the **current** cluster structure — it doesn't update existing assignments
- For best results, run a few more Gibbs sweeps after inserting many new rows
- Unpacked `insert_rows` is fast (pure Python, no JIT). Packed `packed_insert_rows` triggers JIT compilation on first call

## API Reference

- [`insert_rows`](../api/model.md#insert_rows)
- [`packed_insert_rows`](../api/packed-kernels.md#packed_insert_rows)
- [`sample_and_insert`](../api/inference.md#sample_and_insert)
