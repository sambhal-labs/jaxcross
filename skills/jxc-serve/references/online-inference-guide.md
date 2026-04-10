# Online Inference Guide

## Row insertion

```python
from crosscat import packed_insert_rows

# Insert new rows into existing model
# new_rows: shape (n_new, n_cols), dtype float32
packed, data_updated = packed_insert_rows(key, packed, data, new_rows)
```

How it works:
1. Each new row is assigned to a cluster in each view via CRP predictive
2. Sufficient statistics are updated incrementally
3. No Gibbs sweeps are run — call `packed_gibbs_sweep` afterward for refinement

## Batch insertion pattern

For streaming data, batch insertions and sweep periodically:

```python
BATCH_SIZE = 100
SWEEP_INTERVAL = 500  # Sweep after every 500 new rows

buffer = []
total_inserted = 0

for new_row in data_stream:
    buffer.append(new_row)
    
    if len(buffer) >= BATCH_SIZE:
        new_batch = jnp.array(buffer, dtype=jnp.float32)
        key, subkey = jax.random.split(key)
        packed, data = packed_insert_rows(subkey, packed, data, new_batch)
        total_inserted += len(buffer)
        buffer = []
        
        if total_inserted % SWEEP_INTERVAL == 0:
            key, subkey = jax.random.split(key)
            packed = packed_gibbs_sweep(subkey, packed, data, n_sweeps=5)
            print(f"Inserted {total_inserted} rows, ran 5 sweeps")
```

## When to retrain vs. incrementally update

| Scenario | Action |
|----------|--------|
| <20% new rows | Incremental: insert + 5-10 sweeps |
| 20-50% new rows | Moderate: insert + 50-100 sweeps |
| >50% new rows | Full retrain with `/jxc-model` |
| Distribution shift detected | Full retrain |
| New column types needed | Full retrain |

## Monitoring insertion quality

After inserting rows, check that the model still explains the data well:

```python
lj_before = float(packed_log_joint(packed_before, data_before))
lj_after = float(packed_log_joint(packed_after, data_after))

# Log-joint should not decrease significantly
# (slight decrease is normal since more data = more to explain)
per_row_before = lj_before / data_before.shape[0]
per_row_after = lj_after / data_after.shape[0]

if per_row_after < 0.8 * per_row_before:
    print("WARNING: Per-row log-joint dropped significantly after insertion")
```
