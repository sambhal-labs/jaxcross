# Serialization & Checkpointing

## What

Save trained models to disk and load them back. Checkpoint during long inference runs to enable resume-from-failure.

## When to Use

- Persisting trained models for later use
- Resuming interrupted inference runs
- Sharing models between sessions or machines

## Save and Load Packed State

```python
from crosscat import save_packed_state, load_packed_state

# Save (creates a .jxc directory with JSON metadata + NPZ arrays)
save_packed_state(packed, "my_model", column_types=col_types)

# Load
packed, col_types = load_packed_state("my_model")
```

## Save and Load Unpacked State

```python
from crosscat import save_state, load_state

# Save (packs internally)
save_state(state, "my_model")

# Load (pass data= for exact sufficient statistics)
state = load_state("my_model", data=data)
```

## Checkpointing During Inference

Save progress periodically during long-running inference:

```python
from crosscat import save_checkpoint, load_latest_checkpoint

for sweep in range(100):
    key, subkey = jax.random.split(key)
    packed = packed_gibbs_sweep(subkey, packed, data, n_sweeps=1)

    if (sweep + 1) % 10 == 0:
        score = float(packed_log_joint(packed, data))
        save_checkpoint(
            packed, "checkpoints/",
            sweep_number=sweep + 1,
            column_types=col_types,
            log_joint_value=score,
        )
        print(f"Checkpoint at sweep {sweep + 1}, log_joint={score:.1f}")
```

## Resuming from Checkpoint

```python
packed, col_types, sweep_num = load_latest_checkpoint("checkpoints/")
print(f"Resuming from sweep {sweep_num}")

# Continue inference
for sweep in range(sweep_num, 100):
    packed = packed_gibbs_sweep(key, packed, data, n_sweeps=1)
```

## File Format

The `.jxc` format is a directory containing:

- `metadata.json` — column types, version info, dimensions
- `arrays.npz` — all state arrays in NumPy compressed format

## API Reference

- [`save_packed_state`](../api/serialization.md#save_packed_state)
- [`load_packed_state`](../api/serialization.md#load_packed_state)
- [`save_state`](../api/serialization.md#save_state)
- [`load_state`](../api/serialization.md#load_state)
- [`save_checkpoint`](../api/serialization.md#save_checkpoint)
- [`load_latest_checkpoint`](../api/serialization.md#load_latest_checkpoint)
