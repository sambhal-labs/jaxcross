# Serialization

::: crosscat.serialization
    options:
      show_source: false

## Overview

Save and load CrossCat states and checkpoints in `.jxc` format (JSON metadata + NPZ arrays).

## `save_packed_state`

```python
save_packed_state(packed, path, *, column_types=None) -> Path
```

Save a `PackedCrossCatState` to a `.jxc` directory.

**Returns**: `Path` to saved directory.

## `load_packed_state`

```python
load_packed_state(path) -> tuple[PackedCrossCatState, list[ColumnType] | None]
```

Load a packed state from a `.jxc` directory.

**Returns**: `(PackedCrossCatState, column_types)`.

## `save_state`

```python
save_state(state, path) -> Path
```

Save an unpacked `CrossCatState` to disk (packs internally).

**Returns**: `Path` to saved file.

## `load_state`

```python
load_state(path, data=None) -> CrossCatState
```

Load an unpacked `CrossCatState`. When `data` is provided, sufficient statistics are recomputed for exact fidelity.

**Returns**: `CrossCatState`.

## `save_checkpoint`

```python
save_checkpoint(packed, base_path, sweep_number, *, column_types=None, log_joint_value=None) -> Path
```

Save a checkpoint during inference with sweep number metadata.

**Returns**: `Path` to checkpoint directory.

## `load_latest_checkpoint`

```python
load_latest_checkpoint(base_path) -> tuple[PackedCrossCatState, list[ColumnType] | None, int]
```

Load the most recent checkpoint from a base directory.

**Returns**: `(PackedCrossCatState, column_types, sweep_number)`.
