"""State serialization for CrossCat (save/load/checkpoint).

Format: a ``.jxc`` directory containing:
- ``metadata.json`` — schema version, static fields, column types
- ``arrays.npz`` — all JAX arrays as compressed NumPy arrays

No external dependencies beyond ``json`` and ``numpy`` (both already required).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import Array

from crosscat.packed.state import (
    _ARRAY_FIELDS,
    _STATIC_FIELDS,
    PackedCrossCatState,
    pack_state,
    unpack_state,
)
from crosscat.types import ColumnType, CrossCatState

_SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# Packed state save / load
# ---------------------------------------------------------------------------


def save_packed_state(
    packed: PackedCrossCatState,
    path: str | Path,
    *,
    column_types: list[ColumnType] | None = None,
) -> Path:
    """Save a PackedCrossCatState to disk.

    Args:
        packed: The packed state to save.
        path: Directory path (a ``.jxc`` suffix is added if missing).
        column_types: Optional column types (needed to reconstruct CrossCatState later).

    Returns:
        The resolved directory path.
    """
    path = Path(path)
    if path.suffix != ".jxc":
        path = path.with_suffix(".jxc")
    path.mkdir(parents=True, exist_ok=True)

    # Metadata
    metadata: dict = {
        "schema_version": _SCHEMA_VERSION,
        "state_type": "packed",
    }
    for name in _STATIC_FIELDS:
        metadata[name] = int(getattr(packed, name))
    if column_types is not None:
        metadata["column_types"] = [ct.value for ct in column_types]

    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Arrays
    arrays = {name: np.asarray(getattr(packed, name)) for name in _ARRAY_FIELDS}
    np.savez_compressed(path / "arrays.npz", **arrays)

    return path


def load_packed_state(
    path: str | Path,
) -> tuple[PackedCrossCatState, list[ColumnType] | None]:
    """Load a PackedCrossCatState from disk.

    Args:
        path: Directory path (with or without ``.jxc`` suffix).

    Returns:
        (packed_state, column_types) — column_types is None if not saved.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If schema version is unsupported.
    """
    path = Path(path)
    if path.suffix != ".jxc":
        path = path.with_suffix(".jxc")
    if not path.exists():
        raise FileNotFoundError(f"No saved state at {path}")

    # Metadata
    with open(path / "metadata.json") as f:
        metadata = json.load(f)

    version = metadata.get("schema_version", 0)
    if version > _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version {version} (expected <={_SCHEMA_VERSION}). "
            f"This state was saved with a newer version of jax-crosscat."
        )

    # Arrays
    npz = np.load(path / "arrays.npz")
    kwargs = {}
    for name in _ARRAY_FIELDS:
        if name in npz:
            kwargs[name] = jnp.array(npz[name])
        elif name == "hyper_cutpoints":
            # Migration from schema v1: synthesize default cutpoints (+inf padding)
            n_cols = int(metadata["n_cols"])
            max_cats = int(metadata["max_categories"])
            kwargs[name] = jnp.full((n_cols, max_cats - 1), jnp.inf)
        elif name == "hyper_n_cutpoints":
            # Migration from schema v2: infer cutpoint counts from isfinite
            n_cols = int(metadata["n_cols"])
            if "hyper_cutpoints" in kwargs:
                cp = kwargs["hyper_cutpoints"]
                kwargs[name] = jnp.sum(jnp.isfinite(cp), axis=1).astype(jnp.int32)
            else:
                kwargs[name] = jnp.zeros(n_cols, dtype=jnp.int32)
        else:
            raise ValueError(f"Missing required array field '{name}' in saved state")
    for name in _STATIC_FIELDS:
        kwargs[name] = int(metadata[name])

    packed = PackedCrossCatState(**kwargs)

    # Column types
    column_types = None
    if "column_types" in metadata:
        column_types = [ColumnType(v) for v in metadata["column_types"]]

    return packed, column_types


# ---------------------------------------------------------------------------
# CrossCatState save / load (via pack/unpack roundtrip)
# ---------------------------------------------------------------------------


def save_state(state: CrossCatState, path: str | Path) -> Path:
    """Save a CrossCatState to disk.

    Internally packs the state for efficient storage, then saves.

    Args:
        state: The state to save.
        path: Directory path (a ``.jxc`` suffix is added if missing).

    Returns:
        The resolved directory path.
    """
    packed = pack_state(state)
    result = save_packed_state(packed, path, column_types=state.column_types)

    # Mark as CrossCatState origin for load_state
    meta_path = result / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    metadata["state_type"] = "crosscat"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return result


def load_state(path: str | Path, data: Array | None = None) -> CrossCatState:
    """Load a CrossCatState from disk.

    Args:
        path: Directory path (with or without ``.jxc`` suffix).
        data: Optional data matrix. When provided, sufficient statistics are
            recomputed from data for exact fidelity (recommended).

    Returns:
        The reconstructed CrossCatState.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If column_types were not saved (required for unpacking).
    """
    packed, column_types = load_packed_state(path)
    if data is None:
        import warnings

        warnings.warn(
            "Loading state without data: sufficient statistics are reconstructed "
            "from saved arrays, which may lose precision due to float32 storage. "
            "Pass data=... for exact reconstruction.",
            UserWarning,
            stacklevel=2,
        )
    if column_types is None:
        raise ValueError(
            "Cannot load CrossCatState: column_types not found in metadata. "
            "Use load_packed_state() instead and provide column_types manually."
        )
    return unpack_state(packed, column_types, data=data)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    packed: PackedCrossCatState,
    base_path: str | Path,
    sweep_number: int,
    *,
    column_types: list[ColumnType] | None = None,
    log_joint_value: float | None = None,
) -> Path:
    """Save a checkpoint during inference.

    Creates a directory like ``{base_path}/checkpoint_sweep_000050.jxc/``.

    Args:
        packed: Current packed state.
        base_path: Base directory for checkpoints.
        sweep_number: Current sweep number (used in filename).
        column_types: Optional column types to persist.
        log_joint_value: Optional log-joint score to record in metadata.

    Returns:
        The checkpoint directory path.
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"checkpoint_sweep_{sweep_number:06d}"
    ckpt_path = base_path / ckpt_name

    result = save_packed_state(packed, ckpt_path, column_types=column_types)

    # Add checkpoint-specific metadata
    meta_path = result / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    metadata["sweep_number"] = sweep_number
    if log_joint_value is not None:
        metadata["log_joint"] = float(log_joint_value)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return result


def load_latest_checkpoint(
    base_path: str | Path,
) -> tuple[PackedCrossCatState, list[ColumnType] | None, int]:
    """Load the most recent checkpoint.

    Args:
        base_path: Directory containing checkpoint subdirectories.

    Returns:
        (packed_state, column_types, sweep_number).

    Raises:
        FileNotFoundError: If no checkpoints found.
    """
    base_path = Path(base_path)
    ckpts = sorted(base_path.glob("checkpoint_sweep_*.jxc"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {base_path}")

    latest = ckpts[-1]
    packed, column_types = load_packed_state(latest)

    # Read sweep number from metadata
    with open(latest / "metadata.json") as f:
        metadata = json.load(f)
    sweep_number = metadata.get("sweep_number", 0)

    return packed, column_types, sweep_number
