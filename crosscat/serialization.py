"""State serialization for CrossCat (save/load/checkpoint).

Format: a ``.jxc`` directory containing:
- ``metadata.json`` — schema version, static fields, column types
- ``arrays.npz`` — all JAX arrays as compressed NumPy arrays

No external dependencies beyond ``json`` and ``numpy`` (both already required).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from contextlib import contextmanager
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

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 3
_MIN_READER_VERSION = 3


@contextmanager
def _file_lock(lock_path: Path, *, shared: bool = False):
    """Cross-platform advisory file lock (exclusive or shared).

    Uses ``fcntl.flock`` on POSIX and is a no-op on Windows (where file
    locking semantics differ and GPU workflows are uncommon).
    """
    try:
        import fcntl
    except ImportError:
        # Windows: no fcntl, skip locking
        yield
        return

    flag = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(fd, flag)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        # Clean up lock file (best-effort, may fail if another process holds it)
        with contextlib.suppress(OSError):
            lock_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Packed state save / load
# ---------------------------------------------------------------------------


def save_packed_state(
    packed: PackedCrossCatState,
    path: str | Path,
    *,
    column_types: list[ColumnType] | None = None,
    _extra_metadata: dict | None = None,
) -> Path:
    """Save a PackedCrossCatState to disk.

    Args:
        packed: The packed state to save.
        path: Directory path (a ``.jxc`` suffix is added if missing).
        column_types: Optional column types (needed to reconstruct CrossCatState later).
        _extra_metadata: Internal — additional metadata fields written atomically
            under the same lock (used by save_state/save_checkpoint).

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
        "min_reader_version": _MIN_READER_VERSION,
        "state_type": "packed",
    }
    for name in _STATIC_FIELDS:
        metadata[name] = int(getattr(packed, name))
    if column_types is not None:
        metadata["column_types"] = [ct.value for ct in column_types]
    if _extra_metadata:
        metadata.update(_extra_metadata)

    # Write under exclusive lock with atomic rename to prevent corruption
    lock_path = path / ".lock"
    with _file_lock(lock_path):
        # Remove validity marker before writing
        valid_marker = path / ".valid"
        valid_marker.unlink(missing_ok=True)

        # Write metadata to temp file, then atomic rename
        meta_tmp = path / "metadata.json.tmp"
        meta_final = path / "metadata.json"
        with open(meta_tmp, "w") as f:
            json.dump(metadata, f, indent=2)
        meta_tmp.rename(meta_final)

        # Write arrays to temp file, then atomic rename
        # np.savez_compressed appends .npz automatically, so use a stem name
        arrays_tmp_stem = path / "_arrays_tmp"
        arrays_tmp_file = path / "_arrays_tmp.npz"
        arrays_final = path / "arrays.npz"
        arrays = {name: np.asarray(getattr(packed, name)) for name in _ARRAY_FIELDS}
        np.savez_compressed(arrays_tmp_stem, **arrays)
        arrays_tmp_file.rename(arrays_final)

        # Mark as valid only after both files are written
        valid_marker.touch()

    logger.info("Saved packed state to %s (schema v%d)", path, _SCHEMA_VERSION)
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

    # Check validity marker (written after both files complete)
    valid_marker = path / ".valid"
    if not valid_marker.exists() and (path / "metadata.json").exists():
        logger.warning(
            "State at %s may be corrupt (missing .valid marker). "
            "A previous save may have been interrupted.",
            path,
        )

    # Read under shared lock to prevent reading during concurrent writes
    lock_path = path / ".lock"
    with _file_lock(lock_path, shared=True):
        # Metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        version = metadata.get("schema_version", 0)
        min_reader = metadata.get("min_reader_version", 0)
        if min_reader > _SCHEMA_VERSION:
            raise ValueError(
                f"This state requires reader version >={min_reader} but this "
                f"installation supports <={_SCHEMA_VERSION}. Upgrade jax-crosscat."
            )
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

    logger.info("Loaded packed state from %s (schema v%d)", path, version)
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
    return save_packed_state(
        packed,
        path,
        column_types=state.column_types,
        _extra_metadata={"state_type": "crosscat"},
    )


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

    extra: dict = {"sweep_number": sweep_number}
    if log_joint_value is not None:
        extra["log_joint"] = float(log_joint_value)

    return save_packed_state(packed, ckpt_path, column_types=column_types, _extra_metadata=extra)


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
