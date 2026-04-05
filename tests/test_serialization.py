"""Tests for state serialization (save/load/checkpoint)."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import pytest

from crosscat.model import initialize, log_joint
from crosscat.packed import pack_state
from crosscat.serialization import (
    load_latest_checkpoint,
    load_packed_state,
    load_state,
    save_checkpoint,
    save_packed_state,
    save_state,
)
from crosscat.synthetic import generate_crosscat_data
from crosscat.types import ColumnType

pytestmark = pytest.mark.cpu


@pytest.fixture
def mixed_state():
    """Create a CrossCatState with mixed column types."""
    key = jax.random.key(42)
    column_types = [
        ColumnType.CONTINUOUS,
        ColumnType.CATEGORICAL,
        ColumnType.BINARY,
        ColumnType.CYCLIC,
    ]
    result = generate_crosscat_data(
        key,
        n_rows=50,
        column_types=column_types,
        n_views=2,
        n_clusters=2,
    )
    k1, k2 = jax.random.split(key)
    state = initialize(k1, result["data"], column_types).state
    return state, result["data"], column_types


# ---------------------------------------------------------------------------
# Packed state roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_packed_state(mixed_state, tmp_path):
    """Save and load a PackedCrossCatState — all fields should match."""
    state, data, column_types = mixed_state
    packed = pack_state(state)

    path = save_packed_state(packed, tmp_path / "test", column_types=column_types)
    loaded, loaded_types = load_packed_state(path)

    # Static fields
    assert loaded.n_rows == packed.n_rows
    assert loaded.n_cols == packed.n_cols
    assert loaded.max_views == packed.max_views
    assert loaded.max_clusters == packed.max_clusters

    # Array fields
    assert jnp.array_equal(loaded.column_assignments, packed.column_assignments)
    assert jnp.allclose(loaded.column_crp_alpha, packed.column_crp_alpha)
    assert jnp.array_equal(loaded.view_row_assignments, packed.view_row_assignments)
    assert jnp.allclose(loaded.ss_sum_x, packed.ss_sum_x)
    assert jnp.allclose(loaded.ss_cat_counts, packed.ss_cat_counts)

    # Column types preserved
    assert loaded_types == column_types


def test_roundtrip_packed_without_column_types(mixed_state, tmp_path):
    """Save without column_types — load returns None for column_types."""
    state, _, _ = mixed_state
    packed = pack_state(state)

    path = save_packed_state(packed, tmp_path / "no_types")
    _, loaded_types = load_packed_state(path)
    assert loaded_types is None


# ---------------------------------------------------------------------------
# CrossCatState roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_crosscat_state_preserves_log_joint(mixed_state, tmp_path):
    """save_state + load_state preserves log_joint (relative to pack/unpack roundtrip).

    Note: pack_state/unpack_state has inherent suffstats reconstruction error,
    so we compare against a fresh pack/unpack roundtrip, not the original state.
    """
    state, data, column_types = mixed_state

    # Baseline: pack/unpack roundtrip (this is what save_state does internally)
    from crosscat.packed import pack_state, unpack_state

    baseline = unpack_state(pack_state(state), column_types)
    baseline_lj = float(log_joint(baseline, data))

    path = save_state(state, tmp_path / "crosscat_state")
    loaded = load_state(path)
    loaded_lj = float(log_joint(loaded, data))

    assert abs(baseline_lj - loaded_lj) < 1e-3, (
        f"Log joint mismatch: baseline={baseline_lj}, loaded={loaded_lj}"
    )


def test_roundtrip_crosscat_state_preserves_structure(mixed_state, tmp_path):
    """save_state + load_state preserves views and assignments."""
    state, _, _ = mixed_state

    path = save_state(state, tmp_path / "struct")
    loaded = load_state(path)

    assert loaded.n_views == state.n_views
    assert loaded.n_rows == state.n_rows
    assert loaded.n_cols == state.n_cols
    assert jnp.array_equal(
        jnp.array(loaded.column_assignments),
        jnp.array(state.column_assignments),
    )


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def test_checkpoint_save_load(mixed_state, tmp_path):
    """Save multiple checkpoints, load_latest returns the most recent."""
    state, _, column_types = mixed_state
    packed = pack_state(state)
    ckpt_dir = tmp_path / "checkpoints"

    save_checkpoint(packed, ckpt_dir, sweep_number=10, column_types=column_types)
    save_checkpoint(packed, ckpt_dir, sweep_number=20, column_types=column_types)
    save_checkpoint(
        packed,
        ckpt_dir,
        sweep_number=50,
        column_types=column_types,
        log_joint_value=-1234.5,
    )

    loaded, loaded_types, sweep = load_latest_checkpoint(ckpt_dir)
    assert sweep == 50
    assert loaded_types == column_types
    assert loaded.n_rows == packed.n_rows


def test_checkpoint_metadata_contains_log_joint(mixed_state, tmp_path):
    """Checkpoint metadata includes sweep_number and log_joint."""
    state, _, column_types = mixed_state
    packed = pack_state(state)
    ckpt_dir = tmp_path / "ckpt_meta"

    path = save_checkpoint(
        packed,
        ckpt_dir,
        sweep_number=42,
        column_types=column_types,
        log_joint_value=-999.9,
    )

    with open(path / "metadata.json") as f:
        meta = json.load(f)
    assert meta["sweep_number"] == 42
    assert abs(meta["log_joint"] - (-999.9)) < 1e-6


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_load_nonexistent_path_raises(tmp_path):
    """Loading from a nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No saved state"):
        load_packed_state(tmp_path / "nonexistent")


def test_load_bad_schema_version_raises(mixed_state, tmp_path):
    """A schema version mismatch raises ValueError."""
    state, _, column_types = mixed_state
    packed = pack_state(state)
    path = save_packed_state(packed, tmp_path / "bad_version", column_types=column_types)

    # Tamper with schema version
    meta_path = path / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["schema_version"] = 999
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    with pytest.raises(ValueError, match="Unsupported schema version"):
        load_packed_state(path)


def test_load_state_without_column_types_raises(mixed_state, tmp_path):
    """load_state raises if column_types missing from metadata."""
    state, _, _ = mixed_state
    packed = pack_state(state)
    save_packed_state(packed, tmp_path / "no_types_for_crosscat")

    with pytest.raises(ValueError, match="column_types not found"):
        load_state(tmp_path / "no_types_for_crosscat")


def test_no_checkpoints_raises(tmp_path):
    """load_latest_checkpoint raises if directory has no checkpoints."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No checkpoints found"):
        load_latest_checkpoint(empty_dir)


# ---------------------------------------------------------------------------
# .jxc suffix handling
# ---------------------------------------------------------------------------


def test_jxc_suffix_added_automatically(mixed_state, tmp_path):
    """Path without .jxc suffix gets it added automatically."""
    state, _, column_types = mixed_state
    packed = pack_state(state)

    path = save_packed_state(packed, tmp_path / "mystate", column_types=column_types)
    assert path.suffix == ".jxc"
    assert path.exists()

    # Load works with or without suffix
    loaded1, _ = load_packed_state(tmp_path / "mystate")
    loaded2, _ = load_packed_state(tmp_path / "mystate.jxc")
    assert loaded1.n_rows == loaded2.n_rows
