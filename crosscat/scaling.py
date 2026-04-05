"""Scaling utilities for large-dataset inference.

Provides higher-level workflows that combine subsample initialization,
batch insertion, and mini-batch Gibbs sweeps for datasets with 10K+ rows.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from crosscat.model import initialize
from crosscat.packed.kernels import (
    packed_gibbs_sweep,
    packed_insert_rows,
    packed_log_joint,
    packed_transition_column_assignments,
    packed_transition_column_hypers,
    packed_transition_crp_alphas,
    packed_transition_row_assignments_minibatch,
)
from crosscat.packed.state import PackedCrossCatState, pack_state, suggest_max_clusters
from crosscat.types import ColumnType


def subsample_anneal(
    rng_key: Array,
    data: Array,
    column_types: list[ColumnType],
    *,
    initial_size: int = 1000,
    growth_factor: float = 2.0,
    sweeps_per_stage: int = 10,
    max_clusters: int | None = None,
    max_views: int = 16,
    insert_batch_size: int = 5000,
) -> tuple[PackedCrossCatState, Array]:
    """Subsample-annealing: gradually grow the dataset during inference.

    Starts with a small subsample, runs Gibbs sweeps to find structure,
    then progressively inserts more rows and refines. Convergence quality
    depends on ``sweeps_per_stage`` — more sweeps per stage allow the
    chain to mix before new data arrives.

    Stages:
      1. Initialize on ``initial_size`` rows, run sweeps
      2. Grow active rows by ``growth_factor``, insert new batch, run sweeps
      3. Repeat step 2 until all rows are included (last iteration
         serves as the final refinement on the full dataset)

    Args:
        rng_key: JAX PRNG key.
        data: Full data matrix, shape (n_rows, n_cols).
        column_types: Type specification per column.
        initial_size: Number of rows for initial subsample.
        growth_factor: Multiplicative growth per stage (default 2x).
        sweeps_per_stage: Gibbs sweeps per annealing stage.
        max_clusters: Max clusters for pack_state. If None, uses heuristic.
        max_views: Max views for pack_state.
        insert_batch_size: Batch size for packed_insert_rows calls.

    Returns:
        Tuple of (packed_state, reordered_data) where reordered_data has
        subsample rows first, then remaining rows in insertion order.
    """
    n_rows = data.shape[0]
    if max_clusters is None:
        max_clusters = suggest_max_clusters(n_rows)

    initial_size = min(initial_size, n_rows)

    # Stage 0: Initialize on subsample
    k_init, k_sweep, k_rest = jax.random.split(rng_key, 3)
    result = initialize(k_init, data, column_types, subsample_rows=initial_size)
    state = result.state
    sub_idx = result.subsample_idx
    current_data = data[sub_idx]
    packed = pack_state(state, max_clusters=max_clusters, max_views=max_views)

    # Pre-sweeps on initial subsample
    k_sweep, k1 = jax.random.split(k_sweep)
    packed = packed_gibbs_sweep(k1, packed, current_data, n_sweeps=sweeps_per_stage)

    # Track which rows are already included
    included = jnp.zeros(n_rows, dtype=bool).at[sub_idx].set(True)
    current_target = int(initial_size * growth_factor)

    stage = 1
    while int(jnp.sum(included)) < n_rows:
        # Determine how many new rows to add this stage
        n_included = int(jnp.sum(included))
        n_to_add = min(current_target - n_included, n_rows - n_included)
        if n_to_add <= 0:
            break

        # Sample new rows from remaining
        remaining_idx = jnp.where(~included, size=n_rows - n_included)[0]
        k_rest, k_sample, k_insert, k_sweep_stage = jax.random.split(k_rest, 4)

        if n_to_add >= remaining_idx.shape[0]:
            new_idx = remaining_idx
        else:
            chosen = jax.random.choice(
                k_sample, remaining_idx.shape[0], shape=(n_to_add,), replace=False
            )
            new_idx = remaining_idx[chosen]

        new_rows = data[new_idx]
        included = included.at[new_idx].set(True)

        # Insert in batches
        n_insert_batches = (new_rows.shape[0] + insert_batch_size - 1) // insert_batch_size
        for b in range(n_insert_batches):
            batch = new_rows[b * insert_batch_size : (b + 1) * insert_batch_size]
            kb = jax.random.fold_in(k_insert, b)
            packed, current_data = packed_insert_rows(kb, packed, current_data, batch)

        # Sweeps on current dataset
        packed = packed_gibbs_sweep(k_sweep_stage, packed, current_data, n_sweeps=sweeps_per_stage)

        stage += 1
        current_target = int(current_target * growth_factor)

    return packed, current_data


def minibatch_gibbs_sweep(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    *,
    batch_size: int = 10_000,
    n_sweeps: int = 1,
) -> PackedCrossCatState:
    """Run multiple mini-batch Gibbs sweeps.

    Each sweep updates ``batch_size`` randomly sampled rows, then runs
    full column assignment, column hyper, and CRP alpha transitions.
    The row kernel cost is O(B) instead of O(N); column/hyper/CRP
    transitions still operate on the full dataset. Uses a Python
    for-loop over sweeps (separate JIT dispatch per sweep).

    Args:
        rng_key: JAX PRNG key.
        packed: Current packed state.
        data: (n_rows, n_cols) data matrix.
        batch_size: Number of rows to update per sweep.
        n_sweeps: Number of sweeps to run.

    Returns:
        Updated PackedCrossCatState.
    """
    keys = jax.random.split(rng_key, n_sweeps)

    for i in range(n_sweeps):
        k1, k2, k3, k4 = jax.random.split(keys[i], 4)
        packed = packed_transition_row_assignments_minibatch(
            k1, packed, data, batch_size=batch_size
        )
        packed = packed_transition_column_assignments(k2, packed, data)
        packed = packed_transition_column_hypers(k3, packed, data)
        packed = packed_transition_crp_alphas(k4, packed)

    return packed


def gibbs_sweep_early_stopping(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    *,
    max_sweeps: int = 200,
    check_interval: int = 10,
    patience: int = 3,
    min_improvement: float = 0.001,
    batch_size: int | None = None,
) -> tuple[PackedCrossCatState, list[float]]:
    """Run Gibbs sweeps with convergence-based early stopping.

    Monitors log-joint probability every ``check_interval`` sweeps. Stops
    when the relative improvement compared to the **previous checkpoint**
    falls below ``min_improvement`` for ``patience`` consecutive checks.
    This avoids premature stopping during healthy MCMC mixing where the
    log-joint fluctuates around equilibrium.

    If the log-joint becomes NaN or infinite, the loop stops immediately
    with a warning.

    Args:
        rng_key: JAX PRNG key.
        packed: Current packed state.
        data: (n_rows, n_cols) data matrix.
        max_sweeps: Maximum number of sweeps.
        check_interval: Sweeps between convergence checks.
        patience: Number of checks with insufficient improvement before stopping.
        min_improvement: Minimum relative improvement threshold.
        batch_size: If set, use mini-batch row transitions with this batch size.
            If None, use full sweeps.

    Returns:
        Tuple of (final_packed_state, log_joint_history).
    """
    import math
    import warnings

    log_joints: list[float] = []
    stale_count = 0
    prev_lj: float | None = None
    total_sweeps = 0

    while total_sweeps < max_sweeps:
        rng_key, k_sweep = jax.random.split(rng_key)
        sweeps_this_round = min(check_interval, max_sweeps - total_sweeps)

        if batch_size is not None:
            packed = minibatch_gibbs_sweep(
                k_sweep, packed, data, batch_size=batch_size, n_sweeps=sweeps_this_round
            )
        else:
            packed = packed_gibbs_sweep(k_sweep, packed, data, n_sweeps=sweeps_this_round)

        total_sweeps += sweeps_this_round

        # Compute log-joint
        lj = float(packed_log_joint(packed, data))
        log_joints.append(lj)

        # Bail on degenerate state
        if not math.isfinite(lj):
            warnings.warn(
                f"Log-joint became {lj} after {total_sweeps} sweeps. "
                f"Stopping early — state may be degenerate.",
                stacklevel=2,
            )
            break

        # Check convergence against previous checkpoint (not all-time best)
        if prev_lj is not None:
            rel_improvement = (lj - prev_lj) / (abs(prev_lj) + 1e-10)
            if rel_improvement < min_improvement:
                stale_count += 1
            else:
                stale_count = 0

        prev_lj = lj

        if stale_count >= patience:
            break

    return packed, log_joints


def multi_device_gibbs_sweep(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    *,
    n_sweeps: int = 1,
) -> PackedCrossCatState:
    """Run Gibbs sweeps using parallel row scoring across available devices.

    Uses ``packed_transition_row_assignments_parallel`` for row assignments
    (vmap over all rows, scoring against shared suffstats) and standard
    kernels for column/hyper/CRP transitions. This maximizes GPU utilization
    for large datasets.

    For multi-GPU setups, the parallel row scoring naturally distributes
    across devices via JAX's XLA compiler when data is sharded.

    Args:
        rng_key: JAX PRNG key.
        packed: Current packed state.
        data: (n_rows, n_cols) data matrix.
        n_sweeps: Number of sweeps to run.

    Returns:
        Updated PackedCrossCatState.
    """
    from crosscat.packed.kernels import packed_transition_row_assignments_parallel

    keys = jax.random.split(rng_key, n_sweeps)

    for i in range(n_sweeps):
        k1, k2, k3, k4 = jax.random.split(keys[i], 4)
        packed = packed_transition_row_assignments_parallel(k1, packed, data)
        packed = packed_transition_column_assignments(k2, packed, data)
        packed = packed_transition_column_hypers(k3, packed, data)
        packed = packed_transition_crp_alphas(k4, packed)

    return packed


def shard_data_across_devices(data: Array) -> Array:
    """Shard a data matrix across all available JAX devices.

    Distributes rows evenly across devices using JAX's device_put_sharded.
    This is useful for multi-GPU setups where the data matrix exceeds
    single-device memory.

    Args:
        data: (n_rows, n_cols) data matrix.

    Returns:
        Sharded data array distributed across devices.
    """
    devices = jax.devices()
    n_devices = len(devices)
    if n_devices <= 1:
        return data

    n_rows = data.shape[0]
    # Pad to make divisible by n_devices
    remainder = n_rows % n_devices
    if remainder > 0:
        pad_rows = n_devices - remainder
        padding = jnp.full((pad_rows, data.shape[1]), jnp.nan)
        data = jnp.concatenate([data, padding], axis=0)

    # Distribute via JAX sharding
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    mesh_devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = Mesh(mesh_devices, axis_names=("devices",))
    sharding = NamedSharding(mesh, PartitionSpec("devices", None))

    return jax.device_put(data, sharding)
