"""Tests for parallel multi-chain inference via vmap."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.model import initialize, log_joint
from crosscat.packed import (
    pack_state,
    unpack_state,
)
from crosscat.packed.kernels import (
    multi_chain_packed_gibbs_sweep,
    packed_log_joint,
)
from crosscat.packed.state import (
    _ARRAY_FIELDS,
    _STATIC_FIELDS,
    batch_packed_states,
    select_best_chain,
    unbatch_packed_states,
)
from crosscat.synthetic import generate_crosscat_data
from crosscat.types import ColumnType


@pytest.fixture
def multi_chain_setup():
    """Create 4 packed chains from different initializations."""
    key = jax.random.key(42)
    column_types = [
        ColumnType.CONTINUOUS,
        ColumnType.CATEGORICAL,
        ColumnType.BINARY,
    ]
    result = generate_crosscat_data(key, 50, column_types, n_views=2, n_clusters=2)
    data = result["data"]

    packed_list = []
    for i in range(4):
        k = jax.random.fold_in(key, i)
        state = initialize(k, data, column_types)
        packed_list.append(pack_state(state))

    return packed_list, data, column_types


def test_batch_unbatch_roundtrip(multi_chain_setup):
    """Batching then unbatching preserves all fields."""
    packed_list, _, _ = multi_chain_setup
    batched = batch_packed_states(packed_list)
    recovered = unbatch_packed_states(batched, len(packed_list))

    for i, (orig, rec) in enumerate(zip(packed_list, recovered, strict=True)):
        for name in _ARRAY_FIELDS:
            assert jnp.array_equal(getattr(orig, name), getattr(rec, name)), (
                f"Chain {i}, field {name} mismatch"
            )
        for name in _STATIC_FIELDS:
            assert getattr(orig, name) == getattr(rec, name)


def test_packed_log_joint_matches_original(multi_chain_setup):
    """packed_log_joint matches model.log_joint on unpacked state (same suffstats)."""
    packed_list, data, column_types = multi_chain_setup
    packed = packed_list[0]

    # Packed version
    plj = float(packed_log_joint(packed, data))

    # Original version (unpack WITHOUT data — uses same packed suffstats)
    state = unpack_state(packed, column_types)
    olj = float(log_joint(state, data))

    assert abs(plj - olj) < 1.0, f"packed_log_joint={plj}, log_joint={olj}, gap={abs(plj - olj)}"


@pytest.mark.slow
def test_vmap_multi_chain_runs(multi_chain_setup):
    """multi_chain_packed_gibbs_sweep runs and returns correct shapes."""
    packed_list, data, column_types = multi_chain_setup
    n_chains = len(packed_list)

    key = jax.random.key(99)
    batched_result, scores = multi_chain_packed_gibbs_sweep(
        key,
        packed_list,
        data,
        n_sweeps=2,
    )

    # Scores shape
    assert scores.shape == (n_chains,)
    assert jnp.all(jnp.isfinite(scores))

    # Batched result has leading n_chains dimension
    assert batched_result.column_assignments.shape[0] == n_chains


@pytest.mark.slow
def test_multi_chain_selects_best(multi_chain_setup):
    """select_best_chain picks the chain with highest score."""
    packed_list, data, column_types = multi_chain_setup

    key = jax.random.key(123)
    batched_result, scores = multi_chain_packed_gibbs_sweep(
        key,
        packed_list,
        data,
        n_sweeps=2,
    )

    best = select_best_chain(batched_result, scores)
    best_idx = int(jnp.argmax(scores))

    # Verify it picked the right chain
    assert jnp.array_equal(
        best.column_assignments,
        batched_result.column_assignments[best_idx],
    )


@pytest.mark.slow
def test_multi_chain_deterministic(multi_chain_setup):
    """Same key gives same result."""
    packed_list, data, _ = multi_chain_setup
    key = jax.random.key(77)

    _, scores1 = multi_chain_packed_gibbs_sweep(key, packed_list, data, n_sweeps=1)
    _, scores2 = multi_chain_packed_gibbs_sweep(key, packed_list, data, n_sweeps=1)

    assert jnp.allclose(scores1, scores2)
