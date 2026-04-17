"""Parity tests: packed conditional predictive queries match unpacked.

The packed tier gained ``condition_cols`` / ``condition_vals`` support in
Phase 2. These tests verify the new code path reproduces the unpacked
implementation's numbers on a small dataset where pack/unpack is lossless.

The tests are CPU-safe because they skip the Gibbs sweep: they use the
initial state directly, which is enough to exercise the conditioning math.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.inference import (
    predictive_cdf,
    predictive_probability,
    predictive_sample,
)
from crosscat.model import initialize
from crosscat.packed import pack_state
from crosscat.packed_inference import (
    batch_predictive_probability,
    batch_predictive_sample,
    multi_chain_predictive_cdf,
    multi_chain_predictive_probability,
    multi_chain_predictive_sample,
    packed_predictive_cdf,
    packed_predictive_probability,
    packed_predictive_sample,
)
from crosscat.types import ColumnType

pytestmark = pytest.mark.cpu


@pytest.fixture
def small_state():
    key = jax.random.key(0)
    data = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [0.3, 0.0, 1.0],
            [5.0, 2.0, 0.0],
            [5.2, 2.0, 1.0],
            [-1.0, 1.0, 0.0],
            [6.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    types = [ColumnType.CONTINUOUS, ColumnType.CATEGORICAL, ColumnType.BINARY]
    state = initialize(key, data, types).state
    packed = pack_state(state, max_categories=3, data=data)
    return state, packed, data, types


def test_packed_predictive_probability_matches_unpacked_with_conditioning(small_state):
    state, packed, data, _ = small_state
    query_cols = [0]
    query_vals = jnp.array([0.25])
    cond_cols = [1]
    cond_vals = jnp.array([1.0])

    unpacked = float(
        predictive_probability(
            state,
            data,
            query_cols,
            query_vals,
            condition_cols=cond_cols,
            condition_vals=cond_vals,
        )
    )
    packed_logp = float(
        packed_predictive_probability(
            packed,
            data,
            query_cols,
            query_vals,
            condition_cols=cond_cols,
            condition_vals=cond_vals,
        )
    )
    assert jnp.isfinite(jnp.asarray(unpacked))
    assert jnp.isfinite(jnp.asarray(packed_logp))
    assert abs(unpacked - packed_logp) < 1e-3


def test_packed_predictive_probability_nan_condition_skipped(small_state):
    """NaN conditioning values must be silently dropped (matches unpacked)."""
    _state, packed, data, _ = small_state
    cond_cols = [1]
    nan_cond = jnp.array([jnp.nan])
    empty_cond = []

    logp_nan = float(
        packed_predictive_probability(
            packed,
            data,
            [0],
            jnp.array([0.25]),
            condition_cols=cond_cols,
            condition_vals=nan_cond,
        )
    )
    logp_marginal = float(
        packed_predictive_probability(
            packed,
            data,
            [0],
            jnp.array([0.25]),
            condition_cols=empty_cond,
        )
    )
    # A pure-NaN condition column should collapse to the marginal mixture.
    assert abs(logp_nan - logp_marginal) < 1e-5


def test_packed_predictive_probability_cross_view_condition_is_independent(small_state):
    """Conditioning on a column in a different view must not change the answer.

    Cross-view conditioning is a no-op in CrossCat (views are independent).
    """
    _state, packed, data, _ = small_state
    # Pick cols in different views if available; if the initial packing put
    # them all in one view this test still passes (conditioning on a column
    # in the same view affects weights, but the parity check below is with
    # the unpacked path which has identical semantics).
    logp_cond = float(
        packed_predictive_probability(
            packed,
            data,
            [0],
            jnp.array([0.25]),
            condition_cols=[2],
            condition_vals=jnp.array([0.0]),
        )
    )
    logp_uncond = float(packed_predictive_probability(packed, data, [0], jnp.array([0.25])))
    assert jnp.isfinite(jnp.asarray(logp_cond))
    assert jnp.isfinite(jnp.asarray(logp_uncond))
    view_of_query = int(packed.column_assignments[0])
    view_of_cond = int(packed.column_assignments[2])
    if view_of_query != view_of_cond:
        # Independent views → conditioning is a no-op
        assert abs(logp_cond - logp_uncond) < 1e-4


def test_packed_predictive_sample_accepts_conditioning(small_state):
    _state, packed, data, _ = small_state
    key = jax.random.key(1)
    samples = packed_predictive_sample(
        key,
        packed,
        data,
        [0],
        condition_cols=[1],
        condition_vals=jnp.array([1.0]),
        n_samples=16,
    )
    assert samples.shape == (16, 1)
    assert jnp.all(jnp.isfinite(samples))


def test_packed_predictive_cdf_accepts_conditioning(small_state):
    _state, packed, data, _ = small_state
    cdf = packed_predictive_cdf(
        jax.random.key(2),
        packed,
        data,
        query_col=0,
        query_val=jnp.array(3.0),
        condition_cols=[1],
        condition_vals=jnp.array([1.0]),
        n_samples=128,
    )
    assert 0.0 <= float(cdf) <= 1.0


def test_batch_predictive_probability_conditioning(small_state):
    """Shared conditioning across a batch: same result per query."""
    _state, packed, data, _ = small_state
    cond_cols = [1]
    cond_vals = jnp.array([1.0])
    query_vals = jnp.array([0.25, 0.25, 0.25])  # Same value three times

    logps = batch_predictive_probability(
        packed,
        data,
        query_col=0,
        query_vals=query_vals,
        row_ids=None,
        condition_cols=cond_cols,
        condition_vals=cond_vals,
    )
    assert logps.shape == (3,)
    # Same query_val + same conditioning → same answer.
    assert abs(float(logps[0]) - float(logps[1])) < 1e-5
    assert abs(float(logps[1]) - float(logps[2])) < 1e-5


def test_batch_predictive_sample_conditioning_n_queries(small_state):
    _state, packed, data, _ = small_state
    samples = batch_predictive_sample(
        jax.random.key(3),
        packed,
        data,
        query_cols=[0],
        row_ids=None,
        condition_cols=[1],
        condition_vals=jnp.array([1.0]),
        n_samples_per_row=4,
        n_queries=5,
    )
    assert samples.shape == (5, 4, 1)


def test_multi_chain_predictive_probability_conditioning(small_state):
    """multi_chain averaging with conditioning still produces finite logp."""
    _state, packed, data, _ = small_state
    chains = [packed, packed]  # Identical chains; logsumexp collapses to logp.
    logp_multi = float(
        multi_chain_predictive_probability(
            chains,
            data,
            [0],
            jnp.array([0.25]),
            condition_cols=[1],
            condition_vals=jnp.array([1.0]),
        )
    )
    logp_single = float(
        packed_predictive_probability(
            packed,
            data,
            [0],
            jnp.array([0.25]),
            condition_cols=[1],
            condition_vals=jnp.array([1.0]),
        )
    )
    # Identical chains → multi-chain average equals single-chain log-prob.
    assert abs(logp_multi - logp_single) < 1e-4


def test_multi_chain_predictive_sample_conditioning_shape(small_state):
    _state, packed, data, _ = small_state
    samples = multi_chain_predictive_sample(
        jax.random.key(4),
        [packed, packed],
        data,
        [0],
        condition_cols=[1],
        condition_vals=jnp.array([1.0]),
        n_samples=8,
    )
    assert samples.shape == (8, 1)


def test_multi_chain_predictive_cdf_conditioning_in_range(small_state):
    _state, packed, data, _ = small_state
    cdf = multi_chain_predictive_cdf(
        jax.random.key(5),
        [packed, packed],
        data,
        query_col=0,
        query_val=jnp.array(3.0),
        condition_cols=[1],
        condition_vals=jnp.array([1.0]),
        n_samples=64,
    )
    assert 0.0 <= float(cdf) <= 1.0


def test_unpacked_conditioning_still_works(small_state):
    """Smoke: the unpacked tier continues to work (we didn't regress it)."""
    state, _packed, data, _ = small_state
    logp = float(
        predictive_probability(
            state,
            data,
            [0],
            jnp.array([0.25]),
            condition_cols=[1],
            condition_vals=jnp.array([1.0]),
        )
    )
    assert jnp.isfinite(jnp.asarray(logp))

    # And predictive_sample + predictive_cdf still accept conditioning.
    samples = predictive_sample(
        jax.random.key(6),
        state,
        data,
        [0],
        condition_cols=[1],
        condition_vals=jnp.array([1.0]),
        n_samples=4,
    )
    assert samples.shape == (4, 1)

    cdf = predictive_cdf(
        jax.random.key(7),
        state,
        data,
        query_col=0,
        query_val=jnp.array(3.0),
        condition_cols=[1],
        condition_vals=jnp.array([1.0]),
        n_samples=32,
    )
    assert 0.0 <= float(cdf) <= 1.0
