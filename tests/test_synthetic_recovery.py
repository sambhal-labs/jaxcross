"""Synthetic data recovery tests for jax-crosscat.

These tests verify that CrossCat inference can recover known generative parameters
from synthetic data. This is the standard validation approach used in the original
CrossCat paper and codebase (see probcomp/crosscat/tests/).

Pattern:
1. Generate data from known CrossCat parameters (views, clusters, component params)
2. Run inference for enough sweeps
3. Assert recovered structure matches ground truth within tolerance
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


def _run_multi_chain(rng_key, data, column_types, n_chains=4, n_sweeps=30):
    """Run multi-chain inference and return the best state by log_joint."""
    from crosscat.model import initialize, log_joint
    from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state

    states = initialize(rng_key, data, column_types, n_chains=n_chains)
    best_state, best_score = None, -jnp.inf
    for i, s in enumerate(states):
        key_i = jax.random.fold_in(rng_key, i + 1000)
        packed = pack_state(s)
        packed = packed_gibbs_sweep(key_i, packed, data, n_sweeps=n_sweeps)
        s = unpack_state(packed, column_types, data=data)
        score = log_joint(s, data)
        if score > best_score:
            best_score = score
            best_state = s
    return best_state


def test_initialization_from_prior(rng_key, synthetic_continuous_data):
    """Verify that initialization from prior produces valid state."""
    from crosscat.model import initialize

    state = initialize(
        rng_key,
        synthetic_continuous_data["data"],
        synthetic_continuous_data["column_types"],
    )
    assert state.n_rows == synthetic_continuous_data["n_rows"]
    assert state.n_cols == synthetic_continuous_data["n_cols"]
    assert state.column_assignments.shape == (synthetic_continuous_data["n_cols"],)
    assert state.n_views >= 1

    # Verify views have valid structure
    for view in state.views:
        assert view.row_assignments.shape == (state.n_rows,)
        assert len(view.column_indices) > 0
        assert view.suffstats is not None


def test_log_joint(rng_key, synthetic_continuous_data):
    """Verify that log_joint returns finite values."""
    from crosscat.model import initialize, log_joint

    state = initialize(
        rng_key,
        synthetic_continuous_data["data"],
        synthetic_continuous_data["column_types"],
    )
    lj = log_joint(state, synthetic_continuous_data["data"])
    assert jnp.isfinite(lj), f"log_joint returned {lj}"
    assert lj < 0, f"log_joint should be negative, got {lj}"


def test_component_suffstats(rng_key):
    """Verify sufficient statistics computation for each component model."""
    import jax

    from crosscat.components import BetaBernoulli, DirichletCategorical, NormalGamma

    k1, k2 = jax.random.split(rng_key)

    # NormalGamma
    data_cont = jax.random.normal(k1, (100,)) * 2.0 + 5.0
    ss = NormalGamma.sufficient_statistics(data_cont)
    assert int(ss.count) == 100
    assert jnp.isclose(ss.sum_x, data_cont.sum(), atol=1e-4)
    assert jnp.isclose(ss.sum_x_sq, (data_cont**2).sum(), atol=1e-4)

    # DirichletCategorical
    data_cat = jax.random.randint(k2, (50,), 0, 5)
    ss_cat = DirichletCategorical.sufficient_statistics(data_cat, 5)
    assert int(ss_cat.count) == 50
    assert int(ss_cat.category_counts.sum()) == 50

    # BetaBernoulli
    data_bin = jax.random.bernoulli(k1, 0.7, (80,)).astype(jnp.float32)
    ss_bin = BetaBernoulli.sufficient_statistics(data_bin)
    assert int(ss_bin.count) == 80
    assert jnp.isclose(ss_bin.sum_x, data_bin.sum())


def test_component_log_marginal(rng_key):
    """Verify log marginal likelihood is finite and reasonable."""
    import jax

    from crosscat.components import NormalGamma
    from crosscat.types import ColumnHypers, ColumnType

    data = jax.random.normal(rng_key, (100,)) * 2.0 + 5.0
    ss = NormalGamma.sufficient_statistics(data)
    hypers = ColumnHypers(
        column_type=ColumnType.CONTINUOUS,
        mu=jnp.array(5.0),
        r=jnp.array(1.0),
        s=jnp.array(4.0),
        nu=jnp.array(2.0),
    )
    lml = NormalGamma.log_marginal_likelihood(ss, hypers)
    assert jnp.isfinite(lml), f"log_marginal_likelihood returned {lml}"


def test_component_posterior_predictive(rng_key):
    """Verify posterior predictive returns valid log probabilities and samples."""
    import jax

    from crosscat.components import NormalGamma
    from crosscat.types import ColumnHypers, ColumnType

    data = jax.random.normal(rng_key, (100,)) * 2.0 + 5.0
    ss = NormalGamma.sufficient_statistics(data)
    hypers = ColumnHypers(
        column_type=ColumnType.CONTINUOUS,
        mu=jnp.array(5.0),
        r=jnp.array(1.0),
        s=jnp.array(4.0),
        nu=jnp.array(2.0),
    )

    # Log predictive density
    log_p = NormalGamma.posterior_predictive_logp(jnp.array(5.0), ss, hypers)
    assert jnp.isfinite(log_p)
    assert log_p < 0  # log probability

    # Samples
    samples = NormalGamma.sample_posterior_predictive(rng_key, ss, hypers, n=500)
    assert samples.shape == (500,)
    # Mean should be close to data mean
    assert abs(float(samples.mean()) - 5.0) < 2.0


def test_single_gibbs_sweep(rng_key, synthetic_continuous_data):
    """Verify that a single Gibbs sweep runs without error and produces valid state."""
    from crosscat.gibbs import gibbs_sweep
    from crosscat.model import initialize

    state = initialize(
        rng_key,
        synthetic_continuous_data["data"],
        synthetic_continuous_data["column_types"],
    )
    state_after = gibbs_sweep(rng_key, state, synthetic_continuous_data["data"], n_sweeps=1)

    assert state_after.n_rows == synthetic_continuous_data["n_rows"]
    assert state_after.n_cols == synthetic_continuous_data["n_cols"]
    assert state_after.n_views >= 1
    for view in state_after.views:
        assert view.row_assignments.shape == (state_after.n_rows,)


@pytest.mark.slow
def test_column_partition_recovery(rng_key, synthetic_continuous_data):
    """Verify that Gibbs inference recovers the true column partition.

    With well-separated clusters and enough sweeps, CrossCat should
    discover that columns 0,1 belong together and columns 2,3 belong
    together (two views).
    """
    state = _run_multi_chain(
        rng_key,
        synthetic_continuous_data["data"],
        synthetic_continuous_data["column_types"],
    )

    # Check column partition via ARI — should be high but not necessarily perfect
    from crosscat.diagnostics import column_partition_ari

    ari = float(column_partition_ari(state, synthetic_continuous_data["true_column_assignments"]))
    assert ari > 0.5, f"Column partition ARI {ari} <= 0.5"


@pytest.mark.slow
def test_row_cluster_recovery(rng_key, synthetic_continuous_data):
    """Verify that Gibbs inference recovers the true row clusters within views."""
    state = _run_multi_chain(
        rng_key,
        synthetic_continuous_data["data"],
        synthetic_continuous_data["column_types"],
    )

    # Each view should have discovered approximately 2 clusters (allow 2-4)
    for v_idx, view in enumerate(state.views):
        n_clusters = len(set(view.row_assignments.tolist()))
        assert 2 <= n_clusters <= 4, f"View {v_idx}: expected 2-4 clusters, got {n_clusters}"


@pytest.mark.slow
def test_posterior_predictive_accuracy(rng_key, synthetic_continuous_data):
    """Verify that posterior predictive samples are calibrated."""
    from crosscat.inference import predictive_sample

    state = _run_multi_chain(
        rng_key,
        synthetic_continuous_data["data"],
        synthetic_continuous_data["column_types"],
    )

    # Predict column 0 given column 1 = -2.0 (should predict ~0.0, cluster 0)
    samples = predictive_sample(
        rng_key,
        state,
        synthetic_continuous_data["data"],
        query_cols=[0],
        condition_cols=[1],
        condition_vals=jnp.array([-2.0]),
    )
    mean_prediction = samples.mean()
    assert abs(mean_prediction - 0.0) < 2.0, f"Expected ~0.0, got {mean_prediction}"
