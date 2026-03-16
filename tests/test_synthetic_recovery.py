"""Synthetic data recovery tests for jax-crosscat.

These tests verify that CrossCat inference can recover known generative parameters
from synthetic data. This is the standard validation approach used in the original
CrossCat paper and codebase (see probcomp/crosscat/tests/).

Pattern:
1. Generate data from known CrossCat parameters (views, clusters, component params)
2. Run inference for enough sweeps
3. Assert recovered structure matches ground truth within tolerance

All tests are currently skipped — they will be enabled as inference is implemented.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest


@pytest.mark.skip(reason="Requires model.initialize — Week 2-4")
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


@pytest.mark.skip(reason="Requires gibbs.gibbs_sweep — Week 4-6")
def test_column_partition_recovery(rng_key, synthetic_continuous_data):
    """Verify that Gibbs inference recovers the true column partition.

    With well-separated clusters and enough sweeps, CrossCat should
    discover that columns 0,1 belong together and columns 2,3 belong
    together (two views).
    """
    from crosscat.gibbs import gibbs_sweep
    from crosscat.model import initialize

    state = initialize(
        rng_key,
        synthetic_continuous_data["data"],
        synthetic_continuous_data["column_types"],
    )
    state = gibbs_sweep(rng_key, state, synthetic_continuous_data["data"], n_sweeps=100)

    # Check column partition: cols 0,1 should be in same view, cols 2,3 in same view
    assert state.column_assignments[0] == state.column_assignments[1]
    assert state.column_assignments[2] == state.column_assignments[3]
    assert state.column_assignments[0] != state.column_assignments[2]


@pytest.mark.skip(reason="Requires gibbs.gibbs_sweep — Week 4-6")
def test_row_cluster_recovery(rng_key, synthetic_continuous_data):
    """Verify that Gibbs inference recovers the true row clusters within views."""
    from crosscat.gibbs import gibbs_sweep
    from crosscat.model import initialize

    state = initialize(
        rng_key,
        synthetic_continuous_data["data"],
        synthetic_continuous_data["column_types"],
    )
    state = gibbs_sweep(rng_key, state, synthetic_continuous_data["data"], n_sweeps=200)

    # Each view should have discovered 2 clusters
    for view in state.views:
        n_clusters = len(set(view.row_assignments.tolist()))
        assert n_clusters == 2, f"Expected 2 clusters, got {n_clusters}"


@pytest.mark.skip(reason="Requires inference.predictive_sample — Week 4-6")
def test_posterior_predictive_accuracy(rng_key, synthetic_continuous_data):
    """Verify that posterior predictive samples are calibrated."""
    from crosscat.gibbs import gibbs_sweep
    from crosscat.inference import predictive_sample
    from crosscat.model import initialize

    state = initialize(
        rng_key,
        synthetic_continuous_data["data"],
        synthetic_continuous_data["column_types"],
    )
    state = gibbs_sweep(rng_key, state, synthetic_continuous_data["data"], n_sweeps=200)

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
    assert abs(mean_prediction - 0.0) < 1.0, f"Expected ~0.0, got {mean_prediction}"
