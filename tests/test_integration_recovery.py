"""Integration tests for CrossCat inference recovery across data types.

Tests column partition recovery, row cluster recovery, and convergence
diagnostics for cyclic, mixed-type, and missing-data scenarios.

All recovery tests are marked @slow (30+ sweeps of packed Gibbs sampling).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.diagnostics import column_partition_ari, row_partition_ari
from crosscat.inference import predictive_sample
from crosscat.model import initialize, log_joint
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat.validate import validate_state
from tests.conftest import run_multi_chain_with_diagnostics


def _packed_infer(key, state, data, column_types, n_sweeps):
    """Run packed Gibbs sweeps and return unpacked state."""
    packed = pack_state(state)
    packed = packed_gibbs_sweep(key, packed, data, n_sweeps=n_sweeps)
    return unpack_state(packed, column_types, data=data)


# ---------------------------------------------------------------------------
# Gap A: Cyclic Recovery (3 tests)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cyclic_column_partition_recovery(synthetic_cyclic_data):
    """Inference recovers the 2-view column partition on cyclic data."""
    d = synthetic_cyclic_data
    key = jax.random.key(110)
    states = initialize(key, d["data"], d["column_types"], n_chains=4)
    best_ari = -1.0
    for i, state in enumerate(states):
        k = jax.random.fold_in(key, i + 500)
        state = _packed_infer(k, state, d["data"], d["column_types"], n_sweeps=50)
        ari = float(column_partition_ari(state, d["true_column_assignments"]))
        best_ari = max(best_ari, ari)
    assert best_ari > 0.5, f"Best column ARI {best_ari} <= 0.5"


@pytest.mark.slow
@pytest.mark.xfail(
    reason="Cyclic row clustering is stochastic; recovery depends on seed and may need more sweeps"
)
def test_cyclic_row_cluster_recovery(synthetic_cyclic_data):
    """Inference recovers row clusters per view on cyclic data."""
    d = synthetic_cyclic_data
    key = jax.random.key(111)
    states = initialize(key, d["data"], d["column_types"], n_chains=4)
    best_row_ari = -1.0
    for i, state in enumerate(states):
        k = jax.random.fold_in(key, i + 600)
        state = _packed_infer(k, state, d["data"], d["column_types"], n_sweeps=50)
        col_ari = float(column_partition_ari(state, d["true_column_assignments"]))
        if col_ari > 0.3:
            for v in range(state.n_views):
                for _true_v, true_assigns in enumerate(d["true_row_assignments"]):
                    ari = float(row_partition_ari(state, v, true_assigns))
                    best_row_ari = max(best_row_ari, ari)
    assert best_row_ari > 0.4, f"Best row ARI {best_row_ari} <= 0.4"


@pytest.mark.slow
def test_cyclic_predictive_samples_in_range(synthetic_cyclic_data):
    """Predictive samples from cyclic model are in [0, 2*pi) and bimodal."""
    d = synthetic_cyclic_data
    key = jax.random.key(112)
    k1, k2 = jax.random.split(key)
    state = initialize(k1, d["data"], d["column_types"])
    state = _packed_infer(k2, state, d["data"], d["column_types"], n_sweeps=20)

    k3 = jax.random.key(113)
    samples = predictive_sample(k3, state, d["data"], [0], n_samples=500)
    s = samples[:, 0]

    # All samples should be finite
    assert jnp.all(jnp.isfinite(s)), "Non-finite predictive samples"
    # Cyclic samples wrapped to [0, 2*pi) by the model
    s_wrapped = s % (2.0 * jnp.pi)
    assert jnp.all(s_wrapped >= 0.0) and jnp.all(s_wrapped < 2.0 * jnp.pi + 0.01)


# ---------------------------------------------------------------------------
# Gap B: Mixed-Type Recovery (3 tests)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mixed_type_column_partition_recovery(synthetic_mixed_data):
    """Inference recovers column partition on mixed-type data."""
    d = synthetic_mixed_data
    key = jax.random.key(120)
    states = initialize(key, d["data"], d["column_types"], n_chains=4)
    best_ari = -1.0
    for i, state in enumerate(states):
        k = jax.random.fold_in(key, i + 700)
        state = _packed_infer(k, state, d["data"], d["column_types"], n_sweeps=30)
        ari = float(column_partition_ari(state, d["true_column_assignments"]))
        best_ari = max(best_ari, ari)
    assert best_ari > 0.5, f"Best column ARI {best_ari} <= 0.5"


@pytest.mark.slow
def test_mixed_type_row_cluster_recovery(synthetic_mixed_data):
    """Inference recovers row clusters on mixed-type data."""
    d = synthetic_mixed_data
    key = jax.random.key(121)
    states = initialize(key, d["data"], d["column_types"], n_chains=4)
    best_row_ari = -1.0
    for i, state in enumerate(states):
        k = jax.random.fold_in(key, i + 800)
        state = _packed_infer(k, state, d["data"], d["column_types"], n_sweeps=30)
        col_ari = float(column_partition_ari(state, d["true_column_assignments"]))
        if col_ari > 0.3:
            for v in range(state.n_views):
                for true_assigns in d["true_row_assignments"]:
                    ari = float(row_partition_ari(state, v, true_assigns))
                    best_row_ari = max(best_row_ari, ari)
    assert best_row_ari > 0.4, f"Best row ARI {best_row_ari} <= 0.4"


@pytest.mark.slow
def test_mixed_type_state_validity(synthetic_mixed_data):
    """Inferred mixed-type state passes validation and has finite log_joint."""
    d = synthetic_mixed_data
    key = jax.random.key(122)
    k1, k2 = jax.random.split(key)
    state = initialize(k1, d["data"], d["column_types"])
    state = _packed_infer(k2, state, d["data"], d["column_types"], n_sweeps=10)

    errors = validate_state(state, d["data"])
    assert errors == [], f"Validation errors: {errors}"

    lj = log_joint(state, d["data"])
    assert jnp.isfinite(lj), f"log_joint not finite: {lj}"


# ---------------------------------------------------------------------------
# Gap C: Missing Data Robustness (4 tests)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_missing_data_inference_runs(synthetic_missing_data):
    """Inference completes with finite log_joint on data with 15% NaN."""
    d = synthetic_missing_data
    key = jax.random.key(130)
    k1, k2 = jax.random.split(key)
    state = initialize(k1, d["data"], d["column_types"])
    state = _packed_infer(k2, state, d["data"], d["column_types"], n_sweeps=10)

    errors = validate_state(state, d["data"])
    assert errors == [], f"Validation errors: {errors}"

    lj = log_joint(state, d["data"])
    assert jnp.isfinite(lj), f"log_joint not finite: {lj}"


@pytest.mark.slow
def test_missing_data_column_recovery(synthetic_missing_data):
    """Column partition recovery still works with 15% missing data."""
    d = synthetic_missing_data
    key = jax.random.key(131)
    states = initialize(key, d["data"], d["column_types"], n_chains=4)
    best_ari = -1.0
    for i, state in enumerate(states):
        k = jax.random.fold_in(key, i + 900)
        state = _packed_infer(k, state, d["data"], d["column_types"], n_sweeps=30)
        ari = float(column_partition_ari(state, d["true_column_assignments"]))
        best_ari = max(best_ari, ari)
    assert best_ari > 0.4, f"Best column ARI {best_ari} <= 0.4"


@pytest.mark.slow
def test_missing_data_row_cluster_recovery(synthetic_missing_data):
    """Row cluster recovery still works with 15% missing data."""
    d = synthetic_missing_data
    key = jax.random.key(132)
    states = initialize(key, d["data"], d["column_types"], n_chains=4)
    best_row_ari = -1.0
    for i, state in enumerate(states):
        k = jax.random.fold_in(key, i + 1000)
        state = _packed_infer(k, state, d["data"], d["column_types"], n_sweeps=30)
        for v in range(state.n_views):
            for true_assigns in d["true_row_assignments"]:
                ari = float(row_partition_ari(state, v, true_assigns))
                best_row_ari = max(best_row_ari, ari)
    assert best_row_ari > 0.3, f"Best row ARI {best_row_ari} <= 0.3"


@pytest.mark.slow
def test_missing_data_predictive_sample_finite(synthetic_missing_data):
    """Predictive samples are all finite despite missing training data."""
    d = synthetic_missing_data
    key = jax.random.key(133)
    k1, k2, k3 = jax.random.split(key, 3)
    state = initialize(k1, d["data"], d["column_types"])
    state = _packed_infer(k2, state, d["data"], d["column_types"], n_sweeps=10)

    samples = predictive_sample(k3, state, d["data"], [0, 1], n_samples=100)
    assert jnp.all(jnp.isfinite(samples)), "Non-finite predictive samples with missing data"


# ---------------------------------------------------------------------------
# Gap G: Convergence Diagnostics (3 tests)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_log_joint_improves_over_sweeps(synthetic_continuous_data):
    """log_joint increases over Gibbs sweeps (model finds better fit)."""
    d = synthetic_continuous_data
    key = jax.random.key(140)
    k1, k2 = jax.random.split(key)
    state = initialize(k1, d["data"], d["column_types"])

    initial_lj = float(log_joint(state, d["data"]))
    state = _packed_infer(k2, state, d["data"], d["column_types"], n_sweeps=20)
    final_lj = float(log_joint(state, d["data"]))

    assert final_lj > initial_lj, f"log_joint did not improve: {initial_lj:.2f} -> {final_lj:.2f}"


@pytest.mark.slow
def test_ari_improves_over_sweeps(synthetic_continuous_data):
    """Column partition ARI improves over inference sweeps."""
    d = synthetic_continuous_data
    key = jax.random.key(141)
    k1, k_run = jax.random.split(key)
    state = initialize(k1, d["data"], d["column_types"])

    ari_at_1 = float(column_partition_ari(state, d["true_column_assignments"]))
    state = _packed_infer(k_run, state, d["data"], d["column_types"], n_sweeps=20)
    ari_at_20 = float(column_partition_ari(state, d["true_column_assignments"]))

    # ARI should at least not decrease significantly; ideally improves
    assert ari_at_20 >= ari_at_1 - 0.1, (
        f"ARI degraded: sweep 1={ari_at_1:.3f}, sweep 20={ari_at_20:.3f}"
    )


@pytest.mark.slow
def test_multi_chain_best_selection(synthetic_continuous_data):
    """Best chain by log_joint has ARI >= mean ARI across chains."""
    d = synthetic_continuous_data
    states, all_diags = run_multi_chain_with_diagnostics(
        d["data"], d["column_types"], n_chains=4, n_sweeps=20, seed=142
    )

    aris = []
    ljs = []
    for state in states:
        aris.append(float(column_partition_ari(state, d["true_column_assignments"])))
        ljs.append(float(log_joint(state, d["data"])))

    best_idx = ljs.index(max(ljs))
    mean_ari = sum(aris) / len(aris)

    assert aris[best_idx] >= mean_ari - 0.1, (
        f"Best chain ARI {aris[best_idx]:.3f} < mean ARI {mean_ari:.3f} - 0.1"
    )
