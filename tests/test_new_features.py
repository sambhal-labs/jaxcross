"""Tests for newly implemented CrossCat features.

Covers: NaN handling, VonMises, row_similarity, observed row distinction,
ARI diagnostics, impute_and_confidence, MH transitions, initialization modes,
row insertion, joint predictive probability, data utilities, constraints,
predictive anomalousness, synthetic data generator, state validation,
nu/df grid sampling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.types import ColumnHypers, ColumnType


@pytest.fixture
def rng_key():
    return jax.random.key(42)


@pytest.fixture
def simple_state(rng_key):
    """A simple 2-view state for testing queries."""
    from crosscat.model import initialize

    n_rows = 100
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    col0 = jnp.where(jnp.arange(n_rows) < 50, 0.0, 5.0) + jax.random.normal(k1, (n_rows,))
    col1 = jnp.where(jnp.arange(n_rows) < 50, -2.0, 3.0) + jax.random.normal(k2, (n_rows,))
    col2 = jnp.where(jnp.arange(n_rows) < 50, 10.0, 20.0) + jax.random.normal(k3, (n_rows,))
    col3 = jnp.where(jnp.arange(n_rows) < 50, -5.0, 5.0) + jax.random.normal(k4, (n_rows,))
    data = jnp.column_stack([col0, col1, col2, col3])
    column_types = [ColumnType.CONTINUOUS] * 4
    state = initialize(rng_key, data, column_types)
    return state, data, column_types


# --- NaN handling ---

class TestNaNHandling:
    def test_normal_gamma_nan(self):
        from crosscat.components import NormalGamma

        data = jnp.array([1.0, 2.0, jnp.nan, 4.0, jnp.nan])
        ss = NormalGamma.sufficient_statistics(data)
        assert int(ss.count) == 3  # only non-NaN
        assert jnp.isclose(ss.sum_x, 7.0, atol=1e-4)

    def test_dirichlet_categorical_nan(self):
        from crosscat.components import DirichletCategorical

        data = jnp.array([0.0, 1.0, jnp.nan, 2.0, jnp.nan])
        ss = DirichletCategorical.sufficient_statistics(data, 3)
        assert int(ss.count) == 3

    def test_beta_bernoulli_nan(self):
        from crosscat.components import BetaBernoulli

        data = jnp.array([1.0, 0.0, jnp.nan, 1.0])
        ss = BetaBernoulli.sufficient_statistics(data)
        assert int(ss.count) == 3
        assert jnp.isclose(ss.sum_x, 2.0)

    def test_ordinal_nan(self):
        from crosscat.components import OrderedLogistic

        data = jnp.array([0.0, jnp.nan, 2.0, 1.0])
        ss = OrderedLogistic.sufficient_statistics(data, 3)
        assert int(ss.count) == 3


# --- Von Mises (Cyclic) ---

class TestVonMises:
    def test_suffstats(self):
        from crosscat.components import VonMises

        data = jnp.array([0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2])
        ss = VonMises.sufficient_statistics(data)
        assert int(ss.count) == 4
        assert ss.sum_sin is not None
        assert ss.sum_cos is not None

    def test_log_marginal(self):
        from crosscat.components import VonMises

        data = jnp.array([0.1, 0.2, 0.3, 0.15])
        ss = VonMises.sufficient_statistics(data)
        hypers = ColumnHypers(
            column_type=ColumnType.CYCLIC, kappa=jnp.array(2.0), vm_mu=jnp.array(0.0)
        )
        lml = VonMises.log_marginal_likelihood(ss, hypers)
        assert jnp.isfinite(lml)

    def test_posterior_predictive(self):
        from crosscat.components import VonMises

        data = jnp.array([0.1, 0.2, 0.3])
        ss = VonMises.sufficient_statistics(data)
        hypers = ColumnHypers(
            column_type=ColumnType.CYCLIC, kappa=jnp.array(2.0), vm_mu=jnp.array(0.0)
        )
        log_p = VonMises.posterior_predictive_logp(jnp.array(0.15), ss, hypers)
        assert jnp.isfinite(log_p)

    def test_sample(self, rng_key):
        from crosscat.components import VonMises

        data = jnp.array([0.1, 0.2, 0.3])
        ss = VonMises.sufficient_statistics(data)
        hypers = ColumnHypers(
            column_type=ColumnType.CYCLIC, kappa=jnp.array(5.0), vm_mu=jnp.array(0.2)
        )
        samples = VonMises.sample_posterior_predictive(rng_key, ss, hypers, n=100)
        assert samples.shape == (100,)
        # All samples should be in [0, 2*pi)
        assert jnp.all(samples >= 0)
        assert jnp.all(samples < 2 * jnp.pi)


# --- Row Similarity ---

class TestRowSimilarity:
    def test_same_cluster_rows(self, simple_state):
        from crosscat.inference import row_similarity

        state, data, _ = simple_state
        # Rows in same cluster should have higher similarity
        sim = row_similarity([state], 0, 1)
        assert 0.0 <= float(sim) <= 1.0

    def test_self_similarity(self, simple_state):
        from crosscat.inference import row_similarity

        state, data, _ = simple_state
        sim = row_similarity([state], 0, 0)
        assert float(sim) == 1.0  # Same row = always same cluster

    def test_target_columns(self, simple_state):
        from crosscat.inference import row_similarity

        state, data, _ = simple_state
        sim = row_similarity([state], 0, 50, target_columns=[0])
        assert 0.0 <= float(sim) <= 1.0


# --- Observed vs Unobserved Row ---

class TestObservedRowDistinction:
    def test_observed_row_prediction(self, rng_key, simple_state):
        from crosscat.inference import predictive_sample

        state, data, _ = simple_state
        # With row_id, should use actual cluster
        samples_obs = predictive_sample(
            rng_key, state, data, [0], row_id=0, n_samples=100
        )
        assert samples_obs.shape == (100, 1)

    def test_unobserved_marginalizes(self, rng_key, simple_state):
        from crosscat.inference import predictive_sample

        state, data, _ = simple_state
        # Without row_id, marginalizes over clusters
        samples_unobs = predictive_sample(
            rng_key, state, data, [0], n_samples=100
        )
        assert samples_unobs.shape == (100, 1)


# --- Convergence Diagnostics ---

class TestDiagnostics:
    def test_ari_perfect(self):
        from crosscat.diagnostics import adjusted_rand_index

        a = jnp.array([0, 0, 1, 1, 2, 2])
        ari = adjusted_rand_index(a, a)
        assert jnp.isclose(ari, 1.0, atol=1e-5)

    def test_ari_permutation(self):
        from crosscat.diagnostics import adjusted_rand_index

        a = jnp.array([0, 0, 1, 1])
        b = jnp.array([1, 1, 0, 0])  # Same partition, different labels
        ari = adjusted_rand_index(a, b)
        assert jnp.isclose(ari, 1.0, atol=1e-5)

    def test_collect_diagnostics(self, simple_state):
        from crosscat.diagnostics import collect_diagnostics

        state, data, _ = simple_state
        diag = collect_diagnostics(state, data)
        assert "log_joint" in diag
        assert "n_views" in diag
        assert "column_crp_alpha" in diag
        assert jnp.isfinite(diag["log_joint"])


# --- Impute and Confidence ---

class TestImputeAndConfidence:
    def test_continuous_impute(self, rng_key, simple_state):
        from crosscat.inference import impute_and_confidence

        state, data, _ = simple_state
        val, conf = impute_and_confidence(rng_key, state, data, 0, n_samples=200)
        assert jnp.isfinite(val)
        assert 0.0 <= float(conf) <= 1.0


# --- Initialization Modes ---

class TestInitializationModes:
    def test_together(self, rng_key):
        from crosscat.model import initialize

        data = jax.random.normal(rng_key, (50, 4))
        column_types = [ColumnType.CONTINUOUS] * 4
        state = initialize(rng_key, data, column_types, initialization="together")
        assert state.n_views == 1

    def test_apart(self, rng_key):
        from crosscat.model import initialize

        data = jax.random.normal(rng_key, (50, 4))
        column_types = [ColumnType.CONTINUOUS] * 4
        state = initialize(rng_key, data, column_types, initialization="apart")
        assert state.n_views == 4

    def test_from_the_prior(self, rng_key):
        from crosscat.model import initialize

        data = jax.random.normal(rng_key, (50, 4))
        column_types = [ColumnType.CONTINUOUS] * 4
        state = initialize(rng_key, data, column_types, initialization="from_the_prior")
        assert state.n_views >= 1


# --- Row Insertion ---

class TestRowInsertion:
    def test_insert_rows(self, rng_key, simple_state):
        from crosscat.model import insert_rows

        state, data, _ = simple_state
        new_rows = jax.random.normal(rng_key, (5, 4))
        new_state, new_data = insert_rows(rng_key, state, data, new_rows)
        assert new_state.n_rows == state.n_rows + 5
        assert new_data.shape[0] == data.shape[0] + 5
        for view in new_state.views:
            assert view.row_assignments.shape[0] == new_state.n_rows


# --- Joint Predictive Probability ---

class TestJointPredictiveProb:
    def test_chain_rule(self, simple_state):
        from crosscat.inference import joint_predictive_probability

        state, data, _ = simple_state
        # Joint probability of two query values
        log_p_joint = joint_predictive_probability(
            state, data, [0, 1], jnp.array([0.0, -2.0])
        )
        assert jnp.isfinite(log_p_joint)
        assert log_p_joint < 0  # log probability


# --- Data Utilities ---

class TestDataUtils:
    def test_guess_column_types(self):
        from crosscat.data_utils import guess_column_types

        data = jnp.column_stack([
            jnp.array([1.1, 2.3, 3.5, 4.7, 5.9]),
            jnp.array([0.0, 1.0, 0.0, 1.0, 0.0]),
            jnp.array([0.0, 1.0, 2.0, 0.0, 1.0]),
        ])
        types = guess_column_types(data)
        assert types[0] == ColumnType.CONTINUOUS
        assert types[1] == ColumnType.BINARY

    def test_gen_column_metadata(self):
        from crosscat.data_utils import gen_column_metadata

        data = jnp.ones((5, 3))
        types = [ColumnType.CONTINUOUS, ColumnType.BINARY, ColumnType.CATEGORICAL]
        meta = gen_column_metadata(data, types, ["a", "b", "c"])
        assert meta["name_to_idx"]["a"] == 0
        assert len(meta["column_metadata"]) == 3


# --- Synthetic Data Generator ---

class TestSyntheticGenerator:
    def test_generate(self, rng_key):
        from crosscat.synthetic import generate_crosscat_data

        result = generate_crosscat_data(
            rng_key,
            n_rows=100,
            column_types=[ColumnType.CONTINUOUS, ColumnType.CONTINUOUS, ColumnType.BINARY],
            n_views=2,
            n_clusters=2,
        )
        assert result["data"].shape == (100, 3)
        assert len(result["true_row_assignments"]) == 2
        assert result["true_column_assignments"].shape == (3,)

    def test_add_missing(self, rng_key):
        from crosscat.synthetic import add_missing_data

        data = jnp.ones((50, 3))
        data_with_nan = add_missing_data(rng_key, data, missing_fraction=0.2)
        n_nan = jnp.sum(jnp.isnan(data_with_nan))
        assert int(n_nan) > 0


# --- State Validation ---

class TestValidation:
    def test_valid_state(self, simple_state):
        from crosscat.validate import validate_state

        state, data, _ = simple_state
        errors = validate_state(state, data)
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_assert_valid(self, simple_state):
        from crosscat.validate import assert_valid_state

        state, data, _ = simple_state
        assert_valid_state(state, data)  # Should not raise


# --- MH Column Transitions ---

class TestMHTransition:
    def test_mh_column_sweep(self, rng_key, simple_state):
        from crosscat.gibbs import transition_column_assignments_mh

        state, data, _ = simple_state
        new_state = transition_column_assignments_mh(rng_key, state, data)
        assert new_state.n_rows == state.n_rows
        assert new_state.n_cols == state.n_cols
        assert new_state.n_views >= 1

    def test_mh_via_gibbs_sweep(self, rng_key, simple_state):
        from crosscat.gibbs import gibbs_sweep

        state, data, _ = simple_state
        new_state = gibbs_sweep(
            rng_key, state, data, n_sweeps=1,
            kernels=("row_assignments", "column_assignments_mh", "crp_alphas"),
        )
        assert new_state.n_views >= 1


# --- Conditional Entropy ---

class TestConditionalEntropy:
    def test_conditional_entropy(self, rng_key, simple_state):
        from crosscat.inference import conditional_entropy

        state, data, _ = simple_state
        h = conditional_entropy(
            rng_key, [state], data,
            target_col=0, given_cols=[1],
            n_samples=50,
        )
        assert jnp.isfinite(h)
        assert float(h) >= 0  # Entropy is non-negative


# --- Predictive CDF ---

class TestPredictiveCDF:
    def test_continuous_cdf_monotone(self, rng_key, simple_state):
        from crosscat.inference import predictive_cdf

        state, data, _ = simple_state
        k1, k2, k3 = jax.random.split(rng_key, 3)
        cdf_low = predictive_cdf(k1, state, data, 0, jnp.array(-10.0))
        cdf_mid = predictive_cdf(k2, state, data, 0, jnp.array(2.5))
        cdf_high = predictive_cdf(k3, state, data, 0, jnp.array(20.0))
        assert float(cdf_low) <= float(cdf_mid) <= float(cdf_high)
        assert float(cdf_low) < 0.5
        assert float(cdf_high) > 0.5

    def test_continuous_cdf_bounds(self, rng_key, simple_state):
        from crosscat.inference import predictive_cdf

        state, data, _ = simple_state
        cdf = predictive_cdf(rng_key, state, data, 0, jnp.array(0.0))
        assert 0.0 <= float(cdf) <= 1.0

    def test_categorical_cdf(self, rng_key):
        from crosscat.inference import predictive_cdf
        from crosscat.model import initialize

        data = jnp.array([
            [0.0], [1.0], [2.0], [0.0], [1.0],
            [2.0], [0.0], [1.0], [2.0], [0.0],
        ])
        column_types = [ColumnType.CATEGORICAL]
        state = initialize(rng_key, data, column_types)
        cdf_all = predictive_cdf(
            rng_key, state, data, 0, jnp.array(2.0)
        )
        # CDF at max category should be ~1.0
        assert float(cdf_all) > 0.99

    def test_binary_cdf(self, rng_key):
        from crosscat.inference import predictive_cdf
        from crosscat.model import initialize

        data = jnp.array([[0.0], [1.0], [0.0], [1.0], [0.0]])
        column_types = [ColumnType.BINARY]
        state = initialize(rng_key, data, column_types)
        cdf_1 = predictive_cdf(rng_key, state, data, 0, jnp.array(1.0))
        assert float(cdf_1) > 0.99


# --- Sample and Insert ---

class TestSampleAndInsert:
    def test_basic(self, rng_key, simple_state):
        from crosscat.inference import sample_and_insert

        state, data, _ = simple_state
        partial = jnp.array([1.0, jnp.nan, 3.0, jnp.nan])
        new_state, new_data, completed = sample_and_insert(
            rng_key, state, data, partial
        )
        assert new_state.n_rows == state.n_rows + 1
        assert new_data.shape[0] == data.shape[0] + 1
        # Observed values preserved
        assert jnp.isclose(completed[0], 1.0)
        assert jnp.isclose(completed[2], 3.0)
        # Missing values filled
        assert jnp.isfinite(completed[1])
        assert jnp.isfinite(completed[3])

    def test_no_missing(self, rng_key, simple_state):
        from crosscat.inference import sample_and_insert

        state, data, _ = simple_state
        full_row = jnp.array([1.0, 2.0, 3.0, 4.0])
        new_state, new_data, completed = sample_and_insert(
            rng_key, state, data, full_row
        )
        assert new_state.n_rows == state.n_rows + 1
        assert jnp.allclose(completed, full_row)

    def test_all_missing(self, rng_key, simple_state):
        from crosscat.inference import sample_and_insert

        state, data, _ = simple_state
        all_nan = jnp.full(4, jnp.nan)
        new_state, new_data, completed = sample_and_insert(
            rng_key, state, data, all_nan
        )
        assert new_state.n_rows == state.n_rows + 1
        assert jnp.all(jnp.isfinite(completed))
