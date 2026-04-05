"""Tests for miscellaneous inference queries.

Covers: row_similarity, observed vs unobserved row prediction,
impute_and_confidence, conditional_entropy, predictive_cdf,
joint_predictive_probability, sample_and_insert.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from crosscat.types import ColumnType

# --- Row Similarity ---


class TestRowSimilarity:
    def test_same_cluster_rows(self, simple_state):
        from crosscat.inference import row_similarity

        state, data, _ = simple_state
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
        samples_obs = predictive_sample(rng_key, state, data, [0], row_id=0, n_samples=100)
        assert samples_obs.shape == (100, 1)

    def test_unobserved_marginalizes(self, rng_key, simple_state):
        from crosscat.inference import predictive_sample

        state, data, _ = simple_state
        samples_unobs = predictive_sample(rng_key, state, data, [0], n_samples=100)
        assert samples_unobs.shape == (100, 1)


# --- Impute and Confidence ---


class TestImputeAndConfidence:
    def test_continuous_impute(self, rng_key, simple_state):
        from crosscat.inference import impute_and_confidence

        state, data, _ = simple_state
        val, conf = impute_and_confidence(rng_key, state, data, 0, n_samples=200)
        assert jnp.isfinite(val)
        assert 0.0 <= float(conf) <= 1.0


# --- Conditional Entropy ---


class TestConditionalEntropy:
    def test_conditional_entropy(self, rng_key, simple_state):
        from crosscat.inference import conditional_entropy

        state, data, _ = simple_state
        h = conditional_entropy(
            rng_key,
            [state],
            data,
            target_col=0,
            given_cols=[1],
            n_samples=50,
        )
        assert jnp.isfinite(h)
        assert float(h) >= 0  # Entropy is non-negative


# --- Predictive CDF ---


class TestPredictiveCDF:
    @pytest.mark.slow
    def test_continuous_cdf_monotone(self, rng_key, simple_state):
        from crosscat.inference import predictive_cdf

        state, data, _ = simple_state
        # Use the same key for all calls so the MC samples are consistent
        cdf_low = predictive_cdf(rng_key, state, data, 0, jnp.array(-100.0), n_samples=2000)
        cdf_mid = predictive_cdf(rng_key, state, data, 0, jnp.array(2.5), n_samples=2000)
        cdf_high = predictive_cdf(rng_key, state, data, 0, jnp.array(100.0), n_samples=2000)
        assert float(cdf_low) <= float(cdf_mid) <= float(cdf_high)
        assert float(cdf_low) < 0.5
        assert float(cdf_high) > 0.5

    @pytest.mark.slow
    def test_continuous_cdf_bounds(self, rng_key, simple_state):
        from crosscat.inference import predictive_cdf

        state, data, _ = simple_state
        cdf = predictive_cdf(rng_key, state, data, 0, jnp.array(0.0), n_samples=2000)
        assert 0.0 <= float(cdf) <= 1.0

    def test_categorical_cdf(self, rng_key):
        from crosscat.inference import predictive_cdf
        from crosscat.model import initialize

        data = jnp.array(
            [
                [0.0],
                [1.0],
                [2.0],
                [0.0],
                [1.0],
                [2.0],
                [0.0],
                [1.0],
                [2.0],
                [0.0],
            ]
        )
        column_types = [ColumnType.CATEGORICAL]
        state = initialize(rng_key, data, column_types).state
        cdf_all = predictive_cdf(rng_key, state, data, 0, jnp.array(2.0))
        assert float(cdf_all) > 0.99

    def test_binary_cdf(self, rng_key):
        from crosscat.inference import predictive_cdf
        from crosscat.model import initialize

        data = jnp.array([[0.0], [1.0], [0.0], [1.0], [0.0]])
        column_types = [ColumnType.BINARY]
        state = initialize(rng_key, data, column_types).state
        cdf_1 = predictive_cdf(rng_key, state, data, 0, jnp.array(1.0))
        assert float(cdf_1) > 0.99


# --- Joint Predictive Probability ---


class TestJointPredictiveProb:
    def test_chain_rule(self, simple_state):
        from crosscat.inference import joint_predictive_probability

        state, data, _ = simple_state
        log_p_joint = joint_predictive_probability(state, data, [0, 1], jnp.array([0.0, -2.0]))
        assert jnp.isfinite(log_p_joint)
        assert log_p_joint < 0  # log probability


# --- Sample and Insert ---


class TestSampleAndInsert:
    def test_basic(self, rng_key, simple_state):
        from crosscat.inference import sample_and_insert

        state, data, _ = simple_state
        partial = jnp.array([1.0, jnp.nan, 3.0, jnp.nan])
        new_state, new_data, completed = sample_and_insert(rng_key, state, data, partial)
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
        new_state, new_data, completed = sample_and_insert(rng_key, state, data, full_row)
        assert new_state.n_rows == state.n_rows + 1
        assert jnp.allclose(completed, full_row)

    def test_all_missing(self, rng_key, simple_state):
        from crosscat.inference import sample_and_insert

        state, data, _ = simple_state
        all_nan = jnp.full(4, jnp.nan)
        new_state, new_data, completed = sample_and_insert(rng_key, state, data, all_nan)
        assert new_state.n_rows == state.n_rows + 1
        assert jnp.all(jnp.isfinite(completed))
