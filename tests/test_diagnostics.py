"""Tests for convergence diagnostics.

Covers: Adjusted Rand Index, collect_diagnostics, random_holdout_mask,
mean_test_log_likelihood, evaluate_imputation, gelman_rubin_rhat,
effective_sample_size.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.diagnostics import (
    adjusted_rand_index,
    collect_diagnostics,
    effective_sample_size,
    evaluate_imputation,
    gelman_rubin_rhat,
    mean_test_log_likelihood,
    random_holdout_mask,
)
from crosscat.model import initialize
from crosscat.types import ColumnType


class TestAdjustedRandIndex:
    def test_perfect_agreement(self):
        a = jnp.array([0, 0, 1, 1, 2, 2])
        ari = adjusted_rand_index(a, a)
        assert jnp.isclose(ari, 1.0, atol=1e-5)

    def test_permutation_invariance(self):
        a = jnp.array([0, 0, 1, 1])
        b = jnp.array([1, 1, 0, 0])  # Same partition, different labels
        ari = adjusted_rand_index(a, b)
        assert jnp.isclose(ari, 1.0, atol=1e-5)

    def test_random_partitions_low_ari(self):
        """Random partitions should have ARI near 0."""
        a = jnp.array([0, 1, 0, 1, 0, 1, 0, 1])
        b = jnp.array([0, 0, 1, 1, 0, 0, 1, 1])
        ari = adjusted_rand_index(a, b)
        assert float(ari) < 0.5


class TestCollectDiagnostics:
    def test_keys_present(self, simple_state):
        state, data, _ = simple_state
        diag = collect_diagnostics(state, data)
        assert "log_joint" in diag
        assert "n_views" in diag
        assert "column_crp_alpha" in diag
        assert jnp.isfinite(diag["log_joint"])


class TestRandomHoldoutMask:
    def test_shape(self):
        key = jax.random.key(42)
        mask = random_holdout_mask(key, 100, 5)
        assert mask.shape == (100, 5)
        assert mask.dtype == jnp.bool_

    def test_fraction(self):
        key = jax.random.key(43)
        mask = random_holdout_mask(key, 1000, 10, holdout_fraction=0.2)
        frac = float(jnp.mean(mask.astype(jnp.float32)))
        assert 0.15 < frac < 0.25  # within reasonable range


class TestMeanTestLogLikelihood:
    def test_returns_finite(self):
        """mean_test_log_likelihood returns a finite scalar."""
        key = jax.random.key(44)
        data = jax.random.normal(key, (30, 3))
        column_types = [ColumnType.CONTINUOUS] * 3
        state = initialize(jax.random.key(45), data, column_types).state
        test_rows = jnp.array([25, 26, 27, 28, 29])
        ll = mean_test_log_likelihood(state, data, test_rows)
        assert jnp.isfinite(ll)
        assert float(ll) < 0  # log-likelihoods are negative


class TestEvaluateImputation:
    def test_returns_dict_with_keys(self):
        """evaluate_imputation returns dict with expected keys."""
        key = jax.random.key(46)
        data = jax.random.normal(key, (30, 3))
        column_types = [ColumnType.CONTINUOUS] * 3
        state = initialize(jax.random.key(47), data, column_types).state
        mask = random_holdout_mask(jax.random.key(48), 30, 3, holdout_fraction=0.1)
        result = evaluate_imputation(state, data, mask, column_types, rng_key=jax.random.key(49))
        assert "mae" in result or "accuracy" in result
        assert "mean_log_likelihood" in result


# ---------------------------------------------------------------------------
# Tests for gelman_rubin_rhat and effective_sample_size
# ---------------------------------------------------------------------------




class TestGelmanRubinRhat:
    def test_converged_chains_near_one(self):
        """Converged chains (same distribution) should give R-hat ≈ 1.0."""
        key = jax.random.key(100)
        traces = jax.random.normal(key, (4, 200))
        rhat = gelman_rubin_rhat(traces)
        assert float(rhat) >= 1.0
        assert float(rhat) < 1.1

    def test_divergent_chains_above_threshold(self):
        """Chains at different means should give R-hat >> 1.1."""
        traces = jnp.array(
            [
                jnp.ones(100) * 0.0 + jax.random.normal(jax.random.key(1), (100,)) * 0.1,
                jnp.ones(100) * 10.0 + jax.random.normal(jax.random.key(2), (100,)) * 0.1,
            ]
        )
        rhat = gelman_rubin_rhat(traces)
        assert float(rhat) > 1.5

    def test_constant_chains_rhat_one(self):
        """Constant chains (stuck) should give R-hat >= 1.0, not < 1.0."""
        traces = jnp.ones((3, 20))
        rhat = gelman_rubin_rhat(traces)
        assert float(rhat) >= 1.0

    def test_error_on_1d(self):
        with pytest.raises(ValueError, match="at least 2 chains"):
            gelman_rubin_rhat(jnp.ones(10))

    def test_error_on_single_chain(self):
        with pytest.raises(ValueError, match="at least 2 chains"):
            gelman_rubin_rhat(jnp.ones((1, 10)))

    def test_error_on_short_chains(self):
        with pytest.raises(ValueError, match="at least 4 samples"):
            gelman_rubin_rhat(jnp.ones((2, 3)))


class TestEffectiveSampleSize:
    def test_iid_ess_near_n(self):
        """IID samples should have ESS close to n_samples."""
        traces = jax.random.normal(jax.random.key(200), (1, 500))
        ess = effective_sample_size(traces)
        # IID: ESS should be close to 500, allow generous tolerance
        assert float(ess) > 200

    def test_correlated_ess_less_than_n(self):
        """Highly correlated traces should have ESS << n_samples."""
        # Random walk: cumulative sum creates strong autocorrelation
        steps = jax.random.normal(jax.random.key(300), (1, 500)) * 0.01
        traces = jnp.cumsum(steps, axis=1)
        ess = effective_sample_size(traces)
        assert float(ess) < 200

    def test_constant_trace_ess_small(self):
        """Constant trace (stuck chain) should give small ESS."""
        traces = jnp.ones((1, 100))
        ess = effective_sample_size(traces)
        assert float(ess) <= 100  # Should be ~1

    def test_1d_input(self):
        """1-D input treated as single chain."""
        traces = jax.random.normal(jax.random.key(400), (200,))
        ess = effective_sample_size(traces)
        assert float(ess) > 0

    def test_multi_chain(self):
        """Multi-chain ESS should be > single-chain ESS."""
        key = jax.random.key(500)
        traces_1 = jax.random.normal(key, (1, 200))
        traces_4 = jax.random.normal(key, (4, 200))
        ess_1 = effective_sample_size(traces_1)
        ess_4 = effective_sample_size(traces_4)
        assert float(ess_4) > float(ess_1)

    def test_error_on_single_sample(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            effective_sample_size(jnp.ones((2, 1)))


class TestInsertRowsValidation:
    def test_wrong_column_count_raises(self):
        """insert_rows with wrong column count should raise ValueError."""
        from crosscat.model import initialize, insert_rows

        key = jax.random.key(600)
        data = jax.random.normal(key, (10, 3))
        state = initialize(jax.random.key(601), data, [ColumnType.CONTINUOUS] * 3).state
        wrong_cols = jax.random.normal(jax.random.key(602), (2, 5))  # 5 cols, not 3
        with pytest.raises(ValueError, match="shape"):
            insert_rows(jax.random.key(603), state, data, wrong_cols)


class TestGrowthFactorGuard:
    def test_growth_factor_one_raises(self):
        """growth_factor=1.0 should raise ValueError."""
        from crosscat.scaling import subsample_anneal

        data = jax.random.normal(jax.random.key(700), (50, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        with pytest.raises(ValueError, match="growth_factor"):
            subsample_anneal(jax.random.key(701), data, col_types, growth_factor=1.0)

    def test_growth_factor_below_one_raises(self):
        from crosscat.scaling import subsample_anneal

        data = jax.random.normal(jax.random.key(702), (50, 3))
        col_types = [ColumnType.CONTINUOUS] * 3
        with pytest.raises(ValueError, match="growth_factor"):
            subsample_anneal(jax.random.key(703), data, col_types, growth_factor=0.5)
