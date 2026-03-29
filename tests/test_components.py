"""Tests for Bayesian component models.

Covers: NaN handling across all component types, VonMises (cyclic) model,
OrderedLogistic cutpoint sensitivity, and component edge cases.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from crosscat.types import ColumnHypers, ColumnType, SufficientStats

# --- NaN handling ---


class TestNaNHandling:
    """NaN values (missing data) should be filtered from sufficient statistics."""

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

    def test_vonmises_nan(self):
        from crosscat.components import VonMises

        data = jnp.array([0.1, jnp.nan, 0.3, jnp.nan, 0.5])
        ss = VonMises.sufficient_statistics(data)
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
            column_type=ColumnType.CYCLIC,
            kappa=jnp.array(2.0),
            vm_a=jnp.array(1.0),
            vm_mu=jnp.array(0.0),
        )
        lml = VonMises.log_marginal_likelihood(ss, hypers)
        assert jnp.isfinite(lml)

    def test_posterior_predictive(self):
        from crosscat.components import VonMises

        data = jnp.array([0.1, 0.2, 0.3])
        ss = VonMises.sufficient_statistics(data)
        hypers = ColumnHypers(
            column_type=ColumnType.CYCLIC,
            kappa=jnp.array(2.0),
            vm_a=jnp.array(1.0),
            vm_mu=jnp.array(0.0),
        )
        log_p = VonMises.posterior_predictive_logp(jnp.array(0.15), ss, hypers)
        assert jnp.isfinite(log_p)

    def test_sample(self, rng_key):
        from crosscat.components import VonMises

        data = jnp.array([0.1, 0.2, 0.3])
        ss = VonMises.sufficient_statistics(data)
        hypers = ColumnHypers(
            column_type=ColumnType.CYCLIC,
            kappa=jnp.array(5.0),
            vm_a=jnp.array(1.0),
            vm_mu=jnp.array(0.2),
        )
        samples = VonMises.sample_posterior_predictive(rng_key, ss, hypers, n=100)
        assert samples.shape == (100,)
        # All samples should be in [0, 2*pi)
        assert jnp.all(samples >= 0)
        assert jnp.all(samples < 2 * jnp.pi)

    def test_sampling_completes_low_kappa(self, rng_key):
        """VonMises rejection sampling terminates even with very low kappa."""
        from crosscat.components import VonMises

        ss = SufficientStats(
            column_type=ColumnType.CYCLIC,
            count=jnp.array(5, dtype=jnp.int32),
            sum_sin=jnp.array(0.1),
            sum_cos=jnp.array(0.1),
        )
        hypers = ColumnHypers(
            column_type=ColumnType.CYCLIC,
            kappa=jnp.array(0.01),  # very low kappa — nearly uniform
            vm_a=jnp.array(0.01),
            vm_mu=jnp.array(0.0),
        )
        samples = VonMises.sample_posterior_predictive(rng_key, ss, hypers, n=10)
        assert samples.shape == (10,)
        assert jnp.all(samples >= 0.0) & jnp.all(samples < 2.0 * jnp.pi)


# --- OrderedLogistic cutpoint sensitivity ---


class TestOrdinalUsesHypers:
    """Tests that OrderedLogistic properly uses cutpoints and prior hypers."""

    def test_log_marginal_uses_cutpoints(self):
        from crosscat.components import OrderedLogistic

        counts = jnp.array([5.0, 3.0, 2.0])
        ss = SufficientStats(
            column_type=ColumnType.ORDINAL,
            count=jnp.array(10, dtype=jnp.int32),
            category_counts=counts,
        )
        hypers_1 = ColumnHypers(
            column_type=ColumnType.ORDINAL,
            mu=jnp.array(0.0),
            s=jnp.array(4.0),
            cutpoints=jnp.array([-1.0, 1.0]),
        )
        hypers_2 = ColumnHypers(
            column_type=ColumnType.ORDINAL,
            mu=jnp.array(0.0),
            s=jnp.array(4.0),
            cutpoints=jnp.array([-0.5, 0.5]),
        )

        lml_1 = OrderedLogistic.log_marginal_likelihood(ss, hypers_1)
        lml_2 = OrderedLogistic.log_marginal_likelihood(ss, hypers_2)
        # Different cutpoints should give different log marginals
        assert not jnp.allclose(lml_1, lml_2)

    def test_posterior_predictive_uses_cutpoints(self):
        from crosscat.components import OrderedLogistic

        counts = jnp.array([5.0, 3.0, 2.0])
        ss = SufficientStats(
            column_type=ColumnType.ORDINAL,
            count=jnp.array(10, dtype=jnp.int32),
            category_counts=counts,
        )
        hypers_1 = ColumnHypers(
            column_type=ColumnType.ORDINAL,
            mu=jnp.array(0.0),
            s=jnp.array(4.0),
            cutpoints=jnp.array([-1.0, 1.0]),
        )
        hypers_2 = ColumnHypers(
            column_type=ColumnType.ORDINAL,
            mu=jnp.array(0.0),
            s=jnp.array(4.0),
            cutpoints=jnp.array([-3.0, 3.0]),
        )

        logp_1 = OrderedLogistic.posterior_predictive_logp(jnp.array(0.0), ss, hypers_1)
        logp_2 = OrderedLogistic.posterior_predictive_logp(jnp.array(0.0), ss, hypers_2)
        # Different cutpoints -> different predictive probabilities
        assert not jnp.allclose(logp_1, logp_2)

    def test_log_marginal_empty_is_zero(self):
        from crosscat.components import OrderedLogistic

        ss = SufficientStats(
            column_type=ColumnType.ORDINAL,
            count=jnp.array(0, dtype=jnp.int32),
            category_counts=jnp.zeros(3),
        )
        hypers = ColumnHypers(
            column_type=ColumnType.ORDINAL,
            mu=jnp.array(0.0),
            s=jnp.array(4.0),
            cutpoints=jnp.array([-1.0, 1.0]),
        )
        assert float(OrderedLogistic.log_marginal_likelihood(ss, hypers)) == 0.0

    @pytest.mark.parametrize("level", [0, 1, 2])
    def test_posterior_predictive_finite(self, level):
        from crosscat.components import OrderedLogistic

        counts = jnp.array([5.0, 3.0, 2.0])
        ss = SufficientStats(
            column_type=ColumnType.ORDINAL,
            count=jnp.array(10, dtype=jnp.int32),
            category_counts=counts,
        )
        hypers = ColumnHypers(
            column_type=ColumnType.ORDINAL,
            mu=jnp.array(0.0),
            s=jnp.array(4.0),
            cutpoints=jnp.array([-1.0, 1.0]),
        )
        logp = OrderedLogistic.posterior_predictive_logp(jnp.array(float(level)), ss, hypers)
        assert jnp.isfinite(logp), f"Non-finite predictive at level {level}: {logp}"
