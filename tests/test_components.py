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


class TestVonMisesKappaScaling:
    """Verify kappa scales the data sufficient statistics in the resultant vector."""

    def test_kappa_affects_log_marginal(self):
        """Log marginal must differ when kappa != 1.0 vs kappa == 1.0 (non-trivial data)."""
        from crosscat.components import VonMises

        data = jnp.array([0.1, 0.5, 1.0, 1.5])
        ss = VonMises.sufficient_statistics(data)
        base_hypers = ColumnHypers(
            column_type=ColumnType.CYCLIC,
            kappa=jnp.array(1.0),
            vm_a=jnp.array(1.0),
            vm_mu=jnp.array(0.0),
        )
        high_kappa_hypers = ColumnHypers(
            column_type=ColumnType.CYCLIC,
            kappa=jnp.array(5.0),
            vm_a=jnp.array(1.0),
            vm_mu=jnp.array(0.0),
        )
        lml_k1 = VonMises.log_marginal_likelihood(ss, base_hypers)
        lml_k5 = VonMises.log_marginal_likelihood(ss, high_kappa_hypers)
        assert jnp.isfinite(lml_k1) and jnp.isfinite(lml_k5)
        assert float(lml_k1) != float(lml_k5), "kappa must affect log marginal"

    def test_kappa_affects_posterior_mean(self):
        """Posterior mean direction must shift toward data as kappa increases."""
        from crosscat.components import VonMises

        # Data clustered near 0.0, prior mean at pi
        data = jnp.array([0.0, 0.1, -0.1])
        ss = VonMises.sufficient_statistics(data)

        low_kappa = ColumnHypers(
            column_type=ColumnType.CYCLIC,
            kappa=jnp.array(0.1),
            vm_a=jnp.array(5.0),
            vm_mu=jnp.array(jnp.pi),
        )
        high_kappa = ColumnHypers(
            column_type=ColumnType.CYCLIC,
            kappa=jnp.array(10.0),
            vm_a=jnp.array(5.0),
            vm_mu=jnp.array(jnp.pi),
        )
        # With low kappa, prior dominates → predictive favors x near pi
        # With high kappa, data dominates → predictive favors x near 0
        lp_low_at_0 = float(VonMises.posterior_predictive_logp(jnp.array(0.0), ss, low_kappa))
        lp_high_at_0 = float(VonMises.posterior_predictive_logp(jnp.array(0.0), ss, high_kappa))
        assert lp_high_at_0 > lp_low_at_0, (
            "Higher kappa should give more weight to data (near 0), "
            f"got lp_high={lp_high_at_0}, lp_low={lp_low_at_0}"
        )

    def test_packed_matches_unpacked_kappa(self):
        """Packed and unpacked VM log marginal agree for kappa != 1."""
        from crosscat.components import VonMises
        from crosscat.packed.components import _vm_log_marginal

        data = jnp.array([0.2, 0.8, 1.5])
        ss = VonMises.sufficient_statistics(data)
        hypers = ColumnHypers(
            column_type=ColumnType.CYCLIC,
            kappa=jnp.array(3.0),
            vm_a=jnp.array(2.0),
            vm_mu=jnp.array(1.0),
        )
        unpacked = float(VonMises.log_marginal_likelihood(ss, hypers))
        packed = float(
            _vm_log_marginal(
                ss.count, ss.sum_sin, ss.sum_cos, hypers.kappa, hypers.vm_a, hypers.vm_mu
            )
        )
        assert jnp.isclose(unpacked, packed, atol=1e-5), (
            f"Packed/unpacked mismatch: unpacked={unpacked}, packed={packed}"
        )
