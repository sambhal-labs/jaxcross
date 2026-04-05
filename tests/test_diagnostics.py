"""Tests for convergence diagnostics.

Covers: Adjusted Rand Index, collect_diagnostics, random_holdout_mask,
mean_test_log_likelihood, evaluate_imputation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from crosscat.diagnostics import (
    adjusted_rand_index,
    collect_diagnostics,
    evaluate_imputation,
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
        state = initialize(jax.random.key(45).state, data, column_types)
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
        state = initialize(jax.random.key(47).state, data, column_types)
        mask = random_holdout_mask(jax.random.key(48), 30, 3, holdout_fraction=0.1)
        result = evaluate_imputation(state, data, mask, column_types, rng_key=jax.random.key(49))
        assert "mae" in result or "accuracy" in result
        assert "mean_log_likelihood" in result
