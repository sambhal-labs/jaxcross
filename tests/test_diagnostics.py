"""Tests for convergence diagnostics.

Covers: Adjusted Rand Index, collect_diagnostics.
"""

from __future__ import annotations

import jax.numpy as jnp

from crosscat.diagnostics import adjusted_rand_index, collect_diagnostics


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


class TestCollectDiagnostics:
    def test_keys_present(self, simple_state):
        state, data, _ = simple_state
        diag = collect_diagnostics(state, data)
        assert "log_joint" in diag
        assert "n_views" in diag
        assert "column_crp_alpha" in diag
        assert jnp.isfinite(diag["log_joint"])
