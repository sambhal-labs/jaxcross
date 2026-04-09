"""Tests for batch ordinal scoring and the ordinal fast-path in kernels.

Covers:
- batch_ol_posterior_predictive_logp parity with scalar _ol_posterior_predictive_logp
- All-ORDINAL view triggers dominant-type fast path in packed Gibbs
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.packed.components import (
    _ol_posterior_predictive_logp,
    batch_ol_posterior_predictive_logp,
)
from crosscat.types import ColumnType

MAX_CATS = 6


class TestBatchOlParity:
    """batch_ol_posterior_predictive_logp must match scalar _ol_posterior_predictive_logp."""

    def _make_ordinal_inputs(self, n_cols: int, key: jax.Array):
        """Generate random ordinal hyperparameters for n_cols columns."""
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        xs = jax.random.randint(k1, (n_cols,), 0, MAX_CATS).astype(jnp.float32)
        counts = jax.random.randint(k2, (n_cols,), 1, 50).astype(jnp.float32)
        cat_counts = jax.random.uniform(k3, (n_cols, MAX_CATS)) * 10.0
        # Sorted cutpoints per column
        raw = jax.random.normal(k4, (n_cols, MAX_CATS - 1))
        cutpoints = jnp.sort(raw, axis=1)
        mu0s = jax.random.normal(k5, (n_cols,))
        s0s = jnp.ones(n_cols) * 2.0
        return xs, counts, cat_counts, cutpoints, mu0s, s0s

    def test_parity_with_scalar(self):
        """Batch output matches loop over scalar function."""
        n_cols = 5
        xs, counts, cat_counts, cutpoints, mu0s, s0s = self._make_ordinal_inputs(
            n_cols, jax.random.key(42)
        )

        # Scalar path: loop over columns
        scalar_results = []
        for j in range(n_cols):
            lp = _ol_posterior_predictive_logp(
                xs[j], counts[j], cat_counts[j], cutpoints[j], mu0s[j], s0s[j]
            )
            scalar_results.append(float(lp))
        scalar_logps = jnp.array(scalar_results)

        # Batch path
        batch_logps = batch_ol_posterior_predictive_logp(
            xs, counts, cat_counts, cutpoints, mu0s, s0s
        )

        assert jnp.allclose(scalar_logps, batch_logps, atol=1e-5), (
            f"Parity mismatch: scalar={scalar_logps}, batch={batch_logps}"
        )

    def test_nan_guards_on_mu0s(self):
        """NaN in mu0s should not produce NaN output (clamped to 0.0)."""
        n_cols = 3
        xs, counts, cat_counts, cutpoints, _, s0s = self._make_ordinal_inputs(
            n_cols, jax.random.key(43)
        )
        nan_mu0s = jnp.array([float("nan"), 0.0, float("nan")])
        result = batch_ol_posterior_predictive_logp(
            xs, counts, cat_counts, cutpoints, nan_mu0s, s0s
        )
        assert jnp.all(jnp.isfinite(result)), f"NaN in output: {result}"

    def test_nan_guards_on_cutpoints(self):
        """NaN in cutpoints should not produce NaN output (clamped to LOGISTIC_INF)."""
        n_cols = 2
        xs, counts, cat_counts, _, mu0s, s0s = self._make_ordinal_inputs(
            n_cols, jax.random.key(44)
        )
        nan_cutpoints = jnp.full((n_cols, MAX_CATS - 1), float("nan"))
        result = batch_ol_posterior_predictive_logp(
            xs, counts, cat_counts, nan_cutpoints, mu0s, s0s
        )
        assert jnp.all(jnp.isfinite(result)), f"NaN in output: {result}"

    def test_single_column(self):
        """Batch with n_cols=1 matches scalar."""
        xs, counts, cat_counts, cutpoints, mu0s, s0s = self._make_ordinal_inputs(
            1, jax.random.key(45)
        )
        scalar = _ol_posterior_predictive_logp(
            xs[0], counts[0], cat_counts[0], cutpoints[0], mu0s[0], s0s[0]
        )
        batch = batch_ol_posterior_predictive_logp(xs, counts, cat_counts, cutpoints, mu0s, s0s)
        assert jnp.isclose(float(scalar), float(batch[0]), atol=1e-5)


class TestOrdinalFastPath:
    """Test that all-ORDINAL views trigger the dominant-type fast path in kernels."""

    @pytest.mark.slow
    def test_all_ordinal_gibbs_sweep(self):
        """Run packed Gibbs sweep on all-ORDINAL data — exercises ordinal fast path."""
        from crosscat import initialize
        from crosscat.packed import pack_state, packed_gibbs_sweep

        key = jax.random.key(100)
        n_rows, n_cols = 30, 4
        n_levels = 5

        # Generate ordinal data: integer values in [0, n_levels)
        data = jax.random.randint(key, (n_rows, n_cols), 0, n_levels).astype(jnp.float32)
        col_types = [ColumnType.ORDINAL] * n_cols

        result = initialize(jax.random.key(101), data, col_types)
        packed = pack_state(result.state, max_clusters=8, max_views=4)

        # This should trigger _compute_dominant_type → ORDINAL_ID → ol_score path
        packed_out = packed_gibbs_sweep(jax.random.key(102), packed, data, n_sweeps=3)

        # Basic sanity: state should be valid (no NaN in assignments)
        assert jnp.all(jnp.isfinite(packed_out.row_assignments))
        assert jnp.all(jnp.isfinite(packed_out.column_assignments))
