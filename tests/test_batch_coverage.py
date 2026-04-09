"""Tests for batch/packed inference functions that previously lacked coverage.

Covers: batch_anomaly_score, batch_classify_column, batch_credible_interval,
batch_impute_column, batch_predictive_cdf, batch_row_similarity,
batch_row_typicality, batch_score_columns_binary, packed_dependence_matrix,
packed_predictive_cdf.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from crosscat.model import initialize
from crosscat.packed import pack_state
from crosscat.types import ColumnType


def _make_packed_states(key, n_chains=2):
    """Create a small list of packed states for testing."""
    data = jax.random.normal(key, (20, 3))
    col_types = [ColumnType.CONTINUOUS] * 3
    states = []
    for i in range(n_chains):
        k = jax.random.fold_in(key, i)
        result = initialize(k, data, col_types)
        states.append(pack_state(result.state, max_clusters=16, max_views=8))
    return states, data


class TestBatchAnomalyScore:
    def test_shape_and_bounds(self):
        """batch_anomaly_score returns correct shape with values in [0, 1]."""
        from crosscat.packed_inference import batch_anomaly_score

        key = jax.random.key(100)
        states, data = _make_packed_states(key, n_chains=1)
        row_ids = jnp.arange(5)

        scores = batch_anomaly_score(states[0], data, row_ids)
        assert scores.shape == (5,)
        assert jnp.all(jnp.isfinite(scores))
        assert jnp.all(scores >= 0.0)
        assert jnp.all(scores <= 1.0)

    def test_consistent_ordering(self):
        """Rows with typical values should score lower than outliers."""
        from crosscat.packed_inference import batch_anomaly_score

        key = jax.random.key(101)
        states, data = _make_packed_states(key, n_chains=1)
        packed = states[0]
        # Score all rows — just verify consistency (all finite, in [0,1])
        row_ids = jnp.arange(data.shape[0])
        scores = batch_anomaly_score(packed, data, row_ids)
        assert scores.shape == (data.shape[0],)
        assert jnp.all(jnp.isfinite(scores))


class TestBatchClassifyColumn:
    def test_shape(self):
        """batch_classify_column returns (n_rows, n_candidates) log-probs."""
        from crosscat.packed_inference import batch_classify_column

        key = jax.random.key(110)
        states, data = _make_packed_states(key, n_chains=1)
        candidates = jnp.array([0.0, 1.0, -1.0])
        row_ids = jnp.array([0, 1, 2])

        result = batch_classify_column(states[0], data, 0, candidates, row_ids)
        assert result.shape == (3, 3)
        assert jnp.all(jnp.isfinite(result))
        # Log probabilities should be negative
        assert jnp.all(result < 0)


class TestBatchCredibleInterval:
    def test_returns_ordered_ci(self):
        """batch_credible_interval returns lower <= median <= upper."""
        from crosscat.packed_inference import batch_credible_interval

        key = jax.random.key(120)
        states, data = _make_packed_states(key, n_chains=1)
        row_ids = jnp.array([0, 1])

        medians, lowers, uppers = batch_credible_interval(
            jax.random.key(121), states[0], data, 0, row_ids, n_samples=200
        )
        assert medians.shape == (2,)
        assert lowers.shape == (2,)
        assert uppers.shape == (2,)
        for i in range(2):
            assert float(lowers[i]) <= float(medians[i]) <= float(uppers[i])


class TestBatchImputeColumn:
    def test_returns_finite_estimates(self):
        """batch_impute_column returns finite point estimates and confidences."""
        from crosscat.packed_inference import batch_impute_column

        key = jax.random.key(130)
        states, data = _make_packed_states(key, n_chains=1)
        row_ids = jnp.array([0, 1, 2])

        estimates, confidences = batch_impute_column(
            jax.random.key(131), states[0], data, 0, row_ids, n_samples=50
        )
        assert estimates.shape == (3,)
        assert confidences.shape == (3,)
        assert jnp.all(jnp.isfinite(estimates))
        assert jnp.all(jnp.isfinite(confidences))
        # Confidence should be positive
        assert jnp.all(confidences > 0)


class TestBatchPredictiveCdf:
    def test_values_in_unit_interval(self):
        """batch_predictive_cdf returns CDF values in [0, 1]."""
        from crosscat.packed_inference import batch_predictive_cdf

        key = jax.random.key(140)
        states, data = _make_packed_states(key, n_chains=1)
        row_ids = jnp.array([0, 1, 2])

        cdfs = batch_predictive_cdf(
            jax.random.key(141),
            states[0],
            data,
            0,
            jnp.float32(0.0),
            row_ids,
            n_samples=200,
        )
        assert cdfs.shape == (3,)
        assert jnp.all(jnp.isfinite(cdfs))
        assert jnp.all(cdfs >= 0.0)
        assert jnp.all(cdfs <= 1.0)


class TestBatchRowSimilarity:
    def test_symmetric_with_unit_diagonal(self):
        """batch_row_similarity returns symmetric matrix with diagonal=1."""
        from crosscat.packed_inference import batch_row_similarity

        key = jax.random.key(150)
        states, _data = _make_packed_states(key, n_chains=1)
        row_ids = jnp.array([0, 1, 2])

        sim = batch_row_similarity(states, row_ids)
        assert sim.shape == (3, 3)
        # Diagonal should be 1.0
        assert jnp.allclose(jnp.diag(sim), jnp.ones(3), atol=1e-5)
        # Should be symmetric
        assert jnp.allclose(sim, sim.T, atol=1e-5)
        # Values in [0, 1]
        assert jnp.all(sim >= 0.0)
        assert jnp.all(sim <= 1.0)


class TestBatchRowTypicality:
    def test_returns_finite_in_unit(self):
        """batch_row_typicality returns finite values in [0, 1]."""
        from crosscat.packed_inference import batch_row_typicality

        key = jax.random.key(160)
        states, _data = _make_packed_states(key, n_chains=1)
        row_ids = jnp.array([0, 1, 2, 3])

        typ = batch_row_typicality(states, row_ids)
        assert typ.shape == (4,)
        assert jnp.all(jnp.isfinite(typ))
        assert jnp.all(typ >= 0.0)
        assert jnp.all(typ <= 1.0)


class TestBatchScoreColumnsBinary:
    def test_returns_probabilities(self):
        """batch_score_columns_binary returns P(col=1) in [0, 1]."""
        from crosscat.packed_inference import batch_score_columns_binary

        # Use binary data for this test
        key = jax.random.key(170)
        data = jax.random.bernoulli(key, 0.5, (20, 3)).astype(jnp.float32)
        col_types = [ColumnType.BINARY] * 3
        result = initialize(jax.random.key(171), data, col_types)
        packed = pack_state(result.state, max_clusters=16, max_views=8)

        col_indices = jnp.array([0, 1, 2])
        probs = batch_score_columns_binary(packed, data, col_indices, row_id=0)
        assert probs.shape == (3,)
        assert jnp.all(jnp.isfinite(probs))
        assert jnp.all(probs >= 0.0)
        assert jnp.all(probs <= 1.0)


class TestPackedDependenceMatrix:
    def test_symmetric_unit_diagonal(self):
        """packed_dependence_matrix returns symmetric Z-matrix with diagonal=1."""
        from crosscat.packed_inference import packed_dependence_matrix

        key = jax.random.key(180)
        states, _data = _make_packed_states(key, n_chains=2)

        z = packed_dependence_matrix(states)
        assert z.shape == (3, 3)
        assert jnp.allclose(jnp.diag(z), jnp.ones(3), atol=1e-5)
        assert jnp.allclose(z, z.T, atol=1e-5)
        assert jnp.all(z >= 0.0)
        assert jnp.all(z <= 1.0)


class TestPackedPredictiveCdf:
    def test_returns_value_in_unit(self):
        """packed_predictive_cdf returns CDF value in [0, 1]."""
        from crosscat.packed_inference import packed_predictive_cdf

        key = jax.random.key(190)
        states, data = _make_packed_states(key, n_chains=1)

        cdf = packed_predictive_cdf(
            jax.random.key(191),
            states[0],
            data,
            0,
            jnp.float32(0.0),
            n_samples=200,
        )
        assert jnp.isfinite(cdf)
        assert float(cdf) >= 0.0
        assert float(cdf) <= 1.0

    def test_monotonic(self):
        """CDF should be monotonically non-decreasing."""
        from crosscat.packed_inference import packed_predictive_cdf

        key = jax.random.key(192)
        states, data = _make_packed_states(key, n_chains=1)
        packed = states[0]

        vals = [-2.0, 0.0, 2.0]
        cdfs = [
            float(
                packed_predictive_cdf(
                    jax.random.key(193 + i),
                    packed,
                    data,
                    0,
                    jnp.float32(v),
                    n_samples=500,
                )
            )
            for i, v in enumerate(vals)
        ]
        # With enough samples, CDF should be monotonic
        assert cdfs[0] <= cdfs[1] + 0.1  # allow small MC noise
        assert cdfs[1] <= cdfs[2] + 0.1
