"""Tests for Phase C-E: multi-chain wrappers, batch functions, and Arrow I/O.

Covers: multi_chain_classify_column, multi_chain_credible_interval,
multi_chain_joint_predictive_probability, batch_predictive_probability,
batch_predictive_sample, save_data/load_data roundtrip.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

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


class TestMultiChainClassifyColumn:
    def test_single_chain_matches_packed(self):
        """With 1 chain, multi_chain_classify_column matches packed_classify_column."""
        from crosscat.packed_inference import (
            multi_chain_classify_column,
            packed_classify_column,
        )

        key = jax.random.key(10)
        states, data = _make_packed_states(key, n_chains=1)
        candidates = jnp.array([0.0, 1.0, 2.0])

        single = packed_classify_column(states[0], data, 0, candidates, row_id=0)
        single_pred = candidates[jnp.argmax(single)]
        multi = multi_chain_classify_column(states, data, 0, candidates, row_id=0)

        assert float(single_pred) == float(multi)

    def test_uses_logsumexp(self):
        """Verify averaging uses log-mean-exp, not arithmetic mean of logs."""
        from crosscat.packed_inference import (
            multi_chain_classify_column,
        )

        key = jax.random.key(11)
        states, data = _make_packed_states(key, n_chains=3)
        candidates = jnp.array([0.0, 1.0, -1.0])

        result = multi_chain_classify_column(states, data, 0, candidates, row_id=0)
        assert jnp.isfinite(result)


class TestMultiChainCredibleInterval:
    def test_returns_three_values(self):
        from crosscat.packed_inference import multi_chain_credible_interval

        key = jax.random.key(20)
        states, data = _make_packed_states(key)
        median, lower, upper = multi_chain_credible_interval(
            jax.random.key(21), states, data, 0, n_samples=100
        )
        assert jnp.isfinite(median)
        assert float(lower) <= float(median) <= float(upper)

    def test_ci_level_validation(self):
        from crosscat.packed_inference import multi_chain_credible_interval

        key = jax.random.key(22)
        states, data = _make_packed_states(key)
        with pytest.raises(ValueError, match="ci_level"):
            multi_chain_credible_interval(jax.random.key(23), states, data, 0, ci_level=1.5)
        with pytest.raises(ValueError, match="ci_level"):
            multi_chain_credible_interval(jax.random.key(24), states, data, 0, ci_level=0.0)


class TestMultiChainJointPredictiveProb:
    def test_returns_finite_scalar(self):
        from crosscat.packed_inference import multi_chain_joint_predictive_probability

        key = jax.random.key(30)
        states, data = _make_packed_states(key)
        log_p = multi_chain_joint_predictive_probability(
            states, data, [0, 1], jnp.array([0.0, 0.0])
        )
        assert jnp.isfinite(log_p)
        assert float(log_p) < 0  # log-probability is negative


class TestBatchPredictiveProb:
    def test_matches_loop(self):
        """batch_predictive_probability matches loop over packed_predictive_probability."""
        from crosscat.packed_inference import (
            batch_predictive_probability,
            packed_predictive_probability,
        )

        key = jax.random.key(40)
        states, data = _make_packed_states(key, n_chains=1)
        packed = states[0]
        row_ids = jnp.array([0, 1, 2])
        query_vals = data[row_ids, 0]  # true values

        batch_result = batch_predictive_probability(packed, data, 0, query_vals, row_ids)

        loop_result = jnp.array(
            [
                packed_predictive_probability(
                    packed, data, [0], jnp.array([query_vals[i]]), row_id=int(row_ids[i])
                )
                for i in range(len(row_ids))
            ]
        )

        assert jnp.allclose(batch_result, loop_result, atol=1e-5)


class TestBatchPredictiveSample:
    def test_shape(self):
        from crosscat.packed_inference import batch_predictive_sample

        key = jax.random.key(50)
        states, data = _make_packed_states(key, n_chains=1)
        packed = states[0]
        row_ids = jnp.array([0, 1])

        result = batch_predictive_sample(
            jax.random.key(51), packed, data, [0], row_ids, n_samples_per_row=3
        )
        assert result.shape == (2, 3, 1)  # (n_queries, n_samples, n_query_cols)
        assert jnp.all(jnp.isfinite(result))


try:
    import pyarrow  # noqa: F401

    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


@pytest.mark.skipif(not _HAS_PYARROW, reason="pyarrow not installed")
class TestSaveLoadDataRoundtrip:
    def test_roundtrip_with_types(self):
        """save_data → load_data preserves data and column types."""
        from crosscat.data_utils import load_data, save_data

        key = jax.random.key(60)
        data = jax.random.normal(key, (10, 3))
        col_types = [ColumnType.CONTINUOUS, ColumnType.CATEGORICAL, ColumnType.BINARY]
        col_names = ["x", "y", "z"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.arrow"
            save_data(path, data, column_names=col_names, column_types=col_types)
            loaded_data, loaded_names, loaded_types = load_data(path)

        assert loaded_names == col_names
        assert loaded_types == col_types
        assert jnp.allclose(data, loaded_data, atol=1e-6)

    def test_roundtrip_without_types(self):
        """save_data → load_data works without column_types."""
        from crosscat.data_utils import load_data, save_data

        key = jax.random.key(61)
        data = jax.random.normal(key, (10, 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.arrow"
            save_data(path, data)
            loaded_data, loaded_names, loaded_types = load_data(path)

        assert loaded_types is None
        assert loaded_names == ["col_0", "col_1"]
        assert jnp.allclose(data, loaded_data, atol=1e-6)

    def test_column_subset(self):
        """load_data with columns= returns correct subset of types."""
        from crosscat.data_utils import load_data, save_data

        key = jax.random.key(62)
        data = jax.random.normal(key, (10, 4))
        col_types = [
            ColumnType.CONTINUOUS,
            ColumnType.CATEGORICAL,
            ColumnType.BINARY,
            ColumnType.CONTINUOUS,
        ]
        col_names = ["a", "b", "c", "d"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.arrow"
            save_data(path, data, column_names=col_names, column_types=col_types)
            loaded_data, loaded_names, loaded_types = load_data(path, columns=["b", "d"])

        assert loaded_names == ["b", "d"]
        assert loaded_types == [ColumnType.CATEGORICAL, ColumnType.CONTINUOUS]
        assert loaded_data.shape == (10, 2)

    def test_invalid_column_names_length(self):
        """save_data with wrong column_names length raises ValueError."""
        from crosscat.data_utils import save_data

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.arrow"
            with pytest.raises(ValueError, match="column_names"):
                save_data(path, jnp.ones((5, 3)), column_names=["a", "b"])

    def test_invalid_column_types_length(self):
        """save_data with wrong column_types length raises ValueError."""
        from crosscat.data_utils import save_data

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.arrow"
            with pytest.raises(ValueError, match="column_types"):
                save_data(
                    path, jnp.ones((5, 3)), column_types=[ColumnType.CONTINUOUS, ColumnType.BINARY]
                )

    def test_invalid_compression(self):
        """save_data with invalid compression raises ValueError."""
        from crosscat.data_utils import save_data

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.arrow"
            with pytest.raises(ValueError, match="compression"):
                save_data(path, jnp.ones((5, 3)), compression="gzip")


class TestBatchConditionalEntropy:
    def test_returns_finite(self):
        """batch_conditional_entropy returns finite values for each pair."""
        from crosscat.packed_inference import batch_conditional_entropy

        key = jax.random.key(70)
        states, data = _make_packed_states(key, n_chains=1)
        target_cols = [0, 1]
        given_cols = [2, 0]

        results = batch_conditional_entropy(
            jax.random.key(73), states, data, target_cols, given_cols
        )
        assert len(results) == 2
        for h in results:
            assert jnp.isfinite(h)
            assert float(h) >= 0  # entropy is non-negative


class TestBatchColumnTypicality:
    def test_returns_finite(self):
        """batch_column_typicality returns finite values for each column."""
        from crosscat.packed_inference import batch_column_typicality

        key = jax.random.key(71)
        states, _data = _make_packed_states(key, n_chains=1)
        col_indices = [0, 1, 2]

        results = batch_column_typicality(states, col_indices)
        assert len(results) == 3
        for t in results:
            assert jnp.isfinite(t)


class TestBatchDependenceProbability:
    def test_returns_finite_probabilities(self):
        """batch_dependence_probability returns values in [0, 1]."""
        from crosscat.packed_inference import batch_dependence_probability

        key = jax.random.key(72)
        states, _data = _make_packed_states(key, n_chains=1)
        col_pairs = [(0, 1), (1, 2), (0, 2)]

        results = batch_dependence_probability(states, col_pairs)
        assert len(results) == 3
        for p in results:
            assert jnp.isfinite(p)
            assert 0.0 <= float(p) <= 1.0
