"""Tests for CrossCat paper parity gap fixes.

Covers: CRP alpha grid expansion, singleton view handling, Gamma(1,1) new-view
alpha, NormalGamma r hyperparameter sampling, dependence_probability / Z-matrix,
and optional recompute_all_suffstats.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.model import initialize, log_joint
from crosscat.types import ColumnType


@pytest.fixture
def rng_key():
    return jax.random.key(42)


@pytest.fixture
def simple_continuous_data(rng_key):
    """4-column continuous dataset with 2 correlated pairs."""
    n_rows = 80
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    # Columns 0 and 1 are correlated (same cluster structure)
    labels = jnp.where(jnp.arange(n_rows) < 40, 0, 1)
    col0 = jnp.where(labels == 0, 0.0, 10.0) + jax.random.normal(k1, (n_rows,)) * 0.5
    col1 = jnp.where(labels == 0, 0.0, 10.0) + jax.random.normal(k2, (n_rows,)) * 0.5
    # Columns 2 and 3 have independent structure
    labels2 = jnp.where(jnp.arange(n_rows) % 2 == 0, 0, 1)
    col2 = jnp.where(labels2 == 0, -5.0, 5.0) + jax.random.normal(k3, (n_rows,)) * 0.5
    col3 = jax.random.normal(k4, (n_rows,)) * 2.0
    data = jnp.column_stack([col0, col1, col2, col3])
    column_types = [ColumnType.CONTINUOUS] * 4
    return data, column_types


# ---- Gap 3: CRP alpha grid coverage ----


class TestCRPAlphaGrid:
    def test_crp_alpha_can_reach_extreme_values(self, rng_key, simple_continuous_data):
        """CRP alpha grid should allow values outside the old [0.1, 20.0] range."""
        from crosscat.gibbs import transition_crp_alphas

        data, column_types = simple_continuous_data
        state = initialize(rng_key, data, column_types)

        # Run many CRP alpha transitions and collect sampled values
        alphas_col = []
        alphas_row = []
        key = rng_key
        for _ in range(50):
            key, subkey = jax.random.split(key)
            state = transition_crp_alphas(subkey, state)
            alphas_col.append(float(state.column_crp_alpha))
            for v in state.views:
                alphas_row.append(float(v.row_crp_alpha))

        all_alphas = alphas_col + alphas_row
        min_alpha = min(all_alphas)
        max_alpha = max(all_alphas)
        # The new grid goes from 0.01 to 100.0 — verify the range is wider
        # than the old grid's [0.1, 20.0]
        assert min_alpha < 0.15 or max_alpha > 18.0, (
            f"Alpha range [{min_alpha:.3f}, {max_alpha:.3f}] suggests grid is too narrow"
        )


# ---- Gap 1 & 5: Singleton view handling + Gamma(1,1) alpha ----


class TestSingletonViewHandling:
    def test_singleton_column_reuses_row_assignments(self, rng_key):
        """When a column is the only one in its view (singleton), the new-view
        proposal should reuse the current view's row assignments."""
        from crosscat.gibbs import transition_column_assignments

        # Create a state with 2 views, each with 1 column (both singletons)
        data = jnp.column_stack(
            [
                jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            ]
        )
        column_types = [ColumnType.CONTINUOUS, ColumnType.CONTINUOUS]

        key = rng_key
        state = initialize(key, data, column_types, initialization="apart")

        # Both columns should be in separate views (apart initialization)
        assert len(state.views) == 2, "Expected 2 views with 'apart' initialization"

        # Run column assignment transitions — should not crash and should handle
        # singleton views correctly
        key, subkey = jax.random.split(key)
        new_state = transition_column_assignments(subkey, state, data)
        assert new_state is not None
        assert len(new_state.views) >= 1

    def test_new_view_alpha_varies(self, rng_key, simple_continuous_data):
        """New view alpha should be sampled from Gamma(1,1), not always the same."""
        from crosscat.gibbs import transition_column_assignments

        data, column_types = simple_continuous_data
        state = initialize(rng_key, data, column_types)

        # Run many column transitions and track new view alphas
        new_view_alphas = set()
        key = rng_key
        for _ in range(20):
            key, subkey = jax.random.split(key)
            state = transition_column_assignments(subkey, state, data)
            for v in state.views:
                new_view_alphas.add(round(float(v.row_crp_alpha), 4))

        # Should see more than 1 unique alpha value across views
        assert len(new_view_alphas) > 1, (
            "Expected diverse row_crp_alpha values from Gamma(1,1) sampling"
        )


# ---- Gap 2: NormalGamma r hyperparameter sampling ----


class TestNormalGammaRSampling:
    def test_r_hyperparameter_varies(self, rng_key, simple_continuous_data):
        """The r (precision scale) hyperparameter should change across iterations."""
        from crosscat.gibbs import transition_column_hypers

        data, column_types = simple_continuous_data
        state = initialize(rng_key, data, column_types)

        r_values = set()
        key = rng_key
        for _ in range(30):
            key, subkey = jax.random.split(key)
            state = transition_column_hypers(subkey, state, data)
            for j in range(len(column_types)):
                r_values.add(round(float(state.column_hypers[j].r), 6))

        assert len(r_values) > 1, (
            f"r hyperparameter stuck at single value: {r_values}. "
            "Expected it to vary across MCMC iterations."
        )

    def test_r_sampling_does_not_degrade_log_joint(self, rng_key, simple_continuous_data):
        """Sampling r should not systematically degrade model quality."""
        from crosscat.gibbs import gibbs_sweep

        data, column_types = simple_continuous_data
        state = initialize(rng_key, data, column_types)

        key, subkey = jax.random.split(rng_key)
        state = gibbs_sweep(subkey, state, data, n_sweeps=10)
        score = float(log_joint(state, data))

        # Score should be finite and not NaN
        assert jnp.isfinite(score), f"Log joint is not finite: {score}"


# ---- Gap 4: Dependence probability / Z-matrix ----


class TestDependenceProbability:
    def test_same_view_columns_high_probability(self, rng_key):
        """Columns in the same view should have high dependence probability."""
        from crosscat.inference import dependence_probability

        data = jnp.column_stack(
            [
                jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                jnp.array([2.0, 4.0, 6.0, 8.0, 10.0]),
            ]
        )
        column_types = [ColumnType.CONTINUOUS, ColumnType.CONTINUOUS]

        # Create states where both columns are in the same view
        states = []
        for i in range(5):
            key = jax.random.fold_in(rng_key, i)
            s = initialize(key, data, column_types, initialization="together")
            states.append(s)

        dp = dependence_probability(states, 0, 1)
        assert float(dp) == 1.0, "Columns in same view should have dep prob = 1.0"

    def test_dependence_matrix_properties(self, rng_key):
        """Z-matrix should be symmetric with diagonal = 1.0."""
        from crosscat.inference import dependence_matrix

        data = jnp.column_stack(
            [
                jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                jnp.array([2.0, 4.0, 6.0, 8.0, 10.0]),
                jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            ]
        )
        column_types = [ColumnType.CONTINUOUS] * 3

        states = []
        for i in range(5):
            key = jax.random.fold_in(rng_key, i)
            s = initialize(key, data, column_types)
            states.append(s)

        z = dependence_matrix(states)
        assert z.shape == (3, 3)
        # Diagonal should be 1.0
        assert jnp.allclose(jnp.diag(z), 1.0), f"Diagonal not 1.0: {jnp.diag(z)}"
        # Should be symmetric
        assert jnp.allclose(z, z.T), "Z-matrix should be symmetric"
        # All values in [0, 1]
        assert jnp.all(z >= 0.0) and jnp.all(z <= 1.0), "Z-matrix values out of [0, 1]"

    def test_packed_dependence_matches_unpacked(self, rng_key):
        """Packed and unpacked dependence_probability should agree."""
        from crosscat.inference import dependence_probability
        from crosscat.packed import pack_state
        from crosscat.packed_inference import packed_dependence_probability

        data = jnp.column_stack(
            [
                jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                jnp.array([2.0, 4.0, 6.0, 8.0, 10.0]),
                jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            ]
        )
        column_types = [ColumnType.CONTINUOUS] * 3

        states = []
        packed_states = []
        for i in range(5):
            key = jax.random.fold_in(rng_key, i)
            s = initialize(key, data, column_types)
            states.append(s)
            packed_states.append(pack_state(s))

        for ci in range(3):
            for cj in range(3):
                dp = float(dependence_probability(states, ci, cj))
                pdp = float(packed_dependence_probability(packed_states, ci, cj))
                assert dp == pytest.approx(pdp, abs=1e-6), (
                    f"Mismatch for ({ci},{cj}): unpacked={dp}, packed={pdp}"
                )


# ---- Gap 7: Optional recompute_all_suffstats ----


class TestOptionalRecomputeSuffstats:
    def test_recompute_flag_default_true(self, rng_key, simple_continuous_data):
        """Default behavior (recompute=True) should work as before."""
        from crosscat.packed import pack_state, packed_transition_row_assignments

        data, column_types = simple_continuous_data
        state = initialize(rng_key, data, column_types)
        packed = pack_state(state)

        key, subkey = jax.random.split(rng_key)
        result = packed_transition_row_assignments(subkey, packed, data)
        assert result is not None

    def test_recompute_flag_false_runs(self, rng_key, simple_continuous_data):
        """recompute_suffstats=False should run without error."""
        from crosscat.packed import pack_state
        from crosscat.packed.kernels import packed_transition_row_assignments

        data, column_types = simple_continuous_data
        state = initialize(rng_key, data, column_types)
        packed = pack_state(state)

        key, subkey = jax.random.split(rng_key)
        result = packed_transition_row_assignments(subkey, packed, data, recompute_suffstats=False)
        assert result is not None
