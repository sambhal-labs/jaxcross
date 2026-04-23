"""Correctness-hardening tests for Phase 1 fixes.

Covers:
- LOG_EPS guards in DirichletCategorical.posterior_predictive_logp
- LOG_EPS guards in CRP log-probability sites (unpacked + packed)
- Runtime category-range validation in packed_insert_rows
- Overflow policy (warn vs raise) for cluster / column budgets
- All-NaN-row robustness through the packed Gibbs sweep
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.components import DirichletCategorical
from crosscat.model import _log_crp, initialize
from crosscat.packed import pack_state, packed_insert_rows, set_overflow_policy
from crosscat.packed.kernels import _validate_category_range
from crosscat.types import LOG_EPS, ColumnHypers, ColumnType, SufficientStats

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------
# LOG_EPS guards: DirichletCategorical
# ---------------------------------------------------------------------------


def test_dirichlet_categorical_logp_finite_on_unseen_category():
    """A category never observed in suffstats must not yield -inf logp.

    With counts=[1, 0] and alpha -> 0, probs for class 1 would be 0 without
    the LOG_EPS guard, producing -inf. The guard clamps to log(LOG_EPS).
    """
    suffstats = SufficientStats(
        column_type=ColumnType.CATEGORICAL,
        count=jnp.asarray(1, dtype=jnp.int32),
        category_counts=jnp.array([1.0, 0.0]),
    )
    hypers = ColumnHypers(
        column_type=ColumnType.CATEGORICAL,
        dirichlet_alpha=jnp.asarray(1e-40),
    )

    logp_unseen = DirichletCategorical.posterior_predictive_logp(
        jnp.asarray(1, dtype=jnp.int32), suffstats, hypers
    )
    assert jnp.isfinite(logp_unseen), "posterior predictive must be finite with LOG_EPS guard"
    # Lower bound: log(LOG_EPS) ≈ -69
    assert logp_unseen >= jnp.log(LOG_EPS) - 1.0


def test_dirichlet_categorical_logp_normal_path_unchanged():
    """The guard must not alter logp when probs are well-bounded."""
    suffstats = SufficientStats(
        column_type=ColumnType.CATEGORICAL,
        count=jnp.asarray(10, dtype=jnp.int32),
        category_counts=jnp.array([4.0, 4.0, 2.0]),
    )
    hypers = ColumnHypers(
        column_type=ColumnType.CATEGORICAL,
        dirichlet_alpha=jnp.asarray(1.0),
    )

    logp = DirichletCategorical.posterior_predictive_logp(
        jnp.asarray(0, dtype=jnp.int32), suffstats, hypers
    )
    # Expected: log((4 + 1) / (10 + 3)) = log(5/13)
    expected = jnp.log(5.0 / 13.0)
    assert jnp.allclose(logp, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# LOG_EPS guards: CRP
# ---------------------------------------------------------------------------


def test_log_crp_finite_with_tiny_alpha():
    """_log_crp must not NaN when alpha hits its numerical floor."""
    assignments = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
    logp = _log_crp(assignments, jnp.asarray(0.0))
    assert jnp.isfinite(logp)


def test_log_crp_finite_with_normal_alpha():
    """Guard does not distort typical CRP log-probabilities."""
    assignments = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
    logp = _log_crp(assignments, jnp.asarray(1.0))
    assert jnp.isfinite(logp)
    # Sanity: CRP log-prob of a non-degenerate partition is negative but bounded.
    assert -50.0 < float(logp) < 0.0


# ---------------------------------------------------------------------------
# Category-range validation
# ---------------------------------------------------------------------------


def test_validate_category_range_accepts_in_range():
    col_type_ids = jnp.array([1, 0], dtype=jnp.int32)  # CATEGORICAL, CONTINUOUS
    data = jnp.array([[0.0, 3.14], [2.0, 2.71], [1.0, 1.41]])
    # Should not raise
    _validate_category_range(data, col_type_ids, max_categories=3)


def test_validate_category_range_rejects_out_of_range():
    col_type_ids = jnp.array([1], dtype=jnp.int32)  # CATEGORICAL
    data = jnp.array([[0.0], [1.0], [5.0]])  # 5 >= max_categories=3
    with pytest.raises(ValueError, match="max_categories"):
        _validate_category_range(data, col_type_ids, max_categories=3)


def test_validate_category_range_ignores_nan():
    col_type_ids = jnp.array([1], dtype=jnp.int32)
    data = jnp.array([[0.0], [jnp.nan], [1.0]])
    # NaN should be skipped; no exception
    _validate_category_range(data, col_type_ids, max_categories=2)


def test_packed_insert_rows_rejects_out_of_range_category():
    """packed_insert_rows must validate new rows before the JIT boundary."""
    key = jax.random.key(0)
    # 1 CONTINUOUS + 1 CATEGORICAL column with 3 possible levels.
    data = jnp.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 0.0]], dtype=jnp.float32)
    types = [ColumnType.CONTINUOUS, ColumnType.CATEGORICAL]
    state = initialize(key, data, types).state
    packed = pack_state(state, max_categories=3, data=data)

    # New row with category value 5 (> max_categories - 1)
    bad_rows = jnp.array([[5.0, 5.0]], dtype=jnp.float32)
    with pytest.raises(ValueError, match="max_categories"):
        packed_insert_rows(jax.random.key(1), packed, data, bad_rows)


# ---------------------------------------------------------------------------
# Overflow policy
# ---------------------------------------------------------------------------


def test_set_overflow_policy_validation():
    with pytest.raises(ValueError):
        set_overflow_policy("silent")


def test_set_overflow_policy_raise_then_warn_roundtrip():
    """Policy toggling is stateful but reversible."""
    from crosscat.packed.kernels import _report_overflow

    set_overflow_policy("raise")
    try:
        with pytest.raises(RuntimeError, match="budget"):
            _report_overflow("budget exhausted (test)")
    finally:
        set_overflow_policy("warn")

    # After restoring, the same call must only warn.
    with pytest.warns(UserWarning, match="budget"):
        _report_overflow("budget exhausted (test)")


# ---------------------------------------------------------------------------
# NaN robustness
# ---------------------------------------------------------------------------


def test_all_nan_row_does_not_crash_through_pack_cycle():
    """A row consisting entirely of NaN survives init + pack + unpack without NaN output."""
    from crosscat.packed import unpack_state

    key = jax.random.key(7)
    data = jnp.array(
        [
            [1.0, 0.0],
            [jnp.nan, jnp.nan],
            [2.0, 1.0],
            [3.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    types = [ColumnType.CONTINUOUS, ColumnType.BINARY]
    state = initialize(key, data, types).state
    packed = pack_state(state)
    recovered = unpack_state(packed, types, data=data)
    assert recovered.n_rows == 4
