"""Property-based tests for CrossCat mathematical invariants.

Uses Hypothesis to verify properties hold for all valid inputs,
not just hand-picked examples. Covers:
- Sufficient statistics add/remove roundtrip (inverse operations)
- Pack/unpack roundtrip (structural preservation)
- Component model scoring (empty cluster = 0, finite outputs)
- Unified type dispatch parity (matches type-specific functions)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from crosscat.packed.components import (
    _bb_log_marginal,
    _bb_posterior_predictive_logp,
    _dc_log_marginal,
    _dc_posterior_predictive_logp,
    _ng_log_marginal,
    _ng_posterior_predictive_logp,
    _vm_log_marginal,
    _vm_posterior_predictive_logp,
    unified_log_marginal,
    unified_posterior_predictive_logp,
)
from crosscat.packed.state import (
    BINARY_ID,
    CATEGORICAL_ID,
    CONTINUOUS_ID,
    CYCLIC_ID,
    ORDINAL_ID,
)
from crosscat.packed.suffstats import _add_row_to_suffstats, _remove_row_from_suffstats

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Bounded positive floats for hyperparameters
pos_float = st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)

# Bounded floats for data values
data_float = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)

# Small integers for category indices
category_int = st.integers(min_value=0, max_value=7)

# Binary values
binary_val = st.sampled_from([0.0, 1.0])

# Cyclic values in [0, 2*pi)
cyclic_float = st.floats(min_value=0.0, max_value=6.28, allow_nan=False, allow_infinity=False)

# Count (non-negative int)
count_st = st.integers(min_value=1, max_value=200)

MAX_CATS = 8
MAX_COLS = 4
MAX_CLUSTERS = 4


# ---------------------------------------------------------------------------
# Suffstat add/remove roundtrip tests
# ---------------------------------------------------------------------------


def _make_empty_suffstats(n_cols, max_clusters, max_cats):
    """Create zeroed suffstat arrays."""
    return (
        jnp.zeros((max_clusters, n_cols), dtype=jnp.int32),  # counts
        jnp.zeros((max_clusters, n_cols), dtype=jnp.float32),  # sum_x
        jnp.zeros((max_clusters, n_cols), dtype=jnp.float32),  # sum_x_sq
        jnp.zeros((max_clusters, n_cols, max_cats), dtype=jnp.float32),  # cat_counts
        jnp.zeros((max_clusters, n_cols), dtype=jnp.float32),  # sum_sin
        jnp.zeros((max_clusters, n_cols), dtype=jnp.float32),  # sum_cos
    )


@given(
    values=st.lists(data_float, min_size=MAX_COLS, max_size=MAX_COLS),
    cluster_id=st.integers(min_value=0, max_value=MAX_CLUSTERS - 1),
)
@settings(max_examples=50, deadline=None)
def test_suffstat_add_remove_roundtrip_continuous(values, cluster_id):
    """Adding then removing a row recovers original suffstats (continuous)."""
    col_indices = jnp.arange(MAX_COLS, dtype=jnp.int32)
    col_type_ids = jnp.full(MAX_COLS, CONTINUOUS_ID, dtype=jnp.int32)
    row_data = jnp.array(values, dtype=jnp.float32)
    cluster_id = jnp.array(cluster_id, dtype=jnp.int32)

    ss = _make_empty_suffstats(MAX_COLS, MAX_CLUSTERS, MAX_CATS)
    ss_added = _add_row_to_suffstats(
        *ss, cluster_id, row_data, col_indices, col_type_ids, MAX_CATS
    )
    ss_removed = _remove_row_from_suffstats(
        *ss_added, cluster_id, row_data, col_indices, col_type_ids, MAX_CATS
    )

    for orig, recovered in zip(ss, ss_removed, strict=True):
        np.testing.assert_allclose(np.array(recovered), np.array(orig), atol=1e-5)


@given(
    values=st.lists(binary_val, min_size=MAX_COLS, max_size=MAX_COLS),
    cluster_id=st.integers(min_value=0, max_value=MAX_CLUSTERS - 1),
)
@settings(max_examples=50, deadline=None)
def test_suffstat_add_remove_roundtrip_binary(values, cluster_id):
    """Adding then removing a row recovers original suffstats (binary)."""
    col_indices = jnp.arange(MAX_COLS, dtype=jnp.int32)
    col_type_ids = jnp.full(MAX_COLS, BINARY_ID, dtype=jnp.int32)
    row_data = jnp.array(values, dtype=jnp.float32)
    cluster_id = jnp.array(cluster_id, dtype=jnp.int32)

    ss = _make_empty_suffstats(MAX_COLS, MAX_CLUSTERS, MAX_CATS)
    ss_added = _add_row_to_suffstats(
        *ss, cluster_id, row_data, col_indices, col_type_ids, MAX_CATS
    )
    ss_removed = _remove_row_from_suffstats(
        *ss_added, cluster_id, row_data, col_indices, col_type_ids, MAX_CATS
    )

    for orig, recovered in zip(ss, ss_removed, strict=True):
        np.testing.assert_allclose(np.array(recovered), np.array(orig), atol=1e-5)


@given(
    values=st.lists(category_int, min_size=MAX_COLS, max_size=MAX_COLS),
    cluster_id=st.integers(min_value=0, max_value=MAX_CLUSTERS - 1),
)
@settings(max_examples=50, deadline=None)
def test_suffstat_add_remove_roundtrip_categorical(values, cluster_id):
    """Adding then removing a row recovers original suffstats (categorical)."""
    col_indices = jnp.arange(MAX_COLS, dtype=jnp.int32)
    col_type_ids = jnp.full(MAX_COLS, CATEGORICAL_ID, dtype=jnp.int32)
    row_data = jnp.array(values, dtype=jnp.float32)
    cluster_id = jnp.array(cluster_id, dtype=jnp.int32)

    ss = _make_empty_suffstats(MAX_COLS, MAX_CLUSTERS, MAX_CATS)
    ss_added = _add_row_to_suffstats(
        *ss, cluster_id, row_data, col_indices, col_type_ids, MAX_CATS
    )
    ss_removed = _remove_row_from_suffstats(
        *ss_added, cluster_id, row_data, col_indices, col_type_ids, MAX_CATS
    )

    for orig, recovered in zip(ss, ss_removed, strict=True):
        np.testing.assert_allclose(np.array(recovered), np.array(orig), atol=1e-5)


@given(
    values=st.lists(cyclic_float, min_size=MAX_COLS, max_size=MAX_COLS),
    cluster_id=st.integers(min_value=0, max_value=MAX_CLUSTERS - 1),
)
@settings(max_examples=50, deadline=None)
def test_suffstat_add_remove_roundtrip_cyclic(values, cluster_id):
    """Adding then removing a row recovers original suffstats (cyclic)."""
    col_indices = jnp.arange(MAX_COLS, dtype=jnp.int32)
    col_type_ids = jnp.full(MAX_COLS, CYCLIC_ID, dtype=jnp.int32)
    row_data = jnp.array(values, dtype=jnp.float32)
    cluster_id = jnp.array(cluster_id, dtype=jnp.int32)

    ss = _make_empty_suffstats(MAX_COLS, MAX_CLUSTERS, MAX_CATS)
    ss_added = _add_row_to_suffstats(
        *ss, cluster_id, row_data, col_indices, col_type_ids, MAX_CATS
    )
    ss_removed = _remove_row_from_suffstats(
        *ss_added, cluster_id, row_data, col_indices, col_type_ids, MAX_CATS
    )

    for orig, recovered in zip(ss, ss_removed, strict=True):
        np.testing.assert_allclose(np.array(recovered), np.array(orig), atol=1e-5)


@given(
    values=st.lists(category_int, min_size=MAX_COLS, max_size=MAX_COLS),
    cluster_id=st.integers(min_value=0, max_value=MAX_CLUSTERS - 1),
)
@settings(max_examples=50, deadline=None)
def test_suffstat_add_remove_roundtrip_ordinal(values, cluster_id):
    """Adding then removing a row recovers original suffstats (ordinal)."""
    col_indices = jnp.arange(MAX_COLS, dtype=jnp.int32)
    col_type_ids = jnp.full(MAX_COLS, ORDINAL_ID, dtype=jnp.int32)
    row_data = jnp.array(values, dtype=jnp.float32)
    cluster_id = jnp.array(cluster_id, dtype=jnp.int32)

    ss = _make_empty_suffstats(MAX_COLS, MAX_CLUSTERS, MAX_CATS)
    ss_added = _add_row_to_suffstats(
        *ss, cluster_id, row_data, col_indices, col_type_ids, MAX_CATS
    )
    ss_removed = _remove_row_from_suffstats(
        *ss_added, cluster_id, row_data, col_indices, col_type_ids, MAX_CATS
    )

    for orig, recovered in zip(ss, ss_removed, strict=True):
        np.testing.assert_allclose(np.array(recovered), np.array(orig), atol=1e-5)


@given(
    cluster_id=st.integers(min_value=0, max_value=MAX_CLUSTERS - 1),
)
@settings(max_examples=50, deadline=None)
def test_suffstat_nan_does_not_bias_ordinal(cluster_id):
    """NaN values in ordinal columns should not change any suffstat."""
    col_indices = jnp.arange(MAX_COLS, dtype=jnp.int32)
    col_type_ids = jnp.full(MAX_COLS, ORDINAL_ID, dtype=jnp.int32)
    row_data = jnp.full(MAX_COLS, jnp.nan, dtype=jnp.float32)
    cluster_id = jnp.array(cluster_id, dtype=jnp.int32)

    ss = _make_empty_suffstats(MAX_COLS, MAX_CLUSTERS, MAX_CATS)
    ss_after = _add_row_to_suffstats(
        *ss, cluster_id, row_data, col_indices, col_type_ids, MAX_CATS
    )

    # All suffstats should remain zero — NaN contributes nothing
    for orig, after in zip(ss, ss_after, strict=True):
        np.testing.assert_allclose(np.array(after), np.array(orig), atol=1e-10)


@given(
    v1=st.lists(data_float, min_size=MAX_COLS, max_size=MAX_COLS),
    v2=st.lists(data_float, min_size=MAX_COLS, max_size=MAX_COLS),
    cluster_id=st.integers(min_value=0, max_value=MAX_CLUSTERS - 1),
)
@settings(max_examples=50, deadline=None)
def test_suffstat_add_commutative(v1, v2, cluster_id):
    """Adding rows in any order produces same suffstats (continuous)."""
    col_indices = jnp.arange(MAX_COLS, dtype=jnp.int32)
    col_type_ids = jnp.full(MAX_COLS, CONTINUOUS_ID, dtype=jnp.int32)
    row1 = jnp.array(v1, dtype=jnp.float32)
    row2 = jnp.array(v2, dtype=jnp.float32)
    cid = jnp.array(cluster_id, dtype=jnp.int32)

    ss = _make_empty_suffstats(MAX_COLS, MAX_CLUSTERS, MAX_CATS)

    # Order 1: row1 then row2
    ss_12 = _add_row_to_suffstats(*ss, cid, row1, col_indices, col_type_ids, MAX_CATS)
    ss_12 = _add_row_to_suffstats(*ss_12, cid, row2, col_indices, col_type_ids, MAX_CATS)

    # Order 2: row2 then row1
    ss_21 = _add_row_to_suffstats(*ss, cid, row2, col_indices, col_type_ids, MAX_CATS)
    ss_21 = _add_row_to_suffstats(*ss_21, cid, row1, col_indices, col_type_ids, MAX_CATS)

    for a, b in zip(ss_12, ss_21, strict=True):
        np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-5)


@given(
    values=st.lists(
        st.one_of(data_float, st.just(float("nan"))),
        min_size=MAX_COLS,
        max_size=MAX_COLS,
    ),
    cluster_id=st.integers(min_value=0, max_value=MAX_CLUSTERS - 1),
)
@settings(max_examples=50, deadline=None)
def test_suffstat_nan_transparency(values, cluster_id):
    """NaN values produce zero contribution — add then remove recovers original."""
    col_indices = jnp.arange(MAX_COLS, dtype=jnp.int32)
    col_type_ids = jnp.full(MAX_COLS, CONTINUOUS_ID, dtype=jnp.int32)
    row_data = jnp.array(values, dtype=jnp.float32)
    cid = jnp.array(cluster_id, dtype=jnp.int32)

    ss = _make_empty_suffstats(MAX_COLS, MAX_CLUSTERS, MAX_CATS)
    ss_added = _add_row_to_suffstats(*ss, cid, row_data, col_indices, col_type_ids, MAX_CATS)
    ss_removed = _remove_row_from_suffstats(
        *ss_added, cid, row_data, col_indices, col_type_ids, MAX_CATS
    )

    for orig, recovered in zip(ss, ss_removed, strict=True):
        np.testing.assert_allclose(np.array(recovered), np.array(orig), atol=1e-5)


# ---------------------------------------------------------------------------
# Pack/unpack roundtrip tests
# ---------------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=9999))
@settings(max_examples=10, deadline=None)
def test_pack_unpack_roundtrip_continuous(seed):
    """Pack then unpack preserves column assignments and row assignments."""
    from crosscat.model import initialize
    from crosscat.packed.state import pack_state, unpack_state
    from crosscat.types import ColumnType

    key = jax.random.key(seed)
    n_rows, n_cols = 20, 4
    data = jax.random.normal(key, (n_rows, n_cols))
    col_types = [ColumnType.CONTINUOUS] * n_cols

    k1, k2 = jax.random.split(key)
    state = initialize(k1, data, col_types).state
    packed = pack_state(state)
    recovered = unpack_state(packed, col_types, data=data)

    np.testing.assert_array_equal(
        np.array(state.column_assignments), np.array(recovered.column_assignments)
    )
    assert len(state.views) == len(recovered.views)
    for v_orig, v_rec in zip(state.views, recovered.views, strict=True):
        np.testing.assert_array_equal(
            np.array(v_orig.row_assignments), np.array(v_rec.row_assignments)
        )


@given(seed=st.integers(min_value=0, max_value=9999))
@settings(max_examples=10, deadline=None)
def test_pack_unpack_preserves_hypers(seed):
    """Pack then unpack preserves hyperparameters."""
    from crosscat.model import initialize
    from crosscat.packed.state import pack_state, unpack_state
    from crosscat.types import ColumnType

    key = jax.random.key(seed)
    n_rows, n_cols = 20, 4
    data = jax.random.normal(key, (n_rows, n_cols))
    col_types = [ColumnType.CONTINUOUS] * n_cols

    state = initialize(key, data, col_types).state
    packed = pack_state(state)
    recovered = unpack_state(packed, col_types, data=data)

    for h_orig, h_rec in zip(state.column_hypers, recovered.column_hypers, strict=True):
        np.testing.assert_allclose(float(h_orig.mu), float(h_rec.mu), rtol=1e-5)
        np.testing.assert_allclose(float(h_orig.r), float(h_rec.r), rtol=1e-5)


@given(seed=st.integers(min_value=0, max_value=9999))
@settings(max_examples=10, deadline=None)
def test_pack_unpack_mixed_types(seed):
    """Pack/unpack roundtrip works for mixed column types."""
    from crosscat.model import initialize
    from crosscat.packed.state import pack_state, unpack_state
    from crosscat.types import ColumnType

    key = jax.random.key(seed)
    n_rows = 30
    k1, k2, k3 = jax.random.split(key, 3)
    cont = jax.random.normal(k1, (n_rows, 2))
    binary = jax.random.bernoulli(k2, 0.5, (n_rows, 2)).astype(jnp.float32)
    data = jnp.concatenate([cont, binary], axis=1)
    col_types = [
        ColumnType.CONTINUOUS,
        ColumnType.CONTINUOUS,
        ColumnType.BINARY,
        ColumnType.BINARY,
    ]

    state = initialize(key, data, col_types).state
    packed = pack_state(state)
    recovered = unpack_state(packed, col_types, data=data)

    np.testing.assert_array_equal(
        np.array(state.column_assignments), np.array(recovered.column_assignments)
    )


# ---------------------------------------------------------------------------
# Component model scoring: empty cluster = 0
# ---------------------------------------------------------------------------


@given(
    mu=pos_float,
    r=pos_float,
    s=pos_float,
    nu=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_ng_log_marginal_empty_cluster_is_zero(mu, r, s, nu):
    """Normal-Gamma log marginal for empty cluster (n=0) is 0."""
    result = _ng_log_marginal(
        jnp.array(0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(mu),
        jnp.array(r),
        jnp.array(s),
        jnp.array(nu),
    )
    assert float(result) == 0.0


@given(dir_alpha=pos_float)
@settings(max_examples=50, deadline=None)
def test_dc_log_marginal_empty_cluster_is_zero(dir_alpha):
    """Dirichlet-Categorical log marginal for empty cluster (n=0) is 0."""
    cat_counts = jnp.zeros(MAX_CATS, dtype=jnp.float32)
    result = _dc_log_marginal(jnp.array(0), cat_counts, jnp.array(dir_alpha))
    assert float(result) == 0.0


@given(alpha=pos_float, beta=pos_float)
@settings(max_examples=50, deadline=None)
def test_bb_log_marginal_empty_cluster_is_zero(alpha, beta):
    """Beta-Bernoulli log marginal for empty cluster (n=0) is 0."""
    result = _bb_log_marginal(jnp.array(0), jnp.array(0.0), jnp.array(alpha), jnp.array(beta))
    assert float(result) == 0.0


@given(
    kappa=pos_float,
    vm_a=pos_float,
    vm_mu=cyclic_float,
)
@settings(max_examples=50, deadline=None)
def test_vm_log_marginal_empty_cluster_is_zero(kappa, vm_a, vm_mu):
    """Von Mises log marginal for empty cluster (n=0) is 0."""
    result = _vm_log_marginal(
        jnp.array(0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(kappa),
        jnp.array(vm_a),
        jnp.array(vm_mu),
    )
    assert float(result) == 0.0


# ---------------------------------------------------------------------------
# Component model scoring: finite outputs for valid inputs
# ---------------------------------------------------------------------------


@given(
    n=count_st,
    mu=pos_float,
    r=pos_float,
    s=pos_float,
    nu=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    x=data_float,
)
@settings(max_examples=50, deadline=None)
def test_ng_posterior_predictive_finite(n, mu, r, s, nu, x):
    """Normal-Gamma posterior predictive produces finite values."""
    sum_x = jnp.array(float(n) * mu)
    sum_x_sq = jnp.array(float(n) * (s + mu**2))
    result = _ng_posterior_predictive_logp(
        jnp.array(x),
        jnp.array(n),
        sum_x,
        sum_x_sq,
        jnp.array(mu),
        jnp.array(r),
        jnp.array(s),
        jnp.array(nu),
    )
    assert jnp.isfinite(result), f"Non-finite NG predictive: {result}"


@given(
    n=count_st,
    dir_alpha=pos_float,
    x=category_int,
)
@settings(max_examples=50, deadline=None)
def test_dc_posterior_predictive_finite(n, dir_alpha, x):
    """Dirichlet-Categorical posterior predictive produces finite values."""
    cat_counts = jnp.ones(MAX_CATS, dtype=jnp.float32) * (n / MAX_CATS)
    result = _dc_posterior_predictive_logp(
        jnp.array(float(x)),
        jnp.array(n),
        cat_counts,
        jnp.array(dir_alpha),
    )
    assert jnp.isfinite(result), f"Non-finite DC predictive: {result}"


@given(n=count_st, alpha=pos_float, beta=pos_float, x=binary_val)
@settings(max_examples=50, deadline=None)
def test_bb_posterior_predictive_finite(n, alpha, beta, x):
    """Beta-Bernoulli posterior predictive produces finite values."""
    sum_x = jnp.array(float(n) * 0.5)
    result = _bb_posterior_predictive_logp(
        jnp.array(x),
        jnp.array(n),
        sum_x,
        jnp.array(alpha),
        jnp.array(beta),
    )
    assert jnp.isfinite(result), f"Non-finite BB predictive: {result}"


@given(
    n=count_st,
    kappa=pos_float,
    vm_a=pos_float,
    vm_mu=cyclic_float,
    x=cyclic_float,
)
@settings(max_examples=50, deadline=None)
def test_vm_posterior_predictive_finite(n, kappa, vm_a, vm_mu, x):
    """Von Mises posterior predictive produces finite values."""
    result = _vm_posterior_predictive_logp(
        jnp.array(x),
        jnp.array(n),
        jnp.array(0.0),
        jnp.array(float(n)),
        jnp.array(kappa),
        jnp.array(vm_a),
        jnp.array(vm_mu),
    )
    assert jnp.isfinite(result), f"Non-finite VM predictive: {result}"


# ---------------------------------------------------------------------------
# Unified type dispatch parity
# ---------------------------------------------------------------------------


@given(
    n=count_st,
    mu=pos_float,
    r=pos_float,
    s=pos_float,
    nu=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_unified_log_marginal_matches_ng(n, mu, r, s, nu):
    """unified_log_marginal with CONTINUOUS_ID matches _ng_log_marginal."""
    sum_x = jnp.array(float(n) * mu)
    sum_x_sq = jnp.array(float(n) * (s + mu**2))
    cat_counts = jnp.zeros(MAX_CATS, dtype=jnp.float32)

    specific = _ng_log_marginal(
        jnp.array(n),
        sum_x,
        sum_x_sq,
        jnp.array(mu),
        jnp.array(r),
        jnp.array(s),
        jnp.array(nu),
    )
    unified = unified_log_marginal(
        jnp.array(CONTINUOUS_ID),
        jnp.array(n),
        sum_x,
        sum_x_sq,
        cat_counts,
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(mu),
        jnp.array(r),
        jnp.array(s),
        jnp.array(nu),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.full(MAX_CATS - 1, jnp.inf),
    )

    np.testing.assert_allclose(float(unified), float(specific), rtol=1e-5)


@given(n=count_st, dir_alpha=pos_float)
@settings(max_examples=50, deadline=None)
def test_unified_log_marginal_matches_dc(n, dir_alpha):
    """unified_log_marginal with CATEGORICAL_ID matches _dc_log_marginal."""
    cat_counts = jnp.ones(MAX_CATS, dtype=jnp.float32) * (n / MAX_CATS)

    specific = _dc_log_marginal(jnp.array(n), cat_counts, jnp.array(dir_alpha))
    unified = unified_log_marginal(
        jnp.array(CATEGORICAL_ID),
        jnp.array(n),
        jnp.array(0.0),
        jnp.array(0.0),
        cat_counts,
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(dir_alpha),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.full(MAX_CATS - 1, jnp.inf),
    )

    np.testing.assert_allclose(float(unified), float(specific), rtol=1e-5)


@given(n=count_st, alpha=pos_float, beta=pos_float)
@settings(max_examples=50, deadline=None)
def test_unified_log_marginal_matches_bb(n, alpha, beta):
    """unified_log_marginal with BINARY_ID matches _bb_log_marginal."""
    sum_x = jnp.array(float(n) * 0.5)
    cat_counts = jnp.zeros(MAX_CATS, dtype=jnp.float32)

    specific = _bb_log_marginal(jnp.array(n), sum_x, jnp.array(alpha), jnp.array(beta))
    unified = unified_log_marginal(
        jnp.array(BINARY_ID),
        jnp.array(n),
        sum_x,
        jnp.array(0.0),
        cat_counts,
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(alpha),
        jnp.array(beta),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.full(MAX_CATS - 1, jnp.inf),
    )

    np.testing.assert_allclose(float(unified), float(specific), rtol=1e-5)


@given(
    n=count_st,
    x=data_float,
    mu=pos_float,
    r=pos_float,
    s=pos_float,
    nu=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_unified_posterior_predictive_matches_ng(n, x, mu, r, s, nu):
    """Unified posterior predictive matches NG for CONTINUOUS_ID."""
    sum_x = jnp.array(float(n) * mu)
    sum_x_sq = jnp.array(float(n) * (s + mu**2))
    cat_counts = jnp.zeros(MAX_CATS, dtype=jnp.float32)

    specific = _ng_posterior_predictive_logp(
        jnp.array(x),
        jnp.array(n),
        sum_x,
        sum_x_sq,
        jnp.array(mu),
        jnp.array(r),
        jnp.array(s),
        jnp.array(nu),
    )
    unified = unified_posterior_predictive_logp(
        jnp.array(x),
        jnp.array(CONTINUOUS_ID),
        jnp.array(n),
        sum_x,
        sum_x_sq,
        cat_counts,
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(mu),
        jnp.array(r),
        jnp.array(s),
        jnp.array(nu),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.full(MAX_CATS - 1, jnp.inf),
    )

    np.testing.assert_allclose(float(unified), float(specific), rtol=1e-5)


@given(n=count_st, x=binary_val, alpha=pos_float, beta=pos_float)
@settings(max_examples=50, deadline=None)
def test_unified_posterior_predictive_matches_bb(n, x, alpha, beta):
    """unified_posterior_predictive_logp with BINARY_ID matches _bb_posterior_predictive_logp."""
    sum_x = jnp.array(float(n) * 0.5)
    cat_counts = jnp.zeros(MAX_CATS, dtype=jnp.float32)

    specific = _bb_posterior_predictive_logp(
        jnp.array(x),
        jnp.array(n),
        sum_x,
        jnp.array(alpha),
        jnp.array(beta),
    )
    unified = unified_posterior_predictive_logp(
        jnp.array(x),
        jnp.array(BINARY_ID),
        jnp.array(n),
        sum_x,
        jnp.array(0.0),
        cat_counts,
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(alpha),
        jnp.array(beta),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.full(MAX_CATS - 1, jnp.inf),
    )

    np.testing.assert_allclose(float(unified), float(specific), rtol=1e-5)


@given(
    n=count_st,
    x=cyclic_float,
    kappa=pos_float,
    vm_a=pos_float,
    vm_mu=cyclic_float,
)
@settings(max_examples=50, deadline=None)
def test_batch_vm_matches_unified_vm(n, x, kappa, vm_a, vm_mu):
    """batch_vm_posterior_predictive_logp matches unified for CYCLIC_ID."""
    from crosscat.packed.components import batch_vm_posterior_predictive_logp

    sum_sin = jnp.array(float(n) * jnp.sin(jnp.array(vm_mu)))
    sum_cos = jnp.array(float(n) * jnp.cos(jnp.array(vm_mu)))

    specific = _vm_posterior_predictive_logp(
        jnp.array(x),
        jnp.array(n),
        sum_sin,
        sum_cos,
        jnp.array(kappa),
        jnp.array(vm_a),
        jnp.array(vm_mu),
    )
    batch = batch_vm_posterior_predictive_logp(
        jnp.array([x]),
        jnp.array([n], dtype=jnp.float32),
        jnp.array([float(sum_sin)]),
        jnp.array([float(sum_cos)]),
        jnp.array([kappa]),
        jnp.array([vm_a]),
        jnp.array([vm_mu]),
    )

    np.testing.assert_allclose(float(batch[0]), float(specific), rtol=1e-5)


# ---------------------------------------------------------------------------
# Component scoring: log marginal is non-positive
# ---------------------------------------------------------------------------


@given(
    n=count_st,
    mu=pos_float,
    r=pos_float,
    s=pos_float,
    nu=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_ng_log_marginal_is_finite(n, mu, r, s, nu):
    """Normal-Gamma log marginal produces finite values for valid inputs."""
    sum_x = jnp.array(float(n) * mu)
    sum_x_sq = jnp.array(float(n) * (s + mu**2))
    result = _ng_log_marginal(
        jnp.array(n),
        sum_x,
        sum_x_sq,
        jnp.array(mu),
        jnp.array(r),
        jnp.array(s),
        jnp.array(nu),
    )
    assert jnp.isfinite(result), f"Non-finite NG log marginal: {result}"


@given(n=count_st, alpha=pos_float, beta=pos_float)
@settings(max_examples=50, deadline=None)
def test_bb_log_marginal_is_finite(n, alpha, beta):
    """Beta-Bernoulli log marginal produces finite values for valid inputs."""
    sum_x = jnp.array(float(n) * 0.5)
    result = _bb_log_marginal(jnp.array(n), sum_x, jnp.array(alpha), jnp.array(beta))
    assert jnp.isfinite(result), f"Non-finite BB log marginal: {result}"


# ---------------------------------------------------------------------------
# Ordered logistic property tests
# ---------------------------------------------------------------------------


@given(mu0=data_float, s0=pos_float)
@settings(max_examples=50, deadline=None)
def test_ol_log_marginal_empty_cluster_is_zero(mu0, s0):
    """Ordered logistic log marginal for empty cluster (n=0) is 0."""
    from crosscat.packed.components import _ol_log_marginal

    cat_counts = jnp.zeros(MAX_CATS, dtype=jnp.float32)
    cutpoints = jnp.linspace(-2.0, 2.0, MAX_CATS - 1)
    result = _ol_log_marginal(
        jnp.array(0),
        cat_counts,
        cutpoints,
        jnp.array(mu0),
        jnp.array(s0),
    )
    assert float(result) == 0.0


@given(
    n=count_st,
    mu0=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    s0=pos_float,
)
@settings(max_examples=50, deadline=None)
def test_ol_log_marginal_is_finite(n, mu0, s0):
    """Ordered logistic log marginal produces finite values for valid inputs."""
    from crosscat.packed.components import _ol_log_marginal

    cat_counts = jnp.ones(MAX_CATS, dtype=jnp.float32) * (n / MAX_CATS)
    cutpoints = jnp.linspace(-2.0, 2.0, MAX_CATS - 1)
    result = _ol_log_marginal(
        jnp.array(n),
        cat_counts,
        cutpoints,
        jnp.array(mu0),
        jnp.array(s0),
    )
    assert jnp.isfinite(result), f"Non-finite OL log marginal: {result}"


@given(
    n=count_st,
    x=st.integers(min_value=0, max_value=MAX_CATS - 1),
    mu0=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    s0=pos_float,
)
@settings(max_examples=50, deadline=None)
def test_ol_posterior_predictive_finite(n, x, mu0, s0):
    """Ordered logistic posterior predictive produces finite values."""
    from crosscat.packed.components import _ol_posterior_predictive_logp

    cat_counts = jnp.ones(MAX_CATS, dtype=jnp.float32) * (n / MAX_CATS)
    cutpoints = jnp.linspace(-2.0, 2.0, MAX_CATS - 1)
    result = _ol_posterior_predictive_logp(
        jnp.array(float(x)),
        jnp.array(n),
        cat_counts,
        cutpoints,
        jnp.array(mu0),
        jnp.array(s0),
    )
    assert jnp.isfinite(result), f"Non-finite OL predictive: {result}"


@given(
    n=count_st,
    mu0=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    s0=pos_float,
)
@settings(max_examples=50, deadline=None)
def test_unified_log_marginal_matches_ol(n, mu0, s0):
    """unified_log_marginal with ORDINAL_ID matches _ol_log_marginal."""
    from crosscat.packed.components import _ol_log_marginal

    cat_counts = jnp.ones(MAX_CATS, dtype=jnp.float32) * (n / MAX_CATS)
    cutpoints = jnp.linspace(-2.0, 2.0, MAX_CATS - 1)

    specific = _ol_log_marginal(
        jnp.array(n),
        cat_counts,
        cutpoints,
        jnp.array(mu0),
        jnp.array(s0),
    )
    unified = unified_log_marginal(
        jnp.array(ORDINAL_ID),
        jnp.array(n),
        jnp.array(0.0),
        jnp.array(0.0),
        cat_counts,
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(mu0),
        jnp.array(1.0),
        jnp.array(s0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(0.0),
        cutpoints,
    )

    np.testing.assert_allclose(float(unified), float(specific), rtol=1e-5)


# ---------------------------------------------------------------------------
# Serialization roundtrip property tests
# ---------------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=10, deadline=None)
def test_serialize_deserialize_roundtrip(seed, tmp_path_factory):
    """Serializing then deserializing a packed state preserves all fields."""
    from crosscat.model import initialize
    from crosscat.packed.state import pack_state
    from crosscat.serialization import load_packed_state, save_packed_state
    from crosscat.types import ColumnType

    key = jax.random.key(seed)
    data = jax.random.normal(key, (30, 3))
    column_types = [ColumnType.CONTINUOUS, ColumnType.BINARY, ColumnType.CONTINUOUS]
    data = data.at[:, 1].set(jnp.where(data[:, 1] > 0, 1.0, 0.0))

    state = initialize(key, data, column_types).state
    packed = pack_state(state)

    path = tmp_path_factory.mktemp("ser") / f"test_{seed}.jxc"
    save_packed_state(packed, path, column_types=column_types)
    loaded, _ = load_packed_state(path)

    for field in [
        "view_row_assignments",
        "column_assignments",
        "view_row_crp_alpha",
        "column_crp_alpha",
    ]:
        orig = getattr(packed, field)
        rec = getattr(loaded, field)
        np.testing.assert_array_equal(np.array(orig), np.array(rec))


# ---------------------------------------------------------------------------
# Initialize invariant property tests
# ---------------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=15, deadline=None)
def test_initialize_valid_assignments(seed):
    """initialize() always produces valid row/column assignments."""
    from crosscat.model import initialize
    from crosscat.types import ColumnType

    key = jax.random.key(seed)
    data = jax.random.normal(key, (20, 4))
    column_types = [ColumnType.CONTINUOUS] * 4

    state = initialize(key, data, column_types).state

    # Column assignments cover all views
    assert state.n_views == len(state.views)
    for view in state.views:
        assert view.row_assignments.shape[0] == 20
        # All row assignments are valid cluster indices
        assert int(jnp.max(view.row_assignments)) < len(view.suffstats)


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=15, deadline=None)
def test_pack_unpack_preserves_assignments(seed):
    """Pack then unpack preserves row and column assignments exactly."""
    from crosscat.model import initialize
    from crosscat.packed.state import pack_state, unpack_state
    from crosscat.types import ColumnType

    key = jax.random.key(seed)
    data = jax.random.normal(key, (25, 3))
    column_types = [ColumnType.CONTINUOUS] * 3

    state = initialize(key, data, column_types).state
    packed = pack_state(state)
    recovered = unpack_state(packed, column_types, data=data)

    # Column assignments preserved
    np.testing.assert_array_equal(
        np.array(state.column_assignments),
        np.array(recovered.column_assignments),
    )
    # Row assignments preserved per view
    for v_orig, v_rec in zip(state.views, recovered.views, strict=True):
        np.testing.assert_array_equal(
            np.array(v_orig.row_assignments),
            np.array(v_rec.row_assignments),
        )
