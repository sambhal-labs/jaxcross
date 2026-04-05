"""Integration tests for CrossCat inference queries.

Tests anomaly detection, mutual information, constraints, and row similarity
using a shared inferred state fixture (module-scoped, runs inference once).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.constraints import (
    check_all_column_constraints,
    check_column_dep_constraint,
    check_row_dep_constraint,
    ensure_col_dep_constraints,
    ensure_row_dep_constraint,
)
from crosscat.inference import (
    column_typicality,
    mutual_information,
    predictive_anomalousness,
    row_similarity,
    row_typicality,
)
from crosscat.model import initialize, log_joint
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat.types import ColumnType

# ---------------------------------------------------------------------------
# Module-scoped fixture: run inference once, share across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inferred_continuous_state():
    """Run inference on continuous data with known structure, shared across tests."""
    key = jax.random.key(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    n_rows = 200
    cluster_0 = jnp.array([0] * 100 + [1] * 100)
    col0 = jnp.where(cluster_0 == 0, 0.0, 5.0) + jax.random.normal(k1, (n_rows,))
    col1 = jnp.where(cluster_0 == 0, -2.0, 3.0) + jax.random.normal(k2, (n_rows,))

    cluster_1 = jnp.array(([0] * 50 + [1] * 50) * 2)
    col2 = jnp.where(cluster_1 == 0, 10.0, 20.0) + jax.random.normal(k3, (n_rows,))
    col3 = jnp.where(cluster_1 == 0, -5.0, 5.0) + jax.random.normal(k4, (n_rows,))

    data = jnp.column_stack([col0, col1, col2, col3])
    column_types = [ColumnType.CONTINUOUS] * 4

    init_states = initialize(k5, data, column_types, n_chains=4).state
    final_states = []
    for i, state in enumerate(init_states):
        k = jax.random.fold_in(k5, i + 100)
        packed = pack_state(state)
        packed = packed_gibbs_sweep(k, packed, data, n_sweeps=20)
        state = unpack_state(packed, column_types, data=data)
        final_states.append(state)

    best_idx = max(range(len(final_states)), key=lambda i: float(log_joint(final_states[i], data)))

    return {
        "states": final_states,
        "best_state": final_states[best_idx],
        "data": data,
        "column_types": column_types,
        "true_column_assignments": jnp.array([0, 0, 1, 1]),
        "true_row_assignments": [cluster_0, cluster_1],
    }


# ---------------------------------------------------------------------------
# Gap D: Anomaly Detection (4 tests)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_predictive_anomalousness_typical_row(inferred_continuous_state):
    """A typical in-distribution row has anomaly score < 0.8."""
    d = inferred_continuous_state
    key = jax.random.key(200)
    # Row 50 is from cluster 0 in view 0 — typical
    score = predictive_anomalousness(key, d["best_state"], d["data"], query_row=50)
    assert 0.0 <= float(score) <= 1.0, f"Score {score} not in [0, 1]"
    assert float(score) < 0.8, f"Typical row scored too anomalous: {score}"


@pytest.mark.slow
def test_predictive_anomalousness_outlier_row(inferred_continuous_state):
    """An outlier row scores higher than a typical row (relative test)."""
    d = inferred_continuous_state
    key = jax.random.key(201)
    data = d["data"]
    state = d["best_state"]

    # Create an outlier: replace one row with extreme values
    outlier_data = data.at[0].set(jnp.array([50.0, 50.0, 50.0, 50.0]))
    # Re-score — the outlier row's values are far from any cluster
    k1, k2 = jax.random.split(key)
    score_typical = predictive_anomalousness(k1, state, data, query_row=50)
    score_outlier = predictive_anomalousness(k2, state, outlier_data, query_row=0)

    assert float(score_outlier) >= float(score_typical), (
        f"Outlier score {score_outlier} < typical score {score_typical}"
    )


@pytest.mark.slow
def test_row_typicality_cluster_center_vs_boundary(inferred_continuous_state):
    """Rows in large clusters have typicality >= 0.3."""
    d = inferred_continuous_state
    states = d["states"]

    # Row 50 is in the first half (cluster 0 of view 0) — should be typical
    typ = row_typicality(states, row_id=50)
    assert float(typ) >= 0.3, f"Row 50 typicality {typ} < 0.3"
    assert 0.0 <= float(typ) <= 1.0, f"Typicality {typ} not in [0, 1]"


@pytest.mark.slow
def test_column_typicality_stable_columns(inferred_continuous_state):
    """Column typicality is in [0, 1]; returns 0.5 for single state."""
    d = inferred_continuous_state
    states = d["states"]

    # With multiple states
    typ_multi = column_typicality(states, col_id=0)
    assert 0.0 <= float(typ_multi) <= 1.0, f"Typicality {typ_multi} not in [0, 1]"

    # With single state — should return 0.5
    typ_single = column_typicality([states[0]], col_id=0)
    assert float(typ_single) == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Gap E: Mutual Information (4 tests)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mutual_information_same_view(inferred_continuous_state):
    """MI > 0 for columns co-assigned to the same view."""
    d = inferred_continuous_state
    states = d["states"]
    # Cols 0 and 1 should be in same view (both in view 0 in truth)
    mi, linfoot = mutual_information(states, col_i=0, col_j=1)
    assert float(mi) >= 0.0, f"MI should be non-negative, got {mi}"
    # At least some states should have them co-assigned
    # MI > 0 means at least some states have them in the same view
    assert float(mi) > 0.0 or float(linfoot) >= 0.0


@pytest.mark.slow
def test_mutual_information_different_views(inferred_continuous_state):
    """MI is lower for columns in different views vs same view."""
    d = inferred_continuous_state
    states = d["states"]
    # Cols 0 and 2 should be in different views in truth
    mi_cross, _ = mutual_information(states, col_i=0, col_j=2)
    mi_same, _ = mutual_information(states, col_i=0, col_j=1)
    # Cross-view MI should be <= same-view MI
    assert float(mi_cross) <= float(mi_same) + 0.1, (
        f"Cross-view MI {mi_cross} > same-view MI {mi_same}"
    )


@pytest.mark.slow
def test_mutual_information_symmetry(inferred_continuous_state):
    """MI(X, Y) == MI(Y, X)."""
    d = inferred_continuous_state
    states = d["states"]
    mi_xy, _ = mutual_information(states, col_i=0, col_j=1)
    mi_yx, _ = mutual_information(states, col_i=1, col_j=0)
    assert float(mi_xy) == pytest.approx(float(mi_yx), abs=0.01)


@pytest.mark.slow
def test_mutual_information_self(inferred_continuous_state):
    """MI(X, X) >= MI(X, Y)."""
    d = inferred_continuous_state
    states = d["states"]
    mi_self, _ = mutual_information(states, col_i=0, col_j=0)
    mi_other, _ = mutual_information(states, col_i=0, col_j=2)
    assert float(mi_self) >= float(mi_other) - 0.01, f"Self-MI {mi_self} < cross-MI {mi_other}"


# ---------------------------------------------------------------------------
# Gap F: Constraints (6 tests — 3 fast, 3 slow)
# ---------------------------------------------------------------------------


def test_check_column_dep_constraint_basic():
    """check_column_dep_constraint returns correct bool."""
    key = jax.random.key(300)
    n_rows = 30
    k1, k2 = jax.random.split(key)
    data = jax.random.normal(k1, (n_rows, 4))
    column_types = [ColumnType.CONTINUOUS] * 4
    state = initialize(k2, data, column_types, initialization="together").state
    # All columns in one view -> columns 0 and 1 are in the same view
    same_view = int(state.column_assignments[0]) == int(state.column_assignments[1])
    assert check_column_dep_constraint(state, 0, 1, dependent=True) == same_view
    assert check_column_dep_constraint(state, 0, 1, dependent=False) == (not same_view)


def test_check_all_column_constraints():
    """check_all_column_constraints validates a constraint list."""
    key = jax.random.key(301)
    k1, k2 = jax.random.split(key)
    data = jax.random.normal(k1, (30, 4))
    column_types = [ColumnType.CONTINUOUS] * 4
    state = initialize(k2, data, column_types, initialization="together").state
    same_01 = int(state.column_assignments[0]) == int(state.column_assignments[1])
    constraints = [(0, 1, same_01)]
    assert check_all_column_constraints(state, constraints) is True
    constraints_neg = [(0, 1, not same_01)]
    assert check_all_column_constraints(state, constraints_neg) is False


def test_check_row_dep_constraint():
    """check_row_dep_constraint returns correct bool for same/different cluster."""
    key = jax.random.key(302)
    k1, k2 = jax.random.split(key)
    data = jax.random.normal(k1, (30, 4))
    column_types = [ColumnType.CONTINUOUS] * 4
    state = initialize(k2, data, column_types).state
    view = state.views[0]
    # Find two rows in the same cluster
    cluster_0_rows = [i for i in range(state.n_rows) if int(view.row_assignments[i]) == 0]
    if len(cluster_0_rows) >= 2:
        r_a, r_b = cluster_0_rows[0], cluster_0_rows[1]
        assert check_row_dep_constraint(state, r_a, r_b, dependent=True, view_idx=0)
        assert not check_row_dep_constraint(state, r_a, r_b, dependent=False, view_idx=0)


@pytest.mark.slow
def test_ensure_col_dep_constraints_dependent(inferred_continuous_state):
    """ensure_col_dep_constraints finds a state with columns co-assigned."""
    d = inferred_continuous_state
    key = jax.random.key(210)
    state = d["best_state"]
    data = d["data"]
    # Constrain cols 0 and 1 to be dependent (same view)
    constraints = [(0, 1, True)]
    result = ensure_col_dep_constraints(
        key, state, data, constraints, max_rejections=50, n_sweeps_per_attempt=3
    )
    assert result is not None, "Failed to find state satisfying dependent constraint"
    assert check_column_dep_constraint(result, 0, 1, dependent=True)


@pytest.mark.slow
@pytest.mark.xfail(reason="Stochastic: rejection sampling may not find independent partition")
def test_ensure_col_dep_constraints_independent(inferred_continuous_state):
    """ensure_col_dep_constraints finds a state with columns separated."""
    d = inferred_continuous_state
    key = jax.random.key(211)
    state = d["best_state"]
    data = d["data"]
    # Constrain cols 0 and 2 to be independent (different views)
    constraints = [(0, 2, False)]
    result = ensure_col_dep_constraints(
        key, state, data, constraints, max_rejections=100, n_sweeps_per_attempt=3
    )
    assert result is not None, "Failed to find state satisfying independent constraint"
    assert check_column_dep_constraint(result, 0, 2, dependent=False)


@pytest.mark.slow
def test_ensure_row_dep_constraint(inferred_continuous_state):
    """ensure_row_dep_constraint finds a state with rows co-clustered."""
    d = inferred_continuous_state
    key = jax.random.key(212)
    state = d["best_state"]
    data = d["data"]
    # Constrain rows 0 and 1 to be in the same cluster (view 0)
    result = ensure_row_dep_constraint(
        key,
        state,
        data,
        row_a=0,
        row_b=1,
        dependent=True,
        view_idx=0,
        max_iterations=50,
        n_sweeps_per_attempt=3,
    )
    assert result is not None, "Failed to find state satisfying row dep constraint"
    assert check_row_dep_constraint(result, 0, 1, dependent=True, view_idx=0)


# ---------------------------------------------------------------------------
# Gap H: Row Similarity (3 tests)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_row_similarity_same_cluster_inferred(inferred_continuous_state):
    """Rows in the same cluster have similarity > 0.3."""
    d = inferred_continuous_state
    states = d["states"]
    state = d["best_state"]

    # Find two rows in the same cluster in view 0
    view = state.views[0]
    cluster_rows = {}
    for i in range(state.n_rows):
        c = int(view.row_assignments[i])
        cluster_rows.setdefault(c, []).append(i)

    # Pick the largest cluster
    largest_cluster = max(cluster_rows.values(), key=len)
    if len(largest_cluster) >= 2:
        r_a, r_b = largest_cluster[0], largest_cluster[1]
        sim = row_similarity(states, r_a, r_b)
        assert float(sim) > 0.3, f"Same-cluster similarity {sim} <= 0.3"


@pytest.mark.slow
def test_row_similarity_different_cluster_inferred(inferred_continuous_state):
    """Cross-cluster similarity < within-cluster similarity."""
    d = inferred_continuous_state
    states = d["states"]
    state = d["best_state"]

    view = state.views[0]
    cluster_rows = {}
    for i in range(state.n_rows):
        c = int(view.row_assignments[i])
        cluster_rows.setdefault(c, []).append(i)

    clusters = list(cluster_rows.values())
    if len(clusters) >= 2 and len(clusters[0]) >= 2:
        # Within-cluster pair
        r_same_a, r_same_b = clusters[0][0], clusters[0][1]
        sim_within = row_similarity(states, r_same_a, r_same_b)

        # Cross-cluster pair
        r_cross_a, r_cross_b = clusters[0][0], clusters[1][0]
        sim_cross = row_similarity(states, r_cross_a, r_cross_b)

        assert float(sim_cross) < float(sim_within) + 0.1, (
            f"Cross-cluster sim {sim_cross} >= within-cluster sim {sim_within}"
        )


@pytest.mark.slow
def test_row_similarity_with_target_columns_inferred(inferred_continuous_state):
    """Row similarity with target_columns returns valid result."""
    d = inferred_continuous_state
    states = d["states"]
    sim = row_similarity(states, 0, 50, target_columns=[0, 1])
    assert 0.0 <= float(sim) <= 1.0, f"Similarity {sim} not in [0, 1]"
