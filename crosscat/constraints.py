"""Column and row dependency constraint enforcement.

Maps to original CrossCat:
- LocalEngine.ensure_col_dep_constraints() — rejection sampling
- LocalEngine.ensure_row_dep_constraint() — iterative enforcement
"""

from __future__ import annotations

import jax
from jax import Array

from crosscat.packed import pack_state, packed_gibbs_step, unpack_state
from crosscat.types import CrossCatState


def check_column_dep_constraint(
    state: CrossCatState,
    col_a: int,
    col_b: int,
    dependent: bool,
) -> bool:
    """Check if a column dependency constraint is satisfied.

    Args:
        state: CrossCat state.
        col_a: First column index.
        col_b: Second column index.
        dependent: True if columns should be in the same view.

    Returns:
        True if constraint is satisfied.
    """
    same_view = int(state.column_assignments[col_a]) == int(state.column_assignments[col_b])
    return same_view == dependent


def check_all_column_constraints(
    state: CrossCatState,
    constraints: list[tuple[int, int, bool]],
) -> bool:
    """Check if all column dependency constraints are satisfied.

    Args:
        state: CrossCat state.
        constraints: List of (col_a, col_b, dependent) tuples.

    Returns:
        True if all constraints are satisfied.
    """
    return all(check_column_dep_constraint(state, a, b, dep) for a, b, dep in constraints)


def _count_satisfied(
    state: CrossCatState,
    constraints: list[tuple[int, int, bool]],
) -> int:
    """Count how many column constraints are currently satisfied."""
    return sum(1 for a, b, dep in constraints if check_column_dep_constraint(state, a, b, dep))


def _failed_constraints(
    state: CrossCatState,
    constraints: list[tuple[int, int, bool]],
) -> list[tuple[int, int, bool]]:
    """Return list of unsatisfied column constraints."""
    return [
        (a, b, dep)
        for a, b, dep in constraints
        if not check_column_dep_constraint(state, a, b, dep)
    ]


def ensure_col_dep_constraints(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
    constraints: list[tuple[int, int, bool]],
    *,
    max_rejections: int = 100,
    n_sweeps_per_attempt: int = 5,
    return_diagnostics: bool = False,
) -> CrossCatState | None | tuple[CrossCatState | None, dict]:
    """Enforce column dependency constraints via rejection sampling.

    Maps to original LocalEngine.ensure_col_dep_constraints().

    Repeatedly runs inference sweeps and checks if constraints are satisfied.
    Returns the first state that satisfies all constraints, or None if
    max_rejections is exceeded.

    Args:
        rng_key: JAX PRNG key.
        state: Initial CrossCat state.
        data: Observation matrix.
        constraints: List of (col_a, col_b, dependent) tuples.
            dependent=True means columns should be in the same view.
            dependent=False means columns should be in different views.
        max_rejections: Maximum number of rejection attempts.
        n_sweeps_per_attempt: Gibbs sweeps per attempt.
        return_diagnostics: If True, return ``(state, diagnostics)`` tuple.

    Returns:
        State satisfying constraints, or None if not found.
        When ``return_diagnostics=True``, returns ``(state_or_none, diagnostics)``
        where diagnostics is a dict with keys: ``success``, ``n_attempts``,
        ``best_n_satisfied``, ``constraint_failures``.
    """
    n_total = len(constraints)
    best_satisfied = 0
    diagnostics = {
        "success": False,
        "n_attempts": 0,
        "best_n_satisfied": 0,
        "constraint_failures": list(constraints),
    }

    packed = pack_state(state)
    for _attempt in range(max_rejections):
        for _ in range(n_sweeps_per_attempt):
            rng_key, step_key = jax.random.split(rng_key)
            packed = packed_gibbs_step(step_key, packed, data)
        state = unpack_state(packed, state.column_types, data=data)
        diagnostics["n_attempts"] += 1

        satisfied = _count_satisfied(state, constraints)
        if satisfied > best_satisfied:
            best_satisfied = satisfied
            diagnostics["best_n_satisfied"] = best_satisfied

        if satisfied == n_total:
            diagnostics["success"] = True
            diagnostics["constraint_failures"] = []
            if return_diagnostics:
                return state, diagnostics
            return state

    diagnostics["constraint_failures"] = _failed_constraints(state, constraints)
    if return_diagnostics:
        return None, diagnostics
    return None


def check_row_dep_constraint(
    state: CrossCatState,
    row_a: int,
    row_b: int,
    dependent: bool,
    *,
    view_idx: int | None = None,
) -> bool:
    """Check if a row dependency constraint is satisfied.

    Args:
        state: CrossCat state.
        row_a: First row index.
        row_b: Second row index.
        dependent: True if rows should be in the same cluster.
        view_idx: Specific view to check (if None, checks all views).

    Returns:
        True if constraint is satisfied.
    """
    views = [state.views[view_idx]] if view_idx is not None else state.views
    for view in views:
        same_cluster = int(view.row_assignments[row_a]) == int(view.row_assignments[row_b])
        if same_cluster != dependent:
            return False
    return True


def ensure_row_dep_constraint(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
    row_a: int,
    row_b: int,
    dependent: bool,
    *,
    view_idx: int | None = None,
    max_iterations: int = 100,
    n_sweeps_per_attempt: int = 5,
    return_diagnostics: bool = False,
) -> CrossCatState | None | tuple[CrossCatState | None, dict]:
    """Enforce a row dependency constraint via rejection.

    Maps to original LocalEngine.ensure_row_dep_constraint().

    Args:
        rng_key: JAX PRNG key.
        state: CrossCat state.
        data: Observation matrix.
        row_a: First row index.
        row_b: Second row index.
        dependent: True = same cluster, False = different clusters.
        view_idx: Specific view to constrain.
        max_iterations: Maximum attempts.
        n_sweeps_per_attempt: Sweeps per attempt.
        return_diagnostics: If True, return ``(state, diagnostics)`` tuple.

    Returns:
        State satisfying constraint, or None.
        When ``return_diagnostics=True``, returns ``(state_or_none, diagnostics)``
        where diagnostics is a dict with keys: ``success``, ``n_attempts``.
    """
    diagnostics = {"success": False, "n_attempts": 0}

    packed = pack_state(state)
    for _attempt in range(max_iterations):
        for _ in range(n_sweeps_per_attempt):
            rng_key, step_key = jax.random.split(rng_key)
            packed = packed_gibbs_step(step_key, packed, data)
        state = unpack_state(packed, state.column_types, data=data)
        diagnostics["n_attempts"] += 1

        if check_row_dep_constraint(state, row_a, row_b, dependent, view_idx=view_idx):
            diagnostics["success"] = True
            if return_diagnostics:
                return state, diagnostics
            return state

    if return_diagnostics:
        return None, diagnostics
    return None
