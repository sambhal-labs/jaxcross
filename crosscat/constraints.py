"""Column and row dependency constraint enforcement.

Maps to original CrossCat:
- LocalEngine.ensure_col_dep_constraints() — rejection sampling
- LocalEngine.ensure_row_dep_constraint() — iterative enforcement
"""

from __future__ import annotations

import jax
from jax import Array

from crosscat.gibbs import gibbs_sweep
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
    return all(
        check_column_dep_constraint(state, a, b, dep)
        for a, b, dep in constraints
    )


def ensure_col_dep_constraints(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
    constraints: list[tuple[int, int, bool]],
    *,
    max_rejections: int = 100,
    n_sweeps_per_attempt: int = 5,
) -> CrossCatState | None:
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

    Returns:
        State satisfying constraints, or None if not found.
    """
    for attempt in range(max_rejections):
        rng_key, subkey = jax.random.split(rng_key)
        state = gibbs_sweep(subkey, state, data, n_sweeps=n_sweeps_per_attempt)

        if check_all_column_constraints(state, constraints):
            return state

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
) -> CrossCatState | None:
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

    Returns:
        State satisfying constraint, or None.
    """
    for attempt in range(max_iterations):
        rng_key, subkey = jax.random.split(rng_key)
        state = gibbs_sweep(
            subkey, state, data,
            n_sweeps=n_sweeps_per_attempt,
            kernels=("row_assignments",),
        )

        if check_row_dep_constraint(state, row_a, row_b, dependent, view_idx=view_idx):
            return state

    return None
