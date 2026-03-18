"""State validation utilities.

Maps to original CrossCat validate_utils.py.
Provides consistency checks for CrossCat state, hyperparameters, and suffstats.
"""

from __future__ import annotations

import jax.numpy as jnp

from crosscat.types import ColumnType, CrossCatState


class ValidationError(ValueError):
    """Raised when state validation fails."""


def validate_state(state: CrossCatState, data=None) -> list[str]:
    """Validate consistency of a CrossCatState.

    Maps to original validate_utils.py consistency checks.

    Args:
        state: CrossCat state to validate.
        data: Optional data array for additional checks.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    # Check dimensions
    if state.column_assignments.shape[0] != state.n_cols:
        errors.append(
            f"column_assignments length {state.column_assignments.shape[0]} "
            f"!= n_cols {state.n_cols}"
        )

    if len(state.column_hypers) != state.n_cols:
        errors.append(f"column_hypers length {len(state.column_hypers)} != n_cols {state.n_cols}")

    if len(state.column_types) != state.n_cols:
        errors.append(f"column_types length {len(state.column_types)} != n_cols {state.n_cols}")

    # Check views
    n_views = int(jnp.max(state.column_assignments)) + 1
    if len(state.views) != n_views:
        errors.append(f"Number of views {len(state.views)} != max assignment + 1 = {n_views}")

    # Check each view
    all_cols_in_views = set()
    for v_idx, view in enumerate(state.views):
        # Row assignments shape
        if view.row_assignments.shape[0] != state.n_rows:
            errors.append(
                f"View {v_idx}: row_assignments length {view.row_assignments.shape[0]} "
                f"!= n_rows {state.n_rows}"
            )

        # Column indices should match assignments
        expected_cols = set(
            int(j) for j in range(state.n_cols) if int(state.column_assignments[j]) == v_idx
        )
        actual_cols = set(int(c) for c in view.column_indices.tolist())
        if expected_cols != actual_cols:
            errors.append(
                f"View {v_idx}: column_indices {actual_cols} != expected {expected_cols}"
            )
        all_cols_in_views.update(actual_cols)

        # Suffstats structure
        if view.suffstats is not None:
            n_clusters = int(jnp.max(view.row_assignments)) + 1
            if len(view.suffstats) != n_clusters:
                errors.append(
                    f"View {v_idx}: {len(view.suffstats)} cluster suffstats "
                    f"!= {n_clusters} clusters"
                )
            for c_idx, cluster_ss in enumerate(view.suffstats):
                if len(cluster_ss) != len(view.column_indices):
                    errors.append(
                        f"View {v_idx}, cluster {c_idx}: "
                        f"{len(cluster_ss)} column suffstats != "
                        f"{len(view.column_indices)} columns"
                    )

        # CRP alpha should be positive
        if float(view.row_crp_alpha) <= 0:
            errors.append(f"View {v_idx}: row_crp_alpha = {float(view.row_crp_alpha)} <= 0")

    # All columns should be assigned
    expected_all = set(range(state.n_cols))
    if all_cols_in_views != expected_all:
        missing = expected_all - all_cols_in_views
        errors.append(f"Columns {missing} not assigned to any view")

    # Column CRP alpha should be positive
    if float(state.column_crp_alpha) <= 0:
        errors.append(f"column_crp_alpha = {float(state.column_crp_alpha)} <= 0")

    # Validate hyperparameters
    for j, hypers in enumerate(state.column_hypers):
        col_type = state.column_types[j]
        if hypers.column_type != col_type:
            errors.append(
                f"Column {j}: hypers type {hypers.column_type} != column type {col_type}"
            )

        if col_type == ColumnType.CONTINUOUS:
            if hypers.mu is None or hypers.r is None or hypers.s is None or hypers.nu is None:
                errors.append(f"Column {j}: continuous hypers missing (mu/r/s/nu)")
            elif float(hypers.r) <= 0 or float(hypers.s) <= 0 or float(hypers.nu) <= 0:
                errors.append(f"Column {j}: continuous hypers must be positive (r/s/nu)")

        elif col_type == ColumnType.CATEGORICAL:
            if hypers.dirichlet_alpha is None:
                errors.append(f"Column {j}: categorical hypers missing dirichlet_alpha")
            elif float(hypers.dirichlet_alpha) <= 0:
                errors.append(f"Column {j}: dirichlet_alpha must be positive")

        elif col_type == ColumnType.BINARY:
            if hypers.alpha is None or hypers.beta is None:
                errors.append(f"Column {j}: binary hypers missing alpha/beta")
            elif float(hypers.alpha) <= 0 or float(hypers.beta) <= 0:
                errors.append(f"Column {j}: alpha/beta must be positive")

        elif col_type == ColumnType.CYCLIC:
            if hypers.kappa is None or hypers.vm_mu is None:
                errors.append(f"Column {j}: cyclic hypers missing kappa/vm_mu")
            elif float(hypers.kappa) <= 0:
                errors.append(f"Column {j}: kappa must be positive")

    # Check data compatibility
    if data is not None:
        if data.shape[0] != state.n_rows:
            errors.append(f"Data rows {data.shape[0]} != state.n_rows {state.n_rows}")
        if data.shape[1] != state.n_cols:
            errors.append(f"Data cols {data.shape[1]} != state.n_cols {state.n_cols}")

    return errors


def assert_valid_state(state: CrossCatState, data=None) -> None:
    """Assert that a CrossCatState is valid.

    Raises:
        ValidationError: If validation fails.
    """
    errors = validate_state(state, data)
    if errors:
        raise ValidationError("State validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
