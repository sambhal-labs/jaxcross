"""Gibbs sampling transition kernels for CrossCat in JAX.

Maps to the transition methods in original State.cpp:
- transition_features()                    -> transition_column_assignments
- transition_row_partition_assignments()   -> transition_row_assignments
- transition_column_hyperparameters()      -> transition_column_hypers
- transition_column_crp_alpha()            -> transition_crp_alphas (outer)
- transition_row_partition_hyperparameters() -> transition_crp_alphas (inner)

Key JAX design decisions:
- Column assignment sweep uses jax.lax.scan over columns (replaces C++ for-loop)
- Row assignment sweep uses jax.vmap across views (replaces sequential per-view loop)
- Hyperparameter sampling uses BlackJAX NUTS (replaces grid-based Gibbs)
- All kernels operate on CrossCatState dataclass (replaces X_L/X_D dicts)
"""

from __future__ import annotations

from jax import Array

from crosscat.types import CrossCatState


def transition_column_assignments(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
) -> CrossCatState:
    """Gibbs sweep over column-to-view assignments (outer DP).

    This is the core novelty of jax-crosscat — no published GPU implementation exists.

    Maps to original State::transition_features() in State.cpp (~200 lines).

    For each column j:
    1. Remove column j from its current view (update sufficient stats)
    2. For each existing view v, compute log p(z_j = v | z_{-j}, data, hypers)
       using CRP prior + marginal likelihood of column j's data under view v's clusters
    3. Also compute probability of a new singleton view
    4. Sample new assignment from categorical distribution
    5. Add column j to chosen view (update sufficient stats)

    Uses jax.lax.scan to iterate over columns within a single XLA kernel,
    avoiding Python-level loops and enabling GPU acceleration.

    Args:
        rng_key: JAX PRNG key.
        state: Current CrossCat state.
        data: Observation matrix, shape (n_rows, n_cols).

    Returns:
        Updated CrossCatState with new column assignments.
    """
    raise NotImplementedError(
        "outer DP column partitioning via jax.lax.scan — Week 2-4 deliverable"
    )


def transition_row_assignments(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
) -> CrossCatState:
    """Gibbs sweep over row-to-cluster assignments (inner DP), all views in parallel.

    Maps to original State::transition_row_partition_assignments() in State.cpp,
    which calls View::transition_row_partition_assignments() per view.

    For each view v (in parallel via jax.vmap):
        For each row i:
        1. Remove row i from its current cluster (update sufficient stats)
        2. For each existing cluster c, compute log p(z_i = c | z_{-i}, data_row, hypers)
           using CRP prior + product of component model likelihoods across view's columns
        3. Also compute probability of a new singleton cluster
        4. Sample new assignment from categorical distribution
        5. Add row i to chosen cluster (update sufficient stats)

    Uses jax.vmap across views to parallelize independent row clusterings on GPU.

    Args:
        rng_key: JAX PRNG key.
        state: Current CrossCat state.
        data: Observation matrix, shape (n_rows, n_cols).

    Returns:
        Updated CrossCatState with new row assignments per view.
    """
    raise NotImplementedError(
        "inner DP row clustering via jax.vmap — Week 4-6 deliverable"
    )


def transition_column_hypers(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
) -> CrossCatState:
    """Gibbs sample component model hyperparameters for each column.

    Maps to original State::transition_column_hyperparameters() which calls
    ComponentModel::sample_hypers() per column using grid-based Gibbs.

    For each column j:
        For each hyperparameter h of column j's component model:
        1. Evaluate marginal likelihood over all clusters at each grid point
        2. Sample from categorical over grid weighted by likelihood * prior

    In this JAX implementation, we may replace grids with BlackJAX NUTS
    for continuous hyperparameters, or retain grids for efficiency.

    Args:
        rng_key: JAX PRNG key.
        state: Current CrossCat state.
        data: Observation matrix, shape (n_rows, n_cols).

    Returns:
        Updated CrossCatState with new column hyperparameters.
    """
    raise NotImplementedError("column hyperparameter sampling — Week 6-7 deliverable")


def transition_crp_alphas(
    rng_key: Array,
    state: CrossCatState,
) -> CrossCatState:
    """Sample CRP concentration parameters via BlackJAX NUTS.

    Replaces original grid-based sampling from:
    - State::transition_column_crp_alpha() — outer DP alpha
    - State::transition_row_partition_hyperparameters() — inner DP alphas per view

    The original used grid-based Gibbs (evaluate CRP log-prob at each grid point,
    sample from categorical). BlackJAX NUTS is more efficient and doesn't require
    manual grid construction.

    Args:
        rng_key: JAX PRNG key.
        state: Current CrossCat state.

    Returns:
        Updated CrossCatState with new CRP alpha values.
    """
    raise NotImplementedError(
        "BlackJAX NUTS hyperparameter sampling — Week 6-7 deliverable"
    )


def gibbs_sweep(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
    *,
    n_sweeps: int = 1,
    kernels: tuple[str, ...] = (
        "column_assignments",
        "row_assignments",
        "column_hypers",
        "crp_alphas",
    ),
) -> CrossCatState:
    """Run one or more full Gibbs sweeps combining all transition kernels.

    Maps to original LocalEngine.analyze() which calls State.transition()
    with a configurable kernel_list.

    Uses jax.lax.scan to iterate sweeps, keeping the entire chain within
    a single XLA computation for maximum GPU utilization.

    Args:
        rng_key: JAX PRNG key.
        state: Current CrossCat state.
        data: Observation matrix, shape (n_rows, n_cols).
        n_sweeps: Number of full sweeps to run.
        kernels: Which transition kernels to include per sweep.

    Returns:
        Updated CrossCatState after all sweeps.
    """
    raise NotImplementedError("full Gibbs sweep via jax.lax.scan — Week 4-6 deliverable")
