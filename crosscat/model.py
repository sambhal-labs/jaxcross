"""CrossCat model initialization and joint probability.

Maps to the original State.h / State.cpp from probcomp/crosscat.
The original State class managed initialization from prior, transition
dispatching, and state serialization. Here we separate concerns:
- model.py: initialization and scoring
- gibbs.py: transition kernels
- types.py: state representation
"""

from __future__ import annotations

from jax import Array

from crosscat.types import ColumnType, CrossCatState


def initialize(
    rng_key: Array,
    data: Array,
    column_types: list[ColumnType],
    *,
    n_chains: int = 1,
    column_crp_alpha: float = 1.0,
    row_crp_alpha: float = 1.0,
) -> CrossCatState | list[CrossCatState]:
    """Initialize CrossCat state(s) from the prior.

    Maps to original LocalEngine.initialize() which calls State.__init__
    with initialization='from_the_prior'.

    Procedure (following original CrossCat):
    1. Sample column-to-view assignments from CRP(column_crp_alpha)
    2. For each view, sample row-to-cluster assignments from CRP(row_crp_alpha)
    3. Compute sufficient statistics for each (cluster, column) pair
    4. Initialize column hyperparameters from data-driven defaults

    Args:
        rng_key: JAX PRNG key.
        data: Observation matrix, shape (n_rows, n_cols).
        column_types: Type specification per column.
        n_chains: Number of independent chains to initialize.
        column_crp_alpha: Initial outer DP concentration.
        row_crp_alpha: Initial inner DP concentration per view.

    Returns:
        Single CrossCatState if n_chains=1, else list of states.
    """
    raise NotImplementedError("initialization from prior — Week 2-4 deliverable")


def log_joint(state: CrossCatState, data: Array) -> Array:
    """Compute the joint log probability of state and data.

    log p(state, data) = log p(column_partition | alpha_outer)
                       + sum_v log p(row_partition_v | alpha_v)
                       + sum_v sum_c sum_col log p(data[rows_c, col] | hypers_col)
                       + log p(alpha_outer) + sum_v log p(alpha_v)
                       + sum_col log p(hypers_col)

    Maps to scoring logic spread across original State.cpp, View.cpp,
    and numerics.cpp (calc_crp_logp, calc_continuous_logp, etc.).

    Args:
        state: Current CrossCat state.
        data: Observation matrix, shape (n_rows, n_cols).

    Returns:
        Scalar log joint probability.
    """
    raise NotImplementedError("joint log probability — Week 2-4 deliverable")
