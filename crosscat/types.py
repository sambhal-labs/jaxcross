"""JAX-native state representations for CrossCat.

Maps the original CrossCat (X_L, X_D) dictionary pair into typed dataclasses
compatible with JAX transformations (jit, vmap, scan).

Original CrossCat state structure (probcomp/crosscat):
    X_L = {
        'column_partition': {'assignments': [...], 'hypers': {'alpha': ...}, 'counts': [...]},
        'column_hypers': [{'r': ..., 'nu': ..., 's': ..., 'mu': ...}, ...],
        'view_state': [{
            'row_partition_model': {'hypers': {'alpha': ...}, 'counts': [...]},
            'column_component_suffstats': [[{...}, ...], ...],
        }, ...]
    }
    X_D = [[cluster_idx_per_row], ...]  # one list per view
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array


class ColumnType(enum.Enum):
    """Supported column data types.

    Original CrossCat supported: continuous, multinomial, cyclic.
    This implementation adds: ordinal, binary.
    """

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BINARY = "binary"


@dataclass
class SufficientStats:
    """Sufficient statistics for a single cluster-column pair.

    For continuous (Normal-Gamma):
        count, sum_x, sum_x_sq — mirrors original ContinuousComponentModel suffstats.
    For categorical (Dirichlet-Categorical):
        count, category_counts — mirrors original MultinomialComponentModel suffstats.
    For binary (Beta-Bernoulli):
        count, sum_x — number of observations and number of 1s.
    For ordinal (Ordered Logistic):
        count, level_counts — observations per ordinal level.
    """

    column_type: ColumnType
    count: Array  # scalar int — number of observations in this cluster-column
    # Continuous
    sum_x: Array | None = None  # scalar float
    sum_x_sq: Array | None = None  # scalar float
    # Categorical / Ordinal
    category_counts: Array | None = None  # shape (n_categories,)


@dataclass
class ColumnHypers:
    """Per-column hyperparameters.

    Maps to original X_L['column_hypers'][col_idx].

    For continuous (Normal-Gamma conjugate):
        mu, r, s, nu — Normal-Inverse-Gamma prior parameters.
        Original: {'mu': float, 'r': float, 's': float, 'nu': float}
    For categorical (Dirichlet-Categorical):
        dirichlet_alpha — symmetric Dirichlet concentration.
        Original: {'dirichlet_alpha': float}
    For binary (Beta-Bernoulli):
        alpha, beta — Beta prior parameters.
    For ordinal (Ordered Logistic):
        cutpoints — ordered threshold parameters.
    """

    column_type: ColumnType
    # Continuous (Normal-Gamma)
    mu: Array | None = None  # prior mean
    r: Array | None = None  # prior precision scale
    s: Array | None = None  # prior variance scale
    nu: Array | None = None  # prior degrees of freedom
    # Categorical
    dirichlet_alpha: Array | None = None
    # Binary
    alpha: Array | None = None
    beta: Array | None = None
    # Ordinal
    cutpoints: Array | None = None


@dataclass
class ViewState:
    """State for a single view (column group) in the CrossCat partition.

    Maps to one element of X_L['view_state'] combined with X_D[view_idx].

    A view contains:
    - A set of columns assigned to it
    - An independent row clustering (inner DP)
    - Sufficient statistics per (cluster, column) pair
    """

    # Which columns belong to this view — shape (n_cols_in_view,)
    column_indices: Array
    # Row-to-cluster assignments — shape (n_rows,)
    # Maps to X_D[view_idx]
    row_assignments: Array
    # CRP concentration for row clustering in this view
    # Maps to X_L['view_state'][v]['row_partition_model']['hypers']['alpha']
    row_crp_alpha: Array  # scalar float
    # Sufficient statistics: nested structure [cluster_idx][col_idx_in_view]
    # For JAX compatibility, stored as padded arrays rather than ragged lists
    suffstats: Any = None  # will be a pytree of arrays


@dataclass
class CrossCatState:
    """Full CrossCat state — the JAX equivalent of (X_L, X_D).

    The two-level Dirichlet Process structure:
    - Outer DP: partitions columns into views (column_assignments + column_crp_alpha)
    - Inner DP: independently clusters rows within each view (ViewState.row_assignments)

    Each (cluster, column) pair maintains sufficient statistics for its
    component model, enabling collapsed Gibbs sampling without storing
    per-observation parameters.
    """

    # Column-to-view assignments — shape (n_columns,)
    # Maps to X_L['column_partition']['assignments']
    column_assignments: Array
    # CRP concentration for column partitioning (outer DP)
    # Maps to X_L['column_partition']['hypers']['alpha']
    column_crp_alpha: Array  # scalar float
    # Per-column hyperparameters — one per column
    column_hypers: list[ColumnHypers]
    # Per-column type specification
    column_types: list[ColumnType]
    # View states — one per unique view
    views: list[ViewState]
    # Number of data rows (needed for CRP calculations)
    n_rows: int
    # Number of data columns
    n_cols: int

    @property
    def n_views(self) -> int:
        """Number of active views (column groups)."""
        return len(self.views)

    @property
    def view_counts(self) -> Array:
        """Number of columns per view."""
        return jnp.bincount(self.column_assignments, length=self.n_views)
