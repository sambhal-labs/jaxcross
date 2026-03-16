"""jax-crosscat: GPU-accelerated nonparametric cross-categorization in JAX."""

__version__ = "0.1.0"

from crosscat.types import (
    ColumnHypers,
    ColumnType,
    CrossCatState,
    SufficientStats,
    ViewState,
)

__all__ = [
    "ColumnHypers",
    "ColumnType",
    "CrossCatState",
    "SufficientStats",
    "ViewState",
]
