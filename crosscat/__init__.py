"""jax-crosscat: GPU-accelerated nonparametric cross-categorization in JAX."""

__version__ = "0.2.0"

from crosscat.gibbs import gibbs_sweep
from crosscat.inference import (
    mutual_information,
    predictive_probability,
    predictive_sample,
)
from crosscat.model import initialize, insert_rows, log_joint
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
    "gibbs_sweep",
    "initialize",
    "insert_rows",
    "log_joint",
    "mutual_information",
    "predictive_probability",
    "predictive_sample",
]
