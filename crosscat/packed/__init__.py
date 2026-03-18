"""crosscat.packed — JIT-compatible packed state and kernels for CrossCat.

Sub-modules:
    state       — PackedCrossCatState dataclass, pack/unpack conversions
    components  — Conjugate component scoring (log marginal, posterior predictive)
    suffstats   — Sufficient statistics computation and incremental updates
    kernels     — Gibbs kernels (row/column assignments, hypers, CRP alphas, sweep)
"""

from crosscat.packed.components import (
    unified_log_marginal,
    unified_posterior_predictive_logp,
    unified_sample_posterior_predictive,
)
from crosscat.packed.kernels import (
    packed_gibbs_sweep,
    packed_transition_column_assignments,
    packed_transition_column_hypers,
    packed_transition_crp_alphas,
    packed_transition_row_assignments,
)
from crosscat.packed.state import (
    _ARRAY_FIELDS,
    _ID_TO_TYPE,
    _STATIC_FIELDS,
    _TYPE_TO_ID,
    BINARY_ID,
    CATEGORICAL_ID,
    CONTINUOUS_ID,
    CYCLIC_ID,
    ORDINAL_ID,
    PackedCrossCatState,
    pack_state,
    unpack_state,
)
from crosscat.packed.suffstats import (
    _add_row_to_suffstats,
    _remove_row_from_suffstats,
    compute_suffstats_vectorized,
    recompute_all_suffstats,
)

__all__ = [
    "BINARY_ID",
    "CATEGORICAL_ID",
    "CONTINUOUS_ID",
    "CYCLIC_ID",
    "ORDINAL_ID",
    "PackedCrossCatState",
    "_ARRAY_FIELDS",
    "_ID_TO_TYPE",
    "_STATIC_FIELDS",
    "_TYPE_TO_ID",
    "_add_row_to_suffstats",
    "_remove_row_from_suffstats",
    "compute_suffstats_vectorized",
    "pack_state",
    "packed_gibbs_sweep",
    "packed_transition_column_assignments",
    "packed_transition_column_hypers",
    "packed_transition_crp_alphas",
    "packed_transition_row_assignments",
    "recompute_all_suffstats",
    "unified_log_marginal",
    "unified_posterior_predictive_logp",
    "unified_sample_posterior_predictive",
    "unpack_state",
]
