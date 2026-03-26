"""crosscat.packed — JIT-compatible packed state and kernels for CrossCat.

Sub-modules:
    state       — PackedCrossCatState dataclass, pack/unpack conversions
    components  — Conjugate component scoring (log marginal, posterior predictive)
    suffstats   — Sufficient statistics computation and incremental updates
    kernels     — Gibbs kernels (row/column assignments, hypers, CRP alphas, sweep)
    aot_cache   — XLA persistent compilation cache
"""

from crosscat.packed.aot_cache import clear_cache, compile_kernels, enable_xla_cache
from crosscat.packed.components import (
    unified_log_marginal,
    unified_posterior_predictive_logp,
    unified_sample_posterior_predictive,
)
from crosscat.packed.kernels import (
    multi_chain_packed_gibbs_sweep,
    packed_gibbs_step,
    packed_gibbs_sweep,
    packed_insert_rows,
    packed_log_joint,
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
    batch_packed_states,
    pack_state,
    select_best_chain,
    unbatch_packed_states,
    unpack_state,
)
from crosscat.packed.suffstats import (
    _add_row_to_suffstats,
    _remove_row_from_suffstats,
    compute_suffstats_vectorized,
    recompute_all_suffstats,
)

# Auto-enable XLA persistent cache so compiled kernels are reused across runs.
enable_xla_cache()

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
    "batch_packed_states",
    "clear_cache",
    "compile_kernels",
    "compute_suffstats_vectorized",
    "enable_xla_cache",
    "multi_chain_packed_gibbs_sweep",
    "pack_state",
    "packed_gibbs_step",
    "packed_gibbs_sweep",
    "packed_insert_rows",
    "packed_log_joint",
    "packed_transition_column_assignments",
    "packed_transition_column_hypers",
    "packed_transition_crp_alphas",
    "packed_transition_row_assignments",
    "recompute_all_suffstats",
    "select_best_chain",
    "unbatch_packed_states",
    "unified_log_marginal",
    "unified_posterior_predictive_logp",
    "unified_sample_posterior_predictive",
    "unpack_state",
]
