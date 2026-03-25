"""jax-crosscat: GPU-accelerated nonparametric cross-categorization in JAX."""

__version__ = "0.9.0"

from crosscat.constraints import (
    check_column_dep_constraint,
    ensure_col_dep_constraints,
    ensure_row_dep_constraint,
)
from crosscat.data_utils import (
    discretize_column,
    gen_column_metadata,
    guess_column_type,
    guess_column_types,
    read_csv,
    write_csv,
)
from crosscat.diagnostics import (
    adjusted_rand_index,
    collect_diagnostics,
    column_partition_ari,
    evaluate_imputation,
    mean_test_log_likelihood,
    random_holdout_mask,
    row_partition_ari,
)
from crosscat.gibbs import gibbs_sweep
from crosscat.inference import (
    column_typicality,
    conditional_entropy,
    credible_interval,
    dependence_matrix,
    dependence_probability,
    impute_and_confidence,
    joint_predictive_probability,
    mutual_information,
    predictive_anomalousness,
    predictive_cdf,
    predictive_probability,
    predictive_sample,
    row_similarity,
    row_typicality,
    sample_and_insert,
)
from crosscat.model import initialize, insert_rows, log_joint
from crosscat.packed import (
    PackedCrossCatState,
    batch_packed_states,
    multi_chain_packed_gibbs_sweep,
    pack_state,
    packed_gibbs_sweep,
    packed_log_joint,
    packed_transition_column_assignments,
    packed_transition_column_hypers,
    packed_transition_crp_alphas,
    packed_transition_row_assignments,
    select_best_chain,
    unbatch_packed_states,
    unpack_state,
)
from crosscat.packed_inference import (
    packed_anomaly_score,
    packed_dependence_matrix,
    packed_dependence_probability,
    packed_impute_and_confidence,
    packed_mutual_information,
    packed_predictive_cdf,
    packed_predictive_probability,
    packed_predictive_sample,
    packed_row_similarity,
)
from crosscat.serialization import (
    load_latest_checkpoint,
    load_packed_state,
    load_state,
    save_checkpoint,
    save_packed_state,
    save_state,
)
from crosscat.synthetic import add_missing_data, generate_crosscat_data
from crosscat.types import (
    ColumnHypers,
    ColumnType,
    CrossCatState,
    SufficientStats,
    ViewState,
)
from crosscat.validate import assert_valid_state, validate_state

__all__ = [
    # Types
    "ColumnHypers",
    "ColumnType",
    "CrossCatState",
    "PackedCrossCatState",
    "SufficientStats",
    "ViewState",
    # Model
    "initialize",
    "insert_rows",
    "log_joint",
    # Gibbs
    "gibbs_sweep",
    # Inference
    "column_typicality",
    "conditional_entropy",
    "credible_interval",
    "dependence_matrix",
    "dependence_probability",
    "impute_and_confidence",
    "joint_predictive_probability",
    "mutual_information",
    "predictive_anomalousness",
    "predictive_cdf",
    "predictive_probability",
    "predictive_sample",
    "row_similarity",
    "row_typicality",
    "sample_and_insert",
    # Constraints
    "check_column_dep_constraint",
    "ensure_col_dep_constraints",
    "ensure_row_dep_constraint",
    # Diagnostics
    "adjusted_rand_index",
    "collect_diagnostics",
    "column_partition_ari",
    "evaluate_imputation",
    "mean_test_log_likelihood",
    "random_holdout_mask",
    "row_partition_ari",
    # Data utilities
    "discretize_column",
    "gen_column_metadata",
    "guess_column_type",
    "guess_column_types",
    "read_csv",
    "write_csv",
    # Synthetic
    "add_missing_data",
    "generate_crosscat_data",
    # Validation
    "assert_valid_state",
    "validate_state",
    # Serialization
    "load_latest_checkpoint",
    "load_packed_state",
    "load_state",
    "save_checkpoint",
    "save_packed_state",
    "save_state",
    # Packed state
    "batch_packed_states",
    "multi_chain_packed_gibbs_sweep",
    "pack_state",
    "packed_gibbs_sweep",
    "packed_log_joint",
    "packed_transition_column_assignments",
    "packed_transition_column_hypers",
    "packed_transition_crp_alphas",
    "packed_transition_row_assignments",
    "select_best_chain",
    "unbatch_packed_states",
    "unpack_state",
    # Packed inference
    "packed_anomaly_score",
    "packed_dependence_matrix",
    "packed_dependence_probability",
    "packed_impute_and_confidence",
    "packed_mutual_information",
    "packed_predictive_cdf",
    "packed_predictive_probability",
    "packed_predictive_sample",
    "packed_row_similarity",
]
