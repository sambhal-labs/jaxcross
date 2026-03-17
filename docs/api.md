# API Reference

Complete reference for the jax-crosscat public API.

## Core Types

### `ColumnType`

Enum specifying column data types.

| Value | Description | Component Model |
|-------|-------------|-----------------|
| `CONTINUOUS` | Real-valued data | NormalGamma (Normal-Inverse-Gamma) |
| `CATEGORICAL` | Unordered integer labels | DirichletCategorical |
| `BINARY` | 0 or 1 | BetaBernoulli |
| `ORDINAL` | Ordered integer levels | OrderedLogistic (Dirichlet-Multinomial) |
| `CYCLIC` | Angles in [0, 2*pi) | VonMises |

### `CrossCatState`

Full model state containing column partition, row clusterings, hyperparameters, and sufficient statistics.

| Field | Type | Description |
|-------|------|-------------|
| `column_assignments` | `Array (n_cols,)` | Column-to-view mapping |
| `column_crp_alpha` | `Array (scalar)` | Outer DP concentration |
| `column_hypers` | `list[ColumnHypers]` | Per-column hyperparameters |
| `column_types` | `list[ColumnType]` | Per-column type |
| `views` | `list[ViewState]` | View states |
| `n_rows` | `int` | Number of data rows |
| `n_cols` | `int` | Number of data columns |
| `n_views` | `int` (property) | Number of active views |

### `ViewState`

State for a single view (column group).

| Field | Type | Description |
|-------|------|-------------|
| `column_indices` | `Array (n_cols_in_view,)` | Which columns belong to this view |
| `row_assignments` | `Array (n_rows,)` | Row-to-cluster mapping |
| `row_crp_alpha` | `Array (scalar)` | Inner DP concentration |
| `suffstats` | `list[list[SufficientStats]]` | `suffstats[cluster][col_in_view]` |

---

## Model (`crosscat.model`)

### `initialize(rng_key, data, column_types, *, n_chains=1, column_crp_alpha=1.0, row_crp_alpha=1.0, initialization="from_the_prior")`

Create initial CrossCat state(s).

| Parameter | Type | Description |
|-----------|------|-------------|
| `rng_key` | `Array` | JAX PRNG key |
| `data` | `Array (n_rows, n_cols)` | Observation matrix |
| `column_types` | `list[ColumnType]` | Type per column |
| `n_chains` | `int` | Number of independent initializations |
| `initialization` | `str` | `"from_the_prior"`, `"together"`, or `"apart"` |

**Returns**: `CrossCatState` if `n_chains=1`, else `list[CrossCatState]`.

### `log_joint(state, data)`

Compute joint log probability of state and data.

**Returns**: Scalar log probability.

### `insert_rows(rng_key, state, data, new_rows)`

Insert new rows via CRP predictive (no re-inference on existing rows).

**Returns**: `(updated_state, updated_data)`.

---

## Gibbs Sampling (`crosscat.gibbs`)

### `gibbs_sweep(rng_key, state, data, *, n_sweeps=1, kernels=("row_assignments", "column_assignments", "column_hypers", "crp_alphas"))`

Run full Gibbs sweeps combining all transition kernels.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_sweeps` | `int` | Number of full iterations |
| `kernels` | `tuple[str]` | Which kernels to include per sweep |

**Available kernels**: `"row_assignments"`, `"column_assignments"`, `"column_assignments_mh"`, `"column_hypers"`, `"crp_alphas"`

**Returns**: Updated `CrossCatState`.

---

## Inference Queries (`crosscat.inference`)

### `predictive_probability(state, data, query_cols, query_vals, *, condition_cols=None, condition_vals=None, row_id=None)`

Conditional predictive probability: p(query | conditions, state).

**Returns**: Scalar log probability.

### `predictive_sample(rng_key, state, data, query_cols, *, condition_cols=None, condition_vals=None, n_samples=1000, row_id=None)`

Draw samples from the posterior predictive.

**Returns**: `Array (n_samples, len(query_cols))`.

### `predictive_cdf(rng_key, state, data, query_col, query_val, *, condition_cols=None, condition_vals=None, row_id=None, n_samples=10000)`

Posterior predictive CDF: P(X <= value). Analytic for discrete types, MC for continuous/cyclic.

**Returns**: Scalar in [0, 1].

### `mutual_information(states, col_i, col_j, *, n_samples=1000)`

Estimate mutual information between two columns, averaged over posterior samples.

**Returns**: `(mi, linfoot_correlation)`.

### `row_similarity(states, row_a, row_b, *, target_columns=None)`

Probability that two rows are in the same cluster, averaged over views and posterior samples.

**Returns**: Scalar in [0, 1].

### `row_typicality(states, row_id)`

Structural typicality score for a row (low = anomalous).

**Returns**: Scalar in [0, 1].

### `column_typicality(states, col_id)`

Structural typicality score for a column (consistency of view assignment across samples).

**Returns**: Scalar in [0, 1].

### `predictive_anomalousness(rng_key, state, data, query_row, *, n_samples=1000)`

Predictive anomaly score for a row (high = anomalous).

**Returns**: Scalar in [0, 1].

### `impute_and_confidence(rng_key, state, data, query_col, *, condition_cols=None, condition_vals=None, row_id=None, n_samples=1000)`

Impute a missing value with confidence. Continuous: median + IQR-based confidence. Discrete: mode + mode frequency.

**Returns**: `(point_estimate, confidence_score)`.

### `sample_and_insert(rng_key, state, data, partial_row)`

Fill NaN entries via predictive sampling, then insert the completed row.

**Returns**: `(updated_state, updated_data, completed_row)`.

### `conditional_entropy(rng_key, states, data, target_col, given_cols, *, n_samples=500)`

Estimate H(target | given) via Monte Carlo.

**Returns**: Scalar conditional entropy (nats).

---

## Constraints (`crosscat.constraints`)

### `check_column_dep_constraint(state, col_a, col_b, dependent)`

Check if two columns satisfy a dependency constraint.

### `ensure_col_dep_constraints(rng_key, state, data, constraints, *, max_rejections=100, n_sweeps_per_attempt=5)`

Find a state satisfying all column constraints via rejection sampling.

| Parameter | Type | Description |
|-----------|------|-------------|
| `constraints` | `list[tuple[int, int, bool]]` | `(col_a, col_b, dependent)` tuples |

**Returns**: `CrossCatState | None` (None if max_rejections exceeded).

### `ensure_row_dep_constraint(rng_key, state, data, row_a, row_b, dependent, *, view_idx=None, max_iterations=100)`

Find a state where two rows are in the same/different cluster.

**Returns**: `CrossCatState | None`.

---

## Diagnostics (`crosscat.diagnostics`)

### `adjusted_rand_index(assignments_true, assignments_pred)`

ARI between two partitions. 1 = perfect, 0 = random, <0 = anti-correlated.

### `column_partition_ari(state, true_assignments)`

ARI of inferred column partition vs ground truth.

### `row_partition_ari(state, view_idx, true_assignments)`

ARI of row partition in a specific view.

### `collect_diagnostics(state, data)`

Per-sweep diagnostic metrics: log_joint, n_views, CRP alphas, cluster counts.

### `mean_test_log_likelihood(state, data, test_rows)`

Held-out log-likelihood on specified test rows.

---

## Packed State (`crosscat.packed_state`)

### `pack_state(state, *, max_views=16, max_clusters=32, max_categories=16)`

Convert `CrossCatState` to JIT-compatible `PackedCrossCatState` with padded arrays.

### `unpack_state(packed, column_types)`

Convert back to `CrossCatState`.

### `packed_gibbs_sweep(rng_key, packed, data, *, n_sweeps=1, kernels=("row_assignments", "column_hypers", "crp_alphas"))`

Run Gibbs sweeps on packed state.

---

## Synthetic Data (`crosscat.synthetic`)

### `generate_crosscat_data(rng_key, n_rows, column_types, *, n_views=2, n_clusters=2, cluster_separation=5.0)`

Generate data from a known CrossCat generative model. Returns dict with `data`, `column_types`, `true_column_assignments`, `true_row_assignments`.

### `add_missing_data(rng_key, data, missing_fraction=0.1)`

Inject random NaN values into data.
