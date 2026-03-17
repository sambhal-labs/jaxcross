# Phase 3: Full JAX Vectorization — Design Spec

## Goal

Replace all Python for-loops in packed Gibbs kernels and inference queries with
`jax.lax.scan`, `jax.vmap`, and vectorized array operations, enabling full
`jax.jit` compilation and GPU acceleration via Google Colab.

## Scope

### In scope
- Vectorize `packed_transition_row_assignments` (lax.scan over rows, vmap over clusters/columns)
- Vectorize `packed_transition_column_hypers` (vmap over columns x grid x clusters)
- Vectorize `packed_transition_crp_alphas` (vmap over views x grid)
- JIT-compilable `packed_gibbs_sweep_v2` wrapping all three
- New `packed_inference.py` with vectorized inference queries on `PackedCrossCatState`
- `unified_sample_posterior_predictive` for JIT-compatible sampling dispatch
- Incremental suffstat helpers (remove-row / add-row)
- Correctness tests comparing packed v2 against unpacked ground truth
- Colab GPU benchmark notebook

### Out of scope (deferred)
- Column assignment kernel vectorization (outer DP — dynamic view creation/destruction)
- Unified API that auto-detects packed vs unpacked state (UX improvement, later)
- Multi-chain parallelism via pmap/vmap (Approach 3, follow-up)

## Architecture

### Backward Compatibility: Dual API (Option A)

New packed functions coexist alongside originals. Users explicitly choose:

```python
# Unpacked (existing, unchanged)
from crosscat.gibbs import gibbs_sweep
from crosscat.inference import predictive_sample

# Packed v2 (new, JIT-compilable)
from crosscat.packed_state import packed_gibbs_sweep_v2
from crosscat.packed_inference import packed_predictive_sample
```

Both APIs work on CPU and GPU. JAX handles device placement transparently.

### File Changes

| File | Action | Description |
|------|--------|-------------|
| `crosscat/packed_state.py` | Edit | Incremental suffstats, vmap scoring, lax.scan row kernel, vmap hypers/CRP, unified sampler, `packed_gibbs_sweep_v2` |
| `crosscat/packed_inference.py` | New | Vectorized inference on PackedCrossCatState |
| `crosscat/__init__.py` | Edit | Export new packed inference functions |
| `tests/test_packed_kernels_v2.py` | New | Correctness tests vs unpacked ground truth |
| `notebooks/gpu_benchmark.ipynb` | New | Colab GPU benchmarking notebook |

### Unchanged files
- `crosscat/gibbs.py` — unpacked reference implementation
- `crosscat/inference.py` — unpacked reference implementation
- `crosscat/components.py`, `crosscat/types.py` — no changes needed
- All existing tests

### Prerequisite: Add `max_cols_per_view` to static fields

The suffstat arrays have shape `(max_views, max_clusters, max_cols_per_view, ...)` but
`max_cols_per_view` is computed dynamically in `pack_state()` and not stored. This must
become a static field on `PackedCrossCatState` (added to `_STATIC_FIELDS`) so that
`lax.scan` and `vmap` can determine array shapes at trace time. All column loops inside
JIT-compiled functions iterate over the full `max_cols_per_view` dimension with masking
via `jnp.where(col_indices[li] >= 0, ...)` to skip padding columns.

---

## JAX Tracing Constraints

These constraints apply throughout the design and must be respected in implementation:

1. **`jnp.bincount` `length` must be a Python integer** (static, not traced). Always use
   `packed.max_clusters` or `packed.n_rows` (static fields), never a traced value.
2. **No `int()` or `.item()` inside JIT-compiled code.** Use traced array indexing
   (e.g., `arr.at[traced_idx].add(val)`) instead of `arr[int(idx)]`.
3. **`lax.scan` body must have fixed-shape carry.** All arrays in the carry tuple must
   have shapes known at trace time. Use padded arrays with masking.
4. **`n_sweeps` in `packed_gibbs_sweep_v2`** is a static shape parameter — changing it
   causes recompilation. This is the expected JAX behavior for `lax.scan` lengths.

---

## Detailed Design

### 1. Vectorized Row Assignment Kernel

The hottest path: O(n_rows x n_clusters x n_columns) per view per sweep.

#### Current bottlenecks
- `for v in range(n_views)` — sequential over views
- `for i in range(n_rows)` — sequential over rows (required for Gibbs)
- `for c in range(max_clusters)` inside `_score_row_all_clusters` — sequential scoring
- `for li in range(n_columns)` inside cluster scoring — sequential over columns

#### View loop strategy

Views are processed via `lax.fori_loop` (or `lax.scan`), NOT a Python for-loop,
so the entire row assignment kernel remains inside a single JIT compilation.
The padded array structure (all views share the same `max_cols_per_view` and
`max_clusters` dimensions) makes this safe. Inactive views (where
`view_mask[v] == False`) are skipped via `jnp.where` masking in the scan body.

#### New design

```python
def packed_transition_row_assignments_v2(rng_key, packed, data):
    # Process views via lax.fori_loop (max_views is small, typically 2-8)
    # Within each view:

    # Step 1: vmap over columns to score one row against one cluster
    def score_row_cluster(row_data, col_indices, type_ids, ss_slice, hypers):
        per_col_logp = vmap(unified_posterior_predictive_logp, ...)(...)
        return jnp.sum(jnp.where(valid_mask, per_col_logp, 0.0))

    # Step 2: vmap over clusters to score one row against ALL clusters
    score_all_clusters = vmap(score_row_cluster, in_axes=(..., 0, ...))

    # Step 3: lax.scan over rows (sequential — Gibbs correctness)
    def scan_body(carry, row_idx):
        assigns, ss, n_clust, rng = carry
        # Incremental: remove row_idx from current cluster suffstats
        # Score all clusters via score_all_clusters (vmapped)
        # Append new-cluster score (empty suffstats, CRP alpha prior)
        # Sample assignment from categorical
        # Incremental: add row_idx to chosen cluster suffstats
        return (new_assigns, new_ss, new_n_clust, new_rng), None

    (final_assigns, final_ss, ...), _ = lax.scan(
        scan_body, init_carry, jnp.arange(n_rows)
    )
```

#### Incremental sufficient statistics

Instead of full recompute after each row reassignment, maintain running suffstats:

```python
def _remove_row_from_suffstats(ss_counts, ss_sum_x, ss_sum_x_sq,
                                ss_cat_counts, ss_sum_sin, ss_sum_cos,
                                cluster_id, row_data, col_indices, col_type_ids):
    """Subtract one row's contribution from a cluster's suffstats.

    All updates are vectorized over columns with jnp.where on type_id.
    """
    # NaN handling: if row value is NaN for a column, that column's
    # contribution is zero — the row was never counted in those suffstats.
    # Guard: is_valid = ~jnp.isnan(x) & (col_idx >= 0)
    # All deltas are multiplied by is_valid before applying.

    # For continuous (type_id == 0):
    #   delta_count = is_valid.astype(int), delta_sum_x = x * is_valid,
    #   delta_sum_x_sq = x**2 * is_valid
    # For categorical/ordinal (type_id == 1 or 2):
    #   delta_count = is_valid.astype(int)
    #   Category index update uses .at[traced_idx].add(-1) with clamped index:
    #     cat_idx = jnp.clip(x.astype(jnp.int32), 0, max_categories - 1)
    #     ss_cat_counts = ss_cat_counts.at[cluster_id, col_li, cat_idx].add(-is_valid)
    #   (NOT int(x) — traced index via .at[] is required inside JIT)
    # For binary (type_id == 3):
    #   delta_count = is_valid, delta_sum_x = x * is_valid
    # For cyclic (type_id == 4):
    #   delta_count = is_valid, delta_sin = sin(x) * is_valid,
    #   delta_cos = cos(x) * is_valid

    # Apply deltas: ss_counts = ss_counts.at[cluster_id, :].add(-delta_count)
    # etc. for each suffstat array

def _add_row_to_suffstats(...):
    """Add one row's contribution to a cluster's suffstats.
    Same structure as _remove, with positive deltas. Same NaN guard."""
```

This reduces per-row work from O(n_rows x n_cols) to O(n_cols).

#### New cluster handling

- Slot `max_clusters - 1` is reserved as the "new cluster" proposal.
- Its suffstats are always zeros (prior predictive).
- If chosen, `n_clusters` increments (capped at `max_clusters - 1` to preserve the proposal slot).
- **Cluster budget exhaustion**: When `n_clusters >= max_clusters - 1`, the new-cluster
  option is excluded by setting its log probability to `-inf` in the categorical. The row
  is forced into an existing cluster. A test case covers this edge condition.
- Cluster compaction (removing empty clusters, remapping indices) happens once at the end
  of the full row scan, not per-row.

#### JIT-compatible cluster compaction

After the `lax.scan` completes, empty clusters are removed and indices remapped
using only traced array operations (no `jnp.unique` or Python loops):

```python
cluster_counts = jnp.bincount(assignments, length=max_clusters)
alive = cluster_counts > 0                    # (max_clusters,) bool
new_ids = jnp.cumsum(alive) - 1               # contiguous 0,1,2,...
remapped = new_ids[assignments]               # remap all assignments
n_clusters_final = jnp.sum(alive)
```

### 2. Vectorized Column Hypers Kernel

#### New design

```python
def packed_transition_column_hypers_v2(rng_key, packed, data):
    def process_one_column(key, col_idx, type_id, view_idx, local_idx, ...):
        # Gather suffstats: ss_arrays[view_idx, :, local_idx]
        # For each hyper to sample (e.g., s, mu, nu for continuous):
        #   vmap score_grid_point over grid → shape (n_grid,)
        #     where score_grid_point vmaps _ng_log_marginal over clusters → sum
        #   categorical sample from grid scores
        # jnp.where on type_id selects which hyper update to apply
        return updated_hyper_values

    # vmap over ALL columns — independent sampling
    new_hypers = vmap(process_one_column)(keys, arange(n_cols), type_ids, ...)
```

#### Grid scoring vectorization

```python
def _score_hyper_grid_continuous(grid_vals, ss_counts_col, ss_sum_x_col,
                                  ss_sum_x_sq_col, fixed_hypers):
    """Score a grid of hyper values across all clusters simultaneously."""
    # vmap over grid points:
    #   vmap over clusters: _ng_log_marginal(ss[c], grid_val)
    #   sum over clusters
    # Returns: shape (n_grid,)
    def score_one(grid_val):
        per_cluster = vmap(_ng_log_marginal, in_axes=(0,0,0,None,None,None,None))(
            ss_counts_col, ss_sum_x_col, ss_sum_x_sq_col, *fixed_hypers, grid_val
        )
        return jnp.sum(per_cluster)
    return vmap(score_one)(grid_vals)
```

### 3. Vectorized CRP Alpha Kernel

#### New design

```python
def packed_transition_crp_alphas_v2(rng_key, packed):
    def log_crp_score(assignments, alpha_val, n):
        counts = jnp.bincount(assignments, length=max_clusters).astype(jnp.float32)
        n_clusters = jnp.sum(counts > 0)
        valid_counts = jnp.where(counts > 0, counts, 1.0)
        return (n_clusters * jnp.log(alpha_val)
                + jnp.sum(jnp.where(counts > 0, gammaln(valid_counts), 0.0))
                - gammaln(n + alpha_val) + gammaln(alpha_val)
                - alpha_val)  # Exp(1) prior

    # Outer CRP: vmap over grid
    outer_scores = vmap(lambda a: log_crp_score(packed.column_assignments, a, n_cols))(alpha_grid)
    new_col_alpha = alpha_grid[jax.random.categorical(key0, outer_scores)]

    # Inner CRP: vmap over views x grid
    view_grid_scores = vmap(
        lambda assigns: vmap(lambda a: log_crp_score(assigns, a, n_rows))(alpha_grid)
    )(packed.view_row_assignments)  # (max_views, n_grid)

    new_view_alphas = vmap(
        lambda k, s: alpha_grid[jax.random.categorical(k, s)]
    )(view_keys, view_grid_scores)
```

Zero Python loops. Entire kernel is one JIT-compiled function.

### 4. Packed Gibbs Sweep v2

```python
@jax.jit
def packed_gibbs_sweep_v2(rng_key, packed, data, n_sweeps=1):
    def one_sweep(carry, _):
        packed_state, rng = carry
        k1, k2, k3, rng = jax.random.split(rng, 4)
        packed_state = packed_transition_row_assignments_v2(k1, packed_state, data)
        packed_state = packed_transition_column_hypers_v2(k2, packed_state, data)
        packed_state = packed_transition_crp_alphas_v2(k3, packed_state)
        return (packed_state, rng), None

    (result, _), _ = jax.lax.scan(one_sweep, (packed, rng_key), jnp.arange(n_sweeps))
    return result
```

Full sweep becomes a single `lax.scan` — one JIT compilation, zero Python overhead.

### 5. Packed Inference (packed_inference.py)

New file with vectorized inference functions operating on `PackedCrossCatState`.

#### unified_sample_posterior_predictive (in packed_state.py)

```python
def unified_sample_posterior_predictive(rng_key, type_id, count, sum_x, sum_x_sq,
                                         cat_counts, sum_sin, sum_cos, hypers):
    """Sample from posterior predictive for any column type via jnp.where."""
    # Compute sample for ALL types, select correct one
    cont_sample = _ng_sample(rng_key, count, sum_x, sum_x_sq, mu, r, s, nu)
    cat_sample = _dc_sample(rng_key, count, cat_counts, dir_alpha)
    binary_sample = _bb_sample(rng_key, count, sum_x, alpha, beta)
    ordinal_sample = _dc_sample(rng_key, count, cat_counts, 1.0)  # uses Dir-Cat with alpha=1.0
    cyclic_sample = _vm_sample(rng_key, count, sum_sin, sum_cos, kappa, vm_mu)

    return jnp.where(type_id == CONTINUOUS_ID, cont_sample,
           jnp.where(type_id == CATEGORICAL_ID, cat_sample,
           jnp.where(type_id == ORDINAL_ID, ordinal_sample,
           jnp.where(type_id == BINARY_ID, binary_sample, cyclic_sample))))
```

#### Inference functions

| Function | Vectorization strategy |
|----------|----------------------|
| `packed_predictive_probability` | vmap over clusters (logsumexp), sum over query cols |
| `packed_predictive_sample` | vmap over n_samples; within each: sample cluster then sample value |
| `packed_mutual_information` | vmap over states, vectorized cluster counting |
| `packed_row_similarity` | vmap over states x views, boolean comparison |
| `packed_impute_and_confidence` | Calls packed_predictive_sample + vectorized median/mode |
| `packed_anomaly_score` | vmap packed_predictive_probability over all columns |
| `packed_predictive_cdf` | Analytic (vmap over categories x clusters) or MC (via packed_predictive_sample) |

All functions accept `PackedCrossCatState` and return JAX arrays. No Python loops.

### 6. Testing Strategy

New file: `tests/test_packed_kernels_v2.py`

| Test | Method |
|------|--------|
| `test_scan_row_assignments_matches_original` | Same data+seed, compare assignments array |
| `test_vmap_column_hypers_matches_original` | Same data+seed, compare hyper values |
| `test_vmap_crp_alphas_matches_original` | Same data+seed, compare alpha values |
| `test_incremental_suffstats_correctness` | Remove+add row equals full recompute (`atol=1e-5` for float32) |
| `test_packed_predictive_probability_matches_original` | Compare log probs within tolerance |
| `test_packed_predictive_sample_distribution` | KS test: packed vs unpacked samples |
| `test_packed_mutual_information_matches_original` | Compare MI values within tolerance |
| `test_packed_row_similarity_matches_original` | Compare similarity scores |
| `test_full_packed_sweep_jit_compiles` | `jax.jit(packed_gibbs_sweep_v2)` runs without error |
| `test_packed_sweep_deterministic` | Same key produces same output |
| `test_mixed_column_types` | Correctness with continuous + categorical + binary + cyclic |
| `test_packed_anomaly_score_matches_original` | Compare anomaly scores within tolerance |
| `test_cluster_budget_exhaustion` | When `n_clusters >= max_clusters - 1`, new-cluster option excluded |

Correctness tolerance: `atol=1e-4` for log probabilities, `atol=1e-5` for suffstats, KS test p-value > 0.05 for samples.

### 7. Colab Notebook

File: `notebooks/gpu_benchmark.ipynb`

```
Cell 1: Setup
  - pip install jaxcross
  - Detect backend: jax.devices() → CPU/GPU/TPU
  - Print JAX version, device info

Cell 2: Synthetic data generation
  - Small: 100 rows x 10 cols (3 continuous, 3 categorical, 2 binary, 2 cyclic)
  - Medium: 1000 rows x 20 cols
  - Large: 5000 rows x 50 cols

Cell 3: Benchmark Gibbs sweep
  - Unpacked gibbs_sweep: time 1, 5, 10 sweeps
  - Packed v1 (Python loops): time 1, 5, 10 sweeps
  - Packed v2 (vectorized): time 1, 5, 10 sweeps (including JIT compile time)
  - Packed v2 post-JIT: time 1, 5, 10 sweeps (exclude first-call compile)
  - Table + bar chart

Cell 4: Benchmark inference queries
  - predictive_sample: 100, 1000, 10000 samples
  - predictive_probability: batch of 100 queries
  - Compare unpacked vs packed
  - Table + bar chart

Cell 5: Scaling analysis
  - Fix columns=20, vary rows: 100, 500, 1000, 2000, 5000
  - Fix rows=1000, vary columns: 5, 10, 20, 50
  - Line plots: time vs size

Cell 6: Memory usage
  - jax.device_memory_profile() before/after
  - Compare state sizes

Cell 7: Correctness verification
  - Run both unpacked and packed on same data
  - Compare suffstats, assignments, predictive distributions
  - Print pass/fail summary
```

## Success Criteria

1. `jax.jit(packed_gibbs_sweep_v2)` compiles and runs without error
2. All correctness tests pass (packed v2 matches unpacked reference)
3. No Python for-loops remain in any packed kernel or packed inference function
4. Colab notebook runs end-to-end on free tier (T4 GPU)
5. Measurable speedup on GPU (target: 10x+ for medium datasets)
6. CPU performance is not degraded for medium+ datasets (>= 100 rows); small datasets
   may see overhead from `lax.scan`/`vmap` dispatch vs Python loops — this is acceptable
7. All existing tests continue to pass (no regressions)
