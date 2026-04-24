# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-04-24 — [diff](https://github.com/sambhal-labs/jaxcross/compare/v1.0.0...v1.0.1)

Documentation overhaul. No code changes; all `1.0.0` public APIs remain source-compatible.

### Added
- **Complete API narrative coverage** — handwritten entries for all `packed_inference.py` functions
  (40 total): added missing 7 `batch_*` wrappers (`batch_predictive_probability`, `batch_predictive_sample`,
  `batch_conditional_entropy`, `batch_column_typicality`, `batch_dependence_probability`,
  `batch_joint_predictive_probability`, `batch_sample_and_insert`) and 4 `multi_chain_*` wrappers
  (`multi_chain_classify_column`, `multi_chain_credible_interval`,
  `multi_chain_joint_predictive_probability`, `multi_chain_sample_and_insert`) to
  `docs/api/packed-inference.md`. Added a "See Also" cross-reference block.
- **High-level I/O section** in `docs/api/data-utils.md` — `save_data` / `load_data`
  (Arrow IPC with embedded column type metadata) are now documented up front as the
  recommended entry points.
- **Algorithm pages** under `docs/architecture/algorithms/` — 5 new deep-dives on each Gibbs
  kernel with math, pseudocode, JAX pitfalls, and hyperparameter guidance:
  `row-gibbs.md`, `column-gibbs.md`, `hyper-transitions.md`, `crp-alpha.md`,
  `ordered-logistic-grid.md`. Linked from `docs/architecture/gibbs-kernels.md`
  and added to `mkdocs.yml` navigation.
- **probcomp/crosscat parity appendix** at `docs/reference/probcomp-parity.md` —
  feature-by-feature comparison table, API differences, and intentional divergences.
- **Newcomer on-ramp**: "Choose Your Path" triage block on `docs/index.md`, expanded
  "When to Use CrossCat (and when not to)" on `docs/getting-started/concepts.md`,
  "60-second" vs "10-minute" paths on `docs/getting-started/quickstart.md`, "What do
  you want to do?" decision table on `README.md`.
- **Production/GPU polish**:
    - Memory footprint table + low-VRAM guidance in `docs/guides/gpu-packed.md`.
    - `jax.pmap` pattern for multi-GPU in `docs/guides/multi-chain.md`.
    - "Profiling JAX kernels" section in `docs/architecture/performance.md`
      (`jax.profiler.trace`, `block_until_ready`, TensorBoard).
- **`LOGISTIC_INF` constant documentation** in `docs/api/types.md` plus `Related` cross-links.

### Changed
- `docs/api/index.md` — module map expanded to 17 modules; added "Three Tiers of
  Inference Functions" table (`packed_*` / `batch_*` / `multi_chain_*`).
- `README.md` — fixed count drift: packed_inference is 40 public functions (16 packed +
  15 batch + 9 multi-chain), 134 total exports across 18 modules.
- `CLAUDE.md` — clarified `packed_inference.py` scope (40) vs full packed-path exports
  (54) vs library-wide (134).

### Removed
- `notebooks/gpu_benchmark.ipynb` — stale 4 KB stub referencing "packed v2"
  terminology. Canonical benchmarks live in [benchmarks/](benchmarks/) — see
  `jit_benchmark.ipynb` for per-sweep timing.

## [1.0.0] - 2026-04-17 — [diff](https://github.com/sambhal-labs/jaxcross/compare/v0.12.0...v1.0.0)

First production release. Wraps up the four-phase hardening plan driven by
the production-readiness audit (see the Phase 1–4 pull requests below).
Bumps `Development Status` to `5 - Production/Stable`. No breaking API
changes — every prior `0.12.0` call remains source-compatible.

### Added
- **Phase 1 — correctness hardening** (#117)
  - `LOG_EPS` guard in `DirichletCategorical.posterior_predictive_logp`
    to match the existing `BetaBernoulli` / `OrderedLogistic` pattern
  - `jnp.maximum(alpha, LOG_EPS)` on every unguarded CRP `log(alpha)`
    site across `model.py`, `gibbs.py`, and `packed/kernels.py`
  - `_validate_category_range()` host-side check wired into
    `packed_insert_rows()` to replace silent clip-to-`max_categories-1`
    with a `ValueError`
  - `set_overflow_policy("warn"|"raise")` public API (+
    `JAXCROSS_OVERFLOW_POLICY` env var) for strict production handling
    of `max_clusters` / `max_cols_per_view` budget overflows
- **Phase 2 — conditional inference on the packed tier** (#118)
  - `condition_cols` / `condition_vals` support added to
    `packed_predictive_probability` / `..._sample` / `..._cdf`, plus the
    corresponding `batch_*` and `multi_chain_*` wrappers
  - `_cluster_weights_conditioned_packed()` helper matching the
    unpacked cluster-weight math (NaN skipping, cross-view independence)
  - `_resolve_cluster_weights()` single dispatch enforcing
    `row_id > condition_cols > marginal` precedence across all tiers
- **Phase 3 — test coverage + MI batching** (#119)
  - `batch_mutual_information(packed_states, column_types, col_pairs,
    …)` — Python-loops a `(n_pairs, 2)` array of column pairs, using
    `fold_in` for per-pair RNG independence; exported from the top-level
    `crosscat` namespace
  - `tests/test_packed_parity_extended.py` — 15 new packed/unpacked
    parity tests across `dependence_probability` / `dependence_matrix`
    / `row_similarity` / `row_typicality` / `column_typicality` /
    `predictive_probability` / `predictive_sample` / `predictive_cdf` /
    `predictive_anomalousness` / `mutual_information` /
    `conditional_entropy`
- **Phase 4 — release polish** (#120)
  - `TBLogger.log_convergence(traces, step, *, metric_name)` — logs
    `rhat/{metric}` + `ess/{metric}` scalars, silently skipping metrics
    when preconditions aren't met
  - Dedicated prose for `gelman_rubin_rhat` + `effective_sample_size`
    in `docs/api/diagnostics.md` including the "cannot be JIT-compiled"
    note for ESS
  - `docs/architecture/gibbs-kernels.md` — new section on the
    synchronous parallel row kernel's shared-baseline approximation and
    recommended 3–5 parallel / 1 sequential alternation cadence

### Changed
- `gibbs_sweep_early_stopping` relative-improvement denominator moved
  from `abs(prev_lj) + 1e-10` to `max(abs(prev_lj), 1.0)` so the
  patience mechanism still triggers when `prev_lj` is near zero (#120)
- `Development Status` classifier flipped to `5 - Production/Stable`

### Fixed
- Previously unguarded `jnp.log(alpha)` sites could NaN if any future
  hyperprior change produced a near-zero alpha; guards are defensive
  no-ops on every call path exercised by the current sampler (#117)

## [0.12.0] - 2026-04-09 — [diff](https://github.com/sambhal-labs/jaxcross/compare/v0.11.0...v0.12.0)

### Added
- **Production readiness (Phases A–F)** (#100, #101, #104, #105)
  - Gelman-Rubin R-hat and Effective Sample Size convergence diagnostics
  - Batch ordinal scoring (`batch_ol_posterior_predictive_logp`)
  - `growth_factor` and `initial_size` validation guards in `subsample_anneal`
  - Shape validation in `insert_rows()` for mismatched column counts
  - Centralized numerical constants (`LOG_EPS`, `LOGISTIC_INF`, `ORDINAL_N_GRID`) in `types.py`
- **18 missing inference functions** (#101, #104) — completes the feature matrix:
  - 7 batch functions: `batch_predictive_probability`, `batch_predictive_sample`,
    `batch_conditional_entropy`, `batch_column_typicality`, `batch_dependence_probability`,
    `batch_joint_predictive_probability`, `batch_sample_and_insert`
  - 9 multi-chain wrappers: `multi_chain_classify_column`, `multi_chain_credible_interval`,
    `multi_chain_joint_predictive_probability`, `multi_chain_sample_and_insert`,
    `multi_chain_predictive_probability`, `multi_chain_predictive_sample`,
    `multi_chain_anomaly_score`, `multi_chain_impute_and_confidence`,
    `multi_chain_predictive_cdf`
  - 2 packed queries: `packed_joint_predictive_probability`, `packed_sample_and_insert`
- **Arrow-first data I/O** (#101) — `save_data()`/`load_data()` convenience wrappers
  with column type metadata stored in Arrow schema
- **Atomic serialization writes** (#105) — temp file + rename + `.valid` marker
  prevents checkpoint corruption on crash
- **Test coverage** (#105) — 61 new tests (409 total), including batch coverage
  for 10 previously untested functions
- **Von Mises kappa scaling fix** (#103) — resultant vector scaled by kappa
  per Mardia & Jupp §5.3

### Changed
- Von Mises `_sample_von_mises_bf` kappa threshold unified to `1e-8`
- `load_packed_state` warns when `.valid` marker is missing
- Gold-standard workflow updated to use `multi_chain_packed_gibbs_sweep` (vmap)
  with convergence monitoring (Rhat/ESS) and checkpointing
- Documentation updated across all guides to use modern multi-chain pattern

## [0.11.0] - 2026-04-05 — [diff](https://github.com/sambhal-labs/jaxcross/compare/v0.10.1...v0.11.0)

### Added
- **Phase 1: Scaling quick wins** (#83)
  - JIT-compiled `packed_insert_rows` for online/streaming inference
  - `subsample_rows` parameter in `initialize()` — CRP-sample a subset for
    fast initialization on large datasets
  - `suggest_max_clusters(n_rows)` heuristic for automatic `max_clusters`
  - Frozen `InitResult` dataclass returned from `initialize()`
- **Phase 2: Memory optimization** (#84)
  - `read_csv_chunked()` — streaming CSV reader for large files with bounded
    memory, warnings on unparseable values and mismatched row lengths
  - `save_npy()` / `load_npy_mmap()` — uncompressed `.npy` save with true
    NumPy memory-mapped loading for multi-GB datasets (formerly `save_npz`
    / `load_npz_mmap`, which are now deprecated aliases)
  - `read_parquet()` / `write_parquet()` — Apache Parquet integration via
    pyarrow (optional dependency)
  - Memory-efficient column scoring in `_score_column_in_view`: replaced
    O(N*K) membership matrix with O(K)-output `jnp.bincount` operations —
    reduces per-column scoring memory from 128 MB to 4 MB at 1M rows (K=32)
- **Phase 3: Algorithmic scaling** (#84)
  - `packed_transition_row_assignments_minibatch` — mini-batch Gibbs kernel
    that samples `batch_size` rows per sweep (O(B) instead of O(N))
  - `subsample_anneal()` — gradually grows dataset during inference (init on
    small sample, double active rows per stage, insert + sweep)
  - `minibatch_gibbs_sweep()` — mini-batch row + full column/hyper/CRP sweeps
  - `gibbs_sweep_early_stopping()` — convergence-based early stopping with
    NaN/inf detection and previous-checkpoint relative improvement
  - New `crosscat.scaling` module for all scaling utilities
- 100K-row scaling benchmark (`benchmarks/scaling_100k_benchmark.py`)
- 33 new tests covering all Phase 2+3 features including structural
  invariants, CSV edge cases, mmap verification, and NaN bail-out

### Changed
- `read_csv()` refactored to use shared `_parse_rows()` with warnings on
  unparseable values and mismatched row lengths (parity with `read_csv_chunked`)
- `_parse_rows()` tracks exact unparseable count (not capped at 5)
- `save_npy` / `load_npy_mmap` warn when input file extension differs from
  `.npy` (suffix is silently rewritten for true mmap support)
- `save_npz` / `load_npz_mmap` are now deprecated aliases that emit
  `DeprecationWarning` and delegate to the new names

## [0.10.1] - 2026-03-30 — [diff](https://github.com/sambhal-labs/jaxcross/compare/v0.10.0...v0.10.1)

### Added
- **Von Mises batch fast path** (`batch_vm_posterior_predictive_logp`) for
  cyclic-dominant views — bypasses 5-way `jnp.where` dispatch, matching the
  existing fast paths for binary, continuous, and categorical types
- `hyper_n_cutpoints` field in `PackedCrossCatState` — stores actual cutpoint
  count per column for lossless ordinal pack/unpack roundtrips
- Optional `data` parameter in `pack_state()` for category value validation
  (raises `ValueError` if values >= `max_categories`)
- Runtime warning via `jax.debug.callback` when Gibbs column transitions
  cause a view to exceed `max_cols_per_view`
- Cyclic-only benchmark section in `benchmarks/jit_benchmark.py`
- 8 new tests: ordinal suffstat roundtrip, NaN-does-not-bias-ordinal,
  Von Mises dispatch parity, expanded diagnostics coverage
  (`random_holdout_mask`, `mean_test_log_likelihood`, `evaluate_imputation`)

### Changed
- Ordinal cutpoint sampling bounds widened from [-10, +10] to [-100, +100]
  to support data with larger ordinal ranges
- `unpack_state()` now uses `hyper_n_cutpoints` instead of `jnp.isfinite()`
  to determine actual cutpoint count (backward-compatible via schema migration)
- Serialization schema bumped to v3 (auto-migrates from v2 by inferring
  cutpoint counts from `jnp.isfinite`)
- `enable_xla_cache()` wrapped in try-except to prevent import failure
- `OrderedLogistic` refactored: extracted `_mu_grid_and_weights()` and
  `_averaged_probs()` helpers, eliminating ~24 lines of duplication

### Removed
- Dead code: `_compute_suffstats_for_column()` and `_component_log_marginal()`
  from `gibbs.py` (defined but never called)
- Dead code: `_shape_signature()` from `packed/aot_cache.py`
- Unused `view_counts` property from `CrossCatState`

### Fixed
- Stale grid size comment `(25,)` → `(64,)` for binary hyper grid in
  `packed/kernels.py`

## [0.10.0] - 2026-03-26 — [diff](https://github.com/sambhal-labs/jaxcross/compare/v0.9.0...v0.10.0)

### Added
- **True ordered logistic component model** with cumulative link function:
  P(Y=k | μ, cutpoints) = σ(c_k - μ) - σ(c_{k-1} - μ). Replaces the
  Dirichlet-Categorical stub that ignored ordinal structure. Uses 31-point
  grid integration over latent location μ with Normal prior.
- `hyper_cutpoints` field in `PackedCrossCatState` — per-column ordered
  thresholds, shape (n_cols, max_categories - 1), padded with +inf
- Cutpoint Gibbs transition kernel: sequential sampling via `lax.scan`
  with ordering constraint, plus mu prior variance grid sampling
- **Kernel splitting** for independent JIT compilation: `@jax.jit` on all
  4 packed Gibbs sub-kernels + `packed_gibbs_step()` for constraint loops
- **XLA persistent cache** auto-enabled on `crosscat.packed` import
- `compile_kernels()` warm-up function for pre-compiling sub-kernels
- **Property-based tests** via Hypothesis (28 tests): suffstat roundtrips,
  component scoring invariants, type dispatch parity, NaN transparency

### Changed
- `OrderedLogistic` class rewritten with cumulative link (non-conjugate,
  grid integration) instead of Dirichlet-Categorical alias
- Ordinal initialization: `cutpoints=linspace(-2, 2, K-1)`, `mu=0`, `s=4`
  instead of `dirichlet_alpha=1`
- Unified scoring functions (`unified_log_marginal`, etc.) now accept
  `cutpoints` parameter for ordinal support
- Ordinal columns use general scoring path (not categorical fast path)
- Constraint enforcement uses `packed_gibbs_step` (split kernels) instead
  of monolithic `packed_gibbs_sweep`

### Fixed
- AOT cache: `compile_and_cache()` now returns early on cache hit
- `enable_xla_cache()` made idempotent, respects existing JAX config
- Missing `hyper_vm_a` argument in `packed_inference.py` `_sample_one_column`

## [0.9.0] - 2026-03-24 — [diff](https://github.com/sambhal-labs/jaxcross/compare/v0.8.0...v0.9.0)

### Added
- **12x faster packed Gibbs kernels** via three optimizations:
  1. Vectorized column scanning: replaced `lax.scan` over columns with
     `jax.vmap` in row scoring and batched scatter in suffstat updates
  2. Type-specialized scoring: `batch_bb_posterior_predictive_logp`,
     `batch_ng_posterior_predictive_logp`, `batch_dc_posterior_predictive_logp`
     skip 5-way type dispatch for uniform-type views
  3. AOT compilation caching: `enable_xla_cache()` activates JAX's persistent
     compilation cache to skip 20+ min recompilation on subsequent runs
- MNIST Paper Benchmark notebook (`benchmarks/mnist_paper_colab.ipynb`)
  reproducing Section 3.2 of Mansinghka et al. (2016): Z-matrix (Fig 13b),
  pixel dependence map (Fig 13c), classification ROC (Fig 15), and pixel
  inpainting (Fig 14)
- Checkpoint/resume support in benchmark notebooks via `save_checkpoint()`
  and `load_latest_checkpoint()` — sessions resume from last saved state
- `_compute_dominant_type()` helper for detecting uniform-type views

### Changed
- `_score_row_one_cluster`: sequential `lax.scan` over columns replaced with
  parallel `jax.vmap(unified_posterior_predictive_logp)` over all columns
- `_remove_row_from_suffstats` / `_add_row_to_suffstats`: sequential
  `lax.scan` replaced with batched `.at[cluster_id, li_range].add()`
- `_score_row_all_clusters`: accepts optional `dominant_type` for
  type-specialized fast path

### Performance
- 1000 rows × 257 cols (16×16 MNIST): **238s → 20s per sweep** (12x speedup)
- 100 rows × 65 cols: **38s → 4.8s per sweep** (7.9x speedup)
- 50 rows × 11 cols: **25s → 4.5s per sweep** (5.5x speedup)
- JIT compilation time reduced from 20+ min to ~23s for 257 columns
- Full MNIST benchmark (10 chains × 100 sweeps × 257 cols) completes in
  ~3.5 hours on P100 (previously ~40+ hours)

### Fixed
- Mutual information estimation: removed self-normalized importance weighting
  that biased MI upward by ~5-15%
- BetaBernoulli `posterior_predictive_logp`: clamped `log(1-p)` to prevent
  `-inf` when p approaches 1.0
- NormalGamma `posterior_predictive_logp`: clamped `scale_sq` before `sqrt`
  to prevent NaN from negative floating-point artifacts
- Imputation confidence: replaced ad-hoc `exp(-IQR/std)` with principled
  inverse-variance measure `1/(1+std)`

## [0.8.0] - 2026-03-20 — [diff](https://github.com/sambhal-labs/jaxcross/compare/v0.7.0...v0.8.0)

### Changed
- VonMises sampling: rejection sampling (uniform proposal on [0, 2π), accept/reject
  against predictive logp) matching original `CyclicComponentModel::get_draw_constrained()`
- Mutual information estimation: Monte Carlo sampling matching original
  `inference_utils.estimate_MI_sample()` — draws (x,y) from joint predictive,
  computes MI = E[log p(x,y) - log p(x) - log p(y)] with importance weighting
- Von Mises kappa grid: `linspace(kappa_est, N*kappa_est, 31)` anchored at MLE
  estimate, matching original `construct_cyclic_specific_hyper_grid()`
- Benchmarks (synthetic + MNIST) switched to packed JIT kernels for ~10x speedup

### Fixed
- `packed/kernels.py`: missing `hyper_vm_a` argument in `_score_row_one_cluster`
  and `_score_column_in_view` calls for new-cluster scoring
- `tests/test_new_features.py`: VonMises tests missing `vm_a` hyperparameter

## [0.7.0] - 2026-03-20 — [diff](https://github.com/sambhal-labs/jaxcross/compare/v0.6.0...v0.7.0)

### Added
- Benchmark results infrastructure: JSON metrics + matplotlib PNG charts saved to `results/`
- Shared benchmark utilities (`benchmarks/utils.py`): result persistence, convergence plots, Z-matrix heatmaps
- Enhanced synthetic benchmark with per-sweep convergence plot, Z-matrix heatmap, and cluster recovery scatter
- MNIST digit clustering benchmark (`benchmarks/mnist_benchmark.py`): PCA-reduced MNIST with
  convergence tracking, Z-matrix, digit-cluster contingency table, and held-out evaluation
- New optional dependency group `benchmark` (matplotlib, scikit-learn)

## [0.6.0] - 2026-03-20 — [diff](https://github.com/sambhal-labs/jaxcross/compare/v0.5.0...v0.6.0)

### Added
- Synthetic recovery benchmark (`benchmarks/paper_synthetic_benchmark.py`) reproducing
  the paper's Figure 7 experiment — measures column ARI, row ARI, and Z-matrix recovery
- Held-out evaluation pipeline: `evaluate_imputation()` and `random_holdout_mask()` in
  `crosscat/diagnostics.py` for cell-level imputation accuracy (MAE, accuracy, log-lik)

### Documentation
- CRP alpha prior: documented Exp(1) vs flat prior divergence from probcomp/crosscat
- Components: added note clarifying collapsed-only inference (no Neal Algorithm 8)

## [0.5.0] - 2026-03-20 — [diff](https://github.com/sambhal-labs/jaxcross/compare/v0.4.0...v0.5.0)

### Changed
- **BREAKING**: Von Mises component model now uses separate `vm_a` (prior concentration)
  and `kappa` (likelihood concentration). Previously `kappa` was incorrectly used for both.
  Existing serialized states with Von Mises columns will need `vm_a` added.
- All hyperparameter grids switched to data-dependent ranges with N_GRID=31,
  matching the original probcomp/crosscat grid construction:
  - `s`: log-spaced [SSD/100, SSD] (was 7 fixed multipliers)
  - `mu`: linear [min(data), max(data)] (was 11 pts around mean)
  - `nu`: log-spaced [1, N] (was 7 fixed powers of 2)
  - `r`: log-spaced [1/N, N] (was 10 fixed values)
  - CRP alpha: log-spaced [1/N, N] with separate col/row grids (was 50 fixed pts)
  - `dirichlet_alpha`: log-spaced [1/N, N] (was 7 fixed values)
  - Von Mises `kappa`: log-spaced [0.01, N] (was 7 fixed values)
  - Binary `a`, `b`: 8×8 log-spaced [1/N, N] (was 5×5 fixed)

### Added
- Von Mises `vm_a` hyperparameter (prior concentration on mean direction)
  in `ColumnHypers`, `PackedCrossCatState`, and all scoring functions
- Grid-based Gibbs sampling for all 3 Von Mises hyperparameters:
  `kappa` (31 pts), `vm_a` (31 pts), `vm_mu` (31 pts)
- Separate inner/outer CRP alpha grids scaled to row/column count

### Fixed
- Von Mises log marginal likelihood formula: correctly uses `vm_a` for prior
  and `kappa` for likelihood (was conflating both as `kappa`)

## [0.4.0] - 2026-03-19 — [diff](https://github.com/sambhal-labs/jaxcross/compare/v0.3.0...v0.4.0)

### Added
- Interactive Streamlit dashboard for CrossCat analysis (`dashboard/`)

## [0.3.0] - 2026-03-18 — [diff](https://github.com/sambhal-labs/jaxcross/compare/v0.2.0...v0.3.0)

### Added
- JIT-compiled packed Gibbs kernels using `jax.lax.scan` and `jax.vmap`:
  - `packed_transition_row_assignments` — nested lax.scan over views and rows
  - `packed_transition_column_assignments` — lax.scan over columns with view compaction
  - `packed_transition_column_hypers` — vmap over columns with unified type dispatch
  - `packed_transition_crp_alphas` — vmap over grid + views
  - `packed_gibbs_sweep` — full sweep via lax.scan over n_sweeps
- Vectorized inference on packed state (`crosscat/packed_inference.py`):
  - `packed_predictive_probability`, `packed_predictive_sample`, `packed_predictive_cdf`
  - `packed_mutual_information`, `packed_row_similarity`
  - `packed_impute_and_confidence`, `packed_anomaly_score`
- GPU benchmark notebook (`notebooks/gpu_benchmark.ipynb`)
- Integration test suite:
  - `tests/test_integration_recovery.py`: 13 tests (cyclic, mixed-type, missing data)
  - `tests/test_integration_queries.py`: 17 tests (anomaly, MI, constraints, similarity)
- CI: coverage tracking (60% min), mypy, slow test gate, release workflow, pre-commit hooks

### Changed
- **BREAKING**: Split `crosscat/packed_state.py` into `crosscat/packed/` sub-package:
  - `packed/state.py` — PackedCrossCatState, pack/unpack
  - `packed/components.py` — conjugate scoring (log marginal, posterior predictive)
  - `packed/suffstats.py` — sufficient statistics computation
  - `packed/kernels.py` — all Gibbs kernels
- **BREAKING**: Removed `_v2` suffix from all packed kernel functions
  (e.g., `packed_gibbs_sweep_v2` is now `packed_gibbs_sweep`)
- `crosscat/packed_state.py` is now a deprecation shim — import from `crosscat.packed` instead
- `pyproject.toml`: extended ruff rules (B, SIM), added mypy config, coverage deps

### Removed
- v1 packed kernels (Python for-loop implementations) — replaced by JIT-compiled versions

## [0.2.0] - 2026-03-17 — [diff](https://github.com/sambhal-labs/jaxcross/compare/v0.1.0...v0.2.0)

### Added
- Packed state representation for JIT-compatible inference
  - `PackedCrossCatState` dataclass registered as JAX pytree
  - `pack_state()` / `unpack_state()` conversion functions
  - Vectorized sufficient statistics via matrix operations
  - Unified scoring functions for JIT-compatible column type dispatch
- Test fixtures for cyclic, mixed-type, and missing data scenarios
- Multi-chain inference helper with diagnostics collection
- Performance benchmark (`benchmarks/jit_benchmark.py`)

## [0.1.0] - 2025-01-15

### Added
- Initial release with full CrossCat feature parity
- Five conjugate component models: NormalGamma, DirichletCategorical, BetaBernoulli, OrderedLogistic, VonMises
- Collapsed Gibbs MCMC kernels: row assignments, column assignments (Gibbs + MH), hyperparameters, CRP alphas
- Posterior predictive queries: probability, sampling, CDF, joint probability
- Analysis utilities: mutual information, row/column typicality, anomaly detection, imputation
- Dependency constraint enforcement (column and row)
- Convergence diagnostics (ARI, log_joint tracking)
- Synthetic data generation and missing data injection
- State validation and data utilities
- CSV I/O and column type detection
