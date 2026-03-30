# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
