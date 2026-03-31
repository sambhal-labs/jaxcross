# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX-CrossCat is a GPU-accelerated reimplementation of [probcomp/crosscat](https://github.com/probcomp/crosscat) using JAX. It implements a two-level Dirichlet Process mixture model: an outer DP partitions columns into "views", and an inner DP per view clusters rows. All parameters are collapsed out via conjugate Bayesian component models — only cluster assignments and hyperparameters are sampled.

## Commands

```bash
# Install with dev dependencies (using uv)
uv sync --extra dev

# Install with GPU support
uv sync --extra dev --extra gpu

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_synthetic_recovery.py

# Run a single test function
uv run pytest tests/test_new_features.py::test_function_name

# Exclude slow tests
uv run pytest -m "not slow"

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

## Architecture

The package is `crosscat/` with these core modules:

- **types.py** — Dataclasses for all state: `CrossCatState` (full model), `ViewState` (one column group with row clustering), `ColumnHypers`, `SufficientStats`, `ColumnType` enum (CONTINUOUS, CATEGORICAL, ORDINAL, BINARY, CYCLIC), and `LOG_EPS` numerical stability constant.

- **components.py** — Bayesian component models (`NormalGamma`, `DirichletCategorical`, `BetaBernoulli`, `OrderedLogistic`, `VonMises`). Each provides `sufficient_statistics()`, `log_marginal_likelihood()`, and `posterior_predictive_logp()`. All are conjugate except `OrderedLogistic` which uses grid integration over a latent location parameter.

- **model.py** — State initialization (`initialize()`), scoring (`log_joint()`), and row insertion (`insert_rows()`). Uses Chinese Restaurant Process sampling for cluster assignments and data-driven hyperparameter defaults.

- **gibbs.py** — Collapsed Gibbs MCMC kernels: `transition_row_assignments()`, `transition_column_assignments()`, `transition_column_hypers()`, `transition_crp_alphas()`, and `gibbs_sweep()` which runs a full iteration. **Note:** The unpacked gibbs.py path uses Python for-loops and is extremely slow — always prefer the packed path for inference.

- **inference.py** — Posterior predictive queries (unpacked path): `predictive_probability()`, `predictive_sample()`, `predictive_cdf()`, `mutual_information()`, `dependence_probability()`, `dependence_matrix()`, `impute_and_confidence()`, `predictive_anomalousness()`, `row_similarity()`, `row_typicality()`, `column_typicality()`, `sample_and_insert()`, `credible_interval()`, `conditional_entropy()`, `joint_predictive_probability()`.

- **packed/** — JIT-compatible packed state sub-package:
  - `state.py` — `PackedCrossCatState` dataclass, `pack_state()`, `unpack_state()`, `batch_packed_states()`, `unbatch_packed_states()`, `select_best_chain()`
  - `components.py` — unified scoring (log marginal, posterior predictive) via `jnp.where` type dispatch + batch-vectorized type-specialized scoring (`batch_bb_posterior_predictive_logp`, `batch_ng_posterior_predictive_logp`, `batch_dc_posterior_predictive_logp`) + ordered logistic grid integration (`_ol_log_marginal`, `_ol_posterior_predictive_logp`)
  - `suffstats.py` — vectorized sufficient statistics (matrix ops, batched scatter add/remove)
  - `kernels.py` — all Gibbs kernels (`packed_gibbs_sweep`, `packed_gibbs_step`, row/column assignments, hypers, CRP alphas, `packed_insert_rows`) via `vmap`/`lax.scan` with type-specialized fast paths. Sub-kernels have `@jax.jit` for independent compilation.
  - `aot_cache.py` — XLA persistent compilation cache (`enable_xla_cache()`, `compile_kernels()`, `clear_cache()`)

- **packed_inference.py** — Vectorized inference queries on packed state. Full parity with inference.py plus multi-chain and batch support:
  - **Single-state:** `packed_predictive_probability`, `packed_predictive_sample`, `packed_predictive_cdf`, `packed_anomaly_score`, `packed_impute_and_confidence`, `packed_credible_interval`, `packed_row_typicality`, `packed_column_typicality`, `packed_conditional_entropy`, `packed_joint_predictive_probability`, `packed_sample_and_insert`, `packed_classify_column`
  - **Batch (vectorized over rows via vmap — production-recommended):** `batch_anomaly_score`, `batch_row_typicality`, `batch_impute_column`, `batch_classify_column`, `batch_score_columns_binary`, `batch_row_similarity`, `batch_predictive_cdf`, `batch_credible_interval`
  - **Multi-state (already accept lists):** `packed_mutual_information`, `packed_dependence_matrix`, `packed_dependence_probability`, `packed_row_similarity`
  - **Multi-chain wrappers:** `multi_chain_predictive_probability`, `multi_chain_predictive_sample`, `multi_chain_anomaly_score`, `multi_chain_impute_and_confidence`, `multi_chain_predictive_cdf`

- **constraints.py** — Enforces column/row dependency constraints via packed Gibbs rejection sampling.
- **diagnostics.py** — Convergence metrics (Adjusted Rand Index, held-out likelihood, imputation evaluation).
- **serialization.py** — Save/load states and checkpoints in `.jxc` format (JSON metadata + NPZ arrays).
- **synthetic.py** — Synthetic data generation from known CrossCat generative model, missing data injection.
- **data_utils.py** — CSV I/O, column type detection, discretization.
- **validate.py** — State consistency checking.
- **../contrib/fingerprint.py** — Entity behavioral fingerprinting (LaborLens-specific, not part of core).

## Packed vs Unpacked Paths

**Always prefer the packed path.** The unpacked path (`gibbs.py`, `inference.py`) uses Python for-loops and is 10-100x slower. The packed path (`packed/kernels.py`, `packed_inference.py`) uses JAX JIT compilation with `lax.scan`/`vmap` and runs on GPU.

Typical workflow:
```python
state = initialize(key, data, column_types)
packed = pack_state(state)
packed = packed_gibbs_sweep(key, packed, data, n_sweeps=100)
state = unpack_state(packed, column_types, data=data)
```

For streaming/online inference:
```python
packed, data = packed_insert_rows(key, packed, data, new_rows)
```

**Batch queries (production-recommended for multi-row operations):**
```python
from crosscat import batch_anomaly_score, batch_row_typicality, batch_impute_column
scores = batch_anomaly_score(packed, data, jnp.arange(n_rows))       # all rows, one JIT call
typicality = batch_row_typicality([packed], jnp.arange(n_rows))      # structural typicality
values, confs = batch_impute_column(key, packed, data, col, row_ids) # impute column for rows
```
Always prefer `batch_*` functions over Python loops with single-row packed functions — they use `jax.vmap` for 10-100x speedup.

**Two sweep modes:**
- `packed_gibbs_sweep` — uses `lax.scan` for maximum throughput in production (multi-sweep, multi-chain). First compile is cached by XLA persistent cache (auto-enabled on import).
- `packed_gibbs_step` — calls `@jax.jit` sub-kernels independently (4 smaller compilations). Used by constraint enforcement and interactive/exploratory workflows.

**Compilation caching:** XLA persistent cache is auto-enabled when `crosscat.packed` is imported. Use `compile_kernels(packed, data)` to pre-compile all sub-kernels for a given shape.

## Key Patterns

- **JAX idioms**: Uses `jax.lax.scan` for sequential loops (view/row sweeps), `jax.vmap` for parallel operations (column scoring, cluster scoring, row clustering across views), `jax.jit` for compilation. All state is immutable — operations return new arrays.
- **Vectorized column scoring**: Row scoring in `_score_row_one_cluster` uses `jax.vmap(unified_posterior_predictive_logp)` over all columns simultaneously (not sequential `lax.scan`). This is the key optimization that gave 12x speedup in v0.9.0.
- **Type-specialized fast paths**: `_compute_dominant_type()` detects when all columns in a view share the same type (e.g., all BINARY for MNIST). When dominant, `_score_row_one_cluster_typed` bypasses the 5-way `jnp.where` dispatch and calls type-specific batch functions directly (`batch_bb_posterior_predictive_logp`, `batch_ng_posterior_predictive_logp`, `batch_dc_posterior_predictive_logp`, `batch_vm_posterior_predictive_logp`).
- **Batched suffstat updates**: `_add_row_to_suffstats` / `_remove_row_from_suffstats` use `.at[].add()` scatter operations over all columns at once instead of sequential `lax.scan`.
- **Numerical stability**: `LOG_EPS = 1e-30` constant in `types.py` used throughout for underflow protection. All files import from `crosscat.types`.
- **Deterministic RNG**: Always use `jax.random.key()` and `jax.random.split()` for reproducibility.
- **NaN transparency**: Missing data is represented as NaN and silently filtered during sufficient statistic computation.
- **Docstring cross-references**: Many functions include "Maps to original..." comments linking to the probcomp/crosscat equivalent.

## Testing

- **Do NOT run pytest locally** — tests require JAX JIT compilation which is slow even on GPU. Run on Kaggle P100 via `notebooks/run_tests.ipynb`.
- **CI (GitHub Actions)** runs lint + format + type check only (~1 min). No pytest in CI.
- **Kaggle setup**: Use `pip install -e . --no-deps` to preserve Kaggle's pre-installed JAX+CUDA stack. Do NOT use `uv sync --extra gpu` on Kaggle (causes ptxas version mismatch).
- **Test markers**: `@pytest.mark.slow` for GPU-heavy tests (30+ Gibbs sweeps). `@pytest.mark.xfail` for 2 known flaky tests (stochastic recovery).
- **Test suite**: 185+ fast tests (including 33 Hypothesis property tests), 31 slow tests.
- **Property tests**: `tests/test_property.py` uses Hypothesis to verify mathematical invariants (suffstat roundtrips for all 5 types, component scoring, type dispatch parity, NaN safety) across random inputs.

## Benchmarks

- **WDI macroeconomic benchmark** (`benchmarks/wdi_macroeconomic_benchmark.ipynb`): Real-world World Bank data (~200 countries × 30+ continuous indicators). Multi-chain inference with checkpointing, dependence discovery, country anomaly detection (`batch_row_typicality`), similarity clustering (`batch_row_similarity`), holdout imputation (`batch_impute_column`), mutual information. **This is the reference notebook for production API patterns.**
- **MNIST paper benchmark** (`benchmarks/mnist_paper_colab.ipynb`): Reproduces Section 3.2 of Mansinghka et al. (2016). 16×16 binary MNIST (257 cols), 10 chains × 100 sweeps on P100. Validates Z-matrix, pixel dependence map, inpainting (93% accuracy), and classification (79% accuracy).
- **Synthetic benchmark** (`benchmarks/paper_synthetic_benchmark.py`): Figure 7 recovery with known ground truth.
- **JIT benchmark** (`benchmarks/jit_benchmark.py`): Per-sweep timing comparison.
- Run notebooks on Kaggle (P100) for GPU-accelerated benchmarks.

## Git Workflow

- **Always use feature branches** — never commit directly to main.
- Branch naming: `feat/`, `fix/`, `perf/`, `chore/` prefixes.
- Create PRs via `gh pr create` and merge via `gh pr merge --merge`.
- GitHub Actions free-tier quota is limited — CI may fail when exhausted.

## Code Style

- Python 3.11+, ruff for linting (rules: E, F, I, W, UP, B, SIM), line length 99
- Type hints throughout
- Private functions prefixed with `_`

## Docs Index

IMPORTANT: Prefer retrieval-led reasoning — read the referenced doc before making changes to related code. Each doc is annotated with what it covers so you can find the right one quickly.

### Getting Started — `docs/getting-started/`
- `installation.md` — pip/uv install, CPU/CUDA/ROCm/Kaggle/Colab setup
- `quickstart.md` — 5-minute end-to-end example (load → initialize → pack → sweep → query)
- `concepts.md` — CrossCat model explanation (two-level DP, CRP, views, clusters, column types)

### Architecture — `docs/architecture/`
- `overview.md` — Module map, data flow, package structure
- `model.md` — The CrossCat generative model, component models math, collapsed inference
- `gibbs-kernels.md` — Row/column assignment transitions, hyper transitions, CRP alpha sampling
- `packed-state.md` — PackedCrossCatState design, padding strategy, pack/unpack roundtrip
- `jax-patterns.md` — JAX idioms used (vmap, lax.scan, jit, immutable state, RNG threading)
- `performance.md` — Optimization history, type-specialized fast paths, batched suffstats, numerical stability

### Feature Guides — `docs/guides/`
- `data-loading.md` — CSV I/O, column type auto-detection, ORDINAL/CYCLIC manual override, NaN handling
- `initialization.md` — Single/multi-chain init, initialization modes, CRP alpha, hyperparameter defaults
- `inference.md` — Packed vs unpacked paths, kernel selection, sweep configuration, convergence
- `gpu-packed.md` — Pack/unpack workflow, padding dimensions, batch queries for production, JIT timing
- `multi-chain.md` — Parallel chains, multi_chain_* wrappers, best chain selection
- `constraints.md` — Column/row dependency constraints, rejection sampling, diagnostics
- `serialization.md` — save_state/load_state, save_packed_state/load_packed_state, checkpointing
- `xla-cache.md` — XLA persistent compilation cache, enable_xla_cache(), compile_kernels()
- `missing-data.md` — NaN transparency, missing data patterns, imputation strategies
- `online-learning.md` — Row insertion (packed_insert_rows), streaming workflows
- `diagnostics.md` — ARI, log-joint tracking, holdout evaluation, convergence monitoring
- `dashboard.md` — Streamlit interactive dashboard

### Query Guides — `docs/guides/queries/`
- `predictive-probability.md` — Predictive probability, CDF, joint probability, batch_predictive_cdf, batch_credible_interval
- `sampling.md` — Conditional sampling, credible intervals, predictive_sample
- `anomaly-detection.md` — Anomaly scores, row/column typicality, batch_anomaly_score, batch_row_typicality
- `dependence.md` — Z-matrix (dependence_matrix), pairwise dependence probability
- `imputation.md` — Missing value imputation, confidence scores, batch_impute_column, holdout evaluation
- `mutual-information.md` — MI estimation, Linfoot correlation, conditional entropy
- `row-similarity.md` — Co-clustering probability, batch_row_similarity, similarity matrix

### API Reference — `docs/api/`
- `types.md` — ColumnType, CrossCatState, ViewState, PackedCrossCatState, ColumnHypers, SufficientStats
- `components.md` — NormalGamma, DirichletCategorical, BetaBernoulli, OrderedLogistic, VonMises
- `model.md` — initialize(), log_joint(), insert_rows()
- `gibbs.md` — gibbs_sweep() (unpacked path)
- `inference.md` — All unpacked inference functions (15 functions)
- `packed-state.md` — pack_state(), unpack_state(), batch_packed_states(), select_best_chain()
- `packed-components.md` — Unified scoring functions, type dispatch, batch type-specialized scoring
- `packed-kernels.md` — packed_gibbs_sweep/step, all packed transition kernels, packed_insert_rows
- `packed-inference.md` — All packed/batch/multi-chain inference functions (28+ functions including batch_*)
- `packed-suffstats.md` — Vectorized sufficient statistics, scatter add/remove
- `aot-cache.md` — enable_xla_cache(), compile_kernels(), clear_cache()
- `serialization.md` — save/load state, packed state, checkpoints
- `synthetic.md` — generate_crosscat_data(), add_missing_data()
- `constraints.md` — ensure_col_dep_constraints(), check_* functions
- `diagnostics.md` — ARI, collect_diagnostics(), evaluate_imputation(), packed_evaluate_imputation()
- `data-utils.md` — read_csv(), guess_column_types(), discretize_column()
- `validation.md` — validate_state(), assert_valid_state()

### Examples — `docs/examples/`
- `wdi-macroeconomics.md` — Real-world WDI benchmark: multi-chain with checkpointing, batch queries, dependence/anomaly/similarity/imputation analysis on ~200 countries × 30+ indicators
- `mnist.md` — MNIST paper reproduction: binary pixel data, inpainting, classification
- `csv-workflow.md` — 10-step CSV end-to-end: load → init → pack → sweep → query → save (uses batch functions)

## Common Workflows

### Adding a new component model
Read: docs/architecture/model.md, docs/api/components.md
1. `crosscat/components.py` — add class with `sufficient_statistics`, `log_marginal_likelihood`, `posterior_predictive_logp`, `sample_posterior_predictive`
2. `crosscat/packed/components.py` — add `_XX_log_marginal`, `_XX_posterior_predictive_logp`, `_XX_sample`; update 3 `unified_*` functions with new branch
3. `crosscat/packed/state.py` — add any new hyper fields to `PackedCrossCatState` + `_ARRAY_FIELDS`; update `pack_state`/`unpack_state`
4. `crosscat/packed/kernels.py` — thread new hypers through all scoring functions; add hyper transition in `packed_transition_column_hypers`; update `packed_insert_rows` constructor
5. `crosscat/model.py` — add initialization in `_default_hypers`
6. `crosscat/gibbs.py` — add hyper transition
7. `crosscat/packed_inference.py` — thread new hypers through inference calls
8. `tests/test_property.py` — add empty=0, finite, dispatch parity tests
9. `crosscat/serialization.py` — bump `_SCHEMA_VERSION`, add migration in `load_packed_state`

### Adding a new inference query
Read: docs/guides/inference.md, docs/api/inference.md, docs/api/packed-inference.md
1. `crosscat/inference.py` — add unpacked implementation
2. `crosscat/packed_inference.py` — add packed single-row implementation (`packed_*`)
3. `crosscat/packed_inference.py` — add batch implementation (`batch_*`) using `jax.vmap` over rows
4. `crosscat/__init__.py` — export all three (unpacked, packed, batch)
5. `tests/` — add unit test + parity test (see `test_packed_inference_parity.py`)
6. `docs/api/packed-inference.md` — document packed + batch functions
7. `docs/guides/queries/` — add or update the relevant query guide with batch examples

### Debugging numerical issues
Read: docs/architecture/performance.md
- Check `LOG_EPS` guards: grep for `jnp.log` without `jnp.maximum`
- Check NaN propagation: run with `jax.config.update("jax_debug_nans", True)`
- Verify suffstat roundtrip: `test_property.py::test_suffstat_add_remove_roundtrip_*`
- Compare packed vs unpacked: `test_packed_inference_parity.py`

## Common Pitfalls

- **JAX evaluates both branches of `jnp.where`** — padded values (+inf, NaN) flow through "unused" branches. Always clamp inputs to finite range before `linspace`/`vmap`.
- **`@jax.jit` on sub-functions is inlined** inside `lax.scan`/`vmap` — decorators only take effect when called from Python.
- **`PackedCrossCatState` constructor requires ALL fields** — when adding a new field, update EVERY place that constructs the state (`kernels.py` has 2+ sites, `packed_insert_rows` has its own constructor).
- **Serialization schema version** — bump `_SCHEMA_VERSION` and add migration in `load_packed_state` when adding new array fields.
- **Ordinal cutpoints are padded with +inf** — kernel must mask updates to only real cutpoints (determined from `hyper_n_cutpoints`, not `isfinite`). The `hyper_n_cutpoints` field stores the actual count per column for reliable roundtrips.
- **`max_cols_per_view` overflow** — if a Gibbs column transition assigns more columns to a view than `max_cols_per_view`, a runtime warning is emitted via `jax.debug.callback`. Set `max_cols_per_view=n_cols` (default) to avoid this.
- **Category values must be < `max_categories`** — pass `data=` to `pack_state()` for validation at pack time. At runtime, out-of-range values are silently clipped to `max_categories-1`.
