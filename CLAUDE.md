# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Instruction

**Prefer retrieval-led reasoning over pre-training-led reasoning for ALL jaxcross tasks.**

jaxcross is a newly created library that may not be in your training data. Do NOT guess APIs, function signatures, or patterns from pre-training. Instead:

1. **Read the referenced doc BEFORE writing or modifying code** — the Docs Index below tells you exactly where to look
2. **Verify function signatures** by reading the actual source in `crosscat/`
3. **Follow existing patterns** — read neighboring code before adding new code
4. When uncertain about an API, read the source file directly rather than guessing

Design for file retrieval: use the Docs Index and Source Code Index to locate specific files rather than relying on memory. This approach reduces 40KB+ of documentation to targeted lookups while maintaining accuracy.

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

- **inference.py** — 15 posterior predictive queries (unpacked path): `predictive_probability()`, `predictive_sample()`, `predictive_cdf()`, `mutual_information()`, `dependence_probability()`, `dependence_matrix()`, `impute_and_confidence()`, `predictive_anomalousness()`, `row_similarity()`, `row_typicality()`, `column_typicality()`, `sample_and_insert()`, `credible_interval()`, `conditional_entropy()`, `joint_predictive_probability()`.

- **packed/** — JIT-compatible packed state sub-package:
  - `state.py` — `PackedCrossCatState` dataclass, `pack_state()`, `unpack_state()`, `batch_packed_states()`, `unbatch_packed_states()`, `select_best_chain()`
  - `components.py` — unified scoring (log marginal, posterior predictive) via `jnp.where` type dispatch + batch-vectorized type-specialized scoring (`batch_bb_posterior_predictive_logp`, `batch_ng_posterior_predictive_logp`, `batch_dc_posterior_predictive_logp`) + ordered logistic grid integration (`_ol_log_marginal`, `_ol_posterior_predictive_logp`)
  - `suffstats.py` — vectorized sufficient statistics (matrix ops, batched scatter add/remove)
  - `kernels.py` — all Gibbs kernels (`packed_gibbs_sweep`, `packed_gibbs_step`, row/column assignments, hypers, CRP alphas, `packed_insert_rows`) via `vmap`/`lax.scan` with type-specialized fast paths. Sub-kernels have `@jax.jit` for independent compilation.
  - `aot_cache.py` — XLA persistent compilation cache (`enable_xla_cache()`, `compile_kernels()`, `clear_cache()`)

- **packed_inference.py** — Vectorized inference queries on packed state. Full parity with inference.py plus batch and multi-chain support (37 public functions):
  - **Single-state packed_ (12):** `packed_classify_column`, `packed_predictive_probability`, `packed_predictive_sample`, `packed_predictive_cdf`, `packed_anomaly_score`, `packed_impute_and_confidence`, `packed_credible_interval`, `packed_row_typicality`, `packed_column_typicality`, `packed_conditional_entropy`, `packed_joint_predictive_probability`, `packed_sample_and_insert`
  - **Multi-state packed_ (4, accept lists):** `packed_mutual_information`, `packed_dependence_matrix`, `packed_dependence_probability`, `packed_row_similarity`
  - **Batch (13):** `batch_anomaly_score`, `batch_impute_column`, `batch_row_typicality`, `batch_credible_interval`, `batch_predictive_cdf`, `batch_row_similarity`, `batch_classify_column`, `batch_score_columns_binary`, `batch_predictive_probability`, `batch_predictive_sample`, `batch_conditional_entropy`, `batch_column_typicality`, `batch_dependence_probability`
  - **Multi-chain wrappers (8):** `multi_chain_predictive_probability`, `multi_chain_predictive_sample`, `multi_chain_anomaly_score`, `multi_chain_impute_and_confidence`, `multi_chain_predictive_cdf`, `multi_chain_classify_column`, `multi_chain_credible_interval`, `multi_chain_joint_predictive_probability`

- **constraints.py** — Enforces column/row dependency constraints via packed Gibbs rejection sampling.
- **diagnostics.py** — Convergence metrics (Adjusted Rand Index, held-out likelihood, imputation evaluation, Gelman-Rubin R-hat, Effective Sample Size).
- **serialization.py** — Save/load states and checkpoints in `.jxc` format (JSON metadata + NPZ arrays).
- **synthetic.py** — Synthetic data generation from known CrossCat generative model, missing data injection.
- **data_utils.py** — Data I/O (Arrow IPC preferred, CSV, Parquet, NPY), column type detection, discretization. Use `save_data()`/`load_data()` for Arrow-first workflows with column type metadata.
- **scaling.py** — Large-dataset workflows: `subsample_anneal()`, `minibatch_gibbs_sweep()`, `parallel_gibbs_sweep()`, `gibbs_sweep_early_stopping()`. Combines subsample initialization, batch insertion, and mini-batch Gibbs sweeps for 10K+ row datasets.
- **tb_logger.py** — TensorBoard logging via `tensorboardX`. `TBLogger` context manager logs per-sweep diagnostics (scalars, histograms). Designed to consume the dict returned by `diagnostics.collect_diagnostics()`. Requires optional `tensorboardX` dependency.
- **validate.py** — State consistency checking.
- **../contrib/fingerprint.py** — Entity behavioral fingerprinting (LaborLens-specific, not part of core).
- **../paper/** — LaTeX paper sources (`main.tex`, `references.bib`, figures).
- **../notebooks/** — `run_tests.ipynb` (Kaggle test runner), `intro_tutorial.ipynb`, `gpu_benchmark.ipynb`.

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

- **Do NOT run pytest locally** — tests require JAX JIT compilation which is slow even on GPU. Run on Kaggle via `notebooks/run_tests.ipynb` (2xT4 with pmap preferred over single P100).
- **CI (GitHub Actions)** runs lint + format + type check + CPU-safe tests (`pytest -m cpu --timeout=120`). No GPU tests in CI.
- **Kaggle setup**: Use `pip install -e . --no-deps` to preserve Kaggle's pre-installed JAX+CUDA stack. Do NOT use `uv sync --extra gpu` on Kaggle (causes ptxas version mismatch).
- **Test markers**: `@pytest.mark.slow` for GPU-heavy tests (30+ Gibbs sweeps). `@pytest.mark.xfail` for 3 known flaky tests (stochastic recovery).
- **Test suite**: 279 fast tests (including 34 Hypothesis property tests), 69 slow tests (348 total).
- **Property tests**: `tests/test_property.py` uses Hypothesis to verify mathematical invariants (suffstat roundtrips for all 5 types, component scoring, type dispatch parity, NaN safety) across random inputs.

## Benchmarks

- **MNIST benchmark** (`benchmarks/mnist_benchmark.ipynb`): Binary MNIST (257 cols), multi-chain sweeps. Validates Z-matrix, pixel dependence map, inpainting, and classification.
- **MNIST PCA benchmark** (`benchmarks/mnist_pca_benchmark.ipynb`): PCA-reduced MNIST variant.
- **Synthetic benchmark** (`benchmarks/paper_synthetic_benchmark.ipynb`): Figure 7 recovery with known ground truth.
- **JIT benchmark** (`benchmarks/jit_benchmark.ipynb`): Per-sweep timing comparison.
- **Scalability benchmarks**: `scalability_benchmark.ipynb`, `scaling_10k_benchmark.ipynb`, `scaling_100k_benchmark.ipynb`, `scaling_1m_benchmark.ipynb`.
- Run notebooks on Kaggle (2xT4 with pmap preferred) for GPU-accelerated benchmarks.

## Git Workflow

- **Always use feature branches** — never commit directly to main.
- Branch naming: `feat/`, `fix/`, `perf/`, `chore/` prefixes.
- Create PRs via `gh pr create` and merge via `gh pr merge --merge`.
- GitHub Actions free-tier quota is limited — CI may fail when exhausted.

## Code Style

- Python 3.11+, ruff for linting (rules: E, F, I, W, UP, B, SIM), line length 99
- Type hints throughout
- Private functions prefixed with `_`

## Source Code Index

IMPORTANT: When modifying a module, read the source file first. Do not rely on the architecture summary above — it is compressed and may lag behind the actual code.

|root: ./crosscat
|.:{__init__.py,types.py,components.py,model.py,gibbs.py,inference.py,packed_inference.py,constraints.py,diagnostics.py,serialization.py,synthetic.py,data_utils.py,scaling.py,tb_logger.py,validate.py}
|packed:{__init__.py,state.py,components.py,suffstats.py,kernels.py,aot_cache.py}

## Docs Index

IMPORTANT: Prefer retrieval-led reasoning — read the referenced doc BEFORE making changes to related code. This is not optional — reading the doc first prevents generating code with wrong signatures or patterns.

|root: ./docs
|.:{index.md,faq.md,glossary.md,contributing.md,roadmap.md,changelog.md}
|getting-started:{installation.md,quickstart.md,concepts.md}
|architecture:{overview.md,model.md,gibbs-kernels.md,packed-state.md,jax-patterns.md,performance.md}
|guides:{index.md,data-loading.md,initialization.md,inference.md,gpu-packed.md,multi-chain.md,constraints.md,serialization.md,xla-cache.md,missing-data.md,online-learning.md,diagnostics.md,dashboard.md,scaling.md,tb-logger.md,tips-and-tricks.md}
|guides/queries:{predictive-probability.md,sampling.md,anomaly-detection.md,dependence.md,imputation.md,mutual-information.md,row-similarity.md}
|api:{index.md,types.md,components.md,model.md,gibbs.md,inference.md,packed-state.md,packed-components.md,packed-kernels.md,packed-inference.md,packed-suffstats.md,aot-cache.md,serialization.md,synthetic.md,constraints.md,diagnostics.md,data-utils.md,scaling.md,tb-logger.md,validation.md}
|use-cases:{customer-segmentation.md,anomaly-detection.md,missing-data.md,scientific-exploration.md}
|examples:{csv-workflow.md,mnist.md,wdi-macroeconomics.md}

## Benchmarks Index

IMPORTANT: Read the WDI benchmark notebook for the latest, fastest code patterns. It is the gold-standard reference for production workflows.

|root: ./benchmarks
|.:{jit_benchmark.ipynb,paper_synthetic_benchmark.ipynb,mnist_benchmark.ipynb,mnist_pca_benchmark.ipynb,scalability_benchmark.ipynb,scaling_10k_benchmark.ipynb,scaling_100k_benchmark.ipynb,scaling_1m_benchmark.ipynb,wdi_macroeconomic_benchmark.ipynb,utils.py}

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
Read: docs/guides/inference.md, docs/api/inference.md
1. `crosscat/inference.py` — add unpacked implementation
2. `crosscat/packed_inference.py` — add packed implementation
3. `crosscat/__init__.py` — export both
4. `tests/` — add unit test + parity test (see `test_packed_inference_parity.py`)

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
- **`initialize()` returns `InitResult`, not a bare state** — access `.state` to get the `CrossCatState`. When `n_chains > 1`, `.state` is a list.
- **ORDINAL and CYCLIC are never auto-detected** — `guess_column_types()` only detects CONTINUOUS, CATEGORICAL, BINARY. Always set ORDINAL and CYCLIC manually.
- **Not on PyPI** — install from source via `uv sync` or `pip install -e .`, not `pip install jax-crosscat`.

## Gold-Standard Workflow

Read the WDI benchmark notebook (`benchmarks/wdi_macroeconomic_benchmark.ipynb`) for the latest, fastest patterns. Every new feature or use-case should follow this structure:

```python
import jax
import jax.numpy as jnp
import numpy as np
from crosscat import initialize, log_joint
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat.types import ColumnType

# 1. Data loading
data = jnp.array(raw_data, dtype=jnp.float32)
col_types = [ColumnType.CONTINUOUS, ColumnType.CATEGORICAL, ...]

# 2. Multi-chain initialization
key = jax.random.key(42)
result = initialize(key, data, col_types, n_chains=4)
packed_states = [pack_state(s, max_views=16, max_clusters=32) for s in result.state]

# 3. Packed Gibbs inference (GPU)
final_packed = []
for i, packed in enumerate(packed_states):
    k = jax.random.fold_in(key, i + 100)
    packed = packed_gibbs_sweep(k, packed, data, n_sweeps=200)
    final_packed.append(packed)

# 4. Select best chain
final_states = [unpack_state(p, col_types, data=data) for p in final_packed]
scores = [float(log_joint(s, data)) for s in final_states]
best_idx = int(np.argmax(scores))

# 5. Inference queries
from crosscat import packed_dependence_matrix, batch_anomaly_score
z_matrix = packed_dependence_matrix(final_packed)
anomaly_scores = batch_anomaly_score(final_packed[best_idx], data, jnp.arange(data.shape[0]))
```

## API Quick Reference

### Anomaly Detection
```python
from crosscat import (
    predictive_anomalousness,    # Single row, unpacked
    batch_anomaly_score,         # All rows, packed (PREFERRED)
    row_typicality,              # Structural anomaly (unpacked, multi-state)
    batch_row_typicality,        # Structural anomaly, batch (packed)
    multi_chain_anomaly_score,   # Multi-chain ensemble
    column_typicality,           # Column-level anomaly (unpacked)
    packed_column_typicality,    # Column-level, packed
)
```

### Imputation & Missing Data
```python
from crosscat import (
    impute_and_confidence,           # Single cell (unpacked)
    batch_impute_column,             # Batch imputation (packed, PREFERRED)
    packed_impute_and_confidence,    # Single cell (packed)
    sample_and_insert,               # Impute + insert row (unpacked)
    packed_sample_and_insert,        # Impute + insert row (packed)
    multi_chain_impute_and_confidence,  # Multi-chain
)
```

### Predictive Inference
```python
from crosscat import (
    predictive_probability,          # P(query | conditions) (unpacked)
    predictive_sample,               # Draw samples (unpacked)
    predictive_cdf,                  # P(X <= val) (unpacked)
    packed_predictive_probability,   # (packed)
    packed_predictive_sample,        # (packed)
    packed_predictive_cdf,           # (packed)
    batch_predictive_probability,    # Per-row log prob (packed)
    batch_predictive_sample,         # Per-row samples (packed)
    multi_chain_predictive_probability,  # Multi-chain
    multi_chain_predictive_sample,       # Multi-chain
)
```

### Classification
```python
from crosscat import (
    packed_classify_column,          # Argmax predictive (packed)
    batch_classify_column,           # Batch classification (packed)
    batch_score_columns_binary,      # Binary column probabilities (packed)
    multi_chain_classify_column,     # Ensemble classification (multi-chain)
)
```

### Dependency Discovery
```python
from crosscat import (
    dependence_probability,          # Pairwise P(col_i ~ col_j) (unpacked)
    dependence_matrix,               # Full Z-matrix (unpacked, multi-state)
    packed_dependence_probability,   # Pairwise (packed, accepts list)
    packed_dependence_matrix,        # Full Z-matrix (packed, accepts list, PREFERRED)
    batch_dependence_probability,    # Multiple column pairs (packed)
    mutual_information,              # MI + Linfoot correlation (multi-state)
    packed_mutual_information,       # MI (packed, accepts list)
    conditional_entropy,             # H(target | given) (unpacked)
    packed_conditional_entropy,      # H(target | given) (packed, accepts list)
    batch_conditional_entropy,       # Multiple targets (packed)
    batch_column_typicality,         # Multiple columns (packed)
)
```

### Credible Intervals
```python
from crosscat import (
    credible_interval,               # Percentile CI (unpacked)
    packed_credible_interval,        # (packed)
    batch_credible_interval,         # Multiple rows (packed)
    multi_chain_credible_interval,   # Pooled across chains
)
```

### Serialization & Checkpointing
```python
from crosscat import (
    save_state, load_state,                  # Single state .jxc
    save_packed_state, load_packed_state,     # Packed state .jxc
    save_checkpoint, load_latest_checkpoint,  # Checkpoint directory
    save_data, load_data,                    # Arrow IPC with metadata (PREFERRED)
)
```

### Diagnostics
```python
from crosscat import (
    log_joint,                       # Model score (for convergence)
    collect_diagnostics,             # Per-sweep metrics dict
    adjusted_rand_index,             # ARI between partitions
    gelman_rubin_rhat,               # R-hat convergence (multi-chain)
    effective_sample_size,           # ESS (multi-chain)
    random_holdout_mask,             # Cell-level holdout mask
    packed_evaluate_imputation,      # Holdout imputation metrics (packed, PREFERRED)
    evaluate_imputation,             # (unpacked)
)
```

### Scaling (10K+ rows)
```python
from crosscat import (
    subsample_anneal,                # Subsample → grow → full inference
    minibatch_gibbs_sweep,           # Mini-batch row transitions
    parallel_gibbs_sweep,            # Parallel row scoring
    gibbs_sweep_early_stopping,      # Stop when log_joint plateaus
)
```
