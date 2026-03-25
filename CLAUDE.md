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

- **types.py** — Dataclasses for all state: `CrossCatState` (full model), `ViewState` (one column group with row clustering), `ColumnHypers`, `SufficientStats`, and `ColumnType` enum (CONTINUOUS, CATEGORICAL, ORDINAL, BINARY, CYCLIC).

- **components.py** — Conjugate Bayesian component models (`NormalGamma`, `DirichletCategorical`, `BetaBernoulli`, `OrderedLogistic`, `VonMises`). Each provides `sufficient_statistics()`, `log_marginal_likelihood()`, and `posterior_predictive_logp()`.

- **model.py** — State initialization (`initialize()`), scoring (`log_joint()`), and row insertion (`insert_rows()`). Uses Chinese Restaurant Process sampling for cluster assignments and data-driven hyperparameter defaults.

- **gibbs.py** — Collapsed Gibbs MCMC kernels: `transition_row_assignments()`, `transition_column_assignments()`, `transition_column_hypers()`, `transition_crp_alphas()`, and `gibbs_sweep()` which runs a full iteration.

- **inference.py** — Posterior predictive queries: `predictive_probability()`, `predictive_sample()`, `predictive_cdf()`, `mutual_information()`, `dependence_probability()`, `dependence_matrix()`, `impute_and_confidence()`, `predictive_anomalousness()`, `row_similarity()`, `row_typicality()`, `column_typicality()`, `sample_and_insert()`, `credible_interval()`, `conditional_entropy()`, `joint_predictive_probability()`.

- **packed/** — JIT-compatible packed state sub-package:
  - `state.py` — `PackedCrossCatState` dataclass, `pack_state()`, `unpack_state()`
  - `components.py` — unified scoring (log marginal, posterior predictive) via `jnp.where` type dispatch + batch-vectorized type-specialized scoring (`batch_bb_posterior_predictive_logp`, `batch_ng_posterior_predictive_logp`, `batch_dc_posterior_predictive_logp`)
  - `suffstats.py` — vectorized sufficient statistics (matrix ops, batched scatter add/remove)
  - `kernels.py` — all Gibbs kernels (`packed_gibbs_sweep`, row/column assignments, hypers, CRP alphas) via `vmap`/`lax.scan` with type-specialized fast paths (`_compute_dominant_type`, `_score_row_one_cluster_typed`)
  - `aot_cache.py` — XLA persistent compilation cache (`enable_xla_cache()`, `clear_cache()`)

- **packed_inference.py** — Vectorized inference queries on packed state (predictive, MI, anomaly, similarity).

- **constraints.py** — Enforces column/row dependency constraints during inference.
- **diagnostics.py** — Convergence metrics (Adjusted Rand Index, held-out likelihood, imputation evaluation).
- **serialization.py** — Save/load states and checkpoints in `.jxc` format (JSON metadata + NPZ arrays).
- **synthetic.py** — Synthetic data generation from known CrossCat generative model, missing data injection.
- **data_utils.py** — CSV I/O, column type detection, discretization.
- **validate.py** — State consistency checking.
- **packed_state.py** — Legacy deprecation shim; import from `crosscat.packed` instead.
- **../contrib/fingerprint.py** — Entity behavioral fingerprinting (LaborLens-specific, not part of core).

## Key Patterns

- **JAX idioms**: Uses `jax.lax.scan` for sequential loops (view/row sweeps), `jax.vmap` for parallel operations (column scoring, cluster scoring, row clustering across views), `jax.jit` for compilation. All state is immutable — operations return new arrays.
- **Vectorized column scoring**: Row scoring in `_score_row_one_cluster` uses `jax.vmap(unified_posterior_predictive_logp)` over all columns simultaneously (not sequential `lax.scan`). This is the key optimization that gave 12x speedup in v0.9.0.
- **Type-specialized fast paths**: `_compute_dominant_type()` detects when all columns in a view share the same type (e.g., all BINARY for MNIST). When dominant, `_score_row_one_cluster_typed` bypasses the 5-way `jnp.where` dispatch and calls type-specific batch functions directly (`batch_bb_posterior_predictive_logp`, etc.).
- **Batched suffstat updates**: `_add_row_to_suffstats` / `_remove_row_from_suffstats` use `.at[].add()` scatter operations over all columns at once instead of sequential `lax.scan`.
- **Deterministic RNG**: Always use `jax.random.key()` and `jax.random.split()` for reproducibility.
- **NaN transparency**: Missing data is represented as NaN and silently filtered during sufficient statistic computation.
- **Docstring cross-references**: Many functions include "Maps to original..." comments linking to the probcomp/crosscat equivalent.

## Benchmarks

- **MNIST paper benchmark** (`benchmarks/mnist_paper_colab.ipynb`): Reproduces Section 3.2 of Mansinghka et al. (2016). 16×16 binary MNIST (257 cols), 10 chains × 100 sweeps on P100. Validates Z-matrix, pixel dependence map, inpainting (93% accuracy), and classification (79% accuracy).
- **Synthetic benchmark** (`benchmarks/paper_synthetic_benchmark.py`): Figure 7 recovery with known ground truth.
- **JIT benchmark** (`benchmarks/jit_benchmark.py`): Per-sweep timing comparison.
- Run notebooks on Kaggle (P100) or Colab (T4/A100) for GPU-accelerated benchmarks.

## Code Style

- Python 3.11+, ruff for linting (rules: E, F, I, W, UP, B, SIM), line length 99
- Type hints throughout
- Private functions prefixed with `_`
