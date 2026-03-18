# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX-CrossCat is a GPU-accelerated reimplementation of [probcomp/crosscat](https://github.com/probcomp/crosscat) using JAX. It implements a two-level Dirichlet Process mixture model: an outer DP partitions columns into "views", and an inner DP per view clusters rows. All parameters are collapsed out via conjugate Bayesian component models — only cluster assignments and hyperparameters are sampled.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_synthetic_recovery.py

# Run a single test function
pytest tests/test_new_features.py::test_function_name

# Exclude slow tests
pytest -m "not slow"

# Lint
ruff check .

# Format
ruff format .
```

## Architecture

The package is `crosscat/` with these core modules:

- **types.py** — Dataclasses for all state: `CrossCatState` (full model), `ViewState` (one column group with row clustering), `ColumnHypers`, `SufficientStats`, and `ColumnType` enum (CONTINUOUS, CATEGORICAL, ORDINAL, BINARY, CYCLIC).

- **components.py** — Conjugate Bayesian component models (`NormalGamma`, `DirichletCategorical`, `BetaBernoulli`, `OrderedLogistic`, `VonMises`). Each provides `sufficient_statistics()`, `log_marginal_likelihood()`, and `posterior_predictive_logp()`.

- **model.py** — State initialization (`initialize()`), scoring (`log_joint()`), and row insertion (`insert_rows()`). Uses Chinese Restaurant Process sampling for cluster assignments and data-driven hyperparameter defaults.

- **gibbs.py** — Collapsed Gibbs MCMC kernels: `transition_row_assignments()`, `transition_column_assignments()`, `transition_column_hypers()`, `transition_crp_alphas()`, and `gibbs_sweep()` which runs a full iteration.

- **inference.py** — Posterior predictive queries: `predictive_probability()`, `predictive_sample()`, `predictive_cdf()`, `mutual_information()`, `impute_and_confidence()`, `anomaly_score()`, `row_similarity()`, `sample_and_insert()`.

- **packed/** — JIT-compatible packed state sub-package:
  - `state.py` — `PackedCrossCatState` dataclass, `pack_state()`, `unpack_state()`
  - `components.py` — unified scoring (log marginal, posterior predictive) via `jnp.where` type dispatch
  - `suffstats.py` — vectorized sufficient statistics (matrix ops, incremental add/remove)
  - `kernels.py` — all Gibbs kernels (`packed_gibbs_sweep`, row/column assignments, hypers, CRP alphas) via `lax.scan`/`vmap`

- **packed_inference.py** — Vectorized inference queries on packed state (predictive, MI, anomaly, similarity).

- **constraints.py** — Enforces column/row dependency constraints during inference.
- **diagnostics.py** — Convergence metrics (Adjusted Rand Index, etc.).
- **data_utils.py** — CSV I/O and column type detection.
- **validate.py** — State consistency checking.
- **../contrib/fingerprint.py** — Entity behavioral fingerprinting (LaborLens-specific, not part of core).

## Key Patterns

- **JAX idioms**: Uses `jax.lax.scan` for sequential loops (column sweeps), `jax.vmap` for parallel operations (row clustering across views), `jax.jit` for compilation. All state is immutable — operations return new arrays.
- **Deterministic RNG**: Always use `jax.random.key()` and `jax.random.split()` for reproducibility.
- **NaN transparency**: Missing data is represented as NaN and silently filtered during sufficient statistic computation.
- **Docstring cross-references**: Many functions include "Maps to original..." comments linking to the probcomp/crosscat equivalent.

## Code Style

- Python 3.11+, ruff for linting (rules: E, F, I, W, UP), line length 99
- Type hints throughout
- Private functions prefixed with `_`
