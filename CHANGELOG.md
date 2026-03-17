# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Packed state representation (`crosscat/packed_state.py`) for JIT-compatible inference
  - `PackedCrossCatState` dataclass registered as JAX pytree
  - `pack_state()` / `unpack_state()` conversion functions
  - Vectorized sufficient statistics via matrix operations
  - Unified scoring functions for JIT-compatible column type dispatch
  - Packed row assignment, hyperparameter, and CRP alpha kernels
  - `packed_gibbs_sweep()` for full packed inference
- Integration test suite for recovery across data types
  - `tests/test_integration_recovery.py`: 13 tests (cyclic, mixed-type, missing data, convergence)
  - `tests/test_integration_queries.py`: 17 tests (anomaly, MI, constraints, similarity)
  - `tests/test_packed_state.py`: pack/unpack roundtrip and kernel correctness
- Test fixtures for cyclic, mixed-type, and missing data scenarios
- Multi-chain inference helper with diagnostics collection
- Performance benchmark (`benchmarks/jit_benchmark.py`)
- CI coverage tracking with `pytest-cov` (60% minimum)
- Type checking with `mypy` in CI
- Slow test gate: `pytest -m slow` runs on push to main only
- PyPI release workflow (`.github/workflows/release.yml`)
- Pre-commit hooks for ruff lint and format
- Expanded API exports in `crosscat/__init__.py`

### Changed
- `pyproject.toml`: added mypy config, coverage dependencies, extended ruff rules (B, SIM)
- `.github/workflows/ci.yml`: added coverage, mypy, and slow test jobs

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
