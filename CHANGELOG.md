# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-03-18

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

## [0.2.0] - 2026-03-17

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
