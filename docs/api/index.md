# API Reference

Complete reference for the jax-crosscat public API, organized by module.

## Module Map

| Module | Purpose | Page |
|--------|---------|------|
| [`crosscat.types`](types.md) | State dataclasses: `CrossCatState`, `ViewState`, `SufficientStats`, `ColumnHypers`, `ColumnType` | [Types & State](types.md) |
| [`crosscat.model`](model.md) | `initialize()`, `log_joint()`, `insert_rows()` | [Model](model.md) |
| [`crosscat.components`](components.md) | Conjugate models: `NormalGamma`, `DirichletCategorical`, `BetaBernoulli`, `OrderedLogistic`, `VonMises` | [Components](components.md) |
| [`crosscat.gibbs`](gibbs.md) | MCMC kernels: `gibbs_sweep()` | [Gibbs Sampling](gibbs.md) |
| [`crosscat.inference`](inference.md) | Posterior queries: predictive sampling, anomaly detection, MI, imputation, similarity | [Inference Queries](inference.md) |
| [`crosscat.constraints`](constraints.md) | Column/row dependency constraint enforcement | [Constraints](constraints.md) |
| [`crosscat.diagnostics`](diagnostics.md) | ARI, convergence metrics, held-out evaluation | [Diagnostics](diagnostics.md) |
| [`crosscat.data_utils`](data-utils.md) | CSV I/O, column type detection, discretization | [Data Utilities](data-utils.md) |
| [`crosscat.serialization`](serialization.md) | Save/load states and checkpoints (`.jxc` format) | [Serialization](serialization.md) |
| [`crosscat.synthetic`](synthetic.md) | Synthetic data generation for testing | [Synthetic Data](synthetic.md) |
| [`crosscat.validate`](validation.md) | State consistency checking | [Validation](validation.md) |
| [`crosscat.packed.state`](packed-state.md) | `PackedCrossCatState`, `pack_state()`, `unpack_state()`, batching | [Packed State](packed-state.md) |
| [`crosscat.packed.kernels`](packed-kernels.md) | JIT-compiled Gibbs kernels on packed state | [Packed Kernels](packed-kernels.md) |
| [`crosscat.packed.components`](packed-components.md) | Unified scoring functions for type dispatch | [Packed Components](packed-components.md) |
| [`crosscat.packed_inference`](packed-inference.md) | Vectorized inference queries on packed state | [Packed Inference](packed-inference.md) |
| [`crosscat.packed.suffstats`](packed-suffstats.md) | Vectorized sufficient statistics | [Packed Suffstats](packed-suffstats.md) |
| [`crosscat.packed.aot_cache`](aot-cache.md) | XLA persistent compilation cache | [AOT Cache](aot-cache.md) |
