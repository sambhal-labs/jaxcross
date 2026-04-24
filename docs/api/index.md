# API Reference

Complete reference for the jax-crosscat public API, organized by module. **All 134 symbols** exported from `crosscat` are covered here — either on a dedicated page or via the auto-generated `:::` section of the owning module.

## Module Map

| Module | Purpose | Page |
|--------|---------|------|
| [`crosscat.types`](types.md) | State dataclasses: `CrossCatState`, `ViewState`, `SufficientStats`, `ColumnHypers`, `ColumnType`, `InitResult`, constants (`LOG_EPS`, `LOGISTIC_INF`) | [Types & State](types.md) |
| [`crosscat.model`](model.md) | `initialize()`, `log_joint()`, `insert_rows()` | [Model](model.md) |
| [`crosscat.components`](components.md) | Conjugate models: `NormalGamma`, `DirichletCategorical`, `BetaBernoulli`, `OrderedLogistic`, `VonMises` | [Components](components.md) |
| [`crosscat.gibbs`](gibbs.md) | Unpacked MCMC kernels: `gibbs_sweep()` (slow — prefer packed kernels) | [Gibbs Sampling](gibbs.md) |
| [`crosscat.inference`](inference.md) | Unpacked posterior queries (15 functions): predictive sampling, anomaly detection, MI, imputation, similarity | [Inference Queries](inference.md) |
| [`crosscat.constraints`](constraints.md) | Column/row dependency constraint enforcement (5 functions) | [Constraints](constraints.md) |
| [`crosscat.diagnostics`](diagnostics.md) | ARI, convergence metrics (Rhat, ESS), held-out evaluation (10 functions) | [Diagnostics](diagnostics.md) |
| [`crosscat.data_utils`](data-utils.md) | Arrow/CSV/Parquet/NPY I/O, column type detection, discretization (17 functions) | [Data Utilities](data-utils.md) |
| [`crosscat.scaling`](scaling.md) | 10K+ row workflows: `subsample_anneal`, `minibatch_gibbs_sweep`, `parallel_gibbs_sweep`, `gibbs_sweep_early_stopping` | [Scaling](scaling.md) |
| [`crosscat.serialization`](serialization.md) | Save/load states and checkpoints (`.jxc` format, atomic writes) | [Serialization](serialization.md) |
| [`crosscat.synthetic`](synthetic.md) | Synthetic data generation for testing | [Synthetic Data](synthetic.md) |
| [`crosscat.validate`](validation.md) | State consistency checking (`validate_state`, `assert_valid_state`, `ValidationError`) | [Validation](validation.md) |
| [`crosscat.tb_logger`](tb-logger.md) | TensorBoard logging via `tensorboardX` | [TensorBoard Logger](tb-logger.md) |
| [`crosscat.packed.state`](packed-state.md) | `PackedCrossCatState`, `pack_state()`, `unpack_state()`, batching, `suggest_max_clusters`, `estimate_packed_memory` | [Packed State](packed-state.md) |
| [`crosscat.packed.kernels`](packed-kernels.md) | JIT-compiled Gibbs kernels (`packed_gibbs_sweep`, `packed_gibbs_step`, `multi_chain_packed_gibbs_sweep`, row/column/hyper/alpha transitions, `packed_insert_rows`, `packed_log_joint`) | [Packed Kernels](packed-kernels.md) |
| [`crosscat.packed.components`](packed-components.md) | Unified scoring functions for type dispatch | [Packed Components](packed-components.md) |
| [`crosscat.packed_inference`](packed-inference.md) | Vectorized inference queries on packed state (41 functions: 12 single-state, 4 multi-state, 16 batch, 9 multi-chain) | [Packed Inference](packed-inference.md) |
| [`crosscat.packed.suffstats`](packed-suffstats.md) | Vectorized sufficient statistics | [Packed Suffstats](packed-suffstats.md) |
| [`crosscat.packed.aot_cache`](aot-cache.md) | XLA persistent compilation cache (`enable_xla_cache`, `compile_kernels`, `clear_cache`) | [AOT Cache](aot-cache.md) |

## Three Tiers of Inference Functions

| Prefix | Accepts | Batches over | Example |
|--------|---------|-------------|---------|
| `packed_*` | single `PackedCrossCatState` | nothing (one query) | `packed_anomaly_score(key, packed, data, query_row=5)` |
| `batch_*` | single `PackedCrossCatState` | **rows / queries** (vmap) | `batch_anomaly_score(packed, data, row_ids)` |
| `multi_chain_*` | `list[PackedCrossCatState]` | **chains** (Bayesian model averaging) | `multi_chain_anomaly_score(key, chains, data, query_row=5)` |

Some posterior-structure queries accept a **list** of packed states to average over samples: `packed_dependence_matrix(list)`, `packed_mutual_information(list)`, `packed_row_similarity(list)`, `packed_dependence_probability(list)`.

See the [Packed Inference](packed-inference.md) reference for the full catalog.
