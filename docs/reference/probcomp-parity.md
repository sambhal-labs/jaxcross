# probcomp/crosscat Parity

**jax-crosscat** is a modern reimplementation of the [probcomp/crosscat](https://github.com/probcomp/crosscat) reference implementation from the original JMLR 2016 paper (Mansinghka, Shafto, Jonas, Petschulat, Gasner, Tenenbaum). This page is an appendix tracking what survived, what changed, and why — for readers familiar with the original codebase.

!!! note "Not a migration guide"
    This is a reference appendix for credibility and context, not a tutorial on porting probcomp/crosscat code. For new users, start with the [Core Concepts](../getting-started/concepts.md) and [Quickstart](../getting-started/quickstart.md).

## Feature Matrix

| Feature | probcomp/crosscat | jax-crosscat | Status | Notes |
|---------|-------------------|--------------|--------|-------|
| **Two-level DP model** | yes | yes | Parity | Same mathematical model (outer DP on columns, inner DP on rows). |
| **Collapsed Gibbs sampling** | yes | yes | Parity | Parameters integrated out via conjugate priors. |
| **NormalGamma (continuous)** | yes | yes | Parity | Same Normal-Inverse-Gamma prior parameterisation (`mu`, `r`, `s`, `nu`). |
| **DirichletCategorical** | yes | yes | Parity | Symmetric concentration `alpha`. |
| **BetaBernoulli (binary)** | yes | yes | Parity | `alpha`, `beta` shape hypers. |
| **OrderedLogistic (ordinal)** | yes | yes | Divergence | jaxcross uses grid integration over the latent location; original uses Metropolis-Hastings. Faster and JIT-friendly but grid-coarse — see [ordered-logistic-grid.md](../architecture/algorithms/ordered-logistic-grid.md). |
| **VonMises (cyclic)** | no | yes | Extension | Cyclic variables (angles, time-of-day) were not in the 2016 paper. jaxcross adds them. |
| **Grid Gibbs on hypers** | yes | yes | Parity | Same 31-point grid pattern, data-driven ranges. See [hyper-transitions.md](../architecture/algorithms/hyper-transitions.md). |
| **CRP concentration sampling** | yes | yes | Parity | `alpha_col` outer DP + per-view `alpha_v` inner DP, grid-updated. |
| **Missing data via NaN** | yes | yes | Parity | Same masking strategy. |
| **Constraint enforcement** | yes | yes | Parity | Column/row dependency constraints via rejection sampling. |
| **JIT / GPU acceleration** | no | yes | **Extension** | The big delta. probcomp/crosscat is pure CPython; jaxcross packs state into fixed-size JAX arrays for `vmap`/`lax.scan`/`jit` — 10–100× faster on GPU. |
| **Multi-chain inference** | manual loops | `multi_chain_packed_gibbs_sweep` | Extension | First-class API; `jax.pmap` for multi-GPU. |
| **Online row insertion** | no | `packed_insert_rows` | Extension | Incremental inference without full re-sweeps. |
| **XLA persistent cache** | n/a | yes (auto) | Extension | Compiled kernels persist to disk across sessions. |
| **Unified query API** | yes | yes | Parity+ | All 15 core queries from the original plus 25 more (batch + multi-chain wrappers + classification). |
| **`.jxc` serialization** | Python-native binary format | versioned `.jxc` | Divergence | jaxcross uses atomic writes + schema versioning + NPZ-backed arrays. |
| **Z-matrix (column dependence)** | yes | yes | Parity | `dependence_matrix` / `packed_dependence_matrix`. |
| **Mutual information** | yes | yes | Parity | MC estimate averaged across chains. |
| **Splash screen / GUI** | Navdata UI | Streamlit dashboard | Divergence | Different tooling; same intent (interactive exploration). |
| **Python 2.7** | yes (legacy) | no | Divergence | jaxcross requires Python 3.11+. |

## API-Level Differences

Concepts map one-to-one, but call signatures differ because jaxcross is written in modern Python with JAX conventions.

| Concept | probcomp/crosscat | jax-crosscat |
|---------|-------------------|--------------|
| Create a model | `State(T, cctypes=...)` | `initialize(key, data, column_types)` → `InitResult` |
| Run inference | `state.transition(N=100)` | `packed = packed_gibbs_sweep(key, packed, data, n_sweeps=100)` |
| Query predictive | `state.simulate(...)` | `packed_predictive_sample(key, packed, data, ...)` |
| Query dependence | `state.dependence_probability(i, j)` | `packed_dependence_probability([packed], i, j)` |
| Serialize | Python-native binary | `save_packed_state(packed, path)` → `.jxc` |

Note: jaxcross separates **unpacked** (`CrossCatState`, Python-friendly) from **packed** (`PackedCrossCatState`, JIT-friendly) representations. probcomp/crosscat has only the equivalent of the unpacked form. See [Packed State Design](../architecture/packed-state.md).

## Intentional Divergences

### 1. OrderedLogistic via grid integration

The original uses Metropolis-Hastings over the latent location parameter. jaxcross grid-integrates on a 31-point uniform grid — faster, JIT-compatible, deterministic. The tradeoff: slightly coarser posterior on the location. For typical 5–10 level ordinals the difference is well below MC noise from the row-assignment step.

### 2. VonMises as a first-class component

Cyclic variables (compass bearings, wind direction, time-of-day phases) are common in practice but absent from the 2016 paper. jaxcross adds them as a full component with prior hypers (`kappa`, `vm_a`, `vm_mu`) and grid updates. All queries (anomaly, imputation, predictive) work uniformly across all 5 types.

### 3. Explicit packed/unpacked split

JIT compilation forces fixed-size tensors. jaxcross exposes this reality: `pack_state` materialises padded arrays (with `max_views`, `max_clusters`, `max_cols_per_view`, `max_categories` budgets), `unpack_state` reconstructs the Python-level view. Users pay the small mental overhead in exchange for GPU-accelerated inference. See [the guide](../guides/gpu-packed.md).

### 4. Schema-versioned serialization

The original format is fragile across Python/library versions. jaxcross ships `.jxc`: JSON metadata + NPZ arrays + a `.valid` marker written atomically. Schema version is bumped whenever array fields change, with migrations applied on load. See [Serialization](../api/serialization.md).

## What We Kept

- The **algorithmic core** is unchanged: collapsed Gibbs over row/column assignments, grid Gibbs over hypers, CRP priors throughout.
- The **mathematical invariants** (suffstat roundtrip, score parity across representations) are test-enforced — see `tests/test_property.py` and `tests/test_packed_inference_parity.py`.
- The **query semantics**: `dependence_probability`, `row_similarity`, `mutual_information` mean the same thing. If you ran the MNIST benchmark on probcomp/crosscat you will recognise the Z-matrix and inpainting outputs.

## Citation

If you use jax-crosscat, please cite the original paper — the algorithm is theirs:

```bibtex
@article{mansinghka2016crosscat,
  title={CrossCat: A Fully Bayesian Nonparametric Method for Analyzing
         Heterogeneous, High Dimensional Data},
  author={Mansinghka, Vikash and Shafto, Patrick and Jonas, Eric and
          Petschulat, Cap and Gasner, Max and Tenenbaum, Joshua B},
  journal={Journal of Machine Learning Research},
  volume={17}, number={138}, pages={1--49}, year={2016}
}
```

## See Also

- [Architecture Overview](../architecture/overview.md)
- [Core Concepts](../getting-started/concepts.md)
- [probcomp/crosscat on GitHub](https://github.com/probcomp/crosscat)
