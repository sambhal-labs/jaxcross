---
hide:
  - navigation
---

# JAX-CrossCat

<p align="center" style="font-size: 1.3em; color: #666;">
GPU-accelerated Bayesian cross-categorization in JAX
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/sambhal-labs/jaxcross/actions"><img src="https://github.com/sambhal-labs/jaxcross/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

---

**jax-crosscat** is a modern reimplementation of [probcomp/crosscat](https://github.com/probcomp/crosscat) — the Bayesian nonparametric model that simultaneously discovers which columns are related and how rows cluster within each group. Built on [JAX](https://github.com/jax-ml/jax) for hardware-accelerated inference on CPU, GPU, and TPU.

## Why CrossCat?

Most clustering methods force a single partition over all columns. CrossCat discovers that *different subsets of columns may cluster rows differently*. A dataset of employees might cluster by `(salary, experience)` into seniority tiers, but independently cluster by `(commute_distance, zip_code)` into geographic regions — with no alignment between the two.

CrossCat models this with a **two-level Dirichlet Process**:

<p align="center">
  <img src="diagrams/two-level-dp.svg" alt="Two-Level Dirichlet Process Mixture Model" width="800" />
</p>

## Features

<div class="grid cards" markdown>

-   **5 Column Types**

    ---

    Continuous (Normal-Gamma), Categorical (Dirichlet-Categorical), Binary (Beta-Bernoulli), Ordinal (Ordered Logistic), Cyclic (Von Mises)

-   **Collapsed Gibbs Sampling**

    ---

    All parameters integrated out via conjugacy — only cluster assignments are sampled. No tuning required.

-   **Full Query API**

    ---

    Predictive probability, sampling, CDF, mutual information, anomaly detection, imputation, row similarity

-   **Missing Data**

    ---

    NaN values handled transparently — skipped during sufficient statistic computation

-   **Constraint Enforcement**

    ---

    Enforce column/row dependency constraints during inference to incorporate domain knowledge

-   **Multi-Chain Inference**

    ---

    Initialize multiple independent chains, select best by log-joint, or aggregate queries across chains

-   **GPU Acceleration**

    ---

    JIT-compiled packed state with vectorized kernels — 12x speedup over sequential path

-   **Convergence Diagnostics**

    ---

    Adjusted Rand Index, log-joint tracking, held-out likelihood, imputation evaluation

</div>

## Quick Start

```python
from crosscat import initialize, read_csv, guess_column_types
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state

# 1. Load your data
data, col_names = read_csv("employees.csv")
col_types = guess_column_types(data)

# 2. Initialize and run inference
key = jax.random.key(42)
state = initialize(key, data, col_types)
packed = pack_state(state)
packed = packed_gibbs_sweep(jax.random.key(1), packed, data, n_sweeps=100)
state = unpack_state(packed, col_types, data=data)

# 3. Ask questions
from crosscat import predictive_sample, dependence_matrix
samples = predictive_sample(jax.random.key(2), state, data, query_cols=[0])
z_matrix = dependence_matrix([state])  # which columns are related?
```

[:material-rocket-launch: Get Started](getting-started/quickstart.md){ .md-button .md-button--primary }
[:material-book-open-variant: Feature Guides](guides/index.md){ .md-button }
[:material-code-tags: API Reference](api/index.md){ .md-button }

## Performance

The `crosscat.packed` sub-package provides JIT-compiled kernels with vectorized column scoring and type-specialized fast paths:

| Dataset | Rows x Cols | Per Sweep | 100 Sweeps |
|---------|-------------|-----------|------------|
| Small (mixed types) | 50 x 11 | 4.5s | 7.5 min |
| Medium (binary+cat) | 100 x 65 | 4.8s | 8 min |
| MNIST 16x16 | 1000 x 257 | 12s | 20 min |

Benchmarked on NVIDIA P100 GPU. See [MNIST Benchmark](examples/mnist.md) for the full paper reproduction.

## References

- Mansinghka, V., Shafto, P., Jonas, E., Petschulat, C., Gasner, M., & Tenenbaum, J. B. (2016). *CrossCat: A Fully Bayesian Nonparametric Method for Analyzing Heterogeneous, High Dimensional Data.* JMLR, 17(138), 1-49.
- Original implementation: [probcomp/crosscat](https://github.com/probcomp/crosscat)
