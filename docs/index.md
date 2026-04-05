---
hide:
  - navigation
---

# JAX-CrossCat

<p align="center" style="font-size: 1.3em; color: #666;">
Discover hidden structure in tabular data — automatically.<br/>
No feature engineering. No model selection. Just jaxcross.
</p>

<p align="center">
  <a href="https://github.com/sambhal-labs/jaxcross/releases"><img src="https://img.shields.io/github/v/release/sambhal-labs/jaxcross?color=orange" alt="Release"></a>
  <a href="https://mariadb.com/bsl11/"><img src="https://img.shields.io/badge/License-BSL_1.1-orange.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/sambhal-labs/jaxcross/actions"><img src="https://github.com/sambhal-labs/jaxcross/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/JAX-0.4+-green.svg" alt="JAX">
</p>

<p align="center" style="font-size: 1.1em; font-weight: 600;">
  <strong>12x faster</strong> than sequential inference &nbsp;&middot;&nbsp;
  <strong>5 native</strong> column types &nbsp;&middot;&nbsp;
  <strong>93% accuracy</strong> on MNIST inpainting &nbsp;&middot;&nbsp;
  <strong>Fully Bayesian</strong> — zero tuning
</p>

---

**jax-crosscat** is a modern reimplementation of [probcomp/crosscat](https://github.com/probcomp/crosscat) — the Bayesian nonparametric model that simultaneously discovers which columns are related and how rows cluster within each group. Built on [JAX](https://github.com/jax-ml/jax) for hardware-accelerated inference on CPU, GPU, and TPU.

<p align="center">
  <a href="https://colab.research.google.com/github/sambhal-labs/jaxcross/blob/main/notebooks/intro_tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

## Why jax-crosscat?

Most clustering methods force a single partition over all columns. CrossCat discovers that *different subsets of columns may cluster rows differently*.

A dataset of employees might cluster by `(salary, experience)` into seniority tiers, but independently cluster by `(commute_distance, zip_code)` into geographic regions — with no alignment between the two.

CrossCat models this with a **two-level Dirichlet Process**:

<p align="center">
  <img src="diagrams/two-level-dp.svg" alt="Two-Level Dirichlet Process Mixture Model" width="900" />
</p>

## Use Cases

<div class="grid cards" markdown>

-   **Customer Segmentation**

    ---

    Discover natural segments in mixed-type customer data — demographics, behavior, and spend — without choosing k or encoding categories

-   **Anomaly & Fraud Detection**

    ---

    Score how unusual each row is relative to learned structure. Flag outliers across heterogeneous record types with Bayesian confidence

-   **Missing Data Imputation**

    ---

    Fill missing values with posterior predictive sampling and confidence scores. No separate imputation pipeline needed

-   **Scientific Data Exploration**

    ---

    Uncover which variables are related in genomics, economics, or sensor data without assuming a model structure

-   **Feature Relationship Discovery**

    ---

    Build a dependence matrix showing which features carry information about each other, informing downstream ML pipelines

-   **Streaming & Online Inference**

    ---

    Insert new rows incrementally without re-running full inference. Score new observations against an existing trained model

</div>

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

    Predictive probability, sampling, CDF, mutual information, anomaly detection, imputation, row similarity, credible intervals, conditional entropy

-   **Missing Data**

    ---

    NaN values handled transparently — skipped during sufficient statistic computation, no preprocessing needed

-   **GPU Acceleration**

    ---

    JIT-compiled packed state with vectorized kernels via `vmap`/`lax.scan` — 12x speedup. XLA persistent cache for instant restarts

-   **Batched Operations**

    ---

    Vectorized column scoring, batched suffstat updates, batch posterior predictive for all 5 types, multi-chain wrappers

-   **Multi-Chain Inference**

    ---

    Initialize multiple independent chains, select best by log-joint, or aggregate queries across chains

-   **Streaming / Online**

    ---

    `packed_insert_rows` for incremental row insertion without full re-inference, `sample_and_insert` for posterior-aware insertion

-   **Constraint Enforcement**

    ---

    Column must-link / cannot-link and row dependency constraints via rejection sampling during inference

-   **Production-Ready**

    ---

    Serialization (`.jxc` format), checkpointing, state validation, deterministic RNG for reproducibility

-   **Convergence Diagnostics**

    ---

    Adjusted Rand Index, log-joint tracking, held-out likelihood, imputation evaluation

-   **Interactive Dashboard**

    ---

    Streamlit-based UI for visual exploration — data loading, inference, structure inspection, and querying

</div>

## Quick Start

```python
import jax
from crosscat import initialize, read_csv, guess_column_types
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state

# 1. Load your data
data, col_names = read_csv("employees.csv")
col_types = guess_column_types(data)

# 2. Initialize and run inference
key = jax.random.key(42)
result = initialize(key, data, col_types)
state = result.state
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

## Examples

| Example | Colab | Description |
|---------|-------|-------------|
| **[MNIST Benchmark](examples/mnist.md)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sambhal-labs/jaxcross/blob/main/benchmarks/mnist_paper_colab.ipynb) | Paper reproduction — pixel dependence, inpainting, classification |
| **[WDI Macroeconomics](examples/wdi-macroeconomics.md)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sambhal-labs/jaxcross/blob/main/benchmarks/wdi_macroeconomic_benchmark.ipynb) | Real-world GDP, trade, population — structure discovery |
| **[Intro Tutorial](https://colab.research.google.com/github/sambhal-labs/jaxcross/blob/main/notebooks/intro_tutorial.ipynb)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sambhal-labs/jaxcross/blob/main/notebooks/intro_tutorial.ipynb) | End-to-end walkthrough: synthetic data, inference, 7 query types |

## Getting Help

- **[FAQ & Troubleshooting](faq.md)** — Common questions and solutions
- **[Glossary](glossary.md)** — Key Bayesian terms explained
- **[GitHub Discussions](https://github.com/sambhal-labs/jaxcross/discussions)** — Questions, ideas, show & tell
- **[Issue Tracker](https://github.com/sambhal-labs/jaxcross/issues)** — Bug reports and feature requests

## References

- Mansinghka, V., Shafto, P., Jonas, E., Petschulat, C., Gasner, M., & Tenenbaum, J. B. (2016). *CrossCat: A Fully Bayesian Nonparametric Method for Analyzing Heterogeneous, High Dimensional Data.* JMLR, 17(138), 1-49.
- Original implementation: [probcomp/crosscat](https://github.com/probcomp/crosscat)
