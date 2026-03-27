<p align="center">
  <strong>jax-crosscat</strong>
</p>

<p align="center">
  <em>GPU-accelerated Bayesian cross-categorization in JAX</em>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/sambhal-labs/jaxcross/actions"><img src="https://github.com/sambhal-labs/jaxcross/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://sambhal-labs.github.io/jaxcross/"><img src="https://img.shields.io/badge/docs-mkdocs-blue.svg" alt="Documentation"></a>
</p>

---

**jax-crosscat** is a modern reimplementation of [probcomp/crosscat](https://github.com/probcomp/crosscat) — the Bayesian nonparametric model that simultaneously discovers which columns are related and how rows cluster within each group. Built on [JAX](https://github.com/jax-ml/jax) for hardware-accelerated inference on CPU, GPU, and TPU.

> **[Read the full documentation](https://sambhal-labs.github.io/jaxcross/)**

## Why CrossCat?

Most clustering methods force a single partition over all columns. CrossCat discovers that *different subsets of columns may cluster rows differently*. A dataset of employees might cluster by `(salary, experience)` into seniority tiers, but independently cluster by `(commute_distance, zip_code)` into geographic regions — with no alignment between the two.

<p align="center">
  <img src="docs/diagrams/two-level-dp.svg" alt="Two-Level Dirichlet Process Mixture Model" width="800" />
</p>

## Features

| Feature | Description |
|---------|-------------|
| **5 column types** | Continuous (Normal-Gamma), Categorical (Dirichlet-Categorical), Binary (Beta-Bernoulli), Ordinal (Ordered Logistic), Cyclic (Von Mises) |
| **Collapsed Gibbs** | All parameters integrated out via conjugacy — only cluster assignments are sampled |
| **Full query API** | Predictive probability, sampling, CDF, mutual information, anomaly detection, imputation, row similarity |
| **Missing data** | NaN values handled transparently — skipped during sufficient statistic computation |
| **Constraints** | Enforce column/row dependency constraints during inference |
| **Multi-chain** | Initialize multiple independent chains, select best by log-joint |
| **GPU acceleration** | JIT-compiled packed state with vectorized kernels — 12x speedup |
| **Convergence diagnostics** | Adjusted Rand Index, log-joint tracking, held-out likelihood |

## Quick Start

```python
import jax
from crosscat import initialize, read_csv, guess_column_types, predictive_sample
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state

# 1. Load your CSV data
data, col_names = read_csv("data.csv")
col_types = guess_column_types(data)

# 2. Initialize and run GPU-accelerated inference
key = jax.random.key(42)
state = initialize(key, data, col_types)
packed = pack_state(state)
packed = packed_gibbs_sweep(jax.random.key(1), packed, data, n_sweeps=100)
state = unpack_state(packed, col_types, data=data)

# 3. Query the posterior
samples = predictive_sample(jax.random.key(2), state, data, query_cols=[0])
```

See the **[Quick Start Guide](https://sambhal-labs.github.io/jaxcross/getting-started/quickstart/)** for a complete walkthrough.

## Installation

```bash
uv pip install jax-crosscat            # CPU
uv pip install "jax-crosscat[gpu]"     # GPU (NVIDIA CUDA 13)
```

From source:
```bash
git clone https://github.com/sambhal-labs/jaxcross.git && cd jaxcross
uv sync --extra dev
```

## Documentation

- **[Getting Started](https://sambhal-labs.github.io/jaxcross/getting-started/quickstart/)** — installation, quick start, core concepts
- **[Feature Guides](https://sambhal-labs.github.io/jaxcross/guides/)** — end-to-end guides for every feature
- **[API Reference](https://sambhal-labs.github.io/jaxcross/api/)** — complete function documentation
- **[Architecture](https://sambhal-labs.github.io/jaxcross/architecture/overview/)** — internal design and JAX patterns
- **[Examples](https://sambhal-labs.github.io/jaxcross/examples/csv-workflow/)** — full workflows including MNIST benchmark
- **[Changelog](https://sambhal-labs.github.io/jaxcross/changelog/)** — release history

## Performance

| Dataset | Rows × Cols | Per Sweep | 100 Sweeps |
|---------|-------------|-----------|------------|
| Small (mixed types) | 50 × 11 | 4.5s | 7.5 min |
| Medium (binary+cat) | 100 × 65 | 4.8s | 8 min |
| MNIST 16×16 | 1000 × 257 | 12s | 20 min |

Benchmarked on NVIDIA P100 GPU. See the [MNIST Benchmark](https://sambhal-labs.github.io/jaxcross/examples/mnist/) for the full paper reproduction.

## References

- Mansinghka, V., Shafto, P., Jonas, E., Petschulat, C., Gasner, M., & Tenenbaum, J. B. (2016). *CrossCat: A Fully Bayesian Nonparametric Method for Analyzing Heterogeneous, High Dimensional Data.* JMLR, 17(138), 1-49.
- Original implementation: [probcomp/crosscat](https://github.com/probcomp/crosscat)

## License

[Apache 2.0](LICENSE)
