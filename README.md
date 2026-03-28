<p align="center">
  <img src="docs/diagrams/two-level-dp.svg" alt="JAX-CrossCat" width="720" />
</p>

<h1 align="center">jax-crosscat</h1>

<p align="center">
  <strong>GPU-accelerated Bayesian structure discovery for tabular data</strong>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/sambhal-labs/jaxcross/actions"><img src="https://github.com/sambhal-labs/jaxcross/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://sambhal-labs.github.io/jaxcross/"><img src="https://img.shields.io/badge/docs-mkdocs-blue.svg" alt="Docs"></a>
  <img src="https://img.shields.io/badge/JAX-0.4+-green.svg" alt="JAX">
  <img src="https://img.shields.io/badge/version-0.10.0-orange.svg" alt="Version">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="notebooks/intro_tutorial.ipynb">Interactive Tutorial</a> &middot;
  <a href="https://sambhal-labs.github.io/jaxcross/">Documentation</a> &middot;
  <a href="benchmarks/">Benchmarks</a>
</p>

---

**jax-crosscat** automatically discovers hidden structure in your data — which columns are related, how rows cluster, and how to predict missing values — all without manual feature engineering or model selection.

Built on [JAX](https://github.com/jax-ml/jax) for hardware-accelerated inference on GPU and TPU. A modern reimplementation of [probcomp/crosscat](https://github.com/probcomp/crosscat) (Mansinghka et al., JMLR 2016).

## The Problem

Most clustering methods force a single partition over all columns. Real data doesn't work that way.

An employee dataset might cluster by `(salary, experience)` into seniority tiers, but independently by `(commute_distance, zip_code)` into geographic regions — with no alignment between the two. CrossCat discovers these **multiple overlapping structures** automatically.

## Key Capabilities

<table>
<tr>
<td width="50%">

**Automatic Structure Discovery**
- Discovers which columns are statistically related
- Finds independent clustering structures per column group
- Infers the number of clusters automatically — no k to tune

</td>
<td width="50%">

**Rich Query API**
- Predictive probability, sampling, and CDF
- Anomaly detection and row similarity
- Mutual information and dependence discovery
- Missing value imputation with confidence scores

</td>
</tr>
<tr>
<td>

**Production-Ready**
- 5 column types: continuous, categorical, binary, ordinal, cyclic
- Transparent NaN handling — no preprocessing needed
- Serialization, checkpointing, and convergence diagnostics
- Constraint enforcement for domain knowledge

</td>
<td>

**GPU-Accelerated**
- JIT-compiled packed state with vectorized Gibbs kernels
- 12x speedup over sequential scoring via `vmap`
- Multi-chain inference with automatic best-chain selection
- XLA persistent compilation cache for instant restarts

</td>
</tr>
</table>

## Quick Start

```bash
pip install jax-crosscat               # CPU
pip install "jax-crosscat[gpu]"        # GPU (NVIDIA CUDA)
```

```python
import jax
from crosscat import initialize, predictive_sample, dependence_matrix
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat.types import ColumnType

# Load and configure
data, col_types = your_data, [ColumnType.CONTINUOUS, ColumnType.CATEGORICAL, ...]

# Initialize → Pack → Infer → Unpack → Query
key = jax.random.key(42)
state = initialize(key, data, col_types)
packed = pack_state(state)
packed = packed_gibbs_sweep(jax.random.key(1), packed, data, n_sweeps=100)
state = unpack_state(packed, col_types, data=data)

# Discover column relationships
z_matrix = dependence_matrix([state])  # which columns are related?

# Predict missing values
samples = predictive_sample(jax.random.key(2), state, data, query_cols=[0])
```

> **Want the full walkthrough?** Open the **[Interactive Tutorial](notebooks/intro_tutorial.ipynb)** — covers synthetic data, inference, and 7 query types end-to-end.

## Column Types

CrossCat natively handles mixed-type data — no encoding or preprocessing required:

| Type | Statistical Model | Example Data |
|------|-------------------|-------------|
| `CONTINUOUS` | Normal-Gamma (conjugate) | Salary, temperature, sensor readings |
| `CATEGORICAL` | Dirichlet-Categorical | Department, country code, product category |
| `BINARY` | Beta-Bernoulli | Yes/no flags, presence/absence |
| `ORDINAL` | Ordered Logistic (cumulative link) | Star ratings, education level, severity |
| `CYCLIC` | Von Mises | Wind direction, time of day, compass bearing |

## Query API

After inference, ask questions about your data:

```python
from crosscat import (
    predictive_probability,     # P(col=value | context)
    predictive_sample,          # Draw from posterior predictive
    impute_and_confidence,      # Fill missing values with confidence
    mutual_information,         # Information shared between columns
    dependence_matrix,          # Full pairwise column dependency matrix
    predictive_anomalousness,   # Detect unusual rows
    row_similarity,             # How similar are two rows?
    credible_interval,          # Bayesian credible intervals
    conditional_entropy,        # Remaining uncertainty in a column
)
```

All queries are fully Bayesian — they integrate over cluster assignment uncertainty, not just point estimates. See the [Query Guides](docs/guides/queries/) for detailed examples.

## Performance

| Dataset | Rows x Cols | Per Sweep | 100 Sweeps |
|---------|-------------|-----------|------------|
| Small (mixed types) | 50 x 11 | 4.5s | 7.5 min |
| Medium (binary+cat) | 100 x 65 | 4.8s | 8 min |
| MNIST 16x16 | 1,000 x 257 | 12s | 20 min |

Benchmarked on NVIDIA P100 GPU. See [benchmarks/](benchmarks/) for reproduction scripts including the [MNIST paper benchmark](benchmarks/mnist_paper_colab.ipynb).

## Architecture

<p align="center">
  <img src="docs/diagrams/architecture-pipeline.svg" alt="Architecture Pipeline" width="720" />
</p>

CrossCat uses a **two-level Dirichlet Process** mixture model:
1. **Outer DP** partitions columns into views (column groups)
2. **Inner DP** per view clusters rows independently

All component parameters are collapsed out via conjugate priors — only cluster assignments and hyperparameters are sampled via Gibbs.

The **packed path** converts variable-size Python state into fixed-size JAX arrays for JIT compilation with `lax.scan` and `vmap`, enabling GPU-accelerated inference.

```
CrossCatState  ──pack_state()──▸  PackedCrossCatState  ──packed_gibbs_sweep()──▸  ...  ──unpack_state()──▸  CrossCatState
  (Python)                          (JAX arrays, JIT)                                                        (query-friendly)
```

See [docs/architecture/](docs/architecture/) for deep dives into the model, kernels, and JAX patterns.

## Project Structure

```
crosscat/
├── types.py              # Core dataclasses: CrossCatState, ViewState, ColumnType
├── components.py         # 5 Bayesian component models
├── model.py              # Initialization, scoring, row insertion
├── gibbs.py              # Collapsed Gibbs MCMC kernels
├── inference.py          # 15 posterior predictive queries
├── packed/               # JIT-compiled packed state path
│   ├── state.py          #   Pack/unpack, batching, multi-chain
│   ├── components.py     #   Unified type-dispatched scoring
│   ├── kernels.py        #   Vectorized Gibbs kernels
│   └── suffstats.py      #   Batched sufficient statistics
├── packed_inference.py   # 15 packed queries + 5 multi-chain wrappers
├── constraints.py        # Column/row dependency enforcement
├── diagnostics.py        # ARI, log-joint, held-out likelihood
├── serialization.py      # Save/load in .jxc format
├── synthetic.py          # Synthetic data generation
└── data_utils.py         # CSV I/O, type detection

notebooks/
├── intro_tutorial.ipynb  # Start here — full beginner walkthrough
└── run_tests.ipynb       # Test runner for Kaggle/Colab GPU

benchmarks/
├── mnist_paper_colab.ipynb          # MNIST paper reproduction (Section 3.2)
├── wdi_macroeconomic_benchmark.ipynb # Real-world macroeconomic data
├── paper_synthetic_benchmark.py     # Figure 7 synthetic recovery
└── jit_benchmark.py                 # Per-sweep timing

docs/
├── getting-started/      # Installation, quickstart, concepts
├── guides/               # Feature guides + 7 query-specific guides
├── api/                  # Complete API reference (18 modules)
├── architecture/         # Model design, kernels, JAX patterns
└── examples/             # End-to-end workflows
```

## Documentation

| Resource | Description |
|----------|-------------|
| **[Interactive Tutorial](notebooks/intro_tutorial.ipynb)** | Hands-on notebook: data generation, inference, 7 query types |
| **[Getting Started](docs/getting-started/)** | Installation, quickstart, core concepts |
| **[Feature Guides](docs/guides/)** | Deep dives into every capability |
| **[Query Guides](docs/guides/queries/)** | Dedicated guides for each query type |
| **[API Reference](docs/api/)** | Complete function documentation (88+ functions) |
| **[Architecture](docs/architecture/)** | Internal design, JAX patterns, performance |
| **[Benchmarks](benchmarks/)** | MNIST, synthetic recovery, JIT timing |
| **[Full Docs Site](https://sambhal-labs.github.io/jaxcross/)** | Searchable hosted documentation |

## From Source

```bash
git clone https://github.com/sambhal-labs/jaxcross.git && cd jaxcross
uv sync --extra dev                    # CPU
uv sync --extra dev --extra gpu        # GPU (NVIDIA CUDA)

uv run pytest                          # Run tests
uv run ruff check . && uv run ruff format .  # Lint & format
```

## Citation

If you use jax-crosscat in your research, please cite the original CrossCat paper:

```bibtex
@article{mansinghka2016crosscat,
  title={CrossCat: A Fully Bayesian Nonparametric Method for Analyzing
         Heterogeneous, High Dimensional Data},
  author={Mansinghka, Vikash and Shafto, Patrick and Jonas, Eric and
          Petschulat, Cap and Gasner, Max and Tenenbaum, Joshua B},
  journal={Journal of Machine Learning Research},
  volume={17},
  number={138},
  pages={1--49},
  year={2016}
}
```

## Contributing

We welcome contributions. See [docs/contributing.md](docs/contributing.md) for guidelines.

## License

[Apache 2.0](LICENSE)
