# jax-crosscat

[![PyPI version](https://img.shields.io/pypi/v/jax-crosscat)](https://pypi.org/project/jax-crosscat/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

GPU-accelerated nonparametric cross-categorization in JAX.

A modern reimplementation of [CrossCat](https://github.com/probcomp/crosscat) — the Bayesian
cross-categorization model that simultaneously discovers column groupings ("views") and row
clusters within each view using a two-level Dirichlet Process.

## Key Features

- **Outer DP column partitioning** via collapsed Gibbs with `jax.lax.scan` — full XLA compilation
- **Inner DP row clustering** per view via `jax.vmap` — parallel across all views on GPU
- **Heterogeneous column types**: continuous (Normal-Gamma), categorical (Dirichlet-Categorical), ordinal (ordered logistic), binary (Beta-Bernoulli)
- **BlackJAX NUTS** for hyperparameter sampling (replaces grid-based sampling from original)
- **Posterior predictive queries**: conditional distributions, credible intervals, mutual information, anomaly detection

## Installation

```bash
pip install jax-crosscat
```

For GPU support:
```bash
pip install jax[cuda12]
pip install jax-crosscat
```

## Quick Start

```python
import jax
import crosscat

# Coming soon — library is in active development
```

## Development

```bash
git clone https://github.com/sambhal-labs/jaxcross.git
cd jaxcross
pip install -e ".[dev]"
pytest
```

## Status

**Pre-alpha** — core inference engine under development.

## References

- Mansinghka, V. et al. (2016). CrossCat: A Fully Bayesian Nonparametric Method for Analyzing Heterogeneous, High Dimensional Data.
- Dinari, O. & Zamir, T. (2022). GPU-accelerated Dirichlet Process Mixture Models.

## License

Apache 2.0
