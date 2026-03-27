# Installation

## Requirements

- Python 3.11+
- JAX 0.4+

## Install with pip or uv

We recommend [uv](https://docs.astral.sh/uv/) for fast, reproducible installs.

=== "CPU only"

    ```bash
    uv pip install jax-crosscat
    ```

=== "GPU (NVIDIA CUDA 13)"

    ```bash
    uv pip install "jax-crosscat[gpu]"
    ```

=== "GPU (AMD ROCm)"

    ```bash
    uv pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html
    uv pip install jax-crosscat
    ```

=== "pip"

    ```bash
    pip install jax-crosscat
    # or with GPU:
    pip install "jax-crosscat[gpu]"
    ```

## Install from Source

```bash
git clone https://github.com/sambhal-labs/jaxcross.git
cd jaxcross
uv sync --extra dev
```

## Verify Installation

```python
import crosscat
print(crosscat.__version__)

import jax
print(jax.devices())  # should show GPU if installed with CUDA
```

## Optional Extras

| Extra | Install | Purpose |
|-------|---------|---------|
| `gpu` | `pip install "jax-crosscat[gpu]"` | NVIDIA CUDA 13 support |
| `dashboard` | `pip install "jax-crosscat[dashboard]"` | Streamlit interactive dashboard |
| `benchmark` | `pip install "jax-crosscat[benchmark]"` | Matplotlib, scikit-learn for benchmarks |
| `dev` | `pip install "jax-crosscat[dev]"` | pytest, ruff, mypy for development |
| `docs` | `pip install "jax-crosscat[docs]"` | MkDocs + Material for documentation |

## Kaggle / Colab Setup

On Kaggle (P100) or Colab (T4/A100), JAX with CUDA is pre-installed. Use `--no-deps` to avoid overwriting the platform JAX:

```bash
pip install -e . --no-deps
```

!!! warning "Do NOT use `uv sync --extra gpu` on Kaggle"
    This installs a different CUDA toolkit version that causes `ptxas` version mismatches. Use `pip install -e . --no-deps` instead.
