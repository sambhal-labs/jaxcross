---
name: run-benchmark
description: Run WDI or MNIST benchmark locally on GTX 1650 with appropriate config
disable-model-invocation: true
---

Run a benchmark notebook locally on the GTX 1650 GPU. Usage: `/run-benchmark wdi` or `/run-benchmark mnist`.

1. Verify GPU is available:
   ```python
   import jax; print(jax.devices())
   ```
   If no GPU, warn the user and suggest Kaggle instead.

2. Based on the argument:

   **WDI** (`benchmarks/wdi_macroeconomic_benchmark.ipynb`):
   - Recommended local config: `N_CHAINS=3`, `N_SWEEPS=500`, `N_INDICATORS=80`
   - Expected runtime: ~2-3 hours on GTX 1650
   - Memory: ~1.5 GB VRAM (fits easily in 4 GB)
   - Convert notebook to script and run, or guide user to run in Jupyter

   **MNIST** (`benchmarks/mnist_paper_colab.ipynb`):
   - WARNING: JIT compilation for 257 columns takes 90+ minutes on GTX 1650
   - Recommended: Run on Kaggle P100 instead
   - If user insists on local: `N_CHAINS=2`, `N_SWEEPS=300`, `PIXEL_SIZE=8` (64 pixels instead of 256)
   - Expected runtime: 4-6 hours including JIT

3. Check that `uv sync --extra dev --extra gpu` has been run and JAX sees the GPU.

4. Run the notebook using `uv run jupyter nbconvert --execute --to notebook` or guide the user to open it in Jupyter/VS Code.

5. After completion, summarize key results: log_joint convergence, Z-matrix structure, classification/imputation accuracy, and wall-clock timing per section.

Note: Never run the full benchmark in CI — GPU tests are too slow. This skill is for local development only.
