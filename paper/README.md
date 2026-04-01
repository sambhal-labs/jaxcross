# JAX-CrossCat: Systems Paper

arXiv paper describing the JAX-CrossCat library.

## Building

```bash
# Collect figures from benchmark results
make figures

# Build PDF
make pdf

# Both
make all
```

## Figure Generation Pipeline

Figures come from benchmark outputs. Run these **on GPU** (Kaggle P100 recommended):

| Step | Command | Output |
|------|---------|--------|
| 1. Synthetic recovery | `uv run python benchmarks/paper_synthetic_benchmark.py` | `benchmarks/results/synthetic/` |
| 2. MNIST (Kaggle) | Run `benchmarks/mnist_paper_colab.ipynb` | `benchmarks/results/mnist/` |
| 3. WDI (Kaggle) | Run `benchmarks/wdi_macroeconomic_benchmark.ipynb` | `benchmarks/results/wdi/` |
| 4. Scalability | `uv run python benchmarks/scalability_benchmark.py` | `benchmarks/results/scalability/` |
| 5. Kernel timing | `uv run python benchmarks/jit_benchmark.py` | Console output for Table 2 |

Then run `make figures` to collect PNGs into `paper/figures/`.

## Paper Structure

| Section | Content | Figures/Tables |
|---------|---------|----------------|
| 1. Introduction | Motivation, contributions | — |
| 2. Background | CrossCat model, component models, inference | — |
| 3. System Design | Packed state, type dispatch, vectorized kernels, batched suffstats, XLA cache, inference queries | Architecture diagrams |
| 4. Experiments | Synthetic, MNIST, WDI, scalability | Figs 1-7, Tables 1-2 |
| 5. Software Engineering | Module org, testing, serialization | Table 3 |
| 6. Related Work | Original CrossCat, BayesDB, Loom, JAX ecosystem | — |
| 7. Discussion | Limitations, future work | — |
| 8. Conclusion | Summary | — |
| Appendix A | Inference query reference | Table 4 |

## Placeholder Values

Search for `[from benchmark]` and `[benchmark]` in `main.tex` — these need to be filled with actual numbers from benchmark runs.
