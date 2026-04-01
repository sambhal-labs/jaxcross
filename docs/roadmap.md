# Roadmap

A high-level view of where jax-crosscat has been and where it's headed.

---

## Completed Milestones

### v0.1.0 — Foundation (Jan 2025)
Full CrossCat feature parity: 5 component models, collapsed Gibbs kernels, 15 posterior queries, constraint enforcement, diagnostics, synthetic data generation.

### v0.2.0–v0.3.0 — Packed State & GPU (Mar 2026)
JIT-compiled packed state representation. All Gibbs kernels rewritten with `lax.scan` and `vmap`. Vectorized packed inference queries. Multi-chain support.

### v0.4.0 — Dashboard (Mar 2026)
Interactive Streamlit dashboard for visual CrossCat analysis.

### v0.5.0–v0.8.0 — Accuracy & Benchmarks (Mar 2026)
Data-dependent hyperparameter grids matching the original paper. Von Mises model fixes. MNIST paper benchmark reproduction. Benchmark infrastructure with result persistence.

### v0.9.0 — 12x Performance (Mar 2026)
Vectorized column scoring, type-specialized fast paths, XLA persistent compilation cache. MNIST (1000x257) dropped from 238s to 20s per sweep.

### v0.10.0–v0.10.1 — Ordinal & Polish (Mar 2026)
True ordered logistic component model. Kernel splitting for independent JIT compilation. Property-based tests via Hypothesis. Von Mises batch fast path.

## Current Focus

- Documentation overhaul and developer experience improvements
- Expanding real-world examples and use case guides
- Community infrastructure (GitHub Discussions, FAQ)

## Future Directions

These are areas of active interest. Priority depends on community feedback — [open a discussion](https://github.com/sambhal-labs/jaxcross/discussions) or [upvote an issue](https://github.com/sambhal-labs/jaxcross/issues) to signal what matters to you.

### Performance & Scale
- Larger dataset support (10k+ rows) via memory-efficient kernels
- Multi-device inference (data parallelism across GPUs)
- Adaptive sweep scheduling (early stopping when converged)

### New Capabilities
- Additional component models (count data, zero-inflated, etc.)
- Structured prediction queries (conditional generation of multiple columns)
- Causal discovery integration

### Ecosystem
- PyPI package publication
- Python 3.12+ compatibility testing
- Integration examples with pandas, polars, scikit-learn pipelines
- Docker images for reproducible environments

### Research
- Comparison studies against other nonparametric methods
- Scalability benchmarks on public datasets
- Tutorial notebooks for specific domains (healthcare, finance, NLP features)

---

*Last updated: April 2026*
