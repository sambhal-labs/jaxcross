# Feature Guides

Step-by-step guides for every feature in jax-crosscat. Each guide covers what the feature does, when to use it, and includes complete code examples.

## Getting Data In

- **[Data Loading & Column Types](data-loading.md)** — Load CSV files, detect column types, handle missing values
- **[Model Initialization](initialization.md)** — Single and multi-chain initialization, choosing modes

## Running Inference

- **[Running Inference](inference.md)** — Gibbs sweep configuration, kernel selection, convergence
- **[GPU Acceleration (Packed Path)](gpu-packed.md)** — JIT-compiled packed state for 10-100x speedup
- **[Multi-Chain Inference](multi-chain.md)** — Parallel chains for robust results
- **[XLA Compilation Caching](xla-cache.md)** — Skip recompilation across sessions

## Querying the Posterior

- **[Conditional Sampling](queries/sampling.md)** — Predictive sampling, credible intervals
- **[Anomaly Detection](queries/anomaly-detection.md)** — Row anomaly scores, structural typicality
- **[Dependence Discovery](queries/dependence.md)** — Z-matrix, pairwise dependence probabilities
- **[Imputation](queries/imputation.md)** — Missing value prediction with confidence
- **[Mutual Information](queries/mutual-information.md)** — Information-theoretic dependency strength
- **[Row Similarity](queries/row-similarity.md)** — Co-clustering probability between rows
- **[Predictive CDF & Probability](queries/predictive-probability.md)** — Probability evaluation, CDF

## Advanced Features

- **[Constraint Enforcement](constraints.md)** — Force column/row dependencies
- **[Missing Data Handling](missing-data.md)** — NaN transparency throughout the pipeline
- **[Serialization & Checkpointing](serialization.md)** — Save, load, and checkpoint models
- **[Convergence Diagnostics](diagnostics.md)** — ARI, log-joint tracking, held-out evaluation
- **[Online Learning (Row Insertion)](online-learning.md)** — Add new data without re-inference
- **[Streamlit Dashboard](dashboard.md)** — Interactive visual exploration
