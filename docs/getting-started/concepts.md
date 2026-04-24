# Core Concepts

This page explains the key ideas behind CrossCat to help you understand what the library is doing and why.

## What is Cross-Categorization?

CrossCat is a Bayesian model for **heterogeneous tabular data**. Given a table of observations (rows) and features (columns), CrossCat simultaneously discovers:

1. **Which columns are related** — grouping columns into "views"
2. **How rows cluster within each group** — independently clustering rows per view

This is different from standard clustering, which forces a single grouping over all features.

## Views and Clusters

### Views (Column Groups)

A **view** is a group of columns that CrossCat has determined are statistically dependent. Columns in different views are treated as independent.

For example, in an employee dataset:

- **View 0**: `{salary, experience, title}` — compensation-related columns
- **View 1**: `{zip_code, commute_distance}` — geography-related columns

### Clusters (Row Groups within a View)

Within each view, rows are independently clustered. The same employee can belong to different clusters in different views:

- In View 0: Employee 42 is in Cluster 0 ("Senior tier")
- In View 1: Employee 42 is in Cluster 2 ("Suburban commuters")

This flexibility is what makes CrossCat powerful — it doesn't force a single explanation for all the data.

## The Two-Level Dirichlet Process

CrossCat uses a **two-level Dirichlet Process (DP)** mixture model:

1. **Outer DP** — A Chinese Restaurant Process (CRP) assigns columns to views. The number of views is inferred automatically.
2. **Inner DP** — Within each view, another CRP assigns rows to clusters. The number of clusters per view is also inferred.

The CRP is a distribution over partitions that favors joining existing groups but always allows creating new ones. The concentration parameter `alpha` controls how likely new groups are.

## Column Types

Each column must have a declared type, which determines the statistical model used:

| Type | Values | Statistical Model | Use Cases |
|------|--------|-------------------|-----------|
| `CONTINUOUS` | Any real number | Normal-Gamma (conjugate) | Salary, temperature, height |
| `CATEGORICAL` | Non-negative integers (0, 1, 2, ...) | Dirichlet-Categorical | Department, color, country |
| `BINARY` | 0 or 1 | Beta-Bernoulli | Yes/no flags, presence/absence |
| `ORDINAL` | Ordered integers (1, 2, 3, ...) | Ordered Logistic (cumulative link) | Ratings, education level |
| `CYCLIC` | Floats in [0, 2*pi) | Von Mises | Wind direction, time of day |

## Collapsed Inference

CrossCat uses **collapsed Gibbs sampling**: all component model parameters (means, variances, etc.) are analytically integrated out. The model only stores and samples:

- **Cluster assignments** — which cluster each row belongs to (per view)
- **Column assignments** — which view each column belongs to
- **Hyperparameters** — a few scalars per column (sampled via grid Gibbs)

This means:

- **No parameter tuning** — you don't set learning rates, priors, or model complexity
- **Automatic complexity** — the number of views and clusters is inferred from data
- **Efficient storage** — only sufficient statistics (count, sum, sum-of-squares) are stored per cluster

## Packed vs Unpacked State

The library offers two representations of model state:

### Unpacked (`CrossCatState`)

- Python lists and variable-length arrays
- Easy to inspect and debug
- Cannot be JIT-compiled by JAX
- Used for: queries, diagnostics, inspection

### Packed (`PackedCrossCatState`)

- Fixed-size padded JAX arrays
- Fully JIT-compatible — enables `jax.jit`, `jax.vmap`, `jax.lax.scan`
- 10-100x faster inference on GPU
- Used for: running Gibbs sweeps, GPU-accelerated inference

**Typical workflow:** Initialize unpacked → pack → run inference → unpack → query.

```python
result = initialize(key, data, col_types)       # unpacked
state = result.state
packed = pack_state(state)                       # pack
packed = packed_gibbs_sweep(key, packed, data)   # fast inference
state = unpack_state(packed, col_types, data=data)  # unpack for queries
```

## What Can You Ask?

After running inference, CrossCat supports a rich set of posterior queries:

| Query | Question it answers |
|-------|-------------------|
| **Predictive sampling** | "What values would we expect for column X given Y=y?" |
| **Anomaly detection** | "Is this row unusual compared to the rest of the data?" |
| **Dependence discovery** | "Which columns are statistically related?" |
| **Mutual information** | "How much information does column X carry about column Y?" |
| **Imputation** | "What's the best guess for this missing value, and how confident?" |
| **Row similarity** | "How similar are these two rows?" |
| **Predictive CDF** | "What's P(X <= value)?" |
| **Credible intervals** | "What's the 90% credible interval for column X?" |

All queries are Bayesian — they account for uncertainty in cluster assignments, not just point estimates.

## When to Use CrossCat (and When Not To)

**Good fit**

- You have a **mixed-type tabular dataset** (continuous, categorical, ordinal, binary, cyclic) and want a single model that handles all columns uniformly.
- Missing values are **structurally meaningful** — you want the model to impute them rather than dropping rows.
- You want to **discover structure** — which columns relate, how rows cluster — rather than predict one fixed target.
- You need **calibrated Bayesian uncertainty** on queries (anomaly scores, imputations, predictions), not just point predictions.
- Your dataset is **small-to-medium** (hundreds to low millions of rows, tens to hundreds of columns). See the [Scaling guide](../guides/scaling.md) for the 10K+ workflow.

**Not a good fit**

- You have a **single well-defined prediction target** with abundant labeled data — a supervised model (gradient-boosted trees, linear/logistic regression, neural net) will outperform CrossCat on pure predictive accuracy.
- Your features are **images, audio, or free text** — CrossCat is a tabular model. Extract features first or use a domain-specific model.
- You need **sub-millisecond inference latency** — posterior queries involve MC sampling and multi-chain averaging, not a single forward pass.
- Your data is **extremely wide** (tens of thousands of columns) without meaningful type-level structure — the column-assignment Gibbs step scales with column count.

## Further Reading

- [**Quickstart**](quickstart.md) — Run your first model in 60 seconds.
- [**Glossary**](../glossary.md) — Formal definitions for *view*, *cluster*, *CRP*, *Z-matrix*, *component model*.
- [**Architecture → Overview**](../architecture/overview.md) — How the pieces fit together.
- [**Architecture → Algorithms**](../architecture/algorithms/row-gibbs.md) — Math and code for each Gibbs kernel.
- [**Original CrossCat paper**](https://jmlr.org/papers/v17/11-392.html) — Mansinghka et al., JMLR 2016.
