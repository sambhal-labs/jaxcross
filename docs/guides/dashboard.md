# Streamlit Dashboard

## What

An interactive web dashboard for exploring CrossCat models visually — load data, run inference, and query the posterior through a graphical interface.

## Setup

```bash
uv sync --extra dashboard
streamlit run dashboard/app.py
```

## Pages

| Page | Description |
|------|-------------|
| **Data Loading** | Upload CSV or generate synthetic data |
| **Inference** | Run Gibbs sweeps with configurable kernels |
| **Structure** | Inspect column partition and row clusterings |
| **Dependencies** | Z-matrix heatmap (column dependency probabilities) |
| **Convergence** | Log-joint trace, ARI over sweeps |
| **Anomalies** | Per-row anomaly scores |
| **Predictions** | Conditional sampling and imputation |
| **Similarity** | Row similarity matrix |

## Tips

- Start with synthetic data to understand the interface
- Use the "Structure" page to see how columns are grouped into views
- The "Dependencies" page shows the Z-matrix heatmap — look for block structure
- The dashboard uses the unpacked path, so it's best for small-medium datasets
