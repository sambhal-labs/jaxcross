# Streamlit Dashboard

An interactive web dashboard for exploring CrossCat models visually — load data, run inference, and query the posterior through a graphical interface.

## Setup

Install the dashboard dependencies:

```bash
uv sync --extra dashboard
# or
pip install "jax-crosscat[dashboard]"
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501` in your browser.

## Pages

The dashboard is organized into 8 pages, each focused on a different aspect of the CrossCat workflow:

### Data Loading

Upload a CSV file or generate synthetic data with known structure. The dashboard auto-detects column types and displays a data preview with type annotations.

- **CSV upload**: Drag and drop or browse for a file
- **Synthetic data**: Configure number of rows, views, clusters, and column types
- **Column type override**: Manually set types if auto-detection is incorrect

### Inference

Run Gibbs sampling with configurable parameters:

- **Number of sweeps**: How many full Gibbs iterations to run
- **Kernel selection**: Choose which transition kernels to apply (row assignments, column assignments, hyperparameters, CRP alphas)
- **Progress tracking**: Real-time sweep counter and log-joint display

### Structure

Inspect the learned model structure:

- **Column partition**: Which columns are grouped into each view
- **Row clusterings**: How rows are clustered within each view
- **Cluster sizes**: Bar chart of row counts per cluster per view

### Dependencies

The Z-matrix heatmap showing pairwise column dependency probabilities. Look for block structure — groups of columns with high mutual dependency (bright squares) indicate discovered views.

### Convergence

Monitor inference quality:

- **Log-joint trace**: Plot of log-joint probability over sweeps — should plateau
- **ARI over sweeps**: If ground truth is available (synthetic data), shows Adjusted Rand Index convergence

### Anomalies

Per-row anomaly scores ranked from most to least unusual. High scores indicate rows that don't fit well into any cluster. Useful for outlier detection and data quality assessment.

### Predictions

Interactive conditional sampling and imputation:

- Select a query column and optional context columns
- Set context values to condition on
- View predicted distributions and sampled values
- Impute missing values with confidence scores

### Similarity

Row similarity matrix as a heatmap. Bright cells indicate pairs of rows that frequently co-cluster across views. Useful for finding similar entities.

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| Port | 8501 | Set via `streamlit run dashboard/app.py --server.port 8080` |
| Max upload size | 200MB | Set via `--server.maxUploadSize 500` |

## Limitations

- The dashboard uses the **unpacked path** for simplicity, so inference is slower than the packed path. Best for small-to-medium datasets (up to ~500 rows, ~50 columns).
- Multi-chain inference is not yet supported in the dashboard — it runs a single chain.
- Large datasets may cause the browser tab to slow down due to heatmap rendering.

## Tips

- **Start with synthetic data** to understand the interface — you can verify that the model recovers known structure
- **Use the Structure page** after inference to see how columns are grouped into views
- **Watch the log-joint trace** on the Convergence page — if it's still climbing, run more sweeps
- **Export results** by copying the state from the dashboard and loading it in a script for packed-path analysis
