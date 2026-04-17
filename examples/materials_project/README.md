# Materials Project: Dielectric Constant Discovery

Predict dielectric constants for ~150K Materials Project materials using CrossCat,
avoiding expensive DFPT calculations. Demonstrates structure discovery, calibrated
uncertainty quantification, and Bayesian Model Averaging on a consumer GPU.

## Key Results

- **Ionic dielectric holdout R² = 0.81** with 99% CI coverage of 99.7%
- **49,566 new materials** predicted with high confidence (out of 147K)
- **5 physically meaningful views** discovered consistently across 4 MCMC chains
- Rhat = 1.007 (converged), runtime ~4 hours on GTX 1650

## Prerequisites

- [jaxcross](../../) installed (`uv sync --extra dev`)
- JAX with GPU support (`uv sync --extra gpu`)
- Materials Project API key (set `MP_API_KEY` env var) — only needed for `fetch_mp_data.py`
- Python dependencies: `pymatgen`, `fpdf`, `scikit-learn` (for baselines)

## Pipeline

Run scripts in order from the repo root:

| Step | Script | What it does | Runtime |
|------|--------|-------------|---------|
| 1 | `fetch_mp_data.py` | Fetch ~154K materials via MP REST API | ~10 min |
| 2 | `preprocess_mp_data.py` | Standardize 23 columns, split train/new | ~1 min |
| 3 | `run_local_multichain.py` | 4 MCMC chains × 100 sweeps from checkpoint | ~4 hours (GTX 1650) |
| 4 | `analyze_multichain.py` | Convergence, structure, anomalies, imputation eval | ~10 min |
| 5 | `predict_dielectric.py` | Holdout validation + screening predictions | ~5 min |
| 6 | `impute_dielectric_bma.py` | BMA predictions for ~147K new materials | ~30 min |
| 7 | `baseline_comparison.py` | Compare CrossCat vs MICE vs Random Forest | ~5 min |
| 8 | `generate_pdf.py` | 9-page PDF report with all figures | ~1 min |

```bash
# Example: run the full pipeline
uv run python examples/materials_project/fetch_mp_data.py
uv run python examples/materials_project/preprocess_mp_data.py
uv run python examples/materials_project/run_local_multichain.py
uv run python examples/materials_project/analyze_multichain.py
uv run python examples/materials_project/predict_dielectric.py
uv run python examples/materials_project/impute_dielectric_bma.py
uv run python examples/materials_project/baseline_comparison.py
uv run python examples/materials_project/generate_pdf.py
```

## Outputs

Results are saved to `results/multichain_results/`:

- `dielectric_predictions.csv` — 7,327 training materials with predictions and CI
- `predicted_dielectric_123k.csv` — 49,566 high-confidence new material predictions
- `baseline_comparison.csv` — R²/MAE comparison across methods
- `analysis_summary.json` — convergence diagnostics, imputation metrics, classification results
- `z_matrix.npy` — dependence matrix (23×23) averaged over 4 chains
- 12 PNG figures embedded in the final PDF report

## Column Catalog (23 properties)

| # | Property | Type |
|---|----------|------|
| 0 | Band Gap | Continuous |
| 1 | Is Metal | Binary |
| 2 | Electronic Dielectric | Continuous |
| 3 | Ionic Dielectric | Continuous |
| 4 | Total Dielectric | Continuous |
| 5 | Formation Energy | Continuous |
| 6 | Energy Above Hull | Continuous |
| 7 | Is Stable | Binary |
| 8 | Density | Continuous |
| 9 | Volume | Continuous |
| 10 | N Sites | Continuous |
| 11 | N Elements | Continuous |
| 12 | Crystal System | Categorical (7) |
| 13 | Bulk Modulus | Continuous |
| 14 | Shear Modulus | Continuous |
| 15 | Elastic Anisotropy | Continuous |
| 16 | Poisson Ratio | Continuous |
| 17 | Piezo e_ij_max | Continuous |
| 18 | Avg Electronegativity | Continuous |
| 19 | Avg Ionic Radius | Continuous |
| 20 | Laue Class | Ordinal (11) |
| 21 | Magnetization | Continuous |
| 22 | Magnetic Ordering | Categorical |

## Notebook

`discovery_v2.ipynb` provides an interactive walkthrough of the same pipeline,
suitable for running on Kaggle (2×T4 GPUs with pmap).
