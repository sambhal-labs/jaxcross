# Materials Project Structure Discovery

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sambhal-labs/jaxcross/blob/main/examples/materials_project/discovery_v2.ipynb)

An industry-grade use case analyzing ~7,000 materials from the [Materials Project](https://materialsproject.org/) database and predicting dielectric constants for ~150,000 candidate materials. Every ML paper on Materials Project data predicts one property at a time (band gap, bulk modulus, dielectric constant) using supervised learning. This example demonstrates CrossCat's unique capability: **unsupervised joint structure discovery** across all material properties simultaneously — with calibrated uncertainty and zero feature engineering.

!!! info "Scripts & Notebook"
    The full pipeline is at [`examples/materials_project/`](https://github.com/sambhal-labs/jaxcross/tree/main/examples/materials_project) — 8 standalone scripts plus an interactive notebook. Run on a GTX 1650 (4GB VRAM) or Kaggle (T4/P100) for GPU acceleration. Requires a free [Materials Project API key](https://materialsproject.org/api).

## Results at a Glance

| Metric | Value |
|--------|-------|
| Training materials | 7,327 (with DFPT dielectric data) |
| New materials predicted | 49,566 high-confidence (of 147K total) |
| Ionic dielectric holdout R² | 0.81 |
| 90% CI calibration | 95.6% coverage (conservative) |
| 99% CI calibration | 99.7% coverage |
| Structure discovered | 5 physically meaningful views |
| Convergence (Rhat) | 1.007 |
| Compute | 4 chains x 100 sweeps, ~4 hours on GTX 1650 |

## Dataset

~7,000 materials with dielectric data (v2025.09.25), enriched with elasticity, piezoelectric, and summary properties. The dataset has natural sparsity: elasticity data is available for only ~28% of the dielectric subset — ideal for CrossCat's native NaN handling.

**Mixed column types (23 columns):**

- **CONTINUOUS (18):** band gap, formation energy, energy above hull, density, volume, nsites, nelements, dielectric constants (total, ionic, electronic), bulk modulus, shear modulus, elastic anisotropy, Poisson ratio, piezo e_ij_max, avg electronegativity, avg ionic radius, magnetization
- **BINARY (2):** is_stable, is_metal
- **CATEGORICAL (2):** crystal system (7 values), magnetic ordering
- **ORDINAL (1):** Laue class (11 values)

```python
from mp_api.client import MPRester
from emmet.core.summary import HasProps

with MPRester(API_KEY) as mpr:
    # Fetch all materials with dielectric data
    docs = mpr.materials.summary.search(has_props=[HasProps.dielectric])
    mpids = [str(doc.material_id) for doc in docs]

    # Enrich with elasticity and piezoelectric (sparse subsets)
    elastic_docs = mpr.materials.elasticity.search(material_ids=mpids)
    piezo_docs = mpr.materials.piezoelectric.search(material_ids=mpids)
```

Data is cached to Parquet after the first API call, so subsequent runs are instant. Preprocessing applies crystal system encoding, Laue class mapping, log transforms for high-dynamic-range columns, and IQR clamping.

## Pipeline Overview

The example is organized as 8 standalone scripts, each doing one step:

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `fetch_mp_data.py` | Fetch ~154K materials via Materials Project REST API |
| 2 | `preprocess_mp_data.py` | Standardize 23 columns, split train (7K) / new (147K) |
| 3 | `run_local_multichain.py` | 4 MCMC chains x 100 sweeps from checkpoint |
| 4 | `analyze_multichain.py` | Convergence, structure, anomalies, imputation eval |
| 5 | `predict_dielectric.py` | Holdout validation + dielectric screening predictions |
| 6 | `impute_dielectric_bma.py` | Bayesian Model Averaging for 147K new materials |
| 7 | `baseline_comparison.py` | Compare CrossCat vs MICE vs Random Forest |
| 8 | `generate_pdf.py` | 9-page PDF report with all figures |

## Multi-Chain Inference

The pipeline runs 4 independent MCMC chains from a sweep-300 checkpoint, using `packed_gibbs_step` for memory efficiency on consumer GPUs.

!!! note "Why `packed_gibbs_step` instead of `packed_gibbs_sweep`?"
    On GPUs with 4GB VRAM, the `lax.scan`-based `packed_gibbs_sweep` may exceed memory during JIT compilation for 23 columns. `packed_gibbs_step` calls 4 smaller `@jax.jit` sub-kernels independently, fitting within memory constraints while maintaining the same mathematical behavior.

```python
from crosscat import packed_log_joint
from crosscat.packed import packed_gibbs_step
from crosscat.serialization import load_packed_state, save_packed_state

N_CHAINS = 4
N_SWEEPS = 100

for chain_idx in range(N_CHAINS):
    packed, col_types = load_packed_state(checkpoint_path)
    key = jax.random.key(42 + chain_idx * 1000)

    for sweep in range(N_SWEEPS):
        key, subkey = jax.random.split(key)
        packed = packed_gibbs_step(subkey, packed, data_jax)

        if (sweep + 1) % 10 == 0:
            lj = float(packed_log_joint(packed, data_jax))
            print(f"Chain {chain_idx} sweep {sweep+1}: log_joint={lj:,.1f}")

    save_packed_state(packed, f"chain_{chain_idx}.jxc", column_types=col_types)
```

## Convergence Diagnostics

With 4 chains, standard convergence diagnostics confirm mixing:

```python
from crosscat.diagnostics import gelman_rubin_rhat, effective_sample_size

traces_jax = jnp.array(log_joint_traces)  # (4, 10) — 4 chains x 10 diag points
rhat = float(gelman_rubin_rhat(traces_jax))   # 1.007 — converged
ess = float(effective_sample_size(traces_jax))  # ~100-200 per chain
```

!!! note "Rhat for structure-learning models"
    CrossCat's partition space is combinatorial (Bell number B(23) ~ 4.6 x 10^18 possible view structures). Different chains legitimately settle into different posterior modes, so Rhat — designed for unimodal targets — may stay elevated. The key diagnostic is that per-chain log-joint traces stabilize. We select the best chain for row-level queries and average over all chains for dependence structure.

## Dependence Structure Discovery (Z-Matrix)

The flagship result — which material properties are jointly dependent?

```python
from crosscat import packed_dependence_matrix

z_matrix = np.array(packed_dependence_matrix(all_chains))
```

The Z-matrix reveals 5 physically meaningful property groupings, consistent across all 4 chains:

| View | Columns | Interpretation |
|------|---------|----------------|
| 0 | Band gap, formation energy, E above hull, is_stable, density, volume, nsites, nelements, crystal system, electronegativity, ionic radius, magnetization | Structural / thermodynamic |
| 1 | Electronic dielectric, bulk modulus, shear modulus, Poisson ratio, Laue class, is_metal, ordering | Electronic / mechanical |
| 2 | Ionic dielectric, total dielectric | Dielectric pair |
| 3 | Piezo e_ij_max | Piezoelectric (singleton) |
| 4 | Elastic anisotropy | Elastic anisotropy (singleton) |

Key findings:
- Ionic and total dielectric constants form their own view — they share information that no other property captures
- Electronic dielectric clusters with mechanical properties (bulk/shear modulus), suggesting shared electronic structure origins
- Piezoelectric and elastic anisotropy are singletons — they don't share dependency structure with other measured properties

## Holdout Validation

10% of observed values are masked and imputed to validate prediction quality:

```python
from crosscat import batch_impute_column, batch_credible_interval

key, k1, k2 = jax.random.split(key, 3)

# Point predictions
predictions, confidence = batch_impute_column(
    k1, best_packed, data_jax,
    query_col=ionic_dielectric_col,
    row_ids=jnp.array(holdout_rows),
)

# Credible intervals
medians, ci_lo, ci_hi = batch_credible_interval(
    k2, best_packed, data_jax,
    query_col=ionic_dielectric_col,
    row_ids=jnp.array(holdout_rows),
    ci_level=0.90,
)
```

**Per-column holdout results (best chain):**

| Column | MAE | RMSE | R² | N holdout |
|--------|-----|------|----|-----------|
| Band Gap | 0.33 | 0.52 | 0.63 | 732 |
| Ionic Dielectric | 0.33 | 0.58 | 0.81 | 732 |
| Formation Energy | 0.28 | 0.42 | 0.74 | 732 |
| Bulk Modulus | 0.41 | 0.57 | 0.52 | ~200 |
| Shear Modulus | 0.39 | 0.56 | 0.47 | ~200 |
| Avg Electronegativity | 0.12 | 0.18 | 0.72 | 732 |

**Calibration:** The credible intervals are well-calibrated and slightly conservative — desirable for materials screening where false positives waste expensive compute:

| CI Level | Expected Coverage | Actual Coverage |
|----------|------------------|-----------------|
| 90% | 90% | 95.6% |
| 95% | 95% | 98.4% |
| 99% | 99% | 99.7% |

## Baseline Comparison

The same 10% holdout is used to compare CrossCat against two standard imputation methods:

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# MICE (multiple imputation by chained equations)
mice = IterativeImputer(max_iter=10, random_state=42)
mice_imputed = mice.fit_transform(data_with_holdout_masked)

# Random Forest (per-column supervised model)
for col in test_cols:
    rf = RandomForestRegressor(n_estimators=100, max_depth=20)
    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_holdout)
```

**R² comparison on ionic dielectric:**

| Method | R² | Notes |
|--------|-----|-------|
| **Random Forest** | 0.92 | Requires curated features, separate model per target |
| **CrossCat** | 0.81 | No feature engineering, structure discovery, calibrated CI |
| **MICE** | 0.48 | Assumes linear relationships |

CrossCat reaches 88% of Random Forest accuracy while providing capabilities RF cannot: structure discovery, calibrated uncertainty, mixed-type native handling, anomaly detection, and mutual information — all from a single model.

## Bayesian Model Averaging for New Materials

The headline practical result: predicting dielectric constants for 147,552 materials that lack DFPT data.

```python
from crosscat import batch_impute_column, packed_insert_rows

# For each of 4 chains:
for chain in all_chains:
    # Insert new materials in batches (5000 rows at a time)
    packed_aug, data_aug = packed_insert_rows(
        rng_key, chain, train_data, new_materials_batch
    )

    # Predict ionic dielectric in batches (500 rows at a time)
    predictions, confidence = batch_impute_column(
        key, packed_aug, data_aug,
        query_col=ionic_dielectric_col,
        row_ids=jnp.array(new_row_ids),
    )
    per_chain_predictions.append(predictions)

# Bayesian Model Averaging across 4 chains
bma_mean = np.mean(per_chain_predictions, axis=0)
bma_std = np.std(per_chain_predictions, axis=0)
confidence = 1.0 / (1.0 + bma_std)
```

**BMA results:**

- 147,552 total predictions generated
- **49,566 pass high-confidence filter** (99% CI relative precision < 1.0)
- Mean cross-chain std: 0.27 (chains agree closely)
- Mean confidence score: 0.795
- Predicted ionic dielectric range: 4.8 - 80.6

## Anomaly Detection

Score all materials for typicality to identify unusual property combinations:

```python
from crosscat import batch_anomaly_score, batch_row_typicality

# Anomaly scores (lower = more anomalous)
anom_scores = np.array(
    batch_anomaly_score(best_packed, data_jax, jnp.arange(n_rows))
)

# Row typicality (multi-chain average)
typ_scores = np.array(batch_row_typicality(all_chains, jnp.arange(n_rows)))
```

Materials with unusual property combinations (e.g., high band gap but metallic behavior, or extreme dielectric with low density) surface as anomalies. Per-column drilldown identifies *which* specific properties are surprising.

## Mutual Information

Quantify nonlinear relationships between material property pairs:

```python
from crosscat import packed_mutual_information

mi_val, mi_std = packed_mutual_information(
    all_chains, column_types,
    col_i=bulk_modulus_col, col_j=shear_modulus_col,
    rng_key=jax.random.key(99),
)
linfoot = float(np.sqrt(1 - np.exp(-2 * float(mi_val))))
```

Selected pairs (Linfoot correlation, 0-1 scale):

| Pair | Linfoot |
|------|---------|
| Bulk Modulus <-> Shear Modulus | 0.94 |
| Band Gap <-> Is Metal | 0.85 |
| Formation Energy <-> E Above Hull | 0.78 |
| Crystal System <-> Laue Class | 0.72 |
| Band Gap <-> Electronic Dielectric | 0.65 |

## Metallicity Classification

CrossCat classifies metals vs non-metals from the joint model (no separate classifier):

```python
from crosscat import batch_classify_column

log_probs = np.array(batch_classify_column(
    best_packed, data_jax,
    target_col=is_metal_col,
    candidate_vals=jnp.array([0.0, 1.0]),
    row_ids=jnp.arange(n_rows),
))
```

**Results:** F1 = 0.85, Accuracy = 0.87 (threshold optimized over grid search).

## Key Takeaways

- **Joint structure discovery** reveals physically meaningful property groupings that no supervised approach can provide — 5 views emerge consistently across 4 independent chains.
- **DFPT dielectric screening** is the headline practical result: R²=0.81 for ionic dielectric with well-calibrated credible intervals. CrossCat predicts missing dielectric constants from structural and compositional features, saving 5-10x the DFPT compute cost.
- **Bayesian Model Averaging** across 4 chains provides robust uncertainty estimates. The cross-chain standard deviation serves as a practical quality filter, identifying 49,566 trustworthy predictions out of 147K candidates.
- **Native NaN handling** makes CrossCat ideal for materials databases with natural sparsity — no need to drop incomplete rows or impute before modeling.
- **Consumer GPU feasible** — the entire pipeline runs on a GTX 1650 (4GB VRAM) in ~4 hours using `packed_gibbs_step` for memory-efficient inference.
- **Single model, many queries** — structure discovery, imputation, anomaly detection, classification, and mutual information all come from the same trained CrossCat model.
