# Materials Project Structure Discovery

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sambhal-labs/jaxcross/blob/main/examples/materials_project_discovery.ipynb)

An industry-grade use case analyzing ~7,000 materials from the [Materials Project](https://materialsproject.org/) database. Every ML paper on Materials Project data predicts one property at a time (band gap, bulk modulus, dielectric constant) using supervised learning. This notebook demonstrates CrossCat's unique capability: **unsupervised joint structure discovery** across all material properties simultaneously — without labels.

!!! info "Notebook"
    The full example is at [`examples/materials_project_discovery.ipynb`](https://github.com/sambhal-labs/jaxcross/blob/main/examples/materials_project_discovery.ipynb). Run it on Kaggle (T4 or P100) for GPU acceleration. Requires a free [Materials Project API key](https://materialsproject.org/api).

## Dataset

~7,000 materials with dielectric data (v2025.09.25), enriched with elasticity, piezoelectric, and summary properties. The dataset has natural sparsity: elasticity data is available for only ~28% of the dielectric subset — ideal for CrossCat's native NaN handling.

**Mixed column types (20 columns):**

- **CONTINUOUS (16):** band gap, formation energy, energy above hull, density, volume, nsites, dielectric constants (total, ionic, electronic), refractive index, bulk modulus, shear modulus, elastic anisotropy, Poisson ratio, piezo e_ij_max, magnetization
- **BINARY (2):** is_stable, is_metal
- **CATEGORICAL (2):** crystal system (7 values), magnetic ordering

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

Data is cached to Parquet after the first API call, so subsequent runs are instant.

## Multi-Chain Inference

The notebook runs multi-chain packed Gibbs sampling with convergence monitoring:

```python
from crosscat import (
    initialize, pack_state, select_best_chain,
    multi_chain_packed_gibbs_sweep, unbatch_packed_states,
)
from crosscat.diagnostics import gelman_rubin_rhat

key = jax.random.key(42)
result = initialize(key, data_jax, column_types, n_chains=4)
chains = [pack_state(s, max_views=16, max_clusters=32, data=data_jax)
          for s in result.state]

# Run in diagnostic batches with checkpointing
for sweep in range(0, N_SWEEPS, DIAG_EVERY):
    key, subkey = jax.random.split(key)
    batched, scores = multi_chain_packed_gibbs_sweep(
        subkey, chains, data_jax, n_sweeps=DIAG_EVERY
    )
    chains = unbatch_packed_states(batched, N_CHAINS)

    # Convergence check
    rhat = float(gelman_rubin_rhat(jnp.array(log_joint_traces)))

best = select_best_chain(batched, scores)
all_chains = unbatch_packed_states(batched, N_CHAINS)
```

## Dependence Structure Discovery (Z-Matrix)

The flagship result — which material properties are jointly dependent?

```python
from crosscat import packed_dependence_matrix

z_matrix = np.array(packed_dependence_matrix(all_chains))
```

The Z-matrix reveals physically meaningful groupings: electronic properties (band gap, dielectric constants, refractive index) cluster together, mechanical properties (bulk modulus, shear modulus) form their own group, and thermodynamic stability indicators (formation energy, energy above hull, is_stable) are jointly modeled.

## Anomaly Detection

Score all materials for typicality in a single vectorized call:

```python
from crosscat import batch_row_typicality

typicality = np.array(batch_row_typicality(best_packed, jnp.arange(n_rows)))
```

Materials with unusual property combinations (e.g., high band gap but metallic behavior) surface as anomalies. Per-column drilldown using `packed_predictive_probability` identifies *which* specific properties are surprising for each anomalous material.

## Missing Property Imputation

The headline practical result: predict missing mechanical properties from electronic and structural data. DFT elasticity calculations are computationally expensive — CrossCat can fill gaps using discovered dependency structure.

```python
from crosscat import batch_impute_column

key, subkey = jax.random.split(key)
values, confidences = batch_impute_column(
    subkey, best_packed, data_jax,
    query_col=bulk_modulus_col,
    row_ids=jnp.array(missing_rows),
)
```

Quality is validated with a 10% holdout: mask observed values, impute, and measure MAE/RMSE/R² per column. The notebook reports per-column metrics and parity plots (true vs. predicted).

## Mutual Information

Quantify nonlinear relationships between material property pairs:

```python
from crosscat import packed_mutual_information

mi = packed_mutual_information(all_chains, col_i=band_gap_col, col_j=dielectric_col)
linfoot = float(np.sqrt(1 - np.exp(-2 * float(mi))))
```

The Linfoot correlation (normalized MI, 0–1 scale) captures nonlinear relationships that Pearson correlation misses — critical for materials data where property relationships are often highly nonlinear.

## Generative Classification

CrossCat predicts metallicity without a dedicated classifier:

```python
from crosscat import batch_classify_column

predictions = batch_classify_column(
    key, best_packed, data_jax,
    query_col=is_metal_col,
    row_ids=jnp.arange(n_rows),
)
```

This uses the full posterior predictive P(is_metal | all other properties), averaging over all discovered structures via Bayesian model averaging.

## Key Takeaways

- **Joint structure discovery** reveals physically meaningful property groupings that no supervised approach can provide.
- **Native NaN handling** makes CrossCat ideal for materials databases with natural sparsity — no need to drop incomplete rows or impute before modeling.
- **Imputation from structure** can predict expensive-to-compute mechanical properties from cheaper electronic/structural data, potentially accelerating materials screening.
- **Mixed column types** (continuous, binary, categorical) are handled natively within a single model — no separate preprocessing pipelines needed.
- **Anomaly detection** identifies materials with unusual property combinations for further experimental investigation.
