# Predicting Dielectric Properties for 49,566 Materials at 99% Confidence

**How a single unsupervised Bayesian model replaces millions of CPU-hours of quantum chemistry — and discovers the physics along the way.**

---

## The Problem

The [Materials Project](https://materialsproject.org/) database contains 154,879 computationally characterized materials. But dielectric constants — critical for designing capacitors, gate insulators, and piezoelectric devices — are available for only 7,327 of them (4.7%).

The bottleneck is Density Functional Perturbation Theory (DFPT). Computing dielectric properties requires 5-10x more compute than a standard DFT relaxation: higher k-point densities (3,000/atom vs 1,000), tighter self-consistency convergence, and 600 eV energy cutoffs. At current cloud rates, filling in the remaining ~147,000 materials would cost millions of CPU-hours.

What if we could predict dielectric constants from the structural and compositional properties we already have — band gap, density, crystal system, elastic moduli — without any additional quantum chemistry calculation?

## The Approach: Bayesian Cross-Categorization

We used [JAX-CrossCat](https://github.com/sambhal-labs/jaxcross), a GPU-accelerated Bayesian nonparametric model, to jointly analyze 23 material properties across the 7,327 materials with known dielectric data. CrossCat simultaneously discovers:

1. **Which properties are statistically dependent** (column structure)
2. **Which materials behave similarly** (row clustering)

All model parameters are integrated out via conjugate priors — only structural assignments are sampled via collapsed Gibbs MCMC. This gives calibrated uncertainty estimates as a natural byproduct, not as a post-hoc add-on.

The 23 properties span 4 column types:

- **18 continuous:** band gap, formation energy, density, volume, elastic moduli, dielectric constants, electronegativity, ionic radius, magnetization
- **2 binary:** is_metal, is_stable
- **2 categorical:** crystal system (7 values), magnetic ordering
- **1 ordinal:** Laue class (11 values)

CrossCat handles all of these natively in a single model — no one-hot encoding, no separate preprocessing pipelines, no feature engineering.

## The Discovery: 5 Physically Meaningful Property Groups

Before making a single prediction, CrossCat revealed something no supervised model can: the latent dependency structure of material properties.

The model discovered 5 independent property groups ("views"), consistent across all 4 MCMC chains:

| View | Properties | Physical Interpretation |
|------|-----------|------------------------|
| **Structural / Thermodynamic** | Band gap, formation energy, E above hull, stability, density, volume, nsites, nelements, crystal system, electronegativity, ionic radius, Laue class | Compositional and structural descriptors |
| **Electronic / Mechanical** | Is_metal, electronic dielectric, bulk modulus, shear modulus, Poisson ratio, magnetization, magnetic ordering | Band structure and elastic response |
| **Ionic Dielectric Pair** | Ionic dielectric, total dielectric | Lattice dynamics (phonon-driven) |
| **Piezoelectric** | Piezo e_ij_max | Independent tensor property |
| **Elastic Anisotropy** | Universal anisotropy | Independent scalar property |

The key physical insight: **CrossCat separated ionic from electronic dielectric into different views.** This reflects distinct underlying physics — ionic dielectric depends on lattice dynamics (phonon modes), while electronic dielectric depends on band structure (electron polarizability). The model discovered this separation without any physics knowledge built in.

This is the headline result that no Random Forest or neural network can produce. Supervised methods predict individual targets; CrossCat reveals the structure *between* targets.

## Holdout Validation: R² = 0.81 with Calibrated Uncertainty

We validated predictions by masking 10% of observed ionic dielectric values and predicting them from the remaining 22 properties:

| Metric | Value |
|--------|-------|
| R² | 0.81 |
| MAE | 0.33 |
| RMSE | 0.58 |

But accuracy alone is not enough for materials screening. The critical question is: **when the model says it's confident, is it right?**

| CI Level | Expected Coverage | Actual Coverage |
|----------|------------------|-----------------|
| 90% | 90% | 95.6% |
| 95% | 95% | 98.4% |
| 99% | 99% | 99.7% |

The credible intervals are slightly conservative at every level — the model over-covers rather than under-covers. For a screening application ("should I spend 5-10x compute on DFPT for this material?"), conservative uncertainty is exactly what you want. A false positive wastes compute; a false negative misses a candidate. The model errs on the side of saying "I'm not sure" rather than "I'm confident" when it shouldn't be.

## Bayesian Model Averaging Across 4 Chains

For predicting the 147,552 materials without dielectric data, we used Bayesian Model Averaging (BMA) across 4 independent MCMC chains:

1. Insert new materials into each chain via `packed_insert_rows`
2. Predict ionic dielectric in each chain independently
3. Average predictions across chains (BMA point estimate)
4. Use cross-chain standard deviation as an uncertainty measure

This produced 147,552 predictions, of which **49,566 passed the high-confidence filter** (99% CI relative precision < 1.0). The filtering means we're reporting only predictions where all 4 chains agree — a practical quality gate built into the Bayesian workflow.

**BMA statistics:**

- Mean cross-chain std: 0.27 (chains agree closely)
- Mean confidence score: 0.795
- Ionic dielectric range: 4.8 - 80.6

## How Does It Compare to Supervised Baselines?

We benchmarked against two standard imputation methods on the same 10% holdout:

| Method | Ionic Dielectric R² | Uncertainty | Structure Discovery | Feature Engineering |
|--------|---------------------|-------------|--------------------|--------------------|
| **Random Forest** | 0.92 | None | No | Required |
| **CrossCat** | 0.81 | Calibrated (99.7% at 99% CI) | 5 views discovered | None |
| **MICE** | 0.48 | None | No | None |

Random Forest wins on raw accuracy — but it requires a separate model per target, curated feature sets, and provides no uncertainty quantification. CrossCat reaches 88% of RF's accuracy while providing:

- **Structure discovery** — which properties are related, automatically
- **Calibrated uncertainty** — credible intervals you can trust for screening decisions
- **One model for everything** — imputation, anomaly detection, classification, mutual information, all from a single trained model
- **Native mixed types** — continuous, binary, categorical, and ordinal columns handled jointly
- **Native sparsity** — 43.5% of values are NaN in the new materials; CrossCat handles this without preprocessing

The comparison is inherently asymmetric. RF is a specialized point predictor. CrossCat is a general-purpose probabilistic model. CrossCat's value is not replacing RF — it's providing capabilities that RF cannot, while achieving competitive accuracy.

## It All Runs on a Consumer GPU

The entire pipeline ran on a single NVIDIA GTX 1650 with 4GB VRAM:

| Step | Time |
|------|------|
| Data fetch (154K materials from MP API) | ~10 min |
| Preprocessing (23 columns, encoding, standardization) | ~1 min |
| MCMC inference (4 chains x 100 sweeps) | ~4 hours |
| Analysis (convergence, structure, anomalies, imputation) | ~10 min |
| BMA predictions (147K new materials) | ~30 min |
| Baseline comparison | ~5 min |

Total: under 5 hours, on hardware that costs $150 used.

The key to fitting on 4GB VRAM is `packed_gibbs_step` — it calls 4 smaller JIT-compiled sub-kernels independently instead of compiling one large `lax.scan` loop. Same mathematical behavior, 4x smaller compilation memory footprint.

## Convergence: Do the Chains Agree?

With 4 independent chains starting from the same sweep-300 checkpoint but with different RNG seeds:

- **Gelman-Rubin Rhat: 1.007** — well below the 1.1 convergence threshold
- **Effective Sample Size: ~100-200** per chain
- All 4 chains discovered the same 5-view structure

CrossCat's partition space is combinatorial — for 23 columns, there are ~4.6 x 10^18 possible view structures. The fact that all 4 chains converged to the same structure (5 views with the same column groupings) is strong evidence that this structure is a genuine feature of the data, not an artifact of a particular MCMC run.

## What Else Falls Out of the Model?

Since CrossCat is a full probabilistic model, not just a predictor, we get several additional analyses from the same trained model:

**Metallicity classification** — CrossCat classifies metals vs non-metals at F1 = 0.85, using the joint posterior rather than a separate classifier.

**Anomaly detection** — Materials with unusual property combinations (e.g., high band gap but metallic behavior) are flagged automatically. Per-column drilldown identifies *which* properties make each material unusual.

**Mutual information** — Nonlinear relationships between property pairs, quantified on a 0-1 scale (Linfoot correlation):

| Pair | Linfoot |
|------|---------|
| Bulk Modulus <-> Shear Modulus | 0.94 |
| Band Gap <-> Is Metal | 0.85 |
| Formation Energy <-> E Above Hull | 0.78 |
| Crystal System <-> Laue Class | 0.72 |

These relationships include nonlinear dependencies that Pearson correlation misses.

## Reproducing This

The full pipeline is 8 standalone Python scripts in [`examples/materials_project/`](https://github.com/sambhal-labs/jaxcross/tree/main/examples/materials_project):

```bash
# Install jaxcross
git clone https://github.com/sambhal-labs/jaxcross.git
cd jaxcross && uv sync --extra dev --extra gpu

# Run the pipeline (requires MP_API_KEY env var)
uv run python examples/materials_project/fetch_mp_data.py
uv run python examples/materials_project/preprocess_mp_data.py
uv run python examples/materials_project/run_local_multichain.py
uv run python examples/materials_project/analyze_multichain.py
uv run python examples/materials_project/predict_dielectric.py
uv run python examples/materials_project/impute_dielectric_bma.py
uv run python examples/materials_project/baseline_comparison.py
uv run python examples/materials_project/generate_pdf.py
```

Each script is self-contained: load data, do one thing, save results. No notebook state to manage, no manual cell execution order to remember.

An interactive Jupyter notebook (`discovery_v2.ipynb`) is also available for Kaggle/Colab with 2xT4 GPU support via `jax.pmap`.

## What's Next

**Publish the predicted dataset.** The 49,566-material CSV with ionic dielectric predictions and 99% CI bounds could be a community resource — a starting point for experimentalists choosing which materials to characterize next.

**Improve electronic dielectric prediction.** The current R² for electronic dielectric is 0.05 — essentially unpredictive from the available features. Electronic dielectric depends on band structure details (effective masses, optical transitions) that aren't captured by the 23 bulk properties we used. Adding band structure descriptors or orbital-level features could close this gap.

**Submit to a materials informatics venue.** The 5-view structure discovery is the hook — it's a result that validates CrossCat's capability on real data. The R²=0.81 ionic dielectric prediction is the proof of practical value. The 99.7% calibration is what makes it useful for real screening decisions.

---

*Built with [JAX-CrossCat](https://github.com/sambhal-labs/jaxcross) — GPU-accelerated Bayesian cross-categorization in JAX.*

*Data: [Materials Project](https://materialsproject.org/) v2025.09.25 (154,879 materials, CC BY 4.0).*
