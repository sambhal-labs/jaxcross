# Related Work — NHANES Structure Discovery + Diabetes Prediction

Reference notes for paper / blog post positioning. Last updated 2026-04-26
(after held-out evaluation completed).

## TL;DR

There is **no published CrossCat-on-NHANES** paper. The closest peers are:

1. For **structure discovery** (column views + row clusters): Long et al. 2024 in
   *Nature Cardiovascular Research* — 10 cardiometabolic phenotypes via
   k-prototypes / GMM on imputed NHANES 1988-2018.
2. For **diabetes prediction**: a small library of NHANES-cycle-specific ML papers
   reporting AUC 0.79–0.90, all supervised, all point-prediction-only.
3. For **multiple imputation**: standard Rubin's-rules MI methodology
   (1990s-era), regularly applied to NHANES, no calibrated credible-interval
   reporting on held-out cells.

Our jaxcross work is differentiated on three axes simultaneously: (a) view-level
column structure + DP-learned row clusters, (b) **held-out 90 % credible-interval
coverage = 89 % on biomarker cells the model never saw during training**,
(c) reproducible on a $300 GPU. None of the prior work combines all three —
and **none of them report empirical held-out CI coverage at all**.

## Where our held-out numbers land

| Metric | Our held-out result | Best literature peer | Verdict |
|---|---|---|---|
| Diabetes AUC (held-out 1,742 test rows) | 0.851 [95 % CI 0.817–0.883] | Mehrabkhani 2025: 0.817 | Comparable; we cover their point estimate at the lower bound |
| Diabetes AUC vs Liu 2023 (high-risk subset) | 0.851 [0.817–0.883] | 0.903 | Their cohort is 2,355 (high-risk only) vs our 9,254; cohort breadth differs |
| Diabetes AUC vs Dinh 2019 with-labs | 0.851 [0.817–0.883] | 0.957 | Supervised XGBoost with 123 hand-engineered features beats us on raw AUC; we win on calibration |
| 90 % CI held-out coverage on biomarkers | 89.0 % (1,432 cells) | None reported | Unique contribution — closest analogue is Rubin-style MI confidence intervals, methodologically very different |
| Single-cycle vs pooled cycles | NHANES 2017-2018 only | All literature pools 3-15 cycles | Avoids assay-drift + non-stationarity confounds |

---

## Diabetes prediction on NHANES (closest to our AUC = 0.973)

| Paper | Cycles | n | Method | Held-out AUC | What's missing vs ours |
|---|---|---|---|---|---|
| Mehrabkhani et al. 2025, BMJ Open Diabetes Res Care | 2007–2018 | 29,509 | XGBoost (lifestyle only) | **0.817** | Point predictions; no uncertainty |
| Liu et al. 2023, Archives of Medical Science | 2013–2018 | **2,355** (high-risk subset) | XGBoost (19 risk factors) | **0.903** | High-risk subset only |
| Dinh et al. 2019, BMC Med Inform Decis Mak | 1999–2014 | ~21k | XGBoost ensemble (123 features) | 0.862 lifestyle / **0.957 with-labs** | Ensemble; highest reported with labs |
| Prediabetes EHR + NHANES recalibration, BMC MIDM 2024 | EHR + 2017–2020 | mixed | Logistic + recal | n/a | Recalibration framework |

**All of these are supervised, all give point predictions only.** None of them
report 90 / 95 % credible interval calibration.

Our jaxcross result for comparison:
- DIQ010 in-sample AUC = 0.973, Brier = 0.035, log-loss = 0.105 (n_obs = 8,709)
- 90 %-CI empirical coverage on 6 biomarkers: **91.5 % (mean), 90.6–92.6 %** —
  no equivalent in this list.

To make the AUC comparison apples-to-apples we need a held-out test split (see
*Held-out evaluation plan* below).

---

## Unsupervised clustering / phenotyping (closest to our 3-view discovery)

| Paper | Venue | Method | Output |
|---|---|---|---|
| **Long et al. 2024** — *Cardiometabolic and renal phenotypes and transitions in the United States population* | **Nature Cardiovascular Research** | k-prototypes / GMM on imputed NHANES 1988–2018 | **10 phenotypes** (low-risk, high-BP, severe-obesity, severe-hyperglycemia, low-DBP-low-eGFR, …), tracked over time |
| Cha et al. 2024 | *iScience* | Unsupervised clustering UK Biobank + Taiwan Biobank | **5 metabolic-syndrome endotypes** (non-descriptive, hypertensive, obese, lipodystrophy-like, hyperglycemic) |
| Pediatric obesity subtypes 2025 | *Sci Rep* | Hierarchical clustering NHANES youth | Age-specific obesity subtypes |
| Body-composition phenotypes NHANES 2022 | *Prev Med* | Latent class analysis | 4 body-composition classes |
| OSA combined unsup-sup phenotyping 2021 | *Sci Rep* | Unsupervised + supervised OSA cohort | Phenotypes for sleep apnea subtypes |

**Long et al. 2024 is the gold-standard NHANES structure-discovery paper.**
It is the right citation for the "structure discovery" angle. Where we differ:

| Axis | Long et al. 2024 | Our jaxcross work |
|---|---|---|
| Output | Single-axis: 10 row clusters | **Two-axis: 3 column views × {8, 4, 1} row clusters** |
| Mixed type | Imputed first, then continuous | **Native mixed-type DP mixture per view** |
| Number of clusters | Fixed-K | **DP-learned (auto)** |
| Uncertainty on cluster membership | None reported | **Posterior + Rhat + view-consistency ARI = 1.000** |
| Imputation calibration | Not addressed | **90.6–92.6 % empirical 90 % CI coverage** |
| Variable co-variation | Not addressed | **3-view partition (clinical / diabetes / income)** |

---

## Multiple imputation precedent

Standard NHANES imputation methodology is Rubin's-rules multiple imputation
(MI) — generate M complete datasets, analyze each, combine via Rubin's
formulas. Examples:

- **NHANES III multiple-imputation report**, CDC 2001 — canonical reference
- **Schenker et al. 2016**, PMC5096983 — MI for fully-missing repeated
  accelerometer measures
- **NHANES-CMS Medicaid linkage 2020**, PMC7437981 — MI for linkage ineligibility
- **MICE (van Buuren) tutorials** — practical MI in R

These methods give **point estimates and Wald CIs** that combine within-imputation
and between-imputation variance. They do not directly produce per-row
posterior-predictive credible intervals, and empirical CI coverage is usually
not reported.

Our `batch_credible_interval` story is qualitatively different:
posterior-predictive intervals from a single Bayesian model with
**directly-checkable empirical coverage** — the regulator-friendly framing that
Rubin-style MI doesn't naturally provide.

---

## Direct CrossCat precedents on clinical data

- **Mansinghka, Shafto, Jonas, Petschulat, Gasner, Tenenbaum, "CrossCat: A
  Fully Bayesian Nonparametric Method for Analyzing Heterogeneous, High
  Dimensional Data"**, JMLR 2016 — original CrossCat paper, with synthetic
  + small clinical demos.
- **Saad & Mansinghka 2016**, arXiv 1608.05347, *Probabilistic Data Analysis with
  Probabilistic Programming* — broader methodology paper.
- **GenSQL (PACMPL 2024)** — probabilistic-programming SQL on top of CrossCat;
  evaluated on AutoML for clinical-trial oversight in three real-world
  proprietary clinical trials. **Not on NHANES, not open.**
- **InferenceQL** — tools from MIT Probabilistic Computing Project that build
  on CrossCat for clinical AutoML.

**There is no published CrossCat-on-NHANES paper.** Open-source jaxcross +
this NHANES recipe makes the methodology reproducible for the community —
that itself is a contribution.

---

## Where to position our paper

Three honest framings, ordered by venue ambition:

### 1. NeurIPS Datasets and Benchmarks 2026 (recommended for ML community)

**Title (draft):** "JAX-CrossCat for NHANES: Open-Source Bayesian-Nonparametric
Structure Discovery with Calibrated Uncertainty on a Public Clinical Dataset."

**Pitch:** Open-source GPU-accelerated CrossCat applied to a canonical clinical
benchmark; reproducible on a $300 GPU; calibrated 90 % credible intervals;
unsupervised diabetes-axis recovery (ARI 0.656); held-out diabetes AUC
[pending Phase 3]. Companion library (jaxcross) + dataset recipe.

**Strengths:** Hits a community need (no open CrossCat-on-public-clinical work),
trivially reproducible, calibrated uncertainty is novel for the venue.

**Weaknesses:** Requires held-out comparison vs Mehrabkhani 2025 (0.817 AUC) and
the 0.903-AUC paper to land. We currently only have in-sample numbers.

### 2. JAMA Open / Lancet Digital Health (clinical data note)

**Title (draft):** "Calibrated Bayesian Discovery of the Diabetes Axis in
NHANES 2017-2018."

**Pitch:** Reframed for clinicians — emphasize the unsupervised diabetes-axis
discovery and the regulator-friendly 90 % CI calibration story.

**Strengths:** Strongest clinical-impact framing; the 4-cluster diabetes-axis
phenotype gradient is intuitive to endocrinologists.

**Weaknesses:** Needs a clinician co-author to land; longer review cycle (6-12
months). Better as a follow-up after the arXiv preprint draws clinical interest.

### 3. arXiv preprint (recommended first step)

**Title (draft):** "Calibrated Bayesian Structure Discovery on NHANES 2017-2018
via JAX-Accelerated CrossCat."

**Pitch:** 6-page methods + results paper. Ship within 2 weeks of held-out
results landing. Use as calling card for inbound from the Long et al. 2024 group
and from clinical-AutoML startups (Tempus, Volpara, Owkin, etc.).

**Strengths:** Fastest, no review cycle, full control over framing, attracts
inbound for the commercial play.

**Weaknesses:** No formal review; needs a strong README + repo to back it up
(we already have that).

---

## Held-out evaluation — DONE (Phase 3, 2026-04-26)

Stratified 80 / 20 split: 7,403 train + 1,851 test, plus 5 % cell-mask on 6
biomarkers in the train fold (1,432 cells held out). Cold-start 4 chains × 150
sweeps inference (110 min on GTX 1650), then `packed_insert_rows` of test rows
into the best chain, `batch_classify_column` for diabetes, and
`batch_credible_interval` on the masked cells.

### Held-out diabetes classification (n = 1,742 with observed DIQ010)

| Metric | Point | 95 % bootstrap CI |
|---|---|---|
| AUC | 0.851 | [0.817, 0.883] |
| Brier | 0.068 | [0.060, 0.077] |
| log-loss | 0.346 | [0.281, 0.418] |
| ECE (10-bin) | 0.057 | well-calibrated |

### Held-out CI coverage on 1,432 masked biomarker cells

| Column | n_cells | 50 % | 90 % | 95 % | MAE (z) |
|---|---:|---:|---:|---:|---:|
| LBXGH (HbA1c) | 241 | 55.2 % | 88.4 % | 92.9 % | 0.391 |
| LBXSGL (glucose) | 235 | 47.7 % | 86.8 % | 91.5 % | 0.486 |
| BMXBMI | 320 | 52.2 % | 88.8 % | 91.6 % | 0.511 |
| BPXSY1 (systolic BP) | 253 | 47.8 % | 90.1 % | 93.7 % | 0.642 |
| LBXTC (total chol.) | 270 | 54.4 % | 90.4 % | 95.9 % | 0.716 |
| LBDLDL | 113 | 42.5 % | 89.4 % | 95.6 % | 0.835 |
| **Cell-weighted aggregate** | **1,432** | **~50 %** | **~89.0 %** | **~93.3 %** | — |

Held-out 90 % CI mean coverage = **89.0 %**, **within 1 % of nominal**.

### In-sample → held-out drop

| | In-sample (Phase 2) | Held-out (Phase 3) | Drop |
|---|---|---|---|
| AUC | 0.973 | 0.851 | -0.122 |
| 90 % CI coverage | 91.5 % | 89.0 % | -2.5 % |
| 95 % CI coverage | 95.3 % | 93.3 % | -2.0 % |

The CI calibration story is essentially preserved under held-out evaluation.
The 12-point AUC drop is honest and expected (in-sample = trained-on-row-with-label).

---

## Sources

- [Mehrabkhani et al. 2025 — NHANES 2007-2018 diabetes ML](https://pmc.ncbi.nlm.nih.gov/articles/PMC11931972/)
- [Long et al. 2024 — Cardiometabolic and renal phenotypes (Nature Cardiovascular Research)](https://www.nature.com/articles/s44161-023-00391-y)
- [Cha et al. 2024 — MetS endotypes UK + Taiwan Biobank (iScience)](https://pubmed.ncbi.nlm.nih.gov/39040048/)
- [Cardiometabolic NHANES 2015-2018 network ML (BMC Public Health 2025)](https://link.springer.com/article/10.1186/s12889-025-23483-9)
- [Pediatric obesity subtypes NHANES (Sci Rep 2025)](https://www.nature.com/articles/s41598-025-24524-4)
- [Prediabetes EHR+NHANES recalibration (BMC Med Inform Decis Mak 2024)](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-024-02803-w)
- [Mansinghka, Shafto et al. — CrossCat JMLR 2016](https://www.semanticscholar.org/paper/CrossCat:-A-Fully-Bayesian-Nonparametric-Method-for-Mansinghka-Shafto/b37903a9b41e717599b28c6aa3d595d1bc223950)
- [Saad & Mansinghka — Probabilistic Data Analysis with Probabilistic Programming (arXiv 2016)](https://arxiv.org/abs/1608.05347)
- [GenSQL — clinical AutoML on top of CrossCat (PACMPL 2024)](https://dl.acm.org/doi/10.1145/3656409)
- [NHANES Multiple Imputation reference (CDC, 2001)](https://wwwn.cdc.gov/Nchs/Data/Nhanes3/7a/doc/mimodels.pdf)
- [Schenker et al. — NHANES MI for accelerometer data (PMC 2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5096983/)
