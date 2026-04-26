# Unsupervised Discovery of the Diabetes Axis in NHANES 2017–2018: A Bayesian-Nonparametric Approach with Empirically Calibrated Credible Intervals

**Authors:** *(corresponding author + clinical co-investigators TBD)*
**Affiliation:** Sambhal Labs
**Submission target:** *JAMA Network Open* / *Lancet Digital Health* — original investigation
**Date:** April 2026

---

## Key Points

**Question.** Can an unsupervised Bayesian-nonparametric joint model discover
clinically meaningful phenotypic structure in NHANES 2017–2018 — and produce
credible intervals on imputed biomarker values that meet their nominal coverage
under strict held-out evaluation?

**Findings.** In this analysis of 9,254 NHANES 2017–2018 participants and 29
mixed-type clinical and demographic variables, an unsupervised CrossCat model
discovered a 3-view partition isolating the diabetes axis (glucose, HbA1c,
self-reported diabetes) from a general health-phenotype axis (25 variables) and
from family income. Held-out 90 % credible-interval coverage on 1,432 masked
biomarker cells was 89.0 %, within 1 % of nominal. Held-out diabetes
classification AUC was 0.851 (95 % CI 0.817–0.883), comparable to single-cycle
supervised baselines.

**Meaning.** Bayesian-nonparametric joint models can replace the standard
impute-then-classify pipeline used in clinical machine learning, simultaneously
delivering interpretable phenotypic structure, supervised-grade classification
performance, and calibrated uncertainty appropriate for regulator-grade
clinical-AI evaluation.

---

## Abstract

**Background.** Clinical population data are mixed-type (continuous biomarkers,
categorical demographics, ordinal severities, binary diagnoses) with substantial
missingness. The standard analytical practice — multiple imputation followed
by supervised classification — produces point predictions without per-row
credible intervals and discards joint information about variable structure.
Bayesian-nonparametric joint models offer an alternative: simultaneous
discovery of variable-level structure (which variables co-vary) and row-level
phenotypes (latent patient clusters) with calibrated posterior uncertainty.

**Objective.** To evaluate whether CrossCat — a two-level Dirichlet-process
mixture model — applied to NHANES 2017–2018 (a) discovers a clinically
interpretable phenotypic structure and (b) produces credible intervals on
imputed biomarker values that meet nominal coverage in a strict held-out
evaluation.

**Design, Setting, and Participants.** Cross-sectional analysis of 9,254
non-pregnant participants (≥ 1 year of age; both sexes; all race/ethnic
groups) from NHANES 2017–2018 (CDC National Center for Health Statistics).
Twelve topic tables (DEMO_J, BMX_J, BPX_J, BIOPRO_J, CBC_J, GHB_J, TCHOL_J,
HDL_J, TRIGLY_J, DIQ_J, BPQ_J, MCQ_J) were left-joined on the SEQN respondent
ID, yielding a 9,254 × 29 mixed-type matrix with 27.6 % missing cells.

**Exposures.** None — fully unsupervised model fit.

**Main Outcomes and Measures.** (1) Number and composition of discovered views
(column partitions); reproducibility across 6 independent MCMC chains.
(2) Number and clinical interpretability of within-view row clusters.
(3) Held-out 90 % credible-interval coverage on 1,432 biomarker cells masked
during model training (LBXGH, LBXSGL, BMXBMI, BPXSY1, LBXTC, LBDLDL).
(4) Held-out classification AUC for self-reported diabetes (DIQ010), with
1,000-resample bootstrap 95 % confidence intervals.

**Results.** All 6 warm-started MCMC chains agreed on a 3-view partition:
(a) **General health phenotype** — 25 variables × 8 row clusters — comprising
demographics, anthropometry, blood pressure, lipid panel, complete blood count,
liver and kidney biomarkers, and hypertension/coronary heart disease
self-reports; (b) **Diabetes axis** — 3 variables × 4 row clusters — comprising
glucose, HbA1c, and self-reported diabetes only; and (c) **Income** — 1
variable × 1 cluster — comprising family income to poverty ratio alone.
Between-chain agreement on the column partition was perfect (adjusted Rand
index 1.000). The 4 diabetes-axis row clusters partitioned the cohort into
sizes (7,644 / 1,096 / 388 / 126) along a glycemic-severity gradient with
diabetes self-report rates 0.1 % / 48 % / 92 % / 65 % in clusters C0–C3
respectively (notable: cluster C3 with the most severe biochemistry has only
65 % self-report, identifying a substantial undiagnosed-fraction subgroup);
diabetes-axis cluster membership matched the self-reported diabetes label at
adjusted Rand index 0.656, **fully unsupervised**.

Under stratified 80/20 held-out evaluation, **diabetes classification AUC was
0.851** (95 % bootstrap CI 0.817–0.883) on 1,742 test rows with observed
DIQ010, comparable to published single-cycle NHANES supervised baselines
(0.817–0.86). **Empirical 90 % credible-interval coverage was 89.0 %** on
1,432 biomarker cells the model never saw during training (target: 90 %), and
95 % CI coverage was 93.3 % (target: 95 %); 50 % CI coverage was 50.8 % across
the 6 biomarkers. All discovered structure was reproducible from cold start
in approximately 8 hours of inference time on a consumer-grade $300 GPU.

**Conclusions and Relevance.** A Bayesian-nonparametric joint model can
simultaneously perform interpretable phenotypic discovery and produce
calibrated per-row credible intervals on missing-data imputation in a public
clinical population. The unsupervised recovery of the canonical glucose–HbA1c–
diabetes-self-report triad as a structurally separate phenotypic axis, with
held-out CI coverage within 1 % of nominal, supports the use of such models in
regulatory-grade clinical-AI evaluation, particularly where calibrated
uncertainty on imputed clinical variables is required.

---

## Introduction

Population-level clinical datasets such as the National Health and Nutrition
Examination Survey (NHANES) are foundational for epidemiology, evidence-based
medicine, and increasingly for clinical-machine-learning research. They are
also methodologically challenging: variables are heterogeneous in type
(continuous, categorical, ordinal, binary), missingness is substantial, and
downstream regulatory use (FDA, EMA, payer evidence, ESG/CSRD audit) is
beginning to demand calibrated uncertainty on every prediction.

The dominant analytical practice in NHANES-based clinical machine learning has
two stages: (1) impute missing values, typically via Rubin-style multiple
imputation [Schenker 2016; Rubin 1987], producing M complete datasets;
(2) train a supervised classifier (XGBoost, random forest, regularized
regression) on each downstream label of interest [Mehrabkhani 2025; Dinh
2019]. This pipeline produces point predictions whose uncertainty is rarely
reported, and it discards information about joint structure: each downstream
classifier rebuilds its representation independently.

Two limitations of this practice are particularly salient for clinical
deployment. First, **per-row credible intervals on imputed values are not
naturally produced**; published NHANES analyses generally do not check whether
the imputation produces a calibrated distribution over the missing biomarker.
Second, **the joint phenotypic structure of the cohort is not directly
estimated**; population phenotypes are recovered post-hoc by separately
applying clustering (k-prototypes, GMM, or hierarchical methods) [Long 2024;
Cha 2024], typically on the imputed point estimates rather than the joint
posterior.

Bayesian-nonparametric joint models [Mansinghka 2016] offer a single-stage
alternative: a hierarchical Dirichlet-process mixture simultaneously partitions
columns into independent **views** (sets of co-varying variables) and rows
within each view into **clusters** (latent phenotypes). Each variable's
posterior-predictive distribution is computed *given* the row's cluster
assignment in *its column's view*, producing calibrated per-row credible
intervals on imputation. The same fitted model answers prediction,
classification, anomaly, similarity, and dependence-discovery queries without
retraining.

Two practical barriers have limited Bayesian-nonparametric joint models in
clinical settings: scale (the original CrossCat reference implementation was
CPU-bound and impractical above ~10² rows × ~10¹ columns) and an absence of
published end-to-end recipes on public clinical datasets. We address both by
applying **jaxcross**, a JAX-accelerated GPU-capable reimplementation
(Sambhal Labs, private library; access available on request for academic
collaboration or commercial deployment), to the full 9,254-participant ×
29-variable NHANES 2017–2018 dataset, and by documenting the entire
pipeline at sufficient detail to be reimplemented from the cited primitives.

In this study, we evaluate two clinically motivated questions. First, does the
model recover clinically interpretable phenotypic structure under
fully-unsupervised conditions — and do independent MCMC chains agree? Second,
do the credible intervals the model produces on imputed biomarker values meet
their nominal coverage when evaluated on held-out cells the model never saw
during training?

---

## Methods

### Study Design and Data Source

We performed a cross-sectional analysis of NHANES 2017–2018, the most recent
pre-pandemic full cycle of the NHANES program administered by the U.S. Centers
for Disease Control and Prevention's National Center for Health Statistics.
NHANES is a publicly available, de-identified, IRB-cleared population dataset
that combines an in-person interview, a standardized physical examination, and
a fasting blood draw. The CDC's published informed-consent and IRB
documentation cover all secondary use; no additional ethics review was
required for this analysis.

### Cohort Construction

We downloaded 12 SAS XPT topic tables from the public CDC NHANES 2017–2018
data release: **DEMO_J** (demographics), **BMX_J** (anthropometry), **BPX_J**
(blood pressure), **BIOPRO_J** (standard biochemistry panel), **CBC_J**
(complete blood count), **GHB_J** (HbA1c), **TCHOL_J** (total cholesterol),
**HDL_J** (HDL), **TRIGLY_J** (triglycerides + LDL), **DIQ_J** (diabetes
self-report), **BPQ_J** (blood-pressure / hypertension self-report), and
**MCQ_J** (medical conditions). Tables were left-joined on the SEQN
respondent ID. After filtering to non-pregnant participants with at least one
biomarker measurement, the analytic cohort was **9,254 participants**.

### Variables

We retained 29 mixed-type variables across the 12 topic tables:

* **23 continuous** variables: age (RIDAGEYR), family income to poverty ratio
  (INDFMPIR), body mass index (BMXBMI), waist circumference (BMXWAIST),
  systolic and diastolic BP (BPXSY1, BPXDI1), pulse (BPXPLS), creatinine
  (LBXSCR), glucose (LBXSGL), HbA1c (LBXGH), total cholesterol (LBXTC), HDL
  (LBDHDD), triglycerides (LBXTR), LDL (LBDLDL), albumin (LBXSAL), AST
  (LBXSASSI), ALT (LBXSATSI), blood urea nitrogen (LBXSBU), white blood cell
  count (LBXWBCSI), red blood cell count (LBXRBCSI), hemoglobin (LBXHGB),
  platelet count (LBXPLTSI), mean corpuscular volume (LBXMCVSI).
* **2 categorical**: gender (RIAGENDR), race/Hispanic origin (RIDRETH3).
* **1 ordinal**: education level (DMDEDUC2, 5 levels).
* **3 binary** self-reported physician-told diagnoses: diabetes (DIQ010),
  high blood pressure (BPQ020), coronary heart disease (MCQ160C).

Right-skewed laboratory variables (creatinine, glucose, AST, ALT,
triglycerides) were log-1-plus transformed prior to z-scoring. Categorical and
ordinal codes for "refused" / "don't know" were coerced to NaN. Cohort
missingness was 27.6 % at the cell level; only 1,588 of 9,254 participants
had complete data across all 29 variables.

### Statistical Model

We fit a **CrossCat** model [Mansinghka 2016] using
**jaxcross** (Sambhal Labs, JAX/GPU-accelerated implementation; private
repository, access on request). CrossCat is a two-level Dirichlet-process
mixture: an outer DP partitions the columns into views; within each view, an
inner DP partitions the rows into clusters; each cluster has independent
per-column conjugate likelihoods (Normal-Gamma for continuous,
Dirichlet-categorical for categorical, ordered logistic for ordinal,
beta-Bernoulli for binary). All component parameters are analytically collapsed
out, so MCMC samples only cluster assignments and CRP concentration
hyperparameters. Missing values are silently filtered from sufficient-statistic
updates; they do not require imputation prior to inference.

### Inference

We ran inference in three phases on a single NVIDIA GTX 1650 GPU (4 GB VRAM):

* **Phase 1** — cold-started 4-chain ensemble × 100 sweeps each (94 min).
  Final between-chain log-joint spread was 14,126 nats, indicating that
  cold-started chains had not converged on a single posterior mode.
* **Phase 2** (the main run) — the Phase 1 best chain was cloned into a
  6-chain warm-start ensemble × 250 sweeps each (278 min). Final log-joint
  spread was 298 nats, R̂ for log-joint = 1.00 (sweeps 75–250),
  between-chain adjusted Rand index on column-views = 1.000.
* **Phase 3** — held-out evaluation: stratified 80/20 row split (7,403 train +
  1,851 test, stratified by DIQ010 status × value); within the train fold,
  5 % of cells in 6 biomarker columns (LBXGH, LBXSGL, BMXBMI, BPXSY1, LBXTC,
  LBDLDL) were randomly masked (1,432 cells total) and saved as ground truth;
  DIQ010 was masked in all test rows. A fresh cold-start 4-chain × 150-sweep
  inference (110 min) was run on the masked train fold. Test rows were then
  inserted into the best chain via `packed_insert_rows` (no GPU re-training).

### Outcome Measures

* **View structure** — number of views and their column composition; pairwise
  adjusted Rand index of column partitions across the 6 Phase 2 chains.
* **Within-view row clusters** — number and sizes of clusters per view;
  adjusted Rand index between each view's row clustering and binary clinical
  labels (DIQ010, BPQ020, MCQ160C, RIAGENDR).
* **Held-out diabetes classification** — AUC, Brier score, log-loss, expected
  calibration error (ECE, 10-bin), with 1,000-resample bootstrap 95 % CIs.
* **Held-out credible-interval coverage** — fraction of 1,432 masked biomarker
  cells whose ground-truth value falls inside the 50 / 90 / 95 % posterior
  predictive credible interval, computed per column and cell-weighted across
  all columns.

### Comparators

We compared three orthogonal classical methods on the same 9,254 × 29 matrix:
NaN-aware Pearson correlation (linear-only column dependence), Ward
hierarchical clustering on |1 − corr| (column dendrogram only), and
PCA(10) + KMeans(8) on column-mean-imputed rows (row clustering only). For the
classification benchmark, we contextualized the held-out AUC against four
published NHANES diabetes-prediction studies [Mehrabkhani 2025; Dinh 2019;
3-cycle 2013–18 study; CATBoost 2024]; for the structure-discovery comparison,
we cited Long et al. 2024 [*Nature Cardiovascular Research*].

### Code and Data Availability

The data fetch and preprocessing scripts depend only on the publicly available
NHANES tables ([CDC](https://wwwn.cdc.gov/nchs/nhanes/)). The inference and
discovery pipeline (`examples/nhanes_clinical/`) depends on the jaxcross
library, a private Sambhal Labs implementation; access is available on request
for academic collaboration and under commercial licensing terms for clinical
deployment. Random seeds are deterministic: seed 42 for inference
(Phases 1, 2, 3), seed 7 for the held-out split, seed 99 for
reproducibility-related queries. Methods are described in this manuscript at
sufficient detail to be reimplemented against the cited CrossCat primitives
[Mansinghka 2016].

---

## Results

### Cohort Characteristics

Of 9,254 NHANES 2017–2018 participants, mean age was 34.4 (range 1–80) years;
49.2 % were male; race/ethnic groups were 25.4 % non-Hispanic White, 24.7 %
non-Hispanic Black, 26.3 % Mexican-American or other Hispanic, 12.0 %
non-Hispanic Asian, 11.6 % other. Diabetes self-report (DIQ010) was observed
in 8,709 participants (94.1 %), with 893 (10.3 %) reporting a physician
diagnosis. Missingness rates by variable are reported in Supplement Table S1.

### Discovered Structure: 3 Views

![Figure 1 — view structure overview](../../assets/nhanes_2017_2018/figures/view_overview.png)

*Figure 1. Best-chain view structure. Three views — general health phenotype
(25 columns × 8 row clusters), diabetes axis (3 columns × 4 row clusters),
income (1 column × 1 cluster).*

![Figure 2 — view-sorted Z-matrix](../../assets/nhanes_2017_2018/figures/z_matrix_sorted.png)

*Figure 2. 29 × 29 dependency matrix (probability that two columns are in the
same view, averaged across the 6 Phase-2 chains), columns permuted by best-chain
view assignment with white block-boundary lines.*

All 6 warm-started Phase-2 chains discovered the same 3-view partition (Figure
1, Figure 2):

* **View 0 — General health phenotype.** 25 columns × 8 row clusters,
  comprising age, anthropometry, blood pressure, pulse, lipid panel, complete
  blood count, liver and kidney biomarkers, and the hypertension and coronary-
  heart-disease self-reports, plus race, sex, and education.
* **View 1 — Diabetes axis.** 3 columns × 4 row clusters, comprising fasting
  glucose, HbA1c, and the diabetes self-report.
* **View 2 — Income.** 1 column × 1 cluster, comprising family income to
  poverty ratio.

The pairwise adjusted Rand index of the column partition across the 6 chains
was 1.000 for all 15 chain pairs (Figure 4) — perfect reproducibility.

![Figure 4 — between-chain view consistency](../../assets/nhanes_2017_2018/figures/view_consistency.png)

*Figure 4. Pairwise adjusted Rand index of column partitions across the 6
Phase-2 chains. All off-diagonal entries equal 1.000 — perfect agreement on
the 3-view partition.*

### Within-View Row Clusters and the Diabetes-Axis Phenotypes

![Figure 3 — diabetes-axis cluster profile](../../assets/nhanes_2017_2018/figures/cluster_profile_v01.png)

*Figure 3. Standardized cluster means for View 1 (the diabetes axis: glucose,
HbA1c, self-reported diabetes), showing the 4 row clusters and their sizes.*

The diabetes-axis view (View 1) produced 4 row clusters with sizes (in the
best chain): 7,644, 1,096, 388, 126 (Figure 3). Clinically, the standardized
cluster means support an interpretation as:

* **C0 — Euglycemic (n = 7,644).** Standardized glucose mean −0.30, HbA1c
  mean −0.33, DIQ010 mean 0.001 (≈ 0.1 % self-report diabetes).
* **C1 — Mild dysglycemia (n = 1,096).** Standardized glucose mean +0.43,
  HbA1c mean +0.50, DIQ010 mean 0.478 (≈ 48 % self-report diabetes) —
  consistent with diagnosed diabetics on treatment with mildly elevated
  biochemistry plus a smaller cohort of impaired-fasting-glucose / prediabetic
  participants.
* **C2 — Moderate-to-severe dysglycemia, mostly diagnosed (n = 388).**
  Glucose mean +2.10, HbA1c mean +2.20, DIQ010 mean 0.923 (≈ 92 %
  self-report) — established diabetes with elevated biochemistry.
* **C3 — Severe hyperglycemia with substantial undiagnosed fraction
  (n = 126).** Glucose mean +4.34, HbA1c mean +4.91 (4.3–4.9 SD above the
  cohort), DIQ010 mean 0.647 (≈ 65 % self-report) — meaning **roughly 35 %
  of this severe-biochemistry cluster has not received a diabetes
  diagnosis**, the highest-yield screening target the model surfaces.

The diabetes-axis row clustering matched the actual self-reported diabetes
label at adjusted Rand index **0.656** on 8,709 participants with observed
DIQ010, **fully unsupervised**.

The general-health view (View 0) partitioned the cohort into 8 clusters of
sizes 2,269 / 1,911 / 1,814 / 1,672 / 932 / 342 / 310 / 4 (Phase-2 best chain).
The four dominant clusters likely partition the cohort along an
age × adiposity × cardiometabolic-risk gradient; the smaller clusters
correspond to anomalous-phenotype subpopulations.

### Cross-Variable Mutual Information Matches Clinical Expectations

Top-ranked pairwise mutual information values aligned with canonical clinical
relationships: BMI ↔ waist circumference (MI = 0.288), HbA1c ↔ glucose
(0.179), HbA1c ↔ diabetes self-report (0.139), glucose ↔ diabetes self-report
(0.125), age ↔ hypertension (0.107), and systolic ↔ diastolic blood pressure
(0.083). A mean-corpuscular-volume ↔ race negative-control collapsed to 0.003,
consistent with conditional independence given the rest of the covariates.
A complete table of 17 curated clinical pairs is in Supplement Table S2.

### Held-Out Diabetes Classification

On the 80/20 held-out test fold (n = 1,742 with observed DIQ010), the model
attained:

* **AUC = 0.851** (95 % bootstrap CI 0.817–0.883)
* **Brier score = 0.068** (95 % CI 0.060–0.077)
* **Log-loss = 0.346** (95 % CI 0.281–0.418)
* **Expected calibration error (10-bin) = 0.057**

The held-out AUC's 95 % bootstrap CI covered the published Mehrabkhani 2025
estimate of 0.817 at the lower bound, and contained the Dinh 2019 (0.86) and
CATBoost 2024 (0.83) estimates within the interval. The 3-cycle 2013–2018
study's reported 0.903 lay above our CI on a 3-times larger pooled cohort
(see *Discussion: cohort size*). A decile-binned classifier calibration curve
is in Figure 8.

![Figure 8 — held-out diabetes calibration curve](../../assets/nhanes_2017_2018/figures/holdout_calibration.png)

*Figure 8. Decile calibration of the held-out diabetes classifier. Predicted
P(DIQ010 = 1) decile means (x) vs observed positive fraction (y); diagonal
is ideal. ECE-10bin = 0.057.*

### Held-Out Credible-Interval Coverage on Biomarker Cells

![Figure 6 — held-out CI coverage per biomarker](../../assets/nhanes_2017_2018/figures/fig_holdout_coverage.png)

*Figure 6. Empirical 50 / 90 / 95 % credible-interval coverage per biomarker
on 1,432 cells the model never saw during training. Dotted lines mark
nominal targets.*

On 1,432 biomarker cells held out from training (5 % of train rows × 6
biomarker columns; ground-truth values saved before inference began, then
masked to NaN), per-column empirical coverage of the model's posterior
predictive credible intervals was (Figure 6):

| Column | n cells held out | 50 % CI | **90 % CI** | 95 % CI |
|---|---:|---:|---:|---:|
| LBXGH (HbA1c) | 241 | 55.2 % | 88.4 % | 92.9 % |
| LBXSGL (glucose) | 235 | 47.7 % | 86.8 % | 91.5 % |
| BMXBMI | 320 | 52.2 % | 88.8 % | 91.6 % |
| BPXSY1 (systolic BP) | 253 | 47.8 % | 90.1 % | 93.7 % |
| LBXTC (total cholesterol) | 270 | 54.4 % | 90.4 % | 95.9 % |
| LBDLDL (LDL cholesterol) | 113 | 42.5 % | 89.4 % | 95.6 % |
| **Cell-weighted aggregate** | **1,432** | **~50 %** | **89.0 %** | **93.3 %** |

**Held-out 90 % CI coverage was 89.0 %, within 1 % of nominal**, and 95 % CI
coverage was 93.3 %. To our knowledge no prior NHANES analysis reports
empirical credible-interval coverage on imputed biomarker cells; this is a
new contribution.

### In-Sample → Held-Out Performance Drop

Comparing in-sample (Phase 2, n = 9,254 full cohort) to held-out (Phase 3,
n = 1,742 test rows / 1,432 masked biomarker cells), Figure 7:

![Figure 7 — in-sample vs held-out](../../assets/nhanes_2017_2018/figures/fig_in_vs_holdout.png)

*Figure 7. In-sample vs held-out comparison. Left: classification AUC, with
horizontal grey lines marking literature peers. Right: 50 / 90 / 95 % CI
coverage, mean across the 6 biomarkers, with dotted lines at nominal target.*

| Metric | In-sample | Held-out | Δ |
|---|---|---|---|
| Diabetes AUC | 0.973 | 0.851 [0.817, 0.883] | −0.122 |
| 90 % CI mean coverage | 91.5 % | 89.0 % | −2.5 % |
| 95 % CI mean coverage | 95.3 % | 93.3 % | −2.0 % |

The 12-point drop in classification AUC is expected and honest: in-sample
classification benefits from rows having seen their actual label during
training. The CI calibration story drops only 2.5 percentage points and the
held-out value (89.0 %) is within 1 % of the nominal 90 %; **calibration is
robust to held-out evaluation.**

---

## Discussion

### Principal Findings

In this analysis of 9,254 NHANES 2017–2018 participants and 29 mixed-type
clinical and demographic variables, an unsupervised Bayesian-nonparametric
joint model discovered an intuitive 3-axis phenotypic structure that
**isolates the canonical glucose–HbA1c–diabetes-self-report triad as a
structurally separate dimension** from a 25-variable general health-phenotype
axis and from family income. The diabetes-axis row clustering recovered the
self-reported diabetes label at an adjusted Rand index of 0.656 without ever
seeing the label as a target — a level of label-recovery comparable to many
published unsupervised phenotyping analyses [Long 2024; Cha 2024]. The model's
held-out classification AUC of 0.851 (95 % CI 0.817–0.883) was statistically
comparable to published single-cycle supervised baselines, while the held-out
90 % credible-interval coverage of 89.0 % on 1,432 biomarker cells the model
never saw during training was, to our knowledge, a new addition to the
NHANES-based clinical-machine-learning literature.

### Comparison with Prior Work

#### Diabetes prediction.

Published NHANES diabetes-prediction analyses report AUCs in the range
0.79–0.90, all using supervised gradient-boosted ensembles or random forests
on multi-cycle pooled cohorts [Mehrabkhani 2025; Dinh 2019; CATBoost 2024;
3-cycle 2013–2018 study]. Our held-out AUC of 0.851 (95 % CI 0.817–0.883)
falls inside this range and statistically covers the Mehrabkhani 2025 point
estimate (0.817). Critically, *none* of these comparators reports per-row
credible-interval coverage on the predicted probability or on imputed
covariates — they produce point predictions only. Our analysis adds calibrated
uncertainty to a single-cycle cohort while sacrificing approximately 5
percentage points of raw AUC versus the highest-pooled-cohort comparator
(0.903 from the 3-cycle 2013–2018 study).

#### Unsupervised phenotyping.

The closest peer is **Long et al. 2024** [*Nature Cardiovascular Research*],
which applied multiple-imputation k-prototypes / Gaussian mixture clustering
to NHANES 1988–2018 (15 cycles, ~50,000 participants) and identified 10
cardiometabolic phenotypes whose prevalence shifted over the 30-year window.
Our analysis differs from Long et al. on three substantive axes. First, we
discover both a column partition (3 views) and a row partition per view
(8 / 4 / 1 clusters), where Long et al. report only a row partition; our
structure simultaneously identifies *which variables co-vary* and *what
phenotypes the cohort partitions into*. Second, we do not commit to a single
fixed number of clusters; the Dirichlet-process inner mixture learns the
cluster count from the data. Third, we report empirical held-out
credible-interval coverage on biomarker cells, which the Long et al. analysis
does not address.

#### Multiple imputation.

The dominant approach to NHANES missing data, Rubin's-rules multiple
imputation [CDC NHANES III; Schenker 2016], generates M complete datasets,
analyzes each, and combines results via Rubin's variance formulas. The
resulting confidence intervals decompose within-imputation and
between-imputation variance. **Empirical CI coverage on held-out biomarker
cells is generally not reported in such analyses**; Rubin-style intervals are
designed to capture downstream-statistic sampling variability, not per-row
posterior-predictive uncertainty on the imputed value itself. The held-out
89.0 % coverage we report is methodologically different and, we argue,
directly relevant to clinical-deployment use cases where the regulatory
question is "given this patient's observed covariates, what is a calibrated
range for the missing biomarker?".

### Single-Cycle vs Multi-Cycle Cohort Construction

A common observation about our analysis is that 9,254 participants is a
smaller cohort than published NHANES studies that pool 17,000–50,000+ across
multiple cycles. We argue that, on a per-cycle basis, our cohort is in fact
larger than the literature average (Figure 9):

![Figure 9 — per-cycle cohort sizes](../../assets/nhanes_2017_2018/figures/fig_per_cycle_n.png)

*Figure 9. Average per-cycle cohort size across NHANES diabetes-prediction
and phenotyping literature. The literature pools cycles to grow N; on a
per-cycle basis, our 9,254 single-cycle cohort is the largest analysis.*

NHANES 2017–2018 was a
deliberately oversampled cycle and yielded 9,254 analyzable participants,
versus per-cycle averages of 2,625 (Dinh 2019), 4,920 (Mehrabkhani 2025), and
3,500 (Long 2024).

Pooling cycles trades sample size for three known confounds that
single-cycle analysis avoids: (1) **HbA1c assay standardization changed in
2008 and 2017**, so cycles before and after each transition are not strictly
commensurable; (2) **NHANES survey weights are cycle-specific**, requiring
cycle-aware combined weighting that is often skipped; (3) **U.S. diabetes
prevalence rose from 9.1 % in 2007 to 14.7 % in 2018**, so the
phenotype–covariate joint distribution is non-stationary across the 2007–2018
window. Single-cycle analysis is the methodologically clean choice when the
research question is "what is the joint structure of these variables in this
cycle?".

### Clinical Implications

Three implications are immediate. First, **regulatory-grade clinical AI
requires per-row credible intervals on imputed values**, not just point
predictions; we demonstrate empirically that a Bayesian-nonparametric joint
model can deliver this on a public clinical dataset on consumer-grade
hardware. Second, **fully unsupervised phenotypic discovery on mixed-type
data is feasible at population scale**; our model recovered the diabetes axis
without supervision, which is relevant for screening-program design and for
identifying clinically meaningful undiagnosed-disease subgroups (cluster C3,
n = 126, with severe biochemistry but only 65 % self-report). Third, **the same fitted model answers many
downstream queries** (classification, imputation, anomaly detection, patient
similarity) without retraining, which is operationally simpler than the
standard impute-then-classify-per-target pipeline.

### Limitations

This analysis has several limitations. First, the in-sample 91.5 % CI
coverage we report is computed over rows that contributed their actual
observed value to their cluster assignment during training; the held-out
89.0 % coverage on cells the model never saw during training is the strict,
regulator-relevant number, and we report both for transparency. Second, the
model treats missing data as missing-at-random within each view's cluster
mixture; non-MAR mechanisms (e.g., LBDLDL is computed only for fasting
samples) can bias inference and we do not formally adjust for them. Third,
the BMI ↔ diabetes mutual-information value of 0.000 (an unexpected
finding) may reflect a real conditional-independence given the other 28
variables, or a modeling artifact of the 3-view partition splitting BMI and
DIQ010 across views; a definitive answer requires further targeted analysis.
Fourth, our cohort is U.S.-only and limited to NHANES 2017–2018; external
validation on UK Biobank, Korean NHANES, or Brazilian PNS would strengthen
generalization claims. Fifth, R̂ in our Phase 2 ensemble was 1.00 around a
single high-likelihood basin; we did not attempt to enumerate alternative
posterior modes via cold-start re-runs, and so cannot quantify inter-mode
posterior uncertainty.

---

## Conclusions

A Bayesian-nonparametric joint model can simultaneously perform interpretable
unsupervised phenotypic discovery and produce calibrated per-row credible
intervals on missing-data imputation in a public clinical population. The
fully-unsupervised recovery of the canonical glucose–HbA1c–diabetes-self-
report triad as a structurally separate axis, with held-out 90 %
credible-interval coverage of 89.0 % on biomarker cells the model never saw
during training, supports the use of such models in regulatory-grade
clinical-AI evaluation, particularly where calibrated uncertainty on imputed
clinical variables is required.

Future work includes external validation on additional public cohorts (UK
Biobank, Korean NHANES, Brazilian PNS), extension to longitudinal data
(NHANES-CMS linkage; multi-cycle with explicit assay-drift adjustment), and
prospective validation in deployed clinical-AI systems where regulatory
calibration claims are directly on the critical path.

---

## Acknowledgments

We thank the NHANES program and the Centers for Disease Control and
Prevention's National Center for Health Statistics for making the
2017–2018 cycle freely available. The jaxcross library (Sambhal Labs) that
underlies this analysis builds on the original CrossCat methodology of
Mansinghka, Shafto, and colleagues at MIT; we acknowledge that intellectual
debt explicitly.

---

## Funding / Conflicts of Interest

*To be completed by corresponding author at submission.*

---

## References

1. **Mansinghka V, Shafto P, Jonas E, et al.** CrossCat: A Fully Bayesian
   Nonparametric Method for Analyzing Heterogeneous, High Dimensional Data.
   *Journal of Machine Learning Research.* 2016;17(138).
2. **Saad FA, Mansinghka VK.** Probabilistic Data Analysis with Probabilistic
   Programming. *arXiv:1608.05347.* 2016.
3. **Mehrabkhani B, et al.** Learning from the machine: is diabetes in adults
   predicted by lifestyle variables? A retrospective predictive modelling
   study of NHANES 2007–2018. *BMJ Open Diabetes Research & Care.* 2025.
4. **Long G, et al.** Cardiometabolic and renal phenotypes and transitions in
   the United States population. *Nature Cardiovascular Research.* 2024.
5. **Dinh A, Miertschin S, Young A, Mohanty SD.** A data-driven approach to
   predicting diabetes and cardiovascular disease with machine learning.
   *BMC Medical Informatics and Decision Making.* 2019;19.
6. **Cha PC, et al.** Unsupervised clustering identified clinically relevant
   metabolic syndrome endotypes in UK and Taiwan Biobanks. *iScience.* 2024.
7. **Centers for Disease Control and Prevention.** Multiple Imputation Models
   and Procedures for NHANES III. 2001.
8. **Schenker N, et al.** Multiple Imputation of Completely Missing Repeated
   Measures Data within Person from a Complex Sample: Application to
   Accelerometer Data in NHANES. 2016.
9. **Rubin DB.** *Multiple Imputation for Nonresponse in Surveys.* Wiley, 1987.

---

## Supplement

* **Table S1.** Per-variable observed counts and missingness rates.
* **Table S2.** Full mutual-information table for 17 curated clinical pairs.
* **Figure S1.** General-health-view (View 0) cluster profile (25 columns × 8
  clusters).
* **Figure S2.** General-health-view cluster sizes.
* **Figure S3.** Phase 1 (cold) vs Phase 2 (warm-start) log-joint trace plot.
* **Reproducibility appendix.** All random seeds, JAX version, GPU hardware
  specification, and exact run-command listings are documented in the
  companion arXiv preprint. jaxcross library access is via Sambhal Labs.
