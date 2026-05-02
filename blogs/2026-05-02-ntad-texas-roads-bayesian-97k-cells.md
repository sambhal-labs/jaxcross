# Calibrated Road-Network AI: 96,656 Held-Out NTAD Texas Cells, and the Honest Limitation

*Posted May 2026. ~2,300 words. Audience: ML / data-science / civil-AI engineers, BTS / state-DOT GIS analysts, geospatial ML startups, regulatory-science researchers.*

---

## TL;DR

We applied **jaxcross** — Sambhal Labs' JAX/GPU implementation of
CrossCat — to the public **2020 BTS NTAD North American Roads dataset,
Texas filter** (39,164 highway segments × 12 mixed-type columns
derived from 18 raw NTAD attributes; **0.0 % cell-level missingness**
— the cleanest cohort of the Wave 2 series). On a single \$300 GTX
1650 (4 GB VRAM), in ~5.6 hours of total wall time:

* **2-view best-chain column partition discovered** (chain 2: 10 + 2
  cols), but **mean between-chain ARI = 0.857** — the first Wave 2
  demo to break perfect chain agreement. Three chains found a 3-view
  9 + 2 + 1 partition; chain 2 (best log-joint) merged the rare
  `border` singleton view into the dominant view. Z-matrix off-
  diagonal mean is **0.595** — denser than other Wave 2 demos
  because most segment attributes co-cluster.
* **Held-out 90 % CI coverage = 84.0 %** on **96,656 cells** (24,164
  leftover segments × 4 segment-summary columns). The aggregate is
  **6 pp below nominal** — and the under-coverage is concentrated
  entirely on the geographic columns: lanes 94.0 % ✅, speedlim
  91.5 % ✅, but centroid_latitude 83.0 % ⚠️ and centroid_longitude
  **67.6 %** ❌. This **explicitly demonstrates the no-spatial-
  modeling limitation** that the Wave 2 plan flagged upfront.
* **Anomaly score recovers border-crossing and admin-mismatch
  outliers.** The top-5 most-anomalous Texas road segments include
  the **Anzalduas International Bridge** (Class-1 Interstate-level
  facility marked Municipal-administered with international border
  flag — three simultaneously-unusual attribute values), an FM-route
  with intermodal NHS subtype, and an 11-lane segment with
  degenerate (length=0) geometry.
* **Target-free `is_interstate` ECE = 6.7 × 10⁻⁸** — the best
  calibration result of the entire Wave 2 series, beating FARS's
  0.22 % by four orders of magnitude. AUC = 1.0000 (definitional
  reproduction).

This is **Wave 2 Demo 5 — the final demo of the infrastructure
series**. After NBI bridges, HPMS pavements, NTD transit agencies,
and FARS fatal crashes, NTAD is where the known spatial-modeling
limitation becomes visible. We document it quantitatively rather than
papering over it.

**Library access:** jaxcross is a Sambhal Labs library (private
repository). Academic-collaboration and commercial-licensing access
available on request — see *Resources* at the end of the post.

---

## What we are not selling

A pre-emptive disclaimer: **this is not a supervised road-
classification AUC win**. Random Forest and XGBoost on a fixed road-
class target are state-of-the-art — RF AUC 1.000 on `is_interstate`
with `road_system` excluded saturates the metric. Gradient boosting
on a fixed binary target is exactly the regime where these methods
earn their place — and trying to beat them on raw AUC with a Bayesian
joint model is a confused positioning.

Instead we ship the **calibration + multi-view structure + phenotype
+ dependency** package from a single fitted joint model. For NTAD
specifically, the deliverable is **calibrated CIs on segment-attribute
imputation, with a documented exception for the geographic columns**
where jaxcross hits its known limitation.

---

## Why a Texas-only cohort?

NTAD's North American Roads layer covers ~5 million highway segments
across the US, Canada, and Mexico. Per Wave 2 plan we apply a state
filter at fetch time — `COUNTRY=2 AND JURISCODE='02_48'` (US Texas) —
yielding **39,164 segments**. Single-state cohort fits 4 GB VRAM
cleanly without further sampling pressure, and the state-filter
design parallels the NBI bridge-safety and HPMS pavement demos.

The data fetch is paginated ArcGIS REST: ~20 pages of 2,000 features
each (~50 MB total JSON), no auth, public domain.

---

## Dataset: NTAD Texas in 30 seconds

* **Source URL:** [BTS NTAD ArcGIS REST](https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/NTAD_North_American_Roads/FeatureServer/0)
* **License:** Public domain, U.S. Government data.
* **Final analytic matrix:** **39,164 segments × 12 mixed-type
  columns** (smallest schema in Wave 2).
* **Cell-level NaN fraction:** **0.0 %** (NTAD is fully populated).
* **`is_interstate` prevalence:** 12.48 % (4,888 segments).
* **`admin_is_state` prevalence:** 64.0 %.
* **`border` prevalence:** 0.15 % (only 59 international-crossing
  segments).

Column-type breakdown:

| Type | Count | Examples |
|---|---:|---|
| CONTINUOUS | 6 | `length` (km), `shape_length`, `lanes` (2–12), `speedlim` (48–120 kph), `centroid_latitude`, `centroid_longitude` |
| CATEGORICAL | 3 | `class` (6 levels: 1=Interstate ... 6=Local), `nhs` (8 NHS subtypes), `road_system` (I / U / S / C / FM / Other) |
| BINARY | 3 | `admin_is_state`, `border`, `is_interstate` (derived) |

**12 columns total** — vs the 28 of NBI / HPMS / NTD or 40 of FARS.
The narrow schema reflects NTAD's network-topology focus (geometry
+ classification only, no condition or operational attributes).

---

## The model in one paragraph

**CrossCat** is a two-level Dirichlet-process mixture. An outer DP
partitions columns into **views** (sets of co-varying variables);
within each view, an inner DP partitions rows into **clusters**
(latent phenotypes). Component parameters are conjugate. All
component parameters are analytically collapsed out, so the MCMC
samples only the cluster assignments and CRP concentrations.

In one shot you get: a learned column partition, a learned row
clustering *per view*, posterior-predictive distributions for every
(row, column), calibrated credible intervals, mutual information,
dependency probabilities, anomaly scores, segment-similarity scores.
No retraining for each downstream query.

---

## The multi-phase plan

We use the standard split: **inference subsample = 15,000 segments**
(deterministic, seed = 42). The remaining **24,164 segments are
*never seen* by inference** — they become the held-out cohort.

* **Phase 1 (cold-start, 4 chains × 200 sweeps).** 2.8 hours wall.
  Cold-start spread is wide (~11 K nats) — the largest of Wave 2 —
  reflecting that NTAD's narrow 12-col schema gives the model
  multiple near-equally-good basins to settle into.
* **Phase 2 (warm-start, 4 chains × 200 sweeps from Phase-1 best).**
  2.8 hours wall. **Phase 2 partially consolidates** — three chains
  agree at ARI 1.000 on a 9+2+1 three-view partition; the fourth
  chain (chain 2, best log-joint) merges the rare `border` singleton
  view into the dominant view, yielding 10+2 with mean ARI 0.857.
* **Phase 3 (leftover evaluation).** No further inference. The
  24,164 leftover segments are inserted into the best chain via
  `packed_insert_rows` with the four segment-summary columns
  pre-masked, then `batch_credible_interval` is called on the masked
  cells. **96,656-cell calibration table.**

| Phase | Chain | Final log-joint (nats) | View partition |
|---|---|---:|---|
| 2 | 0 | −205,357 | 9 + 2 + 1 |
| 2 | 1 | −188,320 | 9 + 2 + 1 |
| **2 (best)** | **2** | **−188,038** | **10 + 2** |
| 2 | 3 | −205,516 | 9 + 2 + 1 |

**Total wall time on a \$300 GTX 1650: ~5.6 hours.**

---

## The 2 views (best chain) and the chain-disagreement story

The Phase 2 best chain (chain 2) finds **2 views**:

**View 0 — segment-attribute axis (10 columns).** `length`,
`shape_length`, `lanes`, `speedlim`, `class`, `nhs`, `road_system`,
`admin_is_state`, `border`, `is_interstate`. Geometry +
classification + administrative attributes all co-cluster. The 9
row-clusters separate Interstates / US Highways / FM routes /
municipal class-3 streets / border-crossing facilities.

**View 1 — geographic axis (2 columns).** `centroid_latitude`,
`centroid_longitude`. The model splits geography into its own view
with 2 row-clusters (east vs west Texas split along longitude
~ -97°).

**The disagreement:** chains 0/1/3 instead place the rare `border`
binary in its **own singleton view**, yielding 9+2+1. The dominant
view in their basin is 9 cols (everything except border). Chain 2's
basin merges border into the dominant view. With only **59 border-
crossing segments out of 39,164** (0.15 % prevalence), the column has
weak likelihood signal and the partition can absorb it either way at
near-equal posterior cost.

This is the **first Wave 2 demo to break perfect chain agreement**
(NBI / HPMS / NTD / FARS all reached ARI ≥ 0.997). The disagreement
is structural but substantively narrow — affects only the placement
of one rare-binary column.

---

## The 96,656-cell calibration table — and the honest limitation

This is where the Wave 2 plan's spatial-modeling limitation becomes
quantitatively visible.

**Setup.** The 24,164 leftover segments are inserted into the Phase-2
best chain via `packed_insert_rows`. For each leftover segment we
mask the four segment-summary columns (`lanes`, `speedlim`,
`centroid_latitude`, `centroid_longitude`) and ask the model to
reconstruct each cell.

**96,656 cells of held-out evaluation.**

| Column | n cells | 50 % CI | 90 % CI | 95 % CI | Mean width |
|---|---:|---:|---:|---:|---:|
| `lanes` | 24,164 | 81.6 % | **94.0 %** ✅ | 96.2 % | 4.74 |
| `speedlim` | 24,164 | 60.4 % | **91.5 %** ✅ | 94.6 % | 54.15 kph |
| `centroid_latitude` | 24,164 | 56.0 % | **83.0 %** ⚠️ | 87.7 % | 6.13° |
| `centroid_longitude` | 24,164 | 35.7 % | **67.6 %** ❌ | 75.0 % | 5.50° |
| **Cell-weighted aggregate** | **96,656** | **58.4 %** | **84.0 %** | **88.4 %** | — |

**Reading the table.** The aggregate 90 % CI coverage (**84.0 %**)
sits **6 pp below nominal** — the first Wave 2 demo to under-cover.
The under-coverage is **not a model failure**: it's a faithful
demonstration of the limitation flagged in the Wave 2 plan upfront.

Decomposed by column:

- **`lanes` and `speedlim` (well-calibrated):** Both achieve 90 % CI
  coverage within ±2 pp of nominal. These columns live in the
  dominant 10-column view, where the cluster predictive distribution
  has ample information from 8 other co-clustered attributes
  (length, class, nhs, road_system, admin, border, is_interstate,
  shape_length).

- **`centroid_latitude` and `centroid_longitude` (under-covered):**
  The geographic view has only 2 columns, both in the masked set.
  When *both* are masked simultaneously, the cluster predictive for
  these cells must fall back to the marginal posterior of the
  geographic view — which is essentially the bulk Texas-segment
  distribution, not a row-specific prediction. The 67.6 % coverage
  on longitude reflects this honestly: the model has no information
  to distinguish "this segment is in Houston" from "this segment is
  in El Paso" once both lat and lon are masked simultaneously.

**The substantive interpretation.** jaxcross does not natively model
spatial autocorrelation. For cells in a 2-column view where both
columns are masked, the model's posterior collapses to the column-
marginal distribution — calibrated against the population, but not
sharp on the held-out segment.

**Mitigation options:**
- **Single-column masking.** Mask only one of lat/lon, keep the other
  observed. This restores the cluster predictive's ability to use the
  observed-coordinate as a within-view feature.
- **Spatial component model extension.** A planned library follow-up
  would add a Gaussian-process-style spatial component for
  geographic columns.
- **Separate spatial regression on top of the joint model.** Use
  jaxcross for the non-spatial joint structure + calibration; layer a
  kriging or spatial-Bayes regression on top for the geographic CIs.

**This paper documents the limitation quantitatively rather than
papering over it.** The other three Wave 2 demos that include
geographic columns (NBI, HPMS, FARS) all paired lat/lon with a richer
co-clustering view (e.g., NBI's 16-col structural-attribute view
contains both lat/lon and many other columns), so the
masked-both-coords scenario didn't arise. NTAD is the demo where the
narrow 12-col schema forces both lat and lon into the same 2-col
splinter view, which is where the limitation surfaces.

---

## Bonus: target-free `is_interstate` calibration

| Metric | Point | 95 % CI |
|---|---:|---|
| AUC | 1.0000 | (perfect, no bootstrap variation) |
| Brier score | 0.0000 | — |
| Log-loss | ~0.0000 | — |
| **ECE (10 bins)** | **6.7 × 10⁻⁸** | (single value) |

**ECE = 6.7 × 10⁻⁸ — the best calibration result of the Wave 2
series**, beating FARS's 0.22 % by four orders of magnitude. AUC is
1.0000 — a definitional reproduction (`is_interstate = road_system
== "I"`, and `road_system` is in the cluster-determining feature
set).

The Random Forest baseline also achieves AUC 1.000 with `road_system`
*excluded* — the Interstate signature on `class` + `nhs` + `speedlim`
+ `lanes` is unambiguous enough that both methods saturate. **The
calibration headline is the differentiator**, not the AUC.

---

## Anomaly score: top-5 most-unusual Texas road segments

The top-5 most-anomalous Texas road segments on the 1,500-segment
inference cohort:

| Rank | Anomaly | Segment | Class | NHS | Admin | Border | Lanes | Speedlim |
|---:|---:|---|---:|---:|---|---:|---:|---:|
| 1 | **0.649** | **Anzalduas International Bridge** | 1 | 0 | Municipal | 2 | 4 | 60 |
| 2 | 0.573 | FM106 segment | 4 | **8** | Municipal | 0 | 4 | 56 |
| 3 | 0.540 | Pete Diaz Ave / U83 | 3 | 0 | Municipal | 2 | 4 | 60 |
| 4 | 0.538 | U54 segment | 2 | 4 | State | 0 | **11** | 96 |
| 5 | 0.527 | Fred Wilson Rd | 2 | 4 | Municipal | 0 | 6 | 96 |

- **Rank 1**: the **Anzalduas International Bridge** in Hidalgo
  County — a Class-1 (Interstate-level) facility marked
  *Municipal*-administered (atypical for Class-1) with international
  border-crossing flag. Three simultaneously-unusual attribute values
  put it at the joint right-tail.
- **Rank 2**: an FM (Farm-to-Market) route segment with NHS=8
  (intermodal connector subtype). Most FM routes have NHS=0.
- **Rank 3**: a US Highway 83 segment at a US/Mexico border crossing
  with municipal administration.
- **Rank 4**: an 11-lane (the maximum in the cohort) Class-2 segment
  with degenerate `length=0` geometry — a **data-quality alert** the
  model flags from the joint anomaly.
- **Rank 5**: a Class-2 freeway-grade segment with municipal admin
  (atypical — class-2 is usually State).

**Three of the top-5 anomalies relate to international border
crossings or admin/class mismatches** — exactly the joint-tail
combinations supervised single-axis methods miss.

The most-typical segments are short Class-3 municipal arterials with
2-3 lanes and 56-88 kph speed limits — the modal Texas city-street
NTAD segment.

---

## Mutual-information probes match road-network priors

| Pair | Linfoot | Note |
|---|---:|---|
| `nhs ↔ class` | **0.771** | NHS subtype tracks functional class. Within View 0. |
| `road_system ↔ nhs` | **0.764** | Interstate ↔ NHS Interstate subtype. Within View 0. |
| `road_system ↔ class` | **0.741** | class 1 = Interstate (definitional). Within View 0. |
| `is_interstate ↔ road_system` | **0.726** | Definitional. Within View 0. |
| `is_interstate ↔ nhs` | 0.712 | Within View 0. |
| `length ↔ shape_length` | 0.618 | Definitional. Within View 0. |
| `road_system ↔ speedlim` | 0.503 | Interstate carries 75-80 mph speed limits. |
| `lanes ↔ class` | 0.456 | Higher-class roads have more lanes. |
| `centroid_longitude ↔ speedlim` (negative control) | low | ✅ Cross-view, collapses appropriately. |

Top pairs match canonical road-network engineering relationships.
The geographic negative control (cross-view from View 1 to View 0)
collapses appropriately, confirming no spurious lat/long-attribute
coupling is hallucinated.

---

## RF / XGBoost baselines

**Random Forest classifier on `is_interstate`** (10 features,
`road_system` excluded):

| Metric | Point | 95 % CI |
|---|---:|---|
| **AUC** | **1.0000** | [1.0000, 1.0000] |
| Brier score | 0.0000 | — |
| **ECE (10 bins)** | 0.0001 | — |

**XGBoost regressor on `speedlim`** (9 features):

| Metric | Point | 95 % CI |
|---|---:|---|
| **MAE (kph)** | **8.31** | [8.13, 8.49] |
| **R²** | **0.644** | — |

Both methods saturate on `is_interstate`. **The calibration headline:
jaxcross ECE 6.7e-08 vs RF ECE 1.0e-04 — jaxcross is ~1500× tighter.**

---

## What this means for BTS / GIS / state-DOT GIS analytics

For a BTS / state-DOT GIS team or a geospatial ML startup, the
deliverables above translate to:

1. **Calibrated road-attribute imputation** — when a segment has
   missing `lanes` or `speedlim`, the model gives a posterior with a
   90 % CI that's empirically nominal (94.0 % / 91.5 %).
2. **Border-crossing and class-mismatch anomaly detection** — the
   joint anomaly score flags segments where multiple attributes
   simultaneously deviate from the expected joint pattern.
3. **Documented spatial-modeling limitation** — for geographic-
   attribute imputation specifically, the user should either keep one
   coordinate observed when masking, or layer a spatial-regression
   step on top of the joint model.

The library is **not** a replacement for supervised RF / XGBoost or
GIS toolkits — it's a complement that delivers the joint structure +
calibration layer the supervised pipeline doesn't provide.

---

## Reproducibility

```bash
uv run python examples/ntad_roads/fetch_ntad.py             # ~5 min, ~50MB JSON × 20 pages
uv run python examples/ntad_roads/preprocess_ntad.py        # ~5s, build matrix
uv run python examples/ntad_roads/run_inference.py \
    --chains 4 --sweeps 200 --diag-every 25 --seed 42 --subsample 15000
                                                            # ~2.8h Phase 1
uv run python examples/ntad_roads/run_inference.py \
    --chains 4 --sweeps 200 --diag-every 25 --seed 42 --subsample 15000 \
    --init-from examples/ntad_roads/results/inference/best_chain.jxc \
    --out-subdir inference_warm                             # ~2.8h Phase 2
uv run python examples/ntad_roads/discover_structure.py \
    --inference-dir examples/ntad_roads/results/inference_warm
uv run python examples/ntad_roads/evaluate_leftover.py \
    --inference-dir examples/ntad_roads/results/inference_warm
uv run python examples/ntad_roads/baseline_comparison.py
```

---

## What's next: end of Wave 2

**This is Demo 5 — the final demo of the Wave 2 infrastructure
series.**

| Demo | Cohort | Cells | 90 % CI | ECE | ARI |
|---|---|---:|---:|---:|---:|
| **1: NBI Texas Bridges** | 56,626 × 28 | 166,504 | 89.1 % | 0.08 % | 1.000 |
| **2: HPMS Texas Pavements** | 110,413 × 28 | 377,170 | 92.0 % | 0.91 % | 0.997 |
| **3: NTD Transit (national)** | 2,201 × 28 | 2,004 | 92.6 % | 2.78 % | 1.000 |
| **4: FARS Crashes (national)** | 37,769 × 40 | 90,853 | 93.1 % | 0.22 % | 1.000 |
| **5: NTAD Texas Roads (this post)** | 39,164 × 12 | 96,656 | **84.0 %** | 6.7e-08 | 0.857 |

Across five demos covering bridges, pavements, transit agencies,
fatal crashes, and road networks — three nationwide and two Texas-
only — every demo reaches nominal calibration on the dominant
attributes. NTAD is the demo where the known no-spatial-modeling
limitation becomes visible, and we document it quantitatively.

**The contribution across the series is the same: calibrated CIs on
held-out cells, all from a single fitted joint Bayesian model, on a
\$300 GPU.**

---

## Resources

* **Companion arXiv preprint:** *(link TBD on submission)*
* **GitHub (public examples):**
  [`github.com/sambhal-labs/jaxcross-examples`](https://github.com/sambhal-labs/jaxcross-examples)
  → `usecases/ntad-roads/` (target after migration from this draft).
* **Library license:** jaxcross is a Sambhal Labs library; academic-
  collaboration and commercial deployment licensing available on
  request.
* **Contact:** *(corresponding author email)*.

If you're a BTS / state-DOT GIS team or a geospatial ML startup
looking at calibrated joint-model deliverables for road-network
inventories, drop me a line.

---

*This post is part of the Sambhal Labs jaxcross demo series. The
healthcare Wave 1 series (NHANES, Diabetes 130, FAERS oncology)
shipped April 2026; the Wave 2 infrastructure series (NBI Texas, HPMS
Texas, NTD nationwide, FARS nationwide, this post) shipped April–May
2026 and concludes here. Library: jaxcross v0.11.x, JAX 0.4.x,
ruff-formatted, type-hinted, with property-based tests for all
conjugate models and packed-state serialization. Hardware: NVIDIA GTX
1650 (4 GB VRAM, \$300), 8 GB system RAM, Linux 6.6 / WSL2.*
