# Materials Property Discovery

Predict expensive-to-compute material properties from cheap structural and compositional features. Screen 150K+ candidate materials with calibrated uncertainty — no supervised labels, no feature engineering.

## The Challenge

High-throughput materials databases like [Materials Project](https://materialsproject.org/) contain 150,000+ materials, but expensive properties — dielectric constants (DFPT), elasticity tensors, piezoelectric coefficients — are available for only a small fraction. DFPT dielectric calculations cost 5-10x more than standard DFT relaxation, creating a bottleneck for materials screening.

Supervised ML approaches (random forests, graph neural networks) can predict individual properties with high accuracy, but they require curated feature sets, separate models per target, and cannot quantify prediction uncertainty or discover which material properties are fundamentally related.

## Why CrossCat Fits

- **Joint structure discovery** — learns which material properties are statistically related (e.g., ionic and total dielectric form a cluster, elastic moduli form another), without domain-specific feature engineering
- **Calibrated uncertainty** — credible intervals are well-calibrated (99.7% coverage at 99% CI), so you know which predictions to trust for experimental follow-up
- **Native sparsity handling** — materials databases have natural block-missingness (only 28% have elasticity data). CrossCat handles NaN transparently, no imputation preprocessing needed
- **Mixed column types** — continuous properties, binary flags (is_metal), categorical (crystal system), and ordinal (Laue class) are modeled jointly in a single model

## Workflow

```python
import jax
import jax.numpy as jnp
import numpy as np
from crosscat import initialize, batch_impute_column, batch_credible_interval
from crosscat import packed_dependence_matrix, packed_insert_rows
from crosscat.packed import pack_state, packed_gibbs_step
from crosscat.serialization import save_packed_state, load_packed_state
from crosscat.types import ColumnType

# 23 material properties: band gap, dielectric (3), elastic (4), structural, ...
col_types = [
    ColumnType.CONTINUOUS,    # band_gap
    ColumnType.BINARY,        # is_metal
    ColumnType.CONTINUOUS,    # e_electronic
    ColumnType.CONTINUOUS,    # e_ionic (target)
    ColumnType.CONTINUOUS,    # e_total
    ColumnType.CONTINUOUS,    # formation_energy
    ColumnType.CONTINUOUS,    # energy_above_hull
    ColumnType.BINARY,        # is_stable
    ColumnType.CONTINUOUS,    # density
    ColumnType.CONTINUOUS,    # volume
    ColumnType.CONTINUOUS,    # nsites
    ColumnType.CONTINUOUS,    # nelements
    ColumnType.CATEGORICAL,   # crystal_system (7 values)
    ColumnType.CONTINUOUS,    # bulk_modulus
    ColumnType.CONTINUOUS,    # shear_modulus
    ColumnType.CONTINUOUS,    # elastic_anisotropy
    ColumnType.CONTINUOUS,    # poisson_ratio
    ColumnType.CONTINUOUS,    # piezo_e_ij_max
    ColumnType.CONTINUOUS,    # avg_electronegativity
    ColumnType.CONTINUOUS,    # avg_ionic_radius
    ColumnType.ORDINAL,       # laue_class (11 values)
    ColumnType.CONTINUOUS,    # magnetization
    ColumnType.CATEGORICAL,   # magnetic_ordering
]

# Train on 7,327 materials with dielectric data
data_jax = jnp.array(train_data, dtype=jnp.float32)  # (7327, 23)
key = jax.random.key(42)
result = initialize(key, data_jax, col_types)
packed = pack_state(result.state, max_views=16, max_clusters=32, data=data_jax)

# Run 4 independent chains from checkpoint
for sweep in range(100):
    key, subkey = jax.random.split(key)
    packed = packed_gibbs_step(subkey, packed, data_jax)

# Discover property relationships
all_chains = [chain_0, chain_1, chain_2, chain_3]  # loaded from checkpoints
z_matrix = np.array(packed_dependence_matrix(all_chains))

# Predict dielectric for 147K new materials via row insertion
key, k_insert, k_impute = jax.random.split(key, 3)
packed_aug, data_aug = packed_insert_rows(k_insert, packed, data_jax, new_materials)
predictions, confidence = batch_impute_column(
    k_impute, packed_aug, data_aug,
    query_col=3,  # ionic dielectric
    row_ids=jnp.arange(len(train_data), len(train_data) + len(new_materials)),
)

# Uncertainty quantification: 99% credible intervals
medians, ci_lo, ci_hi = batch_credible_interval(
    jax.random.key(99), packed_aug, data_aug,
    query_col=3, row_ids=candidate_rows, ci_level=0.99,
)
```

## What You Get

1. **Structure discovery** — 5 physically meaningful views: structural/thermodynamic, electronic/mechanical, dielectric pair, piezoelectric, elastic anisotropy
2. **Dielectric predictions** — ionic dielectric at R²=0.81, outperforming MICE (R²=0.48) and reaching 88% of Random Forest accuracy (R²=0.92) — without any feature engineering
3. **Calibrated credible intervals** — 90% CI coverage of 95.6%, 99% CI coverage of 99.7%. The model reliably communicates when it is uncertain
4. **49,566 high-confidence predictions** — out of 147K candidate materials, filtered by cross-chain agreement (Bayesian Model Averaging over 4 chains)
5. **Anomaly detection** — materials with unusual property combinations flagged for experimental investigation
6. **Mutual information** — quantify nonlinear relationships between property pairs (e.g., bulk modulus and shear modulus: Linfoot=0.94)

## Tips

- **Use multi-chain BMA for predictions on new materials** — insert rows into each chain independently, average predictions, and use cross-chain standard deviation as an uncertainty measure
- **Run 300+ total sweeps for materials data** — the 23-column mixed-type structure needs more sweeps than simpler datasets to stabilize view assignments
- **`packed_gibbs_step` over `packed_gibbs_sweep`** on consumer GPUs (4GB VRAM) — the step variant uses 4 smaller JIT compilations instead of one large `lax.scan`, fitting within memory constraints
- See the [full Materials Project example](../examples/materials-project.md) for the complete 8-script pipeline with data fetching, preprocessing, and PDF report generation
