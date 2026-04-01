# Scientific Data Exploration

Discover relationships between variables in scientific datasets without assuming a model structure. Ideal for exploratory analysis of genomics, economics, environmental, and sensor data.

## The Challenge

Scientific datasets often have many variables with unknown relationships. Researchers need to discover which variables are related, how they cluster, and what structure exists — before committing to specific hypotheses or models.

## Why CrossCat Fits

- **Nonparametric** — discovers the number of variable groups and clusters automatically
- **Structure discovery** — the dependence matrix reveals which variables carry information about each other
- **Mixed types** — handles continuous measurements, categorical labels, ordinal scales, and cyclic quantities (e.g., angles, time of day)
- **Hypothesis generation** — use discovered structure to inform follow-up statistical analyses

## Example: Macroeconomic Data

The [WDI Macroeconomics benchmark](../examples/wdi-macroeconomics.md) demonstrates this workflow on World Bank data (~200 countries, 30+ economic indicators).

```python
import jax
from crosscat import initialize, dependence_matrix, mutual_information
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat.types import ColumnType

# All continuous economic indicators
col_types = [ColumnType.CONTINUOUS] * n_cols

# Multi-chain inference for robust results
states = []
for i in range(4):
    state = initialize(jax.random.key(i), data, col_types)
    packed = pack_state(state)
    packed = packed_gibbs_sweep(jax.random.key(i + 100), packed, data, n_sweeps=100)
    states.append(unpack_state(packed, col_types, data=data))

# Discover variable relationships
z_matrix = dependence_matrix(states)
# Reveals clusters like:
#   {GDP_per_capita, life_expectancy, education_spend} — development indicators
#   {CO2_emissions, energy_use, industry_share} — industrialization indicators
#   {trade_balance, FDI, exports} — trade indicators

# Quantify specific relationships
mi = mutual_information(states, col_i=0, col_j=5, n_samples=1000)
print(f"MI between GDP and life expectancy: {mi:.3f}")
```

## What You Get

1. **Dependence matrix (Z-matrix)** — a heatmap showing which variables are statistically related, revealing block structure in your data
2. **Variable groupings (views)** — which sets of variables cluster together, suggesting latent factors
3. **Row clusters** — within each variable group, how observations cluster (e.g., country development tiers)
4. **Mutual information** — quantitative measure of pairwise variable dependence
5. **Anomaly detection** — identify unusual observations (e.g., countries that don't fit expected patterns)
6. **Imputation** — predict missing measurements with uncertainty quantification

## Applications

- **Genomics** — discover gene expression modules, identify co-regulated genes
- **Economics** — find latent development factors, cluster countries by economic profile
- **Environmental science** — relate climate variables, identify sensor anomalies
- **Clinical data** — discover symptom clusters, find patient subtypes
- **Sensor networks** — identify related sensor channels, detect sensor drift

## Tips

- **Always use multi-chain** for scientific analysis — single chains may get stuck in local modes
- The **dependence matrix** is your primary output for exploration — visualize it as a heatmap and look for block structure
- Use **column typicality** to identify which variables are most tightly integrated into the model structure
- For large datasets (1000+ rows), the packed path makes this feasible on GPU — see the [GPU guide](../guides/gpu-packed.md)
- **Compare with domain knowledge** — if known relationships don't appear, run more sweeps or add more chains
