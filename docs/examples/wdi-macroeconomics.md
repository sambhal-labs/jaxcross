# WDI Macroeconomic Benchmark

A real-world benchmark analyzing World Bank Development Indicators across ~200 countries. This notebook demonstrates the full production workflow: multi-chain inference with checkpointing, dependence discovery, country-level anomaly detection, similarity clustering, holdout imputation validation, and mutual information analysis.

!!! info "Notebook"
    The full benchmark is at [`benchmarks/wdi_macroeconomic_benchmark.ipynb`](https://github.com/sambhal-labs/jaxcross/blob/main/benchmarks/wdi_macroeconomic_benchmark.ipynb). Run it on Kaggle (P100) for GPU acceleration.

## Dataset

~200 countries with 30+ macroeconomic indicators (all continuous): GDP per capita, life expectancy, education expenditure, CO2 emissions, trade balance, etc. Fetched from the World Bank API via `wbgapi`, with time-collapsed values (most recent non-null per country) and coverage filtering.

```python
import jax.numpy as jnp
import numpy as np

# Data is a float32 array with NaN for missing values
data_jax = jnp.array(df_selected.values.astype(np.float32))
column_types = [ColumnType.CONTINUOUS] * n_cols
```

## Multi-Chain Inference with Checkpointing

The benchmark runs multiple chains with periodic checkpointing for resilience:

```python
from crosscat import initialize, save_checkpoint, load_latest_checkpoint
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat import collect_diagnostics
import gc

N_CHAINS = 4
N_SWEEPS = 200

for chain_idx in range(N_CHAINS):
    # Resume from checkpoint if available
    try:
        packed, _, start_sweep = load_latest_checkpoint(ckpt_dir)
    except FileNotFoundError:
        state = initialize(init_keys[chain_idx], data_jax, column_types)
        packed = pack_state(state)
        start_sweep = 0

    # Run in batches for diagnostics + checkpointing
    for sweep in range(start_sweep, N_SWEEPS, diag_interval):
        batch = min(diag_interval, N_SWEEPS - sweep)
        key, subkey = jax.random.split(key)
        packed = packed_gibbs_sweep(subkey, packed, data_jax, n_sweeps=batch)

        # Temporary unpack for diagnostics (free immediately)
        state_tmp = unpack_state(packed, column_types, data=data_jax)
        diag = collect_diagnostics(state_tmp, data_jax)
        del state_tmp

        # Checkpoint periodically
        save_checkpoint(packed, ckpt_dir, sweep + batch, column_types=column_types)

    del packed
    gc.collect()  # Free GPU memory between chains
```

Key patterns: temporary unpacking for diagnostics, explicit garbage collection for GPU memory, checkpoint/resume for long runs.

## Dependence Discovery (Z-Matrix)

The Z-matrix reveals which economic indicators are jointly modeled:

```python
from crosscat import packed_dependence_matrix

packed_states = [pack_state(s) for s in states]
z_matrix = np.array(packed_dependence_matrix(packed_states))
```

Hierarchical clustering of the Z-matrix reveals block structure — groups of indicators that the model places in the same view (e.g., health indicators cluster together, economic output clusters together).

## Country Anomaly Analysis

Score all countries for typicality in a single vectorized call:

```python
from crosscat import batch_row_typicality

typicality = np.array(
    batch_row_typicality(packed_states, jnp.arange(n_rows))
)

# Most atypical countries
anomaly_df = pd.DataFrame({
    "Country": country_labels,
    "Typicality": typicality,
}).sort_values("Typicality")

print(anomaly_df.head(10))
```

For detailed per-indicator anomaly analysis, use `packed_anomaly_score` on individual countries and compare their indicator values against population medians.

## Country Similarity

Compute pairwise similarity across all countries:

```python
from crosscat import batch_row_similarity

sim_matrix = np.array(
    batch_row_similarity(packed_states, jnp.arange(n_rows))
)
```

Hierarchical clustering of the similarity matrix reveals country groupings that emerge from the data — often aligning with regional and income-level classifications, but sometimes revealing unexpected similarities.

## Holdout Imputation Validation

Test imputation accuracy by masking 10% of observed values:

```python
from crosscat import batch_impute_column
from collections import defaultdict

# Mask random observed cells
data_masked = data_np.copy()
data_masked[holdout_mask] = np.nan
data_masked_jax = jnp.array(data_masked)

# Group holdout cells by column, impute in batches
col_to_rows = defaultdict(list)
for r, c in holdout_cells:
    col_to_rows[int(c)].append(int(r))

for col, rows in col_to_rows.items():
    row_arr = jnp.array(rows)
    key, subkey = jax.random.split(key)
    vals, confs = batch_impute_column(subkey, best_packed, data_masked_jax, col, row_arr)
```

The benchmark reports MAE, RMSE, and correlation between imputed and true held-out values.

## Mutual Information

Quantify predictive relationships between indicator pairs:

```python
from crosscat import packed_mutual_information

mi_val, linfoot = packed_mutual_information(
    packed_states, column_types,
    col_a, col_b,
    rng_key=mi_key,
    n_samples=500,
)
print(f"MI: {float(mi_val):.3f} nats, Linfoot: {float(linfoot):.3f}")
```

The Linfoot correlation is a normalized [0, 1] measure derived from MI — easier to interpret than raw nats.

## Key Takeaways

- **Batch functions** (`batch_anomaly_score`, `batch_row_typicality`, `batch_impute_column`, `batch_row_similarity`) are essential for real-world datasets — they vectorize queries over rows in a single GPU call.
- **Multi-chain inference** with checkpointing provides robust posterior estimates and resilience to interruption.
- **Temporary unpacking** for diagnostics and immediate deletion keeps GPU memory usage manageable.
- **Column-grouped imputation** (batch by column) maximizes JIT reuse since each column shares view structure.
