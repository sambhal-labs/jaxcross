# Convergence Diagnostics

## What

Monitor inference progress and evaluate model quality using log-joint tracking, Adjusted Rand Index (ARI), and held-out evaluation.

## When to Use

- Deciding when to stop running sweeps
- Comparing chains or configurations
- Validating recovery on synthetic data with known ground truth

## Log-Joint Tracking

```python
from crosscat import log_joint

for batch in range(20):
    packed = packed_gibbs_sweep(key, packed, data, n_sweeps=5)
    state = unpack_state(packed, col_types, data=data)
    score = log_joint(state, data)
    print(f"Sweep {(batch+1)*5}: log_joint={score:.1f}, views={state.n_views}")
```

The log-joint should increase and plateau. If it's still climbing, run more sweeps.

## Per-Sweep Diagnostics

```python
from crosscat.diagnostics import collect_diagnostics

diag = collect_diagnostics(state, data)
print(f"Log joint: {diag['log_joint']:.1f}")
print(f"Views: {diag['n_views']}")
print(f"Clusters per view: {diag['n_clusters_per_view']}")
```

## Adjusted Rand Index (Synthetic Data)

When you have ground truth assignments (e.g., from synthetic data), compare with ARI:

```python
from crosscat.diagnostics import column_partition_ari, row_partition_ari

# Column partition recovery
col_ari = column_partition_ari(state, true_col_assignments)
print(f"Column ARI: {col_ari:.3f}")  # 1.0 = perfect

# Row partition recovery per view
for v in range(state.n_views):
    for true_assigns in true_row_assignments:
        ari = row_partition_ari(state, v, true_assigns)
        print(f"  View {v} row ARI: {ari:.3f}")
```

## General ARI

```python
from crosscat.diagnostics import adjusted_rand_index

ari = adjusted_rand_index(true_labels, predicted_labels)
```

## Held-Out Evaluation

Test model quality by holding out random cells and measuring imputation accuracy:

```python
from crosscat.diagnostics import random_holdout_mask, evaluate_imputation

# Create holdout mask (10% of cells)
mask = random_holdout_mask(key, data.shape[0], data.shape[1], holdout_fraction=0.1)

# Evaluate
metrics = evaluate_imputation(state, data, mask, col_types, rng_key=key)
# Returns: MAE (continuous), accuracy (discrete), log-likelihood
```

## Mean Test Log-Likelihood

```python
from crosscat.diagnostics import mean_test_log_likelihood

# Score on specific test rows
test_rows = jnp.array([100, 101, 102, 103, 104])
mll = mean_test_log_likelihood(state, data, test_rows)
print(f"Mean test log-likelihood: {mll:.3f}")
```

## API Reference

- [`collect_diagnostics`](../api/diagnostics.md#collect_diagnostics)
- [`column_partition_ari`](../api/diagnostics.md#column_partition_ari)
- [`row_partition_ari`](../api/diagnostics.md#row_partition_ari)
- [`evaluate_imputation`](../api/diagnostics.md#evaluate_imputation)
- [`mean_test_log_likelihood`](../api/diagnostics.md#mean_test_log_likelihood)
