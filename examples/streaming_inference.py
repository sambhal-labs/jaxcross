"""Streaming inference example — train, stream new data, query.

Demonstrates JAX-CrossCat's online learning capabilities:
1. Train on an initial batch of synthetic data
2. Stream new rows in batches (simulating real-time data arrival)
3. Run incremental Gibbs sweeps after each batch
4. Query the updated model (anomaly detection, dependence, imputation)

Designed to run on a GTX 1650 (4GB VRAM) in ~5 minutes.
"""

import time

import jax
import jax.numpy as jnp

from crosscat import (
    dependence_matrix,
    impute_and_confidence,
    initialize,
    log_joint,
    predictive_anomalousness,
    predictive_sample,
)
from crosscat.packed import (
    pack_state,
    packed_gibbs_sweep,
    packed_insert_rows,
    unpack_state,
)
from crosscat.types import ColumnType

# ---------------------------------------------------------------------------
# 1. Generate synthetic data with known structure
# ---------------------------------------------------------------------------

print("=" * 60)
print("JAX-CrossCat Streaming Inference Example")
print("=" * 60)
print(f"\nBackend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}\n")

key = jax.random.key(42)

# Simulate employee data: salary, years_exp, department, is_manager
# Two clusters: junior (low salary, few years, dept 0-1) and senior
n_initial = 50
n_cols = 4
column_types = [
    ColumnType.CONTINUOUS,  # salary
    ColumnType.CONTINUOUS,  # years_experience
    ColumnType.CATEGORICAL,  # department (0, 1, 2)
    ColumnType.BINARY,  # is_manager
]
col_names = ["salary", "years_exp", "department", "is_manager"]

# Generate two clusters
key, k1, k2 = jax.random.split(key, 3)

# Junior cluster (60% of data)
n_junior = int(n_initial * 0.6)
junior = jnp.column_stack(
    [
        40000 + jax.random.normal(k1, (n_junior,)) * 5000,  # salary ~40k
        1 + jax.random.exponential(k1, (n_junior,)) * 2,  # 1-5 years
        jax.random.categorical(k1, jnp.log(jnp.array([0.5, 0.4, 0.1])), shape=(n_junior,)).astype(
            jnp.float32
        ),
        jax.random.bernoulli(k1, 0.05, (n_junior,)).astype(jnp.float32),  # rarely manager
    ]
)

# Senior cluster (40% of data)
n_senior = n_initial - n_junior
senior = jnp.column_stack(
    [
        85000 + jax.random.normal(k2, (n_senior,)) * 10000,  # salary ~85k
        8 + jax.random.exponential(k2, (n_senior,)) * 3,  # 8-15 years
        jax.random.categorical(k2, jnp.log(jnp.array([0.2, 0.3, 0.5])), shape=(n_senior,)).astype(
            jnp.float32
        ),
        jax.random.bernoulli(k2, 0.6, (n_senior,)).astype(jnp.float32),  # often manager
    ]
)

data = jnp.concatenate([junior, senior], axis=0)
print(f"Initial data: {data.shape[0]} rows x {data.shape[1]} cols")
print(f"Column types: {[ct.value for ct in column_types]}")
print(f"Columns: {col_names}\n")

# ---------------------------------------------------------------------------
# 2. Train initial model
# ---------------------------------------------------------------------------

print("-" * 60)
print("Phase 1: Initial Training")
print("-" * 60)

key, k_init, k_sweep = jax.random.split(key, 3)
state = initialize(k_init, data, column_types)
packed = pack_state(state, max_views=8, max_clusters=16)

t0 = time.time()
packed = packed_gibbs_sweep(k_sweep, packed, data, n_sweeps=30)
t1 = time.time()
print(f"30 sweeps in {t1 - t0:.1f}s (includes JIT compilation)")

state = unpack_state(packed, column_types, data=data)
lj = log_joint(state, data)
print(f"Log joint: {float(lj):.1f}")
print(
    f"Discovered {state.n_views} views, "
    f"{[len(v.suffstats) for v in state.views]} clusters per view"
)

# Initial dependence structure
z = dependence_matrix([state])
print("\nDependence matrix (Z-matrix):")
for i, name_i in enumerate(col_names):
    row = " ".join(f"{z[i, j]:.2f}" for j in range(n_cols))
    print(f"  {name_i:12s}  {row}")

# ---------------------------------------------------------------------------
# 3. Stream new data in batches
# ---------------------------------------------------------------------------

print("\n" + "-" * 60)
print("Phase 2: Streaming Inference")
print("-" * 60)

n_batches = 5
batch_size = 10
sweeps_per_batch = 5

for batch_idx in range(n_batches):
    key, k_batch, k_sweep = jax.random.split(key, 3)

    # Simulate incoming batch — mix of junior and senior employees
    n_j = batch_size // 2
    n_s = batch_size - n_j
    k_j, k_s = jax.random.split(k_batch)

    new_junior = jnp.column_stack(
        [
            40000 + jax.random.normal(k_j, (n_j,)) * 5000,
            1 + jax.random.exponential(k_j, (n_j,)) * 2,
            jax.random.categorical(k_j, jnp.log(jnp.array([0.5, 0.4, 0.1])), shape=(n_j,)).astype(
                jnp.float32
            ),
            jax.random.bernoulli(k_j, 0.05, (n_j,)).astype(jnp.float32),
        ]
    )
    new_senior = jnp.column_stack(
        [
            85000 + jax.random.normal(k_s, (n_s,)) * 10000,
            8 + jax.random.exponential(k_s, (n_s,)) * 3,
            jax.random.categorical(k_s, jnp.log(jnp.array([0.2, 0.3, 0.5])), shape=(n_s,)).astype(
                jnp.float32
            ),
            jax.random.bernoulli(k_s, 0.6, (n_s,)).astype(jnp.float32),
        ]
    )
    new_rows = jnp.concatenate([new_junior, new_senior], axis=0)

    # Insert new rows into packed state
    t0 = time.time()
    packed, data = packed_insert_rows(k_batch, packed, data, new_rows)

    # Run incremental sweeps to incorporate new data
    packed = packed_gibbs_sweep(k_sweep, packed, data, n_sweeps=sweeps_per_batch)
    t1 = time.time()

    state = unpack_state(packed, column_types, data=data)
    lj = log_joint(state, data)
    print(
        f"Batch {batch_idx + 1}: +{batch_size} rows → {data.shape[0]} total | "
        f"log_joint={float(lj):.1f} | {t1 - t0:.1f}s"
    )

# ---------------------------------------------------------------------------
# 4. Query the updated model
# ---------------------------------------------------------------------------

print("\n" + "-" * 60)
print("Phase 3: Posterior Queries on Updated Model")
print("-" * 60)

state = unpack_state(packed, column_types, data=data)
print(f"\nFinal state: {data.shape[0]} rows, {state.n_views} views")

# Anomaly detection — flag unusual employees
print("\n--- Anomaly Detection ---")
key, k_anom = jax.random.split(key)
for row_idx in [0, n_initial - 1, data.shape[0] - 1]:
    score = predictive_anomalousness(k_anom, state, data, query_row=row_idx)
    print(
        f"Row {row_idx}: anomaly={float(score):.3f} | "
        f"salary={float(data[row_idx, 0]):.0f}, "
        f"years={float(data[row_idx, 1]):.1f}, "
        f"dept={int(data[row_idx, 2])}, "
        f"mgr={int(data[row_idx, 3])}"
    )

# Test a truly anomalous row — high salary but junior
print("\n--- Scoring a Synthetic Anomaly ---")
anomalous_row = jnp.array([[120000.0, 1.0, 0.0, 0.0]])
temp_packed, temp_data = packed_insert_rows(k_anom, packed, data, anomalous_row)
temp_state = unpack_state(temp_packed, column_types, data=temp_data)
anomaly_score = predictive_anomalousness(
    k_anom, temp_state, temp_data, query_row=temp_data.shape[0] - 1
)
print(f"Anomaly (120k salary, 1yr exp): score={float(anomaly_score):.3f}")

# Imputation — predict missing salary
print("\n--- Imputation ---")
key, k_imp = jax.random.split(key)
value, confidence = impute_and_confidence(k_imp, state, data, query_col=0)
print(f"Predicted salary: ${float(value):,.0f} (confidence: {float(confidence):.2f})")

# Predictive sampling
print("\n--- Predictive Sampling ---")
key, k_samp = jax.random.split(key)
samples = predictive_sample(k_samp, state, data, query_cols=[0], n_samples=1000)
print(
    f"Salary distribution: mean=${float(jnp.mean(samples)):,.0f}, "
    f"std=${float(jnp.std(samples)):,.0f}, "
    f"min=${float(jnp.min(samples)):,.0f}, "
    f"max=${float(jnp.max(samples)):,.0f}"
)

# Final dependence matrix
print("\n--- Updated Dependence Matrix ---")
z = dependence_matrix([state])
for i, name_i in enumerate(col_names):
    row = " ".join(f"{z[i, j]:.2f}" for j in range(n_cols))
    print(f"  {name_i:12s}  {row}")

print("\n" + "=" * 60)
print("Streaming inference complete!")
print("=" * 60)
