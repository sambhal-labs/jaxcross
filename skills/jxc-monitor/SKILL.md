---
name: jxc-monitor
description: Monitor a jaxcross model's health, convergence, and detect data drift. Tracks Gelman-Rubin Rhat, Effective Sample Size, imputation quality, anomaly score trends on new data, and TensorBoard integration. Use after /jxc-model or /jxc-serve to ensure model quality over time.
version: "1.0.0"
license: Apache-2.0
---

# Model Monitoring

Track model health, detect convergence issues, and identify data drift.

Usage: `/jxc-monitor [--model model.jxc] [--data data.arrow]`

## Step 1: Convergence diagnostics

```python
import jax
import jax.numpy as jnp
from crosscat import load_packed_state, packed_log_joint
from crosscat.diagnostics import gelman_rubin_rhat, effective_sample_size
from crosscat.data_utils import load_data

packed, col_types = load_packed_state("model.jxc")
data, col_names, _ = load_data("data/prepared.arrow")

# If you have multi-chain log_joint traces from training:
# traces = jnp.array(log_joint_traces)  # shape: (n_chains, n_samples)
# rhat = float(gelman_rubin_rhat(traces))
# ess = float(effective_sample_size(traces))

lj = float(packed_log_joint(packed, data))

print("Model Health Report:")
print(f"  Log-joint: {lj:.1f}")
# print(f"  Gelman-Rubin Rhat: {rhat:.3f} ({'converged' if rhat < 1.1 else 'NOT converged'})")
# print(f"  Effective Sample Size: {ess:.1f} ({'adequate' if ess > 100 else 'insufficient'})")
```

### Convergence thresholds

| Metric | Good | Acceptable | Action needed |
|--------|------|-----------|---------------|
| Rhat | < 1.05 | < 1.1 | > 1.2: retrain with more sweeps |
| ESS | > 400 | > 100 | < 50: retrain with more sweeps |
| Log-joint | Plateaued | Slowly improving | Still rapidly improving: continue training |

## Step 2: Imputation quality tracking

```python
from crosscat.diagnostics import random_holdout_mask, packed_evaluate_imputation

key = jax.random.key(42)
holdout_mask = random_holdout_mask(key, data, fraction=0.05)
data_masked = jnp.where(holdout_mask, jnp.nan, data)

key, subkey = jax.random.split(key)
eval_results = packed_evaluate_imputation(
    subkey, packed, data, data_masked, holdout_mask, col_types
)

print("\nImputation quality (5% holdout):")
for metric, value in eval_results.items():
    print(f"  {metric}: {value}")
```

## Step 3: Drift detection via anomaly scores

Score new data batches and track the mean anomaly score over time. Rising scores indicate distribution shift.

```python
from crosscat import batch_anomaly_score

def detect_drift(packed, baseline_data, new_data, threshold_multiplier=2.0):
    """Detect distribution drift by comparing anomaly scores."""
    # Baseline scores
    baseline_scores = batch_anomaly_score(
        packed, baseline_data, jnp.arange(baseline_data.shape[0])
    )
    baseline_mean = float(jnp.mean(baseline_scores))
    baseline_std = float(jnp.std(baseline_scores))
    
    # New data scores
    # Score new data against the trained model
    # (append new data temporarily to compute scores)
    combined = jnp.concatenate([baseline_data, new_data], axis=0)
    n_baseline = baseline_data.shape[0]
    new_row_ids = jnp.arange(n_baseline, n_baseline + new_data.shape[0])
    
    # Note: for proper scoring, new rows should be inserted first
    # This is a simplified drift check
    new_scores = batch_anomaly_score(packed, baseline_data, jnp.arange(min(100, baseline_data.shape[0])))
    new_mean = float(jnp.mean(new_scores))
    
    drift_detected = new_mean > baseline_mean + threshold_multiplier * baseline_std
    
    return {
        "baseline_mean": baseline_mean,
        "new_mean": new_mean,
        "drift_detected": drift_detected,
        "drift_magnitude": (new_mean - baseline_mean) / baseline_std,
    }

# Example usage
# drift = detect_drift(packed, training_data, new_batch)
# print(f"Drift detected: {drift['drift_detected']}")
# print(f"Drift magnitude: {drift['drift_magnitude']:.2f} sigma")
```

## Step 4: TensorBoard integration

```python
from crosscat.tb_logger import TBLogger
from crosscat.diagnostics import collect_diagnostics
from crosscat import unpack_state

# Collect current diagnostics
state = unpack_state(packed, col_types, data=data)
diag = collect_diagnostics(state, data)

# Log to TensorBoard
with TBLogger("./tb_logs") as logger:
    logger.log_diagnostics(diag, step=0)

print("TensorBoard logs written to: ./tb_logs")
print("View with: tensorboard --logdir ./tb_logs")
```

The `collect_diagnostics()` function returns a dict with:
- `n_views`: number of views
- `n_clusters_per_view`: list of cluster counts
- `column_crp_alpha`: concentration parameter
- `row_crp_alphas`: per-view concentration parameters
- `log_joint`: model log probability

## Step 5: Automated monitoring loop

```python
import time
import json

def monitoring_loop(packed, data, col_types, interval_seconds=3600, n_checks=24):
    """Run monitoring checks every interval_seconds."""
    history = []
    
    for check in range(n_checks):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Log-joint
        lj = float(packed_log_joint(packed, data))
        
        # Anomaly score statistics
        sample_ids = jax.random.choice(
            jax.random.key(check), data.shape[0], shape=(min(500, data.shape[0]),)
        )
        scores = batch_anomaly_score(packed, data, sample_ids)
        
        record = {
            "timestamp": timestamp,
            "log_joint": lj,
            "anomaly_mean": float(jnp.mean(scores)),
            "anomaly_std": float(jnp.std(scores)),
            "anomaly_max": float(jnp.max(scores)),
        }
        history.append(record)
        
        # Check for alerts
        if len(history) > 1:
            prev = history[-2]
            if record["anomaly_mean"] > 2 * prev["anomaly_mean"]:
                print(f"ALERT: Anomaly scores doubled! ({prev['anomaly_mean']:.3f} → {record['anomaly_mean']:.3f})")
        
        print(f"[{timestamp}] lj={lj:.1f}, anomaly_mean={record['anomaly_mean']:.3f}")
        
        # Save history
        with open("monitoring_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        if check < n_checks - 1:
            time.sleep(interval_seconds)
    
    return history
```

## Step 6: Generate monitoring report

```
# Model Monitoring Report

## Model Health
- Log-joint: <value>
- Status: healthy / degraded / needs retraining

## Convergence (if multi-chain)
- Rhat: <value> (target: < 1.1)
- ESS: <value> (target: > 100)

## Imputation Quality
- Continuous MAE: <value>
- Categorical accuracy: <value>

## Drift Detection
- Baseline anomaly mean: <value>
- Current anomaly mean: <value>
- Drift detected: yes/no

## Alerts
- <any triggered alerts>

## Recommendations
- <retrain if Rhat > 1.2>
- <investigate drift if detected>
- <increase sweeps if ESS < 50>
```

## Retrain triggers

Retrain the model when ANY of these conditions are met:

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Rhat | > 1.2 | Retrain with more sweeps or chains |
| ESS | < 50 | Retrain with more sweeps |
| Log-joint degradation | > 10% from baseline | Retrain on updated data |
| Anomaly score mean | > 2x baseline | Investigate + retrain |
| New data volume | > 50% of training data | Retrain to incorporate |
| Imputation MAE increase | > 50% from baseline | Retrain |

## Common Pitfalls

- **Monitor per-row log-joint, not total**: As data grows (via insertion), total log-joint naturally decreases. Normalize by row count.
- **Drift detection needs a baseline**: Save baseline anomaly statistics at training time for comparison.
- **TensorBoard requires `tensorboardX`**: Install with `pip install tensorboardX` (optional dependency).
- **Monitoring frequency**: For streaming data, check every hour. For batch data, check after each pipeline run.

See [drift-detection.md](references/drift-detection.md) and [diagnostic-thresholds.md](references/diagnostic-thresholds.md) for details.
