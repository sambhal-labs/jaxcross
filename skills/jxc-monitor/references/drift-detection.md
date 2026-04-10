# Drift Detection Guide

## What is distribution drift?

Distribution drift occurs when new data no longer follows the same distribution as the training data. This causes model predictions to become unreliable.

## Detection strategies

### Anomaly score trending
The simplest approach: score new data batches with `batch_anomaly_score()` and track the mean over time.

```python
# Establish baseline
baseline_scores = batch_anomaly_score(packed, data, jnp.arange(data.shape[0]))
baseline_mean = float(jnp.mean(baseline_scores))
baseline_std = float(jnp.std(baseline_scores))

# Score new batch
new_scores = batch_anomaly_score(packed, data, new_row_ids)
new_mean = float(jnp.mean(new_scores))

# Drift if new mean is significantly higher
z_score = (new_mean - baseline_mean) / baseline_std
if z_score > 2:
    print(f"Drift detected: z={z_score:.1f}")
```

### Per-column monitoring
Track distribution statistics per column on new data:
- Continuous: mean, std shift
- Categorical: category frequency changes
- Binary: positive rate shift

### Log-joint monitoring
Track per-row log-joint over time:
```python
per_row_lj = packed_log_joint(packed, data) / data.shape[0]
```
A decreasing trend indicates the model is becoming a worse fit.

## Response to drift

| Drift severity | Action |
|---------------|--------|
| Mild (z < 3) | Continue monitoring, increase check frequency |
| Moderate (3 < z < 5) | Insert new data + 50-100 incremental sweeps |
| Severe (z > 5) | Full retrain with new data |
| Structural (new categories, new column distributions) | Full retrain required |
