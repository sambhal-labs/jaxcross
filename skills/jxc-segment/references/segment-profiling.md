# Segment Profiling Guide

## Profile statistics by column type

### Continuous
- Mean, median, std, min, max
- Distribution shape: skewness, modality
- Compare to global distribution — what makes this segment different?

### Categorical
- Mode (most common category)
- Top-3 categories with proportions
- Entropy (how concentrated vs spread across categories)

### Binary
- Positive rate (proportion of 1s)
- Compare to global positive rate

### Ordinal
- Median (middle rank)
- Range (min to max ordinal value)

### Cyclic
- Circular mean: `atan2(mean(sin(x)), mean(cos(x)))`
- Circular std: `sqrt(-2 * log(R))` where `R = sqrt(mean(sin(x))^2 + mean(cos(x))^2)`

## Segment differentiation

Identify what makes each segment unique by comparing to the global distribution:

```python
for j in view.column_indices:
    global_mean = np.nanmean(data[:, j])
    global_std = np.nanstd(data[:, j])
    
    segment_mean = np.nanmean(data[mask, j])
    
    # Z-score of segment mean vs global
    z = (segment_mean - global_mean) / (global_std / np.sqrt(mask.sum()))
    
    if abs(z) > 2:
        direction = "higher" if z > 0 else "lower"
        print(f"  {col_names[j]}: significantly {direction} than average (z={z:.1f})")
```

## Naming segments

Give segments human-readable names based on their profiles:

```python
# Example: auto-generate names from top features
for cluster_id in cluster_ids:
    mask = assignments == cluster_id
    descriptors = []
    for j in view.column_indices:
        global_mean = np.nanmean(data[:, j])
        seg_mean = np.nanmean(data[mask, j])
        if seg_mean > global_mean + np.nanstd(data[:, j]):
            descriptors.append(f"high-{col_names[j]}")
        elif seg_mean < global_mean - np.nanstd(data[:, j]):
            descriptors.append(f"low-{col_names[j]}")
    
    name = ", ".join(descriptors[:3]) or "average"
    print(f"  Segment {cluster_id}: '{name}' ({mask.sum()} rows)")
```
