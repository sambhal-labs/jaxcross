# Threshold Selection Guide

## Percentile-based (simple, robust)

```python
# Top 1%
threshold = float(jnp.percentile(scores, 99))

# Top 5%
threshold = float(jnp.percentile(scores, 95))

# Top N rows
N = 100
threshold = float(jnp.sort(scores)[-N])
```

**Pros**: Simple, deterministic, always returns a fixed proportion.
**Cons**: Always flags exactly N% of rows, even if there are no real anomalies.

## Statistical (z-score based)

```python
mean = float(jnp.mean(scores))
std = float(jnp.std(scores))

# 2-sigma (flags ~2.3% if normally distributed)
threshold = mean + 2 * std

# 3-sigma (flags ~0.1%)
threshold = mean + 3 * std
```

**Pros**: Adapts to the score distribution — flags fewer rows when data is homogeneous.
**Cons**: Assumes approximately normal distribution of scores.

## IQR-based (robust to heavy tails)

```python
q75 = float(jnp.percentile(scores, 75))
q25 = float(jnp.percentile(scores, 25))
iqr = q75 - q25
threshold = q75 + 1.5 * iqr  # Standard box plot rule
```

**Pros**: Robust to outliers in the score distribution itself.
**Cons**: May be too aggressive for heavy-tailed distributions.

## Domain-specific (recommended for production)

Let domain experts set the threshold based on the cost of false positives vs false negatives:

```python
# If false positives are expensive (e.g., human review per flagged row)
threshold = float(jnp.percentile(scores, 99.5))  # Very conservative

# If false negatives are expensive (e.g., fraud detection)
threshold = float(jnp.percentile(scores, 90))  # Aggressive, review more
```

## Evaluating threshold quality

If you have labeled anomalies (ground truth):

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

y_true = known_anomaly_labels  # 0 or 1
y_scores = np.array(scores)

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)
print(f"Average Precision: {ap:.3f}")
```
