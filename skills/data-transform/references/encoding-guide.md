# Encoding Guide

## Categorical Encoding Strategies

### Standard: Sorted unique values
```python
unique_vals = sorted(df[col].dropna().unique())
mapping = {v: i for i, v in enumerate(unique_vals)}
```
Best for: most cases. Deterministic, reproducible.

### Frequency-based ordering
```python
freq_order = df[col].value_counts().index.tolist()
mapping = {v: i for i, v in enumerate(freq_order)}
```
Best for: when you want the most common category to be 0.

### Handling unseen categories (new data)
```python
# Load saved encoding
with open("encodings.json") as f:
    saved = json.load(f)

mapping = saved[col]
# Map, replacing unseen with NaN
df[col] = df[col].map(mapping)  # Unseen values become NaN
n_unseen = df[col].isna().sum() - original_nan_count
if n_unseen > 0:
    print(f"WARNING: {n_unseen} unseen categories in '{col}' mapped to NaN")
```

### High-cardinality columns (>50 categories)
Options:
1. **Keep as-is**: jaxcross handles it, but `max_categories` must be large enough
2. **Group rare categories**: Map categories with <N occurrences to "OTHER"
3. **Drop the column**: If cardinality is too high to be informative
4. **Discretize**: For ordered high-cardinality (e.g., age → age groups)

```python
# Group rare categories
counts = df[col].value_counts()
rare = counts[counts < 10].index
df[col] = df[col].replace(rare, "OTHER")
```

## Ordinal Encoding

The key is defining the correct order. Common ordinal scales:

```python
ordinal_scales = {
    "education": ["none", "primary", "secondary", "bachelors", "masters", "doctorate"],
    "income_bracket": ["low", "lower_middle", "middle", "upper_middle", "high"],
    "satisfaction": ["very_dissatisfied", "dissatisfied", "neutral", "satisfied", "very_satisfied"],
    "frequency": ["never", "rarely", "sometimes", "often", "always"],
    "severity": ["none", "mild", "moderate", "severe", "critical"],
    "size": ["xs", "s", "m", "l", "xl", "xxl"],
    "priority": ["low", "medium", "high", "critical"],
    "agreement": ["strongly_disagree", "disagree", "neutral", "agree", "strongly_agree"],
}
```

## Cyclic Encoding Details

### Why radians?
The Von Mises distribution (jaxcross CYCLIC type) operates on radians in [0, 2*pi). Values near 0 and near 2*pi are close together on the circle.

### Conversion formulas
```python
# General formula
radians = ((value - offset) % period) / period * 2 * np.pi

# Hour of day (0-23)
radians = hour * 2 * np.pi / 24

# Day of week (0=Monday, 6=Sunday)
radians = dow * 2 * np.pi / 7

# Month (1-12)
radians = (month - 1) * 2 * np.pi / 12

# Compass bearing (0-360 degrees)
radians = degrees * np.pi / 180

# Day of year (1-366)
radians = (day - 1) * 2 * np.pi / 365.25
```

### Verify encoding
After encoding, all values should be in [0, 2*pi):
```python
assert (df[col] >= 0).all() and (df[col] < 2 * np.pi).all()
```
