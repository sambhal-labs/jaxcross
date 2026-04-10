---
name: data-quality
description: Profile, validate, and assess tabular data quality before modeling. Detects schema issues, missing data patterns, distribution anomalies, outliers, type mismatches, and jaxcross column type compatibility. Use when preparing data for any ML pipeline or before running /data-transform.
version: "1.0.0"
license: Apache-2.0
---

# Data Quality Assessment

Profile and validate tabular data, producing a structured quality report with actionable recommendations.

Usage: `/data-quality <file_path>`

Examples:
- `/data-quality data/raw_data.csv`
- `/data-quality data/transactions.parquet`

## Step 1: Load and inspect

```python
import pandas as pd
import numpy as np

# Auto-detect format
file_path = "<user_provided_path>"
if file_path.endswith(".parquet"):
    df = pd.read_parquet(file_path)
elif file_path.endswith(".arrow") or file_path.endswith(".feather"):
    df = pd.read_feather(file_path)
elif file_path.endswith((".xlsx", ".xls")):
    df = pd.read_excel(file_path)
else:
    df = pd.read_csv(file_path)

print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nFirst 5 rows:")
print(df.head())
```

## Step 2: Schema validation

Check for structural issues that must be fixed before modeling:

```python
issues = []

# Duplicate column names
dupes = df.columns[df.columns.duplicated()].tolist()
if dupes:
    issues.append(f"CRITICAL: Duplicate column names: {dupes}")

# Constant columns (zero variance)
constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
if constant_cols:
    issues.append(f"WARNING: Constant columns (consider dropping): {constant_cols}")

# Entirely empty columns
empty_cols = [c for c in df.columns if df[c].isna().all()]
if empty_cols:
    issues.append(f"CRITICAL: Entirely empty columns: {empty_cols}")

# Duplicate rows
n_dup_rows = df.duplicated().sum()
if n_dup_rows > 0:
    issues.append(f"INFO: {n_dup_rows} duplicate rows ({100*n_dup_rows/len(df):.1f}%)")

# Mixed-type columns (strings + numbers)
for col in df.columns:
    if df[col].dtype == object:
        numeric_frac = pd.to_numeric(df[col], errors="coerce").notna().mean()
        if 0.1 < numeric_frac < 0.9:
            issues.append(f"WARNING: Column '{col}' has mixed types ({numeric_frac:.0%} numeric)")
```

## Step 3: Missing data analysis

```python
# Per-column missing rates
missing = df.isnull().sum()
missing_pct = 100 * missing / len(df)
missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct.round(1),
}).sort_values("missing_pct", ascending=False)
missing_report = missing_report[missing_report.missing_count > 0]

print("\nMissing Data Report:")
print(missing_report)

# Flag high-missing columns
high_missing = missing_pct[missing_pct > 50].index.tolist()
if high_missing:
    issues.append(f"WARNING: Columns with >50% missing (consider dropping): {high_missing}")

# Row-level completeness
row_missing = df.isnull().sum(axis=1)
fully_missing_rows = (row_missing == df.shape[1]).sum()
if fully_missing_rows > 0:
    issues.append(f"CRITICAL: {fully_missing_rows} rows are entirely empty")

# Correlated missingness (MNAR indicator)
if missing.sum() > 0:
    missing_corr = df.isnull().corr()
    high_corr_pairs = []
    for i in range(len(missing_corr.columns)):
        for j in range(i+1, len(missing_corr.columns)):
            if abs(missing_corr.iloc[i, j]) > 0.7:
                high_corr_pairs.append(
                    (missing_corr.columns[i], missing_corr.columns[j],
                     round(missing_corr.iloc[i, j], 2))
                )
    if high_corr_pairs:
        issues.append(f"INFO: Correlated missingness detected (may be MNAR): {high_corr_pairs}")
```

See [missing-data-patterns.md](references/missing-data-patterns.md) for MCAR/MAR/MNAR detection.

## Step 4: Distribution profiling

### Numeric columns
```python
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    s = df[col].dropna()
    if len(s) == 0:
        continue
    
    stats = {
        "min": s.min(),
        "max": s.max(),
        "mean": s.mean(),
        "std": s.std(),
        "median": s.median(),
        "skewness": s.skew(),
        "n_unique": s.nunique(),
        "n_zeros": (s == 0).sum(),
    }
    
    # Outlier detection (>3 sigma)
    z_scores = np.abs((s - s.mean()) / s.std())
    n_outliers = (z_scores > 3).sum()
    if n_outliers > 0:
        issues.append(f"INFO: '{col}' has {n_outliers} outliers (>3 sigma)")
    
    # High skewness
    if abs(stats["skewness"]) > 2:
        issues.append(f"INFO: '{col}' is highly skewed ({stats['skewness']:.1f}), consider log transform")
    
    print(f"\n{col}: {stats}")
```

### Categorical/string columns
```python
cat_cols = df.select_dtypes(include=["object", "category"]).columns

for col in cat_cols:
    s = df[col].dropna()
    n_unique = s.nunique()
    top_values = s.value_counts().head(5)
    
    print(f"\n{col}: {n_unique} unique values")
    print(f"  Top 5: {dict(top_values)}")
    
    # High cardinality warning
    if n_unique > 100:
        issues.append(f"WARNING: '{col}' has high cardinality ({n_unique} unique values)")
    
    # Rare categories
    rare = s.value_counts()
    rare_cats = rare[rare < 5].index.tolist()
    if len(rare_cats) > 5:
        issues.append(f"INFO: '{col}' has {len(rare_cats)} rare categories (<5 occurrences)")
```

### Binary columns
```python
for col in numeric_cols:
    unique_vals = df[col].dropna().unique()
    if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        balance = df[col].mean()
        if balance < 0.05 or balance > 0.95:
            issues.append(f"WARNING: Binary column '{col}' is imbalanced ({balance:.1%} positive)")
```

See [profiling-checklist.md](references/profiling-checklist.md) for the full 30+ quality checks.

## Step 5: jaxcross compatibility check

Map each column to a candidate `ColumnType`:

```python
type_recommendations = {}

for col in df.columns:
    s = df[col].dropna()
    
    if s.dtype == object:
        # String column — needs integer encoding
        type_recommendations[col] = {
            "current": "string",
            "action": "ENCODE as integers (0-indexed)",
            "suggested_type": "CATEGORICAL",
            "n_categories": s.nunique(),
        }
    elif set(s.unique()).issubset({0, 1, 0.0, 1.0}):
        type_recommendations[col] = {
            "current": str(s.dtype),
            "action": "Ready (binary)",
            "suggested_type": "BINARY",
        }
    elif s.dtype in [np.int64, np.int32] and s.nunique() <= 20:
        type_recommendations[col] = {
            "current": str(s.dtype),
            "action": "Ready (low-cardinality integer)",
            "suggested_type": "CATEGORICAL",
            "n_categories": s.nunique(),
        }
    elif s.dtype in [np.float64, np.float32]:
        if s.nunique() <= 20:
            type_recommendations[col] = {
                "current": str(s.dtype),
                "action": "Float with few unique values — consider CATEGORICAL",
                "suggested_type": "CATEGORICAL or CONTINUOUS",
                "n_unique": s.nunique(),
            }
        else:
            type_recommendations[col] = {
                "current": str(s.dtype),
                "action": "Ready (continuous)",
                "suggested_type": "CONTINUOUS",
            }
    else:
        type_recommendations[col] = {
            "current": str(s.dtype),
            "action": "Review manually",
            "suggested_type": "UNKNOWN",
        }

print("\njaxcross Column Type Recommendations:")
for col, rec in type_recommendations.items():
    print(f"  {col}: {rec['suggested_type']} ({rec['action']})")
```

**Important:** jaxcross cannot auto-detect ORDINAL or CYCLIC types. Flag columns that might be:
- **ORDINAL**: education level, satisfaction rating, Likert scales, letter grades
- **CYCLIC**: hour of day, day of week, month, compass bearing, angle

See [type-mapping-guide.md](references/type-mapping-guide.md) for the full decision tree.

## Step 6: Output quality report

Print a structured markdown report:

```
# Data Quality Report

## Summary
- File: <path>
- Shape: N rows x M columns
- Overall completeness: X%
- Issues found: N critical, M warnings, K info

## Critical Issues
<list from issues>

## Warnings
<list from issues>

## Column Type Recommendations
<table from step 5>

## Next Steps
- Run `/data-transform <path>` to fix encoding issues
- Drop columns: <high_missing_or_constant>
- Set ORDINAL/CYCLIC manually: <flagged_columns>
```

## Common Pitfalls

- **String "nan" vs actual NaN**: Check for literal strings "nan", "null", "N/A", "missing", "" that pandas doesn't auto-detect as NaN. Use `df.replace(["nan", "null", "N/A", "missing", ""], np.nan)`.
- **Integer columns with NaN**: Pandas promotes int→float when NaN is present. The float column may look continuous but is actually categorical. Check `nunique()`.
- **Date columns parsed as strings**: If a column looks like dates, pandas might keep it as strings. Check with `pd.to_datetime(df[col], errors="coerce")`.
- **Encoding issues**: If you see garbled characters, try re-reading with `encoding="latin-1"` or `encoding="utf-8-sig"`.
