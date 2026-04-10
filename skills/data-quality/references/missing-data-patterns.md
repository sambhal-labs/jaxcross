# Missing Data Patterns Reference

## Three Types of Missingness

### MCAR (Missing Completely At Random)
- Missingness is independent of both observed and unobserved data
- Example: sensor randomly drops readings due to hardware glitch
- **Detection**: Little's MCAR test, or check if missing rate differs across subgroups
- **Safe to**: drop rows or impute with mean/mode
- **jaxcross handles well**: NaN values are silently filtered in sufficient statistics

### MAR (Missing At Random)
- Missingness depends on observed data but not the missing value itself
- Example: younger people skip the "income" question in a survey
- **Detection**: missingness in column A correlates with values in column B
- **Safe to**: impute using the model (jaxcross imputation handles this)
- **jaxcross handles well**: CrossCat captures dependencies, so imputation leverages related columns

### MNAR (Missing Not At Random)
- Missingness depends on the unobserved value itself
- Example: high-income people skip the "income" question
- **Detection**: hard to detect definitively; check if missingness patterns are non-random
- **Caution**: simple imputation may be biased; consider sensitivity analysis

## Detection Heuristics

### Check for MCAR vs MAR
```python
# Split data by missingness of target column
target_col = "income"
has_value = df[df[target_col].notna()]
is_missing = df[df[target_col].isna()]

# Compare distributions of other columns
for col in df.columns:
    if col == target_col or df[col].dtype == object:
        continue
    stat, pval = scipy.stats.mannwhitneyu(
        has_value[col].dropna(), is_missing[col].dropna(),
        alternative="two-sided"
    )
    if pval < 0.01:
        print(f"MAR signal: '{col}' distribution differs by '{target_col}' missingness (p={pval:.4f})")
```

### Check for correlated missingness
```python
missing_corr = df.isnull().corr()
# High correlation (>0.7) between missing indicators suggests block missingness
# or MNAR (same mechanism causes both to be missing)
```

### Missing rate by category
```python
# If missing rate differs across a categorical variable, it's MAR
for cat_col in categorical_columns:
    rates = df.groupby(cat_col)[target_col].apply(lambda x: x.isna().mean())
    if rates.max() - rates.min() > 0.1:
        print(f"MAR: '{target_col}' missing rate varies by '{cat_col}'")
```

## Recommendations for jaxcross

| Pattern | Recommendation |
|---------|---------------|
| MCAR, <5% missing | Leave as NaN — jaxcross filters automatically |
| MCAR, 5-30% missing | Leave as NaN — use `/jxc-impute` after modeling |
| MCAR, >50% missing | Consider dropping column |
| MAR | Leave as NaN — CrossCat leverages correlated columns for imputation |
| MNAR | Leave as NaN but note in analysis that imputations may be biased |
| Block missingness | Common in surveys; jaxcross views naturally separate complete vs incomplete variable groups |
