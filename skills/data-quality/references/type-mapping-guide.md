# Column Type Mapping Guide

## Decision Tree: Raw Data Type → jaxcross ColumnType

```
Is the column a string?
├── Yes → needs integer encoding
│   ├── Only 2 unique values → BINARY (encode as 0/1)
│   ├── Has natural ordering (low/med/high, education levels) → ORDINAL
│   └── No ordering → CATEGORICAL
│
└── No (numeric)
    ├── Only values are 0 and 1 → BINARY
    ├── Integer with ≤ 20 unique values
    │   ├── Has natural ordering → ORDINAL
    │   └── No ordering → CATEGORICAL
    ├── Represents an angle, compass bearing, or direction → CYCLIC
    ├── Represents hour of day, day of week, month → CYCLIC
    └── Continuous float → CONTINUOUS
```

## ColumnType Details

### CONTINUOUS
- **Model**: Normal-Gamma (conjugate Gaussian)
- **Use for**: measurements, prices, temperatures, heights, weights
- **Data format**: any float value, NaN for missing
- **Auto-detected**: Yes, by `guess_column_types()`

### CATEGORICAL
- **Model**: Dirichlet-Categorical (conjugate multinomial)
- **Use for**: country codes, product types, colors, unordered labels
- **Data format**: integers 0, 1, 2, ..., K-1 where K < `max_categories`
- **Auto-detected**: Yes (integer columns with ≤20 unique or <2% unique/total ratio)
- **Encoding required**: strings must be converted to 0-indexed integers

### BINARY
- **Model**: Beta-Bernoulli (conjugate binomial)
- **Use for**: yes/no, true/false, 0/1 flags, presence/absence
- **Data format**: 0.0 or 1.0
- **Auto-detected**: Yes (columns with only values 0 and 1)

### ORDINAL
- **Model**: Ordered Logistic (latent variable with learned cutpoints)
- **Use for**: Likert scales, education levels, satisfaction ratings, severity
- **Data format**: integers 0, 1, 2, ..., K-1 in meaningful order
- **Auto-detected**: No — must be set manually
- **Common examples**: rating (1-5), education (none/high-school/bachelors/masters/phd), pain scale (0-10)

### CYCLIC
- **Model**: Von Mises (circular normal)
- **Use for**: angles, compass bearings, hour of day, day of week, month
- **Data format**: radians in [0, 2*pi)
- **Auto-detected**: No — must be set manually
- **Encoding required**: convert to radians before modeling

## Encoding Cheat Sheet

| Raw Type | jaxcross Type | Encoding |
|----------|--------------|----------|
| "male"/"female" | BINARY | {"female": 0, "male": 1} |
| "low"/"med"/"high" | ORDINAL | {"low": 0, "med": 1, "high": 2} |
| "red"/"blue"/"green" | CATEGORICAL | {"red": 0, "blue": 1, "green": 2} |
| 23.5, 67.1, ... | CONTINUOUS | None needed |
| 0, 1, 0, 1, ... | BINARY | None needed |
| hour: 0-23 | CYCLIC | `hour * 2*pi/24` |
| compass: 0-360 | CYCLIC | `deg * pi/180` |
| month: 1-12 | CYCLIC | `(month-1) * 2*pi/12` |
| day of week: 0-6 | CYCLIC | `dow * 2*pi/7` |

## `max_categories` Sizing

The `max_categories` parameter in `pack_state()` must be >= the number of unique values in any categorical column. Default is auto-detected from data.

- If you have a column with 50 categories, `max_categories` must be >= 50
- Higher values waste memory but don't affect correctness
- Pass `data=` to `pack_state()` for automatic validation
