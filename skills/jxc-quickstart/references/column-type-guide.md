# Column Type Decision Guide

## Quick Decision Tree

```
Is this column a measurement/continuous value?
  → CONTINUOUS (temperature, price, height, GDP)

Is this column binary (only 0 and 1)?
  → BINARY (yes/no, true/false, present/absent)

Is this column unordered categories?
  → CATEGORICAL (country, color, product_type)

Does this column have a natural order?
  → ORDINAL (education_level, satisfaction_rating, pain_scale)

Does this column wrap around (end connects to beginning)?
  → CYCLIC (hour_of_day, compass_bearing, month_of_year)
```

## Detailed Type Reference

### CONTINUOUS (NormalGamma model)
- Real-valued measurements
- Examples: temperature, salary, stock_price, age, BMI
- No encoding needed (keep as float)
- Handles NaN natively

### CATEGORICAL (DirichletCategorical model)
- Unordered discrete categories
- Must be encoded as integers: 0, 1, 2, ..., K-1
- Examples: country_code, product_category, color
- Auto-detected by `guess_column_types()` for integer columns with ≤20 unique values

### BINARY (BetaBernoulli model)
- Exactly two values: 0 and 1
- Examples: is_active, has_default, survived
- Auto-detected by `guess_column_types()`

### ORDINAL (OrderedLogistic model)
- Ordered discrete categories
- Encode as integers: 0, 1, 2, ..., K-1 in meaningful order
- Examples: education (none=0, hs=1, bachelors=2, masters=3, phd=4)
- **Never auto-detected** — must set manually

### CYCLIC (VonMises model)
- Values on a circle where the maximum wraps to the minimum
- Must be in radians: [0, 2*pi)
- Examples: hour_of_day, day_of_week, compass_bearing, wind_direction
- **Never auto-detected** — must set manually
- Convert: `radians = value * 2 * pi / period`

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|------------|-----|
| Using CONTINUOUS for a Likert scale (1-5) | Model assumes normal distribution | Use ORDINAL |
| Using CATEGORICAL for hour of day | 23 and 0 treated as unrelated | Use CYCLIC |
| Using CONTINUOUS for a column with 3 unique values | Model fits unnecessary density | Use CATEGORICAL |
| Not encoding strings before modeling | `initialize()` will crash | Use `/data-transform` |
| Using BINARY for a 3-valued flag | Silently wrong model | Use CATEGORICAL |

## Setting types manually

```python
from crosscat.types import ColumnType

col_types = guess_column_types(data)

# Override specific columns by index
col_types[col_names.index("education")] = ColumnType.ORDINAL
col_types[col_names.index("hour")] = ColumnType.CYCLIC
col_types[col_names.index("compass")] = ColumnType.CYCLIC
```
