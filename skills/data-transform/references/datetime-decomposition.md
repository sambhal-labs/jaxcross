# Datetime Decomposition Guide

## Why decompose?

jaxcross operates on float32 arrays. Raw datetime objects can't be used directly. Decomposing into numeric components lets jaxcross discover temporal patterns automatically.

## Standard decomposition

```python
dt = pd.to_datetime(df["timestamp"])

# Cyclic components (encode as radians)
df["hour"] = dt.dt.hour * 2 * np.pi / 24          # CYCLIC
df["day_of_week"] = dt.dt.dayofweek * 2 * np.pi / 7  # CYCLIC
df["month"] = (dt.dt.month - 1) * 2 * np.pi / 12     # CYCLIC

# Linear components
df["year"] = dt.dt.year.astype(float)               # CONTINUOUS
df["day_of_month"] = dt.dt.day.astype(float)         # CONTINUOUS or CATEGORICAL

# Optional: day of year for seasonal patterns
df["day_of_year"] = (dt.dt.dayofyear - 1) * 2 * np.pi / 365.25  # CYCLIC

# Drop original
df = df.drop(columns=["timestamp"])
```

## Which components to keep?

| Pattern to detect | Components needed |
|-------------------|-------------------|
| Time-of-day effects | hour (CYCLIC) |
| Day-of-week effects | day_of_week (CYCLIC) |
| Seasonal effects | month (CYCLIC) or day_of_year (CYCLIC) |
| Long-term trends | year (CONTINUOUS) |
| Monthly billing cycles | day_of_month (CONTINUOUS) |

## Column type assignments

| Component | ColumnType | Reason |
|-----------|-----------|--------|
| hour (radians) | CYCLIC | 23:00 is close to 00:00 |
| day_of_week (radians) | CYCLIC | Sunday is close to Monday |
| month (radians) | CYCLIC | December is close to January |
| year | CONTINUOUS | Linear progression |
| day_of_month | CONTINUOUS | 1-28/31, no wrap-around needed |
| is_weekend | BINARY | 0/1 flag |

## Timezone handling

```python
# Normalize to UTC before decomposition
if dt.dt.tz is not None:
    dt = dt.dt.tz_convert("UTC")
else:
    # Assume UTC if no timezone
    pass

# Or convert to local timezone if that's what matters
dt = dt.dt.tz_convert("US/Eastern")
```

## Duration/elapsed time

If you have a duration column (e.g., "session_length"), keep it as CONTINUOUS:
```python
df["duration_seconds"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
```
