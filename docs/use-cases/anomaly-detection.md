# Anomaly Detection

Score how unusual each record is relative to the learned data structure. Detect outliers, fraud, and data quality issues in heterogeneous datasets.

## The Challenge

Anomaly detection in mixed-type data is hard. Most methods work on numeric data only. Rule-based systems miss complex multivariate patterns. CrossCat scores anomalies across all column types simultaneously, with no manual threshold tuning.

## Why CrossCat Fits

- **Model-free** — learns the normal structure from data, then flags deviations
- **Mixed types** — anomaly scoring works across continuous, categorical, binary, ordinal, and cyclic columns
- **Interpretable** — you can drill down to which columns make a row unusual
- **Bayesian uncertainty** — confidence in anomaly scores, not just binary flags

## Workflow

```python
import jax
import jax.numpy as jnp
from crosscat import initialize, predictive_anomalousness, row_typicality
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state

# Train the model
key = jax.random.key(42)
result = initialize(key, data, col_types)
state = result.state
packed = pack_state(state)
packed = packed_gibbs_sweep(jax.random.key(1), packed, data, n_sweeps=100)
state = unpack_state(packed, col_types, data=data)

# Score all rows
anomaly_scores = []
for i in range(data.shape[0]):
    score = predictive_anomalousness(jax.random.key(i), state, data, query_row=i)
    anomaly_scores.append(float(score))

# Or use row typicality (lower = more atypical)
typicality = []
for i in range(data.shape[0]):
    t = row_typicality([state], row_id=i)
    typicality.append(float(t))

# Flag top anomalies
import numpy as np
scores = np.array(anomaly_scores)
top_anomalies = np.argsort(scores)[-10:]  # Top 10 most anomalous
```

## Packed Version (Faster)

```python
from crosscat import packed_anomaly_score, packed_row_typicality

# Score individual rows on packed state
score = packed_anomaly_score(jax.random.key(3), packed, data, query_row=5)
typicality = packed_row_typicality([packed], row_id=5)

# Multi-chain scoring for robust estimates
from crosscat import multi_chain_anomaly_score
score = multi_chain_anomaly_score(jax.random.key(4), packed_states, data, query_row=5)
```

## Interpreting Scores

**Anomaly score** (`predictive_anomalousness`): Higher = more anomalous. The score is the negative log-probability of the row under the posterior predictive. There's no universal threshold — compare scores within your dataset.

**Row typicality** (`row_typicality`): Probability that the row's cluster assignments are typical given the CRP prior. Lower = more atypical. Values near 0 indicate the row is in small or unusual clusters.

## Applications

- **Fraud detection** — flag transactions that don't match learned spending patterns across amount, merchant category, time, and location
- **Quality control** — identify manufacturing records with unusual combinations of sensor readings
- **Data cleaning** — find data entry errors (e.g., salary=50 when others are 50000)
- **Healthcare** — flag patient records with unusual combinations of symptoms, lab values, and demographics

## Tips

- Use **multi-chain inference** for more stable anomaly scores
- Compare anomaly scores **relative to other rows** — absolute values depend on dataset size and dimensionality
- For per-column anomaly analysis, score the row with and without specific columns to identify which features drive the anomaly
- Run at least 100 sweeps — anomaly detection is sensitive to model quality
