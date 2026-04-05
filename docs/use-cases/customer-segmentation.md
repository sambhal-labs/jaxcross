# Customer Segmentation

Discover natural customer segments from mixed-type data without choosing k, encoding categories, or building separate models.

## The Challenge

Customer data is inherently mixed-type: continuous (revenue, session duration), categorical (plan tier, region), binary (churned, opted in), ordinal (satisfaction rating). Traditional clustering methods require encoding everything as numeric and choosing the number of clusters upfront.

CrossCat handles all these types natively and discovers the number of segments automatically.

## Why CrossCat Fits

- **Mixed types** — no need to one-hot encode categories or discretize continuous values
- **Multiple structures** — customers may segment differently by purchasing behavior vs. engagement vs. demographics, and CrossCat discovers these independently
- **Uncertainty** — posterior predictive gives confidence scores, not just point estimates
- **Imputation** — fill missing customer fields with Bayesian confidence

## Workflow

```python
import jax
import jax.numpy as jnp
from crosscat import initialize, dependence_matrix, impute_and_confidence
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat.types import ColumnType

# Define column types
col_types = [
    ColumnType.CONTINUOUS,    # revenue
    ColumnType.CONTINUOUS,    # session_duration
    ColumnType.CATEGORICAL,   # plan_tier (0=free, 1=pro, 2=enterprise)
    ColumnType.CATEGORICAL,   # region (0=NA, 1=EU, 2=APAC)
    ColumnType.BINARY,        # churned (0/1)
    ColumnType.ORDINAL,       # satisfaction (1-5)
]

# Initialize and run inference
key = jax.random.key(42)
result = initialize(key, data, col_types)
state = result.state
packed = pack_state(state)
packed = packed_gibbs_sweep(jax.random.key(1), packed, data, n_sweeps=100)
state = unpack_state(packed, col_types, data=data)

# Discover which features are related
z_matrix = dependence_matrix([state])
# z_matrix might reveal: {revenue, plan_tier} cluster together (monetization)
# while {session_duration, satisfaction} form a separate view (engagement)

# Predict churn for a new customer
from crosscat import predictive_probability
churn_prob = predictive_probability(
    state, data, query_cols=[4], query_vals=jnp.array([1.0]),  # P(churned=1)
    condition_cols=[0, 2, 5],                       # given revenue, plan, satisfaction
    condition_vals=jnp.array([500.0, 0.0, 2.0])    # low revenue, free tier, low satisfaction
)
```

## What You Get

1. **Dependence matrix** — which customer attributes are statistically related
2. **Multiple segmentations** — customers segmented by monetization, engagement, and geography independently
3. **Segment assignments** — each customer assigned to clusters in each view
4. **Churn prediction** — posterior probability of churn given observed attributes
5. **Missing data imputation** — fill gaps in customer profiles with confidence scores
6. **Anomaly detection** — flag unusual customers (potential fraud or data quality issues)

## Tips

- Start with 100 sweeps and check convergence via the log-joint trace
- Use multi-chain inference (4-10 chains) for robust segment discovery
- The dependence matrix is your most valuable output — it tells you which business questions can be answered from which features
