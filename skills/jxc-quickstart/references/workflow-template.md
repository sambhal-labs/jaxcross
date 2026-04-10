# Quickstart Workflow Template

## Complete copy-paste script

```python
#!/usr/bin/env python3
"""jaxcross quickstart: data file → first insights."""

import jax
import jax.numpy as jnp
from crosscat import (
    initialize, pack_state, packed_gibbs_sweep, packed_log_joint,
    packed_dependence_matrix, batch_anomaly_score, unpack_state,
    save_packed_state, guess_column_types, read_csv,
)

# ── Config ──────────────────────────────────────────────────────
FILE_PATH = "data/prepared.csv"
N_SWEEPS = 100
SEED = 42

# ── Load ────────────────────────────────────────────────────────
data, col_names = read_csv(FILE_PATH)
data = jnp.array(data, dtype=jnp.float32)
col_types = guess_column_types(data)
print(f"Data: {data.shape}, Types: {[ct.name for ct in col_types]}")

# ── Train ───────────────────────────────────────────────────────
key = jax.random.key(SEED)
result = initialize(key, data, col_types)
packed = pack_state(result.state, data=data)

key, subkey = jax.random.split(key)
packed = packed_gibbs_sweep(subkey, packed, data, n_sweeps=N_SWEEPS)
print(f"Log-joint: {float(packed_log_joint(packed, data)):.1f}")

# ── Dependencies ────────────────────────────────────────────────
z = packed_dependence_matrix([packed])
n = len(col_names)
pairs = sorted(
    [(col_names[i], col_names[j], float(z[i, j]))
     for i in range(n) for j in range(i+1, n)],
    key=lambda x: -x[2]
)
print("\nTop dependencies:")
for a, b, s in pairs[:10]:
    print(f"  {a} <-> {b}: {s:.2f}")

# ── Anomalies ───────────────────────────────────────────────────
scores = batch_anomaly_score(packed, data, jnp.arange(data.shape[0]))
for idx in jnp.argsort(-scores)[:5]:
    print(f"  Row {int(idx)}: score={float(scores[idx]):.3f}")

# ── Save ────────────────────────────────────────────────────────
save_packed_state(packed, "model.jxc", column_types=col_types)
print(f"\nSaved to model.jxc")
```
