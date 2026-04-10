# Production Serving Template

## Complete serving script

```python
#!/usr/bin/env python3
"""jaxcross model server with online inference."""

import jax
import jax.numpy as jnp
import logging
import time
from crosscat import (
    load_packed_state, packed_log_joint, estimate_packed_memory,
    packed_insert_rows, packed_gibbs_sweep,
    batch_anomaly_score, batch_impute_column, batch_classify_column,
)
from crosscat.packed.aot_cache import compile_kernels
from crosscat.data_utils import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossCatServer:
    def __init__(self, model_path, data_path):
        self.packed, self.col_types = load_packed_state(model_path)
        self.data, self.col_names, _ = load_data(data_path)
        self.key = jax.random.key(42)
        self.baseline_lj = float(packed_log_joint(self.packed, self.data))
        
        logger.info(f"Model loaded: {self.data.shape}")
        logger.info(f"Memory: {estimate_packed_memory(self.packed)/1e6:.1f} MB")
        
        # Pre-compile
        compile_kernels(self.packed, self.data)
        logger.info("XLA kernels compiled")
    
    def score_anomaly(self, row_ids):
        return batch_anomaly_score(self.packed, self.data, jnp.array(row_ids))
    
    def impute(self, col_name, row_ids):
        self.key, subkey = jax.random.split(self.key)
        col_idx = self.col_names.index(col_name)
        return batch_impute_column(subkey, self.packed, self.data, col_idx, jnp.array(row_ids))
    
    def classify(self, target_col, row_ids):
        self.key, subkey = jax.random.split(self.key)
        col_idx = self.col_names.index(target_col)
        return batch_classify_column(subkey, self.packed, self.data, col_idx, jnp.array(row_ids))
    
    def ingest(self, new_rows, n_sweeps=5):
        self.key, subkey = jax.random.split(self.key)
        self.packed, self.data = packed_insert_rows(subkey, self.packed, self.data, new_rows)
        self.key, subkey = jax.random.split(self.key)
        self.packed = packed_gibbs_sweep(subkey, self.packed, self.data, n_sweeps=n_sweeps)
        logger.info(f"Ingested {new_rows.shape[0]} rows, ran {n_sweeps} sweeps")
    
    def health_check(self):
        lj = float(packed_log_joint(self.packed, self.data))
        degradation = (self.baseline_lj - lj) / abs(self.baseline_lj)
        healthy = degradation < 0.1
        return {"healthy": healthy, "log_joint": lj, "baseline": self.baseline_lj}


if __name__ == "__main__":
    server = CrossCatServer("model.jxc", "data/prepared.arrow")
    
    # Example usage
    scores = server.score_anomaly(list(range(10)))
    print(f"Anomaly scores for first 10 rows: {scores}")
    
    health = server.health_check()
    print(f"Health: {health}")
```
