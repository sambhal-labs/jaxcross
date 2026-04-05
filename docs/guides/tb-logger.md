# TensorBoard Logging

## Setup

Install the optional `tensorboardX` dependency:

```bash
pip install tensorboardX
```

## Basic Usage

Use `TBLogger` as a context manager to log per-sweep diagnostics:

```python
from crosscat import collect_diagnostics, packed_gibbs_sweep, unpack_state
from crosscat.tb_logger import TBLogger

with TBLogger("runs/experiment1") as tb:
    for sweep in range(100):
        key = jax.random.fold_in(jax.random.key(0), sweep)
        packed = packed_gibbs_sweep(key, packed, data, n_sweeps=1)
        state = unpack_state(packed, col_types, data=data)
        metrics = collect_diagnostics(state, data)
        tb.log_sweep(metrics, sweep)
```

## What Gets Logged

`collect_diagnostics()` returns a dict with:

| Metric | Type | Description |
|--------|------|-------------|
| `log_joint` | scalar | Log-joint probability of state + data |
| `n_views` | scalar | Number of active views |
| `n_clusters_per_view` | array | Cluster count per view |

Scalar metrics appear as line charts in TensorBoard. Array metrics are logged as their mean value (with `mean/` prefix) and as histograms.

## Viewing Results

Launch TensorBoard:

```bash
tensorboard --logdir runs
```

Then open `http://localhost:6006` in your browser.

## Integration with Early Stopping

Combine TBLogger with `gibbs_sweep_early_stopping` by logging the returned log-joint history:

```python
from crosscat import gibbs_sweep_early_stopping
from crosscat.tb_logger import TBLogger

packed, log_joints = gibbs_sweep_early_stopping(
    key, packed, data,
    max_sweeps=200, check_interval=10,
)

# Log the convergence curve after the fact
with TBLogger("runs/convergence") as tb:
    for i, lj in enumerate(log_joints):
        tb.log_sweep({"log_joint": lj}, i * 10)
```

## API Reference

- [`TBLogger`](../api/tb-logger.md#tblogger)
- [`collect_diagnostics`](../api/diagnostics.md)
