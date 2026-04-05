# TensorBoard Logger

::: crosscat.tb_logger
    options:
      show_source: false

## Overview

Thin wrapper around `tensorboardX.SummaryWriter` to log per-sweep diagnostics from [`collect_diagnostics()`](diagnostics.md). Requires the optional `tensorboardX` dependency:

```bash
pip install tensorboardX
```

See the [TensorBoard Logging Guide](../guides/tb-logger.md) for usage patterns.

---

## `TBLogger`

```python
class TBLogger:
    def __init__(self, log_dir: str = "runs", **kwargs) -> None: ...
    def log_sweep(self, metrics: dict, step: int) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> TBLogger: ...
    def __exit__(self, *args) -> None: ...
```

Context manager for logging CrossCat diagnostics to TensorBoard.

| Parameter | Type | Description |
|-----------|------|-------------|
| `log_dir` | `str` | Directory for TensorBoard event files (default `"runs"`) |
| `**kwargs` | | Extra arguments passed to `tensorboardX.SummaryWriter` |

### `log_sweep`

Log a diagnostics dict from `collect_diagnostics()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `metrics` | `dict` | Dict from `collect_diagnostics(state, data)` |
| `step` | `int` | Sweep number (x-axis in TensorBoard) |

Scalar metrics are logged directly. Array/list metrics are logged as their mean value with a `mean/` prefix and as histograms.

### Usage

```python
from crosscat.tb_logger import TBLogger
from crosscat import collect_diagnostics

with TBLogger("runs/experiment1") as tb:
    for sweep in range(n_sweeps):
        packed = packed_gibbs_sweep(key, packed, data, n_sweeps=1)
        state = unpack_state(packed, column_types, data=data)
        metrics = collect_diagnostics(state, data)
        tb.log_sweep(metrics, sweep)
```

Then visualize with:

```bash
tensorboard --logdir runs
```
