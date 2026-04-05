"""TensorBoard logging for CrossCat inference monitoring.

Provides a thin wrapper around ``tensorboardX.SummaryWriter`` to log
per-sweep diagnostics from :func:`crosscat.diagnostics.collect_diagnostics`.

Requires the optional ``tensorboardX`` dependency::

    pip install tensorboardX

Usage::

    from crosscat.tb_logger import TBLogger

    with TBLogger("runs/experiment1") as tb:
        for sweep in range(n_sweeps):
            packed = packed_gibbs_sweep(key, packed, data, n_sweeps=1)
            state = unpack_state(packed, column_types, data=data)
            metrics = collect_diagnostics(state, data)
            tb.log_sweep(metrics, sweep)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _require_tensorboardx():
    """Import tensorboardX or raise a clear error."""
    try:
        from tensorboardX import SummaryWriter

        return SummaryWriter
    except ImportError:
        raise ImportError(
            "tensorboardX is required for TensorBoard logging. "
            "Install with: pip install tensorboardX"
        ) from None


class TBLogger:
    """Thin wrapper for logging CrossCat diagnostics to TensorBoard.

    Args:
        log_dir: Directory for TensorBoard event files.
        **kwargs: Extra arguments passed to ``SummaryWriter``.
    """

    def __init__(self, log_dir: str = "runs", **kwargs: Any) -> None:
        SummaryWriter = _require_tensorboardx()
        self._writer = SummaryWriter(log_dir, **kwargs)
        logger.info("TensorBoard logging to %s", log_dir)

    def log_sweep(self, metrics: dict, step: int) -> None:
        """Log a diagnostics dict from ``collect_diagnostics()``.

        Scalar metrics are logged directly. List/array metrics are logged
        as their mean value with a ``mean/`` prefix.

        Args:
            metrics: Dict from :func:`crosscat.diagnostics.collect_diagnostics`.
            step: Sweep number (used as the x-axis in TensorBoard).
        """
        import numpy as np

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, step)
            elif isinstance(value, (list, np.ndarray)):
                arr = np.asarray(value, dtype=np.float64)
                if arr.ndim == 0:
                    self._writer.add_scalar(key, float(arr), step)
                elif arr.ndim == 1:
                    self._writer.add_scalar(f"mean/{key}", float(np.mean(arr)), step)
                    self._writer.add_histogram(key, arr, step)

    def close(self) -> None:
        """Flush and close the underlying writer."""
        self._writer.close()

    def __enter__(self) -> TBLogger:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
