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
            else:
                # Convert JAX arrays, lists, and numpy arrays uniformly
                try:
                    arr = np.asarray(value, dtype=np.float64)
                except (TypeError, ValueError):
                    continue
                if arr.ndim == 0:
                    self._writer.add_scalar(key, float(arr), step)
                elif arr.ndim == 1:
                    self._writer.add_scalar(f"mean/{key}", float(np.mean(arr)), step)
                    self._writer.add_histogram(key, arr, step)

    def log_convergence(
        self,
        traces,
        step: int,
        *,
        metric_name: str = "log_joint",
    ) -> dict[str, float]:
        """Compute and log Gelman-Rubin R-hat and ESS on a multi-chain trace.

        Requires ``traces`` to be convertible to a 2-D array of shape
        ``(n_chains, n_samples)``. R-hat needs at least 2 chains and 4
        samples per chain; ESS needs at least 2 samples. If either
        precondition is unmet the corresponding metric is silently skipped
        — the intent is that this method is safe to call every ``k`` sweeps
        without a guard, letting the diagnostics appear once enough history
        accumulates.

        Both diagnostics are also returned so callers can log them through
        other sinks or assert in tests.

        Args:
            traces: Array-like of shape ``(n_chains, n_samples)`` — the
                statistic (typically ``log_joint``) tracked per sweep per
                chain.
            step: TensorBoard x-axis step (usually the sweep counter).
            metric_name: TensorBoard tag prefix. R-hat logs as
                ``rhat/{metric_name}`` and ESS as ``ess/{metric_name}``.

        Returns:
            Dict with keys ``rhat`` and ``ess`` (values that were not
            computable are omitted).
        """
        import numpy as np

        from crosscat.diagnostics import effective_sample_size, gelman_rubin_rhat

        arr = np.asarray(traces, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError(
                f"traces must be 1-D or 2-D (n_chains, n_samples); got shape {arr.shape}"
            )

        n_chains, n_samples = arr.shape
        results: dict[str, float] = {}

        if n_chains >= 2 and n_samples >= 4:
            rhat_val = float(gelman_rubin_rhat(arr))
            self._writer.add_scalar(f"rhat/{metric_name}", rhat_val, step)
            results["rhat"] = rhat_val

        if n_samples >= 2:
            ess_val = float(effective_sample_size(arr))
            self._writer.add_scalar(f"ess/{metric_name}", ess_val, step)
            results["ess"] = ess_val

        return results

    def close(self) -> None:
        """Flush and close the underlying writer."""
        self._writer.close()

    def __enter__(self) -> TBLogger:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
