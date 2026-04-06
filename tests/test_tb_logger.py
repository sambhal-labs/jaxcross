"""Tests for TBLogger (TensorBoard integration)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.cpu


def _make_mock_summary_writer():
    """Create a mock SummaryWriter class."""
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    return mock_cls, mock_instance


def test_tblogger_import_error_message():
    """TBLogger raises ImportError with install instructions when tensorboardX missing."""
    with patch.dict("sys.modules", {"tensorboardX": None}):
        from crosscat.tb_logger import _require_tensorboardx

        with pytest.raises(ImportError, match="tensorboardX is required"):
            _require_tensorboardx()


def test_tblogger_log_sweep_scalars():
    """log_sweep sends scalar metrics to add_scalar."""
    mock_cls, mock_writer = _make_mock_summary_writer()

    with patch("crosscat.tb_logger._require_tensorboardx", return_value=mock_cls):
        from crosscat.tb_logger import TBLogger

        tb = TBLogger("test_dir")
        tb.log_sweep({"log_joint": -100.5, "n_views": 3}, step=5)

    mock_writer.add_scalar.assert_any_call("log_joint", -100.5, 5)
    mock_writer.add_scalar.assert_any_call("n_views", 3, 5)


def test_tblogger_log_sweep_1d_array():
    """log_sweep sends 1D arrays to add_histogram and mean to add_scalar."""
    import numpy as np

    mock_cls, mock_writer = _make_mock_summary_writer()

    with patch("crosscat.tb_logger._require_tensorboardx", return_value=mock_cls):
        from crosscat.tb_logger import TBLogger

        tb = TBLogger("test_dir")
        tb.log_sweep({"cluster_sizes": np.array([10, 20, 30])}, step=1)

    # Should log mean as scalar
    calls = [str(c) for c in mock_writer.add_scalar.call_args_list]
    assert any("mean/cluster_sizes" in c for c in calls)
    # Should log histogram
    mock_writer.add_histogram.assert_called_once()


def test_tblogger_closed_state_guard():
    """log_sweep raises RuntimeError after close()."""
    mock_cls, mock_writer = _make_mock_summary_writer()

    with patch("crosscat.tb_logger._require_tensorboardx", return_value=mock_cls):
        from crosscat.tb_logger import TBLogger

        tb = TBLogger("test_dir")
        tb.close()
        with pytest.raises(RuntimeError, match="closed"):
            tb.log_sweep({"x": 1.0}, step=0)


def test_tblogger_context_manager():
    """TBLogger works as a context manager and calls close()."""
    mock_cls, mock_writer = _make_mock_summary_writer()

    with patch("crosscat.tb_logger._require_tensorboardx", return_value=mock_cls):
        from crosscat.tb_logger import TBLogger

        with TBLogger("test_dir") as tb:
            tb.log_sweep({"x": 1.0}, step=0)

    mock_writer.close.assert_called_once()
