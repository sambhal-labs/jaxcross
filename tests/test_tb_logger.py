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


def test_tblogger_after_close_does_not_crash():
    """log_sweep after close() delegates to writer (no guard implemented)."""
    mock_cls, mock_writer = _make_mock_summary_writer()

    with patch("crosscat.tb_logger._require_tensorboardx", return_value=mock_cls):
        from crosscat.tb_logger import TBLogger

        tb = TBLogger("test_dir")
        tb.close()
        # No closed-state guard exists; call should not raise
        tb.log_sweep({"x": 1.0}, step=0)
        mock_writer.add_scalar.assert_called_with("x", 1.0, 0)


def test_tblogger_context_manager():
    """TBLogger works as a context manager and calls close()."""
    mock_cls, mock_writer = _make_mock_summary_writer()

    with patch("crosscat.tb_logger._require_tensorboardx", return_value=mock_cls):
        from crosscat.tb_logger import TBLogger

        with TBLogger("test_dir") as tb:
            tb.log_sweep({"x": 1.0}, step=0)

    mock_writer.close.assert_called_once()


def test_tblogger_empty_1d_array():
    """Empty 1D array is logged without crashing (mean produces NaN)."""
    import numpy as np

    mock_cls, mock_writer = _make_mock_summary_writer()

    with patch("crosscat.tb_logger._require_tensorboardx", return_value=mock_cls):
        from crosscat.tb_logger import TBLogger

        tb = TBLogger("test_dir")
        tb.log_sweep({"empty": np.array([])}, step=0)

    # mean of empty array is NaN — should still be logged
    mock_writer.add_scalar.assert_called_once()
    mock_writer.add_histogram.assert_called_once()


def test_tblogger_2d_array_silently_skipped():
    """2D arrays are silently skipped (no ndim==2 branch)."""
    import numpy as np

    mock_cls, mock_writer = _make_mock_summary_writer()

    with patch("crosscat.tb_logger._require_tensorboardx", return_value=mock_cls):
        from crosscat.tb_logger import TBLogger

        tb = TBLogger("test_dir")
        tb.log_sweep({"matrix": np.ones((3, 3))}, step=0)

    mock_writer.add_scalar.assert_not_called()
    mock_writer.add_histogram.assert_not_called()


def test_tblogger_mixed_scalar_and_array():
    """Mixed scalar + array metrics in a single log_sweep call."""
    import numpy as np

    mock_cls, mock_writer = _make_mock_summary_writer()

    with patch("crosscat.tb_logger._require_tensorboardx", return_value=mock_cls):
        from crosscat.tb_logger import TBLogger

        tb = TBLogger("test_dir")
        tb.log_sweep(
            {"log_joint": -50.0, "cluster_sizes": np.array([5, 10, 15])},
            step=2,
        )

    # Scalar + mean of array = 2 add_scalar calls
    assert mock_writer.add_scalar.call_count == 2
    mock_writer.add_histogram.assert_called_once()
