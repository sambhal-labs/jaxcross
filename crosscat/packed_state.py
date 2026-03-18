"""Backward-compatibility shim — use ``crosscat.packed`` instead.

.. deprecated:: 0.3.0
    This module is retained for backward compatibility.
    Import from ``crosscat.packed`` directly.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "crosscat.packed_state is deprecated — use crosscat.packed instead.",
    DeprecationWarning,
    stacklevel=2,
)

from crosscat.packed import *  # noqa: E402, F401, F403
