"""Inference engine wrapping packed CrossCat state and Gibbs sweeps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

from crosscat.model import initialize
from crosscat.packed import (
    PackedCrossCatState,
    pack_state,
    packed_gibbs_sweep,
    packed_log_joint,
)
from crosscat.types import ColumnType

if TYPE_CHECKING:
    from jax import Array


class InferenceEngine:
    """Manages packed CrossCat state and exposes Gibbs sweep operations.

    Attributes:
        packed: Current ``PackedCrossCatState``.
        column_types: Column type list used for unpacking.
        data: The observation matrix.
    """

    def __init__(
        self,
        data: Array,
        column_types: list[ColumnType],
        max_views: int = 16,
        max_clusters: int = 32,
    ) -> None:
        """Initialize state from data.

        Calls ``crosscat.model.initialize`` to create a ``CrossCatState``
        from the prior, then packs it into a ``PackedCrossCatState``.

        Args:
            data: Observation matrix, shape ``(n_rows, n_cols)``.
            column_types: Type specification per column.
            max_views: Maximum number of views (padding dimension).
            max_clusters: Maximum clusters per view (padding dimension).
        """
        self.data = data
        self.column_types = column_types

        # Initialize from the prior with a fixed seed for reproducibility
        rng_key = jax.random.key(42)
        state = initialize(rng_key, data, column_types).state

        self.packed: PackedCrossCatState = pack_state(
            state,
            max_views=max_views,
            max_clusters=max_clusters,
        )

    def run_sweep(self, rng_key: Array) -> dict:
        """Run a single Gibbs sweep and return summary statistics.

        Executes ``packed_gibbs_sweep`` with ``n_sweeps=1``, then computes
        ``log_joint`` directly on the packed state (avoids costly unpack).

        Args:
            rng_key: JAX PRNG key for this sweep.

        Returns:
            Dictionary with ``log_joint`` (float) and ``n_views`` (int).
        """
        self.packed = packed_gibbs_sweep(
            rng_key,
            self.packed,
            self.data,
            n_sweeps=1,
        )

        lj = float(packed_log_joint(self.packed, self.data))
        n_views = int(self.packed.n_views)

        return {"log_joint": lj, "n_views": n_views}

    def get_packed_state(self) -> PackedCrossCatState:
        """Return the current packed state."""
        return self.packed

    def get_column_types(self) -> list[ColumnType]:
        """Return the column types."""
        return self.column_types
