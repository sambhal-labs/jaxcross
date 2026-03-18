"""Data loading and synthetic generation utilities for the dashboard."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pandas as pd

from crosscat.data_utils import guess_column_type
from crosscat.synthetic import generate_crosscat_data
from crosscat.types import ColumnType

if TYPE_CHECKING:
    from jax import Array


def load_csv_data(
    uploaded_file,
) -> tuple[Array, list[str], list[ColumnType]]:
    """Parse a CSV from a Streamlit UploadedFile (BytesIO).

    Uses pandas for robust CSV parsing, then converts to a JAX array and
    detects column types via ``crosscat.data_utils.guess_column_type``.

    Args:
        uploaded_file: A Streamlit ``UploadedFile`` object (BytesIO-compatible).

    Returns:
        Tuple of (data_array, column_names, column_types).
    """
    # Read into pandas for reliable type coercion and NaN handling
    df = pd.read_csv(io.BytesIO(uploaded_file.read()))

    column_names: list[str] = list(df.columns)

    # Convert all columns to float, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    data = jnp.array(df.values, dtype=jnp.float32)

    # Detect column types
    column_types: list[ColumnType] = []
    n_cols = data.shape[1]
    for j in range(n_cols):
        col_data = data[:, j]
        ct = guess_column_type(col_data)
        column_types.append(ct)

    return data, column_names, column_types


def generate_synthetic(
    key: Array,
    n_rows: int,
    column_types: list[ColumnType],
    n_views: int = 2,
    n_clusters: int = 2,
    cluster_separation: float = 5.0,
) -> dict:
    """Generate synthetic CrossCat data.

    Thin wrapper around ``crosscat.synthetic.generate_crosscat_data`` that
    passes through all arguments.

    Args:
        key: JAX PRNG key.
        n_rows: Number of rows to generate.
        column_types: Type per column.
        n_views: Number of column-group views.
        n_clusters: Number of row clusters per view.
        cluster_separation: Separation between cluster means (continuous cols).

    Returns:
        Dictionary with keys ``data``, ``column_types``,
        ``true_column_assignments``, ``true_row_assignments``,
        ``n_rows``, ``n_cols``.
    """
    return generate_crosscat_data(
        key,
        n_rows,
        column_types,
        n_views=n_views,
        n_clusters=n_clusters,
        cluster_separation=cluster_separation,
    )
