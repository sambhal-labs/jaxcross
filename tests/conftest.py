"""Shared test fixtures for jax-crosscat."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.types import ColumnType


@pytest.fixture
def rng_key():
    """Deterministic JAX PRNG key for reproducible tests."""
    return jax.random.key(42)


@pytest.fixture
def synthetic_continuous_data(rng_key):
    """Synthetic continuous data with known cluster structure.

    2 views, 2 clusters each, 4 columns total:
    - View 0: columns 0, 1 (correlated within clusters)
    - View 1: columns 2, 3 (correlated within clusters, independent of view 0)
    """
    n_rows = 200
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)

    # View 0: 2 clusters with different means
    cluster_0 = jnp.array([0] * 100 + [1] * 100)
    col0 = jnp.where(cluster_0 == 0, 0.0, 5.0) + jax.random.normal(k1, (n_rows,))
    col1 = jnp.where(cluster_0 == 0, -2.0, 3.0) + jax.random.normal(k2, (n_rows,))

    # View 1: different clustering (not aligned with view 0)
    cluster_1 = jnp.array(([0] * 50 + [1] * 50) * 2)
    col2 = jnp.where(cluster_1 == 0, 10.0, 20.0) + jax.random.normal(k3, (n_rows,))
    col3 = jnp.where(cluster_1 == 0, -5.0, 5.0) + jax.random.normal(k4, (n_rows,))

    data = jnp.column_stack([col0, col1, col2, col3])
    column_types = [ColumnType.CONTINUOUS] * 4
    true_column_assignments = jnp.array([0, 0, 1, 1])
    true_row_assignments = [cluster_0, cluster_1]

    return {
        "data": data,
        "column_types": column_types,
        "true_column_assignments": true_column_assignments,
        "true_row_assignments": true_row_assignments,
        "n_rows": n_rows,
        "n_cols": 4,
    }
