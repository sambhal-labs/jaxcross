"""Shared test fixtures for jax-crosscat."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.diagnostics import collect_diagnostics
from crosscat.model import initialize
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state
from crosscat.synthetic import add_missing_data, generate_crosscat_data
from crosscat.types import ColumnType


@pytest.fixture
def rng_key():
    """Deterministic JAX PRNG key for reproducible tests."""
    return jax.random.key(42)


@pytest.fixture
def simple_state(rng_key):
    """A simple 2-view state for testing queries.

    Returns (state, data, column_types) with 100 rows and 4 continuous columns.
    """
    n_rows = 100
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    col0 = jnp.where(jnp.arange(n_rows) < 50, 0.0, 5.0) + jax.random.normal(k1, (n_rows,))
    col1 = jnp.where(jnp.arange(n_rows) < 50, -2.0, 3.0) + jax.random.normal(k2, (n_rows,))
    col2 = jnp.where(jnp.arange(n_rows) < 50, 10.0, 20.0) + jax.random.normal(k3, (n_rows,))
    col3 = jnp.where(jnp.arange(n_rows) < 50, -5.0, 5.0) + jax.random.normal(k4, (n_rows,))
    data = jnp.column_stack([col0, col1, col2, col3])
    column_types = [ColumnType.CONTINUOUS] * 4
    state = initialize(rng_key, data, column_types).state
    return state, data, column_types


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


@pytest.fixture(scope="session")
def synthetic_cyclic_data():
    """Synthetic cyclic data with known cluster structure.

    200 rows, 4 CYCLIC columns, 2 views, 2 clusters.
    Mean angles at 0 and pi for clear separation.
    """
    key = jax.random.key(100)
    column_types = [ColumnType.CYCLIC] * 4
    return generate_crosscat_data(
        key, 200, column_types, n_views=2, n_clusters=2, cluster_separation=5.0
    )


@pytest.fixture(scope="session")
def synthetic_mixed_data():
    """Synthetic mixed-type data with known cluster structure.

    300 rows, 6 columns of different types, 2 views, 2 clusters.
    """
    key = jax.random.key(200)
    column_types = [
        ColumnType.CONTINUOUS,
        ColumnType.CATEGORICAL,
        ColumnType.BINARY,
        ColumnType.ORDINAL,
        ColumnType.CYCLIC,
        ColumnType.CONTINUOUS,
    ]
    return generate_crosscat_data(
        key, 300, column_types, n_views=2, n_clusters=2, cluster_separation=5.0
    )


@pytest.fixture(scope="session")
def synthetic_missing_data():
    """Synthetic continuous data with 15% NaN values injected.

    200 rows, 4 CONTINUOUS columns, 2 views, 2 clusters.
    """
    key = jax.random.key(300)
    k1, k2 = jax.random.split(key)
    column_types = [ColumnType.CONTINUOUS] * 4
    result = generate_crosscat_data(
        k1, 200, column_types, n_views=2, n_clusters=2, cluster_separation=5.0
    )
    result["data"] = add_missing_data(k2, result["data"], missing_fraction=0.15)
    return result


def run_multi_chain_with_diagnostics(data, column_types, *, n_chains=4, n_sweeps=20, seed=42):
    """Run multi-chain inference collecting diagnostics at each sweep.

    Uses packed Gibbs kernels for speed, unpacking periodically for diagnostics.

    Returns:
        Tuple of (final_states, all_diagnostics) where all_diagnostics
        is a list of lists: all_diagnostics[chain][sweep] = diagnostics dict.
    """
    key = jax.random.key(seed)
    init_states = initialize(key, data, column_types, n_chains=n_chains).state
    all_diagnostics = []
    final_states = []
    diag_interval = max(1, n_sweeps // 5)
    for i, state in enumerate(init_states):
        chain_diags = []
        packed = pack_state(state)
        k = jax.random.fold_in(key, i + 1000)
        done = 0
        while done < n_sweeps:
            batch = min(diag_interval, n_sweeps - done)
            k, subkey = jax.random.split(k)
            packed = packed_gibbs_sweep(subkey, packed, data, n_sweeps=batch)
            done += batch
            state = unpack_state(packed, column_types, data=data)
            chain_diags.append(collect_diagnostics(state, data))
        final_states.append(state)
        all_diagnostics.append(chain_diags)
    return final_states, all_diagnostics
