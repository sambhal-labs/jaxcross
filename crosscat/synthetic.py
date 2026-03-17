"""Synthetic data generation for testing and benchmarking.

Maps to original CrossCat synthetic_data_generator.py.
Generates data from known CrossCat generative parameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from crosscat.types import ColumnType


def generate_crosscat_data(
    rng_key: Array,
    n_rows: int,
    column_types: list[ColumnType],
    *,
    n_views: int = 2,
    n_clusters: int = 2,
    cluster_separation: float = 5.0,
) -> dict:
    """Generate synthetic data from a CrossCat generative model.

    Maps to original synthetic_data_generator.py.

    Args:
        rng_key: JAX PRNG key.
        n_rows: Number of rows to generate.
        column_types: Type per column.
        n_views: Number of views (column groups).
        n_clusters: Number of clusters per view.
        cluster_separation: Mean separation between clusters (continuous columns).

    Returns:
        Dictionary with keys:
            'data': Array (n_rows, n_cols)
            'column_types': list[ColumnType]
            'true_column_assignments': Array
            'true_row_assignments': list[Array] (one per view)
            'n_rows': int
            'n_cols': int
    """
    n_cols = len(column_types)
    k_main, k_data = jax.random.split(rng_key)

    # Assign columns to views (round-robin)
    col_assignments = jnp.array([j % n_views for j in range(n_cols)], dtype=jnp.int32)

    # Generate cluster assignments per view
    row_assignments_per_view = []
    view_keys = jax.random.split(k_main, n_views)
    for v in range(n_views):
        # Equal-sized clusters
        cluster_size = n_rows // n_clusters
        assigns = jnp.zeros(n_rows, dtype=jnp.int32)
        for c in range(n_clusters):
            start = c * cluster_size
            end = start + cluster_size if c < n_clusters - 1 else n_rows
            assigns = assigns.at[start:end].set(c)
        # Shuffle
        perm = jax.random.permutation(view_keys[v], n_rows)
        assigns = assigns[perm]
        row_assignments_per_view.append(assigns)

    # Generate data column by column
    data = jnp.zeros((n_rows, n_cols))
    data_keys = jax.random.split(k_data, n_cols)

    for j in range(n_cols):
        view_idx = int(col_assignments[j])
        row_assigns = row_assignments_per_view[view_idx]
        col_type = column_types[j]

        if col_type == ColumnType.CONTINUOUS:
            for c in range(n_clusters):
                mask = row_assigns == c
                n_in_cluster = int(jnp.sum(mask))
                mean = c * cluster_separation
                k1 = jax.random.fold_in(data_keys[j], c)
                vals = mean + jax.random.normal(k1, (n_in_cluster,))
                data = data.at[mask, j].set(vals)

        elif col_type == ColumnType.CATEGORICAL:
            n_cats = max(3, n_clusters + 1)
            for c in range(n_clusters):
                mask = row_assigns == c
                n_in_cluster = int(jnp.sum(mask))
                # Each cluster has a dominant category
                k1 = jax.random.fold_in(data_keys[j], c)
                probs = jnp.ones(n_cats) * 0.1
                probs = probs.at[c % n_cats].set(5.0)
                probs = probs / probs.sum()
                vals = jax.random.categorical(k1, jnp.log(probs), shape=(n_in_cluster,))
                data = data.at[mask, j].set(vals.astype(jnp.float32))

        elif col_type == ColumnType.BINARY:
            for c in range(n_clusters):
                mask = row_assigns == c
                n_in_cluster = int(jnp.sum(mask))
                p = 0.2 + 0.6 * (c / max(n_clusters - 1, 1))
                k1 = jax.random.fold_in(data_keys[j], c)
                vals = jax.random.bernoulli(k1, p, (n_in_cluster,)).astype(jnp.float32)
                data = data.at[mask, j].set(vals)

        elif col_type == ColumnType.ORDINAL:
            n_levels = max(3, n_clusters + 1)
            for c in range(n_clusters):
                mask = row_assigns == c
                n_in_cluster = int(jnp.sum(mask))
                k1 = jax.random.fold_in(data_keys[j], c)
                probs = jnp.ones(n_levels) * 0.1
                probs = probs.at[c % n_levels].set(5.0)
                probs = probs / probs.sum()
                vals = jax.random.categorical(k1, jnp.log(probs), shape=(n_in_cluster,))
                data = data.at[mask, j].set(vals.astype(jnp.float32))

        elif col_type == ColumnType.CYCLIC:
            for c in range(n_clusters):
                mask = row_assigns == c
                n_in_cluster = int(jnp.sum(mask))
                mean_angle = c * (2.0 * jnp.pi / n_clusters)
                k1 = jax.random.fold_in(data_keys[j], c)
                vals = mean_angle + 0.5 * jax.random.normal(k1, (n_in_cluster,))
                vals = vals % (2.0 * jnp.pi)
                data = data.at[mask, j].set(vals)

    return {
        "data": data,
        "column_types": column_types,
        "true_column_assignments": col_assignments,
        "true_row_assignments": row_assignments_per_view,
        "n_rows": n_rows,
        "n_cols": n_cols,
    }


def add_missing_data(
    rng_key: Array,
    data: Array,
    missing_fraction: float = 0.1,
) -> Array:
    """Add random missing values (NaN) to data.

    Args:
        rng_key: JAX PRNG key.
        data: Data array, shape (n_rows, n_cols).
        missing_fraction: Fraction of values to set to NaN.

    Returns:
        Data array with NaN values injected.
    """
    mask = jax.random.bernoulli(rng_key, missing_fraction, data.shape)
    return jnp.where(mask, jnp.nan, data)
