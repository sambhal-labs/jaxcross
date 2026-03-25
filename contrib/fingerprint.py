"""Employer fingerprint extraction from CrossCat posterior.

Not present in original CrossCat — new functionality for LaborLens.

An employer's "fingerprint" is the posterior mean of its row cluster assignment
probabilities across all views. This captures the employer's behavioral signature
across different column groupings discovered by CrossCat.

These vectors enable:
- M3 Strategic Twin Finder: cosine similarity between employer fingerprints
- M4 Anomaly Radar: track fingerprint drift over time
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from crosscat.types import LOG_EPS, CrossCatState


def extract_fingerprint(
    states: list[CrossCatState],
    entity_row_indices: Array,
) -> Array:
    """Extract a fingerprint vector for an entity (employer) from its filing rows.

    For each view v in the CrossCat state:
    1. Compute the empirical cluster assignment distribution for the entity's rows
       p(cluster = c | entity) = count(entity rows in cluster c) / total entity rows
    2. Concatenate these distributions across views

    Average over multiple posterior states for robustness.

    Args:
        states: List of CrossCat posterior states (ensemble).
        entity_row_indices: Row indices belonging to this entity.

    Returns:
        Fingerprint vector — shape (sum of n_clusters across views,).
        Normalized to sum to 1 within each view's segment.
    """
    # Determine max clusters per view across all states for consistent dimensionality
    n_views_max = max(s.n_views for s in states)
    max_clusters_per_view = []
    for v in range(n_views_max):
        max_k = 1
        for s in states:
            if v < s.n_views:
                n_k = int(jnp.max(s.views[v].row_assignments)) + 1
                max_k = max(max_k, n_k)
        max_clusters_per_view.append(max_k)

    total_dim = sum(max_clusters_per_view)
    fingerprint_sum = jnp.zeros(total_dim)
    n_contributions = 0

    for state in states:
        fp_parts = []
        for v in range(n_views_max):
            max_k = max_clusters_per_view[v]
            if v < state.n_views:
                view = state.views[v]
                # Get cluster assignments for entity's rows
                entity_assigns = view.row_assignments[entity_row_indices.astype(jnp.int32)]
                # Compute distribution
                dist = jnp.zeros(max_k)
                for c in range(max_k):
                    dist = dist.at[c].set(jnp.sum(entity_assigns == c).astype(jnp.float32))
                # Normalize
                total = dist.sum()
                dist = jnp.where(total > 0, dist / total, jnp.ones(max_k) / max_k)
            else:
                dist = jnp.ones(max_k) / max_k
            fp_parts.append(dist)

        fp = jnp.concatenate(fp_parts)
        fingerprint_sum = fingerprint_sum + fp
        n_contributions += 1

    return fingerprint_sum / n_contributions


def fingerprint_similarity(
    fingerprint_a: Array,
    fingerprint_b: Array,
) -> Array:
    """Cosine similarity between two employer fingerprints.

    Args:
        fingerprint_a: First employer's fingerprint vector.
        fingerprint_b: Second employer's fingerprint vector.

    Returns:
        Cosine similarity score in [-1, 1].
    """
    norm_a = jnp.linalg.norm(fingerprint_a)
    norm_b = jnp.linalg.norm(fingerprint_b)
    denom = norm_a * norm_b
    return jnp.where(denom > LOG_EPS, jnp.dot(fingerprint_a, fingerprint_b) / denom, 0.0)
