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

from jax import Array

from crosscat.types import CrossCatState


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
    raise NotImplementedError


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
    raise NotImplementedError
