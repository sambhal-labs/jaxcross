"""Functions to extract human-readable info from PackedCrossCatState."""

from __future__ import annotations

import numpy as np

from crosscat.packed import PackedCrossCatState


def extract_structure(packed: PackedCrossCatState) -> list[dict]:
    """Extract a human-readable summary of each view in the packed state.

    Args:
        packed: A ``PackedCrossCatState`` instance.

    Returns:
        List of dicts, one per active view, with keys:

        - ``view_id``: int
        - ``columns``: list of global column indices in this view
        - ``n_clusters``: int
        - ``cluster_sizes``: list of ints (one per active cluster)
        - ``crp_alpha``: float
    """
    n_views = int(packed.n_views)
    result: list[dict] = []

    for v in range(n_views):
        n_cols_v = int(packed.view_n_columns[v])
        columns = [int(packed.view_column_indices[v, j]) for j in range(n_cols_v)]

        n_clusters = int(packed.view_n_clusters[v])

        # Compute cluster sizes from row assignments
        row_assigns = np.asarray(packed.view_row_assignments[v])
        cluster_sizes: list[int] = []
        for c in range(n_clusters):
            cluster_sizes.append(int(np.sum(row_assigns == c)))

        crp_alpha = float(packed.view_row_crp_alpha[v])

        result.append(
            {
                "view_id": v,
                "columns": columns,
                "n_clusters": n_clusters,
                "cluster_sizes": cluster_sizes,
                "crp_alpha": crp_alpha,
            }
        )

    return result


def extract_column_partition(packed: PackedCrossCatState) -> dict[int, list[int]]:
    """Extract the column partition as a view_id -> column indices mapping.

    Args:
        packed: A ``PackedCrossCatState`` instance.

    Returns:
        Dictionary mapping each active view id to its list of global column
        indices.
    """
    n_views = int(packed.n_views)
    partition: dict[int, list[int]] = {}

    for v in range(n_views):
        n_cols_v = int(packed.view_n_columns[v])
        columns = [int(packed.view_column_indices[v, j]) for j in range(n_cols_v)]
        partition[v] = columns

    return partition
