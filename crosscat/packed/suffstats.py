from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from crosscat.packed.state import (
    _ARRAY_FIELDS,
    _STATIC_FIELDS,
    BINARY_ID,
    CATEGORICAL_ID,
    CONTINUOUS_ID,
    CYCLIC_ID,
    ORDINAL_ID,
    PackedCrossCatState,
)

# ---------------------------------------------------------------------------
# Vectorized sufficient statistics computation
# ---------------------------------------------------------------------------


def compute_suffstats_vectorized(
    data: Array,
    column_indices: Array,
    col_type_ids: Array,
    row_assignments: Array,
    n_clusters: int,
    max_clusters: int,
    max_categories: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Compute sufficient statistics for all (cluster, column) pairs using matrix ops.

    Replaces the nested for-loop in _compute_suffstats_for_view.

    Args:
        data: Full data array (n_rows, n_cols).
        column_indices: Column indices in this view (n_cols_view,), -1 for padding.
        col_type_ids: Type IDs for ALL columns (n_cols_total,).
        row_assignments: Row assignments (n_rows,).
        n_clusters: Actual number of clusters.
        max_clusters: Padding dimension.
        max_categories: Padding for category counts.

    Returns:
        Tuple of (counts, sum_x, sum_x_sq, cat_counts, sum_sin, sum_cos)
        with shapes (max_clusters, n_cols_view[, max_categories]).
    """
    # Membership matrix: (n_rows, max_clusters) — binary indicator
    membership = (row_assignments[:, None] == jnp.arange(max_clusters)[None, :]).astype(
        jnp.float32
    )

    # Gather column data, handling -1 padding with column 0 (masked out later)
    safe_indices = jnp.where(column_indices >= 0, column_indices, 0)
    col_data = data[:, safe_indices]  # (n_rows, n_cols_view)
    valid = ~jnp.isnan(col_data)  # (n_rows, n_cols_view)
    clean_data = jnp.where(valid, col_data, 0.0)

    # Mask out padding columns
    col_valid_mask = (column_indices >= 0)[None, :]  # (1, n_cols_view)
    valid = valid & col_valid_mask

    # Counts: membership^T @ valid -> (max_clusters, n_cols_view)
    counts = (membership.T @ valid.astype(jnp.float32)).astype(jnp.int32)

    # Sum and sum-of-squares for continuous/binary
    sum_x = membership.T @ (clean_data * valid.astype(jnp.float32))
    sum_x_sq = membership.T @ (clean_data**2 * valid.astype(jnp.float32))

    # Sin/cos for cyclic
    sin_data = jnp.where(valid, jnp.sin(col_data), 0.0)
    cos_data = jnp.where(valid, jnp.cos(col_data), 0.0)
    sum_sin = membership.T @ sin_data
    sum_cos = membership.T @ cos_data

    # Category counts for categorical/ordinal — via one-hot encoding
    int_data = jnp.where(valid, clean_data.astype(jnp.int32), 0)
    one_hot = jax.nn.one_hot(int_data, max_categories)  # (n_rows, n_cols_view, max_cats)
    valid_one_hot = one_hot * valid[:, :, None].astype(jnp.float32)
    # Einsum: membership (n_rows, max_clusters) x valid_one_hot (n_rows, n_cols_view, max_cats)
    cat_counts = jnp.einsum("rc,rjk->cjk", membership, valid_one_hot)

    return counts, sum_x, sum_x_sq, cat_counts, sum_sin, sum_cos


def recompute_all_suffstats(packed: PackedCrossCatState, data: Array) -> PackedCrossCatState:
    """Recompute all sufficient statistics from data and current assignments.

    This is useful after modifying row or column assignments.
    Uses lax.scan over views for JIT compatibility.
    """
    max_c = packed.max_clusters
    max_cat = packed.max_categories
    max_views = packed.max_views

    def recompute_one_view(carry, v_idx):
        ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos = carry
        is_active = packed.view_mask[v_idx]
        col_indices = packed.view_column_indices[v_idx]  # (max_cpv,) with -1 padding
        row_assigns = packed.view_row_assignments[v_idx]

        counts, sx, sxsq, cc, ssin, scos = compute_suffstats_vectorized(
            data, col_indices, packed.col_type_ids, row_assigns, 0, max_c, max_cat
        )

        # Only update if view is active; otherwise keep old values
        new_ss_c = jnp.where(is_active, counts, ss_c[v_idx])
        new_ss_sx = jnp.where(is_active, sx, ss_sx[v_idx])
        new_ss_sxsq = jnp.where(is_active, sxsq, ss_sxsq[v_idx])
        new_ss_cat = jnp.where(is_active, cc, ss_cat[v_idx])
        new_ss_sin = jnp.where(is_active, ssin, ss_sin[v_idx])
        new_ss_cos = jnp.where(is_active, scos, ss_cos[v_idx])

        ss_c = ss_c.at[v_idx].set(new_ss_c)
        ss_sx = ss_sx.at[v_idx].set(new_ss_sx)
        ss_sxsq = ss_sxsq.at[v_idx].set(new_ss_sxsq)
        ss_cat = ss_cat.at[v_idx].set(new_ss_cat)
        ss_sin = ss_sin.at[v_idx].set(new_ss_sin)
        ss_cos = ss_cos.at[v_idx].set(new_ss_cos)

        return (ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos), None

    init = (
        packed.ss_counts,
        packed.ss_sum_x,
        packed.ss_sum_x_sq,
        packed.ss_cat_counts,
        packed.ss_sum_sin,
        packed.ss_sum_cos,
    )
    (ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos), _ = jax.lax.scan(
        recompute_one_view, init, jnp.arange(max_views)
    )

    return PackedCrossCatState(
        **{
            name: getattr(packed, name)
            for name in _ARRAY_FIELDS
            if name
            not in (
                "ss_counts",
                "ss_sum_x",
                "ss_sum_x_sq",
                "ss_cat_counts",
                "ss_sum_sin",
                "ss_sum_cos",
            )
        },
        ss_counts=ss_counts,
        ss_sum_x=ss_sum_x,
        ss_sum_x_sq=ss_sum_x_sq,
        ss_cat_counts=ss_cat_counts,
        ss_sum_sin=ss_sum_sin,
        ss_sum_cos=ss_sum_cos,
        **{name: getattr(packed, name) for name in _STATIC_FIELDS},
    )


# ---------------------------------------------------------------------------
# Incremental suffstat helpers (single-row add/remove)
# ---------------------------------------------------------------------------


def _remove_row_from_suffstats(
    ss_counts: Array,
    ss_sum_x: Array,
    ss_sum_x_sq: Array,
    ss_cat_counts: Array,
    ss_sum_sin: Array,
    ss_sum_cos: Array,
    cluster_id: Array,
    row_data: Array,
    col_indices: Array,
    col_type_ids: Array,
    max_categories: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Remove one row's contribution from a cluster's suffstats.

    All updates vectorized over columns via lax.scan. NaN values produce zero
    deltas. Uses .at[traced_idx].add() for JIT compatibility.

    Args:
        ss_counts: (max_clusters, max_cols_per_view) int
        ss_sum_x, ss_sum_x_sq: (max_clusters, max_cols_per_view)
        ss_cat_counts: (max_clusters, max_cols_per_view, max_categories)
        ss_sum_sin, ss_sum_cos: (max_clusters, max_cols_per_view)
        cluster_id: scalar traced int — which cluster to update
        row_data: (n_cols_total,) — full row from data matrix
        col_indices: (max_cols_per_view,) int — column indices for this view,
            -1 for padding
        col_type_ids: (n_cols_total,) int — type ID per column
        max_categories: int (static)

    Returns:
        Updated (ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts,
        ss_sum_sin, ss_sum_cos).
    """
    n_cols_v = col_indices.shape[0]

    def update_one_col(carry, li):
        ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos = carry
        col_idx = col_indices[li]
        safe_col_idx = jnp.clip(col_idx, 0, row_data.shape[0] - 1)
        x = row_data[safe_col_idx]
        type_id = col_type_ids[safe_col_idx]
        is_valid = (~jnp.isnan(x)) & (col_idx >= 0)
        is_valid_f = is_valid.astype(jnp.float32)

        # Count delta (applies to all types)
        ss_c = ss_c.at[cluster_id, li].add(-is_valid.astype(jnp.int32))

        # Continuous / Binary: sum_x -= x, sum_x_sq -= x^2
        clean_x = jnp.where(jnp.isnan(x), 0.0, x)
        is_sum_type = (type_id == CONTINUOUS_ID) | (type_id == BINARY_ID)
        sx_delta = clean_x * is_valid_f * is_sum_type.astype(jnp.float32)
        sxsq_delta = clean_x**2 * is_valid_f * (type_id == CONTINUOUS_ID).astype(jnp.float32)
        ss_sx = ss_sx.at[cluster_id, li].add(-sx_delta)
        ss_sxsq = ss_sxsq.at[cluster_id, li].add(-sxsq_delta)

        # Categorical / Ordinal: cat_counts[category] -= 1
        is_cat_type = (type_id == CATEGORICAL_ID) | (type_id == ORDINAL_ID)
        cat_idx = jnp.clip(clean_x.astype(jnp.int32), 0, max_categories - 1)
        cat_delta = is_valid_f * is_cat_type.astype(jnp.float32)
        ss_cat = ss_cat.at[cluster_id, li, cat_idx].add(-cat_delta)

        # Cyclic: sum_sin -= sin(x), sum_cos -= cos(x)
        is_cyc = (type_id == CYCLIC_ID).astype(jnp.float32)
        ss_sin = ss_sin.at[cluster_id, li].add(-jnp.sin(clean_x) * is_valid_f * is_cyc)
        ss_cos = ss_cos.at[cluster_id, li].add(-jnp.cos(clean_x) * is_valid_f * is_cyc)

        return (ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos), None

    carry = (ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos)
    (ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos), _ = jax.lax.scan(
        update_one_col, carry, jnp.arange(n_cols_v)
    )
    return ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos


def _add_row_to_suffstats(
    ss_counts: Array,
    ss_sum_x: Array,
    ss_sum_x_sq: Array,
    ss_cat_counts: Array,
    ss_sum_sin: Array,
    ss_sum_cos: Array,
    cluster_id: Array,
    row_data: Array,
    col_indices: Array,
    col_type_ids: Array,
    max_categories: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Add one row's contribution to a cluster's suffstats.

    Same structure as _remove_row_from_suffstats with positive deltas.
    """
    n_cols_v = col_indices.shape[0]

    def update_one_col(carry, li):
        ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos = carry
        col_idx = col_indices[li]
        safe_col_idx = jnp.clip(col_idx, 0, row_data.shape[0] - 1)
        x = row_data[safe_col_idx]
        type_id = col_type_ids[safe_col_idx]
        is_valid = (~jnp.isnan(x)) & (col_idx >= 0)
        is_valid_f = is_valid.astype(jnp.float32)

        ss_c = ss_c.at[cluster_id, li].add(is_valid.astype(jnp.int32))

        clean_x = jnp.where(jnp.isnan(x), 0.0, x)
        is_sum_type = (type_id == CONTINUOUS_ID) | (type_id == BINARY_ID)
        sx_delta = clean_x * is_valid_f * is_sum_type.astype(jnp.float32)
        sxsq_delta = clean_x**2 * is_valid_f * (type_id == CONTINUOUS_ID).astype(jnp.float32)
        ss_sx = ss_sx.at[cluster_id, li].add(sx_delta)
        ss_sxsq = ss_sxsq.at[cluster_id, li].add(sxsq_delta)

        is_cat_type = (type_id == CATEGORICAL_ID) | (type_id == ORDINAL_ID)
        cat_idx = jnp.clip(clean_x.astype(jnp.int32), 0, max_categories - 1)
        cat_delta = is_valid_f * is_cat_type.astype(jnp.float32)
        ss_cat = ss_cat.at[cluster_id, li, cat_idx].add(cat_delta)

        is_cyc = (type_id == CYCLIC_ID).astype(jnp.float32)
        ss_sin = ss_sin.at[cluster_id, li].add(jnp.sin(clean_x) * is_valid_f * is_cyc)
        ss_cos = ss_cos.at[cluster_id, li].add(jnp.cos(clean_x) * is_valid_f * is_cyc)

        return (ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos), None

    carry = (ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos)
    (ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos), _ = jax.lax.scan(
        update_one_col, carry, jnp.arange(n_cols_v)
    )
    return ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos
