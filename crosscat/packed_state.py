"""Padded, JIT-compatible state representation for CrossCat.

Coexists with the original CrossCatState API via pack_state() / unpack_state().
All data structures use fixed-size padded arrays, enabling jax.jit, jax.vmap,
and jax.lax.scan without Python-level branching.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln

from crosscat.types import ColumnHypers, ColumnType, CrossCatState, SufficientStats, ViewState

# ---------------------------------------------------------------------------
# Column type integer encoding (for JIT-compatible dispatch)
# ---------------------------------------------------------------------------

CONTINUOUS_ID = 0
CATEGORICAL_ID = 1
ORDINAL_ID = 2
BINARY_ID = 3
CYCLIC_ID = 4

_TYPE_TO_ID = {
    ColumnType.CONTINUOUS: CONTINUOUS_ID,
    ColumnType.CATEGORICAL: CATEGORICAL_ID,
    ColumnType.ORDINAL: ORDINAL_ID,
    ColumnType.BINARY: BINARY_ID,
    ColumnType.CYCLIC: CYCLIC_ID,
}

_ID_TO_TYPE = {v: k for k, v in _TYPE_TO_ID.items()}

# ---------------------------------------------------------------------------
# Packed state dataclass
# ---------------------------------------------------------------------------

# Array field names (dynamic pytree children) — order matters for flatten/unflatten
_ARRAY_FIELDS = (
    "column_assignments",
    "column_crp_alpha",
    "n_views",
    "view_mask",
    "col_type_ids",
    "hyper_mu",
    "hyper_r",
    "hyper_s",
    "hyper_nu",
    "hyper_dirichlet_alpha",
    "hyper_alpha",
    "hyper_beta",
    "hyper_kappa",
    "hyper_vm_mu",
    "view_column_indices",
    "view_n_columns",
    "view_row_assignments",
    "view_n_clusters",
    "view_row_crp_alpha",
    "ss_counts",
    "ss_sum_x",
    "ss_sum_x_sq",
    "ss_cat_counts",
    "ss_sum_sin",
    "ss_sum_cos",
)

# Static field names (pytree auxiliary data)
_STATIC_FIELDS = ("n_rows", "n_cols", "max_views", "max_clusters", "max_categories", "max_cols_per_view")


@jax.tree_util.register_pytree_node_class
@dataclass
class PackedCrossCatState:
    """JIT-compatible CrossCat state with padded fixed-size arrays.

    All view data has a leading (max_views,) dimension.
    All cluster data has (max_views, max_clusters) dimensions.
    Invalid entries are masked via view_mask, n_views, n_clusters.
    """

    # Column partition
    column_assignments: Array      # (n_cols,) int
    column_crp_alpha: Array        # scalar
    n_views: Array                 # scalar int
    view_mask: Array               # (max_views,) bool

    # Column type and hyperparameters — flat (n_cols,) arrays
    col_type_ids: Array            # (n_cols,) int
    hyper_mu: Array                # (n_cols,)
    hyper_r: Array                 # (n_cols,)
    hyper_s: Array                 # (n_cols,)
    hyper_nu: Array                # (n_cols,)
    hyper_dirichlet_alpha: Array   # (n_cols,)
    hyper_alpha: Array             # (n_cols,)
    hyper_beta: Array              # (n_cols,)
    hyper_kappa: Array             # (n_cols,)
    hyper_vm_mu: Array             # (n_cols,)

    # View data — leading (max_views,) dimension
    view_column_indices: Array     # (max_views, max_cols_per_view) int, -1=invalid
    view_n_columns: Array          # (max_views,) int
    view_row_assignments: Array    # (max_views, n_rows) int
    view_n_clusters: Array         # (max_views,) int
    view_row_crp_alpha: Array      # (max_views,)

    # Sufficient statistics — (max_views, max_clusters, max_cols_per_view[, max_cats])
    ss_counts: Array               # int
    ss_sum_x: Array
    ss_sum_x_sq: Array
    ss_cat_counts: Array           # (max_views, max_clusters, max_cols_per_view, max_cats)
    ss_sum_sin: Array
    ss_sum_cos: Array

    # Static configuration (not traced by JAX)
    n_rows: int = 0
    n_cols: int = 0
    max_views: int = 16
    max_clusters: int = 32
    max_categories: int = 16
    max_cols_per_view: int = 16

    def tree_flatten(self):
        children = [getattr(self, name) for name in _ARRAY_FIELDS]
        aux_data = tuple(getattr(self, name) for name in _STATIC_FIELDS)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        kwargs = dict(zip(_ARRAY_FIELDS, children, strict=True))
        kwargs.update(dict(zip(_STATIC_FIELDS, aux_data, strict=True)))
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Pack / unpack conversion
# ---------------------------------------------------------------------------


def pack_state(
    state: CrossCatState,
    *,
    max_views: int = 16,
    max_clusters: int = 32,
    max_categories: int = 16,
) -> PackedCrossCatState:
    """Convert a CrossCatState (Python lists) into a PackedCrossCatState (padded arrays).

    Args:
        state: Original CrossCat state.
        max_views: Maximum number of views (padding dimension).
        max_clusters: Maximum clusters per view.
        max_categories: Maximum categories for categorical/ordinal columns.

    Returns:
        Padded, JIT-compatible state.
    """
    n_rows = state.n_rows
    n_cols = state.n_cols
    n_views = state.n_views
    max_cols_per_view = max(len(v.column_indices) for v in state.views)
    max_cols_per_view = max(max_cols_per_view, 1)

    # Column assignments and CRP alpha
    col_assignments = jnp.array(state.column_assignments, dtype=jnp.int32)
    col_crp_alpha = jnp.array(float(state.column_crp_alpha))

    # View mask
    view_mask = jnp.zeros(max_views, dtype=jnp.bool_)
    view_mask = view_mask.at[:n_views].set(True)

    # Column types as int IDs
    col_type_ids = jnp.array([_TYPE_TO_ID[ct] for ct in state.column_types], dtype=jnp.int32)

    # Pack hyperparameters into flat arrays
    hyper_mu = jnp.zeros(n_cols)
    hyper_r = jnp.ones(n_cols)
    hyper_s = jnp.ones(n_cols)
    hyper_nu = jnp.ones(n_cols) * 2.0
    hyper_dirichlet_alpha = jnp.ones(n_cols)
    hyper_alpha = jnp.ones(n_cols)
    hyper_beta = jnp.ones(n_cols)
    hyper_kappa = jnp.ones(n_cols)
    hyper_vm_mu = jnp.ones(n_cols) * jnp.pi

    for j, h in enumerate(state.column_hypers):
        if h.mu is not None:
            hyper_mu = hyper_mu.at[j].set(float(h.mu))
        if h.r is not None:
            hyper_r = hyper_r.at[j].set(float(h.r))
        if h.s is not None:
            hyper_s = hyper_s.at[j].set(float(h.s))
        if h.nu is not None:
            hyper_nu = hyper_nu.at[j].set(float(h.nu))
        if h.dirichlet_alpha is not None:
            hyper_dirichlet_alpha = hyper_dirichlet_alpha.at[j].set(float(h.dirichlet_alpha))
        if h.alpha is not None:
            hyper_alpha = hyper_alpha.at[j].set(float(h.alpha))
        if h.beta is not None:
            hyper_beta = hyper_beta.at[j].set(float(h.beta))
        if h.kappa is not None:
            hyper_kappa = hyper_kappa.at[j].set(float(h.kappa))
        if h.vm_mu is not None:
            hyper_vm_mu = hyper_vm_mu.at[j].set(float(h.vm_mu))

    # Pack view data
    view_column_indices = jnp.full((max_views, max_cols_per_view), -1, dtype=jnp.int32)
    view_n_columns = jnp.zeros(max_views, dtype=jnp.int32)
    view_row_assignments = jnp.zeros((max_views, n_rows), dtype=jnp.int32)
    view_n_clusters = jnp.zeros(max_views, dtype=jnp.int32)
    view_row_crp_alpha = jnp.ones(max_views)

    ss_counts = jnp.zeros((max_views, max_clusters, max_cols_per_view), dtype=jnp.int32)
    ss_sum_x = jnp.zeros((max_views, max_clusters, max_cols_per_view))
    ss_sum_x_sq = jnp.zeros((max_views, max_clusters, max_cols_per_view))
    ss_cat_counts = jnp.zeros(
        (max_views, max_clusters, max_cols_per_view, max_categories)
    )
    ss_sum_sin = jnp.zeros((max_views, max_clusters, max_cols_per_view))
    ss_sum_cos = jnp.zeros((max_views, max_clusters, max_cols_per_view))

    for v_idx, view in enumerate(state.views):
        n_cols_v = len(view.column_indices)
        view_column_indices = view_column_indices.at[v_idx, :n_cols_v].set(
            jnp.array(view.column_indices, dtype=jnp.int32)
        )
        view_n_columns = view_n_columns.at[v_idx].set(n_cols_v)
        view_row_assignments = view_row_assignments.at[v_idx].set(
            jnp.array(view.row_assignments, dtype=jnp.int32)
        )
        nc = int(jnp.max(view.row_assignments)) + 1
        view_n_clusters = view_n_clusters.at[v_idx].set(nc)
        view_row_crp_alpha = view_row_crp_alpha.at[v_idx].set(float(view.row_crp_alpha))

        if view.suffstats is not None:
            for c_idx, cluster_ss in enumerate(view.suffstats):
                for l_idx, ss in enumerate(cluster_ss):
                    ss_counts = ss_counts.at[v_idx, c_idx, l_idx].set(int(ss.count))
                    if ss.sum_x is not None:
                        ss_sum_x = ss_sum_x.at[v_idx, c_idx, l_idx].set(float(ss.sum_x))
                    if ss.sum_x_sq is not None:
                        ss_sum_x_sq = ss_sum_x_sq.at[v_idx, c_idx, l_idx].set(
                            float(ss.sum_x_sq)
                        )
                    if ss.category_counts is not None:
                        nc_cats = min(len(ss.category_counts), max_categories)
                        ss_cat_counts = ss_cat_counts.at[
                            v_idx, c_idx, l_idx, :nc_cats
                        ].set(ss.category_counts[:nc_cats])
                    if ss.sum_sin is not None:
                        ss_sum_sin = ss_sum_sin.at[v_idx, c_idx, l_idx].set(
                            float(ss.sum_sin)
                        )
                    if ss.sum_cos is not None:
                        ss_sum_cos = ss_sum_cos.at[v_idx, c_idx, l_idx].set(
                            float(ss.sum_cos)
                        )

    return PackedCrossCatState(
        column_assignments=col_assignments,
        column_crp_alpha=col_crp_alpha,
        n_views=jnp.array(n_views, dtype=jnp.int32),
        view_mask=view_mask,
        col_type_ids=col_type_ids,
        hyper_mu=hyper_mu,
        hyper_r=hyper_r,
        hyper_s=hyper_s,
        hyper_nu=hyper_nu,
        hyper_dirichlet_alpha=hyper_dirichlet_alpha,
        hyper_alpha=hyper_alpha,
        hyper_beta=hyper_beta,
        hyper_kappa=hyper_kappa,
        hyper_vm_mu=hyper_vm_mu,
        view_column_indices=view_column_indices,
        view_n_columns=view_n_columns,
        view_row_assignments=view_row_assignments,
        view_n_clusters=view_n_clusters,
        view_row_crp_alpha=view_row_crp_alpha,
        ss_counts=ss_counts,
        ss_sum_x=ss_sum_x,
        ss_sum_x_sq=ss_sum_x_sq,
        ss_cat_counts=ss_cat_counts,
        ss_sum_sin=ss_sum_sin,
        ss_sum_cos=ss_sum_cos,
        n_rows=n_rows,
        n_cols=n_cols,
        max_views=max_views,
        max_clusters=max_clusters,
        max_categories=max_categories,
        max_cols_per_view=max_cols_per_view,
    )


def unpack_state(
    packed: PackedCrossCatState,
    column_types: list[ColumnType],
) -> CrossCatState:
    """Convert a PackedCrossCatState back into a CrossCatState.

    Args:
        packed: Packed state.
        column_types: Column type list (not stored in packed state).

    Returns:
        Standard CrossCatState with Python lists.
    """
    n_cols = packed.n_cols
    n_rows = packed.n_rows
    n_views = int(packed.n_views)

    # Unpack hyperparameters
    col_hypers = []
    for j in range(n_cols):
        ct = column_types[j]
        if ct == ColumnType.CONTINUOUS:
            h = ColumnHypers(
                column_type=ct,
                mu=packed.hyper_mu[j],
                r=packed.hyper_r[j],
                s=packed.hyper_s[j],
                nu=packed.hyper_nu[j],
            )
        elif ct == ColumnType.CATEGORICAL:
            h = ColumnHypers(column_type=ct, dirichlet_alpha=packed.hyper_dirichlet_alpha[j])
        elif ct == ColumnType.BINARY:
            h = ColumnHypers(
                column_type=ct, alpha=packed.hyper_alpha[j], beta=packed.hyper_beta[j]
            )
        elif ct == ColumnType.ORDINAL:
            h = ColumnHypers(column_type=ct, cutpoints=None)
        elif ct == ColumnType.CYCLIC:
            h = ColumnHypers(
                column_type=ct, kappa=packed.hyper_kappa[j], vm_mu=packed.hyper_vm_mu[j]
            )
        else:
            raise ValueError(f"Unknown column type: {ct}")
        col_hypers.append(h)

    # Unpack views
    views = []
    for v in range(n_views):
        n_cols_v = int(packed.view_n_columns[v])
        col_indices = packed.view_column_indices[v, :n_cols_v]
        row_assigns = packed.view_row_assignments[v]
        nc = int(packed.view_n_clusters[v])

        # Unpack suffstats
        suffstats = []
        for c in range(nc):
            cluster_ss = []
            for li in range(n_cols_v):
                col_idx = int(col_indices[li])
                ct = column_types[col_idx]
                count = packed.ss_counts[v, c, li]
                if ct == ColumnType.CONTINUOUS:
                    ss = SufficientStats(
                        column_type=ct,
                        count=count,
                        sum_x=packed.ss_sum_x[v, c, li],
                        sum_x_sq=packed.ss_sum_x_sq[v, c, li],
                    )
                elif ct in (ColumnType.CATEGORICAL, ColumnType.ORDINAL):
                    ss = SufficientStats(
                        column_type=ct,
                        count=count,
                        category_counts=packed.ss_cat_counts[v, c, li],
                    )
                elif ct == ColumnType.BINARY:
                    ss = SufficientStats(
                        column_type=ct, count=count, sum_x=packed.ss_sum_x[v, c, li]
                    )
                elif ct == ColumnType.CYCLIC:
                    ss = SufficientStats(
                        column_type=ct,
                        count=count,
                        sum_sin=packed.ss_sum_sin[v, c, li],
                        sum_cos=packed.ss_sum_cos[v, c, li],
                    )
                else:
                    raise ValueError(f"Unknown column type: {ct}")
                cluster_ss.append(ss)
            suffstats.append(cluster_ss)

        views.append(
            ViewState(
                column_indices=col_indices,
                row_assignments=row_assigns,
                row_crp_alpha=packed.view_row_crp_alpha[v],
                suffstats=suffstats,
            )
        )

    return CrossCatState(
        column_assignments=packed.column_assignments,
        column_crp_alpha=packed.column_crp_alpha,
        column_hypers=col_hypers,
        column_types=column_types,
        views=views,
        n_rows=n_rows,
        n_cols=n_cols,
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
    membership = (
        row_assignments[:, None] == jnp.arange(max_clusters)[None, :]
    ).astype(jnp.float32)

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
    """
    max_c = packed.max_clusters
    max_cat = packed.max_categories
    n_views = int(packed.n_views)

    ss_counts = packed.ss_counts
    ss_sum_x = packed.ss_sum_x
    ss_sum_x_sq = packed.ss_sum_x_sq
    ss_cat_counts = packed.ss_cat_counts
    ss_sum_sin = packed.ss_sum_sin
    ss_sum_cos = packed.ss_sum_cos

    for v in range(n_views):
        n_cols_v = int(packed.view_n_columns[v])
        col_indices = packed.view_column_indices[v, :n_cols_v]
        row_assigns = packed.view_row_assignments[v]
        nc = int(packed.view_n_clusters[v])

        counts, sx, sxsq, cc, ssin, scos = compute_suffstats_vectorized(
            data, col_indices, packed.col_type_ids, row_assigns, nc, max_c, max_cat
        )

        ss_counts = ss_counts.at[v, :, :n_cols_v].set(counts[:, :n_cols_v])
        ss_sum_x = ss_sum_x.at[v, :, :n_cols_v].set(sx[:, :n_cols_v])
        ss_sum_x_sq = ss_sum_x_sq.at[v, :, :n_cols_v].set(sxsq[:, :n_cols_v])
        ss_cat_counts = ss_cat_counts.at[v, :, :n_cols_v, :].set(cc[:, :n_cols_v, :])
        ss_sum_sin = ss_sum_sin.at[v, :, :n_cols_v].set(ssin[:, :n_cols_v])
        ss_sum_cos = ss_sum_cos.at[v, :, :n_cols_v].set(scos[:, :n_cols_v])

    return PackedCrossCatState(
        **{name: getattr(packed, name) for name in _ARRAY_FIELDS
           if name not in ("ss_counts", "ss_sum_x", "ss_sum_x_sq",
                           "ss_cat_counts", "ss_sum_sin", "ss_sum_cos")},
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
        sxsq_delta = (
            clean_x**2 * is_valid_f * (type_id == CONTINUOUS_ID).astype(jnp.float32)
        )
        ss_sx = ss_sx.at[cluster_id, li].add(-sx_delta)
        ss_sxsq = ss_sxsq.at[cluster_id, li].add(-sxsq_delta)

        # Categorical / Ordinal: cat_counts[category] -= 1
        is_cat_type = (type_id == CATEGORICAL_ID) | (type_id == ORDINAL_ID)
        cat_idx = jnp.clip(clean_x.astype(jnp.int32), 0, max_categories - 1)
        cat_delta = is_valid_f * is_cat_type.astype(jnp.float32)
        ss_cat = ss_cat.at[cluster_id, li, cat_idx].add(-cat_delta)

        # Cyclic: sum_sin -= sin(x), sum_cos -= cos(x)
        is_cyc = (type_id == CYCLIC_ID).astype(jnp.float32)
        ss_sin = ss_sin.at[cluster_id, li].add(
            -jnp.sin(clean_x) * is_valid_f * is_cyc
        )
        ss_cos = ss_cos.at[cluster_id, li].add(
            -jnp.cos(clean_x) * is_valid_f * is_cyc
        )

        return (ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos), None

    carry = (ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos)
    (ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos), _ = (
        jax.lax.scan(update_one_col, carry, jnp.arange(n_cols_v))
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
        sxsq_delta = (
            clean_x**2 * is_valid_f * (type_id == CONTINUOUS_ID).astype(jnp.float32)
        )
        ss_sx = ss_sx.at[cluster_id, li].add(sx_delta)
        ss_sxsq = ss_sxsq.at[cluster_id, li].add(sxsq_delta)

        is_cat_type = (type_id == CATEGORICAL_ID) | (type_id == ORDINAL_ID)
        cat_idx = jnp.clip(clean_x.astype(jnp.int32), 0, max_categories - 1)
        cat_delta = is_valid_f * is_cat_type.astype(jnp.float32)
        ss_cat = ss_cat.at[cluster_id, li, cat_idx].add(cat_delta)

        is_cyc = (type_id == CYCLIC_ID).astype(jnp.float32)
        ss_sin = ss_sin.at[cluster_id, li].add(
            jnp.sin(clean_x) * is_valid_f * is_cyc
        )
        ss_cos = ss_cos.at[cluster_id, li].add(
            jnp.cos(clean_x) * is_valid_f * is_cyc
        )

        return (ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos), None

    carry = (ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos)
    (ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos), _ = (
        jax.lax.scan(update_one_col, carry, jnp.arange(n_cols_v))
    )
    return ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos


# ---------------------------------------------------------------------------
# JIT-compatible scoring functions (unified type dispatch)
# ---------------------------------------------------------------------------


def _log_bessel_i0(x: Array) -> Array:
    """Log of modified Bessel function I_0(x)."""
    return jnp.where(
        x < 3.75,
        jnp.log(
            1.0
            + 3.5156229 * (x / 3.75) ** 2
            + 3.0899424 * (x / 3.75) ** 4
            + 1.2067492 * (x / 3.75) ** 6
            + 0.2659732 * (x / 3.75) ** 8
            + 0.0360768 * (x / 3.75) ** 10
            + 0.0045813 * (x / 3.75) ** 12
        ),
        x - 0.5 * jnp.log(2.0 * jnp.pi * jnp.maximum(x, 1e-30)),
    )


def _ng_log_marginal(n, sum_x, sum_x_sq, mu0, r, s, nu):
    """Normal-Gamma log marginal likelihood (element-wise)."""
    n = n.astype(jnp.float32)
    r_n = r + n
    nu_n = nu + n
    nu_s = nu * s
    mean = jnp.where(n > 0, sum_x / jnp.maximum(n, 1.0), 0.0)
    nu_n_s_n = (
        nu_s + sum_x_sq - sum_x**2 / jnp.maximum(n, 1.0)
        + r * n * (mu0 - mean) ** 2 / jnp.maximum(r_n, 1e-30)
    )
    nu_n_s_n = jnp.where(n > 0, nu_n_s_n, nu_s)
    # Clamp to avoid log of negative
    nu_n_s_n = jnp.maximum(nu_n_s_n, 1e-30)

    log_ml = (
        -0.5 * n * jnp.log(2.0 * jnp.pi)
        + 0.5 * jnp.log(r / jnp.maximum(r_n, 1e-30))
        + 0.5 * nu * jnp.log(jnp.maximum(nu_s / 2.0, 1e-30))
        - 0.5 * nu_n * jnp.log(jnp.maximum(nu_n_s_n / 2.0, 1e-30))
        + gammaln(nu_n / 2.0)
        - gammaln(nu / 2.0)
    )
    return jnp.where(n > 0, log_ml, 0.0)


def _dc_log_marginal(n, cat_counts, dir_alpha):
    """Dirichlet-Categorical log marginal likelihood.

    cat_counts: (..., max_categories)
    dir_alpha: scalar or (...,)
    """
    n = n.astype(jnp.float32)
    k = jnp.array(cat_counts.shape[-1], dtype=jnp.float32)
    # Expand dir_alpha if needed
    alpha = dir_alpha
    log_ml = (
        jnp.sum(gammaln(cat_counts + alpha), axis=-1)
        - gammaln(n + k * alpha)
        - k * gammaln(alpha)
        + gammaln(k * alpha)
    )
    return jnp.where(n > 0, log_ml, 0.0)


def _bb_log_marginal(n, sum_x, alpha, beta):
    """Beta-Bernoulli log marginal likelihood."""
    n = n.astype(jnp.float32)
    k = sum_x
    log_ml = (
        gammaln(alpha + beta)
        - gammaln(n + alpha + beta)
        + gammaln(k + alpha)
        - gammaln(alpha)
        + gammaln(n - k + beta)
        - gammaln(beta)
    )
    return jnp.where(n > 0, log_ml, 0.0)


def _vm_log_marginal(n, sum_sin, sum_cos, kappa):
    """Von Mises log marginal likelihood."""
    n = n.astype(jnp.float32)
    r_length = jnp.sqrt(sum_sin**2 + sum_cos**2)
    log_ml = -n * jnp.log(2.0 * jnp.pi) - n * _log_bessel_i0(kappa) + kappa * r_length
    return jnp.where(n > 0, log_ml, 0.0)


def unified_log_marginal(
    type_id, count, sum_x, sum_x_sq, cat_counts, sum_sin, sum_cos,
    mu, r, s, nu, dir_alpha, alpha, beta, kappa, vm_mu,
):
    """Compute log marginal likelihood for any column type without Python branching.

    Computes ALL type results and selects the correct one via jnp.where.
    This wastes trivial compute but enables full JIT compilation.
    """
    continuous_score = _ng_log_marginal(count, sum_x, sum_x_sq, mu, r, s, nu)
    cat_score = _dc_log_marginal(count, cat_counts, dir_alpha)
    binary_score = _bb_log_marginal(count, sum_x, alpha, beta)
    ordinal_score = _dc_log_marginal(count, cat_counts, jnp.ones_like(dir_alpha))
    cyclic_score = _vm_log_marginal(count, sum_sin, sum_cos, kappa)

    return jnp.where(
        type_id == CONTINUOUS_ID, continuous_score,
        jnp.where(
            type_id == CATEGORICAL_ID, cat_score,
            jnp.where(
                type_id == ORDINAL_ID, ordinal_score,
                jnp.where(
                    type_id == BINARY_ID, binary_score,
                    cyclic_score,
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Posterior predictive scoring (for row assignment sweep)
# ---------------------------------------------------------------------------


def _ng_posterior_predictive_logp(x, count, sum_x, sum_x_sq, mu0, r, s, nu):
    """Normal-Gamma posterior predictive log p(x | suffstats, hypers)."""
    n = count.astype(jnp.float32)
    r_n = r + n
    mu_n = (r * mu0 + sum_x) / jnp.maximum(r_n, 1e-30)
    nu_n = nu + n
    nu_s = nu * s
    mean = jnp.where(n > 0, sum_x / jnp.maximum(n, 1.0), 0.0)
    nu_n_s_n = (
        nu_s + sum_x_sq - sum_x**2 / jnp.maximum(n, 1.0)
        + r * n * (mu0 - mean) ** 2 / jnp.maximum(r_n, 1e-30)
    )
    nu_n_s_n = jnp.where(n > 0, nu_n_s_n, nu_s)
    nu_n_s_n = jnp.maximum(nu_n_s_n, 1e-30)

    df = nu_n
    loc = mu_n
    scale_sq = (nu_n_s_n / jnp.maximum(nu_n, 1e-30)) * (1.0 + 1.0 / jnp.maximum(r_n, 1e-30))
    scale = jnp.sqrt(jnp.maximum(scale_sq, 1e-30))
    z = (x - loc) / scale

    log_p = (
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * jnp.log(df * jnp.pi)
        - jnp.log(scale)
        - (df + 1.0) / 2.0 * jnp.log(1.0 + z**2 / jnp.maximum(df, 1e-30))
    )
    return log_p


def _dc_posterior_predictive_logp(x, count, cat_counts, dir_alpha):
    """Dirichlet-Categorical posterior predictive log p(x | suffstats, hypers)."""
    n = count.astype(jnp.float32)
    k = jnp.array(cat_counts.shape[-1], dtype=jnp.float32)
    probs = (cat_counts + dir_alpha) / jnp.maximum(n + k * dir_alpha, 1e-30)
    idx = x.astype(jnp.int32)
    idx = jnp.clip(idx, 0, cat_counts.shape[-1] - 1)
    return jnp.log(jnp.maximum(probs[idx], 1e-30))


def _bb_posterior_predictive_logp(x, count, sum_x, alpha, beta):
    """Beta-Bernoulli posterior predictive log p(x | suffstats, hypers)."""
    n = count.astype(jnp.float32)
    p1 = (sum_x + alpha) / jnp.maximum(n + alpha + beta, 1e-30)
    log_p1 = jnp.log(jnp.maximum(p1, 1e-30))
    log_p0 = jnp.log(jnp.maximum(1.0 - p1, 1e-30))
    return jnp.where(x > 0.5, log_p1, log_p0)


def _vm_posterior_predictive_logp(x, count, sum_sin, sum_cos, kappa, vm_mu):
    """Von Mises posterior predictive log p(x | suffstats, hypers)."""
    total_sin = sum_sin + kappa * jnp.sin(vm_mu)
    total_cos = sum_cos + kappa * jnp.cos(vm_mu)
    r_post = jnp.sqrt(total_sin**2 + total_cos**2)
    mu_post = jnp.arctan2(total_sin, total_cos)
    kappa_post = r_post
    return kappa_post * jnp.cos(x - mu_post) - jnp.log(2.0 * jnp.pi) - _log_bessel_i0(kappa_post)


def unified_posterior_predictive_logp(
    x, type_id, count, sum_x, sum_x_sq, cat_counts, sum_sin, sum_cos,
    mu, r, s, nu, dir_alpha, alpha, beta, kappa, vm_mu,
):
    """Compute posterior predictive logp for any column type without Python branching."""
    cont = _ng_posterior_predictive_logp(x, count, sum_x, sum_x_sq, mu, r, s, nu)
    cat = _dc_posterior_predictive_logp(x, count, cat_counts, dir_alpha)
    binary = _bb_posterior_predictive_logp(x, count, sum_x, alpha, beta)
    ordinal = _dc_posterior_predictive_logp(x, count, cat_counts, jnp.ones_like(dir_alpha))
    cyclic = _vm_posterior_predictive_logp(x, count, sum_sin, sum_cos, kappa, vm_mu)

    return jnp.where(
        type_id == CONTINUOUS_ID, cont,
        jnp.where(
            type_id == CATEGORICAL_ID, cat,
            jnp.where(type_id == ORDINAL_ID, ordinal,
                       jnp.where(type_id == BINARY_ID, binary, cyclic)),
        ),
    )


# ---------------------------------------------------------------------------
# Vectorized row assignment sweep (critical path)
# ---------------------------------------------------------------------------


def _score_row_all_clusters(
    row_data: Array,
    col_indices: Array,
    n_columns: int,
    col_type_ids: Array,
    cluster_counts: Array,
    ss_counts: Array,
    ss_sum_x: Array,
    ss_sum_x_sq: Array,
    ss_cat_counts: Array,
    ss_sum_sin: Array,
    ss_sum_cos: Array,
    hyper_mu: Array,
    hyper_r: Array,
    hyper_s: Array,
    hyper_nu: Array,
    hyper_dirichlet_alpha: Array,
    hyper_alpha: Array,
    hyper_beta: Array,
    hyper_kappa: Array,
    hyper_vm_mu: Array,
    crp_alpha: Array,
    max_clusters: int,
) -> Array:
    """Score a row against all clusters (existing + one new) simultaneously.

    Returns log_probs of shape (max_clusters + 1,) where the last entry
    is the new-cluster score.
    """
    # CRP prior: log(count_c) for existing, log(alpha) for new
    log_prior = jnp.log(jnp.maximum(cluster_counts.astype(jnp.float32), 1e-30))
    # Mask empty clusters
    log_prior = jnp.where(cluster_counts > 0, log_prior, -jnp.inf)

    # Score each cluster: sum of posterior predictive logp across columns
    def score_one_cluster(c):
        log_lik = jnp.array(0.0)
        for li in range(n_columns):
            col_idx = col_indices[li]
            x = row_data[col_idx]
            is_valid = ~jnp.isnan(x) & (col_idx >= 0)
            type_id = col_type_ids[col_idx]

            logp = unified_posterior_predictive_logp(
                x, type_id,
                ss_counts[c, li].astype(jnp.float32),
                ss_sum_x[c, li], ss_sum_x_sq[c, li],
                ss_cat_counts[c, li],
                ss_sum_sin[c, li], ss_sum_cos[c, li],
                hyper_mu[col_idx], hyper_r[col_idx],
                hyper_s[col_idx], hyper_nu[col_idx],
                hyper_dirichlet_alpha[col_idx],
                hyper_alpha[col_idx], hyper_beta[col_idx],
                hyper_kappa[col_idx], hyper_vm_mu[col_idx],
            )
            log_lik = log_lik + jnp.where(is_valid, logp, 0.0)
        return log_lik

    log_liks = jnp.array([score_one_cluster(c) for c in range(max_clusters)])
    log_probs_existing = log_prior + log_liks

    # New cluster: empty suffstats → prior predictive
    log_lik_new = jnp.array(0.0)
    for li in range(n_columns):
        col_idx = col_indices[li]
        x = row_data[col_idx]
        is_valid = ~jnp.isnan(x) & (col_idx >= 0)
        type_id = col_type_ids[col_idx]

        # Empty suffstats (zeros)
        logp = unified_posterior_predictive_logp(
            x, type_id,
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
            jnp.zeros(ss_cat_counts.shape[-1]),
            jnp.array(0.0), jnp.array(0.0),
            hyper_mu[col_idx], hyper_r[col_idx],
            hyper_s[col_idx], hyper_nu[col_idx],
            hyper_dirichlet_alpha[col_idx],
            hyper_alpha[col_idx], hyper_beta[col_idx],
            hyper_kappa[col_idx], hyper_vm_mu[col_idx],
        )
        log_lik_new = log_lik_new + jnp.where(is_valid, logp, 0.0)

    log_prior_new = jnp.log(crp_alpha)
    log_prob_new = log_prior_new + log_lik_new

    return jnp.concatenate([log_probs_existing, jnp.array([log_prob_new])])


def packed_transition_row_assignments(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
) -> PackedCrossCatState:
    """Gibbs sweep over row assignments using packed state.

    Processes each view sequentially, rows sequentially within a view
    (necessary for correctness), but scores all clusters in parallel.
    """
    n_rows = packed.n_rows
    n_views = int(packed.n_views)
    max_c = packed.max_clusters

    view_keys = jax.random.split(rng_key, n_views)

    new_row_assigns = jnp.array(packed.view_row_assignments)
    new_n_clusters = jnp.array(packed.view_n_clusters)

    for v in range(n_views):
        row_keys = jax.random.split(view_keys[v], n_rows)
        row_assigns = new_row_assigns[v]
        n_cols_v = int(packed.view_n_columns[v])
        col_indices = packed.view_column_indices[v, :n_cols_v]
        alpha = packed.view_row_crp_alpha[v]

        # Working copy of suffstats for this view
        v_ss_counts = packed.ss_counts[v]
        v_ss_sum_x = packed.ss_sum_x[v]
        v_ss_sum_x_sq = packed.ss_sum_x_sq[v]
        v_ss_cat = packed.ss_cat_counts[v]
        v_ss_sin = packed.ss_sum_sin[v]
        v_ss_cos = packed.ss_sum_cos[v]

        n_clusters = int(new_n_clusters[v])

        for i in range(n_rows):
            # Cluster counts excluding row i
            counts = jnp.array(
                [jnp.sum(row_assigns.at[i].set(-1) == c) for c in range(max_c)]
            ).astype(jnp.int32)

            # Score all clusters
            log_probs = _score_row_all_clusters(
                data[i], col_indices, n_cols_v, packed.col_type_ids,
                counts,
                v_ss_counts, v_ss_sum_x, v_ss_sum_x_sq, v_ss_cat, v_ss_sin, v_ss_cos,
                packed.hyper_mu, packed.hyper_r, packed.hyper_s, packed.hyper_nu,
                packed.hyper_dirichlet_alpha, packed.hyper_alpha, packed.hyper_beta,
                packed.hyper_kappa, packed.hyper_vm_mu,
                alpha, max_c,
            )

            # Numerical stability
            log_probs = log_probs - jnp.max(log_probs)
            chosen = int(jax.random.categorical(row_keys[i], log_probs))

            if chosen >= max_c:
                # New cluster: use next available slot
                chosen = n_clusters
                n_clusters = min(n_clusters + 1, max_c)

            row_assigns = row_assigns.at[i].set(chosen)

        # Compact cluster indices
        unique_clusters = jnp.unique(row_assigns)
        remap = jnp.full(max_c + 1, 0, dtype=jnp.int32)
        for new_idx, old_idx in enumerate(unique_clusters.tolist()):
            remap = remap.at[int(old_idx)].set(new_idx)
        row_assigns = remap[row_assigns]
        n_clusters = int(jnp.max(row_assigns)) + 1

        new_row_assigns = new_row_assigns.at[v].set(row_assigns)
        new_n_clusters = new_n_clusters.at[v].set(n_clusters)

    # Create updated packed state and recompute suffstats
    packed = PackedCrossCatState(
        **{name: (new_row_assigns if name == "view_row_assignments"
                  else new_n_clusters if name == "view_n_clusters"
                  else getattr(packed, name))
           for name in _ARRAY_FIELDS},
        **{name: getattr(packed, name) for name in _STATIC_FIELDS},
    )
    return recompute_all_suffstats(packed, data)


# ---------------------------------------------------------------------------
# Vectorized hyperparameter sampling
# ---------------------------------------------------------------------------


def packed_transition_column_hypers(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
) -> PackedCrossCatState:
    """Grid-based Gibbs sampling for column hyperparameters using packed state."""
    n_cols = packed.n_cols
    keys = jax.random.split(rng_key, n_cols)

    new_mu = jnp.array(packed.hyper_mu)
    new_r = jnp.array(packed.hyper_r)
    new_s = jnp.array(packed.hyper_s)
    new_nu = jnp.array(packed.hyper_nu)
    new_dir_alpha = jnp.array(packed.hyper_dirichlet_alpha)
    new_alpha = jnp.array(packed.hyper_alpha)
    new_beta = jnp.array(packed.hyper_beta)
    new_kappa = jnp.array(packed.hyper_kappa)

    for j in range(n_cols):
        type_id = int(packed.col_type_ids[j])
        v_idx = int(packed.column_assignments[j])
        n_cols_v = int(packed.view_n_columns[v_idx])

        # Find local index
        local_idx = -1
        for li in range(n_cols_v):
            if int(packed.view_column_indices[v_idx, li]) == j:
                local_idx = li
                break
        if local_idx < 0:
            continue

        nc = int(packed.view_n_clusters[v_idx])

        if type_id == CONTINUOUS_ID:
            col_data = data[:, j]
            data_var = float(jnp.var(col_data)) + 1e-6
            data_mean = float(jnp.mean(col_data))
            data_std = float(jnp.std(col_data)) + 1e-6

            k1, k2, k3 = jax.random.split(keys[j], 3)
            cur_mu = float(new_mu[j])
            cur_r = float(new_r[j])
            cur_nu = float(new_nu[j])

            # Sample s
            s_grid = data_var * jnp.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0])
            scores_s = jnp.array([
                sum(
                    float(_ng_log_marginal(
                        packed.ss_counts[v_idx, c, local_idx],
                        packed.ss_sum_x[v_idx, c, local_idx],
                        packed.ss_sum_x_sq[v_idx, c, local_idx],
                        jnp.array(cur_mu), jnp.array(cur_r), sv, jnp.array(cur_nu),
                    ))
                    for c in range(nc)
                )
                for sv in s_grid
            ])
            scores_s = scores_s - jnp.max(scores_s)
            new_s_val = s_grid[jax.random.categorical(k1, scores_s)]
            new_s = new_s.at[j].set(new_s_val)

            # Sample mu
            mu_grid = data_mean + data_std * jnp.linspace(-2, 2, 11)
            scores_mu = jnp.array([
                sum(
                    float(_ng_log_marginal(
                        packed.ss_counts[v_idx, c, local_idx],
                        packed.ss_sum_x[v_idx, c, local_idx],
                        packed.ss_sum_x_sq[v_idx, c, local_idx],
                        mv, jnp.array(cur_r), new_s_val, jnp.array(cur_nu),
                    ))
                    for c in range(nc)
                )
                for mv in mu_grid
            ])
            scores_mu = scores_mu - jnp.max(scores_mu)
            new_mu_val = mu_grid[jax.random.categorical(k2, scores_mu)]
            new_mu = new_mu.at[j].set(new_mu_val)

            # Sample nu
            nu_grid = jnp.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
            scores_nu = jnp.array([
                sum(
                    float(_ng_log_marginal(
                        packed.ss_counts[v_idx, c, local_idx],
                        packed.ss_sum_x[v_idx, c, local_idx],
                        packed.ss_sum_x_sq[v_idx, c, local_idx],
                        new_mu_val, jnp.array(cur_r), new_s_val, nv,
                    ))
                    for c in range(nc)
                )
                for nv in nu_grid
            ])
            scores_nu = scores_nu - jnp.max(scores_nu)
            new_nu = new_nu.at[j].set(nu_grid[jax.random.categorical(k3, scores_nu)])

        elif type_id == CATEGORICAL_ID:
            alpha_grid = jnp.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
            scores = jnp.array([
                sum(
                    float(_dc_log_marginal(
                        packed.ss_counts[v_idx, c, local_idx],
                        packed.ss_cat_counts[v_idx, c, local_idx],
                        av,
                    ))
                    for c in range(nc)
                )
                for av in alpha_grid
            ])
            scores = scores - jnp.max(scores)
            new_dir_alpha = new_dir_alpha.at[j].set(
                alpha_grid[jax.random.categorical(keys[j], scores)]
            )

        elif type_id == BINARY_ID:
            ab_grid = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
            scores = []
            for a_val in ab_grid:
                for b_val in ab_grid:
                    s = sum(
                        float(_bb_log_marginal(
                            packed.ss_counts[v_idx, c, local_idx],
                            packed.ss_sum_x[v_idx, c, local_idx],
                            a_val, b_val,
                        ))
                        for c in range(nc)
                    )
                    scores.append(s)
            scores = jnp.array(scores)
            scores = scores - jnp.max(scores)
            idx = int(jax.random.categorical(keys[j], scores))
            a_idx, b_idx = divmod(idx, len(ab_grid))
            new_alpha = new_alpha.at[j].set(ab_grid[a_idx])
            new_beta = new_beta.at[j].set(ab_grid[b_idx])

        elif type_id == CYCLIC_ID:
            kappa_grid = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
            scores = jnp.array([
                sum(
                    float(_vm_log_marginal(
                        packed.ss_counts[v_idx, c, local_idx],
                        packed.ss_sum_sin[v_idx, c, local_idx],
                        packed.ss_sum_cos[v_idx, c, local_idx],
                        kv,
                    ))
                    for c in range(nc)
                )
                for kv in kappa_grid
            ])
            scores = scores - jnp.max(scores)
            new_kappa = new_kappa.at[j].set(
                kappa_grid[jax.random.categorical(keys[j], scores)]
            )

    return PackedCrossCatState(
        **{name: (new_mu if name == "hyper_mu"
                  else new_r if name == "hyper_r"
                  else new_s if name == "hyper_s"
                  else new_nu if name == "hyper_nu"
                  else new_dir_alpha if name == "hyper_dirichlet_alpha"
                  else new_alpha if name == "hyper_alpha"
                  else new_beta if name == "hyper_beta"
                  else new_kappa if name == "hyper_kappa"
                  else getattr(packed, name))
           for name in _ARRAY_FIELDS},
        **{name: getattr(packed, name) for name in _STATIC_FIELDS},
    )


# ---------------------------------------------------------------------------
# Vectorized CRP alpha sampling
# ---------------------------------------------------------------------------


def _log_crp_packed(assignments: Array, alpha: Array, n: int) -> Array:
    """Log CRP probability using packed arrays."""
    n_clusters = jnp.max(assignments) + 1
    counts = jnp.bincount(assignments, length=n).astype(jnp.float32)
    # Only count non-empty clusters
    valid_counts = jnp.where(counts > 0, counts, 1.0)
    log_p = (
        n_clusters * jnp.log(alpha)
        + jnp.sum(jnp.where(counts > 0, gammaln(valid_counts), 0.0))
        - gammaln(n + alpha)
        + gammaln(alpha)
    )
    return log_p


def packed_transition_crp_alphas(
    rng_key: Array,
    packed: PackedCrossCatState,
) -> PackedCrossCatState:
    """Sample CRP concentration parameters using packed state."""
    alpha_grid = jnp.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    n_views = int(packed.n_views)
    keys = jax.random.split(rng_key, 1 + n_views)

    # Outer CRP alpha
    from crosscat.model import _log_crp

    log_scores = jnp.array([
        float(_log_crp(packed.column_assignments, av)) - float(av)
        for av in alpha_grid
    ])
    log_scores = log_scores - jnp.max(log_scores)
    new_col_alpha = alpha_grid[jax.random.categorical(keys[0], log_scores)]

    # Inner CRP alphas
    new_view_alpha = jnp.array(packed.view_row_crp_alpha)
    for v in range(n_views):
        assigns = packed.view_row_assignments[v]
        log_scores = jnp.array([
            float(_log_crp(assigns, av)) - float(av)
            for av in alpha_grid
        ])
        log_scores = log_scores - jnp.max(log_scores)
        new_view_alpha = new_view_alpha.at[v].set(
            alpha_grid[jax.random.categorical(keys[v + 1], log_scores)]
        )

    return PackedCrossCatState(
        **{name: (new_col_alpha if name == "column_crp_alpha"
                  else new_view_alpha if name == "view_row_crp_alpha"
                  else getattr(packed, name))
           for name in _ARRAY_FIELDS},
        **{name: getattr(packed, name) for name in _STATIC_FIELDS},
    )


# ---------------------------------------------------------------------------
# Packed Gibbs sweep
# ---------------------------------------------------------------------------


def packed_gibbs_sweep(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    *,
    n_sweeps: int = 1,
    kernels: tuple[str, ...] = ("row_assignments", "column_hypers", "crp_alphas"),
) -> PackedCrossCatState:
    """Run Gibbs sweeps on packed state.

    Note: column_assignments kernel is not yet packed — use the unpacked
    path for that via the wrapper in gibbs.py.
    """
    kernel_map = {
        "row_assignments": lambda k, p, d: packed_transition_row_assignments(k, p, d),
        "column_hypers": lambda k, p, d: packed_transition_column_hypers(k, p, d),
        "crp_alphas": lambda k, p, _d: packed_transition_crp_alphas(k, p),
    }

    for _sweep in range(n_sweeps):
        for kernel_name in kernels:
            if kernel_name not in kernel_map:
                continue
            rng_key, subkey = jax.random.split(rng_key)
            packed = kernel_map[kernel_name](subkey, packed, data)

    return packed
