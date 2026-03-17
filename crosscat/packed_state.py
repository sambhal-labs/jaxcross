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

    init = (packed.ss_counts, packed.ss_sum_x, packed.ss_sum_x_sq,
            packed.ss_cat_counts, packed.ss_sum_sin, packed.ss_sum_cos)
    (ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos), _ = (
        jax.lax.scan(recompute_one_view, init, jnp.arange(max_views))
    )

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


# ---------------------------------------------------------------------------
# Vectorized v2 row scoring (lax.scan over columns, vmap over clusters)
# ---------------------------------------------------------------------------


def _score_row_one_cluster_v2(
    row_data: Array,
    col_indices: Array,
    col_type_ids: Array,
    ss_counts_c: Array,
    ss_sum_x_c: Array,
    ss_sum_x_sq_c: Array,
    ss_cat_counts_c: Array,
    ss_sum_sin_c: Array,
    ss_sum_cos_c: Array,
    hyper_mu: Array,
    hyper_r: Array,
    hyper_s: Array,
    hyper_nu: Array,
    hyper_dir_alpha: Array,
    hyper_alpha: Array,
    hyper_beta: Array,
    hyper_kappa: Array,
    hyper_vm_mu: Array,
    n_columns: Array,
) -> Array:
    """Score a row against ONE cluster using lax.scan over columns.

    Args:
        row_data: (n_cols_total,) full row from data matrix.
        col_indices: (max_cols_per_view,) column indices, -1 for padding.
        col_type_ids: (n_cols_total,) type ID per column.
        ss_counts_c: (max_cols_per_view,) counts for this cluster.
        ss_sum_x_c, ss_sum_x_sq_c: (max_cols_per_view,) sum stats.
        ss_cat_counts_c: (max_cols_per_view, max_categories) category counts.
        ss_sum_sin_c, ss_sum_cos_c: (max_cols_per_view,) cyclic stats.
        hyper_*: (n_cols_total,) hyperparameters indexed by global column.
        n_columns: traced scalar — number of valid columns in this view.

    Returns:
        Scalar log likelihood of row under this cluster.
    """
    n_total_cols = row_data.shape[0]

    def scan_body(log_lik, li):
        col_idx = col_indices[li]
        safe_col_idx = jnp.clip(col_idx, 0, n_total_cols - 1)
        x = row_data[safe_col_idx]
        type_id = col_type_ids[safe_col_idx]

        # Valid if: not padding (-1), within active columns, and not NaN
        is_valid = (col_idx >= 0) & (li < n_columns) & (~jnp.isnan(x))

        logp = unified_posterior_predictive_logp(
            x, type_id,
            ss_counts_c[li].astype(jnp.float32),
            ss_sum_x_c[li], ss_sum_x_sq_c[li],
            ss_cat_counts_c[li],
            ss_sum_sin_c[li], ss_sum_cos_c[li],
            hyper_mu[safe_col_idx], hyper_r[safe_col_idx],
            hyper_s[safe_col_idx], hyper_nu[safe_col_idx],
            hyper_dir_alpha[safe_col_idx],
            hyper_alpha[safe_col_idx], hyper_beta[safe_col_idx],
            hyper_kappa[safe_col_idx], hyper_vm_mu[safe_col_idx],
        )
        log_lik = log_lik + jnp.where(is_valid, logp, 0.0)
        return log_lik, None

    max_cols_per_view = col_indices.shape[0]
    log_lik, _ = jax.lax.scan(scan_body, jnp.array(0.0), jnp.arange(max_cols_per_view))
    return log_lik


def _score_row_all_clusters_v2(
    row_data: Array,
    col_indices: Array,
    n_columns: Array,
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
    hyper_dir_alpha: Array,
    hyper_alpha: Array,
    hyper_beta: Array,
    hyper_kappa: Array,
    hyper_vm_mu: Array,
    crp_alpha: Array,
    max_clusters: int,
) -> Array:
    """Score a row against ALL clusters (existing + one new) using vmap.

    Uses vmap over the cluster axis instead of a Python for-loop.

    Args:
        row_data: (n_cols_total,) full row from data matrix.
        col_indices: (max_cols_per_view,) column indices, -1 for padding.
        n_columns: traced scalar — number of valid columns in this view.
        col_type_ids: (n_cols_total,) type ID per column.
        cluster_counts: (max_clusters,) counts per cluster.
        ss_counts: (max_clusters, max_cols_per_view) int.
        ss_sum_x, ss_sum_x_sq: (max_clusters, max_cols_per_view).
        ss_cat_counts: (max_clusters, max_cols_per_view, max_categories).
        ss_sum_sin, ss_sum_cos: (max_clusters, max_cols_per_view).
        hyper_*: (n_cols_total,) hyperparameters.
        crp_alpha: scalar CRP concentration.
        max_clusters: int (static) — padding size.

    Returns:
        (max_clusters + 1,) array of log probabilities.
    """
    # CRP prior: log(count_c) for existing clusters, -inf for empty
    log_prior = jnp.log(jnp.maximum(cluster_counts.astype(jnp.float32), 1e-30))
    log_prior = jnp.where(cluster_counts > 0, log_prior, -jnp.inf)

    # vmap _score_row_one_cluster_v2 over the cluster dimension (axis 0 of ss_*)
    def score_one(ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos):
        return _score_row_one_cluster_v2(
            row_data, col_indices, col_type_ids,
            ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos,
            hyper_mu, hyper_r, hyper_s, hyper_nu,
            hyper_dir_alpha, hyper_alpha, hyper_beta,
            hyper_kappa, hyper_vm_mu,
            n_columns,
        )

    log_liks = jax.vmap(score_one)(
        ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos
    )
    log_probs_existing = log_prior + log_liks

    # New cluster: empty suffstats -> prior predictive
    max_cols_per_view = col_indices.shape[0]
    max_cats = ss_cat_counts.shape[-1]
    log_lik_new = _score_row_one_cluster_v2(
        row_data, col_indices, col_type_ids,
        jnp.zeros(max_cols_per_view, dtype=jnp.int32),
        jnp.zeros(max_cols_per_view),
        jnp.zeros(max_cols_per_view),
        jnp.zeros((max_cols_per_view, max_cats)),
        jnp.zeros(max_cols_per_view),
        jnp.zeros(max_cols_per_view),
        hyper_mu, hyper_r, hyper_s, hyper_nu,
        hyper_dir_alpha, hyper_alpha, hyper_beta,
        hyper_kappa, hyper_vm_mu,
        n_columns,
    )
    log_prior_new = jnp.log(crp_alpha)
    log_prob_new = log_prior_new + log_lik_new

    return jnp.concatenate([log_probs_existing, jnp.array([log_prob_new])])


def _compact_clusters(
    assignments: Array,
    n_rows: int,
    max_clusters: int,
) -> tuple[Array, Array]:
    """Remap cluster IDs to contiguous 0..K-1 range (JIT-compatible).

    Uses bincount + cumsum instead of jnp.unique (which is not JIT-friendly).

    Args:
        assignments: (n_rows,) int — current cluster assignments.
        n_rows: number of rows (Python int or static).
        max_clusters: padding size (Python int, static).

    Returns:
        (new_assignments, new_n_clusters) where new_n_clusters is a scalar array.
    """
    # Count members per cluster: shape (max_clusters,)
    counts = jnp.bincount(assignments, length=max_clusters)
    occupied = (counts > 0).astype(jnp.int32)  # 1 if cluster has members
    # cumsum gives new IDs: occupied clusters get 1,2,3,... and we subtract 1
    # so first occupied cluster -> 0, second -> 1, etc.
    new_ids = jnp.cumsum(occupied) - 1  # (max_clusters,) — new ID for each old ID
    # For unoccupied clusters, new_ids will have gaps but we never index into them
    new_assignments = new_ids[assignments]
    new_n_clusters = jnp.sum(occupied)
    return new_assignments, new_n_clusters


def packed_transition_row_assignments_v2(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
) -> PackedCrossCatState:
    """Gibbs sweep over row assignments using lax.scan (JIT-compatible).

    Outer lax.scan over views, inner lax.scan over rows. Scores all clusters
    via _score_row_all_clusters_v2 (vmap + lax.scan). After all rows in a view,
    compacts cluster IDs. After all views, recomputes suffstats from scratch.

    Args:
        rng_key: PRNG key.
        packed: Current packed state.
        data: (n_rows, n_cols) data matrix.

    Returns:
        Updated PackedCrossCatState with new row assignments and suffstats.
    """
    n_rows = packed.n_rows
    max_c = packed.max_clusters
    max_views = packed.max_views
    max_cats = packed.max_categories

    # Pre-split keys for all views
    view_keys = jax.random.split(rng_key, max_views)

    def scan_one_view(carry, v_idx):
        """Process one view: inner scan over rows."""
        ra_all, nc_all = carry
        view_key = view_keys[v_idx]
        row_keys = jax.random.split(view_key, n_rows)

        is_active = packed.view_mask[v_idx]
        col_indices = packed.view_column_indices[v_idx]  # (max_cols_per_view,)
        n_columns = packed.view_n_columns[v_idx]
        alpha = packed.view_row_crp_alpha[v_idx]

        # Working suffstats for this view (will be mutated row by row)
        w_ss_c = packed.ss_counts[v_idx]      # (max_c, max_cols_per_view)
        w_ss_sx = packed.ss_sum_x[v_idx]
        w_ss_sxsq = packed.ss_sum_x_sq[v_idx]
        w_ss_cat = packed.ss_cat_counts[v_idx]  # (max_c, max_cols_per_view, max_cats)
        w_ss_sin = packed.ss_sum_sin[v_idx]
        w_ss_cos = packed.ss_sum_cos[v_idx]

        assigns = ra_all[v_idx]  # (n_rows,)
        n_cl = nc_all[v_idx]     # scalar

        def scan_one_row(row_carry, row_idx):
            """Process one row within a view."""
            (r_assigns, r_ss_c, r_ss_sx, r_ss_sxsq, r_ss_cat,
             r_ss_sin, r_ss_cos, r_n_cl) = row_carry
            rk = row_keys[row_idx]
            row_data = data[row_idx]

            old_cluster = r_assigns[row_idx]

            # Remove row from old cluster's suffstats
            r_ss_c, r_ss_sx, r_ss_sxsq, r_ss_cat, r_ss_sin, r_ss_cos = (
                _remove_row_from_suffstats(
                    r_ss_c, r_ss_sx, r_ss_sxsq, r_ss_cat, r_ss_sin, r_ss_cos,
                    old_cluster, row_data, col_indices,
                    packed.col_type_ids, max_cats,
                )
            )

            # Cluster counts excluding this row
            # Temporarily mark this row as an invalid cluster to exclude it
            temp_assigns = r_assigns.at[row_idx].set(max_c)  # out of range
            counts = jnp.bincount(temp_assigns, length=max_c).astype(jnp.int32)

            # Score all clusters
            log_probs = _score_row_all_clusters_v2(
                row_data, col_indices, n_columns, packed.col_type_ids,
                counts,
                r_ss_c, r_ss_sx, r_ss_sxsq, r_ss_cat, r_ss_sin, r_ss_cos,
                packed.hyper_mu, packed.hyper_r, packed.hyper_s, packed.hyper_nu,
                packed.hyper_dirichlet_alpha, packed.hyper_alpha, packed.hyper_beta,
                packed.hyper_kappa, packed.hyper_vm_mu,
                alpha, max_c,
            )

            # If budget exhausted (n_cl >= max_c - 1), block new cluster
            budget_exhausted = r_n_cl >= (max_c - 1)
            log_probs = log_probs.at[max_c].set(
                jnp.where(budget_exhausted, -jnp.inf, log_probs[max_c])
            )

            # Normalize for numerical stability
            log_probs = log_probs - jnp.max(log_probs)

            # Sample
            chosen = jax.random.categorical(rk, log_probs)

            # If chosen == max_c (new cluster), find next free slot
            # Next free slot is the first cluster with count == 0
            # Use argmin on (counts > 0) to find first zero-count cluster
            free_slot = jnp.argmin(counts)  # first cluster with count 0
            # If all occupied, clamp to max_c-1 (budget_exhausted should prevent this)
            is_new = chosen == max_c
            actual_cluster = jnp.where(is_new, free_slot, chosen).astype(jnp.int32)

            # Update n_clusters: if new cluster was chosen, increment
            r_n_cl = jnp.where(is_new, r_n_cl + 1, r_n_cl)

            # Update assignment
            r_assigns = r_assigns.at[row_idx].set(actual_cluster)

            # Add row to chosen cluster's suffstats
            r_ss_c, r_ss_sx, r_ss_sxsq, r_ss_cat, r_ss_sin, r_ss_cos = (
                _add_row_to_suffstats(
                    r_ss_c, r_ss_sx, r_ss_sxsq, r_ss_cat, r_ss_sin, r_ss_cos,
                    actual_cluster, row_data, col_indices,
                    packed.col_type_ids, max_cats,
                )
            )

            new_carry = (r_assigns, r_ss_c, r_ss_sx, r_ss_sxsq, r_ss_cat,
                         r_ss_sin, r_ss_cos, r_n_cl)
            return new_carry, None

        # Run inner scan over rows
        row_init = (assigns, w_ss_c, w_ss_sx, w_ss_sxsq, w_ss_cat,
                    w_ss_sin, w_ss_cos, n_cl)
        (final_assigns, _, _, _, _, _, _, final_n_cl), _ = jax.lax.scan(
            scan_one_row, row_init, jnp.arange(n_rows)
        )

        # Compact cluster IDs
        compacted_assigns, compacted_n_cl = _compact_clusters(
            final_assigns, n_rows, max_c
        )

        # Only update if view is active
        new_ra = jnp.where(is_active, compacted_assigns, ra_all[v_idx])
        new_nc = jnp.where(is_active, compacted_n_cl, nc_all[v_idx])

        ra_all = ra_all.at[v_idx].set(new_ra)
        nc_all = nc_all.at[v_idx].set(new_nc)

        return (ra_all, nc_all), None

    # Outer scan over views
    init_carry = (jnp.array(packed.view_row_assignments),
                  jnp.array(packed.view_n_clusters))
    (new_row_assigns, new_n_clusters), _ = jax.lax.scan(
        scan_one_view, init_carry, jnp.arange(max_views)
    )

    # Create updated packed state with new assignments
    updated = PackedCrossCatState(
        **{name: (new_row_assigns if name == "view_row_assignments"
                  else new_n_clusters if name == "view_n_clusters"
                  else getattr(packed, name))
           for name in _ARRAY_FIELDS},
        **{name: getattr(packed, name) for name in _STATIC_FIELDS},
    )
    return recompute_all_suffstats(updated, data)


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
# Vectorized column hypers sampling (v2 — JIT-compatible via vmap)
# ---------------------------------------------------------------------------


def packed_transition_column_hypers_v2(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
) -> PackedCrossCatState:
    """Grid-based Gibbs sampling for column hyperparameters, vmapped over columns.

    For each column, scores a grid of hyperparameter values across all clusters
    in the column's assigned view. Uses jnp.where on type_id to select which
    type-specific results to keep.

    Fully JIT-compatible: no Python loops over columns, views, or clusters.
    """
    n_cols = packed.n_cols
    max_c = packed.max_clusters
    max_views = packed.max_views
    max_cpv = packed.max_cols_per_view

    # Pre-split keys: one per column
    col_keys = jax.random.split(rng_key, n_cols)

    def _find_local_index(v_idx, col_j):
        """Find local column index within a view using lax.scan."""
        def scan_fn(found_idx, li):
            matches = packed.view_column_indices[v_idx, li] == col_j
            new_idx = jnp.where(matches & (found_idx < 0), li, found_idx)
            return new_idx, None
        local_idx, _ = jax.lax.scan(scan_fn, jnp.array(-1, dtype=jnp.int32),
                                     jnp.arange(max_cpv))
        # Clamp to 0 if not found (should not happen for valid columns)
        return jnp.maximum(local_idx, 0)

    def _score_grid_ng(v_idx, local_idx, mu_val, r_val, s_grid_vals, nu_val):
        """Score a grid of s values for Normal-Gamma. Returns (n_grid,) scores."""
        nc = packed.view_n_clusters[v_idx]
        counts_col = packed.ss_counts[v_idx, :, local_idx]     # (max_c,)
        sum_x_col = packed.ss_sum_x[v_idx, :, local_idx]       # (max_c,)
        sum_x_sq_col = packed.ss_sum_x_sq[v_idx, :, local_idx] # (max_c,)

        def score_one_grid_point(s_val):
            # Score across all clusters, mask inactive
            per_cluster = _ng_log_marginal(
                counts_col, sum_x_col, sum_x_sq_col,
                mu_val, r_val, s_val, nu_val,
            )  # (max_c,)
            masked = jnp.where(jnp.arange(max_c) < nc, per_cluster, 0.0)
            return jnp.sum(masked)

        return jax.vmap(score_one_grid_point)(s_grid_vals)

    def _score_grid_ng_mu(v_idx, local_idx, mu_grid_vals, r_val, s_val, nu_val):
        """Score a grid of mu values for Normal-Gamma."""
        nc = packed.view_n_clusters[v_idx]
        counts_col = packed.ss_counts[v_idx, :, local_idx]
        sum_x_col = packed.ss_sum_x[v_idx, :, local_idx]
        sum_x_sq_col = packed.ss_sum_x_sq[v_idx, :, local_idx]

        def score_one_grid_point(mu_val):
            per_cluster = _ng_log_marginal(
                counts_col, sum_x_col, sum_x_sq_col,
                mu_val, r_val, s_val, nu_val,
            )
            masked = jnp.where(jnp.arange(max_c) < nc, per_cluster, 0.0)
            return jnp.sum(masked)

        return jax.vmap(score_one_grid_point)(mu_grid_vals)

    def _score_grid_ng_nu(v_idx, local_idx, mu_val, r_val, s_val, nu_grid_vals):
        """Score a grid of nu values for Normal-Gamma."""
        nc = packed.view_n_clusters[v_idx]
        counts_col = packed.ss_counts[v_idx, :, local_idx]
        sum_x_col = packed.ss_sum_x[v_idx, :, local_idx]
        sum_x_sq_col = packed.ss_sum_x_sq[v_idx, :, local_idx]

        def score_one_grid_point(nu_val):
            per_cluster = _ng_log_marginal(
                counts_col, sum_x_col, sum_x_sq_col,
                mu_val, r_val, s_val, nu_val,
            )
            masked = jnp.where(jnp.arange(max_c) < nc, per_cluster, 0.0)
            return jnp.sum(masked)

        return jax.vmap(score_one_grid_point)(nu_grid_vals)

    def process_one_column(j):
        """Process column j: sample hypers based on type. Returns updated hyper values."""
        key = col_keys[j]
        type_id = packed.col_type_ids[j]
        v_idx = packed.column_assignments[j]
        local_idx = _find_local_index(v_idx, j)

        k1, k2, k3, k4 = jax.random.split(key, 4)

        # --- Continuous: sample s, then mu, then nu ---
        cur_mu = packed.hyper_mu[j]
        cur_r = packed.hyper_r[j]
        cur_nu = packed.hyper_nu[j]

        # Data statistics for grid construction
        col_data = data[:, j]
        valid_mask = ~jnp.isnan(col_data)
        n_valid = jnp.sum(valid_mask).astype(jnp.float32)
        safe_n = jnp.maximum(n_valid, 1.0)
        data_mean = jnp.sum(jnp.where(valid_mask, col_data, 0.0)) / safe_n
        data_var = jnp.sum(jnp.where(valid_mask, (col_data - data_mean) ** 2, 0.0)) / safe_n
        data_var = data_var + 1e-6
        data_std = jnp.sqrt(data_var)

        # Sample s
        s_grid = data_var * jnp.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0])
        s_scores = _score_grid_ng(v_idx, local_idx, cur_mu, cur_r, s_grid, cur_nu)
        s_scores = s_scores - jnp.max(s_scores)
        new_s_val = s_grid[jax.random.categorical(k1, s_scores)]

        # Sample mu (conditioned on new s)
        mu_grid = data_mean + data_std * jnp.linspace(-2, 2, 11)
        mu_scores = _score_grid_ng_mu(v_idx, local_idx, mu_grid, cur_r, new_s_val, cur_nu)
        mu_scores = mu_scores - jnp.max(mu_scores)
        new_mu_val = mu_grid[jax.random.categorical(k2, mu_scores)]

        # Sample nu (conditioned on new s, new mu)
        nu_grid = jnp.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
        nu_scores = _score_grid_ng_nu(v_idx, local_idx, new_mu_val, cur_r, new_s_val, nu_grid)
        nu_scores = nu_scores - jnp.max(nu_scores)
        new_nu_val = nu_grid[jax.random.categorical(k3, nu_scores)]

        # --- Categorical: sample dirichlet_alpha ---
        nc = packed.view_n_clusters[v_idx]
        cat_alpha_grid = jnp.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
        counts_col_cat = packed.ss_counts[v_idx, :, local_idx]
        cat_counts_col = packed.ss_cat_counts[v_idx, :, local_idx]  # (max_c, max_cats)

        def score_cat_grid(alpha_val):
            per_cluster = _dc_log_marginal(counts_col_cat, cat_counts_col, alpha_val)
            masked = jnp.where(jnp.arange(max_c) < nc, per_cluster, 0.0)
            return jnp.sum(masked)

        cat_scores = jax.vmap(score_cat_grid)(cat_alpha_grid)
        cat_scores = cat_scores - jnp.max(cat_scores)
        new_dir_alpha_val = cat_alpha_grid[jax.random.categorical(k1, cat_scores)]

        # --- Binary: sample alpha, beta from 2D grid ---
        ab_grid = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
        sum_x_col_bb = packed.ss_sum_x[v_idx, :, local_idx]  # (max_c,)

        # Create 2D grid: all combinations
        a_grid_2d = jnp.repeat(ab_grid, ab_grid.shape[0])    # (25,)
        b_grid_2d = jnp.tile(ab_grid, ab_grid.shape[0])      # (25,)

        def score_bb_grid(ab_pair):
            a_val, b_val = ab_pair
            per_cluster = _bb_log_marginal(counts_col_cat, sum_x_col_bb, a_val, b_val)
            masked = jnp.where(jnp.arange(max_c) < nc, per_cluster, 0.0)
            return jnp.sum(masked)

        bb_scores = jax.vmap(score_bb_grid)(jnp.stack([a_grid_2d, b_grid_2d], axis=1))
        bb_scores = bb_scores - jnp.max(bb_scores)
        bb_idx = jax.random.categorical(k1, bb_scores)
        new_alpha_val = a_grid_2d[bb_idx]
        new_beta_val = b_grid_2d[bb_idx]

        # --- Cyclic: sample kappa ---
        kappa_grid = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
        sum_sin_col = packed.ss_sum_sin[v_idx, :, local_idx]  # (max_c,)
        sum_cos_col = packed.ss_sum_cos[v_idx, :, local_idx]  # (max_c,)

        def score_vm_grid(kappa_val):
            per_cluster = _vm_log_marginal(counts_col_cat, sum_sin_col, sum_cos_col, kappa_val)
            masked = jnp.where(jnp.arange(max_c) < nc, per_cluster, 0.0)
            return jnp.sum(masked)

        vm_scores = jax.vmap(score_vm_grid)(kappa_grid)
        vm_scores = vm_scores - jnp.max(vm_scores)
        new_kappa_val = kappa_grid[jax.random.categorical(k1, vm_scores)]

        # --- Select results based on type_id ---
        out_mu = jnp.where(type_id == CONTINUOUS_ID, new_mu_val, packed.hyper_mu[j])
        out_s = jnp.where(type_id == CONTINUOUS_ID, new_s_val, packed.hyper_s[j])
        out_nu = jnp.where(type_id == CONTINUOUS_ID, new_nu_val, packed.hyper_nu[j])
        out_dir_alpha = jnp.where(
            type_id == CATEGORICAL_ID, new_dir_alpha_val, packed.hyper_dirichlet_alpha[j]
        )
        out_alpha = jnp.where(type_id == BINARY_ID, new_alpha_val, packed.hyper_alpha[j])
        out_beta = jnp.where(type_id == BINARY_ID, new_beta_val, packed.hyper_beta[j])
        out_kappa = jnp.where(type_id == CYCLIC_ID, new_kappa_val, packed.hyper_kappa[j])

        return out_mu, packed.hyper_r[j], out_s, out_nu, out_dir_alpha, out_alpha, out_beta, out_kappa

    # vmap over all columns
    (new_mu, new_r, new_s, new_nu, new_dir_alpha,
     new_alpha, new_beta, new_kappa) = jax.vmap(process_one_column)(jnp.arange(n_cols))

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
# Vectorized CRP alpha sampling (v2 — JIT-compatible via vmap)
# ---------------------------------------------------------------------------


def packed_transition_crp_alphas_v2(
    rng_key: Array,
    packed: PackedCrossCatState,
) -> PackedCrossCatState:
    """Sample CRP concentration parameters using vmap (JIT-compatible).

    Scores a grid of alpha values for the outer (column) CRP and each inner
    (row) CRP. Includes Exp(1) prior: log_score -= alpha_val.
    """
    alpha_grid = jnp.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    n_grid = alpha_grid.shape[0]
    max_views = packed.max_views
    n_cols = packed.n_cols
    n_rows = packed.n_rows
    max_c = packed.max_clusters

    k_outer, k_inner = jax.random.split(rng_key)

    # --- Helper: log CRP score for a given assignments vector ---
    def log_crp_score(assignments, alpha_val, length):
        """CRP log probability + Exp(1) prior on alpha."""
        counts = jnp.bincount(assignments, length=length).astype(jnp.float32)
        n_clusters = jnp.sum(counts > 0).astype(jnp.float32)
        valid_counts = jnp.where(counts > 0, counts, 1.0)
        n_total = jnp.sum(counts).astype(jnp.float32)
        log_p = (
            n_clusters * jnp.log(alpha_val)
            + jnp.sum(jnp.where(counts > 0, gammaln(valid_counts), 0.0))
            - gammaln(n_total + alpha_val)
            + gammaln(alpha_val)
        )
        # Exp(1) prior: -alpha_val
        return log_p - alpha_val

    # --- Outer CRP alpha (column assignments) ---
    def score_outer_one(alpha_val):
        return log_crp_score(packed.column_assignments, alpha_val, n_cols)

    outer_scores = jax.vmap(score_outer_one)(alpha_grid)
    outer_scores = outer_scores - jnp.max(outer_scores)
    new_col_alpha = alpha_grid[jax.random.categorical(k_outer, outer_scores)]

    # --- Inner CRP alphas (row assignments per view) ---
    view_keys = jax.random.split(k_inner, max_views)

    def sample_one_view(v_idx):
        """Sample CRP alpha for one view."""
        assigns = packed.view_row_assignments[v_idx]  # (n_rows,)

        def score_inner_one(alpha_val):
            return log_crp_score(assigns, alpha_val, max_c)

        scores = jax.vmap(score_inner_one)(alpha_grid)
        scores = scores - jnp.max(scores)
        chosen = alpha_grid[jax.random.categorical(view_keys[v_idx], scores)]

        # Only update active views
        is_active = packed.view_mask[v_idx]
        return jnp.where(is_active, chosen, packed.view_row_crp_alpha[v_idx])

    new_view_alpha = jax.vmap(sample_one_view)(jnp.arange(max_views))

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
