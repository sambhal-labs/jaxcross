"""Padded, JIT-compatible state representation for CrossCat.

Provides PackedCrossCatState and pack_state() / unpack_state() conversions.
All data structures use fixed-size padded arrays, enabling jax.jit, jax.vmap,
and jax.lax.scan without Python-level branching.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

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
    "hyper_vm_a",
    "hyper_vm_mu",
    "hyper_cutpoints",
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
_STATIC_FIELDS = (
    "n_rows",
    "n_cols",
    "max_views",
    "max_clusters",
    "max_categories",
    "max_cols_per_view",
)


@jax.tree_util.register_pytree_node_class
@dataclass
class PackedCrossCatState:
    """JIT-compatible CrossCat state with padded fixed-size arrays.

    All view data has a leading (max_views,) dimension.
    All cluster data has (max_views, max_clusters) dimensions.
    Invalid entries are masked via view_mask, n_views, n_clusters.
    """

    # Column partition
    column_assignments: Array  # (n_cols,) int
    column_crp_alpha: Array  # scalar
    n_views: Array  # scalar int
    view_mask: Array  # (max_views,) bool

    # Column type and hyperparameters — flat (n_cols,) arrays
    col_type_ids: Array  # (n_cols,) int
    hyper_mu: Array  # (n_cols,)
    hyper_r: Array  # (n_cols,)
    hyper_s: Array  # (n_cols,)
    hyper_nu: Array  # (n_cols,)
    hyper_dirichlet_alpha: Array  # (n_cols,)
    hyper_alpha: Array  # (n_cols,)
    hyper_beta: Array  # (n_cols,)
    hyper_kappa: Array  # (n_cols,)
    hyper_vm_a: Array  # (n_cols,) — prior concentration on mean direction
    hyper_vm_mu: Array  # (n_cols,) — prior mean direction (b)
    hyper_cutpoints: Array  # (n_cols, max_categories - 1) — ordinal cutpoints, +inf=pad

    # View data — leading (max_views,) dimension
    view_column_indices: Array  # (max_views, max_cols_per_view) int, -1=invalid
    view_n_columns: Array  # (max_views,) int
    view_row_assignments: Array  # (max_views, n_rows) int
    view_n_clusters: Array  # (max_views,) int
    view_row_crp_alpha: Array  # (max_views,)

    # Sufficient statistics — (max_views, max_clusters, max_cols_per_view[, max_cats])
    ss_counts: Array  # int
    ss_sum_x: Array
    ss_sum_x_sq: Array
    ss_cat_counts: Array  # (max_views, max_clusters, max_cols_per_view, max_cats)
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
    max_cols_per_view: int | None = None,
) -> PackedCrossCatState:
    """Convert a CrossCatState (Python lists) into a PackedCrossCatState (padded arrays).

    Args:
        state: Original CrossCat state.
        max_views: Maximum number of views (padding dimension).
        max_clusters: Maximum clusters per view.
        max_categories: Maximum categories for categorical/ordinal columns.
        max_cols_per_view: Maximum columns per view. Defaults to ``n_cols``
            (safe for any column assignment). For large datasets (>100 columns),
            setting this to a smaller value (e.g., ``max(32, n_cols // max_views)``)
            reduces memory by up to 10x and speeds up inner scans, but columns
            will be silently dropped if a view exceeds this limit during inference.

    Returns:
        Padded, JIT-compatible state.

    Raises:
        ValueError: If state dimensions exceed max_* limits.
    """
    n_rows = state.n_rows
    n_cols = state.n_cols
    n_views = state.n_views
    if n_views > max_views:
        raise ValueError(
            f"State has {n_views} views but max_views={max_views}. "
            f"Increase max_views to at least {n_views}."
        )
    for v_idx, view in enumerate(state.views):
        n_clusters = len(view.suffstats)
        if n_clusters > max_clusters:
            raise ValueError(
                f"View {v_idx} has {n_clusters} clusters but max_clusters={max_clusters}. "
                f"Increase max_clusters to at least {n_clusters}."
            )
    # Default: n_cols (safe for any column assignment — worst case all columns
    # merge into a single view). For large datasets, users can override.
    if max_cols_per_view is None:
        max_cols_per_view = n_cols
    for v_idx, view in enumerate(state.views):
        n_view_cols = len(view.column_indices)
        if n_view_cols > max_cols_per_view:
            raise ValueError(
                f"View {v_idx} has {n_view_cols} columns but max_cols_per_view="
                f"{max_cols_per_view}. Increase max_cols_per_view to at least "
                f"{n_view_cols}."
            )

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
    hyper_vm_a = jnp.ones(n_cols)
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
        if h.vm_a is not None:
            hyper_vm_a = hyper_vm_a.at[j].set(float(h.vm_a))
        if h.vm_mu is not None:
            hyper_vm_mu = hyper_vm_mu.at[j].set(float(h.vm_mu))

    # Pack ordinal cutpoints — padded with +inf (sigmoid(+inf - μ) = 1 → prob 0)
    hyper_cutpoints = jnp.full((n_cols, max_categories - 1), jnp.inf)
    for j, h in enumerate(state.column_hypers):
        if h.cutpoints is not None:
            n_cp = len(h.cutpoints)
            hyper_cutpoints = hyper_cutpoints.at[j, :n_cp].set(jnp.array(h.cutpoints))

    # Pack view data
    view_column_indices = jnp.full((max_views, max_cols_per_view), -1, dtype=jnp.int32)
    view_n_columns = jnp.zeros(max_views, dtype=jnp.int32)
    view_row_assignments = jnp.zeros((max_views, n_rows), dtype=jnp.int32)
    view_n_clusters = jnp.zeros(max_views, dtype=jnp.int32)
    view_row_crp_alpha = jnp.ones(max_views)

    ss_counts = jnp.zeros((max_views, max_clusters, max_cols_per_view), dtype=jnp.int32)
    ss_sum_x = jnp.zeros((max_views, max_clusters, max_cols_per_view))
    ss_sum_x_sq = jnp.zeros((max_views, max_clusters, max_cols_per_view))
    ss_cat_counts = jnp.zeros((max_views, max_clusters, max_cols_per_view, max_categories))
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
                        ss_sum_x_sq = ss_sum_x_sq.at[v_idx, c_idx, l_idx].set(float(ss.sum_x_sq))
                    if ss.category_counts is not None:
                        nc_cats = len(ss.category_counts)
                        if nc_cats > max_categories:
                            raise ValueError(
                                f"Column has {nc_cats} categories but "
                                f"max_categories={max_categories}. "
                                f"Increase max_categories to at least {nc_cats}."
                            )
                        ss_cat_counts = ss_cat_counts.at[v_idx, c_idx, l_idx, :nc_cats].set(
                            ss.category_counts[:nc_cats]
                        )
                    if ss.sum_sin is not None:
                        ss_sum_sin = ss_sum_sin.at[v_idx, c_idx, l_idx].set(float(ss.sum_sin))
                    if ss.sum_cos is not None:
                        ss_sum_cos = ss_sum_cos.at[v_idx, c_idx, l_idx].set(float(ss.sum_cos))

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
        hyper_vm_a=hyper_vm_a,
        hyper_vm_mu=hyper_vm_mu,
        hyper_cutpoints=hyper_cutpoints,
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
    data: Array | None = None,
) -> CrossCatState:
    """Convert a PackedCrossCatState back into a CrossCatState.

    Args:
        packed: Packed state.
        column_types: Column type list (not stored in packed state).
        data: Optional data matrix. When provided, sufficient statistics are
            recomputed from the data and row assignments instead of being
            unpacked from the padded arrays. This eliminates floating-point
            precision loss from the pack/unpack roundtrip.

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
            # Determine actual level count from data (padded cutpoints may
            # have been sampled to finite values during Gibbs sweeps)
            raw_cp = packed.hyper_cutpoints[j]
            if data is not None:
                col_data = data[:, j]
                valid = col_data[~jnp.isnan(col_data)]
                n_levels = int(jnp.max(valid)) + 1 if valid.size > 0 else 2
            else:
                n_levels = int(jnp.sum(jnp.isfinite(raw_cp))) + 1
                n_levels = max(n_levels, 2)
            trimmed_cp = raw_cp[: n_levels - 1]
            h = ColumnHypers(
                column_type=ct,
                mu=packed.hyper_mu[j],
                s=packed.hyper_s[j],
                cutpoints=trimmed_cp,
            )
        elif ct == ColumnType.CYCLIC:
            h = ColumnHypers(
                column_type=ct,
                kappa=packed.hyper_kappa[j],
                vm_a=packed.hyper_vm_a[j],
                vm_mu=packed.hyper_vm_mu[j],
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

    state = CrossCatState(
        column_assignments=packed.column_assignments,
        column_crp_alpha=packed.column_crp_alpha,
        column_hypers=col_hypers,
        column_types=column_types,
        views=views,
        n_rows=n_rows,
        n_cols=n_cols,
    )

    # Recompute suffstats from data if provided (eliminates precision loss)
    if data is not None:
        from crosscat.model import _compute_suffstats_for_view

        for view in state.views:
            n_clusters = len(view.suffstats)
            view.suffstats = _compute_suffstats_for_view(
                data, view.column_indices, column_types, view.row_assignments, n_clusters
            )

    return state


# ---------------------------------------------------------------------------
# Batching / unbatching utilities for multi-chain inference
# ---------------------------------------------------------------------------


def batch_packed_states(packed_list: list[PackedCrossCatState]) -> PackedCrossCatState:
    """Stack N packed states into a single batched pytree.

    Each array field gets a leading (n_chains,) dimension.
    All states must have identical static fields.
    """
    # Assert all static fields match
    ref = packed_list[0]
    for p in packed_list[1:]:
        for name in _STATIC_FIELDS:
            assert getattr(p, name) == getattr(ref, name), (
                f"Static field {name} differs: {getattr(ref, name)} vs {getattr(p, name)}"
            )

    # Stack array fields
    kwargs = {}
    for name in _ARRAY_FIELDS:
        kwargs[name] = jnp.stack([getattr(p, name) for p in packed_list])
    for name in _STATIC_FIELDS:
        kwargs[name] = getattr(ref, name)
    return PackedCrossCatState(**kwargs)


def unbatch_packed_states(
    batched: PackedCrossCatState, n_chains: int
) -> list[PackedCrossCatState]:
    """Unstack a batched pytree into N individual packed states."""
    result = []
    for i in range(n_chains):
        kwargs = {}
        for name in _ARRAY_FIELDS:
            kwargs[name] = getattr(batched, name)[i]
        for name in _STATIC_FIELDS:
            kwargs[name] = getattr(batched, name)
        result.append(PackedCrossCatState(**kwargs))
    return result


def select_best_chain(batched: PackedCrossCatState, scores: Array) -> PackedCrossCatState:
    """Select the chain with the highest score from a batched state."""
    idx = jnp.argmax(scores)
    kwargs = {}
    for name in _ARRAY_FIELDS:
        kwargs[name] = getattr(batched, name)[idx]
    for name in _STATIC_FIELDS:
        kwargs[name] = getattr(batched, name)
    return PackedCrossCatState(**kwargs)
