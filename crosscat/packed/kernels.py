"""JIT-compatible Gibbs kernels for packed CrossCat state.

All functions use lax.scan and vmap for full JIT compatibility.
Extracted from packed_state.py (v2 kernels) with _v2 suffixes removed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln

from crosscat.packed.components import (
    _bb_log_marginal,
    _dc_log_marginal,
    _ng_log_marginal,
    _vm_log_marginal,
    unified_log_marginal,
    unified_posterior_predictive_logp,
)
from crosscat.packed.state import (
    _ARRAY_FIELDS,
    _STATIC_FIELDS,
    BINARY_ID,
    CATEGORICAL_ID,
    CONTINUOUS_ID,
    CYCLIC_ID,
    PackedCrossCatState,
)
from crosscat.packed.suffstats import (
    _add_row_to_suffstats,
    _remove_row_from_suffstats,
    recompute_all_suffstats,
)

# ---------------------------------------------------------------------------
# Vectorized row scoring (lax.scan over columns, vmap over clusters)
# ---------------------------------------------------------------------------


def _score_row_one_cluster(
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
            x,
            type_id,
            ss_counts_c[li].astype(jnp.float32),
            ss_sum_x_c[li],
            ss_sum_x_sq_c[li],
            ss_cat_counts_c[li],
            ss_sum_sin_c[li],
            ss_sum_cos_c[li],
            hyper_mu[safe_col_idx],
            hyper_r[safe_col_idx],
            hyper_s[safe_col_idx],
            hyper_nu[safe_col_idx],
            hyper_dir_alpha[safe_col_idx],
            hyper_alpha[safe_col_idx],
            hyper_beta[safe_col_idx],
            hyper_kappa[safe_col_idx],
            hyper_vm_mu[safe_col_idx],
        )
        log_lik = log_lik + jnp.where(is_valid, logp, 0.0)
        return log_lik, None

    max_cols_per_view = col_indices.shape[0]
    log_lik, _ = jax.lax.scan(scan_body, jnp.array(0.0), jnp.arange(max_cols_per_view))
    return log_lik


def _score_row_all_clusters(
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

    # vmap _score_row_one_cluster over the cluster dimension (axis 0 of ss_*)
    def score_one(ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos):
        return _score_row_one_cluster(
            row_data,
            col_indices,
            col_type_ids,
            ss_c,
            ss_sx,
            ss_sxsq,
            ss_cat,
            ss_sin,
            ss_cos,
            hyper_mu,
            hyper_r,
            hyper_s,
            hyper_nu,
            hyper_dir_alpha,
            hyper_alpha,
            hyper_beta,
            hyper_kappa,
            hyper_vm_mu,
            n_columns,
        )

    log_liks = jax.vmap(score_one)(
        ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos
    )
    log_probs_existing = log_prior + log_liks

    # New cluster: empty suffstats -> prior predictive
    max_cols_per_view = col_indices.shape[0]
    max_cats = ss_cat_counts.shape[-1]
    log_lik_new = _score_row_one_cluster(
        row_data,
        col_indices,
        col_type_ids,
        jnp.zeros(max_cols_per_view, dtype=jnp.int32),
        jnp.zeros(max_cols_per_view),
        jnp.zeros(max_cols_per_view),
        jnp.zeros((max_cols_per_view, max_cats)),
        jnp.zeros(max_cols_per_view),
        jnp.zeros(max_cols_per_view),
        hyper_mu,
        hyper_r,
        hyper_s,
        hyper_nu,
        hyper_dir_alpha,
        hyper_alpha,
        hyper_beta,
        hyper_kappa,
        hyper_vm_mu,
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


# ---------------------------------------------------------------------------
# Row assignment kernel
# ---------------------------------------------------------------------------


def packed_transition_row_assignments(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    *,
    recompute_suffstats: bool = True,
) -> PackedCrossCatState:
    """Gibbs sweep over row assignments using lax.scan (JIT-compatible).

    Outer lax.scan over views, inner lax.scan over rows. Scores all clusters
    via _score_row_all_clusters (vmap + lax.scan). After all rows in a view,
    compacts cluster IDs. After all views, recomputes suffstats from scratch.

    Args:
        rng_key: PRNG key.
        packed: Current packed state.
        data: (n_rows, n_cols) data matrix.
        recompute_suffstats: If True (default), recompute all sufficient
            statistics from scratch after the sweep. Set to False if a
            subsequent kernel (e.g., column assignments) will recompute anyway.

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
        w_ss_c = packed.ss_counts[v_idx]  # (max_c, max_cols_per_view)
        w_ss_sx = packed.ss_sum_x[v_idx]
        w_ss_sxsq = packed.ss_sum_x_sq[v_idx]
        w_ss_cat = packed.ss_cat_counts[v_idx]  # (max_c, max_cols_per_view, max_cats)
        w_ss_sin = packed.ss_sum_sin[v_idx]
        w_ss_cos = packed.ss_sum_cos[v_idx]

        assigns = ra_all[v_idx]  # (n_rows,)
        n_cl = nc_all[v_idx]  # scalar

        def scan_one_row(row_carry, row_idx):
            """Process one row within a view."""
            (r_assigns, r_ss_c, r_ss_sx, r_ss_sxsq, r_ss_cat, r_ss_sin, r_ss_cos, r_n_cl) = (
                row_carry
            )
            rk = row_keys[row_idx]
            row_data = data[row_idx]

            old_cluster = r_assigns[row_idx]

            # Remove row from old cluster's suffstats
            r_ss_c, r_ss_sx, r_ss_sxsq, r_ss_cat, r_ss_sin, r_ss_cos = _remove_row_from_suffstats(
                r_ss_c,
                r_ss_sx,
                r_ss_sxsq,
                r_ss_cat,
                r_ss_sin,
                r_ss_cos,
                old_cluster,
                row_data,
                col_indices,
                packed.col_type_ids,
                max_cats,
            )

            # Cluster counts excluding this row
            # Temporarily mark this row as an invalid cluster to exclude it
            temp_assigns = r_assigns.at[row_idx].set(max_c)  # out of range
            counts = jnp.bincount(temp_assigns, length=max_c).astype(jnp.int32)

            # Score all clusters
            log_probs = _score_row_all_clusters(
                row_data,
                col_indices,
                n_columns,
                packed.col_type_ids,
                counts,
                r_ss_c,
                r_ss_sx,
                r_ss_sxsq,
                r_ss_cat,
                r_ss_sin,
                r_ss_cos,
                packed.hyper_mu,
                packed.hyper_r,
                packed.hyper_s,
                packed.hyper_nu,
                packed.hyper_dirichlet_alpha,
                packed.hyper_alpha,
                packed.hyper_beta,
                packed.hyper_kappa,
                packed.hyper_vm_mu,
                alpha,
                max_c,
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
            r_ss_c, r_ss_sx, r_ss_sxsq, r_ss_cat, r_ss_sin, r_ss_cos = _add_row_to_suffstats(
                r_ss_c,
                r_ss_sx,
                r_ss_sxsq,
                r_ss_cat,
                r_ss_sin,
                r_ss_cos,
                actual_cluster,
                row_data,
                col_indices,
                packed.col_type_ids,
                max_cats,
            )

            new_carry = (
                r_assigns,
                r_ss_c,
                r_ss_sx,
                r_ss_sxsq,
                r_ss_cat,
                r_ss_sin,
                r_ss_cos,
                r_n_cl,
            )
            return new_carry, None

        # Run inner scan over rows
        row_init = (assigns, w_ss_c, w_ss_sx, w_ss_sxsq, w_ss_cat, w_ss_sin, w_ss_cos, n_cl)
        (final_assigns, _, _, _, _, _, _, final_n_cl), _ = jax.lax.scan(
            scan_one_row, row_init, jnp.arange(n_rows)
        )

        # Compact cluster IDs
        compacted_assigns, compacted_n_cl = _compact_clusters(final_assigns, n_rows, max_c)

        # Only update if view is active
        new_ra = jnp.where(is_active, compacted_assigns, ra_all[v_idx])
        new_nc = jnp.where(is_active, compacted_n_cl, nc_all[v_idx])

        ra_all = ra_all.at[v_idx].set(new_ra)
        nc_all = nc_all.at[v_idx].set(new_nc)

        return (ra_all, nc_all), None

    # Outer scan over views
    init_carry = (jnp.array(packed.view_row_assignments), jnp.array(packed.view_n_clusters))
    (new_row_assigns, new_n_clusters), _ = jax.lax.scan(
        scan_one_view, init_carry, jnp.arange(max_views)
    )

    # Create updated packed state with new assignments
    updated = PackedCrossCatState(
        **{
            name: (
                new_row_assigns
                if name == "view_row_assignments"
                else new_n_clusters
                if name == "view_n_clusters"
                else getattr(packed, name)
            )
            for name in _ARRAY_FIELDS
        },
        **{name: getattr(packed, name) for name in _STATIC_FIELDS},
    )
    if recompute_suffstats:
        return recompute_all_suffstats(updated, data)
    return updated


# ---------------------------------------------------------------------------
# Column hypers kernel (JIT-compatible via vmap)
# ---------------------------------------------------------------------------


def packed_transition_column_hypers(
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
    max_cpv = packed.max_cols_per_view

    # Pre-split keys: one per column
    col_keys = jax.random.split(rng_key, n_cols)

    def _find_local_index(v_idx, col_j):
        """Find local column index within a view using lax.scan."""

        def scan_fn(found_idx, li):
            matches = packed.view_column_indices[v_idx, li] == col_j
            new_idx = jnp.where(matches & (found_idx < 0), li, found_idx)
            return new_idx, None

        local_idx, _ = jax.lax.scan(scan_fn, jnp.array(-1, dtype=jnp.int32), jnp.arange(max_cpv))
        # Clamp to 0 if not found (should not happen for valid columns)
        return jnp.maximum(local_idx, 0)

    def _score_grid_ng(v_idx, local_idx, mu_val, r_val, s_grid_vals, nu_val):
        """Score a grid of s values for Normal-Gamma. Returns (n_grid,) scores."""
        nc = packed.view_n_clusters[v_idx]
        counts_col = packed.ss_counts[v_idx, :, local_idx]  # (max_c,)
        sum_x_col = packed.ss_sum_x[v_idx, :, local_idx]  # (max_c,)
        sum_x_sq_col = packed.ss_sum_x_sq[v_idx, :, local_idx]  # (max_c,)

        def score_one_grid_point(s_val):
            # Score across all clusters, mask inactive
            per_cluster = _ng_log_marginal(
                counts_col,
                sum_x_col,
                sum_x_sq_col,
                mu_val,
                r_val,
                s_val,
                nu_val,
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
                counts_col,
                sum_x_col,
                sum_x_sq_col,
                mu_val,
                r_val,
                s_val,
                nu_val,
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
                counts_col,
                sum_x_col,
                sum_x_sq_col,
                mu_val,
                r_val,
                s_val,
                nu_val,
            )
            masked = jnp.where(jnp.arange(max_c) < nc, per_cluster, 0.0)
            return jnp.sum(masked)

        return jax.vmap(score_one_grid_point)(nu_grid_vals)

    def _score_grid_ng_r(v_idx, local_idx, mu_val, r_grid_vals, s_val, nu_val):
        """Score a grid of r (precision scale) values for Normal-Gamma."""
        nc = packed.view_n_clusters[v_idx]
        counts_col = packed.ss_counts[v_idx, :, local_idx]
        sum_x_col = packed.ss_sum_x[v_idx, :, local_idx]
        sum_x_sq_col = packed.ss_sum_x_sq[v_idx, :, local_idx]

        def score_one_grid_point(r_val):
            per_cluster = _ng_log_marginal(
                counts_col,
                sum_x_col,
                sum_x_sq_col,
                mu_val,
                r_val,
                s_val,
                nu_val,
            )
            masked = jnp.where(jnp.arange(max_c) < nc, per_cluster, 0.0)
            return jnp.sum(masked)

        return jax.vmap(score_one_grid_point)(r_grid_vals)

    def process_one_column(j):
        """Process column j: sample hypers based on type. Returns updated hyper values."""
        key = col_keys[j]
        type_id = packed.col_type_ids[j]
        v_idx = packed.column_assignments[j]
        local_idx = _find_local_index(v_idx, j)

        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        # --- Continuous: sample s, then mu, then nu, then r ---
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

        # Sample r (precision scale) — paper requires all 4 NormalGamma hypers sampled
        r_grid = jnp.array([0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
        r_scores = _score_grid_ng_r(v_idx, local_idx, new_mu_val, r_grid, new_s_val, new_nu_val)
        r_scores = r_scores - jnp.max(r_scores)
        new_r_val = r_grid[jax.random.categorical(k4, r_scores)]

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
        a_grid_2d = jnp.repeat(ab_grid, ab_grid.shape[0])  # (25,)
        b_grid_2d = jnp.tile(ab_grid, ab_grid.shape[0])  # (25,)

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

        cur_vm_mu = packed.hyper_vm_mu[j]

        def score_vm_grid(kappa_val):
            per_cluster = _vm_log_marginal(
                counts_col_cat, sum_sin_col, sum_cos_col, kappa_val, cur_vm_mu
            )
            masked = jnp.where(jnp.arange(max_c) < nc, per_cluster, 0.0)
            return jnp.sum(masked)

        vm_scores = jax.vmap(score_vm_grid)(kappa_grid)
        vm_scores = vm_scores - jnp.max(vm_scores)
        new_kappa_val = kappa_grid[jax.random.categorical(k1, vm_scores)]

        # --- Select results based on type_id ---
        out_mu = jnp.where(type_id == CONTINUOUS_ID, new_mu_val, packed.hyper_mu[j])
        out_r = jnp.where(type_id == CONTINUOUS_ID, new_r_val, packed.hyper_r[j])
        out_s = jnp.where(type_id == CONTINUOUS_ID, new_s_val, packed.hyper_s[j])
        out_nu = jnp.where(type_id == CONTINUOUS_ID, new_nu_val, packed.hyper_nu[j])
        out_dir_alpha = jnp.where(
            type_id == CATEGORICAL_ID, new_dir_alpha_val, packed.hyper_dirichlet_alpha[j]
        )
        out_alpha = jnp.where(type_id == BINARY_ID, new_alpha_val, packed.hyper_alpha[j])
        out_beta = jnp.where(type_id == BINARY_ID, new_beta_val, packed.hyper_beta[j])
        out_kappa = jnp.where(type_id == CYCLIC_ID, new_kappa_val, packed.hyper_kappa[j])

        return (
            out_mu,
            out_r,
            out_s,
            out_nu,
            out_dir_alpha,
            out_alpha,
            out_beta,
            out_kappa,
        )

    # vmap over all columns
    (new_mu, new_r, new_s, new_nu, new_dir_alpha, new_alpha, new_beta, new_kappa) = jax.vmap(
        process_one_column
    )(jnp.arange(n_cols))

    return PackedCrossCatState(
        **{
            name: (
                new_mu
                if name == "hyper_mu"
                else new_r
                if name == "hyper_r"
                else new_s
                if name == "hyper_s"
                else new_nu
                if name == "hyper_nu"
                else new_dir_alpha
                if name == "hyper_dirichlet_alpha"
                else new_alpha
                if name == "hyper_alpha"
                else new_beta
                if name == "hyper_beta"
                else new_kappa
                if name == "hyper_kappa"
                else getattr(packed, name)
            )
            for name in _ARRAY_FIELDS
        },
        **{name: getattr(packed, name) for name in _STATIC_FIELDS},
    )


# ---------------------------------------------------------------------------
# CRP alpha kernel (JIT-compatible via vmap)
# ---------------------------------------------------------------------------


def packed_transition_crp_alphas(
    rng_key: Array,
    packed: PackedCrossCatState,
) -> PackedCrossCatState:
    """Sample CRP concentration parameters using vmap (JIT-compatible).

    Scores a grid of alpha values for the outer (column) CRP and each inner
    (row) CRP. Includes Exp(1) prior: log_score -= alpha_val.
    """
    alpha_grid = jnp.exp(jnp.linspace(jnp.log(0.01), jnp.log(100.0), 50))
    max_views = packed.max_views
    n_cols = packed.n_cols
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
        **{
            name: (
                new_col_alpha
                if name == "column_crp_alpha"
                else new_view_alpha
                if name == "view_row_crp_alpha"
                else getattr(packed, name)
            )
            for name in _ARRAY_FIELDS
        },
        **{name: getattr(packed, name) for name in _STATIC_FIELDS},
    )


# ---------------------------------------------------------------------------
# Column assignment helpers
# ---------------------------------------------------------------------------


def _crp_sample_bounded(
    rng_key: Array, alpha: Array, n: int, max_clusters: int
) -> tuple[Array, Array]:
    """Sample row assignments from CRP, bounded to max_clusters tables.

    Args:
        rng_key: PRNG key.
        alpha: CRP concentration (traced scalar).
        n: Number of items (static int).
        max_clusters: Maximum number of tables (static int).

    Returns:
        (assignments, n_tables) — assignments is (n,) int, n_tables is scalar int.
    """

    def _step(carry, key):
        counts, n_tables = carry
        table_probs = jnp.where(jnp.arange(max_clusters) < n_tables, counts, 0.0)
        can_new = n_tables < max_clusters
        table_probs = table_probs.at[n_tables].set(jnp.where(can_new, alpha, 0.0))
        log_probs = jnp.log(table_probs + 1e-30)
        chosen = jax.random.categorical(key, log_probs)

        is_new = (chosen == n_tables) & can_new
        new_n_tables = jnp.where(is_new, n_tables + 1, n_tables)
        new_counts = counts.at[chosen].add(1.0)
        return (new_counts, new_n_tables), chosen

    keys = jax.random.split(rng_key, n)
    init = (
        jnp.zeros(max_clusters, dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
    )
    (_, n_tables), assignments = jax.lax.scan(_step, init, keys)
    return assignments, n_tables


def _score_column_in_view(
    data_col: Array,
    row_assignments: Array,
    type_id: Array,
    mu: Array,
    r: Array,
    s: Array,
    nu: Array,
    dir_alpha: Array,
    alpha: Array,
    beta: Array,
    kappa: Array,
    vm_mu: Array,
    max_clusters: int,
    max_categories: int,
) -> Array:
    """Log marginal likelihood of one column's data under a view's clustering.

    Computes per-cluster sufficient statistics via matrix ops, then vmaps
    unified_log_marginal over clusters.

    Args:
        data_col: (n_rows,) column data (may contain NaN).
        row_assignments: (n_rows,) int cluster assignments for this view.
        type_id: scalar int column type.
        mu, r, s, nu, dir_alpha, alpha, beta, kappa, vm_mu: scalar hypers.
        max_clusters: static int padding.
        max_categories: static int padding.

    Returns:
        Scalar total log marginal likelihood.
    """
    valid = ~jnp.isnan(data_col)
    clean = jnp.where(valid, data_col, 0.0)
    valid_f = valid.astype(jnp.float32)

    # Membership matrix: (n_rows, max_clusters)
    membership = (row_assignments[:, None] == jnp.arange(max_clusters)[None, :]).astype(
        jnp.float32
    )

    # Per-cluster sufficient statistics for this single column
    counts = (membership.T @ valid_f).astype(jnp.int32)  # (max_clusters,)
    sum_x = membership.T @ (clean * valid_f)  # (max_clusters,)
    sum_x_sq = membership.T @ (clean**2 * valid_f)  # (max_clusters,)
    sum_sin = membership.T @ jnp.where(valid, jnp.sin(data_col), 0.0)
    sum_cos = membership.T @ jnp.where(valid, jnp.cos(data_col), 0.0)

    # Category counts: (max_clusters, max_categories)
    int_data = jnp.where(valid, clean.astype(jnp.int32), 0)
    one_hot = jax.nn.one_hot(int_data, max_categories)  # (n_rows, max_cats)
    cat_counts = membership.T @ (one_hot * valid_f[:, None])  # (max_clusters, max_cats)

    # vmap unified_log_marginal over clusters
    def score_one_cluster(cnt, sx, sxsq, cc, ssin, scos):
        return unified_log_marginal(
            type_id,
            cnt,
            sx,
            sxsq,
            cc,
            ssin,
            scos,
            mu,
            r,
            s,
            nu,
            dir_alpha,
            alpha,
            beta,
            kappa,
            vm_mu,
        )

    log_mls = jax.vmap(score_one_cluster)(
        counts, sum_x, sum_x_sq, cat_counts, sum_sin, sum_cos
    )  # (max_clusters,)

    # Only sum contributions from clusters with data
    return jnp.sum(jnp.where(counts > 0, log_mls, 0.0))


# ---------------------------------------------------------------------------
# Column assignment kernel
# ---------------------------------------------------------------------------


def packed_transition_column_assignments(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
) -> PackedCrossCatState:
    """Gibbs sweep over column-to-view assignments (outer DP), JIT-compatible.

    Uses lax.scan over columns. For each column j:
      1. Remove j from its current view (decrement count).
      2. Score each existing view: CRP prior + log marginal likelihood.
      3. Propose a new singleton view with CRP-sampled row assignments.
      4. Sample new assignment from categorical.
      5. Update column_assignments and view metadata.

    After the scan, rebuilds view_column_indices and recomputes all suffstats.

    Args:
        rng_key: PRNG key.
        packed: Current packed state.
        data: (n_rows, n_cols) data matrix.

    Returns:
        Updated PackedCrossCatState with new column assignments.
    """
    n_rows = packed.n_rows
    n_cols = packed.n_cols
    max_views = packed.max_views
    max_clusters = packed.max_clusters
    max_cats = packed.max_categories
    max_cpv = packed.max_cols_per_view

    def scan_one_col(carry, j):
        (
            col_assigns,
            view_mask,
            view_n_cols,
            view_row_assigns,
            view_n_clusters,
            view_row_crp_alpha,
            rng,
        ) = carry

        k_crp, k_cat, k_alpha, rng = jax.random.split(rng, 4)

        old_view = col_assigns[j]
        data_col = data[:, j]
        type_id = packed.col_type_ids[j]

        # Hyperparameters for column j
        h_mu = packed.hyper_mu[j]
        h_r = packed.hyper_r[j]
        h_s = packed.hyper_s[j]
        h_nu = packed.hyper_nu[j]
        h_dir = packed.hyper_dirichlet_alpha[j]
        h_a = packed.hyper_alpha[j]
        h_b = packed.hyper_beta[j]
        h_k = packed.hyper_kappa[j]
        h_vm = packed.hyper_vm_mu[j]

        # Column counts per view, excluding column j
        counts_excl = view_n_cols.at[old_view].add(-1)

        # --- Score existing views ---
        # CRP prior: log(count_v) for views with columns, -inf otherwise
        log_prior = jnp.log(jnp.maximum(counts_excl.astype(jnp.float32), 1e-30))
        log_prior = jnp.where((counts_excl > 0) & view_mask, log_prior, -jnp.inf)

        # Likelihood: vmap _score_column_in_view over views
        def score_in_view(row_assigns_v):
            return _score_column_in_view(
                data_col,
                row_assigns_v,
                type_id,
                h_mu,
                h_r,
                h_s,
                h_nu,
                h_dir,
                h_a,
                h_b,
                h_k,
                h_vm,
                max_clusters,
                max_cats,
            )

        log_lik = jax.vmap(score_in_view)(view_row_assigns)  # (max_views,)
        log_lik = jnp.where(view_mask & (counts_excl > 0), log_lik, -jnp.inf)

        log_scores_existing = log_prior + log_lik  # (max_views,)

        # --- Score new view proposal ---
        # Paper (Algorithm 8, Neal 1998): if column j is a singleton (only column
        # in its view), reuse the current view's row assignments as the auxiliary
        # variable. Otherwise, sample fresh CRP assignments with alpha from Gamma(1,1).
        log_prior_new = jnp.log(packed.column_crp_alpha)
        is_singleton = counts_excl[old_view] == 0

        # Always compute both paths (JIT-compatible), select via jnp.where
        gamma_alpha = jax.random.gamma(k_alpha, 1.0)
        fresh_row_assigns, fresh_n_clusters = _crp_sample_bounded(
            k_crp,
            gamma_alpha,
            n_rows,
            max_clusters,
        )
        reused_row_assigns = view_row_assigns[old_view]
        reused_n_clusters = view_n_clusters[old_view]
        reused_alpha = view_row_crp_alpha[old_view]

        new_row_assigns = jnp.where(is_singleton, reused_row_assigns, fresh_row_assigns)
        new_n_clusters = jnp.where(is_singleton, reused_n_clusters, fresh_n_clusters)
        new_view_alpha = jnp.where(is_singleton, reused_alpha, gamma_alpha)
        log_lik_new = _score_column_in_view(
            data_col,
            new_row_assigns,
            type_id,
            h_mu,
            h_r,
            h_s,
            h_nu,
            h_dir,
            h_a,
            h_b,
            h_k,
            h_vm,
            max_clusters,
            max_cats,
        )
        log_score_new = log_prior_new + log_lik_new

        # --- Sample assignment ---
        all_scores = jnp.concatenate([log_scores_existing, log_score_new[None]])
        all_scores = all_scores - jnp.max(all_scores)  # numerical stability
        chosen = jax.random.categorical(k_cat, all_scores)

        is_new_view = chosen == max_views

        # Find first inactive slot for new view
        # Use large index for active slots so argmin picks inactive
        slot_priority = jnp.where(view_mask, max_views + 1, jnp.arange(max_views))
        new_slot = jnp.argmin(slot_priority).astype(jnp.int32)

        actual_view = jnp.where(is_new_view, new_slot, chosen)

        # Update column assignments
        new_col_assigns = col_assigns.at[j].set(actual_view)

        # Update view_n_cols: excl already has old_view decremented, now add to actual_view
        new_view_n_cols = counts_excl.at[actual_view].add(1)

        # Update view_mask: activate new slot if needed
        new_view_mask = jnp.where(
            is_new_view,
            view_mask.at[new_slot].set(True),
            view_mask,
        )
        # Deactivate old view if it became empty
        old_view_empty = counts_excl[old_view] == 0
        new_view_mask = jnp.where(
            old_view_empty,
            new_view_mask.at[old_view].set(False),
            new_view_mask,
        )

        # Store new view's row assignments if creating a new view
        new_view_row_assigns = jnp.where(
            is_new_view,
            view_row_assigns.at[new_slot].set(new_row_assigns),
            view_row_assigns,
        )
        new_view_n_clusters = jnp.where(
            is_new_view,
            view_n_clusters.at[new_slot].set(new_n_clusters),
            view_n_clusters,
        )
        new_view_row_crp_alpha = jnp.where(
            is_new_view,
            view_row_crp_alpha.at[new_slot].set(new_view_alpha),
            view_row_crp_alpha,
        )

        new_carry = (
            new_col_assigns,
            new_view_mask,
            new_view_n_cols,
            new_view_row_assigns,
            new_view_n_clusters,
            new_view_row_crp_alpha,
            rng,
        )
        return new_carry, None

    init_carry = (
        packed.column_assignments,
        packed.view_mask,
        packed.view_n_columns,
        packed.view_row_assignments,
        packed.view_n_clusters,
        packed.view_row_crp_alpha,
        rng_key,
    )

    (
        (
            col_assigns,
            view_mask,
            view_n_cols,
            view_row_assigns,
            view_n_clusters,
            view_row_crp_alpha,
            _,
        ),
        _,
    ) = jax.lax.scan(
        scan_one_col,
        init_carry,
        jnp.arange(n_cols),
    )

    n_views = jnp.sum(view_mask.astype(jnp.int32))

    # --- Compact view indices to 0..n_views-1 ---
    # Build a permutation that moves active views to the front.
    # Active slots get low sort keys, inactive slots get high sort keys.
    sort_key = jnp.where(view_mask, jnp.arange(max_views), max_views + jnp.arange(max_views))
    perm = jnp.argsort(sort_key)  # old_idx -> position in sorted order
    # Inverse: for each old view index, what's its new contiguous index?
    inv_perm = jnp.zeros(max_views, dtype=jnp.int32)
    inv_perm = inv_perm.at[perm].set(jnp.arange(max_views, dtype=jnp.int32))

    # Remap column assignments
    compact_col_assigns = inv_perm[col_assigns]

    # Reorder view arrays using perm (gather active views to front)
    compact_view_mask = view_mask[perm]
    compact_view_n_cols = view_n_cols[perm]
    compact_view_row_assigns = view_row_assigns[perm]
    compact_view_n_clusters = view_n_clusters[perm]
    compact_view_row_crp_alpha = view_row_crp_alpha[perm]

    # Rebuild view_column_indices from compacted column_assignments
    def build_view_col_indices(v):
        is_in_view = compact_col_assigns == v  # (n_cols,)
        indices = jnp.where(is_in_view, jnp.arange(n_cols), n_cols)
        sorted_indices = jnp.sort(indices)
        result = sorted_indices[:max_cpv]
        return jnp.where(result < n_cols, result, -1).astype(jnp.int32)

    new_view_col_indices = jax.vmap(build_view_col_indices)(
        jnp.arange(max_views)
    )  # (max_views, max_cpv)

    # Build updated state (suffstats will be recomputed)
    new_packed = PackedCrossCatState(
        **{
            name: (
                compact_col_assigns
                if name == "column_assignments"
                else packed.column_crp_alpha
                if name == "column_crp_alpha"
                else n_views
                if name == "n_views"
                else compact_view_mask
                if name == "view_mask"
                else new_view_col_indices
                if name == "view_column_indices"
                else compact_view_n_cols
                if name == "view_n_columns"
                else compact_view_row_assigns
                if name == "view_row_assignments"
                else compact_view_n_clusters
                if name == "view_n_clusters"
                else compact_view_row_crp_alpha
                if name == "view_row_crp_alpha"
                else getattr(packed, name)
            )
            for name in _ARRAY_FIELDS
        },
        **{name: getattr(packed, name) for name in _STATIC_FIELDS},
    )

    # Recompute all sufficient statistics from scratch
    return recompute_all_suffstats(new_packed, data)


# ---------------------------------------------------------------------------
# Packed Gibbs sweep (JIT-compatible via lax.scan)
# ---------------------------------------------------------------------------


def packed_gibbs_sweep(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    *,
    n_sweeps: int = 1,
) -> PackedCrossCatState:
    """Run full Gibbs sweeps using JIT-compatible kernels via lax.scan.

    Each sweep runs:
      1. Row assignments (packed_transition_row_assignments)
      2. Column assignments (packed_transition_column_assignments)
      3. Column hyperparameters (packed_transition_column_hypers)
      4. CRP alphas (packed_transition_crp_alphas)

    Args:
        rng_key: PRNG key.
        packed: Current packed state.
        data: (n_rows, n_cols) data matrix.
        n_sweeps: Number of full Gibbs sweeps (static int for lax.scan).

    Returns:
        Updated PackedCrossCatState after n_sweeps sweeps.
    """

    def one_sweep(carry, _):
        state, rng = carry
        k1, k2, k3, k4, rng = jax.random.split(rng, 5)
        state = packed_transition_row_assignments(k1, state, data)
        state = packed_transition_column_assignments(k2, state, data)
        state = packed_transition_column_hypers(k3, state, data)
        state = packed_transition_crp_alphas(k4, state)
        return (state, rng), None

    (result, _), _ = jax.lax.scan(one_sweep, (packed, rng_key), jnp.arange(n_sweeps))
    return result


# ---------------------------------------------------------------------------
# Packed log-joint scoring
# ---------------------------------------------------------------------------


def packed_log_joint(packed: PackedCrossCatState, data: Array) -> Array:
    """JIT-compatible log-joint probability on packed state.

    Computes: log CRP(columns) + sum_v log CRP(rows_v)
              + sum_v sum_c sum_col log p(data | suffstats, hypers)
              - Exp(1) priors on CRP alphas
    """
    # --- Column CRP ---
    col_assigns = packed.column_assignments
    n_cols = packed.n_cols
    n_views_val = packed.n_views
    col_alpha = packed.column_crp_alpha

    # Count columns per view
    col_counts = jnp.bincount(col_assigns, length=packed.max_views).astype(jnp.float32)
    # Only active views contribute
    col_crp = (
        n_views_val * jnp.log(col_alpha)
        + jnp.sum(jnp.where(packed.view_mask, gammaln(col_counts), 0.0))
        - gammaln(n_cols + col_alpha)
        + gammaln(col_alpha)
    )

    # --- Row CRP per view (vectorized via vmap) ---
    def _compute_row_crp(v_idx):
        row_assigns = packed.view_row_assignments[v_idx]
        alpha_v = packed.view_row_crp_alpha[v_idx]
        n_clusters_v = packed.view_n_clusters[v_idx]

        # Count rows per cluster using one-hot
        one_hot = (jnp.arange(packed.max_clusters)[None, :] == row_assigns[:, None]).astype(
            jnp.float32
        )
        row_counts = jnp.sum(one_hot, axis=0)

        active_mask = jnp.arange(packed.max_clusters) < n_clusters_v
        crp = (
            n_clusters_v * jnp.log(alpha_v)
            + jnp.sum(jnp.where(active_mask, gammaln(jnp.maximum(row_counts, 1.0)), 0.0))
            - gammaln(packed.n_rows + alpha_v)
            + gammaln(alpha_v)
        )
        return crp

    row_crps = jax.vmap(_compute_row_crp)(jnp.arange(packed.max_views))
    row_crp_total = jnp.sum(jnp.where(packed.view_mask, row_crps, 0.0))

    # --- Data likelihood: sum over views, clusters, columns ---
    def _score_one_view(v_idx):
        n_cols_v = packed.view_n_columns[v_idx]
        n_clusters_v = packed.view_n_clusters[v_idx]

        def _score_one_cluster_col(c_idx, l_idx):
            col_idx = packed.view_column_indices[v_idx, l_idx]
            return unified_log_marginal(
                packed.col_type_ids[col_idx],
                packed.ss_counts[v_idx, c_idx, l_idx],
                packed.ss_sum_x[v_idx, c_idx, l_idx],
                packed.ss_sum_x_sq[v_idx, c_idx, l_idx],
                packed.ss_cat_counts[v_idx, c_idx, l_idx],
                packed.ss_sum_sin[v_idx, c_idx, l_idx],
                packed.ss_sum_cos[v_idx, c_idx, l_idx],
                packed.hyper_mu[col_idx],
                packed.hyper_r[col_idx],
                packed.hyper_s[col_idx],
                packed.hyper_nu[col_idx],
                packed.hyper_dirichlet_alpha[col_idx],
                packed.hyper_alpha[col_idx],
                packed.hyper_beta[col_idx],
                packed.hyper_kappa[col_idx],
                packed.hyper_vm_mu[col_idx],
            )

        # vmap over clusters and columns
        cluster_indices = jnp.arange(packed.max_clusters)
        col_indices = jnp.arange(packed.max_cols_per_view)

        # Score all (cluster, col) pairs
        all_scores = jax.vmap(
            lambda c: jax.vmap(lambda li: _score_one_cluster_col(c, li))(col_indices)
        )(cluster_indices)  # (max_clusters, max_cols_per_view)

        # Mask: only active clusters and columns
        cluster_mask = jnp.arange(packed.max_clusters)[:, None] < n_clusters_v
        col_mask = jnp.arange(packed.max_cols_per_view)[None, :] < n_cols_v
        mask = cluster_mask & col_mask

        return jnp.sum(jnp.where(mask, all_scores, 0.0))

    view_scores = jax.vmap(_score_one_view)(jnp.arange(packed.max_views))
    data_ll = jnp.sum(jnp.where(packed.view_mask, view_scores, 0.0))

    # --- Exp(1) priors on CRP alphas ---
    alpha_prior = -col_alpha - jnp.sum(jnp.where(packed.view_mask, packed.view_row_crp_alpha, 0.0))

    return col_crp + row_crp_total + data_ll + alpha_prior


# ---------------------------------------------------------------------------
# Multi-chain parallel inference via vmap
# ---------------------------------------------------------------------------


def multi_chain_packed_gibbs_sweep(
    rng_key: Array,
    packed_list: list[PackedCrossCatState],
    data: Array,
    *,
    n_sweeps: int = 1,
) -> tuple[PackedCrossCatState, Array]:
    """Run packed Gibbs sweeps across N chains in parallel via vmap.

    Args:
        rng_key: PRNG key (will be split into N subkeys).
        packed_list: List of N PackedCrossCatState (one per chain).
        data: Shared data matrix (n_rows, n_cols).
        n_sweeps: Number of sweeps per chain.

    Returns:
        (batched_result, log_joint_scores) where batched_result has leading
        (n_chains,) dimension on all arrays, and scores is (n_chains,).
    """
    from crosscat.packed.state import batch_packed_states

    n_chains = len(packed_list)
    batched = batch_packed_states(packed_list)
    keys = jax.random.split(rng_key, n_chains)

    # vmap sweep across chains (data is broadcast)
    vmapped_sweep = jax.vmap(lambda k, p: packed_gibbs_sweep(k, p, data, n_sweeps=n_sweeps))
    batched_result = vmapped_sweep(keys, batched)

    # Score each chain
    vmapped_score = jax.vmap(lambda p: packed_log_joint(p, data))
    scores = vmapped_score(batched_result)

    return batched_result, scores
