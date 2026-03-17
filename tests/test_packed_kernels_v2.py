"""Tests for vectorized (v2) packed kernels and packed inference.

Validates correctness by comparing against unpacked reference implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.model import initialize, log_joint
from crosscat.packed_state import (
    _add_row_to_suffstats,
    _remove_row_from_suffstats,
    _score_row_all_clusters,
    _score_row_all_clusters_v2,
    pack_state,
    packed_transition_column_hypers_v2,
    packed_transition_row_assignments_v2,
    unpack_state,
)
from crosscat.types import ColumnType
from crosscat.validate import validate_state


@pytest.fixture
def mixed_packed_state():
    """Mixed-type packed state for testing v2 kernels."""
    key = jax.random.key(42)
    from crosscat.synthetic import generate_crosscat_data

    column_types = [
        ColumnType.CONTINUOUS,
        ColumnType.CATEGORICAL,
        ColumnType.BINARY,
        ColumnType.CYCLIC,
    ]
    result = generate_crosscat_data(key, 50, column_types, n_views=2, n_clusters=2)
    k2 = jax.random.key(43)
    state = initialize(k2, result["data"], column_types)
    packed = pack_state(state, max_clusters=8, max_categories=8)
    return packed, result["data"], column_types


def test_incremental_suffstats_correctness(mixed_packed_state):
    """Remove row then add it back equals original suffstats."""
    packed, data, column_types = mixed_packed_state
    v = 0  # test first view
    row_idx = 5
    n_cols_v = int(packed.view_n_columns[v])
    col_indices = packed.view_column_indices[v, :n_cols_v]
    old_cluster = packed.view_row_assignments[v, row_idx]

    # Remove row
    ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos = _remove_row_from_suffstats(
        packed.ss_counts[v], packed.ss_sum_x[v], packed.ss_sum_x_sq[v],
        packed.ss_cat_counts[v], packed.ss_sum_sin[v], packed.ss_sum_cos[v],
        old_cluster, data[row_idx], col_indices, packed.col_type_ids,
        packed.max_categories,
    )

    # Add row back to same cluster
    ss_c2, ss_sx2, ss_sxsq2, ss_cat2, ss_sin2, ss_cos2 = _add_row_to_suffstats(
        ss_c, ss_sx, ss_sxsq, ss_cat, ss_sin, ss_cos,
        old_cluster, data[row_idx], col_indices, packed.col_type_ids,
        packed.max_categories,
    )

    # Should match original
    assert jnp.allclose(ss_c2[:, :n_cols_v], packed.ss_counts[v, :, :n_cols_v], atol=1e-5)
    assert jnp.allclose(ss_sx2[:, :n_cols_v], packed.ss_sum_x[v, :, :n_cols_v], atol=1e-5)
    assert jnp.allclose(ss_sxsq2[:, :n_cols_v], packed.ss_sum_x_sq[v, :, :n_cols_v], atol=1e-5)
    assert jnp.allclose(ss_sin2[:, :n_cols_v], packed.ss_sum_sin[v, :, :n_cols_v], atol=1e-5)
    assert jnp.allclose(ss_cos2[:, :n_cols_v], packed.ss_sum_cos[v, :, :n_cols_v], atol=1e-5)


def test_score_row_all_clusters_v2_matches_v1(mixed_packed_state):
    """Vectorized row scoring matches original loop-based scoring."""
    packed, data, column_types = mixed_packed_state
    max_c = packed.max_clusters

    # Test on each active view, for a couple of rows
    n_views = int(packed.n_views)
    for v in range(n_views):
        n_cols_v = int(packed.view_n_columns[v])
        col_indices_sliced = packed.view_column_indices[v, :n_cols_v]
        col_indices_full = packed.view_column_indices[v]
        alpha = packed.view_row_crp_alpha[v]

        for row_idx in [0, 5, 10]:
            # Cluster counts excluding this row
            assigns_excl = packed.view_row_assignments[v].at[row_idx].set(-1)
            counts = jnp.array(
                [jnp.sum(assigns_excl == c) for c in range(max_c)]
            ).astype(jnp.int32)

            # v1 (Python loop)
            log_probs_v1 = _score_row_all_clusters(
                data[row_idx], col_indices_sliced, n_cols_v, packed.col_type_ids,
                counts,
                packed.ss_counts[v], packed.ss_sum_x[v], packed.ss_sum_x_sq[v],
                packed.ss_cat_counts[v], packed.ss_sum_sin[v], packed.ss_sum_cos[v],
                packed.hyper_mu, packed.hyper_r, packed.hyper_s, packed.hyper_nu,
                packed.hyper_dirichlet_alpha, packed.hyper_alpha, packed.hyper_beta,
                packed.hyper_kappa, packed.hyper_vm_mu,
                alpha, max_c,
            )

            # v2 (lax.scan + vmap)
            log_probs_v2 = _score_row_all_clusters_v2(
                data[row_idx], col_indices_full,
                packed.view_n_columns[v], packed.col_type_ids,
                counts,
                packed.ss_counts[v], packed.ss_sum_x[v], packed.ss_sum_x_sq[v],
                packed.ss_cat_counts[v], packed.ss_sum_sin[v], packed.ss_sum_cos[v],
                packed.hyper_mu, packed.hyper_r, packed.hyper_s, packed.hyper_nu,
                packed.hyper_dirichlet_alpha, packed.hyper_alpha, packed.hyper_beta,
                packed.hyper_kappa, packed.hyper_vm_mu,
                alpha, max_c,
            )

            assert log_probs_v1.shape == log_probs_v2.shape == (max_c + 1,)

            # Compare finite entries
            both_finite = jnp.isfinite(log_probs_v1) & jnp.isfinite(log_probs_v2)
            if jnp.any(both_finite):
                assert jnp.allclose(
                    jnp.where(both_finite, log_probs_v1, 0.0),
                    jnp.where(both_finite, log_probs_v2, 0.0),
                    atol=1e-4,
                ), (
                    f"v={v}, row={row_idx}: v1 vs v2 mismatch\n"
                    f"  v1={log_probs_v1}\n  v2={log_probs_v2}"
                )

            # Non-finite entries should match (both -inf)
            assert jnp.array_equal(
                jnp.isfinite(log_probs_v1), jnp.isfinite(log_probs_v2)
            ), (
                f"v={v}, row={row_idx}: finite/inf pattern mismatch\n"
                f"  v1 finite={jnp.isfinite(log_probs_v1)}\n"
                f"  v2 finite={jnp.isfinite(log_probs_v2)}"
            )


# ---------------------------------------------------------------------------
# Task 4: lax.scan row assignment kernel tests
# ---------------------------------------------------------------------------


def test_scan_row_assignments_produces_valid_state(mixed_packed_state):
    """packed_transition_row_assignments_v2 produces a valid unpacked state."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(101)
    packed_new = packed_transition_row_assignments_v2(key, packed, data)
    recovered = unpack_state(packed_new, column_types)

    errors = validate_state(recovered, data)
    assert errors == [], f"Validation errors: {errors}"
    lj = float(log_joint(recovered, data))
    assert jnp.isfinite(jnp.array(lj)), f"log_joint not finite: {lj}"


def test_scan_row_assignments_jit_compiles(mixed_packed_state):
    """packed_transition_row_assignments_v2 works under jax.jit."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(202)

    jitted_fn = jax.jit(packed_transition_row_assignments_v2)
    packed_new = jitted_fn(key, packed, data)

    recovered = unpack_state(packed_new, column_types)
    errors = validate_state(recovered, data)
    assert errors == [], f"Validation errors after JIT: {errors}"
    lj = float(log_joint(recovered, data))
    assert jnp.isfinite(jnp.array(lj)), f"log_joint not finite after JIT: {lj}"


# ---------------------------------------------------------------------------
# Task 5: vmap column hypers kernel tests
# ---------------------------------------------------------------------------


def test_vmap_column_hypers_produces_valid_state(mixed_packed_state):
    """packed_transition_column_hypers_v2 produces finite and positive hypers."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(301)
    packed_new = packed_transition_column_hypers_v2(key, packed, data)

    # All hypers should be finite
    assert jnp.all(jnp.isfinite(packed_new.hyper_mu)), "hyper_mu has non-finite values"
    assert jnp.all(jnp.isfinite(packed_new.hyper_r)), "hyper_r has non-finite values"
    assert jnp.all(jnp.isfinite(packed_new.hyper_s)), "hyper_s has non-finite values"
    assert jnp.all(jnp.isfinite(packed_new.hyper_nu)), "hyper_nu has non-finite values"
    assert jnp.all(jnp.isfinite(packed_new.hyper_dirichlet_alpha)), (
        "hyper_dirichlet_alpha has non-finite values"
    )
    assert jnp.all(jnp.isfinite(packed_new.hyper_alpha)), "hyper_alpha has non-finite values"
    assert jnp.all(jnp.isfinite(packed_new.hyper_beta)), "hyper_beta has non-finite values"
    assert jnp.all(jnp.isfinite(packed_new.hyper_kappa)), "hyper_kappa has non-finite values"

    # Positive-definite hypers should be positive
    assert jnp.all(packed_new.hyper_r > 0), "hyper_r should be positive"
    assert jnp.all(packed_new.hyper_s > 0), "hyper_s should be positive"
    assert jnp.all(packed_new.hyper_nu > 0), "hyper_nu should be positive"
    assert jnp.all(packed_new.hyper_dirichlet_alpha > 0), (
        "hyper_dirichlet_alpha should be positive"
    )
    assert jnp.all(packed_new.hyper_alpha > 0), "hyper_alpha should be positive"
    assert jnp.all(packed_new.hyper_beta > 0), "hyper_beta should be positive"
    assert jnp.all(packed_new.hyper_kappa > 0), "hyper_kappa should be positive"


def test_vmap_column_hypers_jit_compiles(mixed_packed_state):
    """packed_transition_column_hypers_v2 works under jax.jit."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(302)

    jitted_fn = jax.jit(packed_transition_column_hypers_v2)
    packed_new = jitted_fn(key, packed, data)

    # Verify all hypers are finite after JIT
    assert jnp.all(jnp.isfinite(packed_new.hyper_mu)), "hyper_mu non-finite after JIT"
    assert jnp.all(jnp.isfinite(packed_new.hyper_s)), "hyper_s non-finite after JIT"
    assert jnp.all(jnp.isfinite(packed_new.hyper_nu)), "hyper_nu non-finite after JIT"
    assert jnp.all(jnp.isfinite(packed_new.hyper_kappa)), "hyper_kappa non-finite after JIT"
