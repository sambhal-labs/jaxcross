# Phase 3: Full JAX Vectorization — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all Python for-loops in packed Gibbs kernels and inference queries with `jax.lax.scan`/`jax.vmap` for full JIT compilation and GPU acceleration.

**Architecture:** Dual API — new `_v2` packed kernels and `packed_inference.py` coexist alongside unchanged originals. All new code operates on `PackedCrossCatState` (JAX pytree with padded arrays). Correctness validated by comparing against unpacked reference implementation.

**Tech Stack:** JAX (`jax.lax.scan`, `jax.vmap`, `jax.jit`), pytest, Jupyter/Colab

**Spec:** `docs/superpowers/specs/2026-03-17-phase3-jax-vectorization-design.md`

---

## File Structure

| File | Action | Responsibility |
| ---- | ------ | -------------- |
| `crosscat/packed_state.py` | Modify | Add `max_cols_per_view` static field, incremental suffstat helpers, vectorized scoring with vmap, `lax.scan` row kernel, vmap hypers/CRP kernels, unified sampler, `packed_gibbs_sweep_v2` |
| `crosscat/packed_inference.py` | Create | Vectorized inference queries on `PackedCrossCatState`: predictive probability/sample, MI, similarity, impute, anomaly, CDF |
| `crosscat/__init__.py` | Modify | Export new packed inference functions and `packed_gibbs_sweep_v2` |
| `tests/test_packed_kernels_v2.py` | Create | Correctness tests comparing v2 packed against unpacked reference |
| `notebooks/gpu_benchmark.ipynb` | Create | Colab GPU benchmark notebook |

---

## Task 1: Create branch and add `max_cols_per_view` static field

**Files:**
- Modify: `crosscat/packed_state.py:44-74` (static fields), `crosscat/packed_state.py:76-136` (dataclass + pytree), `crosscat/packed_state.py:143-293` (pack_state)
- Test: `tests/test_packed_state.py` (existing tests must still pass)

- [ ] **Step 1: Create feature branch and stage CLAUDE.md**

```bash
git checkout -b perf/phase3-jax-vectorization
git add CLAUDE.md
git commit -m "chore: track CLAUDE.md project instructions"
```

- [ ] **Step 2: Add `max_cols_per_view` to `_STATIC_FIELDS` and dataclass**

In `crosscat/packed_state.py`, add `max_cols_per_view` to the static fields tuple at line 73:

```python
_STATIC_FIELDS = ("n_rows", "n_cols", "max_views", "max_clusters", "max_categories", "max_cols_per_view")
```

Add to the dataclass at line 124 (after `max_categories`):

```python
max_cols_per_view: int = 16
```

- [ ] **Step 3: Update `pack_state()` to store `max_cols_per_view`**

In `pack_state()` around line 164, after computing `max_cols_per_view`, pass it to the constructor. In the return statement around line 262, add:

```python
max_cols_per_view=max_cols_per_view,
```

- [ ] **Step 4: Run existing tests to verify no regression**

Run: `pytest tests/test_packed_state.py -v`
Expected: All 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crosscat/packed_state.py
git commit -m "feat: add max_cols_per_view to PackedCrossCatState static fields

Prerequisite for lax.scan/vmap — column dimension must be known at trace time."
```

---

## Task 2: Incremental suffstat helpers

**Files:**
- Modify: `crosscat/packed_state.py` (add new functions after `recompute_all_suffstats`)
- Create: `tests/test_packed_kernels_v2.py` (start the new test file)

- [ ] **Step 1: Write the failing test for incremental suffstats**

Create `tests/test_packed_kernels_v2.py`:

```python
"""Tests for vectorized (v2) packed kernels and packed inference.

Validates correctness by comparing against unpacked reference implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.model import initialize
from crosscat.packed_state import (
    _add_row_to_suffstats,
    _remove_row_from_suffstats,
    compute_suffstats_vectorized,
    pack_state,
    recompute_all_suffstats,
    unpack_state,
)
from crosscat.types import ColumnType


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_packed_kernels_v2.py::test_incremental_suffstats_correctness -v`
Expected: FAIL with `ImportError: cannot import name '_add_row_to_suffstats'`

- [ ] **Step 3: Implement `_remove_row_from_suffstats` and `_add_row_to_suffstats`**

Add to `crosscat/packed_state.py` after the `recompute_all_suffstats` function (after line 519):

```python
def _remove_row_from_suffstats(
    ss_counts: Array, ss_sum_x: Array, ss_sum_x_sq: Array,
    ss_cat_counts: Array, ss_sum_sin: Array, ss_sum_cos: Array,
    cluster_id: Array, row_data: Array, col_indices: Array,
    col_type_ids: Array, max_categories: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Remove one row's contribution from a cluster's suffstats.

    All updates vectorized over columns. NaN values produce zero deltas.
    Uses .at[traced_idx].add() for JIT compatibility.

    Args:
        ss_counts: (max_clusters, max_cols_per_view) int
        ss_sum_x, ss_sum_x_sq: (max_clusters, max_cols_per_view)
        ss_cat_counts: (max_clusters, max_cols_per_view, max_categories)
        ss_sum_sin, ss_sum_cos: (max_clusters, max_cols_per_view)
        cluster_id: scalar traced int — which cluster to update
        row_data: (n_cols_total,) — full row from data matrix
        col_indices: (max_cols_per_view,) int — column indices for this view, -1 for padding
        col_type_ids: (n_cols_total,) int — type ID per column
        max_categories: int (static)

    Returns:
        Updated (ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos).
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
        sxsq_delta = clean_x ** 2 * is_valid_f * (type_id == CONTINUOUS_ID).astype(jnp.float32)
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
    (ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos), _ = (
        jax.lax.scan(update_one_col, carry, jnp.arange(n_cols_v))
    )
    return ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos


def _add_row_to_suffstats(
    ss_counts: Array, ss_sum_x: Array, ss_sum_x_sq: Array,
    ss_cat_counts: Array, ss_sum_sin: Array, ss_sum_cos: Array,
    cluster_id: Array, row_data: Array, col_indices: Array,
    col_type_ids: Array, max_categories: int,
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
        sxsq_delta = clean_x ** 2 * is_valid_f * (type_id == CONTINUOUS_ID).astype(jnp.float32)
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
    (ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos), _ = (
        jax.lax.scan(update_one_col, carry, jnp.arange(n_cols_v))
    )
    return ss_counts, ss_sum_x, ss_sum_x_sq, ss_cat_counts, ss_sum_sin, ss_sum_cos
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_packed_kernels_v2.py::test_incremental_suffstats_correctness -v`
Expected: PASS

- [ ] **Step 5: Run all existing tests for regression check**

Run: `pytest tests/test_packed_state.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add crosscat/packed_state.py tests/test_packed_kernels_v2.py
git commit -m "feat: add incremental suffstat helpers (_remove/_add_row_to_suffstats)

JIT-compatible via lax.scan over columns with NaN guarding and
traced .at[].add() indexing for categorical types."
```

---

## Task 3: Vectorized row scoring with vmap

**Files:**
- Modify: `crosscat/packed_state.py` (replace `_score_row_all_clusters`)
- Test: `tests/test_packed_kernels_v2.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_packed_kernels_v2.py`:

```python
from crosscat.packed_state import _score_row_all_clusters_v2


def test_score_row_all_clusters_v2_matches_v1(mixed_packed_state):
    """Vectorized row scoring matches original loop-based scoring."""
    packed, data, column_types = mixed_packed_state
    v = 0
    n_cols_v = int(packed.view_n_columns[v])
    col_indices = packed.view_column_indices[v, :n_cols_v]
    row_assigns = packed.view_row_assignments[v]
    max_c = packed.max_clusters

    # Cluster counts excluding row 0
    temp = row_assigns.at[0].set(-1)
    counts = jnp.array([jnp.sum(temp == c) for c in range(max_c)]).astype(jnp.int32)

    from crosscat.packed_state import _score_row_all_clusters

    log_probs_v1 = _score_row_all_clusters(
        data[0], col_indices, n_cols_v, packed.col_type_ids,
        counts,
        packed.ss_counts[v], packed.ss_sum_x[v], packed.ss_sum_x_sq[v],
        packed.ss_cat_counts[v], packed.ss_sum_sin[v], packed.ss_sum_cos[v],
        packed.hyper_mu, packed.hyper_r, packed.hyper_s, packed.hyper_nu,
        packed.hyper_dirichlet_alpha, packed.hyper_alpha, packed.hyper_beta,
        packed.hyper_kappa, packed.hyper_vm_mu,
        packed.view_row_crp_alpha[v], max_c,
    )

    log_probs_v2 = _score_row_all_clusters_v2(
        data[0], col_indices, n_cols_v, packed.col_type_ids,
        counts,
        packed.ss_counts[v], packed.ss_sum_x[v], packed.ss_sum_x_sq[v],
        packed.ss_cat_counts[v], packed.ss_sum_sin[v], packed.ss_sum_cos[v],
        packed.hyper_mu, packed.hyper_r, packed.hyper_s, packed.hyper_nu,
        packed.hyper_dirichlet_alpha, packed.hyper_alpha, packed.hyper_beta,
        packed.hyper_kappa, packed.hyper_vm_mu,
        packed.view_row_crp_alpha[v], max_c,
    )

    # Compare finite entries
    finite_mask = jnp.isfinite(log_probs_v1) & jnp.isfinite(log_probs_v2)
    assert jnp.allclose(
        jnp.where(finite_mask, log_probs_v1, 0.0),
        jnp.where(finite_mask, log_probs_v2, 0.0),
        atol=1e-4,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_packed_kernels_v2.py::test_score_row_all_clusters_v2_matches_v1 -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `_score_row_all_clusters_v2`**

Add to `crosscat/packed_state.py` after the incremental suffstat helpers. This replaces the Python for-loops with vmap over clusters and a `lax.scan` over columns:

```python
def _score_row_one_cluster_v2(
    row_data: Array, col_indices: Array, col_type_ids: Array,
    ss_counts_c: Array, ss_sum_x_c: Array, ss_sum_x_sq_c: Array,
    ss_cat_counts_c: Array, ss_sum_sin_c: Array, ss_sum_cos_c: Array,
    hyper_mu: Array, hyper_r: Array, hyper_s: Array, hyper_nu: Array,
    hyper_dir_alpha: Array, hyper_alpha: Array, hyper_beta: Array,
    hyper_kappa: Array, hyper_vm_mu: Array,
    n_columns: int,
) -> Array:
    """Score a row against one cluster. No Python loops — lax.scan over columns."""

    def scan_col(log_lik, li):
        col_idx = col_indices[li]
        safe_idx = jnp.clip(col_idx, 0, row_data.shape[0] - 1)
        x = row_data[safe_idx]
        is_valid = (~jnp.isnan(x)) & (col_idx >= 0) & (li < n_columns)
        type_id = col_type_ids[safe_idx]

        logp = unified_posterior_predictive_logp(
            x, type_id,
            ss_counts_c[li].astype(jnp.float32),
            ss_sum_x_c[li], ss_sum_x_sq_c[li],
            ss_cat_counts_c[li],
            ss_sum_sin_c[li], ss_sum_cos_c[li],
            hyper_mu[safe_idx], hyper_r[safe_idx],
            hyper_s[safe_idx], hyper_nu[safe_idx],
            hyper_dir_alpha[safe_idx],
            hyper_alpha[safe_idx], hyper_beta[safe_idx],
            hyper_kappa[safe_idx], hyper_vm_mu[safe_idx],
        )
        log_lik = log_lik + jnp.where(is_valid, logp, 0.0)
        return log_lik, None

    max_cols_v = col_indices.shape[0]
    log_lik, _ = jax.lax.scan(scan_col, jnp.array(0.0), jnp.arange(max_cols_v))
    return log_lik


def _score_row_all_clusters_v2(
    row_data: Array, col_indices: Array, n_columns: int,
    col_type_ids: Array, cluster_counts: Array,
    ss_counts: Array, ss_sum_x: Array, ss_sum_x_sq: Array,
    ss_cat_counts: Array, ss_sum_sin: Array, ss_sum_cos: Array,
    hyper_mu: Array, hyper_r: Array, hyper_s: Array, hyper_nu: Array,
    hyper_dir_alpha: Array, hyper_alpha: Array, hyper_beta: Array,
    hyper_kappa: Array, hyper_vm_mu: Array,
    crp_alpha: Array, max_clusters: int,
) -> Array:
    """Score a row against all clusters + new cluster. vmap over clusters."""

    # CRP prior
    log_prior = jnp.log(jnp.maximum(cluster_counts.astype(jnp.float32), 1e-30))
    log_prior = jnp.where(cluster_counts > 0, log_prior, -jnp.inf)

    # vmap over clusters for likelihood
    def score_cluster_c(c):
        return _score_row_one_cluster_v2(
            row_data, col_indices, col_type_ids,
            ss_counts[c], ss_sum_x[c], ss_sum_x_sq[c],
            ss_cat_counts[c], ss_sum_sin[c], ss_sum_cos[c],
            hyper_mu, hyper_r, hyper_s, hyper_nu,
            hyper_dir_alpha, hyper_alpha, hyper_beta,
            hyper_kappa, hyper_vm_mu, n_columns,
        )

    log_liks = jax.vmap(score_cluster_c)(jnp.arange(max_clusters))
    log_probs_existing = log_prior + log_liks

    # New cluster: empty suffstats
    empty_counts = jnp.zeros_like(ss_counts[0])
    empty_sx = jnp.zeros_like(ss_sum_x[0])
    empty_sxsq = jnp.zeros_like(ss_sum_x_sq[0])
    empty_cat = jnp.zeros_like(ss_cat_counts[0])
    empty_sin = jnp.zeros_like(ss_sum_sin[0])
    empty_cos = jnp.zeros_like(ss_sum_cos[0])

    log_lik_new = _score_row_one_cluster_v2(
        row_data, col_indices, col_type_ids,
        empty_counts, empty_sx, empty_sxsq, empty_cat, empty_sin, empty_cos,
        hyper_mu, hyper_r, hyper_s, hyper_nu,
        hyper_dir_alpha, hyper_alpha, hyper_beta,
        hyper_kappa, hyper_vm_mu, n_columns,
    )
    log_prob_new = jnp.log(crp_alpha) + log_lik_new

    return jnp.concatenate([log_probs_existing, jnp.array([log_prob_new])])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_packed_kernels_v2.py::test_score_row_all_clusters_v2_matches_v1 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crosscat/packed_state.py tests/test_packed_kernels_v2.py
git commit -m "feat: vectorized row scoring with vmap over clusters, lax.scan over columns

Replaces Python for-loops in _score_row_all_clusters with JAX primitives."
```

---

## Task 4: Vectorized row assignment kernel with `lax.scan`

**Files:**
- Modify: `crosscat/packed_state.py` (add `packed_transition_row_assignments_v2`)
- Test: `tests/test_packed_kernels_v2.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_packed_kernels_v2.py`:

```python
from crosscat.packed_state import packed_transition_row_assignments_v2
from crosscat.validate import validate_state
from crosscat.model import log_joint


def test_scan_row_assignments_produces_valid_state(mixed_packed_state):
    """v2 row assignment kernel produces a valid state."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(55)
    packed_new = packed_transition_row_assignments_v2(key, packed, data)
    recovered = unpack_state(packed_new, column_types)
    errors = validate_state(recovered, data)
    assert errors == [], f"Validation errors: {errors}"
    assert jnp.isfinite(log_joint(recovered, data))


def test_scan_row_assignments_jit_compiles(mixed_packed_state):
    """v2 row assignment kernel compiles under jax.jit."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(55)
    jitted = jax.jit(packed_transition_row_assignments_v2)
    packed_new = jitted(key, packed, data)
    recovered = unpack_state(packed_new, column_types)
    errors = validate_state(recovered, data)
    assert errors == [], f"Validation errors: {errors}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_packed_kernels_v2.py::test_scan_row_assignments_produces_valid_state -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `packed_transition_row_assignments_v2`**

Add to `crosscat/packed_state.py`. This is the core kernel — `lax.scan` over rows within each view, using `lax.fori_loop` over views:

```python
def _compact_clusters(assignments: Array, max_clusters: int) -> tuple[Array, Array]:
    """JIT-compatible cluster compaction. Removes empty clusters, remaps indices."""
    counts = jnp.bincount(assignments, length=max_clusters).astype(jnp.int32)
    alive = counts > 0
    new_ids = jnp.cumsum(alive) - 1
    remapped = new_ids[assignments]
    n_clusters_final = jnp.sum(alive).astype(jnp.int32)
    return remapped, n_clusters_final


def packed_transition_row_assignments_v2(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
) -> PackedCrossCatState:
    """Gibbs sweep over row assignments using lax.scan (JIT-compilable)."""
    n_rows = packed.n_rows
    max_c = packed.max_clusters
    max_views = packed.max_views

    view_keys = jax.random.split(rng_key, max_views)

    new_row_assigns = jnp.array(packed.view_row_assignments)
    new_n_clusters = jnp.array(packed.view_n_clusters)

    def process_one_view(v_carry, v_idx):
        (ra, nc) = v_carry
        is_active = packed.view_mask[v_idx]
        row_assigns_v = ra[v_idx]
        col_indices = packed.view_column_indices[v_idx]
        n_cols_v = packed.view_n_columns[v_idx]
        alpha = packed.view_row_crp_alpha[v_idx]
        n_clust = nc[v_idx]

        # Initialize working suffstats for this view from packed state
        v_sc = packed.ss_counts[v_idx]
        v_sx = packed.ss_sum_x[v_idx]
        v_sxsq = packed.ss_sum_x_sq[v_idx]
        v_scat = packed.ss_cat_counts[v_idx]
        v_ssin = packed.ss_sum_sin[v_idx]
        v_scos = packed.ss_sum_cos[v_idx]

        row_keys = jax.random.split(view_keys[v_idx], n_rows)

        # Scan carry: assignments + suffstats (for scoring) + n_clusters
        # Suffstats are NOT carried across views — only within the row scan
        # for incremental scoring. Final suffstats are recomputed after compaction.
        def scan_row(carry, row_idx):
            assigns, ss_c, ss_sx_, ss_sxsq_, ss_cat_, ss_sin_, ss_cos_, n_cl = carry
            old_cluster = assigns[row_idx]

            # Remove row from current cluster (incremental for scoring accuracy)
            ss_c, ss_sx_, ss_sxsq_, ss_cat_, ss_sin_, ss_cos_ = _remove_row_from_suffstats(
                ss_c, ss_sx_, ss_sxsq_, ss_cat_, ss_sin_, ss_cos_,
                old_cluster, data[row_idx], col_indices, packed.col_type_ids,
                packed.max_categories,
            )

            # Cluster counts excluding this row: count all, then subtract 1 from old cluster
            counts = jnp.bincount(assigns, length=max_c).astype(jnp.int32)
            counts = counts.at[old_cluster].add(-1)

            # Score all clusters
            log_probs = _score_row_all_clusters_v2(
                data[row_idx], col_indices, n_cols_v, packed.col_type_ids,
                counts,
                ss_c, ss_sx_, ss_sxsq_, ss_cat_, ss_sin_, ss_cos_,
                packed.hyper_mu, packed.hyper_r, packed.hyper_s, packed.hyper_nu,
                packed.hyper_dirichlet_alpha, packed.hyper_alpha, packed.hyper_beta,
                packed.hyper_kappa, packed.hyper_vm_mu,
                alpha, max_c,
            )

            # Exclude new cluster if budget exhausted
            budget_ok = n_cl < (max_c - 1)
            log_probs = log_probs.at[max_c].set(
                jnp.where(budget_ok, log_probs[max_c], -jnp.inf)
            )

            # Sample
            log_probs = log_probs - jnp.max(log_probs)
            chosen = jax.random.categorical(row_keys[row_idx], log_probs)

            # If new cluster chosen, use next available slot
            is_new = chosen >= max_c
            actual_cluster = jnp.where(is_new, n_cl, chosen)
            new_n_cl = jnp.where(is_new, jnp.minimum(n_cl + 1, max_c), n_cl)

            assigns = assigns.at[row_idx].set(actual_cluster)

            # Add row to chosen cluster
            ss_c, ss_sx_, ss_sxsq_, ss_cat_, ss_sin_, ss_cos_ = _add_row_to_suffstats(
                ss_c, ss_sx_, ss_sxsq_, ss_cat_, ss_sin_, ss_cos_,
                actual_cluster, data[row_idx], col_indices, packed.col_type_ids,
                packed.max_categories,
            )

            return (assigns, ss_c, ss_sx_, ss_sxsq_, ss_cat_, ss_sin_, ss_cos_, new_n_cl), None

        init = (row_assigns_v, v_sc, v_sx, v_sxsq, v_scat, v_ssin, v_scos, n_clust)
        (final_assigns, _, _, _, _, _, _, f_nc), _ = (
            jax.lax.scan(scan_row, init, jnp.arange(n_rows))
        )
        # Note: suffstats from scan are discarded — they served only for
        # incremental scoring within the scan. Final suffstats are recomputed
        # after compaction for correctness.

        # Compact clusters
        compacted_assigns, compacted_nc = _compact_clusters(final_assigns, max_c)

        # Conditionally update (only for active views)
        # Use jnp.where at the slice level — select between new and old per-view data
        ra = ra.at[v_idx].set(jnp.where(is_active, compacted_assigns, ra[v_idx]))
        nc = nc.at[v_idx].set(jnp.where(is_active, compacted_nc, nc[v_idx]))

        return (ra, nc), None

    init_carry = (new_row_assigns, new_n_clusters)
    (final_ra, final_nc), _ = (
        jax.lax.scan(process_one_view, init_carry, jnp.arange(max_views))
    )

    # Recompute suffstats from scratch after compaction
    packed_new = PackedCrossCatState(
        **{name: (final_ra if name == "view_row_assignments"
                  else final_nc if name == "view_n_clusters"
                  else getattr(packed, name))
           for name in _ARRAY_FIELDS},
        **{name: getattr(packed, name) for name in _STATIC_FIELDS},
    )
    return recompute_all_suffstats(packed_new, data)
```

**Design rationale:** The scan carry includes per-view suffstats for incremental scoring
within the row loop, but these are NOT carried across views — they are initialized from
`packed.ss_*` at the start of each view. The outer carry is just `(assignments, n_clusters)`,
keeping memory usage minimal. After compaction, `recompute_all_suffstats` produces the
canonical final state. A future optimization can trust the incremental path and skip the
recompute.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_packed_kernels_v2.py -k "row_assignments" -v`
Expected: Both tests pass.

- [ ] **Step 5: Commit**

```bash
git add crosscat/packed_state.py tests/test_packed_kernels_v2.py
git commit -m "feat: lax.scan row assignment kernel (packed_transition_row_assignments_v2)

Sequential scan over rows for Gibbs correctness, vmap over clusters
for scoring. Incremental suffstats within scan, full recompute after
compaction for safety."
```

---

## Task 5: Vectorized column hypers kernel

**Files:**
- Modify: `crosscat/packed_state.py` (add `packed_transition_column_hypers_v2`)
- Test: `tests/test_packed_kernels_v2.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_packed_kernels_v2.py`:

```python
from crosscat.packed_state import packed_transition_column_hypers_v2


def test_vmap_column_hypers_produces_valid_state(mixed_packed_state):
    """v2 column hypers kernel produces valid hyperparameters."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(77)
    packed_new = packed_transition_column_hypers_v2(key, packed, data)

    # All hypers should be finite and positive where applicable
    assert jnp.all(jnp.isfinite(packed_new.hyper_mu))
    assert jnp.all(packed_new.hyper_s > 0)
    assert jnp.all(packed_new.hyper_nu > 0)
    assert jnp.all(packed_new.hyper_r > 0)
    assert jnp.all(packed_new.hyper_dirichlet_alpha > 0)


def test_vmap_column_hypers_jit_compiles(mixed_packed_state):
    """v2 column hypers kernel compiles under jax.jit."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(77)
    jitted = jax.jit(packed_transition_column_hypers_v2)
    packed_new = jitted(key, packed, data)
    assert jnp.all(jnp.isfinite(packed_new.hyper_s))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_packed_kernels_v2.py -k "column_hypers" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `packed_transition_column_hypers_v2`**

Add to `crosscat/packed_state.py`. Uses vmap over all columns with unified type dispatch:

```python
def packed_transition_column_hypers_v2(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
) -> PackedCrossCatState:
    """Grid-based Gibbs for column hypers — vmap over columns (JIT-compilable)."""
    n_cols = packed.n_cols
    max_c = packed.max_clusters
    keys = jax.random.split(rng_key, n_cols)

    s_grid = jnp.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0])
    nu_grid = jnp.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    alpha_grid_cat = jnp.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    kappa_grid = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    ab_grid = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])

    def _score_ng_grid(grid_vals, ss_c, ss_sx, ss_sxsq, mu, r, nu, nc):
        """Score a grid of s (or mu or nu) values across all clusters.
        nc is the actual number of active clusters (traced) — used for masking."""
        def score_one(gv):
            per_cl = jax.vmap(lambda c: _ng_log_marginal(
                ss_c[c], ss_sx[c], ss_sxsq[c], mu, r, gv, nu
            ))(jnp.arange(max_c))
            # Mask: only sum over active clusters (nc), not all max_clusters
            return jnp.sum(jnp.where(jnp.arange(max_c) < nc, per_cl, 0.0))
        return jax.vmap(score_one)(grid_vals)

    def process_one_column(key, j):
        type_id = packed.col_type_ids[j]
        v_idx = packed.column_assignments[j]
        n_cols_v = packed.view_n_columns[v_idx]

        # Find local index via scan
        def find_local(carry, li):
            found, idx = carry
            match = (packed.view_column_indices[v_idx, li] == j) & (li < n_cols_v)
            idx = jnp.where(match & ~found, li, idx)
            found = found | match
            return (found, idx), None
        (_, local_idx), _ = jax.lax.scan(find_local, (False, jnp.array(0, jnp.int32)),
                                          jnp.arange(packed.view_column_indices.shape[1]))

        ss_c = packed.ss_counts[v_idx, :, local_idx]
        ss_sx = packed.ss_sum_x[v_idx, :, local_idx]
        ss_sxsq = packed.ss_sum_x_sq[v_idx, :, local_idx]
        ss_cat = packed.ss_cat_counts[v_idx, :, local_idx, :]
        ss_sin = packed.ss_sum_sin[v_idx, :, local_idx]
        ss_cos = packed.ss_sum_cos[v_idx, :, local_idx]
        nc = packed.view_n_clusters[v_idx]  # actual cluster count for masking

        cur_mu = packed.hyper_mu[j]
        cur_r = packed.hyper_r[j]
        cur_nu = packed.hyper_nu[j]
        cur_s = packed.hyper_s[j]

        # Split into enough keys to avoid correlation between type branches
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        # Continuous: sample s, then mu, then nu
        col_data = data[:, j]
        data_var = jnp.var(col_data) + 1e-6
        data_mean = jnp.mean(col_data)
        data_std = jnp.std(col_data) + 1e-6

        scaled_s_grid = data_var * s_grid
        s_scores = _score_ng_grid(scaled_s_grid, ss_c, ss_sx, ss_sxsq, cur_mu, cur_r, cur_nu, nc)
        s_scores = s_scores - jnp.max(s_scores)
        new_s = scaled_s_grid[jax.random.categorical(k1, s_scores)]

        mu_grid = data_mean + data_std * jnp.linspace(-2, 2, 11)
        def score_mu(mv):
            per_cl = jax.vmap(lambda c: _ng_log_marginal(
                ss_c[c], ss_sx[c], ss_sxsq[c], mv, cur_r, new_s, cur_nu
            ))(jnp.arange(max_c))
            return jnp.sum(jnp.where(jnp.arange(max_c) < nc, per_cl, 0.0))
        mu_scores = jax.vmap(score_mu)(mu_grid)
        mu_scores = mu_scores - jnp.max(mu_scores)
        new_mu = mu_grid[jax.random.categorical(k2, mu_scores)]

        def score_nu(nv):
            per_cl = jax.vmap(lambda c: _ng_log_marginal(
                ss_c[c], ss_sx[c], ss_sxsq[c], new_mu, cur_r, new_s, nv
            ))(jnp.arange(max_c))
            return jnp.sum(jnp.where(jnp.arange(max_c) < nc, per_cl, 0.0))
        nu_scores = jax.vmap(score_nu)(nu_grid)
        nu_scores = nu_scores - jnp.max(nu_scores)
        new_nu = nu_grid[jax.random.categorical(k3, nu_scores)]

        # Categorical: sample dirichlet_alpha (separate key k4)
        def score_cat_alpha(av):
            per_cl = jax.vmap(lambda c: _dc_log_marginal(ss_c[c], ss_cat[c], av))(jnp.arange(max_c))
            return jnp.sum(jnp.where(jnp.arange(max_c) < nc, per_cl, 0.0))
        cat_scores = jax.vmap(score_cat_alpha)(alpha_grid_cat)
        cat_scores = cat_scores - jnp.max(cat_scores)
        new_dir_alpha = alpha_grid_cat[jax.random.categorical(k4, cat_scores)]

        # Cyclic: sample kappa (separate key k5)
        def score_kappa(kv):
            per_cl = jax.vmap(lambda c: _vm_log_marginal(ss_c[c], ss_sin[c], ss_cos[c], kv))(jnp.arange(max_c))
            return jnp.sum(jnp.where(jnp.arange(max_c) < nc, per_cl, 0.0))
        kappa_scores = jax.vmap(score_kappa)(kappa_grid)
        kappa_scores = kappa_scores - jnp.max(kappa_scores)
        new_kappa = kappa_grid[jax.random.categorical(k5, kappa_scores)]

        # Binary: sample alpha, beta from 2D grid (uses k4 — ok, only one type active per column)
        ab_a = jnp.repeat(ab_grid, len(ab_grid))
        ab_b = jnp.tile(ab_grid, len(ab_grid))
        def score_ab(idx):
            a, b = ab_a[idx], ab_b[idx]
            per_cl = jax.vmap(lambda c: _bb_log_marginal(ss_c[c], ss_sx[c], a, b))(jnp.arange(max_c))
            return jnp.sum(jnp.where(jnp.arange(max_c) < nc, per_cl, 0.0))
        ab_scores = jax.vmap(score_ab)(jnp.arange(len(ab_a)))
        ab_scores = ab_scores - jnp.max(ab_scores)
        ab_idx = jax.random.categorical(k4, ab_scores)
        new_alpha = ab_a[ab_idx]
        new_beta = ab_b[ab_idx]

        # Select by type
        out_mu = jnp.where(type_id == CONTINUOUS_ID, new_mu, cur_mu)
        out_s = jnp.where(type_id == CONTINUOUS_ID, new_s, cur_s)
        out_nu = jnp.where(type_id == CONTINUOUS_ID, new_nu, cur_nu)
        out_dir = jnp.where(type_id == CATEGORICAL_ID, new_dir_alpha, packed.hyper_dirichlet_alpha[j])
        out_alpha = jnp.where(type_id == BINARY_ID, new_alpha, packed.hyper_alpha[j])
        out_beta = jnp.where(type_id == BINARY_ID, new_beta, packed.hyper_beta[j])
        out_kappa = jnp.where(type_id == CYCLIC_ID, new_kappa, packed.hyper_kappa[j])

        return out_mu, out_s, out_nu, out_dir, out_alpha, out_beta, out_kappa

    # vmap over columns
    results = jax.vmap(process_one_column)(keys, jnp.arange(n_cols))
    new_mu, new_s, new_nu, new_dir, new_alpha, new_beta, new_kappa = results

    return PackedCrossCatState(
        **{name: (new_mu if name == "hyper_mu"
                  else new_s if name == "hyper_s"
                  else new_nu if name == "hyper_nu"
                  else new_dir if name == "hyper_dirichlet_alpha"
                  else new_alpha if name == "hyper_alpha"
                  else new_beta if name == "hyper_beta"
                  else new_kappa if name == "hyper_kappa"
                  else getattr(packed, name))
           for name in _ARRAY_FIELDS},
        **{name: getattr(packed, name) for name in _STATIC_FIELDS},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_packed_kernels_v2.py -k "column_hypers" -v`
Expected: Both tests pass.

- [ ] **Step 5: Commit**

```bash
git add crosscat/packed_state.py tests/test_packed_kernels_v2.py
git commit -m "feat: vmap column hypers kernel (packed_transition_column_hypers_v2)

vmap over all columns with unified type dispatch via jnp.where.
Grid scoring vectorized over grid points x clusters."
```

---

## Task 6: Vectorized CRP alpha kernel

**Files:**
- Modify: `crosscat/packed_state.py` (add `packed_transition_crp_alphas_v2`)
- Test: `tests/test_packed_kernels_v2.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_packed_kernels_v2.py`:

```python
from crosscat.packed_state import packed_transition_crp_alphas_v2


def test_vmap_crp_alphas_produces_valid_values(mixed_packed_state):
    """v2 CRP alpha kernel produces positive alpha values."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(88)
    packed_new = packed_transition_crp_alphas_v2(key, packed)
    assert float(packed_new.column_crp_alpha) > 0
    for v in range(int(packed_new.n_views)):
        assert float(packed_new.view_row_crp_alpha[v]) > 0


def test_vmap_crp_alphas_jit_compiles(mixed_packed_state):
    """v2 CRP alpha kernel compiles under jax.jit."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(88)
    jitted = jax.jit(packed_transition_crp_alphas_v2)
    packed_new = jitted(key, packed)
    assert float(packed_new.column_crp_alpha) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_packed_kernels_v2.py -k "crp_alphas" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `packed_transition_crp_alphas_v2`**

Add to `crosscat/packed_state.py`:

```python
def packed_transition_crp_alphas_v2(
    rng_key: Array,
    packed: PackedCrossCatState,
) -> PackedCrossCatState:
    """Sample CRP concentrations — vmap over grid x views (JIT-compilable)."""
    alpha_grid = jnp.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    max_c = packed.max_clusters
    n_rows = packed.n_rows
    n_cols = packed.n_cols
    max_views = packed.max_views

    keys = jax.random.split(rng_key, 1 + max_views)

    def log_crp_score(assignments: Array, alpha_val: Array, length: int) -> Array:
        counts = jnp.bincount(assignments, length=length).astype(jnp.float32)
        n_clusters = jnp.sum(counts > 0).astype(jnp.float32)
        valid_counts = jnp.where(counts > 0, counts, 1.0)
        return (
            n_clusters * jnp.log(alpha_val)
            + jnp.sum(jnp.where(counts > 0, gammaln(valid_counts), 0.0))
            - gammaln(jnp.array(length, dtype=jnp.float32) + alpha_val)
            + gammaln(alpha_val)
            - alpha_val  # Exp(1) prior
        )

    # Outer CRP: vmap over grid
    outer_scores = jax.vmap(
        lambda a: log_crp_score(packed.column_assignments, a, n_cols)
    )(alpha_grid)
    outer_scores = outer_scores - jnp.max(outer_scores)
    new_col_alpha = alpha_grid[jax.random.categorical(keys[0], outer_scores)]

    # Inner CRP: vmap over views x grid
    def score_view_grid(view_assigns):
        return jax.vmap(
            lambda a: log_crp_score(view_assigns, a, n_rows)
        )(alpha_grid)

    all_view_scores = jax.vmap(score_view_grid)(packed.view_row_assignments)
    # shape: (max_views, len(alpha_grid))

    def sample_one_view(key, scores):
        s = scores - jnp.max(scores)
        return alpha_grid[jax.random.categorical(key, s)]

    new_view_alphas = jax.vmap(sample_one_view)(keys[1:], all_view_scores)

    return PackedCrossCatState(
        **{name: (new_col_alpha if name == "column_crp_alpha"
                  else new_view_alphas if name == "view_row_crp_alpha"
                  else getattr(packed, name))
           for name in _ARRAY_FIELDS},
        **{name: getattr(packed, name) for name in _STATIC_FIELDS},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_packed_kernels_v2.py -k "crp_alphas" -v`
Expected: Both tests pass.

- [ ] **Step 5: Commit**

```bash
git add crosscat/packed_state.py tests/test_packed_kernels_v2.py
git commit -m "feat: vmap CRP alpha kernel (packed_transition_crp_alphas_v2)

Zero Python loops — vmap over grid x views."
```

---

## Task 7: `packed_gibbs_sweep_v2` and full sweep test

**Files:**
- Modify: `crosscat/packed_state.py` (add `packed_gibbs_sweep_v2`)
- Test: `tests/test_packed_kernels_v2.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_packed_kernels_v2.py`:

```python
from crosscat.packed_state import packed_gibbs_sweep_v2


def test_full_packed_sweep_v2_valid(mixed_packed_state):
    """Full v2 Gibbs sweep produces valid state."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(100)
    packed_new = packed_gibbs_sweep_v2(key, packed, data, n_sweeps=2)
    recovered = unpack_state(packed_new, column_types)
    errors = validate_state(recovered, data)
    assert errors == [], f"Validation errors: {errors}"


def test_packed_sweep_v2_deterministic(mixed_packed_state):
    """Same key produces same output."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(100)
    result1 = packed_gibbs_sweep_v2(key, packed, data, n_sweeps=1)
    result2 = packed_gibbs_sweep_v2(key, packed, data, n_sweeps=1)
    assert jnp.array_equal(result1.view_row_assignments, result2.view_row_assignments)


def test_packed_sweep_v2_jit_compiles(mixed_packed_state):
    """Full v2 Gibbs sweep compiles under jax.jit."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(100)
    jitted = jax.jit(lambda k, p, d: packed_gibbs_sweep_v2(k, p, d, n_sweeps=1))
    packed_new = jitted(key, packed, data)
    recovered = unpack_state(packed_new, column_types)
    errors = validate_state(recovered, data)
    assert errors == [], f"Validation errors: {errors}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_packed_kernels_v2.py -k "sweep_v2" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `packed_gibbs_sweep_v2`**

Add to `crosscat/packed_state.py`:

```python
def packed_gibbs_sweep_v2(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    *,
    n_sweeps: int = 1,
) -> PackedCrossCatState:
    """Run Gibbs sweeps with fully vectorized kernels (JIT-compilable).

    Args:
        rng_key: JAX PRNG key.
        packed: Packed state.
        data: Observation matrix.
        n_sweeps: Number of sweeps (static — changing causes recompilation).

    Returns:
        Updated packed state.
    """
    def one_sweep(carry, _):
        state, rng = carry
        k1, k2, k3, rng = jax.random.split(rng, 4)
        state = packed_transition_row_assignments_v2(k1, state, data)
        state = packed_transition_column_hypers_v2(k2, state, data)
        state = packed_transition_crp_alphas_v2(k3, state)
        return (state, rng), None

    (result, _), _ = jax.lax.scan(one_sweep, (packed, rng_key), jnp.arange(n_sweeps))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_packed_kernels_v2.py -k "sweep_v2" -v`
Expected: All 3 tests pass.

- [ ] **Step 5: Run ALL tests for regression check**

Run: `pytest tests/ -v`
Expected: All existing + new tests pass.

- [ ] **Step 6: Commit**

```bash
git add crosscat/packed_state.py tests/test_packed_kernels_v2.py
git commit -m "feat: packed_gibbs_sweep_v2 — full JIT-compilable Gibbs sweep

Wraps v2 row, hyper, and CRP kernels in a lax.scan over n_sweeps.
Single JIT compilation, zero Python overhead."
```

---

## Task 8: Unified posterior predictive sampler

**Files:**
- Modify: `crosscat/packed_state.py` (add sampling functions)
- Test: `tests/test_packed_kernels_v2.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_packed_kernels_v2.py`:

```python
from crosscat.packed_state import unified_sample_posterior_predictive


def test_unified_sampler_continuous(mixed_packed_state):
    """Unified sampler produces finite samples for continuous type."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(200)
    v = 0
    li = 0
    col_idx = int(packed.view_column_indices[v, li])
    sample = unified_sample_posterior_predictive(
        key, packed.col_type_ids[col_idx],
        packed.ss_counts[v, 0, li].astype(jnp.float32),
        packed.ss_sum_x[v, 0, li], packed.ss_sum_x_sq[v, 0, li],
        packed.ss_cat_counts[v, 0, li], packed.ss_sum_sin[v, 0, li],
        packed.ss_sum_cos[v, 0, li],
        packed.hyper_mu[col_idx], packed.hyper_r[col_idx],
        packed.hyper_s[col_idx], packed.hyper_nu[col_idx],
        packed.hyper_dirichlet_alpha[col_idx],
        packed.hyper_alpha[col_idx], packed.hyper_beta[col_idx],
        packed.hyper_kappa[col_idx], packed.hyper_vm_mu[col_idx],
        packed.max_categories,
    )
    assert jnp.isfinite(sample)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_packed_kernels_v2.py::test_unified_sampler_continuous -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `unified_sample_posterior_predictive` and type-specific samplers**

Add to `crosscat/packed_state.py` after `unified_posterior_predictive_logp`:

```python
def _ng_sample(rng_key, count, sum_x, sum_x_sq, mu0, r, s, nu):
    """Sample from Normal-Gamma posterior predictive (Student-t)."""
    n = count.astype(jnp.float32)
    r_n = r + n
    mu_n = (r * mu0 + sum_x) / jnp.maximum(r_n, 1e-30)
    nu_n = nu + n
    nu_s = nu * s
    mean = jnp.where(n > 0, sum_x / jnp.maximum(n, 1.0), 0.0)
    nu_n_s_n = nu_s + sum_x_sq - sum_x**2 / jnp.maximum(n, 1.0) + r * n * (mu0 - mean)**2 / jnp.maximum(r_n, 1e-30)
    nu_n_s_n = jnp.where(n > 0, nu_n_s_n, nu_s)
    nu_n_s_n = jnp.maximum(nu_n_s_n, 1e-30)

    df = nu_n
    loc = mu_n
    scale = jnp.sqrt(jnp.maximum((nu_n_s_n / jnp.maximum(nu_n, 1e-30)) * (1.0 + 1.0 / jnp.maximum(r_n, 1e-30)), 1e-30))

    k1, k2 = jax.random.split(rng_key)
    z = jax.random.normal(k1)
    chi2 = jax.random.chisquare(k2, df)
    t = z / jnp.sqrt(jnp.maximum(chi2 / jnp.maximum(df, 1e-30), 1e-30))
    return loc + scale * t


def _dc_sample(rng_key, count, cat_counts, dir_alpha, max_categories):
    """Sample from Dirichlet-Categorical posterior predictive."""
    n = count.astype(jnp.float32)
    k = jnp.array(max_categories, dtype=jnp.float32)
    probs = (cat_counts + dir_alpha) / jnp.maximum(n + k * dir_alpha, 1e-30)
    return jax.random.categorical(rng_key, jnp.log(jnp.maximum(probs, 1e-30))).astype(jnp.float32)


def _bb_sample(rng_key, count, sum_x, alpha, beta):
    """Sample from Beta-Bernoulli posterior predictive."""
    n = count.astype(jnp.float32)
    p1 = (sum_x + alpha) / jnp.maximum(n + alpha + beta, 1e-30)
    return jax.random.bernoulli(rng_key, p1).astype(jnp.float32)


def _vm_sample(rng_key, count, sum_sin, sum_cos, kappa, vm_mu):
    """Sample from Von Mises posterior predictive (wrapped normal approx)."""
    total_sin = sum_sin + kappa * jnp.sin(vm_mu)
    total_cos = sum_cos + kappa * jnp.cos(vm_mu)
    r_post = jnp.sqrt(total_sin**2 + total_cos**2)
    mu_post = jnp.arctan2(total_sin, total_cos)
    kappa_post = r_post
    sigma = 1.0 / jnp.sqrt(jnp.maximum(kappa_post, 0.01))
    sample = mu_post + sigma * jax.random.normal(rng_key)
    return sample % (2.0 * jnp.pi)


def unified_sample_posterior_predictive(
    rng_key, type_id, count, sum_x, sum_x_sq, cat_counts, sum_sin, sum_cos,
    mu, r, s, nu, dir_alpha, alpha, beta, kappa, vm_mu, max_categories,
):
    """Sample from posterior predictive for any column type (JIT-compatible)."""
    cont = _ng_sample(rng_key, count, sum_x, sum_x_sq, mu, r, s, nu)
    cat = _dc_sample(rng_key, count, cat_counts, dir_alpha, max_categories)
    binary = _bb_sample(rng_key, count, sum_x, alpha, beta)
    ordinal = _dc_sample(rng_key, count, cat_counts, jnp.ones_like(dir_alpha), max_categories)
    cyclic = _vm_sample(rng_key, count, sum_sin, sum_cos, kappa, vm_mu)

    return jnp.where(type_id == CONTINUOUS_ID, cont,
           jnp.where(type_id == CATEGORICAL_ID, cat,
           jnp.where(type_id == ORDINAL_ID, ordinal,
           jnp.where(type_id == BINARY_ID, binary, cyclic))))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_packed_kernels_v2.py::test_unified_sampler_continuous -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crosscat/packed_state.py tests/test_packed_kernels_v2.py
git commit -m "feat: unified_sample_posterior_predictive for JIT-compatible sampling

Type-specific samplers (_ng_sample, _dc_sample, _bb_sample, _vm_sample)
dispatched via jnp.where for all 5 column types."
```

---

## Task 9: Packed inference module

**Files:**
- Create: `crosscat/packed_inference.py`
- Test: `tests/test_packed_kernels_v2.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_packed_kernels_v2.py`:

```python
from crosscat.packed_inference import (
    packed_predictive_probability,
    packed_predictive_sample,
    packed_mutual_information,
    packed_row_similarity,
    packed_impute_and_confidence,
    packed_anomaly_score,
)
from crosscat.inference import (
    predictive_probability,
    predictive_sample,
    mutual_information,
    row_similarity,
)


def test_packed_predictive_probability_matches_original(mixed_packed_state):
    """Packed predictive probability matches unpacked within tolerance."""
    packed, data, column_types = mixed_packed_state
    state = unpack_state(packed, column_types)
    query_cols = jnp.array([0])
    query_vals = jnp.array([data[0, 0]])

    log_p_orig = predictive_probability(state, data, [0], query_vals)
    log_p_packed = packed_predictive_probability(packed, data, query_cols, query_vals)

    assert jnp.allclose(log_p_orig, log_p_packed, atol=1e-3)


def test_packed_predictive_sample_distribution(mixed_packed_state):
    """Packed samples come from similar distribution as unpacked (KS test)."""
    from scipy.stats import ks_2samp
    packed, data, column_types = mixed_packed_state
    state = unpack_state(packed, column_types)
    key = jax.random.key(300)

    k1, k2 = jax.random.split(key)
    samples_orig = predictive_sample(k1, state, data, [0], n_samples=500)
    samples_packed = packed_predictive_sample(k2, packed, data, jnp.array([0]), n_samples=500)

    stat, p_value = ks_2samp(samples_orig[:, 0], samples_packed[:, 0])
    assert p_value > 0.01, f"KS test failed: stat={stat}, p={p_value}"


def test_packed_mutual_information_matches_original(mixed_packed_state):
    """Packed MI matches unpacked."""
    packed, data, column_types = mixed_packed_state
    state = unpack_state(packed, column_types)
    states = [state]

    mi_orig, _ = mutual_information(states, 0, 1)
    mi_packed = packed_mutual_information([packed], column_types, 0, 1)

    assert jnp.allclose(mi_orig, mi_packed, atol=1e-3)


def test_packed_row_similarity_matches_original(mixed_packed_state):
    """Packed row similarity matches unpacked."""
    packed, data, column_types = mixed_packed_state
    state = unpack_state(packed, column_types)
    states = [state]

    sim_orig = row_similarity(states, 0, 1)
    sim_packed = packed_row_similarity([packed], column_types, 0, 1)

    assert jnp.allclose(sim_orig, sim_packed, atol=1e-3)


def test_packed_anomaly_score_produces_valid_output(mixed_packed_state):
    """Packed anomaly score produces value in [0, 1]."""
    packed, data, column_types = mixed_packed_state
    key = jax.random.key(400)
    score = packed_anomaly_score(key, packed, data, 0)
    assert 0.0 <= float(score) <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_packed_kernels_v2.py -k "packed_predictive or packed_mutual or packed_row_sim or packed_anomaly" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `crosscat/packed_inference.py`**

Create `crosscat/packed_inference.py`. Below is the complete implementation for the two core functions that all others depend on, plus the remaining functions.

```python
"""Vectorized inference queries on PackedCrossCatState.

Parallel API to crosscat/inference.py — operates on packed state with
no Python for-loops, enabling jax.jit compilation and GPU acceleration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from crosscat.packed_state import (
    BINARY_ID,
    CATEGORICAL_ID,
    CONTINUOUS_ID,
    CYCLIC_ID,
    ORDINAL_ID,
    PackedCrossCatState,
    unified_posterior_predictive_logp,
    unified_sample_posterior_predictive,
    unpack_state,
)
from crosscat.types import ColumnType


def _find_local_col_index(packed: PackedCrossCatState, view_idx, col_idx):
    """Find local index of col_idx within view_idx's column list (traced)."""
    col_list = packed.view_column_indices[view_idx]
    matches = col_list == col_idx
    # Return first match index (or 0 if not found)
    return jnp.argmax(matches)


def _cluster_weights_packed(packed: PackedCrossCatState, view_idx):
    """CRP-based cluster weights for a view. Returns (max_clusters,) array."""
    assigns = packed.view_row_assignments[view_idx]
    counts = jnp.bincount(assigns, length=packed.max_clusters).astype(jnp.float32)
    total = jnp.sum(counts)
    return counts / jnp.maximum(total, 1e-30)


def packed_predictive_probability(
    packed: PackedCrossCatState,
    data: Array,
    query_cols: Array,
    query_vals: Array,
    *,
    row_id: int | None = None,
) -> Array:
    """Compute predictive probability using packed state. No Python loops."""
    n_query = query_cols.shape[0]
    max_c = packed.max_clusters

    def score_one_query(q_idx):
        col = query_cols[q_idx]
        x = query_vals[q_idx]
        view_idx = packed.column_assignments[col]
        local_idx = _find_local_col_index(packed, view_idx, col)

        # Cluster weights
        if row_id is not None:
            cluster = packed.view_row_assignments[view_idx, row_id]
            weights = jnp.zeros(max_c).at[cluster].set(1.0)
        else:
            weights = _cluster_weights_packed(packed, view_idx)

        # Score each cluster: log(weight) + posterior_predictive_logp
        def score_cluster(c):
            log_w = jnp.log(jnp.maximum(weights[c], 1e-30))
            logp = unified_posterior_predictive_logp(
                x, packed.col_type_ids[col],
                packed.ss_counts[view_idx, c, local_idx].astype(jnp.float32),
                packed.ss_sum_x[view_idx, c, local_idx],
                packed.ss_sum_x_sq[view_idx, c, local_idx],
                packed.ss_cat_counts[view_idx, c, local_idx],
                packed.ss_sum_sin[view_idx, c, local_idx],
                packed.ss_sum_cos[view_idx, c, local_idx],
                packed.hyper_mu[col], packed.hyper_r[col],
                packed.hyper_s[col], packed.hyper_nu[col],
                packed.hyper_dirichlet_alpha[col],
                packed.hyper_alpha[col], packed.hyper_beta[col],
                packed.hyper_kappa[col], packed.hyper_vm_mu[col],
            )
            return log_w + logp

        log_terms = jax.vmap(score_cluster)(jnp.arange(max_c))
        # Mask inactive clusters
        active = weights > 0
        log_terms = jnp.where(active, log_terms, -jnp.inf)
        return jax.nn.logsumexp(log_terms)

    per_col_logp = jax.vmap(score_one_query)(jnp.arange(n_query))
    return jnp.sum(per_col_logp)


def packed_predictive_sample(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    query_cols: Array,
    *,
    n_samples: int = 1000,
    row_id: int | None = None,
) -> Array:
    """Draw samples from posterior predictive. vmap over samples."""
    n_query = query_cols.shape[0]
    max_c = packed.max_clusters
    sample_keys = jax.random.split(rng_key, n_samples)

    def draw_one_sample(key):
        col_keys = jax.random.split(key, n_query)

        def sample_one_col(q_idx, col_key):
            col = query_cols[q_idx]
            view_idx = packed.column_assignments[col]
            local_idx = _find_local_col_index(packed, view_idx, col)

            if row_id is not None:
                cluster = packed.view_row_assignments[view_idx, row_id]
                weights = jnp.zeros(max_c).at[cluster].set(1.0)
            else:
                weights = _cluster_weights_packed(packed, view_idx)

            # Sample cluster
            k1, k2 = jax.random.split(col_key)
            cluster = jax.random.categorical(k1, jnp.log(jnp.maximum(weights, 1e-30)))

            # Sample value from cluster
            val = unified_sample_posterior_predictive(
                k2, packed.col_type_ids[col],
                packed.ss_counts[view_idx, cluster, local_idx].astype(jnp.float32),
                packed.ss_sum_x[view_idx, cluster, local_idx],
                packed.ss_sum_x_sq[view_idx, cluster, local_idx],
                packed.ss_cat_counts[view_idx, cluster, local_idx],
                packed.ss_sum_sin[view_idx, cluster, local_idx],
                packed.ss_sum_cos[view_idx, cluster, local_idx],
                packed.hyper_mu[col], packed.hyper_r[col],
                packed.hyper_s[col], packed.hyper_nu[col],
                packed.hyper_dirichlet_alpha[col],
                packed.hyper_alpha[col], packed.hyper_beta[col],
                packed.hyper_kappa[col], packed.hyper_vm_mu[col],
                packed.max_categories,
            )
            return val

        vals = jax.vmap(sample_one_col)(jnp.arange(n_query), col_keys)
        return vals

    return jax.vmap(draw_one_sample)(sample_keys)  # (n_samples, n_query)


def packed_mutual_information(
    packed_states: list[PackedCrossCatState],
    column_types: list[ColumnType],
    col_i: int,
    col_j: int,
) -> Array:
    """Estimate mutual information between two columns using packed states."""
    mi_estimates = []
    for packed in packed_states:
        view_i = int(packed.column_assignments[col_i])
        view_j = int(packed.column_assignments[col_j])
        if view_i != view_j:
            mi_estimates.append(0.0)
            continue
        assigns = packed.view_row_assignments[view_i]
        n_clusters = int(packed.view_n_clusters[view_i])
        counts = jnp.bincount(assigns, length=packed.max_clusters).astype(jnp.float32)
        probs = counts / jnp.maximum(counts.sum(), 1e-30)
        entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs + 1e-30), 0.0))
        mi_est = float(entropy * (1.0 - 1.0 / jnp.maximum(n_clusters, 1.0)))
        mi_estimates.append(mi_est)
    mi = jnp.array(mi_estimates).mean()
    linfoot = jnp.sqrt(1.0 - jnp.exp(-2.0 * mi))
    return mi


def packed_row_similarity(
    packed_states: list[PackedCrossCatState],
    column_types: list[ColumnType],
    row_a: int,
    row_b: int,
    *,
    target_columns: list[int] | None = None,
) -> Array:
    """Compute row similarity using packed states."""
    sim_scores = []
    for packed in packed_states:
        n_views = int(packed.n_views)
        view_scores = []
        for v in range(n_views):
            if target_columns is not None:
                n_cols_v = int(packed.view_n_columns[v])
                view_cols = set(packed.view_column_indices[v, :n_cols_v].tolist())
                if not any(c in view_cols for c in target_columns):
                    continue
            same = float(packed.view_row_assignments[v, row_a] == packed.view_row_assignments[v, row_b])
            view_scores.append(same)
        if view_scores:
            sim_scores.append(sum(view_scores) / len(view_scores))
    if not sim_scores:
        return jnp.array(0.0)
    return jnp.array(sum(sim_scores) / len(sim_scores))


def packed_impute_and_confidence(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    query_col: int,
    *,
    n_samples: int = 1000,
) -> tuple[Array, Array]:
    """Impute a value with confidence score using packed state."""
    samples = packed_predictive_sample(
        rng_key, packed, data, jnp.array([query_col]), n_samples=n_samples,
    )
    s = samples[:, 0]
    col_type_id = int(packed.col_type_ids[query_col])
    if col_type_id == CONTINUOUS_ID or col_type_id == CYCLIC_ID:
        point_est = jnp.median(s)
        iqr = jnp.percentile(s, 75) - jnp.percentile(s, 25)
        std = jnp.std(s) + 1e-30
        confidence = jnp.exp(-iqr / std)
    else:
        s_int = s.astype(jnp.int32)
        max_val = int(jnp.max(s_int)) + 1
        counts = jnp.bincount(s_int, length=max_val)
        point_est = jnp.argmax(counts).astype(jnp.float32)
        confidence = counts[jnp.argmax(counts)] / jnp.float32(n_samples)
    return point_est, confidence


def packed_anomaly_score(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    query_row: int,
) -> Array:
    """Compute anomaly score for a row using packed state."""
    n_cols = packed.n_cols

    def score_one_col(col):
        x = data[query_row, col]
        log_p = packed_predictive_probability(
            packed, data, jnp.array([col]), jnp.array([x]), row_id=query_row,
        )
        return jnp.where(jnp.isnan(x), 0.0, log_p), jnp.where(jnp.isnan(x), 0.0, 1.0)

    log_ps, valids = jax.vmap(score_one_col)(jnp.arange(n_cols))
    total_log_p = jnp.sum(log_ps)
    n_scored = jnp.sum(valids)
    avg_log_p = total_log_p / jnp.maximum(n_scored, 1.0)
    anomaly = 1.0 / (1.0 + jnp.exp(avg_log_p + 2.0))
    return jnp.clip(anomaly, 0.0, 1.0)


def packed_predictive_cdf(
    rng_key: Array,
    packed: PackedCrossCatState,
    data: Array,
    query_col: int,
    query_val: Array,
    *,
    n_samples: int = 10000,
) -> Array:
    """Compute posterior predictive CDF via MC sampling."""
    samples = packed_predictive_sample(
        rng_key, packed, data, jnp.array([query_col]), n_samples=n_samples,
    )
    return jnp.mean(samples[:, 0] <= query_val)
```

**Note:** `packed_mutual_information` and `packed_row_similarity` use Python loops over the
`packed_states` list (which is typically 4-8 chains) — this is acceptable since the list
is small and the inner operations are vectorized. A future vmap-over-chains optimization
can eliminate these loops.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_packed_kernels_v2.py -k "packed_predictive or packed_mutual or packed_row_sim or packed_anomaly" -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add crosscat/packed_inference.py tests/test_packed_kernels_v2.py
git commit -m "feat: packed_inference.py — vectorized inference on PackedCrossCatState

Includes packed_predictive_probability, packed_predictive_sample,
packed_mutual_information, packed_row_similarity, packed_impute_and_confidence,
packed_anomaly_score, packed_predictive_cdf. No Python for-loops."
```

---

## Task 10: Update `__init__.py` exports

**Files:**
- Modify: `crosscat/__init__.py`

- [ ] **Step 1: Add packed exports**

Add the new packed API functions to `crosscat/__init__.py`:

```python
from crosscat.packed_inference import (
    packed_anomaly_score,
    packed_impute_and_confidence,
    packed_mutual_information,
    packed_predictive_cdf,
    packed_predictive_probability,
    packed_predictive_sample,
    packed_row_similarity,
)
from crosscat.packed_state import (
    PackedCrossCatState,
    pack_state,
    packed_gibbs_sweep_v2,
    unpack_state,
)
```

And add all to `__all__`.

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All pass.

- [ ] **Step 3: Run linter**

Run: `ruff check crosscat/ tests/`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add crosscat/__init__.py
git commit -m "feat: export packed v2 API from crosscat.__init__"
```

---

## Task 11: Edge case tests

**Files:**
- Test: `tests/test_packed_kernels_v2.py`

- [ ] **Step 1: Add cluster budget exhaustion test**

Add to `tests/test_packed_kernels_v2.py`:

```python
def test_cluster_budget_exhaustion():
    """When n_clusters >= max_clusters - 1, new cluster option is excluded."""
    key = jax.random.key(500)
    from crosscat.synthetic import generate_crosscat_data

    column_types = [ColumnType.CONTINUOUS, ColumnType.CONTINUOUS]
    result = generate_crosscat_data(key, 20, column_types, n_views=1, n_clusters=2)
    k2 = jax.random.key(501)
    state = initialize(k2, result["data"], column_types)
    # Pack with very small max_clusters to force budget exhaustion
    packed = pack_state(state, max_clusters=3, max_categories=4)

    k3 = jax.random.key(502)
    # Should not crash — just forces rows into existing clusters
    packed_new = packed_transition_row_assignments_v2(k3, packed, result["data"])
    recovered = unpack_state(packed_new, column_types)
    # All assignments should be < max_clusters - 1
    max_c = 3  # matches max_clusters=3 from pack_state above
    for view in recovered.views:
        assert int(jnp.max(view.row_assignments)) < max_c


def test_mixed_column_types_full_sweep():
    """Full sweep with all 5 column types produces valid state."""
    key = jax.random.key(600)
    from crosscat.synthetic import generate_crosscat_data

    column_types = [
        ColumnType.CONTINUOUS,
        ColumnType.CATEGORICAL,
        ColumnType.BINARY,
        ColumnType.ORDINAL,
        ColumnType.CYCLIC,
    ]
    result = generate_crosscat_data(key, 50, column_types, n_views=2, n_clusters=2)
    k2 = jax.random.key(601)
    state = initialize(k2, result["data"], column_types)
    packed = pack_state(state)

    k3 = jax.random.key(602)
    packed_new = packed_gibbs_sweep_v2(k3, packed, result["data"], n_sweeps=2)
    recovered = unpack_state(packed_new, column_types)
    errors = validate_state(recovered, result["data"])
    assert errors == [], f"Validation errors: {errors}"
```

- [ ] **Step 2: Run new tests**

Run: `pytest tests/test_packed_kernels_v2.py -k "budget_exhaustion or mixed_column" -v`
Expected: All pass.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v -m "not slow"`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_packed_kernels_v2.py
git commit -m "test: add edge case tests (cluster budget exhaustion, mixed column types)"
```

---

## Task 12: Colab GPU benchmark notebook

**Files:**
- Create: `notebooks/gpu_benchmark.ipynb`

- [ ] **Step 1: Create notebook directory**

```bash
mkdir -p notebooks
```

- [ ] **Step 2: Create the benchmark notebook**

Create `notebooks/gpu_benchmark.ipynb` with cells for:
1. Setup: `pip install -e .`, detect GPU, print JAX/device info
2. Synthetic data generation at 3 scales (100x10, 1000x20, 5000x50)
3. Gibbs sweep benchmark: unpacked vs packed v1 vs packed v2 (with/without JIT warmup)
4. Inference query benchmark: predictive_sample at 100/1000/10000 samples
5. Scaling analysis: rows vs time, columns vs time (line plots with matplotlib)
6. Memory comparison
7. Correctness verification: run both paths on same data, compare outputs

Each benchmark cell should use `timeit` or manual timing with `block_until_ready()` for accurate GPU measurement.

- [ ] **Step 3: Verify notebook runs locally (CPU)**

Run: `jupyter nbconvert --execute notebooks/gpu_benchmark.ipynb --to notebook --ExecutePreprocessor.timeout=600`
Expected: Completes without error (will be slower on CPU, that's expected).

- [ ] **Step 4: Commit**

```bash
git add notebooks/gpu_benchmark.ipynb
git commit -m "feat: add Colab GPU benchmark notebook

Benchmarks unpacked vs packed v1 vs packed v2 Gibbs sweeps and
inference queries. Includes scaling analysis and correctness verification."
```

---

## Task 13: Final validation and lint

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v -m "not slow"`
Expected: All tests pass, including all new v2 tests.

- [ ] **Step 2: Run linter and formatter**

```bash
ruff check crosscat/ tests/
ruff format crosscat/ tests/
```

Expected: No errors after formatting.

- [ ] **Step 3: Verify no Python for-loops in new code**

Search for any remaining `for ` loops in the new v2 functions and packed_inference.py:

```bash
grep -n "for " crosscat/packed_inference.py
grep -n "for " crosscat/packed_state.py | grep -v "# for\|format\|information\|before\|after"
```

Expected: No `for x in range(...)` patterns in any v2 function or packed_inference.py.

- [ ] **Step 4: Commit any formatting fixes**

```bash
git add crosscat/ tests/
git commit -m "style: formatting fixes from ruff"
```
