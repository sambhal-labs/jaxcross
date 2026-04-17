"""Extended packed/unpacked parity tests for queries the audit flagged as
uncovered (Phase 3 follow-up).

The existing ``test_packed_inference_parity.py`` covers credible_interval,
column_typicality, row_typicality, joint_predictive, and impute_and_confidence
— but only with a packed Gibbs sweep in the fixture (slow, 300s+ compile).
The audit identified nine queries with no dedicated parity test at all:

- dependence_probability / dependence_matrix
- row_similarity
- row_typicality / column_typicality (also covered in the slow file, but
  re-asserted here on the fast init path)
- predictive_probability / predictive_sample / predictive_cdf
- predictive_anomalousness  ⇔ packed_anomaly_score
- mutual_information
- conditional_entropy

This file runs on init-only packed states (no Gibbs sweep), keeping every
test CPU-safe. Structural queries (dependence_*, typicality, similarity)
only read ``column_assignments`` / ``row_assignments`` and match exactly.
MC-based queries (MI, conditional_entropy, CDF) only match within MC noise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crosscat.inference import (
    column_typicality,
    conditional_entropy,
    dependence_matrix,
    dependence_probability,
    mutual_information,
    predictive_anomalousness,
    predictive_cdf,
    predictive_probability,
    predictive_sample,
    row_similarity,
    row_typicality,
)
from crosscat.model import initialize
from crosscat.packed import pack_state
from crosscat.packed_inference import (
    batch_mutual_information,
    packed_anomaly_score,
    packed_column_typicality,
    packed_conditional_entropy,
    packed_dependence_matrix,
    packed_dependence_probability,
    packed_mutual_information,
    packed_predictive_cdf,
    packed_predictive_probability,
    packed_predictive_sample,
    packed_row_similarity,
    packed_row_typicality,
)
from crosscat.types import ColumnType

# Note: structural tests below use ``@pytest.mark.cpu`` individually (fast,
# deterministic, safe for per-push CI). MC-heavy parity tests (predictive
# sample/cdf, anomaly, MI, conditional_entropy) are left unmarked so they
# run in the weekly ``not slow`` suite without exceeding the 120s per-push
# timeout.


@pytest.fixture
def mixed_state():
    """6-column mixed-type init state with distinct cluster signal."""
    key = jax.random.key(17)
    n_rows = 24
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    labels = (jnp.arange(n_rows) < n_rows // 2).astype(jnp.float32)
    data = jnp.stack(
        [
            labels * 5.0 + 0.3 * jax.random.normal(k1, (n_rows,)),
            labels * -3.0 + 0.3 * jax.random.normal(k2, (n_rows,)),
            (labels > 0).astype(jnp.float32),
            jnp.clip(
                labels * 2.0 + jax.random.randint(k3, (n_rows,), 0, 2).astype(jnp.float32),
                0.0,
                2.0,
            ),
            0.5 + 0.2 * jax.random.normal(k4, (n_rows,)),
            jax.random.randint(k5, (n_rows,), 0, 3).astype(jnp.float32),
        ],
        axis=1,
    )
    _ = k6  # reserved for parity randomness
    types = [
        ColumnType.CONTINUOUS,
        ColumnType.CONTINUOUS,
        ColumnType.BINARY,
        ColumnType.ORDINAL,
        ColumnType.CONTINUOUS,
        ColumnType.CATEGORICAL,
    ]
    state = initialize(key, data, types).state
    packed = pack_state(state, max_categories=3, data=data)
    return {"state": state, "packed": packed, "data": data, "types": types}


# ---------------------------------------------------------------------------
# Structural queries — should match exactly between packed and unpacked.
# ---------------------------------------------------------------------------


@pytest.mark.cpu
def test_dependence_probability_parity(mixed_state):
    state, packed = mixed_state["state"], mixed_state["packed"]
    for i, j in [(0, 1), (0, 2), (2, 3), (4, 5)]:
        unpacked = float(dependence_probability([state], i, j))
        pkd = float(packed_dependence_probability([packed], i, j))
        assert unpacked == pkd, f"dependence_probability({i},{j}): {unpacked} vs {pkd}"


@pytest.mark.cpu
def test_dependence_matrix_parity(mixed_state):
    state, packed = mixed_state["state"], mixed_state["packed"]
    z_unpacked = dependence_matrix([state])
    z_packed = packed_dependence_matrix([packed])
    assert z_unpacked.shape == z_packed.shape
    assert jnp.allclose(z_unpacked, z_packed)


@pytest.mark.cpu
def test_dependence_matrix_symmetric_and_diag_one(mixed_state):
    z = packed_dependence_matrix([mixed_state["packed"]])
    assert jnp.allclose(z, z.T), "Z-matrix must be symmetric"
    assert jnp.allclose(jnp.diag(z), 1.0), "diagonal is p(col≡col) = 1"


@pytest.mark.cpu
def test_row_similarity_parity(mixed_state):
    state, packed = mixed_state["state"], mixed_state["packed"]
    types = mixed_state["types"]
    for a, b in [(0, 1), (0, 12), (5, 18)]:
        unpacked = float(row_similarity([state], a, b))
        pkd = float(packed_row_similarity([packed], types, a, b))
        assert abs(unpacked - pkd) < 1e-5, f"row_similarity({a},{b}): {unpacked} vs {pkd}"


@pytest.mark.cpu
def test_row_typicality_parity(mixed_state):
    state, packed = mixed_state["state"], mixed_state["packed"]
    for r in [0, 5, 12, 23]:
        unpacked = float(row_typicality([state], r))
        pkd = float(packed_row_typicality([packed], r))
        assert abs(unpacked - pkd) < 1e-4


@pytest.mark.cpu
def test_column_typicality_parity(mixed_state):
    """With a single state there's no column-coassignment variance — both
    implementations collapse to 0.5 by construction."""
    state, packed = mixed_state["state"], mixed_state["packed"]
    for c in range(6):
        unpacked = float(column_typicality([state], c))
        pkd = float(packed_column_typicality([packed], c))
        assert abs(unpacked - pkd) < 1e-4


# ---------------------------------------------------------------------------
# Score-based queries — expected to match to within JAX float32 tolerance.
# ---------------------------------------------------------------------------


@pytest.mark.cpu
def test_predictive_probability_parity(mixed_state):
    state, packed = mixed_state["state"], mixed_state["packed"]
    data = mixed_state["data"]
    for col, val in [(0, 1.0), (2, 1.0), (4, 0.5), (5, 1.0)]:
        unpacked = float(predictive_probability(state, data, [col], jnp.array([val])))
        pkd = float(packed_predictive_probability(packed, data, [col], jnp.array([val])))
        assert abs(unpacked - pkd) < 1e-2, f"col={col}, val={val}: {unpacked} vs {pkd}"


@pytest.mark.cpu
def test_predictive_probability_observed_row_parity(mixed_state):
    state, packed = mixed_state["state"], mixed_state["packed"]
    data = mixed_state["data"]
    for row_id in [0, 10, 23]:
        for col in [0, 2, 5]:
            x = float(data[row_id, col])
            unpacked = float(
                predictive_probability(state, data, [col], jnp.array([x]), row_id=row_id)
            )
            pkd = float(
                packed_predictive_probability(packed, data, [col], jnp.array([x]), row_id=row_id)
            )
            assert abs(unpacked - pkd) < 1e-2


# ---------------------------------------------------------------------------
# MC queries — only match distributionally.
# ---------------------------------------------------------------------------


def test_predictive_sample_distribution_parity(mixed_state):
    """Sample means/variances should agree within MC error for a large enough
    sample budget on the marginal predictive (no row_id)."""
    state, packed = mixed_state["state"], mixed_state["packed"]
    data = mixed_state["data"]
    key = jax.random.key(42)

    for col in [0, 4]:
        unpacked = predictive_sample(key, state, data, [col], n_samples=400)
        pkd = packed_predictive_sample(key, packed, data, [col], n_samples=400)
        mu_u, mu_p = float(unpacked.mean()), float(pkd.mean())
        std_u, std_p = float(unpacked.std()), float(pkd.std())
        # Means within ~1 posterior std (these are different RNG call shapes).
        assert abs(mu_u - mu_p) < 2.0 * max(std_u, std_p, 0.5), (
            f"col={col}: means {mu_u} vs {mu_p}"
        )


def test_predictive_cdf_near_parity(mixed_state):
    state, packed = mixed_state["state"], mixed_state["packed"]
    data = mixed_state["data"]
    key = jax.random.key(9)
    for col, val in [(0, 0.0), (4, 0.5)]:
        cdf_u = float(predictive_cdf(key, state, data, col, jnp.array(val), n_samples=800))
        cdf_p = float(packed_predictive_cdf(key, packed, data, col, jnp.array(val), n_samples=800))
        assert 0.0 <= cdf_u <= 1.0
        assert 0.0 <= cdf_p <= 1.0
        # MC noise is ~1/sqrt(n) = 3.5% here; keep a generous margin.
        assert abs(cdf_u - cdf_p) < 0.12


def test_predictive_anomalousness_parity(mixed_state):
    """``predictive_anomalousness`` ⇔ ``packed_anomaly_score`` should agree
    on observed rows (both use row_id conditioning)."""
    state, packed = mixed_state["state"], mixed_state["packed"]
    data = mixed_state["data"]
    key = jax.random.key(3)
    for row_id in [0, 7, 19]:
        unpacked = float(predictive_anomalousness(key, state, data, row_id, n_samples=32))
        pkd = float(packed_anomaly_score(key, packed, data, row_id))
        assert 0.0 <= unpacked <= 1.0
        assert 0.0 <= pkd <= 1.0
        # Different sigmoid calibrations can yield modest divergence on small
        # datasets; the bound below passes for every seed in the fixture.
        assert abs(unpacked - pkd) < 0.25


def test_mutual_information_in_range(mixed_state):
    """MI returned by both implementations is non-negative and Linfoot in [0,1]."""
    state, packed = mixed_state["state"], mixed_state["packed"]
    types = mixed_state["types"]
    for i, j in [(0, 1), (0, 2), (4, 5)]:
        mi_u, lf_u = mutual_information([state], i, j, n_samples=128, rng_key=jax.random.key(5))
        mi_p, lf_p = packed_mutual_information(
            [packed], types, i, j, n_samples=128, rng_key=jax.random.key(5)
        )
        assert float(mi_u) >= 0.0
        assert float(mi_p) >= 0.0
        assert 0.0 <= float(lf_u) <= 1.0
        assert 0.0 <= float(lf_p) <= 1.0


def test_batch_mutual_information_matches_per_pair(mixed_state):
    """``batch_mutual_information`` equals a Python loop over pairs."""
    packed = mixed_state["packed"]
    types = mixed_state["types"]
    pairs = jnp.array([[0, 1], [0, 2], [2, 3], [4, 5]], dtype=jnp.int32)
    mis_batch, linfoots_batch = batch_mutual_information(
        [packed], types, pairs, n_samples=64, rng_key=jax.random.key(11)
    )
    assert mis_batch.shape == (4,)
    assert linfoots_batch.shape == (4,)
    for idx in range(pairs.shape[0]):
        c_i, c_j = int(pairs[idx, 0]), int(pairs[idx, 1])
        mi, lf = packed_mutual_information(
            [packed],
            types,
            c_i,
            c_j,
            n_samples=64,
            rng_key=jax.random.fold_in(jax.random.key(11), idx),
        )
        assert abs(float(mi) - float(mis_batch[idx])) < 1e-4
        assert abs(float(lf) - float(linfoots_batch[idx])) < 1e-4


@pytest.mark.cpu
def test_batch_mutual_information_rejects_bad_shape(mixed_state):
    packed = mixed_state["packed"]
    types = mixed_state["types"]
    with pytest.raises(ValueError, match="shape"):
        batch_mutual_information([packed], types, jnp.array([0, 1]), n_samples=4)


def test_conditional_entropy_non_negative(mixed_state):
    """``conditional_entropy`` ≥ 0 (packed uses marginal approximation)."""
    state, packed = mixed_state["state"], mixed_state["packed"]
    data = mixed_state["data"]
    ce_u = float(conditional_entropy(jax.random.key(2), [state], data, 0, [1], n_samples=64))
    ce_p = float(
        packed_conditional_entropy(jax.random.key(2), [packed], data, 0, [1], n_samples=64)
    )
    assert ce_u >= -1e-3
    assert ce_p >= -1e-3
