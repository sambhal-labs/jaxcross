"""Scoring / component functions for the packed CrossCat representation.

Extracted from crosscat/packed_state.py — JIT-compatible scoring, posterior
predictive scoring, and posterior predictive sampling using unified type
dispatch (all types computed, correct result selected via jnp.where).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln

from crosscat.packed.state import BINARY_ID, CATEGORICAL_ID, CONTINUOUS_ID, ORDINAL_ID
from crosscat.types import LOG_EPS

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
        x - 0.5 * jnp.log(2.0 * jnp.pi * jnp.maximum(x, LOG_EPS)),
    )


def _ng_log_marginal(n, sum_x, sum_x_sq, mu0, r, s, nu):
    """Normal-Gamma log marginal likelihood (element-wise)."""
    n = n.astype(jnp.float32)
    r_n = r + n
    nu_n = nu + n
    nu_s = nu * s
    mean = jnp.where(n > 0, sum_x / jnp.maximum(n, 1.0), 0.0)
    nu_n_s_n = (
        nu_s
        + sum_x_sq
        - sum_x**2 / jnp.maximum(n, 1.0)
        + r * n * (mu0 - mean) ** 2 / jnp.maximum(r_n, LOG_EPS)
    )
    nu_n_s_n = jnp.where(n > 0, nu_n_s_n, nu_s)
    # Clamp to avoid log of negative
    nu_n_s_n = jnp.maximum(nu_n_s_n, LOG_EPS)

    log_ml = (
        -0.5 * n * jnp.log(2.0 * jnp.pi)
        + 0.5 * jnp.log(r / jnp.maximum(r_n, LOG_EPS))
        + 0.5 * nu * jnp.log(jnp.maximum(nu_s / 2.0, LOG_EPS))
        - 0.5 * nu_n * jnp.log(jnp.maximum(nu_n_s_n / 2.0, LOG_EPS))
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


def _ol_level_probs(mu, cutpoints):
    """Compute P(Y=k | μ, cutpoints) for all K levels.

    Uses the cumulative logistic link: P(Y=k) = σ(c_k - μ) - σ(c_{k-1} - μ).
    Cutpoints padded with +inf produce probability 0 for padded levels.
    """
    extended = jnp.concatenate([jnp.array([-1e10]), cutpoints, jnp.array([1e10])])
    cum = jax.nn.sigmoid(extended - mu)
    probs = cum[1:] - cum[:-1]
    return jnp.maximum(probs, LOG_EPS)


_OL_N_GRID = 31  # grid points for μ integration, matches hyper grid size


def _ol_log_marginal(n, cat_counts, cutpoints, mu0, s0):
    """Ordered logistic log marginal likelihood via grid integration over μ.

    Non-conjugate: integrates p(counts | μ, cutpoints) · p(μ) over a μ grid.
    Prior: μ ~ N(mu0, s0). Returns 0.0 for empty clusters (n=0).
    """
    n = n.astype(jnp.float32)
    s0 = jnp.maximum(s0, LOG_EPS)
    half_range = 4.0 * jnp.sqrt(s0)
    mu_grid = jnp.linspace(mu0 - half_range, mu0 + half_range, _OL_N_GRID)
    delta_mu = 2.0 * half_range / (_OL_N_GRID - 1)

    def log_score_at_mu(mu):
        probs = _ol_level_probs(mu, cutpoints)
        log_lik = jnp.sum(cat_counts * jnp.log(probs))
        log_prior = -0.5 * jnp.log(2.0 * jnp.pi * s0) - 0.5 * (mu - mu0) ** 2 / s0
        return log_lik + log_prior

    log_scores = jax.vmap(log_score_at_mu)(mu_grid)
    log_marginal = jax.nn.logsumexp(log_scores) + jnp.log(delta_mu)
    return jnp.where(n > 0, log_marginal, 0.0)


def _ol_posterior_predictive_logp(x, n, cat_counts, cutpoints, mu0, s0):
    """Ordered logistic posterior predictive log p(x=k | data, cutpoints).

    Integrates P(Y=k | μ) · p(μ | data) over μ grid.
    """
    s0 = jnp.maximum(s0, LOG_EPS)
    half_range = 4.0 * jnp.sqrt(s0)
    mu_grid = jnp.linspace(mu0 - half_range, mu0 + half_range, _OL_N_GRID)

    def log_unnorm_posterior(mu):
        probs = _ol_level_probs(mu, cutpoints)
        return jnp.sum(cat_counts * jnp.log(probs)) - 0.5 * (mu - mu0) ** 2 / s0

    log_weights = jax.vmap(log_unnorm_posterior)(mu_grid)
    log_weights = log_weights - jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_weights)

    all_probs = jax.vmap(lambda mu: _ol_level_probs(mu, cutpoints))(mu_grid)
    avg_probs = jnp.sum(weights[:, None] * all_probs, axis=0)

    x_int = jnp.clip(x.astype(jnp.int32), 0, cat_counts.shape[-1] - 1)
    return jnp.log(jnp.maximum(avg_probs[x_int], LOG_EPS))


def _ol_sample(rng_key, n, cat_counts, cutpoints, mu0, s0):
    """Sample from ordered logistic posterior predictive."""
    s0 = jnp.maximum(s0, LOG_EPS)
    half_range = 4.0 * jnp.sqrt(s0)
    mu_grid = jnp.linspace(mu0 - half_range, mu0 + half_range, _OL_N_GRID)

    def log_unnorm_posterior(mu):
        probs = _ol_level_probs(mu, cutpoints)
        return jnp.sum(cat_counts * jnp.log(probs)) - 0.5 * (mu - mu0) ** 2 / s0

    log_weights = jax.vmap(log_unnorm_posterior)(mu_grid)
    log_weights = log_weights - jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_weights)

    all_probs = jax.vmap(lambda mu: _ol_level_probs(mu, cutpoints))(mu_grid)
    avg_probs = jnp.sum(weights[:, None] * all_probs, axis=0)

    return jax.random.categorical(rng_key, jnp.log(jnp.maximum(avg_probs, LOG_EPS))).astype(
        jnp.float32
    )


def _vm_log_marginal(n, sum_sin, sum_cos, kappa, vm_a, vm_mu):
    """Von Mises log marginal likelihood (exact conjugate).

    Integrates out mean direction mu via conjugate von Mises prior.
    kappa = likelihood concentration, vm_a = prior concentration, vm_mu = prior mean (b).
    See Mardia & Jupp (2000), Section 5.3.
    """
    n = n.astype(jnp.float32)
    # Posterior resultant length: data + prior(a, b) contribution
    total_sin = sum_sin + vm_a * jnp.sin(vm_mu)
    total_cos = sum_cos + vm_a * jnp.cos(vm_mu)
    R = jnp.sqrt(total_sin**2 + total_cos**2)
    log_ml = (
        -n * jnp.log(2.0 * jnp.pi)
        + _log_bessel_i0(R)
        - n * _log_bessel_i0(kappa)
        - _log_bessel_i0(vm_a)
    )
    return jnp.where(n > 0, log_ml, 0.0)


def unified_log_marginal(
    type_id: Array,
    count: Array,
    sum_x: Array,
    sum_x_sq: Array,
    cat_counts: Array,
    sum_sin: Array,
    sum_cos: Array,
    mu: Array,
    r: Array,
    s: Array,
    nu: Array,
    dir_alpha: Array,
    alpha: Array,
    beta: Array,
    kappa: Array,
    vm_a: Array,
    vm_mu: Array,
    cutpoints: Array,
) -> Array:
    """Compute log marginal likelihood for any column type without Python branching.

    Computes ALL type results and selects the correct one via jnp.where.
    This wastes trivial compute but enables full JIT compilation.

    Args:
        type_id: Integer column type ID (CONTINUOUS_ID, CATEGORICAL_ID, etc.).
        count: Observation count in cluster.
        sum_x: Sum of values (continuous).
        sum_x_sq: Sum of squared values (continuous).
        cat_counts: Category count vector (categorical/binary/ordinal).
        sum_sin: Sum of sin(x) (cyclic).
        sum_cos: Sum of cos(x) (cyclic).
        mu: Normal-Gamma prior mean.
        r: Normal-Gamma prior count.
        s: Normal-Gamma prior sum-of-squares.
        nu: Normal-Gamma prior degrees of freedom.
        dir_alpha: Dirichlet concentration.
        alpha: Beta-Bernoulli prior successes.
        beta: Beta-Bernoulli prior failures.
        kappa: Von Mises likelihood concentration.
        vm_a: Von Mises prior concentration.
        vm_mu: Von Mises prior mean direction.
        cutpoints: Ordered thresholds for ordinal (max_categories - 1,).

    Returns:
        Scalar log marginal likelihood for the column type indicated by type_id.
    """
    continuous_score = _ng_log_marginal(count, sum_x, sum_x_sq, mu, r, s, nu)
    cat_score = _dc_log_marginal(count, cat_counts, dir_alpha)
    binary_score = _bb_log_marginal(count, sum_x, alpha, beta)
    ordinal_score = _ol_log_marginal(count, cat_counts, cutpoints, mu, s)
    cyclic_score = _vm_log_marginal(count, sum_sin, sum_cos, kappa, vm_a, vm_mu)

    return jnp.where(
        type_id == CONTINUOUS_ID,
        continuous_score,
        jnp.where(
            type_id == CATEGORICAL_ID,
            cat_score,
            jnp.where(
                type_id == ORDINAL_ID,
                ordinal_score,
                jnp.where(
                    type_id == BINARY_ID,
                    binary_score,
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
    mu_n = (r * mu0 + sum_x) / jnp.maximum(r_n, LOG_EPS)
    nu_n = nu + n
    nu_s = nu * s
    mean = jnp.where(n > 0, sum_x / jnp.maximum(n, 1.0), 0.0)
    nu_n_s_n = (
        nu_s
        + sum_x_sq
        - sum_x**2 / jnp.maximum(n, 1.0)
        + r * n * (mu0 - mean) ** 2 / jnp.maximum(r_n, LOG_EPS)
    )
    nu_n_s_n = jnp.where(n > 0, nu_n_s_n, nu_s)
    nu_n_s_n = jnp.maximum(nu_n_s_n, LOG_EPS)

    df = nu_n
    loc = mu_n
    scale_sq = (nu_n_s_n / jnp.maximum(nu_n, LOG_EPS)) * (1.0 + 1.0 / jnp.maximum(r_n, LOG_EPS))
    scale = jnp.sqrt(jnp.maximum(scale_sq, LOG_EPS))
    z = (x - loc) / scale

    log_p = (
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * jnp.log(df * jnp.pi)
        - jnp.log(scale)
        - (df + 1.0) / 2.0 * jnp.log(1.0 + z**2 / jnp.maximum(df, LOG_EPS))
    )
    return log_p


def _dc_posterior_predictive_logp(x, count, cat_counts, dir_alpha):
    """Dirichlet-Categorical posterior predictive log p(x | suffstats, hypers)."""
    n = count.astype(jnp.float32)
    k = jnp.array(cat_counts.shape[-1], dtype=jnp.float32)
    probs = (cat_counts + dir_alpha) / jnp.maximum(n + k * dir_alpha, LOG_EPS)
    idx = x.astype(jnp.int32)
    idx = jnp.clip(idx, 0, cat_counts.shape[-1] - 1)
    return jnp.log(jnp.maximum(probs[idx], LOG_EPS))


def _bb_posterior_predictive_logp(x, count, sum_x, alpha, beta):
    """Beta-Bernoulli posterior predictive log p(x | suffstats, hypers)."""
    n = count.astype(jnp.float32)
    p1 = (sum_x + alpha) / jnp.maximum(n + alpha + beta, LOG_EPS)
    log_p1 = jnp.log(jnp.maximum(p1, LOG_EPS))
    log_p0 = jnp.log(jnp.maximum(1.0 - p1, LOG_EPS))
    return jnp.where(x > 0.5, log_p1, log_p0)


def _vm_posterior_predictive_logp(x, count, sum_sin, sum_cos, kappa, vm_a, vm_mu):
    """Von Mises posterior predictive log p(x | suffstats, hypers)."""
    total_sin = sum_sin + vm_a * jnp.sin(vm_mu)
    total_cos = sum_cos + vm_a * jnp.cos(vm_mu)
    mu_post = jnp.arctan2(total_sin, total_cos)
    return kappa * jnp.cos(x - mu_post) - jnp.log(2.0 * jnp.pi) - _log_bessel_i0(kappa)


def unified_posterior_predictive_logp(
    x: Array,
    type_id: Array,
    count: Array,
    sum_x: Array,
    sum_x_sq: Array,
    cat_counts: Array,
    sum_sin: Array,
    sum_cos: Array,
    mu: Array,
    r: Array,
    s: Array,
    nu: Array,
    dir_alpha: Array,
    alpha: Array,
    beta: Array,
    kappa: Array,
    vm_a: Array,
    vm_mu: Array,
    cutpoints: Array,
) -> Array:
    """Compute posterior predictive logp for any column type without Python branching.

    Computes ALL type results and selects the correct one via jnp.where.

    Args:
        x: New observation value to score.
        type_id: Integer column type ID (CONTINUOUS_ID, CATEGORICAL_ID, etc.).
        count: Observation count in cluster.
        sum_x: Sum of values (continuous).
        sum_x_sq: Sum of squared values (continuous).
        cat_counts: Category count vector (categorical/binary/ordinal).
        sum_sin: Sum of sin(x) (cyclic).
        sum_cos: Sum of cos(x) (cyclic).
        mu: Normal-Gamma prior mean.
        r: Normal-Gamma prior count.
        s: Normal-Gamma prior sum-of-squares.
        nu: Normal-Gamma prior degrees of freedom.
        dir_alpha: Dirichlet concentration.
        alpha: Beta-Bernoulli prior successes.
        beta: Beta-Bernoulli prior failures.
        kappa: Von Mises likelihood concentration.
        vm_a: Von Mises prior concentration.
        vm_mu: Von Mises prior mean direction.
        cutpoints: Ordered thresholds for ordinal (max_categories - 1,).

    Returns:
        Scalar log predictive probability for the column type indicated by type_id.
    """
    cont = _ng_posterior_predictive_logp(x, count, sum_x, sum_x_sq, mu, r, s, nu)
    cat = _dc_posterior_predictive_logp(x, count, cat_counts, dir_alpha)
    binary = _bb_posterior_predictive_logp(x, count, sum_x, alpha, beta)
    ordinal = _ol_posterior_predictive_logp(x, count, cat_counts, cutpoints, mu, s)
    cyclic = _vm_posterior_predictive_logp(x, count, sum_sin, sum_cos, kappa, vm_a, vm_mu)

    return jnp.where(
        type_id == CONTINUOUS_ID,
        cont,
        jnp.where(
            type_id == CATEGORICAL_ID,
            cat,
            jnp.where(
                type_id == ORDINAL_ID, ordinal, jnp.where(type_id == BINARY_ID, binary, cyclic)
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Posterior predictive sampling (JIT-compatible, type-unified)
# ---------------------------------------------------------------------------


def _ng_sample(rng_key, count, sum_x, sum_x_sq, mu0, r, s, nu):
    """Sample from Normal-Gamma posterior predictive (Student-t).

    Uses Normal / sqrt(Chi2/df) representation of Student-t.
    """
    n = count.astype(jnp.float32)
    r_n = r + n
    mu_n = (r * mu0 + sum_x) / jnp.maximum(r_n, LOG_EPS)
    nu_n = nu + n
    nu_s = nu * s
    mean = jnp.where(n > 0, sum_x / jnp.maximum(n, 1.0), 0.0)
    nu_n_s_n = (
        nu_s
        + sum_x_sq
        - sum_x**2 / jnp.maximum(n, 1.0)
        + r * n * (mu0 - mean) ** 2 / jnp.maximum(r_n, LOG_EPS)
    )
    nu_n_s_n = jnp.where(n > 0, nu_n_s_n, nu_s)
    nu_n_s_n = jnp.maximum(nu_n_s_n, LOG_EPS)

    df = nu_n
    loc = mu_n
    scale_sq = (nu_n_s_n / jnp.maximum(nu_n, LOG_EPS)) * (1.0 + 1.0 / jnp.maximum(r_n, LOG_EPS))
    scale = jnp.sqrt(jnp.maximum(scale_sq, LOG_EPS))

    # Student-t sample: loc + scale * Normal(0,1) / sqrt(Chi2(df)/df)
    k1, k2 = jax.random.split(rng_key)
    z = jax.random.normal(k1)
    # Chi2(df) = sum of df standard normals squared; use gamma distribution
    chi2 = 2.0 * jax.random.gamma(k2, df / 2.0)
    chi2 = jnp.maximum(chi2, LOG_EPS)
    return loc + scale * z / jnp.sqrt(chi2 / jnp.maximum(df, LOG_EPS))


def _dc_sample(rng_key, count, cat_counts, dir_alpha):
    """Sample from Dirichlet-Categorical posterior predictive."""
    n = count.astype(jnp.float32)
    k = jnp.array(cat_counts.shape[-1], dtype=jnp.float32)
    probs = (cat_counts + dir_alpha) / jnp.maximum(n + k * dir_alpha, LOG_EPS)
    log_probs = jnp.log(jnp.maximum(probs, LOG_EPS))
    return jax.random.categorical(rng_key, log_probs).astype(jnp.float32)


def _bb_sample(rng_key, count, sum_x, alpha, beta):
    """Sample from Beta-Bernoulli posterior predictive."""
    n = count.astype(jnp.float32)
    p1 = (sum_x + alpha) / jnp.maximum(n + alpha + beta, LOG_EPS)
    return jax.random.bernoulli(rng_key, p1).astype(jnp.float32)


def _vm_sample(rng_key, count, sum_sin, sum_cos, kappa, vm_a, vm_mu):
    """Sample from von Mises posterior predictive via rejection sampling.

    Matches original CyclicComponentModel::get_draw_constrained():
    uniform proposal on [0, 2*pi), accept/reject against predictive logp.
    """
    total_sin = sum_sin + vm_a * jnp.sin(vm_mu)
    total_cos = sum_cos + vm_a * jnp.cos(vm_mu)
    mu_post = jnp.arctan2(total_sin, total_cos)

    # Mode logp (envelope for rejection sampling)
    log_M = kappa - jnp.log(2.0 * jnp.pi) - _log_bessel_i0(kappa)

    max_iters = 1000

    def _cond(state):
        return state[0] & (state[1] < max_iters)

    def _body(state):
        _, itr, sample, key_loop = state
        k1, k2, k3 = jax.random.split(key_loop, 3)
        x = jax.random.uniform(k1) * 2.0 * jnp.pi
        log_u = jnp.log(jax.random.uniform(k2)) + log_M
        log_target = kappa * jnp.cos(x - mu_post) - jnp.log(2.0 * jnp.pi) - _log_bessel_i0(kappa)
        accepted = log_u < log_target
        return (~accepted, itr + 1, jnp.where(accepted, x, sample), k3)

    _, _, sample, _ = jax.lax.while_loop(
        _cond, _body, (jnp.bool_(True), jnp.int32(0), 0.0, rng_key)
    )
    return sample % (2.0 * jnp.pi)


def unified_sample_posterior_predictive(
    rng_key: Array,
    type_id: Array,
    count: Array,
    sum_x: Array,
    sum_x_sq: Array,
    cat_counts: Array,
    sum_sin: Array,
    sum_cos: Array,
    mu: Array,
    r: Array,
    s: Array,
    nu: Array,
    dir_alpha: Array,
    alpha: Array,
    beta: Array,
    kappa: Array,
    vm_a: Array,
    vm_mu: Array,
    cutpoints: Array,
) -> Array:
    """Sample from posterior predictive for any column type without Python branching.

    Computes ALL type samples and selects the correct one via jnp.where.
    This wastes trivial compute but enables full JIT compilation.

    Args:
        rng_key: PRNG key for sampling.
        type_id: Integer column type ID (CONTINUOUS_ID, CATEGORICAL_ID, etc.).
        count: Observation count in cluster.
        sum_x: Sum of values (continuous).
        sum_x_sq: Sum of squared values (continuous).
        cat_counts: Category count vector (categorical/binary/ordinal).
        sum_sin: Sum of sin(x) (cyclic).
        sum_cos: Sum of cos(x) (cyclic).
        mu: Normal-Gamma prior mean.
        r: Normal-Gamma prior count.
        s: Normal-Gamma prior sum-of-squares.
        nu: Normal-Gamma prior degrees of freedom.
        dir_alpha: Dirichlet concentration.
        alpha: Beta-Bernoulli prior successes.
        beta: Beta-Bernoulli prior failures.
        kappa: Von Mises likelihood concentration.
        vm_a: Von Mises prior concentration.
        vm_mu: Von Mises prior mean direction.
        cutpoints: Ordered thresholds for ordinal (max_categories - 1,).

    Returns:
        Scalar sample from the posterior predictive distribution.
    """
    k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)

    cont_sample = _ng_sample(k1, count, sum_x, sum_x_sq, mu, r, s, nu)
    cat_sample = _dc_sample(k2, count, cat_counts, dir_alpha)
    binary_sample = _bb_sample(k3, count, sum_x, alpha, beta)
    cyclic_sample = _vm_sample(k4, count, sum_sin, sum_cos, kappa, vm_a, vm_mu)
    ordinal_sample = _ol_sample(k5, count, cat_counts, cutpoints, mu, s)

    return jnp.where(
        type_id == CONTINUOUS_ID,
        cont_sample,
        jnp.where(
            type_id == CATEGORICAL_ID,
            cat_sample,
            jnp.where(
                type_id == ORDINAL_ID,
                ordinal_sample,
                jnp.where(type_id == BINARY_ID, binary_sample, cyclic_sample),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Batch-vectorized type-specialized scoring (no type dispatch overhead)
# ---------------------------------------------------------------------------


def batch_bb_posterior_predictive_logp(
    xs: Array, counts: Array, sum_xs: Array, alphas: Array, betas: Array
) -> Array:
    """Vectorized Beta-Bernoulli posterior predictive for a batch of columns.

    All inputs are (n_cols,) arrays. Returns (n_cols,) logps.
    Skips all type dispatch — caller must ensure all columns are binary.
    """
    n = counts.astype(jnp.float32)
    p1 = (sum_xs + alphas) / jnp.maximum(n + alphas + betas, LOG_EPS)
    log_p1 = jnp.log(jnp.maximum(p1, LOG_EPS))
    log_p0 = jnp.log(jnp.maximum(1.0 - p1, LOG_EPS))
    return jnp.where(xs > 0.5, log_p1, log_p0)


def batch_ng_posterior_predictive_logp(
    xs: Array,
    counts: Array,
    sum_xs: Array,
    sum_x_sqs: Array,
    mus: Array,
    rs: Array,
    ss: Array,
    nus: Array,
) -> Array:
    """Vectorized Normal-Gamma posterior predictive for a batch of columns.

    All inputs are (n_cols,) arrays. Returns (n_cols,) logps.
    """
    n = counts.astype(jnp.float32)
    r_n = rs + n
    mu_n = (rs * mus + sum_xs) / jnp.maximum(r_n, LOG_EPS)
    nu_n = nus + n
    nu_s = nus * ss
    mean = jnp.where(n > 0, sum_xs / jnp.maximum(n, 1.0), 0.0)
    nu_n_s_n = (
        nu_s
        + sum_x_sqs
        - sum_xs**2 / jnp.maximum(n, 1.0)
        + rs * n * (mus - mean) ** 2 / jnp.maximum(r_n, LOG_EPS)
    )
    nu_n_s_n = jnp.where(n > 0, nu_n_s_n, nu_s)
    nu_n_s_n = jnp.maximum(nu_n_s_n, LOG_EPS)

    df = nu_n
    loc = mu_n
    scale_sq = (nu_n_s_n / jnp.maximum(nu_n, LOG_EPS)) * (1.0 + 1.0 / jnp.maximum(r_n, LOG_EPS))
    scale = jnp.sqrt(jnp.maximum(scale_sq, LOG_EPS))
    z = (xs - loc) / scale

    return (
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * jnp.log(df * jnp.pi)
        - jnp.log(scale)
        - (df + 1.0) / 2.0 * jnp.log(1.0 + z**2 / jnp.maximum(df, LOG_EPS))
    )


def batch_dc_posterior_predictive_logp(
    xs: Array, counts: Array, cat_counts_batch: Array, dir_alphas: Array
) -> Array:
    """Vectorized Dirichlet-Categorical posterior predictive.

    xs: (n_cols,), counts: (n_cols,), cat_counts_batch: (n_cols, max_cats),
    dir_alphas: (n_cols,). Returns (n_cols,) logps.
    """
    n = counts.astype(jnp.float32)
    k = jnp.array(cat_counts_batch.shape[-1], dtype=jnp.float32)
    probs = (cat_counts_batch + dir_alphas[:, None]) / jnp.maximum(
        n[:, None] + k * dir_alphas[:, None], LOG_EPS
    )
    idxs = jnp.clip(xs.astype(jnp.int32), 0, cat_counts_batch.shape[-1] - 1)
    return jnp.log(jnp.maximum(probs[jnp.arange(xs.shape[0]), idxs], LOG_EPS))
