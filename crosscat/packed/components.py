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
        nu_s
        + sum_x_sq
        - sum_x**2 / jnp.maximum(n, 1.0)
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


def _vm_log_marginal(n, sum_sin, sum_cos, kappa, vm_mu):
    """Von Mises log marginal likelihood (exact conjugate).

    Integrates out mean direction mu via conjugate von Mises prior.
    See Mardia & Jupp (2000), Section 5.3.
    """
    n = n.astype(jnp.float32)
    # Posterior resultant length: data + prior contribution
    total_sin = sum_sin + kappa * jnp.sin(vm_mu)
    total_cos = sum_cos + kappa * jnp.cos(vm_mu)
    kappa_post = jnp.sqrt(total_sin**2 + total_cos**2)
    log_ml = (
        -n * jnp.log(2.0 * jnp.pi) + _log_bessel_i0(kappa_post) - (n + 1.0) * _log_bessel_i0(kappa)
    )
    return jnp.where(n > 0, log_ml, 0.0)


def unified_log_marginal(
    type_id,
    count,
    sum_x,
    sum_x_sq,
    cat_counts,
    sum_sin,
    sum_cos,
    mu,
    r,
    s,
    nu,
    dir_alpha,
    alpha,
    beta,
    kappa,
    vm_mu,
):
    """Compute log marginal likelihood for any column type without Python branching.

    Computes ALL type results and selects the correct one via jnp.where.
    This wastes trivial compute but enables full JIT compilation.
    """
    continuous_score = _ng_log_marginal(count, sum_x, sum_x_sq, mu, r, s, nu)
    cat_score = _dc_log_marginal(count, cat_counts, dir_alpha)
    binary_score = _bb_log_marginal(count, sum_x, alpha, beta)
    ordinal_score = _dc_log_marginal(count, cat_counts, jnp.ones_like(dir_alpha))
    cyclic_score = _vm_log_marginal(count, sum_sin, sum_cos, kappa, vm_mu)

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
    mu_n = (r * mu0 + sum_x) / jnp.maximum(r_n, 1e-30)
    nu_n = nu + n
    nu_s = nu * s
    mean = jnp.where(n > 0, sum_x / jnp.maximum(n, 1.0), 0.0)
    nu_n_s_n = (
        nu_s
        + sum_x_sq
        - sum_x**2 / jnp.maximum(n, 1.0)
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
    x,
    type_id,
    count,
    sum_x,
    sum_x_sq,
    cat_counts,
    sum_sin,
    sum_cos,
    mu,
    r,
    s,
    nu,
    dir_alpha,
    alpha,
    beta,
    kappa,
    vm_mu,
):
    """Compute posterior predictive logp for any column type without Python branching."""
    cont = _ng_posterior_predictive_logp(x, count, sum_x, sum_x_sq, mu, r, s, nu)
    cat = _dc_posterior_predictive_logp(x, count, cat_counts, dir_alpha)
    binary = _bb_posterior_predictive_logp(x, count, sum_x, alpha, beta)
    ordinal = _dc_posterior_predictive_logp(x, count, cat_counts, jnp.ones_like(dir_alpha))
    cyclic = _vm_posterior_predictive_logp(x, count, sum_sin, sum_cos, kappa, vm_mu)

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
    mu_n = (r * mu0 + sum_x) / jnp.maximum(r_n, 1e-30)
    nu_n = nu + n
    nu_s = nu * s
    mean = jnp.where(n > 0, sum_x / jnp.maximum(n, 1.0), 0.0)
    nu_n_s_n = (
        nu_s
        + sum_x_sq
        - sum_x**2 / jnp.maximum(n, 1.0)
        + r * n * (mu0 - mean) ** 2 / jnp.maximum(r_n, 1e-30)
    )
    nu_n_s_n = jnp.where(n > 0, nu_n_s_n, nu_s)
    nu_n_s_n = jnp.maximum(nu_n_s_n, 1e-30)

    df = nu_n
    loc = mu_n
    scale_sq = (nu_n_s_n / jnp.maximum(nu_n, 1e-30)) * (1.0 + 1.0 / jnp.maximum(r_n, 1e-30))
    scale = jnp.sqrt(jnp.maximum(scale_sq, 1e-30))

    # Student-t sample: loc + scale * Normal(0,1) / sqrt(Chi2(df)/df)
    k1, k2 = jax.random.split(rng_key)
    z = jax.random.normal(k1)
    # Chi2(df) = sum of df standard normals squared; use gamma distribution
    chi2 = 2.0 * jax.random.gamma(k2, df / 2.0)
    chi2 = jnp.maximum(chi2, 1e-30)
    return loc + scale * z / jnp.sqrt(chi2 / jnp.maximum(df, 1e-30))


def _dc_sample(rng_key, count, cat_counts, dir_alpha):
    """Sample from Dirichlet-Categorical posterior predictive."""
    n = count.astype(jnp.float32)
    k = jnp.array(cat_counts.shape[-1], dtype=jnp.float32)
    probs = (cat_counts + dir_alpha) / jnp.maximum(n + k * dir_alpha, 1e-30)
    log_probs = jnp.log(jnp.maximum(probs, 1e-30))
    return jax.random.categorical(rng_key, log_probs).astype(jnp.float32)


def _bb_sample(rng_key, count, sum_x, alpha, beta):
    """Sample from Beta-Bernoulli posterior predictive."""
    n = count.astype(jnp.float32)
    p1 = (sum_x + alpha) / jnp.maximum(n + alpha + beta, 1e-30)
    return jax.random.bernoulli(rng_key, p1).astype(jnp.float32)


def _vm_sample(rng_key, count, sum_sin, sum_cos, kappa, vm_mu):
    """Sample from von Mises posterior predictive (wrapped normal approximation)."""
    total_sin = sum_sin + kappa * jnp.sin(vm_mu)
    total_cos = sum_cos + kappa * jnp.cos(vm_mu)
    r_post = jnp.sqrt(total_sin**2 + total_cos**2)
    mu_post = jnp.arctan2(total_sin, total_cos)
    kappa_post = r_post

    # Wrapped normal approximation: sigma = 1/sqrt(kappa)
    sigma = 1.0 / jnp.sqrt(jnp.maximum(kappa_post, 1e-30))
    z = jax.random.normal(rng_key)
    # Wrap to [0, 2*pi)
    sample = mu_post + sigma * z
    return sample % (2.0 * jnp.pi)


def unified_sample_posterior_predictive(
    rng_key,
    type_id,
    count,
    sum_x,
    sum_x_sq,
    cat_counts,
    sum_sin,
    sum_cos,
    mu,
    r,
    s,
    nu,
    dir_alpha,
    alpha,
    beta,
    kappa,
    vm_mu,
):
    """Sample from posterior predictive for any column type without Python branching.

    Computes ALL type samples and selects the correct one via jnp.where.
    This wastes trivial compute but enables full JIT compilation.

    Args:
        rng_key: PRNG key for sampling.
        type_id: Integer column type ID.
        count, sum_x, sum_x_sq, cat_counts, sum_sin, sum_cos: Sufficient statistics.
        mu, r, s, nu, dir_alpha, alpha, beta, kappa, vm_mu: Hyperparameters.

    Returns:
        Scalar sample from the posterior predictive distribution.
    """
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)

    cont_sample = _ng_sample(k1, count, sum_x, sum_x_sq, mu, r, s, nu)
    cat_sample = _dc_sample(k2, count, cat_counts, dir_alpha)
    binary_sample = _bb_sample(k3, count, sum_x, alpha, beta)
    cyclic_sample = _vm_sample(k4, count, sum_sin, sum_cos, kappa, vm_mu)
    ordinal_sample = _dc_sample(k2, count, cat_counts, jnp.ones_like(dir_alpha))

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
