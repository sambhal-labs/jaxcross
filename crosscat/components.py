"""Conjugate component models for CrossCat.

Each component model provides:
- Sufficient statistic computation from data (NaN-aware)
- Log marginal likelihood (collapsed — no per-observation parameters)
- Posterior predictive density and sampling

Original CrossCat (probcomp/crosscat) component models:
- ContinuousComponentModel (Normal-Inverse-Gamma)  -> NormalGamma
- MultinomialComponentModel (Dirichlet-Categorical) -> DirichletCategorical
- CyclicComponentModel (Von Mises)                  -> VonMises

New component models for LaborLens use cases:
- OrderedLogistic — ordinal data (wage levels I-IV)
- BetaBernoulli — binary flags (h1b_dependent, willful_violator)

All sufficient_statistics methods filter NaN values, matching the original
CrossCat behavior where missing data is transparently skipped.

NOTE: All component models use collapsed (conjugate) inference — cluster
parameters are integrated out analytically, and only assignments and
hyperparameters are sampled. Uncollapsed inference (Neal Algorithm 8 with
explicit per-cluster parameters) is not implemented. This is sufficient for
all current conjugate models.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln

from crosscat.types import LOG_EPS, ColumnHypers, ColumnType, SufficientStats, log_bessel_i0


def _filter_nan(data: Array) -> Array:
    """Remove NaN values from a 1D data array.

    Matches original CrossCat behavior where NaN observations are
    transparently skipped during sufficient statistic accumulation.
    """
    return data[~jnp.isnan(data)]


# ---------------------------------------------------------------------------
# Normal-Gamma (continuous columns)
# Maps to original ContinuousComponentModel.cpp
# Conjugate model: Normal likelihood with Normal-Inverse-Gamma prior
# Prior: mu | sigma^2 ~ N(mu_0, sigma^2 / r)
#         sigma^2 ~ IG(nu/2, nu*s/2)
# ---------------------------------------------------------------------------


class NormalGamma:
    """Normal-Gamma conjugate model for continuous data.

    Sufficient statistics: count, sum_x, sum_x_sq
    Hyperparameters: mu (prior mean), r (prior precision scale),
                     s (prior variance scale), nu (prior df)

    Maps to original ContinuousComponentModel in cpp_code/src/.
    """

    @staticmethod
    def sufficient_statistics(data: Array) -> SufficientStats:
        """Compute sufficient statistics from data vector.

        Maps to original ContinuousComponentModel::insert_element() accumulation.
        NaN values are filtered before accumulation.

        Args:
            data: 1D array of continuous observations (may contain NaN).

        Returns:
            SufficientStats with count, sum_x, sum_x_sq.
        """
        clean = _filter_nan(data)
        return SufficientStats(
            column_type=ColumnType.CONTINUOUS,
            count=jnp.array(clean.shape[0], dtype=jnp.int32),
            sum_x=jnp.sum(clean),
            sum_x_sq=jnp.sum(clean**2),
        )

    @staticmethod
    def log_marginal_likelihood(suffstats: SufficientStats, hypers: ColumnHypers) -> Array:
        """Log marginal likelihood p(data | hypers) with parameters integrated out.

        Uses the Normal-Inverse-Gamma conjugate integral. The result is:
        log p(x_1..x_n | mu0, r, s, nu) =
            -n/2 log(2pi) + 1/2 log(r/r_n)
            + nu/2 log(nu*s/2) - nu_n/2 log(nu_n*s_n/2)
            + gammaln(nu_n/2) - gammaln(nu/2)

        where the posterior parameters are:
            r_n = r + n
            mu_n = (r*mu0 + sum_x) / r_n
            nu_n = nu + n
            nu_n*s_n = nu*s + sum_x_sq - sum_x^2/n - r_n*(mu_n - mu0)^2 + ...

        Maps to original numerics.cpp::calc_continuous_logp().

        Args:
            suffstats: Cluster sufficient statistics.
            hypers: Column hyperparameters (mu, r, s, nu).

        Returns:
            Scalar log marginal likelihood.
        """
        n = suffstats.count.astype(jnp.float32)
        sum_x = suffstats.sum_x
        sum_x_sq = suffstats.sum_x_sq

        mu0 = hypers.mu
        r = hypers.r
        s = hypers.s
        nu = hypers.nu

        # Posterior parameters
        r_n = r + n
        nu_n = nu + n
        nu_s = nu * s
        nu_n_s_n = (
            nu_s
            + sum_x_sq
            - sum_x**2 / jnp.maximum(n, 1.0)
            + r * n * (mu0 - sum_x / jnp.maximum(n, 1.0)) ** 2 / r_n
        )

        # Log marginal likelihood
        log_ml = (
            -0.5 * n * jnp.log(2.0 * jnp.pi)
            + 0.5 * jnp.log(r / r_n)
            + 0.5 * nu * jnp.log(nu_s / 2.0)
            - 0.5 * nu_n * jnp.log(nu_n_s_n / 2.0)
            + gammaln(nu_n / 2.0)
            - gammaln(nu / 2.0)
        )
        return log_ml

    @staticmethod
    def posterior_predictive_logp(
        x: Array, suffstats: SufficientStats, hypers: ColumnHypers
    ) -> Array:
        """Log posterior predictive density p(x_new | data, hypers).

        The posterior predictive is a Student-t distribution:
            x_new | data ~ t_{nu_n}(mu_n, s_n * (1 + 1/r_n))

        Args:
            x: New observation(s) to evaluate.
            suffstats: Cluster sufficient statistics.
            hypers: Column hyperparameters.

        Returns:
            Log predictive density at x.
        """
        n = suffstats.count.astype(jnp.float32)
        sum_x = suffstats.sum_x
        sum_x_sq = suffstats.sum_x_sq

        mu0 = hypers.mu
        r = hypers.r
        s = hypers.s
        nu = hypers.nu

        # Posterior parameters
        r_n = r + n
        mu_n = (r * mu0 + sum_x) / r_n
        nu_n = nu + n
        nu_s = nu * s
        nu_n_s_n = (
            nu_s
            + sum_x_sq
            - sum_x**2 / jnp.maximum(n, 1.0)
            + r * n * (mu0 - sum_x / jnp.maximum(n, 1.0)) ** 2 / r_n
        )
        # Handle n=0 case: posterior = prior
        nu_n_s_n = jnp.where(n > 0, nu_n_s_n, nu_s)

        # Student-t parameters
        df = nu_n
        loc = mu_n
        scale_sq = (nu_n_s_n / nu_n) * (1.0 + 1.0 / r_n)
        scale = jnp.sqrt(jnp.maximum(scale_sq, LOG_EPS))

        # Student-t log pdf
        z = (x - loc) / scale
        log_p = (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * jnp.log(df * jnp.pi)
            - jnp.log(scale)
            - (df + 1.0) / 2.0 * jnp.log(1.0 + z**2 / df)
        )
        return log_p

    @staticmethod
    def sample_posterior_predictive(
        rng_key: Array, suffstats: SufficientStats, hypers: ColumnHypers, n: int = 1
    ) -> Array:
        """Draw samples from posterior predictive distribution.

        Samples from Student-t by: z ~ t_df, then x = loc + scale * z.

        Args:
            rng_key: JAX PRNG key.
            suffstats: Cluster sufficient statistics.
            hypers: Column hyperparameters.
            n: Number of samples.

        Returns:
            Array of shape (n,) with samples.
        """
        n_obs = suffstats.count.astype(jnp.float32)
        sum_x = suffstats.sum_x
        sum_x_sq = suffstats.sum_x_sq

        mu0 = hypers.mu
        r = hypers.r
        s = hypers.s
        nu = hypers.nu

        r_n = r + n_obs
        mu_n = (r * mu0 + sum_x) / r_n
        nu_n = nu + n_obs
        nu_s = nu * s
        nu_n_s_n = (
            nu_s
            + sum_x_sq
            - sum_x**2 / jnp.maximum(n_obs, 1.0)
            + r * n_obs * (mu0 - sum_x / jnp.maximum(n_obs, 1.0)) ** 2 / r_n
        )
        nu_n_s_n = jnp.where(n_obs > 0, nu_n_s_n, nu_s)

        df = nu_n
        loc = mu_n
        scale = jnp.sqrt((nu_n_s_n / nu_n) * (1.0 + 1.0 / r_n))

        # Sample from Student-t: t = Normal / sqrt(Chi2/df)
        k1, k2 = jax.random.split(rng_key)
        z = jax.random.normal(k1, shape=(n,))
        chi2 = jax.random.chisquare(k2, df, shape=(n,))
        t = z / jnp.sqrt(chi2 / df)
        return loc + scale * t


# ---------------------------------------------------------------------------
# Dirichlet-Categorical (categorical columns)
# Maps to original MultinomialComponentModel.cpp
# Conjugate model: Categorical likelihood with symmetric Dirichlet prior
# ---------------------------------------------------------------------------


class DirichletCategorical:
    """Dirichlet-Categorical conjugate model for categorical data.

    Sufficient statistics: count, category_counts (histogram)
    Hyperparameters: dirichlet_alpha (symmetric concentration)

    Maps to original MultinomialComponentModel in cpp_code/src/.
    """

    @staticmethod
    def sufficient_statistics(data: Array, n_categories: int) -> SufficientStats:
        """Compute sufficient statistics from categorical data.

        NaN values are filtered before accumulation.

        Args:
            data: 1D array of integer category indices (may contain NaN).
            n_categories: Number of possible categories.

        Returns:
            SufficientStats with count and category_counts.
        """
        clean = _filter_nan(data)
        return SufficientStats(
            column_type=ColumnType.CATEGORICAL,
            count=jnp.array(clean.shape[0], dtype=jnp.int32),
            category_counts=jnp.bincount(clean.astype(jnp.int32), length=n_categories),
        )

    @staticmethod
    def log_marginal_likelihood(suffstats: SufficientStats, hypers: ColumnHypers) -> Array:
        """Log marginal likelihood using Dirichlet-Multinomial conjugacy.

        log p(data | alpha) = log [B(counts + alpha) / B(alpha)]
            = sum_k gammaln(count_k + alpha) - gammaln(N + K*alpha)
              - K*gammaln(alpha) + gammaln(K*alpha)

        Maps to original numerics.cpp::calc_multinomial_logp().

        Args:
            suffstats: Cluster sufficient statistics.
            hypers: Column hyperparameters (dirichlet_alpha).

        Returns:
            Scalar log marginal likelihood.
        """
        counts = suffstats.category_counts
        alpha = hypers.dirichlet_alpha
        n = suffstats.count.astype(jnp.float32)
        k = counts.shape[0]

        log_ml = (
            jnp.sum(gammaln(counts + alpha))
            - gammaln(n + k * alpha)
            - k * gammaln(alpha)
            + gammaln(k * alpha)
        )
        return log_ml

    @staticmethod
    def posterior_predictive_logp(
        x: Array, suffstats: SufficientStats, hypers: ColumnHypers
    ) -> Array:
        """Log posterior predictive probability for a category.

        p(x_new = k | data, alpha) = (count_k + alpha) / (N + K * alpha)

        Args:
            x: Category index/indices to evaluate.
            suffstats: Cluster sufficient statistics.
            hypers: Column hyperparameters.

        Returns:
            Log predictive probability.
        """
        counts = suffstats.category_counts
        alpha = hypers.dirichlet_alpha
        n = suffstats.count.astype(jnp.float32)
        k = counts.shape[0]

        probs = (counts + alpha) / (n + k * alpha)
        return jnp.log(probs[x.astype(jnp.int32)])

    @staticmethod
    def sample_posterior_predictive(
        rng_key: Array, suffstats: SufficientStats, hypers: ColumnHypers, n: int = 1
    ) -> Array:
        """Draw samples from posterior predictive (categorical distribution).

        Args:
            rng_key: JAX PRNG key.
            suffstats: Cluster sufficient statistics.
            hypers: Column hyperparameters.
            n: Number of samples.

        Returns:
            Array of shape (n,) with category indices.
        """
        counts = suffstats.category_counts
        alpha = hypers.dirichlet_alpha
        n_obs = suffstats.count.astype(jnp.float32)
        k = counts.shape[0]

        probs = (counts + alpha) / (n_obs + k * alpha)
        return jax.random.categorical(rng_key, jnp.log(probs), shape=(n,))


# ---------------------------------------------------------------------------
# Ordered Logistic (ordinal columns — NEW, not in original CrossCat)
# For wage levels I-IV and similar ordinal data
# Implemented as Dirichlet-Categorical with ordered structure preserved
# ---------------------------------------------------------------------------


class OrderedLogistic:
    """Ordered logistic model for ordinal data.

    Uses a cumulative logistic link function with cutpoints:
    P(Y = k | μ, cutpoints) = σ(c_k - μ) - σ(c_{k-1} - μ)

    Each cluster has a location parameter μ (integrated out via grid).
    Cutpoints c₁ < c₂ < ... < c_{K-1} are per-column hyperparameters.
    Scale σ = 1 (fixed).

    Prior: μ ~ N(hypers.mu, hypers.s)

    Sufficient statistics: count, level_counts (histogram over ordered levels)
    Hyperparameters: cutpoints, mu (prior mean), s (prior variance)
    """

    _N_GRID = 31

    @staticmethod
    def sufficient_statistics(data: Array, n_levels: int) -> SufficientStats:
        """Compute sufficient statistics from ordinal data. NaN-aware."""
        clean = _filter_nan(data)
        return SufficientStats(
            column_type=ColumnType.ORDINAL,
            count=jnp.array(clean.shape[0], dtype=jnp.int32),
            category_counts=jnp.bincount(clean.astype(jnp.int32), length=n_levels),
        )

    @staticmethod
    def _level_probs(mu: Array, cutpoints: Array) -> Array:
        """Compute P(Y=k | μ, cutpoints) for all K levels."""
        extended = jnp.concatenate([jnp.array([-1e10]), cutpoints, jnp.array([1e10])])
        cum = jax.nn.sigmoid(extended - mu)
        probs = cum[1:] - cum[:-1]
        return jnp.maximum(probs, 1e-30)

    @staticmethod
    def log_marginal_likelihood(suffstats: SufficientStats, hypers: ColumnHypers) -> Array:
        """Log marginal likelihood via grid integration over μ.

        Non-conjugate: integrates p(counts | μ, cutpoints) · p(μ) over μ.
        """
        counts = suffstats.category_counts.astype(jnp.float32)
        n = suffstats.count.astype(jnp.float32)
        cutpoints = hypers.cutpoints
        mu0 = float(hypers.mu) if hypers.mu is not None else 0.0
        s0 = float(hypers.s) if hypers.s is not None else 4.0
        s0 = max(s0, 1e-30)

        half_range = 4.0 * jnp.sqrt(s0)
        mu_grid = jnp.linspace(mu0 - half_range, mu0 + half_range, OrderedLogistic._N_GRID)
        delta_mu = 2.0 * half_range / (OrderedLogistic._N_GRID - 1)

        def log_score(mu):
            probs = OrderedLogistic._level_probs(mu, cutpoints)
            log_lik = jnp.sum(counts * jnp.log(probs))
            log_prior = -0.5 * jnp.log(2 * jnp.pi * s0) - 0.5 * (mu - mu0) ** 2 / s0
            return log_lik + log_prior

        log_scores = jax.vmap(log_score)(mu_grid)
        return jnp.where(n > 0, jax.nn.logsumexp(log_scores) + jnp.log(delta_mu), 0.0)

    @staticmethod
    def posterior_predictive_logp(
        x: Array, suffstats: SufficientStats, hypers: ColumnHypers
    ) -> Array:
        """Posterior predictive log p(x=k | data, cutpoints)."""
        counts = suffstats.category_counts.astype(jnp.float32)
        cutpoints = hypers.cutpoints
        mu0 = float(hypers.mu) if hypers.mu is not None else 0.0
        s0 = float(hypers.s) if hypers.s is not None else 4.0
        s0 = max(s0, 1e-30)

        half_range = 4.0 * jnp.sqrt(s0)
        mu_grid = jnp.linspace(mu0 - half_range, mu0 + half_range, OrderedLogistic._N_GRID)

        def log_post(mu):
            probs = OrderedLogistic._level_probs(mu, cutpoints)
            return jnp.sum(counts * jnp.log(probs)) - 0.5 * (mu - mu0) ** 2 / s0

        log_w = jax.vmap(log_post)(mu_grid)
        log_w = log_w - jax.nn.logsumexp(log_w)
        weights = jnp.exp(log_w)

        all_probs = jax.vmap(lambda mu: OrderedLogistic._level_probs(mu, cutpoints))(mu_grid)
        avg_probs = jnp.sum(weights[:, None] * all_probs, axis=0)
        return jnp.log(jnp.maximum(avg_probs[x.astype(jnp.int32)], 1e-30))

    @staticmethod
    def sample_posterior_predictive(
        rng_key: Array, suffstats: SufficientStats, hypers: ColumnHypers, n: int = 1
    ) -> Array:
        """Draw samples from posterior predictive over ordinal levels."""
        counts = suffstats.category_counts.astype(jnp.float32)
        cutpoints = hypers.cutpoints
        mu0 = float(hypers.mu) if hypers.mu is not None else 0.0
        s0 = float(hypers.s) if hypers.s is not None else 4.0
        s0 = max(s0, 1e-30)

        half_range = 4.0 * jnp.sqrt(s0)
        mu_grid = jnp.linspace(mu0 - half_range, mu0 + half_range, OrderedLogistic._N_GRID)

        def log_post(mu):
            probs = OrderedLogistic._level_probs(mu, cutpoints)
            return jnp.sum(counts * jnp.log(probs)) - 0.5 * (mu - mu0) ** 2 / s0

        log_w = jax.vmap(log_post)(mu_grid)
        log_w = log_w - jax.nn.logsumexp(log_w)
        weights = jnp.exp(log_w)

        all_probs = jax.vmap(lambda mu: OrderedLogistic._level_probs(mu, cutpoints))(mu_grid)
        avg_probs = jnp.sum(weights[:, None] * all_probs, axis=0)
        return jax.random.categorical(rng_key, jnp.log(jnp.maximum(avg_probs, 1e-30)), shape=(n,))


# ---------------------------------------------------------------------------
# Beta-Bernoulli (binary columns — NEW, not in original CrossCat)
# For binary flags like h1b_dependent, willful_violator
# ---------------------------------------------------------------------------


class BetaBernoulli:
    """Beta-Bernoulli conjugate model for binary data.

    Not present in original CrossCat. Added for LaborLens binary flag analysis.

    Sufficient statistics: count, sum_x (number of 1s)
    Hyperparameters: alpha, beta (Beta prior parameters)
    """

    @staticmethod
    def sufficient_statistics(data: Array) -> SufficientStats:
        """Compute sufficient statistics from binary data. NaN-aware."""
        clean = _filter_nan(data)
        return SufficientStats(
            column_type=ColumnType.BINARY,
            count=jnp.array(clean.shape[0], dtype=jnp.int32),
            sum_x=jnp.sum(clean),
        )

    @staticmethod
    def log_marginal_likelihood(suffstats: SufficientStats, hypers: ColumnHypers) -> Array:
        """Log marginal likelihood using Beta-Binomial conjugacy.

        log p(data | alpha, beta) = gammaln(alpha + beta) - gammaln(n + alpha + beta)
            + gammaln(k + alpha) - gammaln(alpha)
            + gammaln(n - k + beta) - gammaln(beta)

        where k = sum_x (number of 1s), n = count.
        """
        n = suffstats.count.astype(jnp.float32)
        k = suffstats.sum_x
        a = hypers.alpha
        b = hypers.beta

        log_ml = (
            gammaln(a + b)
            - gammaln(n + a + b)
            + gammaln(k + a)
            - gammaln(a)
            + gammaln(n - k + b)
            - gammaln(b)
        )
        return log_ml

    @staticmethod
    def posterior_predictive_logp(
        x: Array, suffstats: SufficientStats, hypers: ColumnHypers
    ) -> Array:
        """Log posterior predictive probability for a binary outcome.

        p(x=1 | data) = (k + alpha) / (n + alpha + beta)
        """
        n = suffstats.count.astype(jnp.float32)
        k = suffstats.sum_x
        a = hypers.alpha
        b = hypers.beta

        p1 = (k + a) / (n + a + b)
        return jnp.where(
            x > 0.5,
            jnp.log(jnp.maximum(p1, LOG_EPS)),
            jnp.log(jnp.maximum(1.0 - p1, LOG_EPS)),
        )

    @staticmethod
    def sample_posterior_predictive(
        rng_key: Array, suffstats: SufficientStats, hypers: ColumnHypers, n: int = 1
    ) -> Array:
        """Draw samples from posterior predictive (Bernoulli)."""
        n_obs = suffstats.count.astype(jnp.float32)
        k = suffstats.sum_x
        a = hypers.alpha
        b = hypers.beta

        p1 = (k + a) / (n_obs + a + b)
        return jax.random.bernoulli(rng_key, p1, shape=(n,)).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Von Mises (cyclic/circular columns)
# Maps to original CyclicComponentModel.cpp
# For angular/directional data on [0, 2*pi)
# ---------------------------------------------------------------------------


# Alias for backward compatibility with internal callers
_log_bessel_i0 = log_bessel_i0


def _von_mises_sample_best_fisher(key: Array, mu: Array, kappa: Array) -> Array:
    """Sample from von Mises(mu, kappa) using the Best-Fisher algorithm.

    Best & Fisher (1979): acceptance rate > 50% for all kappa.
    This is the standard algorithm used by NumPy/SciPy for von Mises sampling.
    """
    # Best-Fisher parameters
    tau = 1.0 + jnp.sqrt(1.0 + 4.0 * kappa**2)
    rho = (tau - jnp.sqrt(2.0 * tau)) / (2.0 * kappa)
    r = (1.0 + rho**2) / (2.0 * rho)

    max_iters = 1000

    def _cond(state):
        return state[0] & (state[1] < max_iters)

    def _body(state):
        _, itr, sample, key_loop = state
        k1, k2, key_loop = jax.random.split(key_loop, 3)

        u1 = jax.random.uniform(k1)
        z = jnp.cos(jnp.pi * u1)
        f = (1.0 + r * z) / (r + z)
        c = kappa * (r - f)

        u2 = jax.random.uniform(k2)
        accepted = (c * (2.0 - c) > u2) | (jnp.log(c / u2) + 1.0 >= c)

        # f is cos(theta), so theta = arccos(f), with random sign
        theta = jnp.arccos(jnp.clip(f, -1.0, 1.0))
        return (~accepted, itr + 1, jnp.where(accepted, theta, sample), key_loop)

    _, _, theta, _ = jax.lax.while_loop(
        _cond, _body, (jnp.bool_(True), jnp.int32(0), jnp.float32(0.0), key)
    )

    # Random sign and shift to mu
    key, subkey = jax.random.split(key)
    sign = 2.0 * jax.random.bernoulli(subkey).astype(jnp.float32) - 1.0
    return (mu + sign * theta) % (2.0 * jnp.pi)


class VonMises:
    """Von Mises conjugate model for circular/directional data.

    Maps to original CyclicComponentModel in cpp_code/src/.

    Data is on [0, 2*pi). The model uses a von Mises likelihood with
    conjugate prior on the mean direction.

    Sufficient statistics: count, sum_sin, sum_cos
    Hyperparameters:
        kappa — likelihood concentration
        vm_a  — prior concentration on mean direction (a in original)
        vm_mu — prior mean direction (b in original)
    """

    @staticmethod
    def sufficient_statistics(data: Array) -> SufficientStats:
        """Compute sufficient statistics from circular data. NaN-aware."""
        clean = _filter_nan(data)
        return SufficientStats(
            column_type=ColumnType.CYCLIC,
            count=jnp.array(clean.shape[0], dtype=jnp.int32),
            sum_sin=jnp.sum(jnp.sin(clean)),
            sum_cos=jnp.sum(jnp.cos(clean)),
        )

    @staticmethod
    def log_marginal_likelihood(suffstats: SufficientStats, hypers: ColumnHypers) -> Array:
        """Log marginal likelihood for circular data.

        Exact conjugate marginal obtained by integrating out the mean direction
        mu from the von Mises likelihood × von Mises prior:

            p(data | kappa, a, b) = [1/(2*pi)]^n * I_0(R) / [I_0(kappa)^n * I_0(a)]

        where R = ||(sum_sin + a*sin(b), sum_cos + a*cos(b))|| is the posterior
        resultant length combining data with the prior contribution.

        Hyperparameters:
            kappa — likelihood concentration
            vm_a  — prior concentration on mean direction
            vm_mu — prior mean direction (b)

        See Mardia & Jupp (2000), Section 5.3.
        Maps to original calc_marginal_logp in CyclicComponentModel.
        """
        n = suffstats.count.astype(jnp.float32)
        kappa = hypers.kappa
        a = hypers.vm_a
        b = hypers.vm_mu

        # Posterior resultant length: data + prior(a, b)
        total_sin = suffstats.sum_sin + a * jnp.sin(b)
        total_cos = suffstats.sum_cos + a * jnp.cos(b)
        R = jnp.sqrt(total_sin**2 + total_cos**2)

        # log p(data) = -n*log(2*pi) + log I_0(R) - n*log I_0(kappa) - log I_0(a)
        log_ml = (
            -n * jnp.log(2.0 * jnp.pi)
            + _log_bessel_i0(R)
            - n * _log_bessel_i0(kappa)
            - _log_bessel_i0(a)
        )
        return log_ml

    @staticmethod
    def posterior_predictive_logp(
        x: Array, suffstats: SufficientStats, hypers: ColumnHypers
    ) -> Array:
        """Log posterior predictive density for a circular observation.

        Approximation: posterior von Mises with concentration = kappa and
        mean direction from the posterior resultant vector.
        """
        kappa = hypers.kappa
        a = hypers.vm_a

        # Posterior mean direction from resultant of data + prior(a, b)
        total_sin = suffstats.sum_sin + a * jnp.sin(hypers.vm_mu)
        total_cos = suffstats.sum_cos + a * jnp.cos(hypers.vm_mu)
        mu_post = jnp.arctan2(total_sin, total_cos)

        # Von Mises log pdf with likelihood concentration kappa
        log_p = kappa * jnp.cos(x - mu_post) - jnp.log(2.0 * jnp.pi) - _log_bessel_i0(kappa)
        return log_p

    @staticmethod
    def sample_posterior_predictive(
        rng_key: Array, suffstats: SufficientStats, hypers: ColumnHypers, n: int = 1
    ) -> Array:
        """Draw samples from posterior predictive von Mises.

        Uses the Best-Fisher algorithm (Best & Fisher, 1979) which has
        acceptance rate > 50% for all kappa, replacing the previous uniform
        rejection sampler that degraded badly for small kappa and silently
        fell back to 0.0 after 1000 iterations.
        """
        kappa = hypers.kappa
        a = hypers.vm_a

        total_sin = suffstats.sum_sin + a * jnp.sin(hypers.vm_mu)
        total_cos = suffstats.sum_cos + a * jnp.cos(hypers.vm_mu)
        mu_post = jnp.arctan2(total_sin, total_cos)

        def _sample_one(key):
            return _von_mises_sample_best_fisher(key, mu_post, kappa)

        keys = jax.random.split(rng_key, n)
        samples = jax.vmap(_sample_one)(keys)
        return samples


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

_COMPONENT_MAP = {
    ColumnType.CONTINUOUS: NormalGamma,
    ColumnType.CATEGORICAL: DirichletCategorical,
    ColumnType.ORDINAL: OrderedLogistic,
    ColumnType.BINARY: BetaBernoulli,
    ColumnType.CYCLIC: VonMises,
}


def get_component(column_type: ColumnType):
    """Return the component model class for a given column type."""
    return _COMPONENT_MAP[column_type]
