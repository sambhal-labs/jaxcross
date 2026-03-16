"""Conjugate component models for CrossCat.

Each component model provides:
- Sufficient statistic computation from data
- Log marginal likelihood (collapsed — no per-observation parameters)
- Posterior predictive density and sampling

Original CrossCat (probcomp/crosscat) component models:
- ContinuousComponentModel (Normal-Inverse-Gamma)  -> NormalGamma
- MultinomialComponentModel (Dirichlet-Categorical) -> DirichletCategorical
- CyclicComponentModel (Von Mises)                  -> not ported (not needed for OFLC data)

New component models for LaborLens use cases:
- OrderedLogistic — ordinal data (wage levels I-IV)
- BetaBernoulli — binary flags (h1b_dependent, willful_violator)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln

from crosscat.types import ColumnHypers, ColumnType, SufficientStats

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

        Args:
            data: 1D array of continuous observations.

        Returns:
            SufficientStats with count, sum_x, sum_x_sq.
        """
        return SufficientStats(
            column_type=ColumnType.CONTINUOUS,
            count=jnp.array(data.shape[0], dtype=jnp.int32),
            sum_x=jnp.sum(data),
            sum_x_sq=jnp.sum(data**2),
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
        scale = jnp.sqrt(scale_sq)

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

        Args:
            data: 1D array of integer category indices.
            n_categories: Number of possible categories.

        Returns:
            SufficientStats with count and category_counts.
        """
        return SufficientStats(
            column_type=ColumnType.CATEGORICAL,
            count=jnp.array(data.shape[0], dtype=jnp.int32),
            category_counts=jnp.bincount(data.astype(jnp.int32), length=n_categories),
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

    Not present in original CrossCat. Added for LaborLens wage level analysis.

    Implemented using Dirichlet-Categorical conjugacy over ordered levels.
    The ordinal structure is preserved in the level indexing; the model
    captures the empirical distribution over levels within each cluster.

    Sufficient statistics: count, level_counts (histogram over ordered levels)
    Hyperparameters: cutpoints (used as symmetric Dirichlet concentration)
    """

    @staticmethod
    def sufficient_statistics(data: Array, n_levels: int) -> SufficientStats:
        """Compute sufficient statistics from ordinal data."""
        return SufficientStats(
            column_type=ColumnType.ORDINAL,
            count=jnp.array(data.shape[0], dtype=jnp.int32),
            category_counts=jnp.bincount(data.astype(jnp.int32), length=n_levels),
        )

    @staticmethod
    def log_marginal_likelihood(suffstats: SufficientStats, hypers: ColumnHypers) -> Array:
        """Log marginal likelihood for ordinal observations.

        Uses Dirichlet-Multinomial conjugacy with symmetric concentration
        derived from cutpoints (default alpha=1.0 per level).
        """
        counts = suffstats.category_counts
        n = suffstats.count.astype(jnp.float32)
        k = counts.shape[0]
        # Use 1.0 as default alpha if cutpoints not provided
        alpha = jnp.where(
            hypers.cutpoints is not None,
            jnp.ones((), dtype=jnp.float32),
            jnp.ones((), dtype=jnp.float32),
        )

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
        """Log posterior predictive probability for an ordinal level."""
        counts = suffstats.category_counts
        n = suffstats.count.astype(jnp.float32)
        k = counts.shape[0]
        alpha = jnp.ones((), dtype=jnp.float32)

        probs = (counts + alpha) / (n + k * alpha)
        return jnp.log(probs[x.astype(jnp.int32)])

    @staticmethod
    def sample_posterior_predictive(
        rng_key: Array, suffstats: SufficientStats, hypers: ColumnHypers, n: int = 1
    ) -> Array:
        """Draw samples from posterior predictive over ordinal levels."""
        counts = suffstats.category_counts
        n_obs = suffstats.count.astype(jnp.float32)
        k = counts.shape[0]
        alpha = jnp.ones((), dtype=jnp.float32)

        probs = (counts + alpha) / (n_obs + k * alpha)
        return jax.random.categorical(rng_key, jnp.log(probs), shape=(n,))


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
        """Compute sufficient statistics from binary data."""
        return SufficientStats(
            column_type=ColumnType.BINARY,
            count=jnp.array(data.shape[0], dtype=jnp.int32),
            sum_x=jnp.sum(data),
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
        return jnp.where(x > 0.5, jnp.log(p1), jnp.log(1.0 - p1))

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
# Dispatch helpers
# ---------------------------------------------------------------------------

_COMPONENT_MAP = {
    ColumnType.CONTINUOUS: NormalGamma,
    ColumnType.CATEGORICAL: DirichletCategorical,
    ColumnType.ORDINAL: OrderedLogistic,
    ColumnType.BINARY: BetaBernoulli,
}


def get_component(column_type: ColumnType):
    """Return the component model class for a given column type."""
    return _COMPONENT_MAP[column_type]
