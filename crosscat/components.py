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

from jax import Array

from crosscat.types import ColumnHypers, SufficientStats

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
        raise NotImplementedError

    @staticmethod
    def log_marginal_likelihood(suffstats: SufficientStats, hypers: ColumnHypers) -> Array:
        """Log marginal likelihood p(data | hypers) with parameters integrated out.

        Uses the Normal-Inverse-Gamma conjugate integral:
        log p(x_1, ..., x_n | mu_0, r, s, nu) = log Student-t integral

        Maps to original numerics.cpp::calc_continuous_logp().

        Args:
            suffstats: Cluster sufficient statistics.
            hypers: Column hyperparameters (mu, r, s, nu).

        Returns:
            Scalar log marginal likelihood.
        """
        raise NotImplementedError

    @staticmethod
    def posterior_predictive_logp(
        x: Array, suffstats: SufficientStats, hypers: ColumnHypers
    ) -> Array:
        """Log posterior predictive density p(x_new | data, hypers).

        Maps to original sample_utils.py predictive probability computation.

        Args:
            x: New observation(s) to evaluate.
            suffstats: Cluster sufficient statistics.
            hypers: Column hyperparameters.

        Returns:
            Log predictive density at x.
        """
        raise NotImplementedError

    @staticmethod
    def sample_posterior_predictive(
        rng_key: Array, suffstats: SufficientStats, hypers: ColumnHypers, n: int = 1
    ) -> Array:
        """Draw samples from posterior predictive distribution.

        Maps to original sample_utils.py sampling logic.

        Args:
            rng_key: JAX PRNG key.
            suffstats: Cluster sufficient statistics.
            hypers: Column hyperparameters.
            n: Number of samples.

        Returns:
            Array of shape (n,) with samples.
        """
        raise NotImplementedError


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
        raise NotImplementedError

    @staticmethod
    def log_marginal_likelihood(suffstats: SufficientStats, hypers: ColumnHypers) -> Array:
        """Log marginal likelihood using Dirichlet-Multinomial conjugacy.

        log p(data | alpha) = log [B(counts + alpha) / B(alpha)]
        where B is the multivariate Beta function.

        Maps to original numerics.cpp::calc_multinomial_logp().

        Args:
            suffstats: Cluster sufficient statistics.
            hypers: Column hyperparameters (dirichlet_alpha).

        Returns:
            Scalar log marginal likelihood.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Ordered Logistic (ordinal columns — NEW, not in original CrossCat)
# For wage levels I-IV and similar ordinal data
# ---------------------------------------------------------------------------


class OrderedLogistic:
    """Ordered logistic model for ordinal data.

    Not present in original CrossCat. Added for LaborLens wage level analysis.

    Sufficient statistics: count, level_counts (histogram over ordered levels)
    Hyperparameters: cutpoints (ordered thresholds)
    """

    @staticmethod
    def sufficient_statistics(data: Array, n_levels: int) -> SufficientStats:
        """Compute sufficient statistics from ordinal data."""
        raise NotImplementedError

    @staticmethod
    def log_marginal_likelihood(suffstats: SufficientStats, hypers: ColumnHypers) -> Array:
        """Log marginal likelihood for ordinal observations."""
        raise NotImplementedError

    @staticmethod
    def posterior_predictive_logp(
        x: Array, suffstats: SufficientStats, hypers: ColumnHypers
    ) -> Array:
        """Log posterior predictive probability for an ordinal level."""
        raise NotImplementedError

    @staticmethod
    def sample_posterior_predictive(
        rng_key: Array, suffstats: SufficientStats, hypers: ColumnHypers, n: int = 1
    ) -> Array:
        """Draw samples from posterior predictive over ordinal levels."""
        raise NotImplementedError


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
        raise NotImplementedError

    @staticmethod
    def log_marginal_likelihood(suffstats: SufficientStats, hypers: ColumnHypers) -> Array:
        """Log marginal likelihood using Beta-Binomial conjugacy."""
        raise NotImplementedError

    @staticmethod
    def posterior_predictive_logp(
        x: Array, suffstats: SufficientStats, hypers: ColumnHypers
    ) -> Array:
        """Log posterior predictive probability for a binary outcome."""
        raise NotImplementedError

    @staticmethod
    def sample_posterior_predictive(
        rng_key: Array, suffstats: SufficientStats, hypers: ColumnHypers, n: int = 1
    ) -> Array:
        """Draw samples from posterior predictive (Bernoulli)."""
        raise NotImplementedError
