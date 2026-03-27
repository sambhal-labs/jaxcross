# Component Models

::: crosscat.components
    options:
      show_source: false

## Overview

Bayesian conjugate component models. Each column type uses a specific model where parameters are analytically integrated out — only sufficient statistics are stored.

## Common Interface

All component models provide four methods:

| Method | Purpose |
|--------|---------|
| `sufficient_statistics(data, ...)` | Compute sufficient statistics from data |
| `log_marginal_likelihood(suffstats, hypers)` | Log p(data \| hypers) with parameters marginalized |
| `posterior_predictive_logp(x, suffstats, hypers)` | Log p(x_new \| data, hypers) |
| `sample_posterior_predictive(rng_key, suffstats, hypers, n=1)` | Draw from predictive distribution |

## `NormalGamma` (Continuous)

For real-valued data. Uses a Normal-Inverse-Gamma conjugate prior.

**Hyperparameters:** `mu` (prior mean), `r` (prior count), `s` (prior sum-of-squares), `nu` (prior degrees of freedom)

**Sufficient statistics:** `count`, `sum_x`, `sum_x_sq`

```python
from crosscat.components import NormalGamma

ss = NormalGamma.sufficient_statistics(data_column)
logp = NormalGamma.log_marginal_likelihood(ss, hypers)
pred = NormalGamma.posterior_predictive_logp(x_new, ss, hypers)
samples = NormalGamma.sample_posterior_predictive(key, ss, hypers, n=100)
```

## `DirichletCategorical` (Categorical)

For unordered discrete labels (0, 1, 2, ...). Uses a Dirichlet-Categorical conjugate prior.

**Hyperparameters:** `dirichlet_alpha` (concentration per category)

**Sufficient statistics:** `count`, `category_counts`

```python
from crosscat.components import DirichletCategorical

ss = DirichletCategorical.sufficient_statistics(data_column, n_categories=5)
logp = DirichletCategorical.log_marginal_likelihood(ss, hypers)
```

## `BetaBernoulli` (Binary)

For 0/1 data. Uses a Beta-Bernoulli conjugate prior.

**Hyperparameters:** `alpha` (prior successes), `beta` (prior failures)

**Sufficient statistics:** `count`, `category_counts` (counts of 0s and 1s)

```python
from crosscat.components import BetaBernoulli

ss = BetaBernoulli.sufficient_statistics(data_column)
logp = BetaBernoulli.log_marginal_likelihood(ss, hypers)
```

## `OrderedLogistic` (Ordinal)

For ordered categorical data (1, 2, 3, ...). Uses a cumulative link function with grid integration over a latent location parameter. **Non-conjugate** — uses 31-point grid integration.

**Hyperparameters:** `cutpoints` (ordered thresholds), `mu` (latent location), `s` (prior variance)

**Sufficient statistics:** `count`, `category_counts`

The cumulative link function is: P(Y=k | mu, cutpoints) = sigma(c_k - mu) - sigma(c_{k-1} - mu)

```python
from crosscat.components import OrderedLogistic

ss = OrderedLogistic.sufficient_statistics(data_column, n_levels=5)
logp = OrderedLogistic.log_marginal_likelihood(ss, hypers)
```

## `VonMises` (Cyclic)

For angular/circular data in [0, 2*pi). Uses a Von Mises likelihood with conjugate prior.

**Hyperparameters:** `kappa` (likelihood concentration), `vm_a` (prior concentration), `vm_mu` (prior mean direction)

**Sufficient statistics:** `count`, `sum_sin`, `sum_cos`

```python
from crosscat.components import VonMises

ss = VonMises.sufficient_statistics(data_column)
logp = VonMises.log_marginal_likelihood(ss, hypers)
```

## Helper

### `get_component`

```python
get_component(column_type: ColumnType) -> type
```

Return the component model class for a given column type.

```python
from crosscat.components import get_component
from crosscat.types import ColumnType

model_class = get_component(ColumnType.CONTINUOUS)  # returns NormalGamma
```
