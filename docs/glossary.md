# Glossary

Key terms used throughout the jax-crosscat documentation.

---

**Chinese Restaurant Process (CRP)**
:   A distribution over partitions used as a prior for cluster assignments. New items join existing clusters proportionally to their size, or start a new cluster with probability proportional to the concentration parameter `alpha`. CrossCat uses CRP at two levels: one for partitioning columns into views, another (per view) for clustering rows.

**Collapsed Inference**
:   A technique where component model parameters (means, variances, etc.) are analytically integrated out, leaving only discrete cluster assignments to be sampled. This reduces the dimensionality of the sampling problem and typically improves mixing. Four of the five component models in jax-crosscat use collapsed inference via conjugate priors. The ordered logistic model uses grid integration over a latent location parameter because it lacks a conjugate prior.

**Conjugate Prior**
:   A prior distribution that, when combined with a particular likelihood, yields a posterior in the same family. For example, a Normal-Gamma prior is conjugate to a Normal likelihood. Conjugacy enables closed-form computation of marginal likelihoods and posterior predictives — the foundation of collapsed inference in CrossCat.

**Credible Interval**
:   The Bayesian analog of a confidence interval. A 95% credible interval means there is a 95% posterior probability that the true value falls within the interval. Computed via `credible_interval()` by sampling from the posterior predictive.

**Dependence Matrix (Z-matrix)**
:   An N x N matrix (where N = number of columns) showing the posterior probability that each pair of columns belongs to the same view. Values near 1.0 indicate strong statistical dependence; values near 0.0 indicate independence. Computed via `dependence_matrix()`.

**Dirichlet Process (DP)**
:   A distribution over probability distributions, used as a nonparametric prior that allows the number of clusters to grow with the data. The CRP is the sequential representation of a DP. CrossCat uses a two-level DP: outer (columns into views) and inner (rows into clusters per view).

**Gibbs Sampling**
:   A Markov Chain Monte Carlo (MCMC) method that samples each variable in turn from its conditional distribution given the current values of all other variables. In CrossCat, Gibbs sampling is used to update row assignments, column assignments, hyperparameters, and CRP concentration parameters.

**Gibbs Sweep**
:   One complete iteration of all Gibbs sampling kernels: row assignments, column assignments, hyperparameter transitions, and CRP alpha transitions. Multiple sweeps are needed for convergence. Run via `packed_gibbs_sweep(key, packed, data, n_sweeps=100)`.

**Hyperparameter**
:   Parameters of the prior distributions on component model parameters. For example, the Normal-Gamma model has hyperparameters `mu` (prior mean), `r` (prior precision scale), `s` (prior variance scale), and `nu` (prior degrees of freedom). These are sampled during inference via grid-based Gibbs transitions.

**Log-Joint**
:   The log probability of the data and all latent variables (cluster assignments, hyperparameters) under the model. Used as a convergence diagnostic — when log-joint plateaus across multiple chains, the sampler has likely converged. Computed via `log_joint()`.

**Marginal Likelihood**
:   The probability of the observed data in a cluster after integrating out all component parameters. Used to score how well a cluster explains its assigned data points. Each component model provides `log_marginal_likelihood()`.

**Mutual Information (MI)**
:   A measure of the information shared between two columns. Higher MI means knowing one column's value tells you more about the other. Estimated via Monte Carlo sampling in `mutual_information()`. Related to (but distinct from) the dependence matrix.

**Packed State**
:   A representation of CrossCat state as fixed-size JAX arrays with padding, enabling JIT compilation. Variable-size Python lists (views, clusters) are converted to padded arrays with known maximum dimensions. See `pack_state()` / `unpack_state()`.

**Posterior Predictive**
:   The distribution over new data points, integrating over all posterior uncertainty in cluster assignments and parameters. Used by all query functions (sampling, probability, CDF, anomaly scoring). "Fully Bayesian" means predictions account for parameter uncertainty, not just point estimates.

**Sufficient Statistics**
:   Compact summaries of the data in each cluster that are sufficient to compute likelihoods. For example, the Normal-Gamma model tracks `(count, sum, sum_of_squares)` per cluster. Sufficient statistics enable efficient incremental updates when rows are added or removed from clusters.

**View**
:   A group of columns that CrossCat has inferred to be statistically related. Each view has its own independent clustering of rows. For example, in an employee dataset, `{salary, experience}` might form one view (clustering by seniority) while `{zip_code, commute}` forms another (clustering by geography). The number and composition of views is inferred automatically.
