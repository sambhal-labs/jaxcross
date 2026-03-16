"""OFLC dataset benchmark for jax-crosscat.

Trains CrossCat on the AI/ML Engineering SOC cluster (~420K rows) from
DOL OFLC H1B/LCA data and compares against baselines:
- Original CrossCat (C++/Python 2)
- scikit-learn BayesianGaussianMixture
- TabPFN

Metrics:
- Held-out log-likelihood
- Conditional predictive accuracy (wage | SOC, city, wage_level)
- Wall-clock training time (CPU vs GPU)
- Credible interval calibration

This benchmark will be run as part of the arXiv paper (Week 8-9).
"""

from __future__ import annotations


def main():
    raise NotImplementedError("OFLC benchmark — Week 7-8 deliverable")


if __name__ == "__main__":
    main()
