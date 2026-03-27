# MNIST Paper Benchmark

Reproduction of the MNIST experiments from [Mansinghka et al. (2016)](https://jmlr.org/papers/v17/11-392.html), Section 3.2.

## Setup

- 1000 MNIST images downsampled to 16x16 binary pixels
- 256 BetaBernoulli features + 1 Categorical digit label = 257 columns
- 10 independent chains x 100 Gibbs sweeps
- Run on NVIDIA P100 GPU

## Results

### Dependence Matrix (Z-matrix)

Pairwise probability that two columns share the same view, averaged across 10 chains. Block structure reveals groups of dependent pixels. The digit label column (rightmost) shows which pixels carry digit information.

<p align="center">
  <img src="../benchmark-results/mnist-z-matrix.png" alt="Z-matrix" width="600" />
</p>

### Pixel Dependence Spatial Map

Maps each pixel's dependence on the digit label back to the 16x16 grid. Blue = foreground (digit-dependent), magenta = background (independent). Matches Paper Figure 13c.

<p align="center">
  <img src="../benchmark-results/mnist-pixel-dependence.png" alt="Pixel dependence" width="700" />
</p>

### Pixel Inpainting

Predict missing pixels from partial observations. At 30% observed, the model achieves 93.1% pixel accuracy. Matches Paper Figure 14.

<p align="center">
  <img src="../benchmark-results/mnist-inpainting.png" alt="Inpainting" width="700" />
</p>

### Digit-Cluster Contingency

How digits map to inferred row clusters. The model discovers ~30 clusters capturing sub-digit handwriting variation, with clear digit-cluster correspondence.

<p align="center">
  <img src="../benchmark-results/mnist-contingency.png" alt="Contingency" width="800" />
</p>

### Classification ROC

Digit classification via posterior predictive P(digit | pixels) compared against SVM baselines. CrossCat achieves 79% accuracy as a generative model.

<p align="center">
  <img src="../benchmark-results/mnist-classification-roc.png" alt="ROC" width="700" />
</p>

## Metrics Summary

| Metric | Result |
|--------|--------|
| Pixel dependence map | Foreground/background separation matches paper Fig 13c |
| Inpainting accuracy (30% observed) | 93.1% |
| Classification accuracy | 79.0% (generative, no tuning) |
| Posterior views | 4 views (mode across 10 chains) |
| Row clusters | 28-31 per view |
| Total inference time | ~3.5 hours (10 chains x 100 sweeps, P100) |

## Run It Yourself

Open [`benchmarks/mnist_paper_colab.ipynb`](https://github.com/sambhal-labs/jaxcross/blob/main/benchmarks/mnist_paper_colab.ipynb) in Kaggle (P100) or Colab (T4/A100).

The notebook supports checkpoint/resume — long sessions can be interrupted and continued from the last saved state.
