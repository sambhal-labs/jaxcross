# Convergence Diagnostics Guide

## Gelman-Rubin Rhat

Measures how well multiple chains agree. Compares within-chain variance to between-chain variance.

```python
from crosscat.diagnostics import gelman_rubin_rhat

# traces: shape (n_chains, n_samples) — log_joint values per chain
rhat = float(gelman_rubin_rhat(traces))
```

### Interpretation

| Rhat | Status | Action |
|------|--------|--------|
| < 1.01 | Excellent | Chains fully mixed |
| 1.01 – 1.05 | Good | Safe to stop |
| 1.05 – 1.1 | Acceptable | Likely converged, but more sweeps improve quality |
| 1.1 – 1.2 | Marginal | Run more sweeps |
| > 1.2 | Not converged | Significantly more sweeps needed, or increase N_CHAINS |

### Troubleshooting high Rhat

1. **Run more sweeps**: Double N_SWEEPS
2. **Add more chains**: More chains give more mixing opportunities
3. **Check data quality**: Poorly formatted data can cause chains to get stuck
4. **Check max_clusters**: If too low, chains may be constrained
5. **Try different seeds**: Sometimes one chain gets stuck in a local mode

## Effective Sample Size (ESS)

Estimates the number of independent samples, accounting for autocorrelation.

```python
from crosscat.diagnostics import effective_sample_size

# traces: shape (n_chains, n_samples)
ess = float(effective_sample_size(traces))
```

### Interpretation

| ESS | Status | Action |
|-----|--------|--------|
| > 400 | Excellent | More than enough samples |
| 100 – 400 | Good | Adequate for most queries |
| 50 – 100 | Marginal | Acceptable for exploratory analysis |
| < 50 | Insufficient | Run more sweeps or thin the chain |

### Rule of thumb

For reliable posterior estimates, aim for ESS > 100 per chain. With 4 chains, total ESS should be > 400.

## Log-Joint Trajectory

The log-joint should increase and plateau:

```
Sweep 50:  log_joint = -5000
Sweep 100: log_joint = -4200   (improving)
Sweep 150: log_joint = -4100   (slowing down)
Sweep 200: log_joint = -4080   (plateaued — converged)
```

If log-joint is still rapidly improving at the end, run more sweeps.

If log-joint oscillates wildly, check data quality or reduce max_clusters.

## When to stop

Conservative rule: stop when ALL of:
1. Rhat < 1.1
2. ESS > 100
3. Log-joint has plateaued (relative improvement < 0.1% over last 50 sweeps)

Aggressive rule (for quick exploration): stop when Rhat < 1.2 and log-joint has stabilized.
