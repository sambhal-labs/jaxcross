# Diagnostic Thresholds Reference

## Convergence metrics

| Metric | Excellent | Good | Marginal | Action needed |
|--------|-----------|------|----------|---------------|
| Gelman-Rubin Rhat | < 1.01 | < 1.05 | < 1.1 | > 1.2: more sweeps |
| Effective Sample Size | > 400 | > 100 | > 50 | < 50: more sweeps |
| Log-joint relative change | < 0.01% | < 0.1% | < 1% | > 1%: still improving |

## Model quality metrics

| Metric | Good | Acceptable | Poor |
|--------|------|-----------|------|
| Continuous MAE | < 0.5*std | < 1.0*std | > 1.0*std |
| Categorical accuracy | > 80% | > 60% | < 60% |
| Binary accuracy | > 90% | > 75% | < 75% |
| CI coverage (95%) | 92-98% | 85-100% | < 85% |

## Drift thresholds

| Metric | Normal | Warning | Alert |
|--------|--------|---------|-------|
| Anomaly mean z-score | < 2 | 2-5 | > 5 |
| Log-joint degradation | < 5% | 5-15% | > 15% |
| Category frequency shift | < 10% | 10-30% | > 30% |
| Mean shift (continuous) | < 1 std | 1-3 std | > 3 std |

## Operational thresholds

| Metric | Normal | Investigate | Action |
|--------|--------|------------|--------|
| Memory growth | Stable | > 20% increase | Reduce max_clusters or retrain |
| Query latency | < 100ms | 100ms-1s | > 1s: check GPU, recompile |
| New data volume | < 20% of training | 20-50% | > 50%: retrain |

## When to retrain

Retrain when ANY of:
1. Rhat > 1.2 (chains not converged)
2. ESS < 50 (insufficient samples)
3. Log-joint degraded > 10%
4. Anomaly score drift > 5 sigma
5. New data > 50% of training data
6. Imputation quality dropped > 50%
