# Ordered Logistic: Grid Integration

**Source**: [`crosscat/packed/components.py`](https://github.com/sambhal-labs/jaxcross/blob/main/crosscat/packed/components.py) — `_ol_log_marginal`, `_ol_posterior_predictive_logp`.

The `OrderedLogistic` component is the **only non-conjugate** model in jaxcross. It models `K`-level ordinal data via a *cumulative-link* cutpoint model, with a latent location parameter that has no closed-form marginalization. The library handles this by integrating the location out on a fixed grid.

## Math

For an ordinal column with `K` levels and cutpoints `c_0 < c_1 < ... < c_{K-2}`:

```
P(y = k | location λ)  =  σ(c_k − λ) − σ(c_{k-1} − λ)
```

with `c_{-1} = -∞`, `c_{K-1} = +∞`, and `σ(·)` the logistic sigmoid. The location `λ` has a prior `p(λ)` (typically `Normal(0, τ)`). Exact inference requires marginalizing over `λ`:

```
P(y = k | data)  =  ∫ P(y = k | λ) · p(λ | data) dλ
```

No closed form. The library discretizes `λ` onto a grid of `ORDINAL_N_GRID = 31` points spanning `±LOGISTIC_INF / L` for some scale `L`, and computes the integral via weighted sum.

## Algorithm

```
1. Build location grid: λ_grid = linspace(-range, +range, ORDINAL_N_GRID)
2. Compute log prior on grid: log_prior[i] = log p(λ_grid[i])
3. For each observation y:
      log_lik[i] = log P(y | λ_grid[i], cutpoints)  (vectorized over i)
4. Posterior on grid: log_post = log_prior + Σ log_lik  (summed over observations)
5. Log marginal likelihood: logsumexp(log_post) + log(grid_spacing)
6. Posterior predictive: log P(y* = k | data) = logsumexp_i(log_P(y*=k | λ_grid[i]) + log_post[i]) - logsumexp(log_post)
```

Cutpoints are stored padded — a column with `K` levels uses the first `K-1` entries of a `max_cutpoints`-long buffer, with `+LOGISTIC_INF` in the unused slots (see "Pitfalls" below).

## Key Observations

### Grid saturation cap

`LOGISTIC_INF = 1e10` (from [types.py](../../api/types.md)) serves two roles:

1. **Padding for unused cutpoints** — guarantees the sigmoid saturates at 0 or 1 for padded slots, so they contribute no probability mass to any observed level.
2. **Grid endpoint cap** — the location grid is clamped to `±LOGISTIC_INF` so that `σ(·)` evaluates to a finite `0` or `1` even at the edges. This prevents `inf − inf` NaNs when JAX evaluates both branches of a `jnp.where`.

### Grid coarseness

With 31 grid points, the integral is *accurate but not exact*. For most ordinal columns (5–10 levels), this is well within Monte Carlo noise of the row-assignment step; for high-granularity ordinals (20+ levels), consider extending `ORDINAL_N_GRID` at import time. The trade-off is linear cost.

### `hyper_n_cutpoints`

`PackedCrossCatState.hyper_n_cutpoints` stores the *real* cutpoint count per ordinal column. The hyper-transition kernel uses this to mask updates — you cannot use `isfinite(cutpoints)` as a mask because the grid endpoints are finite by construction. This is a subtle but critical invariant.

## Hyperparameter Guidance

- **Cutpoint grid updates.** The hyper-transition kernel resamples each cutpoint on a data-driven grid spanning the observed range of the column. Rarely needs manual tuning.
- **Number of levels `K`.** Set by the data; jaxcross detects the maximum integer value in the column and uses `K - 1` cutpoints.
- **Extending `ORDINAL_N_GRID`.** Override before `import crosscat`:
    ```python
    import os
    os.environ["JAXCROSS_ORDINAL_N_GRID"] = "63"  # (if supported in your version)
    # OR edit crosscat/types.py directly
    import crosscat
    ```
  Note: captured at import time. Re-importing is not sufficient — restart Python.

## Pitfalls

### JAX evaluates both branches

The most infamous jaxcross pitfall: `jnp.where(mask, finite, padded)` evaluates *both* arguments before selecting. Padded cutpoints of `+LOGISTIC_INF` flow through the "unused" branch's `linspace` / `sigmoid` / `vmap` — if any of those overflow to `±inf` or NaN, the result corrupts *even the masked-out branch*.

**Rule**: clamp inputs to a finite range *before* `linspace`, `vmap`, or `logsumexp`. The OL kernels follow this religiously — new code touching this path must too.

### Roundtrip via `hyper_n_cutpoints`

Never reconstruct cutpoint count from the cutpoints array itself (e.g. counting `isfinite` entries). The grid clamp makes the endpoints look "finite" even when they're semantically padding. Always use `hyper_n_cutpoints[col_idx]`.

### Padded slot corruption

When adding a cutpoint (e.g. a column's `K` grows via online learning), the padded slots must stay at `+LOGISTIC_INF`. Kernel writes must mask via `col_idx < hyper_n_cutpoints[c]` — otherwise padded slots drift into the valid range and the likelihood silently becomes wrong.

## Related

- [Hyperparameter Transitions](hyper-transitions.md) — how cutpoints themselves are updated.
- [Types → `LOGISTIC_INF`](../../api/types.md#constants) — the saturation cap constant.
- [Components API → OrderedLogistic](../../api/components.md)
