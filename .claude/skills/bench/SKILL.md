---
name: bench
description: Run quick GPU performance benchmark on packed kernels
disable-model-invocation: true
---

Run a quick benchmark of the packed Gibbs sweep:

1. Check GPU availability with `jax.devices()`
2. Initialize a small CrossCat state (N=200 rows, D=10 columns) using `crosscat.model.initialize`
3. Pack the state with `crosscat.packed_state.pack_state`
4. Run `packed_gibbs_sweep` once to trigger JIT compilation (warmup)
5. Time 10 subsequent sweeps and report mean/std duration
6. Report JAX device used and approximate rows/second

Use `uv run python -c "..."` or create a temporary script and run with `uv run python`.
