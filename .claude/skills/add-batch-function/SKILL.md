---
name: add-batch-function
description: Add a new batch inference function following the established vmap pattern
disable-model-invocation: true
---

Create a new batch inference function in `crosscat/packed_inference.py`. Usage: `/add-batch-function <function_name>`.

1. Read the existing single-row function to understand its signature and logic.

2. Read an existing batch function (e.g., `batch_anomaly_score` or `batch_classify_column`) to follow the established pattern:
   - Function name starts with `batch_`
   - Takes `row_ids: Array` (1D integer array) instead of single `row_id: int`
   - Precomputes view/column metadata outside the vmap
   - Inner function operates on one row, outer vmap parallelizes
   - Uses `_cluster_weights_for_row` for row-specific cluster weights
   - All indexing uses JAX arrays (not Python ints) for vmap compatibility
   - Scalar constants must be `jnp.float32(...)` not Python floats

3. Write the batch function following these rules:
   - Docstring: state the use case, args, returns, and that it's vmapped
   - Use `jax.vmap(_inner)(row_ids)` for the outer row loop
   - Use `jax.vmap` for any inner loops over clusters or columns
   - Never use Python `int()` or `float()` inside vmapped code
   - Test that `jnp.isnan` masking works (no Python `if` inside vmap)

4. Export the function:
   - Add import to `crosscat/__init__.py` (both import line and `__all__` list)
   - Keep alphabetical order in both places

5. Write a quick parity test comparing batch output to individual calls.

6. Run `uv run ruff check crosscat/packed_inference.py crosscat/__init__.py`.
