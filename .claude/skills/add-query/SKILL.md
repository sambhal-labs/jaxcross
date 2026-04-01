---
name: add-query
description: Add a new inference query function to both packed and unpacked paths
disable-model-invocation: true
---

Add a new inference query to JAX-CrossCat. Usage: `/add-query <function_name>` (e.g., `/add-query conditional_mode`).

Before starting, read `docs/guides/inference.md` and `docs/api/inference.md` for context.

Follow these 4 steps in order:

## Step 1: Unpacked implementation (`crosscat/inference.py`)

Read existing query functions to match the pattern:
- Function takes `state: CrossCatState`, `data: Array`, plus query-specific args
- Iterates over views/columns using Python for-loops
- Returns JAX arrays
- Handles missing data (NaN) gracefully
- Add docstring with args, returns, and "Maps to original..." reference if applicable

## Step 2: Packed implementation (`crosscat/packed_inference.py`)

Read existing packed queries to match the pattern:
- Function takes `packed: PackedCrossCatState`, `data: Array`, plus query-specific args
- Uses `jax.vmap` / `jnp.where` instead of Python loops
- Must be JIT-compatible (no Python control flow on data-dependent values)
- Use `jnp.float32(...)` for scalar constants, not Python floats
- NaN masking via `jnp.where(jnp.isnan(...), ...)` not Python `if`

If the query benefits from multi-chain averaging, also add a `multi_chain_<name>` wrapper that takes a list of packed states and averages results.

## Step 3: Export (`crosscat/__init__.py`)

- Add imports for both unpacked and packed versions
- Add both to `__all__` list
- Keep alphabetical order

## Step 4: Tests

Create or update tests:
- Add a unit test in the appropriate test file (or create `tests/test_<name>.py`)
- Add a **parity test** in `tests/test_packed_inference_parity.py` verifying packed output matches unpacked output within `atol=1e-4`
- Use small dimensions (K=3, N=50) for fast execution
- Use `jax.random.key(seed)` for deterministic RNG
- Test with NaN values (missing data) if applicable

## Final checks

- Run `uv run ruff check crosscat/inference.py crosscat/packed_inference.py crosscat/__init__.py`
- Run `uv run ruff format crosscat/inference.py crosscat/packed_inference.py crosscat/__init__.py`
