---
name: gen-test
description: Generate pytest tests for crosscat modules following existing test patterns
disable-model-invocation: true
---

Generate a pytest test for the specified module or function.

Follow these conventions from the existing test suite:
- Use `jax.random.key(seed)` for deterministic RNG
- Use `@pytest.mark.slow` for tests running >10s
- Use conftest.py fixtures where available
- Test with small dimensions (K=3, N=50) for fast execution
- Verify numerical correctness with `jnp.allclose(atol=1e-4)`
- Test NaN handling (missing data) where applicable
- Place in tests/ directory with `test_` prefix
- Import patterns: `import jax`, `import jax.numpy as jnp`, `from crosscat import ...`

Before writing tests, read the target module and at least one existing test file to match style.
