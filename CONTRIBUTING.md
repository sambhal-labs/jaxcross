# Contributing to jax-crosscat

Thanks for your interest in contributing! This guide covers everything you need to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/sambhal-labs/jaxcross.git
cd jaxcross

# Install with dev dependencies (requires uv)
uv sync --extra dev

# With GPU support
uv sync --extra dev --extra gpu
```

## Code Style

- **Python 3.11+** with type hints throughout
- **Formatter/linter**: [ruff](https://docs.astral.sh/ruff/) (rules: E, F, I, W, UP, B, SIM), line length 99
- Private functions prefixed with `_`

```bash
# Lint
uv run ruff check .

# Auto-fix lint issues
uv run ruff check . --fix

# Format
uv run ruff format .

# Type check
uv run mypy crosscat/ --ignore-missing-imports
```

## Pre-commit Hooks

We use pre-commit to enforce style automatically:

```bash
uv tool install pre-commit
pre-commit install
```

This runs ruff check + format on every commit.

## Running Tests

```bash
# Fast tests (~10 min)
uv run pytest -m "not slow"

# Full suite including recovery tests (~30 min)
uv run pytest

# Single test file
uv run pytest tests/test_packed_state.py -v

# Single test function
uv run pytest tests/test_new_features.py::test_function_name
```

## Project Structure

| Directory | Contents |
|-----------|----------|
| `crosscat/` | Core package — types, components, model, inference, Gibbs kernels |
| `crosscat/packed/` | JIT-compatible packed state and vectorized kernels |
| `tests/` | pytest test suite (158 tests: 127 fast + 31 slow) |
| `benchmarks/` | Performance benchmarks (synthetic, MNIST, JIT) |
| `dashboard/` | Interactive Streamlit dashboard |
| `notebooks/` | Jupyter notebooks (GPU benchmarks, examples) |
| `docs/` | Documentation (API reference, architecture, quickstart) |

See [CLAUDE.md](CLAUDE.md) for a detailed module-by-module architecture guide.

## Making Changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature
   ```

2. **Make your changes** — keep PRs focused on a single concern.

3. **Run tests** before committing:
   ```bash
   uv run ruff check . && uv run ruff format --check . && uv run pytest -m "not slow"
   ```

4. **Open a PR** against `main` with a clear description of what and why.

## Key Patterns to Follow

- **JAX idioms**: Use `jax.lax.scan` for loops, `jax.vmap` for parallelism, `jax.jit` for compilation. All state is immutable.
- **Deterministic RNG**: Always thread `jax.random.key()` / `jax.random.split()` — never use global state.
- **NaN transparency**: Missing data is `NaN`. Sufficient statistic computations must filter NaN values.
- **Conjugate models**: All component models are collapsed — parameters are integrated out analytically. Only cluster assignments and hyperparameters are sampled.

## Adding a New Component Model

1. Add the class to `crosscat/components.py` with `sufficient_statistics()`, `log_marginal_likelihood()`, and `posterior_predictive_logp()` methods
2. Add the enum value to `ColumnType` in `crosscat/types.py`
3. Add packed scoring to `crosscat/packed/components.py` (unified `jnp.where` dispatch)
4. Add sufficient statistics to `crosscat/packed/suffstats.py`
5. Add hyperparameter grid to `crosscat/gibbs.py`
6. Add tests covering initialization, scoring, and inference

## Reporting Issues

Please file issues at [github.com/sambhal-labs/jaxcross/issues](https://github.com/sambhal-labs/jaxcross/issues) with:
- What you expected vs what happened
- Minimal reproduction code
- JAX version and hardware (CPU/GPU/TPU)
