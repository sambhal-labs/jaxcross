# Contributing

Thank you for your interest in contributing to jax-crosscat!

## Development Setup

```bash
git clone https://github.com/sambhal-labs/jaxcross.git
cd jaxcross
uv sync --extra dev
```

## Code Style

- Python 3.11+
- [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 99 characters
- Rules: E, F, I, W, UP, B, SIM

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy crosscat/ --ignore-missing-imports
```

## Git Workflow

1. **Always use feature branches** — never commit directly to main
2. Branch naming: `feat/`, `fix/`, `perf/`, `chore/` prefixes
3. Create PRs via `gh pr create` and merge via `gh pr merge --merge`

```bash
git checkout -b feat/my-feature
# ... make changes ...
git add -A && git commit -m "feat: description"
gh pr create --title "feat: description" --body "..."
```

## Testing

Tests require JAX JIT compilation which is slow. **Do not run the full test suite locally** — use Kaggle (P100) via `notebooks/run_tests.ipynb`.

For quick local validation:

```bash
# Fast tests only (~10 min on GPU)
uv run pytest -m "not slow"

# Single test file
uv run pytest tests/test_packed_state.py -v
```

### Test Markers

- `@pytest.mark.slow` — GPU-heavy tests (30+ Gibbs sweeps)
- `@pytest.mark.xfail` — Known flaky tests (stochastic recovery)

### Property Tests

`tests/test_property.py` uses [Hypothesis](https://hypothesis.readthedocs.io/) to verify mathematical invariants across random inputs.

## Adding a New Component Model

1. Add the model class to `crosscat/components.py` with all 4 methods
2. Add the column type to `ColumnType` enum in `crosscat/types.py`
3. Add fields to `ColumnHypers` and `SufficientStats`
4. Add a branch in `crosscat/packed/components.py` unified scoring functions
5. Update `crosscat/packed/suffstats.py` for vectorized computation
6. Add hyperparameter grid in `crosscat/packed/kernels.py`
7. Add tests in `tests/`

## CI

GitHub Actions runs lint + format + type check only (~1 min). No pytest in CI (GPU required). Note that free-tier GitHub Actions quota is limited.

## Documentation

```bash
uv sync --extra docs
mkdocs serve  # local preview at localhost:8000
```
