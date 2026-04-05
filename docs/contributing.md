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
- `@pytest.mark.cpu` — Tests that run on CPU only (no GPU required)
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

GitHub Actions runs lint + format + type check + `@pytest.mark.cpu` tests (~2 min). GPU tests are not run in CI — use Kaggle (P100) for the full suite. Note that free-tier GitHub Actions quota is limited.

## Project Modules

| Module | Purpose |
|--------|---------|
| `crosscat/types.py` | Core dataclasses (`CrossCatState`, `ViewState`, `ColumnType`, `InitResult`) |
| `crosscat/components.py` | 5 Bayesian component models |
| `crosscat/model.py` | Initialization, scoring, row insertion |
| `crosscat/gibbs.py` | Collapsed Gibbs kernels (unpacked path) |
| `crosscat/inference.py` | 15 posterior predictive queries (unpacked) |
| `crosscat/packed/` | JIT-compiled packed state sub-package |
| `crosscat/packed_inference.py` | 15 packed queries + multi-chain wrappers |
| `crosscat/scaling.py` | Large dataset workflows (subsample, minibatch, early stopping) |
| `crosscat/tb_logger.py` | TensorBoard logging |
| `crosscat/constraints.py` | Column/row dependency enforcement |
| `crosscat/diagnostics.py` | Convergence metrics |
| `crosscat/serialization.py` | Save/load in `.jxc` format |
| `crosscat/data_utils.py` | CSV/Parquet/Arrow/NPY I/O, type detection |
| `crosscat/validate.py` | State consistency checking |

## Documentation

```bash
uv sync --extra docs
uv run mkdocs serve  # local preview at localhost:8000
```
