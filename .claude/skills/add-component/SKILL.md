---
name: add-component
description: Scaffold a new Bayesian component model following the 9-step workflow
disable-model-invocation: true
---

Add a new conjugate Bayesian component model to JAX-CrossCat. Usage: `/add-component <ModelName> <ColumnType>` (e.g., `/add-component Poisson COUNT`).

Before starting, read `docs/architecture/model.md` and `docs/api/components.md` for context.

Follow these 9 steps in order:

## Step 1: Unpacked component (`crosscat/components.py`)

Read existing components (e.g., `NormalGamma`, `BetaBernoulli`) to match the pattern:
- Create a class with `sufficient_statistics(data, mask)`, `log_marginal_likelihood(stats, hypers)`, `posterior_predictive_logp(x, stats, hypers)`, and `sample_posterior_predictive(key, stats, hypers)`
- All math in log-space, use `LOG_EPS` from `crosscat.types` for numerical stability
- NaN-safe: filter missing data via mask in sufficient statistics

## Step 2: Packed components (`crosscat/packed/components.py`)

- Add `_XX_log_marginal(stats, hypers)` and `_XX_posterior_predictive_logp(x, stats, hypers)` and `_XX_sample(key, stats, hypers)`
- Update the 3 `unified_*` dispatch functions with a new `jnp.where` branch for the new `ColumnType` enum value

## Step 3: Packed state (`crosscat/packed/state.py`)

- Add any new hyper fields to `PackedCrossCatState` dataclass and `_ARRAY_FIELDS`
- Update `pack_state()` and `unpack_state()` to handle the new fields

## Step 4: Packed kernels (`crosscat/packed/kernels.py`)

- Thread new hypers through all scoring functions
- Add hyper transition in `packed_transition_column_hypers`
- Update `packed_insert_rows` constructor — ensure ALL `PackedCrossCatState` construction sites include the new field

## Step 5: Model initialization (`crosscat/model.py`)

- Add initialization logic and default hyperparameters in `_default_hypers`

## Step 6: Unpacked Gibbs (`crosscat/gibbs.py`)

- Add hyperparameter transition for the new component type

## Step 7: Packed inference (`crosscat/packed_inference.py`)

- Thread new hypers through any inference calls that need them

## Step 8: Property tests (`tests/test_property.py`)

- Add `test_suffstat_empty_is_zero_<type>` — empty stats are zero/identity
- Add `test_suffstat_add_remove_roundtrip_<type>` — add then remove returns to original
- Add `test_component_score_finite_<type>` — scores are finite for valid inputs
- Add `test_type_dispatch_parity_<type>` — packed unified dispatch matches unpacked

## Step 9: Serialization (`crosscat/serialization.py`)

- Bump `_SCHEMA_VERSION`
- Add migration in `load_packed_state` for the new field
- Update `_ARRAY_FIELDS` if new arrays were added

## Step 10: Types (`crosscat/types.py`)

- Add the new `ColumnType` enum value (e.g., `COUNT = 5`)

## Final checks

- Export new type in `crosscat/__init__.py`
- Run `uv run ruff check .` and `uv run ruff format .`
- Verify all `PackedCrossCatState(...)` construction sites include the new field (grep for `PackedCrossCatState(`)
