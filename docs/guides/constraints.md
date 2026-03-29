# Constraint Enforcement

## What

Incorporate domain knowledge by forcing specific columns to be dependent (same view) or independent (different views), and forcing rows to cluster together or apart.

## When to Use

- You know certain features must be related or independent
- Regulatory or business requirements on model structure
- Guided exploration with partial domain knowledge

## Column Dependency Constraints

### Force columns to be dependent (same view)

```python
from crosscat.constraints import ensure_col_dep_constraints

key, subkey = jax.random.split(key)
constrained = ensure_col_dep_constraints(
    subkey, state, data,
    constraints=[(0, 1, True)],  # salary and experience must be in same view
    max_rejections=100,
    n_sweeps_per_attempt=5,
)
```

### Force columns to be independent (different views)

```python
constrained = ensure_col_dep_constraints(
    subkey, state, data,
    constraints=[(0, 3, False)],  # salary and zip_code must be in different views
)
```

### Multiple constraints

```python
constrained = ensure_col_dep_constraints(
    subkey, state, data,
    constraints=[
        (0, 1, True),   # salary ~ experience
        (2, 3, False),  # department != zip_code
    ],
)
```

## Row Dependency Constraints

Force two rows into the same or different cluster:

```python
from crosscat.constraints import ensure_row_dep_constraint

# Force rows 5 and 10 to cluster together in view 0
constrained = ensure_row_dep_constraint(
    key, state, data,
    row_a=5, row_b=10, dependent=True,
    view_idx=0,
    max_iterations=100,
)
```

## Checking Constraints

```python
from crosscat.constraints import check_column_dep_constraint

satisfied = check_column_dep_constraint(state, col_a=0, col_b=1, dependent=True)
print(f"Constraint satisfied: {satisfied}")
```

## How It Works

Constraint enforcement uses **rejection sampling**: it runs Gibbs sweeps and checks whether all constraints are satisfied. If not, it tries again with a new random seed. Returns `None` if `max_rejections` is exceeded.

### Diagnostics

Pass `return_diagnostics=True` to get detailed information about the enforcement attempt:

```python
result, diags = ensure_col_dep_constraints(
    key, state, data,
    constraints=[(0, 1, True), (2, 3, False)],
    return_diagnostics=True,
)

if not diags['success']:
    print(f"Failed after {diags['n_attempts']} attempts")
    print(f"Best: {diags['best_n_satisfied']}/{len(constraints)} satisfied")
    print(f"Failed constraints: {diags['constraint_failures']}")
```

!!! tip
    If constraints frequently fail, try increasing `n_sweeps_per_attempt` or `max_rejections`. Very tight constraints may be incompatible with the data.

## API Reference

- [`ensure_col_dep_constraints`](../api/constraints.md#ensure_col_dep_constraints)
- [`ensure_row_dep_constraint`](../api/constraints.md#ensure_row_dep_constraint)
- [`check_column_dep_constraint`](../api/constraints.md#check_column_dep_constraint)
