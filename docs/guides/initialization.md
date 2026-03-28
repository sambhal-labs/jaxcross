# Model Initialization

## What

Create an initial CrossCat state from data and column types. The state contains column-to-view assignments, row-to-cluster assignments, hyperparameters, and sufficient statistics.

## When to Use

After loading data and specifying column types, initialization is the first modeling step.

## Single Chain

```python
import jax
from crosscat import initialize
from crosscat.types import ColumnType

key = jax.random.key(42)
state = initialize(key, data, col_types)

print(f"Views: {state.n_views}")
print(f"Column assignments: {state.column_assignments}")
for v in range(state.n_views):
    n_clusters = len(set(int(x) for x in state.views[v].row_assignments))
    print(f"  View {v}: {n_clusters} clusters")
```

## Multi-Chain (Recommended)

Different random initializations can lead to different local optima. Running multiple chains and selecting the best gives more robust results:

```python
states = initialize(key, data, col_types, n_chains=4)
# Returns a list of 4 independent CrossCatState objects
```

## Initialization Modes

| Mode | Behavior | When to Use |
|------|----------|-------------|
| `"from_the_prior"` | Sample column and row assignments from CRP priors | Default — good general-purpose start |
| `"together"` | All columns in one view, rows clustered from CRP | Conservative — when you expect most columns are related |
| `"apart"` | Each column in its own view | Exploratory — when you expect many independent groups |

```python
# Conservative start
state = initialize(key, data, col_types, initialization="together")

# Exploratory start
state = initialize(key, data, col_types, initialization="apart")
```

## CRP Concentration Parameters

The `column_crp_alpha` and `row_crp_alpha` parameters control how many groups the CRP prior expects:

- **Higher alpha** → more groups expected
- **Lower alpha** → fewer groups expected
- Default is 1.0, which works well for most datasets

```python
# Encourage more views and more clusters
state = initialize(key, data, col_types,
                   column_crp_alpha=2.0,
                   row_crp_alpha=2.0)
```

!!! tip
    These are prior parameters, not hard constraints. The Gibbs sampler will adjust the actual number of views and clusters during inference.

## Column Type-Specific Hyperparameter Defaults

Hyperparameters are initialized automatically from the data:

| Type | Hyperparameters | Default Initialization |
|------|----------------|----------------------|
| CONTINUOUS | `mu`, `r`, `s`, `nu` | `mu=mean(data)`, `r=1`, `s=var(data)`, `nu=1` |
| CATEGORICAL | `dirichlet_alpha` | `dirichlet_alpha=1.0` |
| BINARY | `alpha`, `beta` | `alpha=1.0`, `beta=1.0` |
| ORDINAL | `cutpoints`, `mu`, `s` | `cutpoints=linspace(-2, 2, K-1)`, `mu=0`, `s=4` |
| CYCLIC | `kappa`, `vm_a`, `vm_mu` | Data-driven MLE estimates |

!!! info "Ordinal columns are non-conjugate"
    The OrderedLogistic model uses grid integration (31-point grid over latent location), which is slower than conjugate models. This is expected and correct.

## Tips

- **Always use multi-chain** for any serious analysis (4+ chains)
- The initial state is random — run at least 50 sweeps before querying
- Use `log_joint(state, data)` to compare chains after inference

## API Reference

- [`initialize`](../api/model.md#initialize)
