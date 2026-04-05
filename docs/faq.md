# FAQ & Troubleshooting

Common questions and solutions for jax-crosscat.

---

## Installation

### JAX doesn't detect my GPU

Verify your JAX installation sees the GPU:

```python
import jax
print(jax.devices())  # Should show [GpuDevice(...)]
```

If it shows only CPU:

1. Check CUDA is installed: `nvidia-smi` should show your GPU
2. Reinstall JAX with CUDA support: `pip install "jax[cuda13]"`
3. Verify CUDA version compatibility — check the [JAX installation docs](https://jax.readthedocs.io/en/latest/installation.html) for supported CUDA versions

### `ptxas` version mismatch on Kaggle / Colab

**Don't** use `uv sync --extra gpu` on Kaggle or Colab. These platforms have JAX + CUDA pre-installed. Instead:

```bash
pip install -e . --no-deps
```

This installs jax-crosscat without touching the platform's JAX/CUDA stack.

### `uv` vs `pip` — which should I use?

We recommend [uv](https://docs.astral.sh/uv/) for local development (faster, reproducible lockfile). Use `pip` on cloud platforms (Kaggle, Colab) where JAX is pre-installed.

```bash
# Local development
uv sync --extra dev

# Cloud platforms
pip install -e . --no-deps
```

### Import error: `No module named 'crosscat'`

Make sure you installed the package, not just cloned the repo:

```bash
# From source
uv sync --extra dev
# or
pip install -e .
```

---

## Performance

### Why is the first sweep so slow?

The first call to `packed_gibbs_sweep` triggers JAX's JIT (just-in-time) compilation. This is a one-time cost:

| Phase | Typical Time | Happens |
|-------|-------------|---------|
| JIT compilation | 20–60s (varies by dataset shape) | First sweep only |
| XLA cache write | ~5s | First run only |
| Subsequent sweeps | 4–12s each | Every sweep |
| Cached restart | ~2s | When XLA cache exists |

Times vary by dataset shape and hardware. See [benchmarks/](https://github.com/sambhal-labs/jaxcross/tree/main/benchmarks) for current numbers.

The **XLA persistent cache** is auto-enabled when you import `crosscat.packed`. On subsequent runs, JIT compilation is skipped entirely.

### How do I use the XLA compilation cache?

It's automatic! The cache activates on `import crosscat.packed`. To pre-compile all kernels for a specific data shape:

```python
from crosscat.packed import compile_kernels
compile_kernels(packed_state, data)  # Warm up all sub-kernels
```

To clear the cache: `from crosscat.packed import clear_cache; clear_cache()`.

See the [XLA Cache Guide](guides/xla-cache.md) for details.

### CPU vs GPU — expected timings

| Setup | 50x11 Dataset | 1000x257 Dataset |
|-------|---------------|-------------------|
| CPU (no JIT) | ~105s/sweep | Not practical |
| GPU (packed, P100) | ~4.5s/sweep | ~12s/sweep |

**Always use the packed path** (`packed_gibbs_sweep`) for real workloads. The unpacked path (`gibbs_sweep`) uses Python for-loops and is 10–100x slower.

### How can I reduce memory usage?

Tune the padding parameters in `pack_state()`:

```python
packed = pack_state(state,
    max_views=5,        # Default: 16
    max_clusters=20,    # Default: 32
    max_categories=10,  # Default: 16
)
```

Smaller padding = less GPU memory. Set these to the maximum you expect during inference, not the theoretical maximum.

---

## Modeling

### How many sweeps do I need?

It depends on dataset size and complexity:

| Dataset Size | Recommended Sweeps | Why |
|-------------|-------------------|-----|
| Small (<100 rows) | 50–100 | Converges quickly |
| Medium (100–1000 rows) | 100–200 | Standard range |
| Large (1000+ rows) | 200–500 | More structure to discover |

**Use multi-chain inference** to assess convergence:

```python
result = initialize(key, data, col_types, n_chains=4)
states = result.state
# Run each chain, then compare with diagnostics
```

Monitor `log_joint` — when it plateaus across chains, you've likely converged. See the [Diagnostics Guide](guides/diagnostics.md).

### How do I choose column types?

| Your Data | Use |
|-----------|-----|
| Real-valued numbers (salary, temperature) | `CONTINUOUS` |
| Unordered categories (department, country) | `CATEGORICAL` |
| True/false, 0/1 | `BINARY` |
| Ordered categories (star ratings 1-5, education level) | `ORDINAL` |
| Angles, periodic values (wind direction, time of day) | `CYCLIC` |

**Automatic detection** is available:

```python
from crosscat import guess_column_types
col_types = guess_column_types(data)
```

### What if I have too many categories?

Category values must be integers < `max_categories` (default: auto-detected from data). If a column has 100+ categories:

1. Consider whether it's truly categorical or should be `CONTINUOUS`
2. Group rare categories into an "other" bin
3. Set `max_categories` explicitly in `pack_state()`

### Can I use CrossCat for classification?

Yes! Use `predictive_probability` or `predictive_sample` with the target column as the query:

```python
# Predict column 0 given columns 1-10
prob = predictive_probability(state, data, query_cols=[0], query_vals=jnp.array([target_class]),
                              condition_cols=[1,2,3,4,5,6,7,8,9,10],
                              condition_vals=new_row[1:11])
```

CrossCat won't match a dedicated classifier on accuracy, but it provides uncertainty estimates and works with mixed column types natively. The [MNIST benchmark](examples/mnist.md) demonstrates 79% classification accuracy on handwritten digits.

### When should I use constraints?

Use column constraints when you have domain knowledge:

- **Must-link**: "These columns should always be in the same view" (e.g., latitude and longitude)
- **Cannot-link**: "These columns should never be in the same view" (e.g., redundant encodings)

```python
from crosscat import ensure_col_dep_constraints
state = ensure_col_dep_constraints(key, state, data,
    constraints=[(0, 1, True), (2, 3, False)])  # True=must-link, False=cannot-link
```

---

## Common Errors

### `ValueError: values >= max_categories`

Your data contains category values larger than `max_categories`. Fix by passing `data=` to `pack_state()`:

```python
packed = pack_state(state, data=data)  # Validates and raises clear error
```

Then either increase `max_categories` or remap your category values to start from 0.

### NaN in log-joint score

Common causes:

1. **Numerical underflow** — Check if your continuous data has extreme values. Standardizing can help.
2. **Empty clusters** — Usually self-correcting after a few sweeps.
3. **Debug mode**: Run with NaN detection enabled:

```python
jax.config.update("jax_debug_nans", True)
# Then run inference — JAX will raise on the first NaN
```

### `TracerConversionError` or "tracer leaked"

This JAX error means a traced value escaped a JIT-compiled function. Common cause: using Python `if` on a JAX array inside JIT. This shouldn't happen with the public API — if it does, please [file an issue](https://github.com/sambhal-labs/jaxcross/issues).

### `max_cols_per_view` overflow warning

During column reassignment, a view received more columns than `max_cols_per_view`. Fix:

```python
packed = pack_state(state, max_cols_per_view=n_cols)  # Default, safest
```

---

## Conceptual

### When should I use CrossCat vs. K-Means / GMM / HDBSCAN?

| Scenario | Best Choice |
|----------|-------------|
| All numeric columns, known k | K-Means |
| All numeric, unknown k, variable density | HDBSCAN |
| Mixed types (numeric + categorical + ordinal) | **CrossCat** |
| Need to know which columns are related | **CrossCat** |
| Need uncertainty estimates on predictions | **CrossCat** |
| Need missing value imputation | **CrossCat** |
| >100k rows, speed is critical | K-Means / HDBSCAN |
| Moderate data (<10k rows), rich queries needed | **CrossCat** |

CrossCat is uniquely suited when you have **mixed-type data** and want to **discover column relationships**, not just row clusters.

### What's the difference between packed and unpacked paths?

| | Unpacked | Packed |
|--|---------|--------|
| State | Python dataclasses with variable-size lists | Fixed-size JAX arrays (padded) |
| Speed | 10–100x slower (Python for-loops) | GPU-accelerated (JIT + vmap) |
| Use for | Debugging, understanding the algorithm | All real workloads |
| Module | `crosscat.gibbs` | `crosscat.packed.kernels` |

**Always use the packed path** unless you're debugging:

```python
# Packed (fast) — use this
from crosscat.packed import pack_state, packed_gibbs_sweep, unpack_state

# Unpacked (slow) — only for debugging
from crosscat import gibbs_sweep
```

---

## Scaling & Production

### How do I handle datasets with 10K+ rows?

Use the `crosscat.scaling` module, which provides four strategies:

1. **Subsample annealing** — start small, grow progressively (`subsample_anneal`)
2. **Mini-batch Gibbs** — update a random subset of rows per sweep (`minibatch_gibbs_sweep`)
3. **Parallel row scoring** — `vmap` over all rows simultaneously (`parallel_gibbs_sweep`)
4. **Early stopping** — stop when log-joint converges (`gibbs_sweep_early_stopping`)

See the [Scaling Guide](guides/scaling.md) for full details.

### How do I estimate GPU memory usage before packing?

```python
from crosscat import estimate_packed_memory

mem = estimate_packed_memory(100_000, 50, max_clusters=16)
print(f"Estimated: {mem['total'] / 1e6:.1f} MB")
```

### What's `InitResult` and why did `initialize()` change?

`initialize()` now returns an `InitResult` instead of a bare state. Access the state via `result.state`:

```python
result = initialize(key, data, col_types)
state = result.state  # CrossCatState (same as before)
```

This wrapper also carries `subsample_idx` when `subsample_rows` is set.

### How do I monitor inference progress?

Use the TensorBoard logger:

```python
from crosscat.tb_logger import TBLogger

with TBLogger("runs/my_experiment") as tb:
    for sweep in range(n_sweeps):
        packed = packed_gibbs_sweep(key, packed, data, n_sweeps=1)
        state = unpack_state(packed, col_types, data=data)
        tb.log_sweep(collect_diagnostics(state, data), sweep)
```

See the [TensorBoard Guide](guides/tb-logger.md).

### Can I load Parquet files directly?

Yes, if `pyarrow` is installed:

```python
from crosscat import read_parquet
data, col_names = read_parquet("data.parquet")
```

See [Data Loading Guide](guides/data-loading.md#parquet-files) for Parquet, Arrow IPC, and NPY formats.
