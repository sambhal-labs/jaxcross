# GPU Memory Guide

## Estimating memory

```python
from crosscat import estimate_packed_memory

mem_bytes = estimate_packed_memory(packed)
print(f"Per-chain memory: {mem_bytes / 1e6:.1f} MB")
print(f"4 chains: {4 * mem_bytes / 1e6:.1f} MB")
```

## Memory formula

Approximate memory per chain:
```
memory ≈ max_views × max_clusters × n_cols × 4 bytes (float32)
       + max_views × n_rows × 4 bytes (row assignments)
       + n_cols × max_clusters × suffstat_size × 4 bytes
```

## Tuning parameters

### max_clusters
- Default: `suggest_max_clusters(n_rows)` ≈ sqrt(n_rows) + constant
- Reduce to save memory: most datasets need 5-30 clusters
- Increase for very heterogeneous data

### max_views
- Default: 16
- Reduce to 8 for memory-constrained setups
- Most datasets use 2-8 views

### max_categories (for categorical columns)
- Default: auto-detected from data
- Only matters for categorical columns
- Set explicitly if you'll see new categories at inference time

## GPU-specific limits

| GPU | VRAM | Practical limit |
|-----|------|----------------|
| GTX 1650 | 4 GB | ~500 rows × 100 cols × 4 chains |
| RTX 3060 | 12 GB | ~2000 rows × 200 cols × 4 chains |
| T4 | 16 GB | ~5000 rows × 300 cols × 4 chains |
| A100 | 40/80 GB | ~50K rows × 500 cols × 8 chains |

These are rough estimates. Use `estimate_packed_memory()` for exact numbers.

## Handling OOM

If you get `jax.errors.OutOfMemoryError`:

1. **Reduce chains**: `N_CHAINS = 2` instead of 4
2. **Reduce max_clusters**: `max_clusters = 16` instead of auto
3. **Reduce max_views**: `max_views = 8` instead of 16
4. **Use subsample annealing**: Start with fewer rows
5. **Set JAX memory fraction**: `os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"` before importing JAX
6. **Use CPU fallback**: Remove GPU extra, JAX falls back to CPU (slower but unlimited memory)
