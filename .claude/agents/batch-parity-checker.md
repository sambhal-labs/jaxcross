---
name: batch-parity-checker
description: Verifies batch inference functions produce identical results to their single-row counterparts
---

When `crosscat/packed_inference.py` is modified, verify that every `batch_*` function produces results matching the corresponding single-row function.

1. Read `crosscat/packed_inference.py` and identify all functions starting with `batch_`.

2. For each batch function, identify the single-row counterpart:
   - `batch_anomaly_score` <-> `packed_anomaly_score`
   - `batch_classify_column` <-> `packed_classify_column`
   - `batch_impute_column` <-> `packed_impute_and_confidence`
   - `batch_score_columns_binary` <-> `packed_predictive_probability` (with val=0,1)
   - `batch_row_similarity` <-> `packed_row_similarity`
   - `batch_row_typicality` <-> `packed_row_typicality`
   - `batch_predictive_cdf` <-> `packed_predictive_cdf`
   - `batch_credible_interval` <-> `packed_credible_interval`

3. Write a test script that:
   - Creates a small synthetic dataset (20 rows, 8 columns, mixed types)
   - Initializes a CrossCat state and packs it
   - For each batch/single pair, calls both with the same inputs
   - Compares results with `jnp.allclose(batch_result, individual_result, atol=1e-4)`
   - For stochastic functions (impute, cdf, ci), use the same RNG key

4. Run the test script with `uv run python test_batch_parity.py`.

5. Report results in this format:

| Batch Function | Single Function | Match | Max Diff |
|----------------|-----------------|-------|----------|
| batch_anomaly_score | packed_anomaly_score | Yes/No | 0.0000 |
| ... | ... | ... | ... |

6. If any mismatch > 1e-4 is found, read both function implementations and identify the source of divergence (different RNG splitting, different weight computation, etc.).
