# C-MAPSS: Predictive Maintenance / RUL Benchmark

End-to-end jaxcross pipeline for Remaining Useful Life (RUL) prediction on the
NASA C-MAPSS turbofan engine degradation dataset. The same code runs on a
single consumer GPU (e.g. GTX 1650) and on multi-GPU setups (e.g. Kaggle 2×T4);
execution mode is auto-selected from `jax.device_count()`.

## Why this dataset

C-MAPSS is the canonical RUL benchmark — every recent LSTM / CNN / Transformer
prognostics paper reports on it. jaxcross's angle is different:

- **Mixed data types**: 21 sensor readings (continuous), operating regime
  (categorical in FD002/FD004), cycle number (ordinal), RUL target (continuous)
- **Calibrated uncertainty**: the headline metric is not just RUL MAE but
  **posterior CI coverage** at 90/95/99% — something point-prediction LSTMs and
  Transformers cannot produce natively
- **No hand-designed features**: RUL is imputed directly from the raw joint
  posterior over sensor-reading clusters

## Prerequisites

- [jaxcross](../../) installed with the `benchmark` extra (brings `polars` + `scikit-learn`):
  ```bash
  uv sync --extra dev --extra gpu --extra benchmark
  ```
  `--extra gpu` is optional (CPU also works, just slow).
- No API key; the fetch script downloads the dataset from a public S3 mirror.

## Pipeline

Run from the repo root in this order:

| Step | Script | What it does | Runtime (GTX 1650) |
|------|--------|-------------|--------------------|
| 1 | `fetch_cmapss.py`       | Download NASA C-MAPSS zip, extract 12 txt files       | ~20 s |
| 2 | `preprocess_cmapss.py`  | Compute RUL, drop constant sensors, z-score, split    | ~30 s (all 4 sets) |
| 3 | `run_inference.py`      | Multi-chain packed Gibbs; auto single- or multi-GPU    | ~15–45 min (FD001) |
| 4 | `evaluate_rul.py`       | Insert test engines, impute RUL, measure CI coverage   | ~2–5 min |

```bash
# Full pipeline for FD001 (the gold-standard RUL benchmark)
uv run python examples/c_mapss/fetch_cmapss.py
uv run python examples/c_mapss/preprocess_cmapss.py FD001
uv run python examples/c_mapss/run_inference.py FD001 --sweeps 200 --chains 4
uv run python examples/c_mapss/evaluate_rul.py FD001 --samples 500
```

To run all four sub-datasets:

```bash
uv run python examples/c_mapss/preprocess_cmapss.py        # all 4
for fd in FD001 FD002 FD003 FD004; do
    uv run python examples/c_mapss/run_inference.py $fd
    uv run python examples/c_mapss/evaluate_rul.py $fd
done
```

## Single-GPU vs multi-GPU

`run_inference.py` detects the JAX device count automatically:

- **1 device** → vmapped multi-chain via `multi_chain_packed_gibbs_sweep` (one JIT compile handles all chains in parallel on the single device).
- **N > 1 devices** → `jax.pmap` distributes `N_CHAINS // N` chains per device (same pattern as `benchmarks/wdi_macroeconomic_benchmark.ipynb`).

No code changes needed; the same command runs on both:

```bash
uv run python examples/c_mapss/run_inference.py FD001        # GTX 1650 — auto-vmap
uv run python examples/c_mapss/run_inference.py FD001        # Kaggle 2×T4 — auto-pmap
```

If you pass `--chains` that does not divide the device count, it is rounded up.

### Running on Kaggle 2×T4 (full 20 631 rows)

A ready-to-run notebook: [kaggle_fd001.ipynb](kaggle_fd001.ipynb).

1. Upload / import the notebook on Kaggle.
2. Settings → Accelerator → **GPU T4 ×2** and enable Internet.
3. Run all. It clones the repo at this branch, installs with `--no-deps` (preserves Kaggle's pre-installed JAX+CUDA stack), then runs:
   - Fetch + preprocess FD001
   - `run_inference.py --chains 8 --sweeps 300` (no `--subsample` → full 20 631 rows)
   - `evaluate_rul.py --samples 1000` (BMA + 90/95/99 % CIs)
   - `baseline_rul.py` + `evaluate_best_chain.py`
   - Consolidated leaderboard printout

Expected inference runtime: ~30-90 min, well within the Kaggle free-tier 12 h weekly GPU budget.

## Column layout (per sub-dataset, after preprocessing)

```
idx  name             type         notes
0    time_in_cycles   ORDINAL      current flight cycle
1    op_setting_1     CONT/CAT     CAT in FD002/FD004 (6 regimes auto-detected)
2    op_setting_2     CONT/CAT
3    op_setting_3     CONT/CAT
4…   sensor_i         CONTINUOUS   z-scored; constant sensors dropped
last rul              CONTINUOUS   target; capped at 125 cycles (Heimes 2008)
```

FD001 typically keeps 14 of 21 sensors after dropping near-constants.
FD002/FD004 replace the three continuous op-setting columns with the
discretized regime id (categorical), which lets the model learn
regime-conditional sensor distributions.

## Evaluating the result

`evaluate_rul.py` reports:

- **MAE / RMSE / R² / bias** of the BMA posterior mean vs published test RULs
- **CI coverage** at 90/95/99% — how often the true RUL falls inside the CI
- **Comparison vs published baselines** on the same test split:

    | Dataset | LSTM (Zheng 2017) | CNN-LSTM (Li 2018) | Transformer (2024-25) |
    |---------|------------------:|-------------------:|----------------------:|
    | FD001   | 13.52             | 12.61              | 11.9                  |
    | FD002   | —                 | 19.61              | 17.2                  |
    | FD003   | 12.64             | —                  | 11.4                  |
    | FD004   | —                 | 23.57              | 19.8                  |

The goal is to land **competitive-or-better MAE** with strictly **better-calibrated
CIs** than what the point-prediction baselines can produce.

## Outputs

```
examples/c_mapss/results/
├── raw/                          train/test/RUL txt files per FD
├── preprocessed/<fd>/            train_data.npy, test_query.npy, test_rul_truth.npy, column_info.json
├── inference/<fd>/               chain_0..N-1.jxc, best_chain.jxc, log_joint_traces.npy, inference_meta.json
└── evaluation/<fd>/              rul_predictions.csv, metrics.json
```

All outputs are reproducible from the seed pinned in each script (`--seed 42`
for inference, `--seed 99` for evaluation by default).

## References

- NASA C-MAPSS data: <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>
- Heimes, F. (2008) "Recurrent neural networks for remaining useful life estimation"
- Zheng, S. et al. (2017) "Long short-term memory network for RUL estimation"
- Li, X. et al. (2018) "Remaining useful life estimation using a deep CNN-LSTM"
- Nature *Scientific Reports* (2025) "A deep learning-based prognostic approach for
  predicting turbofan engine degradation and remaining useful life"
