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

Run from the repo root. **Always run the smoke test before committing to a full run** — it validates the pipeline end-to-end in ~3-5 min.

| Step | Script / command | What it does |
|------|-----------------|--------------|
| 1 | `fetch_cmapss.py` | Download + extract NASA C-MAPSS zip (~20 s) |
| 2 | `preprocess_cmapss.py FD001` | Compute RUL, drop constant sensors, z-score (~30 s) |
| 3a | `run_inference.py FD001 --smoke` | 2 chains × 6 sweeps × 1000 rows end-to-end validation (~3-5 min on any GPU) |
| 3b | `run_inference.py FD001 ...` | Real multi-chain packed Gibbs; auto single- or multi-GPU |
| 4 | `evaluate_rul.py FD001` | Insert test engines, impute RUL, 90/95/99 % CI coverage (~2-5 min) |
| 5 | `baseline_rul.py FD001` | Ridge + RandomForest baselines on the same data (~1 min) |
| 6 | `evaluate_best_chain.py FD001` | Best-chain-only version of the eval for comparison (~1 min) |

### Local full FD001 run (4 GB GTX 1650)

Subsamples to 5000 rows to fit in VRAM; 4 chains × 100 sweeps; ~45 min.

```bash
uv run python examples/c_mapss/fetch_cmapss.py
uv run python examples/c_mapss/preprocess_cmapss.py FD001
uv run python examples/c_mapss/run_inference.py FD001 --smoke              # validate first
uv run python examples/c_mapss/run_inference.py FD001 \
    --chains 4 --sweeps 100 --diag-every 20 --subsample 5000
uv run python examples/c_mapss/evaluate_rul.py FD001 --samples 500
uv run python examples/c_mapss/baseline_rul.py FD001
uv run python examples/c_mapss/evaluate_best_chain.py FD001
```

### Checkpointing and resume

`run_inference.py` checkpoints every `--diag-every` sweeps to `results/inference/<fd>/`.
Safe to kill and re-run with `--resume` (same args as the killed run) — picks up from
`last_completed_sweep` without repeating work.

```bash
# Survives notebook / session death:
uv run python examples/c_mapss/run_inference.py FD001 \
    --chains 4 --sweeps 150 --diag-every 30 --resume
```

Resume rules:
- Checkpoint must match `n_chains` and `data_shape` exactly — otherwise it is ignored and a fresh run starts.
- If `--sweeps` is higher than the saved `last_completed_sweep`, resume runs the remaining sweeps.
- If it is lower or equal, resume is a no-op (nothing to do).
- To explicitly start over, delete `results/inference/<fd>/` before re-running.

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
3. Run cells **sequentially** (not "Run All") — the smoke test (cell 4) must pass before cell 6.

The notebook uses **4 chains × 150 sweeps × full 20 631 rows** with checkpointing every 30 sweeps.

**Measured throughput on 2×T4:** ~40 s per chain-sweep post-compile. `fori_loop`
serializes chains within each device, so with 4 chains each device processes 2
chains serially. Projection: 4 chains × 150 sweeps on 2×T4 ≈ **~3.3 h total**.

If the Kaggle session dies mid-run, the next `run_inference.py` with `--resume`
picks up from the last checkpoint — worst-case loss is 30 sweeps (~40 min), not
the whole run.

**Don't use 8 chains × 300 sweeps on 2×T4.** Measured: ~13.75 h projected, which
busts the Kaggle 12 h weekly quota. Stick with the notebook's 4 × 150 default unless
you have paid Kaggle GPU hours to burn.

## Column layout (per sub-dataset, after preprocessing)

```
idx  name             type         notes
0    time_in_cycles   CONTINUOUS   z-scored on training-set stats
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
