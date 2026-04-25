#!/usr/bin/env python3
"""Run multi-chain Gibbs inference on a C-MAPSS sub-dataset.

Auto-selects the execution mode from the JAX device count:
  - 1 device   -> vmap across chains via multi_chain_packed_gibbs_sweep
  - N devices  -> jax.pmap across devices, CHAINS_PER_DEVICE = N_CHAINS // N
                  (pattern from benchmarks/wdi_macroeconomic_benchmark.ipynb)

Checkpoints the full set of chains + trace + meta after every chunk of
``--diag-every`` sweeps, so a Kaggle session death, interrupt, or OOM only
costs you the sweeps since the last checkpoint (not the whole run).

Outputs (examples/nhanes_clinical/results/inference/<fd>/):
  chain_{i}.jxc               per-chain packed state (latest checkpoint)
  best_chain.jxc              argmax-log-joint chain (latest checkpoint)
  log_joint_traces.npy        (n_chains, n_diag_points) trace so far
  train_used.npy              exact training rows the chains saw
  inference_meta.json         config + timing + last_completed_sweep

Run a 5-minute smoke test first to validate the pipeline end-to-end:

    uv run python examples/nhanes_clinical/run_inference.py --smoke

Full run:

    uv run python examples/nhanes_clinical/run_inference.py \
        --chains 4 --sweeps 100 --diag-every 20

Resume after a crash/interrupt (same args as the killed run):

    uv run python examples/nhanes_clinical/run_inference.py \
        --chains 4 --sweeps 100 --diag-every 20 --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import crosscat.packed.state as _ps
from crosscat import initialize
from crosscat.packed import (
    batch_packed_states,
    pack_state,
    packed_gibbs_sweep,
    unbatch_packed_states,
)
from crosscat.packed.kernels import multi_chain_packed_gibbs_sweep, packed_log_joint
from crosscat.serialization import load_packed_state, save_packed_state
from crosscat.types import ColumnType

PREP_ROOT = Path("examples/nhanes_clinical/results/preprocessed")
OUT_ROOT = Path("examples/nhanes_clinical/results/inference")

_TYPE_MAP = {
    "CONTINUOUS": ColumnType.CONTINUOUS,
    "CATEGORICAL": ColumnType.CATEGORICAL,
    "ORDINAL": ColumnType.ORDINAL,
    "BINARY": ColumnType.BINARY,
    "CYCLIC": ColumnType.CYCLIC,
}


def _load_preprocessed():
    if not PREP_ROOT.exists():
        raise FileNotFoundError(f"Missing {PREP_ROOT} — run preprocess_nhanes.py first")
    train = np.load(PREP_ROOT / "train_data.npy")
    info = json.loads((PREP_ROOT / "column_info.json").read_text())
    column_types = [_TYPE_MAP[c["type"]] for c in info["columns"]]
    return train, column_types, info


def _pack_initial_chains(
    data: jnp.ndarray,
    column_types: list[ColumnType],
    n_chains: int,
    seed: int,
    max_views: int,
    max_clusters: int,
) -> list:
    key = jax.random.key(seed)
    result = initialize(key, data, column_types, n_chains=n_chains)
    states = result.state if n_chains > 1 else [result.state]
    return [pack_state(s, max_views=max_views, max_clusters=max_clusters) for s in states]


# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------


def _save_checkpoint(
    out_dir: Path,
    chains: list,
    traces: np.ndarray,
    meta: dict,
    column_types: list[ColumnType],
    data_np: np.ndarray,
) -> None:
    """Atomically persist every chain + trace + meta. Safe to overwrite."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ci, packed in enumerate(chains):
        save_packed_state(packed, str(out_dir / f"chain_{ci}.jxc"), column_types=column_types)

    last_scores = (
        traces[:, -1]
        if traces.size
        else np.array([float(meta.get(f"_score_{i}", 0.0)) for i in range(len(chains))])
    )
    best_idx = int(np.argmax(last_scores))
    save_packed_state(chains[best_idx], str(out_dir / "best_chain.jxc"), column_types=column_types)
    np.save(out_dir / "log_joint_traces.npy", traces)
    if not (out_dir / "train_used.npy").exists():
        np.save(out_dir / "train_used.npy", data_np)
    (out_dir / "inference_meta.json").write_text(json.dumps(meta, indent=2))


def _try_resume(
    out_dir: Path,
    expected_n_chains: int,
    expected_n_sweeps: int,
    expected_data_shape: tuple[int, int],
) -> tuple[list, np.ndarray, int] | None:
    """Return (chains, traces, sweeps_done) if a compatible checkpoint exists."""
    meta_path = out_dir / "inference_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("n_chains") != expected_n_chains:
        print(
            f"  Resume skipped: checkpoint has {meta.get('n_chains')} chains, "
            f"current run expects {expected_n_chains}"
        )
        return None
    if tuple(meta.get("data_shape", [])) != expected_data_shape:
        print(
            f"  Resume skipped: checkpoint data_shape {meta.get('data_shape')} "
            f"differs from current {list(expected_data_shape)}"
        )
        return None
    sweeps_done = int(meta.get("last_completed_sweep", 0))
    if sweeps_done <= 0:
        return None

    chains: list = []
    for ci in range(expected_n_chains):
        p = out_dir / f"chain_{ci}.jxc"
        if not p.exists():
            print(f"  Resume skipped: {p.name} missing")
            return None
        packed, _ = load_packed_state(str(p))
        chains.append(packed)

    traces_path = out_dir / "log_joint_traces.npy"
    traces = (
        np.load(traces_path)
        if traces_path.exists()
        else np.zeros((expected_n_chains, 0), dtype=np.float32)
    )
    if sweeps_done >= expected_n_sweeps:
        print(
            f"  Checkpoint already at {sweeps_done}/{expected_n_sweeps} sweeps — "
            f"loading final chains without further inference"
        )
    return chains, traces, sweeps_done


# ---------------------------------------------------------------------------
# Run loops — checkpoint after every chunk
# ---------------------------------------------------------------------------


def _run_single_device(
    current_list: list,
    traces_so_far: np.ndarray,
    data: jnp.ndarray,
    data_np: np.ndarray,
    sweeps_done: int,
    n_sweeps: int,
    diag_every: int,
    seed: int,
    out_dir: Path,
    meta_base: dict,
    column_types: list[ColumnType],
) -> tuple[list, np.ndarray]:
    """Vmap across chains on a single device, checkpoint after every chunk."""
    n_chains = len(current_list)
    traces_rows: list[list[float]] = [list(traces_so_far[i]) for i in range(n_chains)]
    start = time.time()
    run_base = meta_base["_run_start_clock"]

    for chunk_start in range(sweeps_done, n_sweeps, diag_every):
        chunk = min(diag_every, n_sweeps - chunk_start)
        chunk_key = jax.random.fold_in(jax.random.key(seed), chunk_start)

        batched, scores = multi_chain_packed_gibbs_sweep(
            chunk_key, current_list, data, n_sweeps=chunk
        )
        jax.block_until_ready(batched.view_row_assignments)
        current_list = unbatch_packed_states(batched, n_chains)

        done = chunk_start + chunk
        for ci, s in enumerate(np.asarray(scores).tolist()):
            traces_rows[ci].append(float(s))
        traces_now = np.array(traces_rows, dtype=np.float32)

        best = float(traces_now[:, -1].max())
        wall = time.time() - run_base
        print(
            f"  Sweep {done:4d}/{n_sweeps}  best log_joint={best:,.1f}  "
            f"({wall:.0f}s total, {time.time() - start:.0f}s this session, "
            f"{n_chains} chains vmapped)",
            flush=True,
        )

        meta = dict(meta_base)
        meta.update(
            last_completed_sweep=done,
            elapsed_seconds=round(wall, 1),
            final_log_joints=traces_now[:, -1].tolist(),
            best_chain_idx=int(np.argmax(traces_now[:, -1])),
        )
        _save_checkpoint(out_dir, current_list, traces_now, meta, column_types, data_np)
        gc.collect()

    return current_list, np.array(traces_rows, dtype=np.float32)


def _run_multi_device(
    current_list: list,
    traces_so_far: np.ndarray,
    data: jnp.ndarray,
    data_np: np.ndarray,
    sweeps_done: int,
    n_sweeps: int,
    diag_every: int,
    seed: int,
    out_dir: Path,
    meta_base: dict,
    column_types: list[ColumnType],
) -> tuple[list, np.ndarray]:
    """pmap across devices, fori_loop within each device, checkpoint after every chunk."""
    n_devices = jax.device_count()
    n_chains = len(current_list)
    assert n_chains % n_devices == 0, (
        f"n_chains ({n_chains}) must be divisible by n_devices ({n_devices})"
    )
    chains_per_device = n_chains // n_devices

    def sweep_chains_on_device(keys, packed_batch, data_in, n_sweeps):
        def body(i, carry):
            packed_b = carry
            single_kwargs = {}
            for name in _ps._ARRAY_FIELDS:
                single_kwargs[name] = getattr(packed_b, name)[i]
            for name in _ps._STATIC_FIELDS:
                single_kwargs[name] = getattr(packed_b, name)
            single = _ps.PackedCrossCatState(**single_kwargs)
            result = packed_gibbs_sweep(keys[i], single, data_in, n_sweeps=n_sweeps)
            new_kwargs = {}
            for name in _ps._ARRAY_FIELDS:
                arr = getattr(packed_b, name)
                new_kwargs[name] = arr.at[i].set(getattr(result, name))
            for name in _ps._STATIC_FIELDS:
                new_kwargs[name] = getattr(packed_b, name)
            return _ps.PackedCrossCatState(**new_kwargs)

        return jax.lax.fori_loop(0, keys.shape[0], body, packed_batch)

    pmap_sweep = jax.pmap(
        sweep_chains_on_device,
        in_axes=(0, 0, None, None),
        static_broadcasted_argnums=(3,),
    )

    def reshape_for_pmap(batched, chain_keys):
        keys_pmap = chain_keys.reshape(n_devices, chains_per_device, *chain_keys.shape[1:])
        kwargs = {}
        for name in _ps._ARRAY_FIELDS:
            arr = getattr(batched, name)
            kwargs[name] = arr.reshape((n_devices, chains_per_device) + arr.shape[1:])
        for name in _ps._STATIC_FIELDS:
            kwargs[name] = getattr(batched, name)
        return keys_pmap, _ps.PackedCrossCatState(**kwargs)

    def unflatten(result_pmap):
        out = []
        for c in range(n_chains):
            dev = c // chains_per_device
            idx = c % chains_per_device
            kwargs = {}
            for name in _ps._ARRAY_FIELDS:
                kwargs[name] = getattr(result_pmap, name)[dev][idx]
            for name in _ps._STATIC_FIELDS:
                kwargs[name] = getattr(result_pmap, name)
            out.append(_ps.PackedCrossCatState(**kwargs))
        return out

    base_key = jax.random.key(seed)
    traces_rows: list[list[float]] = [list(traces_so_far[i]) for i in range(n_chains)]
    start = time.time()
    run_base = meta_base["_run_start_clock"]
    current = current_list

    for chunk_start in range(sweeps_done, n_sweeps, diag_every):
        chunk = min(diag_every, n_sweeps - chunk_start)
        chunk_keys = jax.random.split(jax.random.fold_in(base_key, chunk_start), n_chains)

        batched = batch_packed_states(current)
        keys_pmap, batched_pmap = reshape_for_pmap(batched, chunk_keys)
        result_pmap = pmap_sweep(keys_pmap, batched_pmap, data, chunk)
        jax.tree.map(lambda x: x.block_until_ready(), result_pmap)

        current = unflatten(result_pmap)
        done = chunk_start + chunk

        chunk_scores = [float(packed_log_joint(p, data)) for p in current]
        for ci, lj in enumerate(chunk_scores):
            traces_rows[ci].append(lj)
        traces_now = np.array(traces_rows, dtype=np.float32)

        wall = time.time() - run_base
        # ETA assumes steady-state per-chunk cost after the first chunk.
        chunks_done = traces_now.shape[1]
        chunks_total = (n_sweeps + diag_every - 1) // diag_every
        if chunks_done >= 2:
            per_chunk_recent = wall / chunks_done
            eta = per_chunk_recent * (chunks_total - chunks_done)
            eta_str = f", ETA {eta / 60:.0f} min"
        else:
            eta_str = ""

        print(
            f"  Sweep {done:4d}/{n_sweeps}  best log_joint={max(chunk_scores):,.1f}  "
            f"({wall:.0f}s total, {time.time() - start:.0f}s this session, "
            f"{n_devices} devices){eta_str}",
            flush=True,
        )

        meta = dict(meta_base)
        meta.update(
            last_completed_sweep=done,
            elapsed_seconds=round(wall, 1),
            final_log_joints=traces_now[:, -1].tolist(),
            best_chain_idx=int(np.argmax(traces_now[:, -1])),
        )
        _save_checkpoint(out_dir, current, traces_now, meta, column_types, data_np)
        gc.collect()

    return current, np.array(traces_rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--diag-every", type=int, default=20)
    parser.add_argument("--max-views", type=int, default=16)
    parser.add_argument("--max-clusters", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="If >0, uniformly subsample this many training rows. Useful on low-VRAM GPUs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If a compatible checkpoint exists, pick up from the last completed sweep.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast end-to-end pipeline validation: 2 chains x 6 sweeps x 1000 rows. "
        "Overrides --chains/--sweeps/--diag-every/--subsample.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.chains = 2
        args.sweeps = 6
        args.diag_every = 3
        args.subsample = 1000
        print("SMOKE mode: 2 chains x 6 sweeps x 1000 rows (pipeline validation only)")

    n_devices = jax.device_count()
    mode = "single-device (vmap chains)" if n_devices == 1 else f"pmap across {n_devices} devices"
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"Mode: {mode}")

    n_chains = args.chains
    if n_devices > 1 and n_chains % n_devices:
        rounded = ((n_chains + n_devices - 1) // n_devices) * n_devices
        print(f"  Rounding chains {n_chains} -> {rounded} (must divide n_devices={n_devices})")
        n_chains = rounded

    print(
        f"Config: nhanes_clinical, {n_chains} chains x {args.sweeps} sweeps, "
        f"diag_every={args.diag_every}"
    )

    data_np, column_types, info = _load_preprocessed()
    full_n_rows = data_np.shape[0]
    train_indices: np.ndarray = np.arange(full_n_rows, dtype=np.int64)
    if args.subsample and args.subsample < full_n_rows:
        rng = np.random.default_rng(args.seed)
        sub_idx = rng.choice(full_n_rows, size=args.subsample, replace=False)
        sub_idx.sort()
        train_indices = sub_idx.astype(np.int64)
        data_np = data_np[sub_idx]
        print(
            f"Subsampled training data to {data_np.shape[0]} rows "
            f"(seed={args.seed}; leaves {full_n_rows - data_np.shape[0]} rows as holdout)"
        )
    data = jnp.array(data_np)
    print(
        f"Data: {data.shape[0]} rows x {data.shape[1]} cols, "
        f"NaN fraction {float(jnp.isnan(data).mean()):.2%}"
    )

    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_shape = (data.shape[0], data.shape[1])
    # Save the train_indices now so evaluate_holdout.py can run even if
    # the inference loop is interrupted (it reads the latest checkpoint).
    np.save(out_dir / "train_indices.npy", train_indices)

    # Try resume first if requested
    resume_state: tuple[list, np.ndarray, int] | None = None
    if args.resume:
        resume_state = _try_resume(out_dir, n_chains, args.sweeps, expected_shape)
        if resume_state is not None:
            chains, traces_so_far, sweeps_done = resume_state
            print(f"  Resuming from sweep {sweeps_done}/{args.sweeps}")
        else:
            print("  No compatible checkpoint; starting fresh")

    if resume_state is None:
        print("\nInitializing chains...")
        t0 = time.time()
        chains = _pack_initial_chains(
            data,
            column_types,
            n_chains,
            args.seed,
            max_views=args.max_views,
            max_clusters=args.max_clusters,
        )
        traces_so_far = np.zeros((n_chains, 0), dtype=np.float32)
        sweeps_done = 0
        print(f"  {n_chains} chains initialized in {time.time() - t0:.0f}s")

    meta_base = {
        "dataset": "nhanes_clinical",
        "n_chains": n_chains,
        "n_sweeps": args.sweeps,
        "diag_every": args.diag_every,
        "max_views": args.max_views,
        "max_clusters": args.max_clusters,
        "seed": args.seed,
        "subsample": args.subsample,
        "smoke": args.smoke,
        "n_devices": n_devices,
        "mode": mode,
        "data_shape": list(data.shape),
        "n_total_rows": info.get("n_rows", data.shape[0]),
        "n_columns": info.get("n_cols", data.shape[1]),
        "_run_start_clock": time.time(),
    }

    print(f"\n{'=' * 70}\nRUNNING INFERENCE\n{'=' * 70}")
    t0 = time.time()
    if n_devices == 1:
        finals, traces = _run_single_device(
            chains,
            traces_so_far,
            data,
            data_np,
            sweeps_done,
            args.sweeps,
            args.diag_every,
            args.seed,
            out_dir,
            meta_base,
            column_types,
        )
    else:
        finals, traces = _run_multi_device(
            chains,
            traces_so_far,
            data,
            data_np,
            sweeps_done,
            args.sweeps,
            args.diag_every,
            args.seed,
            out_dir,
            meta_base,
            column_types,
        )
    elapsed = time.time() - t0

    # One final checkpoint (ensures meta's done-state is written once more)
    final_scores = traces[:, -1].tolist() if traces.size else []
    best_idx = int(np.argmax(final_scores)) if final_scores else 0

    final_meta = dict(meta_base)
    final_meta.pop("_run_start_clock", None)
    final_meta.update(
        last_completed_sweep=args.sweeps,
        elapsed_seconds=round(time.time() - meta_base["_run_start_clock"], 1),
        final_log_joints=final_scores,
        best_chain_idx=best_idx,
    )
    _save_checkpoint(out_dir, finals, traces, final_meta, column_types, data_np)
    np.save(out_dir / "train_indices.npy", train_indices)

    print(f"\n{'=' * 70}\nDONE in {elapsed:.0f}s ({elapsed / 60:.1f} min)\n{'=' * 70}")
    for ci, score in enumerate(final_scores):
        marker = "  <-- BEST" if ci == best_idx else ""
        print(f"  Chain {ci}: log_joint={score:,.1f}{marker}")
    print(f"\nSaved to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
