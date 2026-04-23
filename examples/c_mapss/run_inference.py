#!/usr/bin/env python3
"""Run multi-chain Gibbs inference on a C-MAPSS sub-dataset.

Auto-selects the execution mode from the JAX device count:
  - 1 device   -> sequential chains, each using packed_gibbs_sweep
                  (matches examples/materials_project/run_local_multichain.py
                  memory profile; safe on 4 GB GTX 1650)
  - N devices  -> jax.pmap across devices, CHAINS_PER_DEVICE = N_CHAINS // N
                  (matches benchmarks/wdi_macroeconomic_benchmark.ipynb)

Outputs (examples/c_mapss/results/inference/<fd>/):
  chain_{i}.jxc               per-chain final packed state
  best_chain.jxc              argmax-log-joint chain
  log_joint_traces.npy        (N_CHAINS, N_DIAG) diagnostic trace
  inference_meta.json         config + timing + device info

Usage:
    uv run python examples/c_mapss/run_inference.py [FD001] [--sweeps 200] [--chains 4]
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
from crosscat.serialization import save_packed_state
from crosscat.types import ColumnType

PREP_ROOT = Path("examples/c_mapss/results/preprocessed")
OUT_ROOT = Path("examples/c_mapss/results/inference")

_TYPE_MAP = {
    "CONTINUOUS": ColumnType.CONTINUOUS,
    "CATEGORICAL": ColumnType.CATEGORICAL,
    "ORDINAL": ColumnType.ORDINAL,
    "BINARY": ColumnType.BINARY,
    "CYCLIC": ColumnType.CYCLIC,
}


def _load_preprocessed(fd: str):
    fd_dir = PREP_ROOT / fd
    if not fd_dir.exists():
        raise FileNotFoundError(f"Missing {fd_dir} — run preprocess_cmapss.py first")
    train = np.load(fd_dir / "train_data.npy")
    info = json.loads((fd_dir / "column_info.json").read_text())
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


def _run_single_device(
    init_chains: list,
    data: jnp.ndarray,
    n_sweeps: int,
    diag_every: int,
    seed: int,
) -> tuple[list, np.ndarray]:
    """Run N chains on a single device via multi_chain_packed_gibbs_sweep.

    Uses vmap over chains; one JIT compile handles all chains in parallel
    (still single-device, but avoids 4x recompilation from the sequential loop).
    """
    n_chains = len(init_chains)
    traces: list[list[float]] = [[] for _ in range(n_chains)]
    start = time.time()
    current_list = init_chains

    for chunk_start in range(0, n_sweeps, diag_every):
        chunk = min(diag_every, n_sweeps - chunk_start)
        chunk_key = jax.random.fold_in(jax.random.key(seed), chunk_start)

        batched, scores = multi_chain_packed_gibbs_sweep(
            chunk_key, current_list, data, n_sweeps=chunk
        )
        jax.block_until_ready(batched.view_row_assignments)
        current_list = unbatch_packed_states(batched, n_chains)

        done = chunk_start + chunk
        for ci, s in enumerate(np.asarray(scores).tolist()):
            traces[ci].append(float(s))
        best = max(traces[ci][-1] for ci in range(n_chains))
        print(
            f"  Sweep {done:4d}/{n_sweeps}  best log_joint={best:,.1f}  "
            f"({time.time() - start:.0f}s, {n_chains} chains vmapped)",
            flush=True,
        )
        gc.collect()

    return current_list, np.array(traces, dtype=np.float32)


def _run_multi_device(
    init_chains: list,
    data: jnp.ndarray,
    n_sweeps: int,
    diag_every: int,
    seed: int,
) -> tuple[list, np.ndarray]:
    """pmap across devices — one batch of CHAINS_PER_DEVICE per device.

    Layout follows benchmarks/wdi_macroeconomic_benchmark.ipynb:
        keys_pmap      (n_devices, CHAINS_PER_DEVICE, 2)
        batched_pmap   PackedCrossCatState with leading (n_devices, CHAINS_PER_DEVICE)
    """
    n_devices = jax.device_count()
    n_chains = len(init_chains)
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
    traces: list[list[float]] = [[] for _ in range(n_chains)]
    start = time.time()
    current = init_chains

    for chunk_start in range(0, n_sweeps, diag_every):
        chunk = min(diag_every, n_sweeps - chunk_start)

        # Per-chunk independent keys: fold the chunk index into the base key,
        # then split into n_chains scalar keys. Shape: (n_chains,) of key dtype.
        chunk_keys = jax.random.split(jax.random.fold_in(base_key, chunk_start), n_chains)

        batched = batch_packed_states(current)
        keys_pmap, batched_pmap = reshape_for_pmap(batched, chunk_keys)
        result_pmap = pmap_sweep(keys_pmap, batched_pmap, data, chunk)
        jax.tree.map(lambda x: x.block_until_ready(), result_pmap)

        current = unflatten(result_pmap)
        done = chunk_start + chunk

        chunk_scores = [float(packed_log_joint(p, data)) for p in current]
        for ci, lj in enumerate(chunk_scores):
            traces[ci].append(lj)
        print(
            f"  Sweep {done:4d}/{n_sweeps}  best log_joint={max(chunk_scores):,.1f}  "
            f"({time.time() - start:.0f}s on {n_devices} devices)",
            flush=True,
        )
        gc.collect()

    return current, np.array(traces, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fd", nargs="?", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"]
    )
    parser.add_argument("--sweeps", type=int, default=200)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--diag-every", type=int, default=20)
    parser.add_argument("--max-views", type=int, default=16)
    parser.add_argument("--max-clusters", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="If >0, uniformly subsample this many training rows (fixed seed). "
        "Useful on low-VRAM GPUs; 5000 rows is typically sufficient for CrossCat on C-MAPSS.",
    )
    args = parser.parse_args()

    n_devices = jax.device_count()
    mode = "single-device (vmap chains)" if n_devices == 1 else f"pmap across {n_devices} devices"
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"Mode: {mode}")

    # Round chain count to a multiple of device count for pmap paths.
    n_chains = args.chains
    if n_devices > 1 and n_chains % n_devices:
        rounded = ((n_chains + n_devices - 1) // n_devices) * n_devices
        print(f"  Rounding chains {n_chains} -> {rounded} (must divide n_devices={n_devices})")
        n_chains = rounded

    print(f"Config: fd={args.fd}, {n_chains} chains x {args.sweeps} sweeps")

    data_np, column_types, info = _load_preprocessed(args.fd)
    if args.subsample and args.subsample < data_np.shape[0]:
        rng = np.random.default_rng(args.seed)
        sub_idx = rng.choice(data_np.shape[0], size=args.subsample, replace=False)
        sub_idx.sort()
        data_np = data_np[sub_idx]
        print(f"Subsampled training data to {data_np.shape[0]} rows (seed={args.seed})")
    data = jnp.array(data_np)
    print(
        f"Data: {data.shape[0]} rows x {data.shape[1]} cols, "
        f"NaN fraction {float(jnp.isnan(data).mean()):.2%}"
    )

    print("\nInitializing chains...")
    t0 = time.time()
    init_chains = _pack_initial_chains(
        data,
        column_types,
        n_chains,
        args.seed,
        max_views=args.max_views,
        max_clusters=args.max_clusters,
    )
    print(f"  {n_chains} chains initialized in {time.time() - t0:.0f}s")

    print(f"\n{'=' * 70}\nRUNNING INFERENCE\n{'=' * 70}")
    t0 = time.time()
    if n_devices == 1:
        finals, traces = _run_single_device(
            init_chains, data, args.sweeps, args.diag_every, args.seed
        )
    else:
        finals, traces = _run_multi_device(
            init_chains, data, args.sweeps, args.diag_every, args.seed
        )
    elapsed = time.time() - t0

    # Save
    out_dir = OUT_ROOT / args.fd
    out_dir.mkdir(parents=True, exist_ok=True)
    final_scores = [float(packed_log_joint(p, data)) for p in finals]
    best_idx = int(np.argmax(final_scores))

    for ci, packed in enumerate(finals):
        save_packed_state(packed, str(out_dir / f"chain_{ci}.jxc"), column_types=column_types)
    save_packed_state(finals[best_idx], str(out_dir / "best_chain.jxc"), column_types=column_types)
    np.save(out_dir / "log_joint_traces.npy", traces)
    # Persist the exact training array used so evaluate_rul.py can insert
    # test-engine rows into the same base data (important when --subsample is set).
    np.save(out_dir / "train_used.npy", data_np)

    meta = {
        "fd": args.fd,
        "n_chains": n_chains,
        "n_sweeps": args.sweeps,
        "diag_every": args.diag_every,
        "max_views": args.max_views,
        "max_clusters": args.max_clusters,
        "seed": args.seed,
        "n_devices": n_devices,
        "mode": mode,
        "elapsed_seconds": round(elapsed, 1),
        "final_log_joints": final_scores,
        "best_chain_idx": best_idx,
        "data_shape": list(data.shape),
        "n_train_rows": info["n_train_rows"],
        "n_test_engines": info["n_test_engines"],
    }
    (out_dir / "inference_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n{'=' * 70}\nDONE in {elapsed:.0f}s ({elapsed / 60:.1f} min)\n{'=' * 70}")
    for ci, score in enumerate(final_scores):
        marker = "  <-- BEST" if ci == best_idx else ""
        print(f"  Chain {ci}: log_joint={score:,.1f}{marker}")
    print(f"\nSaved to {out_dir}/")
    print(
        f"  best_chain.jxc, chain_{{0..{n_chains - 1}}}.jxc, log_joint_traces.npy, inference_meta.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
