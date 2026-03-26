"""Ahead-of-time compilation caching for packed Gibbs kernels.

Provides XLA persistent compilation caching so the expensive JIT compilation
only happens once per shape config. Subsequent runs load from disk cache.

Usage:
    from crosscat.packed.aot_cache import enable_xla_cache, compile_kernels

    enable_xla_cache()  # Auto-called on import of crosscat.packed

    # Optional: pre-compile kernels for a specific shape
    compile_kernels(packed, data)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import jax

from crosscat.packed.state import _STATIC_FIELDS, PackedCrossCatState

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "jaxcross" / "aot"
_xla_cache_enabled = False


def _shape_signature(packed: PackedCrossCatState, data_shape: tuple) -> str:
    """Create a deterministic hash from static fields + data shape."""
    sig = {name: getattr(packed, name) for name in _STATIC_FIELDS}
    sig["data_shape"] = list(data_shape)
    sig["backend"] = str(jax.default_backend())
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]


def _cache_path(fn_name: str, sig: str, cache_dir: Path) -> Path:
    return cache_dir / f"{fn_name}_{sig}.bin"


def _meta_path(fn_name: str, sig: str, cache_dir: Path) -> Path:
    return cache_dir / f"{fn_name}_{sig}.json"


def compile_and_cache(
    fn,
    fn_name: str,
    *example_args,
    cache_dir: Path | None = None,
):
    """Compile a function via JAX and cache the compiled executable.

    Uses jax.jit().lower().compile() to get a compiled function, then
    stores metadata about the compilation for cache invalidation.

    Args:
        fn: The function to compile (will be wrapped with jax.jit if not already).
        fn_name: Name for the cache key.
        *example_args: Example arguments for tracing.
        cache_dir: Directory for cache files.

    Returns:
        A compiled callable.
    """
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build signature from args
    sig_parts = []
    for arg in example_args:
        if hasattr(arg, "shape"):
            sig_parts.append(f"{arg.shape}_{arg.dtype}")
        elif isinstance(arg, PackedCrossCatState):
            for name in _STATIC_FIELDS:
                sig_parts.append(f"{name}={getattr(arg, name)}")
        else:
            sig_parts.append(str(arg))
    sig = hashlib.sha256("_".join(sig_parts).encode()).hexdigest()[:16]

    meta_file = _meta_path(fn_name, sig, cache_dir)

    if meta_file.exists():
        logger.info("AOT cache hit for %s (sig=%s)", fn_name, sig)
        # XLA persistent cache handles actual executable reuse
        return jax.jit(fn)

    logger.info("Compiling %s (sig=%s)...", fn_name, sig)
    jitted = jax.jit(fn)
    lowered = jitted.lower(*example_args)
    compiled = lowered.compile()

    # Save metadata for diagnostics
    meta = {
        "fn_name": fn_name,
        "sig": sig,
        "backend": str(jax.default_backend()),
        "cost_analysis": str(compiled.cost_analysis()),
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    return compiled


def enable_xla_cache(cache_dir: Path | None = None):
    """Enable JAX's persistent compilation cache.

    This is the most reliable way to cache XLA compilations across runs.
    Configures JAX to persist compiled executables to disk. Idempotent —
    safe to call multiple times.

    Args:
        cache_dir: Directory for the cache. Defaults to ~/.cache/jaxcross/xla.
    """
    global _xla_cache_enabled
    if _xla_cache_enabled:
        return

    # Don't override if user already configured via environment
    if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        _xla_cache_enabled = True
        return

    cache_dir = cache_dir or (Path.home() / ".cache" / "jaxcross" / "xla")
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    _xla_cache_enabled = True
    logger.info("XLA persistent cache enabled at %s", cache_dir)


def compile_kernels(packed: PackedCrossCatState, data) -> None:
    """Pre-compile all Gibbs sub-kernels for the given state shape.

    Call this after packing state to trigger compilation upfront rather than
    on the first inference call. Each kernel compiles independently, so this
    is faster than compiling the monolithic packed_gibbs_sweep.

    Args:
        packed: Packed state (determines compilation shapes).
        data: Data matrix (n_rows, n_cols).
    """
    from crosscat.packed.kernels import (
        packed_transition_column_assignments,
        packed_transition_column_hypers,
        packed_transition_crp_alphas,
        packed_transition_row_assignments,
    )

    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    # Trigger compilation by calling each @jax.jit kernel
    packed_transition_row_assignments(k1, packed, data).view_mask.block_until_ready()
    packed_transition_column_assignments(k2, packed, data).view_mask.block_until_ready()
    packed_transition_column_hypers(k3, packed, data).view_mask.block_until_ready()
    packed_transition_crp_alphas(k4, packed).view_mask.block_until_ready()
    logger.info(
        "All 4 Gibbs sub-kernels compiled for shape: %d rows, %d cols",
        packed.n_rows,
        packed.n_cols,
    )


def clear_cache(cache_dir: Path | None = None):
    """Remove all cached compilations."""
    import shutil

    for d in [
        cache_dir or _DEFAULT_CACHE_DIR,
        Path.home() / ".cache" / "jaxcross" / "xla",
    ]:
        if d.exists():
            shutil.rmtree(d)
            logger.info("Cleared cache at %s", d)
