"""Inference -- run Gibbs sweeps on the loaded data."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import streamlit as st

from dashboard.components.inference_engine import InferenceEngine

# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------

st.header("Inference")

if st.session_state.get("data") is None:
    st.warning("No data loaded. Go to the **Data Loading** page first.")
    st.stop()

data = st.session_state["data"]
n_rows, n_cols = data.shape
st.caption(f"Dataset: **{n_rows}** rows x **{n_cols}** columns")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.subheader("Configuration")

col1, col2 = st.columns(2)
with col1:
    n_sweeps = st.slider("Number of sweeps", 10, 500, 50, step=10, key="inf_n_sweeps")
    max_views = st.slider("Max views (padding)", 4, 32, 16, key="inf_max_views")

with col2:
    max_clusters = st.slider("Max clusters (padding)", 8, 64, 32, key="inf_max_clusters")
    seed = st.number_input("Random seed", value=0, min_value=0, step=1, key="inf_seed")

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------

st.divider()

if st.button("Run Inference", type="primary"):
    column_types = st.session_state["column_types"]

    try:
        # Phase 1: Initialization
        with st.status("Initializing model...", expanded=True) as status:
            st.write("Creating CrossCat state from the prior...")
            engine = InferenceEngine(
                data,
                column_types,
                max_views=max_views,
                max_clusters=max_clusters,
            )
            status.update(label="Model initialized", state="complete")

        # Phase 2: Gibbs sweeps
        sweep_history: list[dict] = []
        progress_bar = st.progress(0, text="Preparing Gibbs sweeps...")

        rng_key = jax.random.key(int(seed))

        for i in range(n_sweeps):
            rng_key, sweep_key = jax.random.split(rng_key)

            # Special message for first sweep (JIT compilation)
            if i == 0:
                progress_bar.progress(
                    0,
                    text=f"Sweep 1/{n_sweeps} -- JIT compiling (this sweep will be slower)...",
                )

            t0 = time.time()
            result = engine.run_sweep(sweep_key)
            elapsed = time.time() - t0

            sweep_history.append(result)

            progress = (i + 1) / n_sweeps
            if i == 0:
                progress_bar.progress(
                    progress,
                    text=(
                        f"Sweep 1/{n_sweeps} done (JIT compile: {elapsed:.1f}s) | "
                        f"log_joint: {result['log_joint']:.2f} | views: {result['n_views']}"
                    ),
                )
            else:
                progress_bar.progress(
                    progress,
                    text=(
                        f"Sweep {i + 1}/{n_sweeps} ({elapsed:.2f}s) | "
                        f"log_joint: {result['log_joint']:.2f} | views: {result['n_views']}"
                    ),
                )

        progress_bar.progress(1.0, text="Inference complete!")

        # Store results in session state
        st.session_state["engine"] = engine
        st.session_state["packed_state"] = engine.get_packed_state()
        st.session_state["sweep_history"] = sweep_history
        st.session_state["inference_done"] = True

        st.success(f"Inference complete: **{n_sweeps}** sweeps finished.")
        st.toast("Inference complete!", icon="✅")

    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.caption("Try adjusting max views/clusters or check that data is valid.")

# ---------------------------------------------------------------------------
# Summary (shown when inference has been run)
# ---------------------------------------------------------------------------

if st.session_state.get("inference_done"):
    st.divider()
    st.subheader("Results Summary")

    history = st.session_state["sweep_history"]
    latest = history[-1]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Sweeps completed", len(history))
    with col_b:
        st.metric("Final log_joint", f"{latest['log_joint']:.2f}")
    with col_c:
        st.metric("Active views", latest["n_views"])

    if len(history) > 1:
        improvement = latest["log_joint"] - history[0]["log_joint"]
        st.write(f"log_joint improvement: **{improvement:+.2f}**")

    st.info("Navigate to **Structure**, **Dependencies**, or other pages to explore results.")
