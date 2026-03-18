"""Convergence -- visualize log_joint and n_views over Gibbs sweeps."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from dashboard.components.visualization import plot_convergence

# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------

st.header("Convergence")

sweep_history = st.session_state.get("sweep_history", [])

if not sweep_history:
    st.warning("No sweep history available. Run inference first on the **Inference** page.")
    st.stop()

# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

st.subheader("log_joint & Number of Views over Sweeps")

fig = plot_convergence(sweep_history)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

st.subheader("Summary")

initial = sweep_history[0]
final = sweep_history[-1]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Initial log_joint", f"{initial['log_joint']:.2f}")
with col2:
    st.metric("Final log_joint", f"{final['log_joint']:.2f}")
with col3:
    improvement = final["log_joint"] - initial["log_joint"]
    st.metric("Improvement", f"{improvement:+.2f}")
with col4:
    st.metric("Final views", final["n_views"])

# Show the full trajectory in a table if user wants
with st.expander("Full sweep history"):
    st.dataframe(
        [
            {
                "sweep": i + 1,
                "log_joint": h["log_joint"],
                "n_views": h["n_views"],
            }
            for i, h in enumerate(sweep_history)
        ],
        use_container_width=True,
    )
