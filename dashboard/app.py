"""JAX-CrossCat Explorer -- Streamlit dashboard entry point.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

import crosscat

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="JAX-CrossCat Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, object] = {
    "data": None,
    "column_names": None,
    "column_types": None,
    "packed_state": None,
    "sweep_history": [],
    "inference_done": False,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("JAX-CrossCat Explorer")

    # Data summary ---------------------------------------------------------
    if st.session_state["data"] is not None:
        data = st.session_state["data"]
        col_names = st.session_state["column_names"]
        col_types = st.session_state["column_types"]
        n_rows, n_cols = data.shape

        st.subheader("Data Summary")
        st.metric("Rows", n_rows)
        st.metric("Columns", n_cols)

        type_counts: dict[str, int] = {}
        for ct in col_types:
            label = ct.value
            type_counts[label] = type_counts.get(label, 0) + 1
        for label, count in sorted(type_counts.items()):
            st.write(f"- **{label}**: {count}")
    else:
        st.info("No data loaded yet.")

    # Inference status -----------------------------------------------------
    st.subheader("Inference Status")
    history = st.session_state["sweep_history"]
    if history:
        latest = history[-1]
        st.metric("Sweeps completed", len(history))
        st.metric("log_joint", f"{latest['log_joint']:.2f}")
        st.metric("Active views", latest["n_views"])
    else:
        st.write("No inference run yet.")

    # Version & reset ------------------------------------------------------
    st.divider()
    st.caption(f"jax-crosscat v{crosscat.__version__}")

    if st.button("Reset", type="primary", use_container_width=True):
        for key, default in _DEFAULTS.items():
            st.session_state[key] = default
        st.rerun()

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

if st.session_state["data"] is None:
    st.header("Welcome to JAX-CrossCat Explorer")
    st.markdown(
        """
        **JAX-CrossCat** is a GPU-accelerated reimplementation of
        [probcomp/crosscat](https://github.com/probcomp/crosscat) using JAX.

        It implements a two-level Dirichlet Process mixture model:

        - **Outer DP** partitions columns into *views* (groups of related columns).
        - **Inner DP** per view clusters rows independently.

        All parameters are collapsed out via conjugate Bayesian component models --
        only cluster assignments and hyperparameters are sampled via Gibbs MCMC.

        ### Getting started

        1. **Load data** -- upload a CSV or generate synthetic data via the
           Data Manager page.
        2. **Run inference** -- execute Gibbs sweeps to learn the model structure.
        3. **Explore** -- inspect the discovered column partition, row clustering,
           mutual information, and anomaly scores.
        """
    )
else:
    st.header("Data loaded")
    st.write(
        f"**{st.session_state['data'].shape[0]}** rows, "
        f"**{st.session_state['data'].shape[1]}** columns"
    )
    if st.session_state["inference_done"]:
        st.success("Inference complete. Use the sidebar to navigate to analysis pages.")
    else:
        st.info("Data is loaded. Run inference to explore the model structure.")
