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

    # Workflow status ---------------------------------------------------------
    data_loaded = st.session_state["data"] is not None
    inference_done = st.session_state.get("inference_done", False)

    if data_loaded:
        data = st.session_state["data"]
        col_names = st.session_state["column_names"]
        col_types = st.session_state["column_types"]
        n_rows, n_cols = data.shape

        st.subheader("Data Summary")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Rows", n_rows)
        with c2:
            st.metric("Columns", n_cols)

        type_counts: dict[str, int] = {}
        for ct in col_types:
            label = ct.value
            type_counts[label] = type_counts.get(label, 0) + 1
        for label, count in sorted(type_counts.items()):
            st.write(f"- **{label}**: {count}")
    else:
        st.info("No data loaded yet. Start on the **Data Loading** page.")

    # Inference status -------------------------------------------------------
    st.subheader("Inference Status")
    history = st.session_state["sweep_history"]
    if history:
        latest = history[-1]
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Sweeps", len(history))
        with c2:
            st.metric("Views", latest["n_views"])
        st.metric("log_joint", f"{latest['log_joint']:.2f}")
    elif data_loaded:
        st.caption("Data is loaded. Go to the **Inference** page to run Gibbs sweeps.")
    else:
        st.caption("Load data first, then run inference.")

    # Workflow guide ----------------------------------------------------------
    st.divider()
    st.subheader("Workflow")
    step1 = "~~1. Load data~~" if data_loaded else "**1. Load data**"
    step2 = "~~2. Run inference~~" if inference_done else "**2. Run inference**"
    step3 = "**3. Explore results**" if inference_done else "3. Explore results"
    st.markdown(f"{step1}  \n{step2}  \n{step3}")

    # Version & reset --------------------------------------------------------
    st.divider()
    st.caption(f"jax-crosscat v{crosscat.__version__}")

    if st.button("Reset All", type="primary", use_container_width=True):
        for key, default in _DEFAULTS.items():
            st.session_state[key] = default
        # Also clear computed results
        for result_key in [
            "mi_matrix",
            "linfoot_matrix",
            "anomaly_scores",
            "pred_samples",
            "pred_sample_cols",
            "impute_result",
            "sim_matrix",
            "sim_row_labels",
            "engine",
        ]:
            st.session_state.pop(result_key, None)
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
           Data Loading page.
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
        st.info("Data is loaded. Navigate to the **Inference** page to run Gibbs sweeps.")
