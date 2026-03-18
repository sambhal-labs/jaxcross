"""Dependencies -- mutual information and Linfoot correlation between columns."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import streamlit as st

from crosscat.packed_inference import packed_mutual_information
from dashboard.components.visualization import plot_mi_matrix

# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------

st.header("Dependencies")

if not st.session_state.get("inference_done"):
    st.warning("No inference results available. Run inference first on the **Inference** page.")
    st.stop()

# ---------------------------------------------------------------------------
# Compute MI matrix
# ---------------------------------------------------------------------------

packed = st.session_state["packed_state"]
column_names = st.session_state["column_names"]
column_types = st.session_state["column_types"]
n_cols = len(column_names)

st.write(
    f"Compute pairwise mutual information for **{n_cols}** columns "
    f"({n_cols * (n_cols - 1) // 2} unique pairs)."
)

if st.button("Compute MI Matrix", type="primary"):
    # Use the current packed state as a single-element list (single chain)
    packed_states = [packed]

    mi_matrix = np.zeros((n_cols, n_cols))
    linfoot_matrix = np.zeros((n_cols, n_cols))

    progress_bar = st.progress(0, text="Computing mutual information...")
    total_pairs = n_cols * (n_cols - 1) // 2
    done = 0

    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            mi, linfoot = packed_mutual_information(packed_states, column_types, i, j)
            mi_val = float(mi)
            lf_val = float(linfoot)

            mi_matrix[i, j] = mi_val
            mi_matrix[j, i] = mi_val
            linfoot_matrix[i, j] = lf_val
            linfoot_matrix[j, i] = lf_val

            done += 1
            if total_pairs > 0:
                progress_bar.progress(done / total_pairs, text=f"Pair {done}/{total_pairs}")

    progress_bar.progress(1.0, text="Done!")

    # Store for display
    st.session_state["mi_matrix"] = mi_matrix
    st.session_state["linfoot_matrix"] = linfoot_matrix

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

if "mi_matrix" in st.session_state:
    tab_mi, tab_linfoot = st.tabs(["Mutual Information", "Linfoot Correlation"])

    with tab_mi:
        st.subheader("Mutual Information")
        fig_mi = plot_mi_matrix(st.session_state["mi_matrix"], column_names)
        st.plotly_chart(fig_mi, use_container_width=True)

    with tab_linfoot:
        st.subheader("Linfoot Correlation")
        st.caption(
            "Linfoot correlation transforms MI into a [0, 1] scale analogous "
            "to Pearson r: linfoot = sqrt(1 - exp(-2 * MI))."
        )
        fig_lf = plot_mi_matrix(st.session_state["linfoot_matrix"], column_names)
        st.plotly_chart(fig_lf, use_container_width=True)
else:
    st.info("Click **Compute MI Matrix** to compute pairwise dependencies.")
