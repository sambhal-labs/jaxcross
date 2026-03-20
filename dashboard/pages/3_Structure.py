"""Structure -- inspect learned column partition and row clustering."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import streamlit as st

from dashboard.components.state_inspector import extract_structure
from dashboard.components.visualization import plot_column_partition, plot_row_clustering

# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------

st.header("Structure")

if not st.session_state.get("inference_done"):
    st.warning("No inference results available. Run inference first on the **Inference** page.")
    st.stop()

# ---------------------------------------------------------------------------
# Extract structure
# ---------------------------------------------------------------------------

packed = st.session_state["packed_state"]
column_names = st.session_state["column_names"]

try:
    views = extract_structure(packed)
    n_views = int(packed.n_views)
except Exception as e:
    st.error(f"Failed to extract structure from model state: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Column partition
# ---------------------------------------------------------------------------

st.subheader("Column Partition")
st.write(f"The model discovered **{n_views}** views (column groups).")

# Build column_assignments array for the visualization
n_cols = len(column_names)
column_assignments = np.array([int(packed.column_assignments[j]) for j in range(n_cols)])

fig_partition = plot_column_partition(column_assignments, column_names, n_views)
st.plotly_chart(fig_partition, use_container_width=True)

# Summary table of views
with st.expander("View summary table", expanded=False):
    for v_info in views:
        cols_in = [column_names[c] for c in v_info["columns"]]
        st.write(
            f"**View {v_info['view_id']}**: {', '.join(cols_in)} "
            f"({v_info['n_clusters']} clusters, CRP alpha={v_info['crp_alpha']:.3f})"
        )

# ---------------------------------------------------------------------------
# Per-view row clustering
# ---------------------------------------------------------------------------

st.subheader("Row Clustering per View")

for view_info in views:
    v = view_info["view_id"]
    cols_in_view = view_info["columns"]
    col_labels = [column_names[c] for c in cols_in_view]

    with st.expander(
        f"View {v} -- {len(cols_in_view)} columns, "
        f"{view_info['n_clusters']} clusters, "
        f"CRP alpha = {view_info['crp_alpha']:.3f}",
        expanded=(v == 0),
    ):
        st.write(f"**Columns:** {', '.join(col_labels)}")
        st.write(f"**CRP alpha:** {view_info['crp_alpha']:.4f}")
        st.write(f"**Cluster sizes:** {view_info['cluster_sizes']}")

        # Row assignments for this view
        row_assigns = np.asarray(packed.view_row_assignments[v])
        fig_clusters = plot_row_clustering(row_assigns, v)
        st.plotly_chart(fig_clusters, use_container_width=True)
