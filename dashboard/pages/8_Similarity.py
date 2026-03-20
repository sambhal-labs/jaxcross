"""Similarity -- compute and visualize pairwise row similarity."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import streamlit as st

from crosscat.packed_inference import packed_row_similarity
from dashboard.components.visualization import plot_similarity_matrix

# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------

st.header("Row Similarity")

if not st.session_state.get("inference_done"):
    st.warning("No inference results available. Run inference first on the **Inference** page.")
    st.stop()

packed = st.session_state["packed_state"]
column_types = st.session_state["column_types"]
data = st.session_state["data"]
n_rows = int(data.shape[0])

# ---------------------------------------------------------------------------
# Row subset selector
# ---------------------------------------------------------------------------

st.subheader("Row Subset")

use_all_rows = st.checkbox(
    "Use all rows",
    value=False,
    key="sim_all_rows",
    help=f"There are {n_rows} rows. Computing all pairs can be slow for large datasets.",
)

if use_all_rows:
    max_rows = n_rows
else:
    max_rows = st.number_input(
        "Max rows to include",
        min_value=2,
        max_value=n_rows,
        value=min(50, n_rows),
        step=5,
        key="sim_max_rows",
    )

n_pairs = max_rows * (max_rows - 1) // 2
st.write(f"Will compute similarity for **{max_rows}** rows ({n_pairs} pairs).")

if n_pairs > 5000:
    st.warning(
        f"Computing {n_pairs} pairs will be slow. "
        "Consider reducing the row count for faster results."
    )
elif n_pairs > 1000:
    st.info(
        f"Computing {n_pairs} pairs may take a moment. "
        "The first pair includes JIT compilation overhead."
    )

# ---------------------------------------------------------------------------
# Compute similarity
# ---------------------------------------------------------------------------

if st.button("Compute Similarity", type="primary"):
    # Use the current packed state as a single-element list (single chain)
    packed_states = [packed]

    subset_size = max_rows
    sim_matrix = np.zeros((subset_size, subset_size))

    # Diagonal is 1.0 (each row is identical to itself)
    np.fill_diagonal(sim_matrix, 1.0)

    progress_bar = st.progress(0, text="Computing pairwise similarity...")
    done = 0

    try:
        for i in range(subset_size):
            for j in range(i + 1, subset_size):
                sim = packed_row_similarity(packed_states, column_types, i, j)
                sim_val = float(sim)
                sim_matrix[i, j] = sim_val
                sim_matrix[j, i] = sim_val

                done += 1
                if done % max(1, n_pairs // 20) == 0 or done == n_pairs:
                    progress_bar.progress(done / n_pairs, text=f"Pair {done}/{n_pairs}")

        progress_bar.progress(1.0, text="Done!")
        st.toast("Similarity matrix computed!", icon="✅")

        row_labels = [f"Row {i}" for i in range(subset_size)]
        st.session_state["sim_matrix"] = sim_matrix
        st.session_state["sim_row_labels"] = row_labels

    except Exception as e:
        st.error(f"Similarity computation failed at pair ({i}, {j}): {e}")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

if "sim_matrix" in st.session_state:
    st.subheader("Similarity Matrix")

    sim_matrix = st.session_state["sim_matrix"]
    row_labels = st.session_state["sim_row_labels"]

    fig = plot_similarity_matrix(sim_matrix, row_labels)
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    # Extract upper triangle (excluding diagonal)
    upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean similarity", f"{upper_tri.mean():.4f}")
    with col2:
        st.metric("Median similarity", f"{np.median(upper_tri):.4f}")
    with col3:
        st.metric("Max similarity", f"{upper_tri.max():.4f}")

    with st.expander("About row similarity"):
        st.write(
            "Row similarity measures how often two rows are assigned to the same cluster "
            "across all views. Values range from **0** (never co-clustered) to **1** "
            "(always co-clustered in every view)."
        )
else:
    st.caption("Click **Compute Similarity** to compute pairwise row similarity.")
