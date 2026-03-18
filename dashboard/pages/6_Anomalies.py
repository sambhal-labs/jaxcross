"""Anomalies -- compute and visualize row-level anomaly scores."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import numpy as np
import pandas as pd
import streamlit as st

from crosscat.packed_inference import packed_anomaly_score
from dashboard.components.visualization import plot_anomaly_distribution

# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------

st.header("Anomalies")

if not st.session_state.get("inference_done"):
    st.warning("No inference results available. Run inference first on the **Inference** page.")
    st.stop()

# ---------------------------------------------------------------------------
# Compute anomaly scores
# ---------------------------------------------------------------------------

packed = st.session_state["packed_state"]
data = st.session_state["data"]
column_names = st.session_state["column_names"]
n_rows = data.shape[0]

st.write(f"Compute anomaly scores for **{n_rows}** rows.")

if st.button("Compute Anomaly Scores", type="primary"):
    scores = []
    rng_key = jax.random.key(0)

    progress_bar = st.progress(0, text="Scoring rows...")

    for row in range(n_rows):
        rng_key, score_key = jax.random.split(rng_key)
        score = packed_anomaly_score(score_key, packed, data, row)
        scores.append(float(score))

        if (row + 1) % max(1, n_rows // 20) == 0 or row == n_rows - 1:
            progress_bar.progress((row + 1) / n_rows, text=f"Row {row + 1}/{n_rows}")

    progress_bar.progress(1.0, text="Done!")

    st.session_state["anomaly_scores"] = np.array(scores)

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

if "anomaly_scores" in st.session_state:
    scores = st.session_state["anomaly_scores"]

    st.subheader("Anomaly Score Distribution")
    fig = plot_anomaly_distribution(scores)
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean score", f"{scores.mean():.4f}")
    with col2:
        st.metric("Median score", f"{np.median(scores):.4f}")
    with col3:
        st.metric("Max score", f"{scores.max():.4f}")

    # Top-10 most anomalous rows
    st.subheader("Top 10 Most Anomalous Rows")
    sorted_indices = np.argsort(scores)[::-1]
    top_10_indices = sorted_indices[:10]

    top_rows = []
    for rank, idx in enumerate(top_10_indices, start=1):
        row_data = {"Rank": rank, "Row": int(idx), "Anomaly Score": f"{scores[idx]:.4f}"}
        # Include first few column values for context
        for j, col_name in enumerate(column_names[:6]):
            row_data[col_name] = f"{float(data[idx, j]):.3f}"
        if len(column_names) > 6:
            row_data["..."] = "..."
        top_rows.append(row_data)

    st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)
else:
    st.info("Click **Compute Anomaly Scores** to score all rows.")
