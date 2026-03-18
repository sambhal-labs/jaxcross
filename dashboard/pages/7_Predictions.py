"""Predictions -- posterior predictive sampling and imputation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from crosscat.packed_inference import packed_impute_and_confidence, packed_predictive_sample

# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------

st.header("Predictions")

if not st.session_state.get("inference_done"):
    st.warning("No inference results available. Run inference first on the **Inference** page.")
    st.stop()

packed = st.session_state["packed_state"]
data = st.session_state["data"]
column_names = st.session_state["column_names"]
column_types = st.session_state["column_types"]
n_cols = len(column_names)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_sample, tab_impute = st.tabs(["Sample", "Impute"])

# ---- Sample ---------------------------------------------------------------

with tab_sample:
    st.subheader("Posterior Predictive Sampling")

    query_cols_names = st.multiselect(
        "Query columns (columns to sample)",
        column_names,
        default=[column_names[0]] if column_names else [],
        key="pred_query_cols",
    )
    query_col_indices = [column_names.index(c) for c in query_cols_names]

    # Optional conditioning (row_id)
    use_condition = st.checkbox("Condition on an observed row", key="pred_use_condition")
    row_id = None
    if use_condition:
        row_id = st.number_input(
            "Row index to condition on",
            min_value=0,
            max_value=int(data.shape[0]) - 1,
            value=0,
            step=1,
            key="pred_row_id",
        )

    n_samples = st.slider("Number of samples", 100, 5000, 1000, step=100, key="pred_n_samples")

    if not query_col_indices:
        st.info("Select at least one query column.")
    else:
        if st.button("Sample", type="primary", key="btn_sample"):
            rng_key = jax.random.key(123)

            with st.spinner("Drawing posterior predictive samples..."):
                samples = packed_predictive_sample(
                    rng_key,
                    packed,
                    data,
                    query_col_indices,
                    n_samples=n_samples,
                    row_id=row_id,
                )

            samples_np = np.asarray(samples)
            st.session_state["pred_samples"] = samples_np
            st.session_state["pred_sample_cols"] = query_cols_names

    # Display samples if available
    if "pred_samples" in st.session_state:
        samples_np = st.session_state["pred_samples"]
        sample_cols = st.session_state["pred_sample_cols"]

        if len(sample_cols) == 1:
            # Histogram for single column
            fig = go.Figure(
                data=go.Histogram(
                    x=samples_np[:, 0],
                    nbinsx=40,
                    marker_color="#1f77b4",
                    opacity=0.8,
                )
            )
            fig.update_layout(
                title=f"Posterior Predictive: {sample_cols[0]}",
                xaxis_title=sample_cols[0],
                yaxis_title="Count",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif len(sample_cols) == 2:
            # Scatter for two columns
            fig = go.Figure(
                data=go.Scattergl(
                    x=samples_np[:, 0],
                    y=samples_np[:, 1],
                    mode="markers",
                    marker=dict(size=3, opacity=0.5, color="#1f77b4"),
                )
            )
            fig.update_layout(
                title=f"Posterior Predictive: {sample_cols[0]} vs {sample_cols[1]}",
                xaxis_title=sample_cols[0],
                yaxis_title=sample_cols[1],
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # More than 2 columns: show histograms for each
            for i, col_name in enumerate(sample_cols):
                fig = go.Figure(
                    data=go.Histogram(
                        x=samples_np[:, i],
                        nbinsx=40,
                        marker_color="#1f77b4",
                        opacity=0.8,
                    )
                )
                fig.update_layout(
                    title=f"Posterior Predictive: {col_name}",
                    xaxis_title=col_name,
                    yaxis_title="Count",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        with st.expander("Sample statistics"):
            for i, col_name in enumerate(sample_cols):
                s = samples_np[:, i]
                st.write(
                    f"**{col_name}**: mean={np.mean(s):.4f}, "
                    f"std={np.std(s):.4f}, "
                    f"median={np.median(s):.4f}"
                )

# ---- Impute ---------------------------------------------------------------

with tab_impute:
    st.subheader("Imputation with Confidence")

    impute_col_name = st.selectbox(
        "Column to impute",
        column_names,
        key="impute_col",
    )
    impute_col_idx = column_names.index(impute_col_name)

    if st.button("Impute", type="primary", key="btn_impute"):
        rng_key = jax.random.key(456)

        with st.spinner("Computing imputation..."):
            point_est, confidence = packed_impute_and_confidence(
                rng_key,
                packed,
                data,
                impute_col_idx,
            )

        point_val = float(point_est)
        conf_val = float(confidence)

        st.session_state["impute_result"] = {
            "column": impute_col_name,
            "point_estimate": point_val,
            "confidence": conf_val,
        }

    # Display imputation result
    if "impute_result" in st.session_state:
        result = st.session_state["impute_result"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                f"Point estimate ({result['column']})",
                f"{result['point_estimate']:.4f}",
            )
        with col2:
            st.metric("Confidence", f"{result['confidence']:.4f}")

        st.caption(
            "For continuous columns, confidence is exp(-IQR/std). "
            "For discrete columns, confidence is the mode frequency."
        )
