"""Data Loading -- upload CSV or generate synthetic CrossCat data."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure crosscat is importable from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import numpy as np
import pandas as pd
import streamlit as st

from crosscat.types import ColumnType
from dashboard.components.data_manager import generate_synthetic, load_csv_data

# ---------------------------------------------------------------------------
# Column type options
# ---------------------------------------------------------------------------

_COLUMN_TYPE_OPTIONS = [ct.value for ct in ColumnType]

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.header("Data Loading")

tab_upload, tab_synthetic = st.tabs(["Upload CSV", "Synthetic Data"])

# ---- Upload CSV -----------------------------------------------------------

with tab_upload:
    st.subheader("Upload a CSV file")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="All columns will be coerced to numeric. Non-numeric values become NaN.",
    )

    if uploaded_file is not None:
        # Parse the file
        data, column_names, column_types = load_csv_data(uploaded_file)

        st.write(f"Parsed **{data.shape[0]}** rows and **{data.shape[1]}** columns.")

        # Preview
        preview_df = pd.DataFrame(np.asarray(data), columns=column_names)
        st.dataframe(preview_df.head(50), use_container_width=True)

        # Let user override column types
        st.subheader("Column Types")
        st.info("Auto-detected types are shown below. Override any column type before confirming.")

        overridden_types: list[ColumnType] = []
        cols_per_row = 4
        for i in range(0, len(column_names), cols_per_row):
            row_cols = st.columns(min(cols_per_row, len(column_names) - i))
            for j, col_widget in enumerate(row_cols):
                idx = i + j
                with col_widget:
                    default_idx = _COLUMN_TYPE_OPTIONS.index(column_types[idx].value)
                    selected = st.selectbox(
                        column_names[idx],
                        _COLUMN_TYPE_OPTIONS,
                        index=default_idx,
                        key=f"col_type_{idx}",
                    )
                    overridden_types.append(ColumnType(selected))

        # Confirm button
        if st.button("Confirm & Load", key="confirm_csv", type="primary"):
            st.session_state["data"] = data
            st.session_state["column_names"] = column_names
            st.session_state["column_types"] = overridden_types
            st.session_state["packed_state"] = None
            st.session_state["sweep_history"] = []
            st.session_state["inference_done"] = False
            st.success("Data loaded successfully!")
            st.rerun()
    else:
        st.info("Upload a CSV file to get started.")

# ---- Synthetic Data -------------------------------------------------------

with tab_synthetic:
    st.subheader("Generate Synthetic Data")

    col_left, col_right = st.columns(2)
    with col_left:
        n_rows = st.slider("Number of rows", 50, 1000, 200, step=10, key="syn_n_rows")
        seed = st.number_input("Random seed", value=42, min_value=0, step=1, key="syn_seed")
        n_views = st.slider("Number of views", 1, 8, 2, key="syn_n_views")

    with col_right:
        n_clusters = st.slider("Clusters per view", 2, 10, 3, key="syn_n_clusters")
        cluster_separation = st.slider(
            "Cluster separation", 1.0, 10.0, 5.0, step=0.5, key="syn_sep"
        )

    # Dynamic column type builder
    st.subheader("Column Types")
    st.caption("Add columns with the desired types. At least one column is required.")

    if "syn_col_types" not in st.session_state:
        # Default: 4 continuous + 2 categorical
        st.session_state["syn_col_types"] = [
            ColumnType.CONTINUOUS.value,
            ColumnType.CONTINUOUS.value,
            ColumnType.CONTINUOUS.value,
            ColumnType.CONTINUOUS.value,
            ColumnType.CATEGORICAL.value,
            ColumnType.CATEGORICAL.value,
        ]

    # Display current column list
    cols_to_remove: list[int] = []
    for i, ct_val in enumerate(st.session_state["syn_col_types"]):
        c1, c2, c3 = st.columns([3, 4, 2])
        with c1:
            st.write(f"**Column {i}**")
        with c2:
            default_idx = _COLUMN_TYPE_OPTIONS.index(ct_val)
            new_val = st.selectbox(
                f"Type for column {i}",
                _COLUMN_TYPE_OPTIONS,
                index=default_idx,
                key=f"syn_ct_{i}",
                label_visibility="collapsed",
            )
            st.session_state["syn_col_types"][i] = new_val
        with c3:
            if st.button("Remove", key=f"syn_rm_{i}"):
                cols_to_remove.append(i)

    # Process removals
    if cols_to_remove:
        for idx in sorted(cols_to_remove, reverse=True):
            st.session_state["syn_col_types"].pop(idx)
        st.rerun()

    # Add column buttons
    add_c1, add_c2 = st.columns(2)
    with add_c1:
        if st.button("+ Add Column", key="syn_add_col"):
            st.session_state["syn_col_types"].append(ColumnType.CONTINUOUS.value)
            st.rerun()

    # Generate button
    st.divider()

    if len(st.session_state["syn_col_types"]) == 0:
        st.warning("Add at least one column before generating data.")
    else:
        if st.button("Generate", key="generate_synthetic", type="primary"):
            col_types = [ColumnType(v) for v in st.session_state["syn_col_types"]]
            key = jax.random.key(int(seed))

            with st.spinner("Generating synthetic data..."):
                result = generate_synthetic(
                    key,
                    n_rows=n_rows,
                    column_types=col_types,
                    n_views=n_views,
                    n_clusters=n_clusters,
                    cluster_separation=float(cluster_separation),
                )

            data = result["data"]
            n_cols = data.shape[1]
            column_names = [f"col_{j}" for j in range(n_cols)]

            st.session_state["data"] = data
            st.session_state["column_names"] = column_names
            st.session_state["column_types"] = col_types
            st.session_state["packed_state"] = None
            st.session_state["sweep_history"] = []
            st.session_state["inference_done"] = False

            st.success(
                f"Generated {data.shape[0]} rows x {data.shape[1]} columns "
                f"({n_views} views, {n_clusters} clusters)."
            )

            # Show preview
            preview_df = pd.DataFrame(np.asarray(data), columns=column_names)
            st.dataframe(preview_df.head(50), use_container_width=True)

            # Show ground truth
            with st.expander("Ground truth"):
                st.write(f"True column assignments: {result['true_column_assignments']}")
                st.write(f"True row assignments shape: {result['true_row_assignments'].shape}")
