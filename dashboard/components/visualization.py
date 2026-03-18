"""Reusable Plotly chart functions for the JAX-CrossCat dashboard."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_column_partition(
    column_assignments: np.ndarray,
    column_names: list[str],
    n_views: int,
) -> go.Figure:
    """Grouped bar chart showing which columns belong to which view.

    Args:
        column_assignments: Integer array of shape ``(n_cols,)`` mapping each
            column to a view index.
        column_names: Human-readable column names.
        n_views: Number of active views.

    Returns:
        A Plotly ``Figure``.
    """
    # Build a matrix: rows=views, cols=columns, value=1 if column in view
    n_cols = len(column_names)
    matrix = np.zeros((n_views, n_cols), dtype=int)
    for j in range(n_cols):
        v = int(column_assignments[j])
        if v < n_views:
            matrix[v, j] = 1

    fig = go.Figure()
    for v in range(n_views):
        cols_in_view = [column_names[j] for j in range(n_cols) if matrix[v, j] == 1]
        fig.add_trace(
            go.Bar(
                x=cols_in_view,
                y=[1] * len(cols_in_view),
                name=f"View {v}",
            )
        )

    fig.update_layout(
        title="Column Partition (columns grouped by view)",
        xaxis_title="Column",
        yaxis_title="",
        yaxis=dict(showticklabels=False),
        barmode="stack",
        showlegend=True,
        height=400,
    )
    return fig


def plot_convergence(sweep_history: list[dict]) -> go.Figure:
    """Dual-axis line chart of log_joint and n_views over sweeps.

    Args:
        sweep_history: List of dicts, each containing ``log_joint`` and
            ``n_views`` keys.

    Returns:
        A Plotly ``Figure``.
    """
    sweeps = list(range(1, len(sweep_history) + 1))
    log_joints = [h["log_joint"] for h in sweep_history]
    n_views_list = [h["n_views"] for h in sweep_history]

    fig = go.Figure()

    # log_joint on primary y-axis
    fig.add_trace(
        go.Scatter(
            x=sweeps,
            y=log_joints,
            mode="lines+markers",
            name="log_joint",
            line=dict(color="#1f77b4"),
        )
    )

    # n_views on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=sweeps,
            y=n_views_list,
            mode="lines+markers",
            name="n_views",
            yaxis="y2",
            line=dict(color="#ff7f0e"),
        )
    )

    fig.update_layout(
        title="Convergence",
        xaxis_title="Sweep",
        yaxis=dict(title="log p(state, data)", titlefont=dict(color="#1f77b4")),
        yaxis2=dict(
            title="Number of views",
            titlefont=dict(color="#ff7f0e"),
            overlaying="y",
            side="right",
        ),
        height=400,
    )
    return fig


def plot_mi_matrix(
    mi_values: np.ndarray,
    column_names: list[str],
) -> go.Figure:
    """Heatmap of pairwise mutual information.

    Args:
        mi_values: Square matrix of shape ``(n_cols, n_cols)``.
        column_names: Column labels.

    Returns:
        A Plotly ``Figure``.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=mi_values,
            x=column_names,
            y=column_names,
            colorscale="Viridis",
            colorbar=dict(title="MI"),
        )
    )
    fig.update_layout(
        title="Mutual Information Matrix",
        xaxis_title="Column",
        yaxis_title="Column",
        height=500,
        width=600,
    )
    return fig


def plot_anomaly_distribution(scores: np.ndarray) -> go.Figure:
    """Histogram of row-level anomaly scores.

    Args:
        scores: 1-D array of anomaly scores in ``[0, 1]``.

    Returns:
        A Plotly ``Figure``.
    """
    fig = go.Figure(
        data=go.Histogram(
            x=scores,
            nbinsx=30,
            marker_color="#d62728",
            opacity=0.8,
        )
    )
    fig.update_layout(
        title="Anomaly Score Distribution",
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        height=400,
    )
    return fig


def plot_row_clustering(
    row_assignments: np.ndarray,
    view_id: int,
) -> go.Figure:
    """Bar chart of cluster sizes for a given view.

    Args:
        row_assignments: Integer array of shape ``(n_rows,)`` with cluster
            assignments for this view.
        view_id: View index (used in chart title).

    Returns:
        A Plotly ``Figure``.
    """
    unique, counts = np.unique(row_assignments, return_counts=True)
    labels = [f"Cluster {int(u)}" for u in unique]

    fig = go.Figure(
        data=go.Bar(
            x=labels,
            y=counts,
            marker_color="#2ca02c",
        )
    )
    fig.update_layout(
        title=f"Row Clustering -- View {view_id}",
        xaxis_title="Cluster",
        yaxis_title="Number of rows",
        height=400,
    )
    return fig


def plot_similarity_matrix(
    sim_matrix: np.ndarray,
    row_labels: list[str],
) -> go.Figure:
    """Heatmap of pairwise row similarity.

    Args:
        sim_matrix: Square matrix of shape ``(n, n)`` with similarity values
            in ``[0, 1]``.
        row_labels: Labels for each row.

    Returns:
        A Plotly ``Figure``.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=sim_matrix,
            x=row_labels,
            y=row_labels,
            colorscale="Blues",
            colorbar=dict(title="Similarity"),
            zmin=0,
            zmax=1,
        )
    )
    fig.update_layout(
        title="Row Similarity Matrix",
        xaxis_title="Row",
        yaxis_title="Row",
        height=500,
        width=600,
    )
    return fig
