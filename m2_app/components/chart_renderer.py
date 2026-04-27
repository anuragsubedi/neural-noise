"""
neural-noise Milestone 2 — Chart Rendering Helper

Renders figures using either st.plotly_chart or st.pyplot depending
on whether the figure is a Plotly Figure or Matplotlib Figure.
"""

import streamlit as st


def render_chart(fig, use_container_width: bool = True):
    """
    Render a chart figure in Streamlit, auto-detecting Plotly vs Matplotlib.

    Args:
        fig: Either a plotly.graph_objects.Figure or matplotlib.figure.Figure
        use_container_width: Whether to stretch to container width
    """
    width = "stretch" if use_container_width else "content"
    if fig is None:
        st.caption("No chart data available.")
        return

    # Check if it's a Plotly figure
    try:
        import plotly.graph_objects as go
        if isinstance(fig, go.Figure):
            st.plotly_chart(fig, width=width)
            return
    except ImportError:
        pass

    # Check if it's a Matplotlib figure
    try:
        import matplotlib.figure
        if isinstance(fig, matplotlib.figure.Figure):
            st.pyplot(fig, use_container_width=use_container_width)  # pyplot still uses old API
            import matplotlib.pyplot as plt
            plt.close(fig)  # Free memory
            return
    except ImportError:
        pass

    # Fallback
    st.warning("Unable to render chart: unknown figure type.")
