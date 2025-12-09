import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from dashboard_utils import (
    stat_display,
    get_available_campaigns,
    load_dataset_with_metadata,
    get_unique_values,
)
###############


def weights_dashboard_app():
    plot_display = {"P(W)": "P_w", "P(λ)": "P_sv", "SVD": "SVD"}

    # Load data
    available_datasets = get_available_campaigns("ana-")
    if not available_datasets:
        st.error("No datasets found.")
        st.stop()

    # Dataset dropdown
    campaign_name = st.sidebar.selectbox("Campaign", available_datasets, index=0)

    df_full, metadata = load_dataset_with_metadata(
        ds_name="weight_study", campaign=campaign_name, hf_version="ana-003"
    )

    model_names = get_unique_values(df_full, "model")
    model_selected = st.sidebar.selectbox("Model", model_names)

    weight_types = get_unique_values(df_full, "weight_type")
    weight_selected = st.sidebar.selectbox("Weight Type", weight_types)

    df = df_full.query(
        f"model == '{model_selected}' and weight_type == '{weight_selected}'"
    )
    # Get architecture dimensions
    d_model = metadata["d_model"]
    n_layers = df["layer"].max() + 1
    n_heads = df["head"].max() + 1

    ########
    st.title("Weight Dashboard")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Info")
    st.sidebar.markdown(f"Model Dimension: {d_model}")
    st.sidebar.markdown(f"Heads: {n_heads}")
    st.sidebar.markdown(f"Layers: {n_layers}")

    ########################################################################
    # Section 1: Single dead distributions
    ########################################################################

    st.header("Weight Distributions for a Single Attention Head")

    col1, col2 = st.columns(2)
    layer = col1.slider("Layer", 0, n_layers - 1, 0)
    head = col2.slider("Head", 0, n_heads - 1, 0)

    entry = df.query(f"layer == {layer} and head == {head}")

    available_plots = ["P(W)"]
    if weight_selected == "W_QK":
        available_plots.extend(["P(λ)", "SVD"])
    plot_type = st.selectbox("Plot", available_plots)

    plot_type_name = plot_display[plot_type]
    h = entry[plot_type_name].iloc[0]

    use_log_1 = st.checkbox("Log scale", key="log_1")
    show_fit = st.checkbox("Show Gaussian fit", key="fit_1")

    xtitle, ytitle = "Weight", "Probability"
    bins = metadata["w_bins"]

    if plot_type_name == "P_sv":
        bins = metadata["sv_bins"]
        xtitle, ytitle = "Singular Value", "Probability"
    elif plot_type_name == "SVD":
        nsvs = len(h)
        bins = np.linspace(-0.5, nsvs - 0.5, nsvs + 1)
        xtitle, ytitle = "Index", "Singular Value"

    h_centers = [0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]
    fig = go.Figure()
    dist_centers = h_centers
    if plot_type == "SVD":
        dist_centers = np.arange(len(h))

    y_min = (h[h != 0].min()) * 0.5
    y_vals = h
    fig.add_trace(go.Bar(x=dist_centers, y=y_vals, name=plot_type))
    if use_log_1:
        fig.update_yaxes(type="log")
    fig.update_layout(xaxis_title=xtitle, yaxis_title=ytitle)

    if show_fit and plot_type == "P(W)":
        mu = entry["fit_mu"].iloc[0]
        sigma = entry["fit_sigma"].iloc[0]
        st.write(f"μ = {mu:.4f}, σ ={sigma:.4f}")
        # Gaussian curve
        from scipy.stats import norm

        fit_curve = norm.pdf(h_centers, mu, sigma)
        # Scale to match histogram
        fit_curve *= np.sum(h) * (h_centers[1] - h_centers[0])
        fig.add_trace(
            go.Scatter(
                x=h_centers,
                y=np.maximum(fit_curve, y_min),
                mode="lines",
                name="Fit",
                line=dict(color="red", width=2),
            )
        )

    st.plotly_chart(fig, width="content")

    # Statistics
    st.subheader("Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Standard Deviation", f"{entry['std'].iloc[0]:.4f}")
    col2.metric("$\sigma$ (Gaussian fit)", f"{entry['fit_sigma'].iloc[0]:.4f}")
    col3.metric("Entropy", f"{entry['entropy'].iloc[0]:.4f}")
    col4.metric("Max", f"{entry['max'].iloc[0]:.4f}")
    col5.metric("Min", f"{entry['min'].iloc[0]:.4f}")

    ########################################################################
    # Section 2: Across layers/heads
    ########################################################################
    st.header("Statistics Across Architecture")
    df_sorted = df.sort_values(["layer", "head"])

    # Multi-select for statistics
    selected_stats = st.multiselect(
        "Select Statistics",
        options=list(stat_display.keys()),
        default=[list(stat_display.keys())[0]],
    )

    if not selected_stats:
        st.warning("Please select at least one statistic")
    else:
        # Option for separate axes
        use_separate_axes = st.checkbox(
            "Use separate Y-axes for each statistic",
            value=False,
            help="Each statistic gets its own Y-axis with matching colors",
        )

        # Prepare data

        # Color palette
        colors = px.colors.qualitative.Plotly[: len(selected_stats)]

        if use_separate_axes:
            # Create figure with secondary y-axes
            from plotly.subplots import make_subplots

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            for idx, stat_display_name in enumerate(selected_stats):
                stat_name = stat_display[stat_display_name]
                stats = df_sorted[stat_name].values.flatten()

                # Determine which axis to use
                use_secondary = idx > 0

                fig.add_trace(
                    go.Scatter(
                        y=stats,
                        mode="lines",
                        name=stat_display_name,
                        line=dict(color=colors[idx]),
                        yaxis="y2" if use_secondary else "y",
                    ),
                    secondary_y=use_secondary,
                )

                # Style the corresponding y-axis
                axis_config = dict(
                    title=stat_display_name,
                    title_font=dict(color=colors[idx]),
                    tickfont=dict(color=colors[idx]),
                )
                if use_secondary:
                    fig.update_yaxes(axis_config, secondary_y=True)
                else:
                    fig.update_yaxes(axis_config, secondary_y=False)
        else:
            # Shared Y-axis
            fig = go.Figure()

            for idx, stat_display_name in enumerate(selected_stats):
                stat_name = stat_display[stat_display_name]
                stats = df_sorted[stat_name].values.flatten()

                fig.add_trace(
                    go.Scatter(
                        y=stats,
                        mode="lines",
                        name=stat_display_name,
                        line=dict(color=colors[idx]),
                    )
                )

        # Common X-axis configuration
        fig.update_xaxes(
            tickmode="array",
            tickvals=[i * n_heads for i in range(n_layers)],
            ticktext=[str(i) for i in range(n_layers)],
            title="Layer",
        )

        # Layer separators
        for xpos in range(n_heads, n_layers * n_heads, n_heads):
            fig.add_shape(
                type="line",
                x0=xpos,
                x1=xpos,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="lightgray", width=1, dash="dot"),
            )

        fig.update_layout(
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig, width="content", key="section2_plot")

    ########################################################################
    # Section 3: 2D Probability Distribution Stack
    ########################################################################

    st.header("Stacked Probability Distributions")

    plot_type_2d = st.selectbox("Distribution", available_plots, key="2d_plot")
    plot_type_name_2d = plot_display[plot_type_2d]
    bins = metadata["w_bins"]
    xtitle_2d, ytitle_2d = "Weight", "Layer"

    if plot_type_name_2d == "P_sv":
        bins = metadata["sv_bins"]
        xtitle_2d, ytitle_2d = "Singular Value", "Layer"
    elif plot_type_name_2d == "SVD":
        nsvs = len(h)
        bins = np.linspace(-0.5, nsvs - 0.5, nsvs + 1)
        xtitle_2d, ytitle_2d = "Index", "Layer"
    h_centers = [0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]

    # Stack histograms into 2D array
    prob_stack = np.array([row[plot_type_name_2d] for _, row in df_sorted.iterrows()])

    use_log_2d = st.checkbox("Log scale", key="log_2d")

    prob_stack = np.maximum(prob_stack, y_min)
    fig = go.Figure(
        data=go.Heatmap(
            z=np.log10(prob_stack) if use_log_2d else prob_stack,
            x=h_centers,
            y=np.arange(n_layers * n_heads),
            colorscale="Viridis",
        )
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=[i * n_heads for i in range(n_layers)],
        ticktext=[str(i) for i in range(n_layers)],
        title="Layer",
    )
    fig.update_layout(xaxis_title=xtitle_2d, yaxis_title=ytitle_2d, height=600)
    st.plotly_chart(fig, width="content")

    ########################################################################
    # Section 4: Per-Layer Head Grid
    ########################################################################

    st.header("Distribution Grid by Layer")

    col1, col2 = st.columns(2)
    layer_grid = col1.selectbox("Layer", range(n_layers), key="grid_layer")
    plot_type_grid = col2.selectbox("Distribution", available_plots, key="grid_plot")
    plot_type_grid_name = plot_display[plot_type_grid]
    show_fit_grid = st.checkbox("Show Gaussian fits", key="fit_grid")
    use_log_grid = st.checkbox("Log scale", key="log_grid")
    # Create subplot grid

    n_cols = 4
    n_rows = int(np.ceil(n_heads / n_cols))
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=[f"Head {i}" for i in range(n_heads)]
    )
    for head in range(n_heads):
        df_h = df.query(f"layer == {layer_grid} and head == {head}")
        h = df_h[plot_type_grid_name].iloc[0]

        row = head // n_cols + 1
        col = head % n_cols + 1
        dist_centers = h_centers
        if plot_type_grid == "SVD":
            dist_centers = np.arange(len(h))
        if show_fit_grid and plot_type_grid == "P(W)":
            from scipy.stats import norm

            mu = df_h["fit_mu"].iloc[0]
            sigma = df_h["fit_sigma"].iloc[0]

            fit_curve = norm.pdf(h_centers, mu, sigma)
            fit_curve *= np.sum(h) * (h_centers[1] - h_centers[0])

            fig.add_trace(
                go.Scatter(
                    x=dist_centers,
                    y=np.maximum(fit_curve, y_min),
                    mode="lines",
                    line=dict(color="red", width=1),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        y_min = (h[h != 0].min()) * 0.5
        y_vals = h
        fig.add_trace(
            go.Bar(x=dist_centers, y=y_vals, name=f"Head {head}", showlegend=False),
            row=row,
            col=col,
        )
        if use_log_grid:
            fig.update_yaxes(type="log")

    fig.update_layout(
        height=200 * n_rows, title=f"Layer {layer_grid} - {plot_type_grid}"
    )

    st.plotly_chart(fig, width="stretch", key="section1_sv")


def render():
    weights_dashboard_app()


if __name__ == "__main__":
    weights_dashboard_app()
