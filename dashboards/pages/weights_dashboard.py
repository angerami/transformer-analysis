import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy import stats as scipy_stats
from dashboard_utils import (
    stat_display,
    get_available_campaigns,
    load_dataset_with_metadata,
    get_unique_values,
    is_HF_environment,
    model_size_from_name,
    create_snapshot_button,
)
###############


def weights_dashboard_app():
    plot_display = {"P(W)": "P_w", "P(λ)": "P_sv", "SVD": "SVD"}
    merge_key = 'merged'
    
    # # Load data
    if is_HF_environment():
        campaign_name = "ana-004"
    else:
        available_datasets = get_available_campaigns("ana-")
        if not available_datasets:
            st.error("No datasets found.")
            st.stop()
        # Dataset dropdown
        campaign_name = st.sidebar.selectbox("Campaign", available_datasets, index=0)


    df_full, metadata = load_dataset_with_metadata(
        ds_name="weight_study", campaign=campaign_name, hf_version="ana-003"
    )

    # Sort models: first by model family (prefix before first '-'), then by size within family
    model_names = sorted(
        get_unique_values(df_full, "model"),
        key=lambda x: (x.split('-')[0], model_size_from_name(x))
    )
    model_selected = st.sidebar.selectbox("Model", model_names)
    if merge_key in metadata and model_selected in metadata[merge_key]:
        metadata = metadata[merge_key][model_selected]

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
    # Section 1: Single Head Distributions
    ########################################################################

    st.header("Section 1: Weight Distributions for a Single Attention Head")

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

    # Set x-axis range for SVD plots
    xaxis_range = None
    if plot_type == "SVD":
        max_sv_index = d_model // n_heads - 1
        xaxis_range = [0, max_sv_index]

    fig.update_layout(xaxis_title=xtitle, yaxis_title=ytitle, xaxis_range=xaxis_range)

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

    # Snapshot button for Section 1
    create_snapshot_button(
        fig=fig,
        metadata={
            "section": "section_1_single_head_distribution",
            "campaign": campaign_name,
            "model": model_selected,
            "weight_type": weight_selected,
            "layer": int(layer),
            "head": int(head),
            "plot_type": plot_type,
            "use_log_scale": use_log_1,
            "show_fit": show_fit,
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "d_model": d_model,
        },
        section_name="weights_section_1",
        key="snapshot_weights_s1"
    )

    # Statistics
    st.subheader("Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Standard Deviation", f"{entry['std'].iloc[0]:.4f}")
    col2.metric("$\sigma$ (Gaussian fit)", f"{entry['fit_sigma'].iloc[0]:.4f}")
    col3.metric("Entropy", f"{entry['entropy'].iloc[0]:.4f}")
    col4.metric("Max", f"{entry['max'].iloc[0]:.4f}")
    col5.metric("Min", f"{entry['min'].iloc[0]:.4f}")

    ########################################################################
    # Section 2: Statistics Across Architecture
    ########################################################################
    st.header("Section 2: Statistics Across Architecture")
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

        # Option to show layer averages
        show_layer_avg = st.checkbox(
            "Show layer averages",
            value=False,
            help="Display the mean value across all heads in each layer as horizontal red lines",
        )

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

                # Add layer averages if requested
                if show_layer_avg:
                    # Compute average value for each x-position (head within layer)
                    avg_values = []
                    for layer_idx in range(n_layers):
                        df_layer = df.query(f"layer == {layer_idx}")
                        layer_avg = df_layer[stat_name].mean()
                        # Repeat the average value for each head in this layer
                        avg_values.extend([layer_avg] * n_heads)

                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(avg_values))),
                            y=avg_values,
                            mode="lines",
                            line=dict(color="red", width=3, dash="dash"),
                            name="Layer average" if idx == 0 else None,
                            showlegend=(idx == 0),
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

                # Add layer averages if requested
                if show_layer_avg:
                    # Compute average value for each x-position (head within layer)
                    avg_values = []
                    for layer_idx in range(n_layers):
                        df_layer = df.query(f"layer == {layer_idx}")
                        layer_avg = df_layer[stat_name].mean()
                        # Repeat the average value for each head in this layer
                        avg_values.extend([layer_avg] * n_heads)

                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(avg_values))),
                            y=avg_values,
                            mode="lines",
                            line=dict(color="red", width=3, dash="dash"),
                            name="Layer average" if idx == 0 else None,
                            showlegend=(idx == 0),
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

        # Snapshot button for Section 2
        create_snapshot_button(
            fig=fig,
            metadata={
                "section": "section_2_statistics_across_architecture",
                "campaign": campaign_name,
                "model": model_selected,
                "weight_type": weight_selected,
                "selected_statistics": selected_stats,
                "use_separate_axes": use_separate_axes,
                "show_layer_avg": show_layer_avg,
                "n_layers": int(n_layers),
                "n_heads": int(n_heads),
                "d_model": d_model,
            },
            section_name="weights_section_2",
            key="snapshot_weights_s2"
        )

    ########################################################################
    # Section 3: Stacked Probability Distributions
    ########################################################################

    st.header("Section 3: Stacked Probability Distributions")

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
    if plot_type_name_2d == "SVD":
        max_sv_index = d_model // n_heads - 1
        fig.update_xaxes(range=[0, max_sv_index])
    fig.update_layout(xaxis_title=xtitle_2d, yaxis_title=ytitle_2d, height=600)
    st.plotly_chart(fig, width="content")

    # Snapshot button for Section 3
    create_snapshot_button(
        fig=fig,
        metadata={
            "section": "section_3_stacked_probability_distributions",
            "campaign": campaign_name,
            "model": model_selected,
            "weight_type": weight_selected,
            "plot_type": plot_type_2d,
            "plot_type_column": plot_type_name_2d,
            "use_log_scale": use_log_2d,
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "d_model": d_model,
        },
        section_name="weights_section_3",
        key="snapshot_weights_s3"
    )

    ########################################################################
    # Section 4: Distribution Grid by Layer
    ########################################################################

    st.header("Section 4: Distribution Grid by Layer")

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

    # Set x-axis range for SVD plots
    if plot_type_grid == "SVD":
        max_sv_index = d_model // n_heads - 1
        fig.update_xaxes(range=[0, max_sv_index])

    fig.update_layout(
        height=200 * n_rows, title=f"Layer {layer_grid} - {plot_type_grid}"
    )

    st.plotly_chart(fig, width="stretch", key="section1_sv")

    # Snapshot button for Section 4
    create_snapshot_button(
        fig=fig,
        metadata={
            "section": "section_4_distribution_grid_by_layer",
            "campaign": campaign_name,
            "model": model_selected,
            "weight_type": weight_selected,
            "layer": int(layer_grid),
            "plot_type": plot_type_grid,
            "plot_type_column": plot_type_grid_name,
            "show_fit": show_fit_grid,
            "use_log_scale": use_log_grid,
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "d_model": d_model,
        },
        section_name="weights_section_4",
        key="snapshot_weights_s4"
    )

    ########################################################################
    # Section 5: Scatter Plot - Compare Two Statistics
    ########################################################################

    st.header("Section 5: Compare Two Statistics")

    col1, col2 = st.columns(2)
    x_stat_display = col1.selectbox("X-axis Statistic", options=list(stat_display.keys()), key="scatter_x")
    y_stat_display = col2.selectbox("Y-axis Statistic", options=list(stat_display.keys()), index=1, key="scatter_y")

    x_stat = stat_display[x_stat_display]
    y_stat = stat_display[y_stat_display]

    # Generate colors for each layer
    # Reserve first color for layer 0, cycle through remaining colors for other layers
    all_colors = px.colors.qualitative.Plotly
    layer_0_color = all_colors[0]  # Reserved color for layer 0
    other_colors = all_colors[1:]  # Colors for layers 1+

    def get_layer_color(layer_idx):
        if layer_idx == 0:
            return layer_0_color
        else:
            # Cycle through remaining colors for layers 1+
            return other_colors[(layer_idx - 1) % len(other_colors)]

    # Create scatter plot
    fig = go.Figure()

    for layer_idx in range(n_layers):
        df_layer = df.query(f"layer == {layer_idx}")
        x_vals = df_layer[x_stat].values
        y_vals = df_layer[y_stat].values

        # Create hover text showing layer and head info
        hover_text = [f"Layer {layer_idx}, Head {h}<br>{x_stat_display}: {x:.4f}<br>{y_stat_display}: {y:.4f}"
                      for h, x, y in zip(df_layer["head"].values, x_vals, y_vals)]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                name=f"Layer {layer_idx}",
                marker=dict(
                    color=get_layer_color(layer_idx),
                    size=8,
                    opacity=0.7
                ),
                hovertext=hover_text,
                hoverinfo="text"
            )
        )

    fig.update_layout(
        xaxis_title=x_stat_display,
        yaxis_title=y_stat_display,
        hovermode="closest",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        height=600
    )

    st.plotly_chart(fig, width="content", key="scatter_plot")

    # Snapshot button for Section 5
    create_snapshot_button(
        fig=fig,
        metadata={
            "section": "section_5_compare_two_statistics",
            "campaign": campaign_name,
            "model": model_selected,
            "weight_type": weight_selected,
            "x_statistic": x_stat_display,
            "y_statistic": y_stat_display,
            "x_statistic_column": x_stat,
            "y_statistic_column": y_stat,
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "d_model": d_model,
        },
        section_name="weights_section_5",
        key="snapshot_weights_s5"
    )

    ########################################################################
    # Section 6: Singular Values Across Architecture (W_QK only)
    ########################################################################

    if weight_selected == "W_QK":
        st.header("Section 6: Singular Values Across Architecture")

        # Get the number of singular values from the first entry
        first_svd = df_sorted["SVD"].iloc[0]
        n_svs = len(first_svd)

        # Build options: pre-computed stats + derived SV stats + individual SVs
        sv_options = {}

        # Add pre-computed statistics
        for display_name, col_name in stat_display.items():
            sv_options[display_name] = ("stat", col_name)

        # Add derived SV statistics
        sv_options["SV Mean"] = ("derived", "mean")
        sv_options["SV Variance"] = ("derived", "variance")
        sv_options["SV Skewness"] = ("derived", "skewness")
        sv_options["SV Kurtosis"] = ("derived", "kurtosis")
        sv_options["Σσ"] = ("derived", "sum")
        sv_options["Σσ²"] = ("derived", "sum_squares")
        sv_options["Participation Ratio"] = ("derived", "participation_ratio")
        sv_options["Normalized Participation Ratio"] = ("derived", "normalized_participation_ratio")
        sv_options["Spectral Entropy"] = ("derived", "spectral_entropy")
        sv_options["Condition Number"] = ("derived", "condition_number")
        sv_options["Stable Rank"] = ("derived", "stable_rank")

        # Add individual singular values (these appear last)
        for k in range(n_svs):
            sv_options[f"SV[{k}]"] = ("sv", k)

        # Multi-select for what to plot
        selected_sv_stats = st.multiselect(
            "Select Statistics/Singular Values to Plot",
            options=list(sv_options.keys()),
            default=[list(stat_display.keys())[0]],
            key="sv_stats_select"
        )

        if not selected_sv_stats:
            st.warning("Please select at least one statistic or singular value")
        else:
            # Option for separate axes
            use_separate_axes_sv = st.checkbox(
                "Use separate Y-axes for each statistic",
                value=False,
                help="Each statistic gets its own Y-axis with matching colors",
                key="separate_axes_sv"
            )

            # Option to show layer averages
            show_layer_avg_sv = st.checkbox(
                "Show layer averages",
                value=False,
                help="Display the mean value across all heads in each layer as horizontal red lines",
                key="layer_avg_sv"
            )

            # Color palette
            colors_sv = px.colors.qualitative.Plotly[: len(selected_sv_stats)]

            # Helper function to compute derived SV statistic
            def compute_derived_sv_stat(svd_array, stat_type):
                """Compute derived statistics from singular values."""
                sv = np.array(svd_array)
                d_head = d_model // n_heads  # Number of non-zero SVs expected

                if stat_type == "mean":
                    return np.mean(sv)
                elif stat_type == "variance":
                    return np.var(sv)
                elif stat_type == "skewness":
                    return scipy_stats.skew(sv)
                elif stat_type == "kurtosis":
                    return scipy_stats.kurtosis(sv)
                elif stat_type == "sum":
                    return np.sum(sv)
                elif stat_type == "sum_squares":
                    return np.sum(sv**2)
                elif stat_type == "participation_ratio":
                    sum_sv = np.sum(sv)
                    sum_sv2 = np.sum(sv**2)
                    return (sum_sv**2) / sum_sv2 if sum_sv2 > 0 else 0
                elif stat_type == "normalized_participation_ratio":
                    sum_sv = np.sum(sv)
                    sum_sv2 = np.sum(sv**2)
                    participation_ratio = (sum_sv**2) / sum_sv2 if sum_sv2 > 0 else 0
                    return participation_ratio / d_head if d_head > 0 else 0
                elif stat_type == "spectral_entropy":
                    sv2 = sv**2
                    sum_sv2 = np.sum(sv2)
                    if sum_sv2 > 0:
                        p = sv2 / sum_sv2
                        p = p[p > 0]  # Remove zeros to avoid log(0)
                        return -np.sum(p * np.log(p))
                    return 0
                elif stat_type == "condition_number":
                    # Use only the d_head largest SVs (low-rank structure)
                    sv_nonzero = sv[:d_head]
                    if len(sv_nonzero) > 0 and sv_nonzero[-1] > 0:
                        return sv_nonzero[0] / sv_nonzero[-1]
                    return 0
                elif stat_type == "stable_rank":
                    sum_sv2 = np.sum(sv**2)
                    max_sv2 = sv[0]**2
                    return sum_sv2 / max_sv2 if max_sv2 > 0 else 0

                return 0

            # Helper function to extract values
            def get_values(option_name):
                option_type, option_data = sv_options[option_name]
                values = []

                for _, row in df_sorted.iterrows():
                    if option_type == "stat":
                        values.append(row[option_data])
                    elif option_type == "sv":
                        svd_array = row["SVD"]
                        values.append(svd_array[option_data])
                    elif option_type == "derived":
                        svd_array = row["SVD"]
                        values.append(compute_derived_sv_stat(svd_array, option_data))

                return np.array(values)

            if use_separate_axes_sv:
                # Create figure with secondary y-axes
                from plotly.subplots import make_subplots

                fig = make_subplots(specs=[[{"secondary_y": True}]])

                for idx, option_name in enumerate(selected_sv_stats):
                    stats = get_values(option_name)

                    # Determine which axis to use
                    use_secondary = idx > 0

                    fig.add_trace(
                        go.Scatter(
                            y=stats,
                            mode="lines",
                            name=option_name,
                            line=dict(color=colors_sv[idx]),
                            yaxis="y2" if use_secondary else "y",
                        ),
                        secondary_y=use_secondary,
                    )

                    # Add layer averages if requested
                    if show_layer_avg_sv:
                        # Compute average value for each x-position (head within layer)
                        avg_values = []
                        option_type, option_data = sv_options[option_name]
                        for layer_idx in range(n_layers):
                            df_layer = df.query(f"layer == {layer_idx}")

                            # Compute layer average for this option
                            layer_values = []
                            for _, row in df_layer.iterrows():
                                if option_type == "stat":
                                    layer_values.append(row[option_data])
                                elif option_type == "sv":
                                    layer_values.append(row["SVD"][option_data])
                                elif option_type == "derived":
                                    layer_values.append(compute_derived_sv_stat(row["SVD"], option_data))

                            layer_avg = np.mean(layer_values)
                            # Repeat the average value for each head in this layer
                            avg_values.extend([layer_avg] * n_heads)

                        fig.add_trace(
                            go.Scatter(
                                x=list(range(len(avg_values))),
                                y=avg_values,
                                mode="lines",
                                line=dict(color="red", width=3, dash="dash"),
                                name="Layer average" if idx == 0 else None,
                                showlegend=(idx == 0),
                                yaxis="y2" if use_secondary else "y",
                            ),
                            secondary_y=use_secondary,
                        )

                    # Style the corresponding y-axis
                    axis_config = dict(
                        title=option_name,
                        title_font=dict(color=colors_sv[idx]),
                        tickfont=dict(color=colors_sv[idx]),
                    )
                    if use_secondary:
                        fig.update_yaxes(axis_config, secondary_y=True)
                    else:
                        fig.update_yaxes(axis_config, secondary_y=False)
            else:
                # Shared Y-axis
                fig = go.Figure()

                for idx, option_name in enumerate(selected_sv_stats):
                    stats = get_values(option_name)

                    fig.add_trace(
                        go.Scatter(
                            y=stats,
                            mode="lines",
                            name=option_name,
                            line=dict(color=colors_sv[idx]),
                        )
                    )

                    # Add layer averages if requested
                    if show_layer_avg_sv:
                        # Compute average value for each x-position (head within layer)
                        avg_values = []
                        option_type, option_data = sv_options[option_name]
                        for layer_idx in range(n_layers):
                            df_layer = df.query(f"layer == {layer_idx}")

                            # Compute layer average for this option
                            layer_values = []
                            for _, row in df_layer.iterrows():
                                if option_type == "stat":
                                    layer_values.append(row[option_data])
                                elif option_type == "sv":
                                    layer_values.append(row["SVD"][option_data])
                                elif option_type == "derived":
                                    layer_values.append(compute_derived_sv_stat(row["SVD"], option_data))

                            layer_avg = np.mean(layer_values)
                            # Repeat the average value for each head in this layer
                            avg_values.extend([layer_avg] * n_heads)

                        fig.add_trace(
                            go.Scatter(
                                x=list(range(len(avg_values))),
                                y=avg_values,
                                mode="lines",
                                line=dict(color="red", width=3, dash="dash"),
                                name="Layer average" if idx == 0 else None,
                                showlegend=(idx == 0),
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

            st.plotly_chart(fig, width="content", key="section6_plot")

            # Snapshot button for Section 6
            create_snapshot_button(
                fig=fig,
                metadata={
                    "section": "section_6_singular_values_across_architecture",
                    "campaign": campaign_name,
                    "model": model_selected,
                    "weight_type": weight_selected,
                    "selected_sv_stats": selected_sv_stats,
                    "use_separate_axes": use_separate_axes_sv,
                    "show_layer_avg": show_layer_avg_sv,
                    "n_layers": int(n_layers),
                    "n_heads": int(n_heads),
                    "d_model": d_model,
                },
                section_name="weights_section_6",
                key="snapshot_weights_s6"
            )

    ########################################################################
    # Section 7: Scatter Plot with Singular Values (W_QK only)
    ########################################################################

    if weight_selected == "W_QK":
        st.header("Section 7: Compare Statistics/Singular Values")

        # Get the number of singular values from the first entry
        first_svd = df_sorted["SVD"].iloc[0]
        n_svs = len(first_svd)

        # Build options: pre-computed stats + derived SV stats + individual SVs
        sv_scatter_options = {}

        # Add pre-computed statistics
        for display_name, col_name in stat_display.items():
            sv_scatter_options[display_name] = ("stat", col_name)

        # Add derived SV statistics
        sv_scatter_options["SV Mean"] = ("derived", "mean")
        sv_scatter_options["SV Variance"] = ("derived", "variance")
        sv_scatter_options["SV Skewness"] = ("derived", "skewness")
        sv_scatter_options["SV Kurtosis"] = ("derived", "kurtosis")
        sv_scatter_options["Σσ"] = ("derived", "sum")
        sv_scatter_options["Σσ²"] = ("derived", "sum_squares")
        sv_scatter_options["Participation Ratio"] = ("derived", "participation_ratio")
        sv_scatter_options["Normalized Participation Ratio"] = ("derived", "normalized_participation_ratio")
        sv_scatter_options["Spectral Entropy"] = ("derived", "spectral_entropy")
        sv_scatter_options["Condition Number"] = ("derived", "condition_number")
        sv_scatter_options["Stable Rank"] = ("derived", "stable_rank")

        # Add individual singular values (these appear last)
        for k in range(n_svs):
            sv_scatter_options[f"SV[{k}]"] = ("sv", k)

        col1, col2 = st.columns(2)
        x_option = col1.selectbox(
            "X-axis Statistic/SV",
            options=list(sv_scatter_options.keys()),
            key="scatter_sv_x"
        )
        y_option = col2.selectbox(
            "Y-axis Statistic/SV",
            options=list(sv_scatter_options.keys()),
            index=1,
            key="scatter_sv_y"
        )

        # Helper function to compute derived SV statistic (same as Section 6)
        def compute_derived_sv_stat_s7(svd_array, stat_type):
            """Compute derived statistics from singular values."""
            sv = np.array(svd_array)
            d_head = d_model // n_heads  # Number of non-zero SVs expected

            if stat_type == "mean":
                return np.mean(sv)
            elif stat_type == "variance":
                return np.var(sv)
            elif stat_type == "skewness":
                return scipy_stats.skew(sv)
            elif stat_type == "kurtosis":
                return scipy_stats.kurtosis(sv)
            elif stat_type == "sum":
                return np.sum(sv)
            elif stat_type == "sum_squares":
                return np.sum(sv**2)
            elif stat_type == "participation_ratio":
                sum_sv = np.sum(sv)
                sum_sv2 = np.sum(sv**2)
                return (sum_sv**2) / sum_sv2 if sum_sv2 > 0 else 0
            elif stat_type == "normalized_participation_ratio":
                sum_sv = np.sum(sv)
                sum_sv2 = np.sum(sv**2)
                participation_ratio = (sum_sv**2) / sum_sv2 if sum_sv2 > 0 else 0
                return participation_ratio / d_head if d_head > 0 else 0
            elif stat_type == "spectral_entropy":
                sv2 = sv**2
                sum_sv2 = np.sum(sv2)
                if sum_sv2 > 0:
                    p = sv2 / sum_sv2
                    p = p[p > 0]  # Remove zeros to avoid log(0)
                    return -np.sum(p * np.log(p))
                return 0
            elif stat_type == "condition_number":
                # Use only the d_head largest SVs (low-rank structure)
                sv_nonzero = sv[:d_head]
                if len(sv_nonzero) > 0 and sv_nonzero[-1] > 0:
                    return sv_nonzero[0] / sv_nonzero[-1]
                return 0
            elif stat_type == "stable_rank":
                sum_sv2 = np.sum(sv**2)
                max_sv2 = sv[0]**2
                return sum_sv2 / max_sv2 if max_sv2 > 0 else 0

            return 0

        # Helper function to extract value for a single row
        def get_value_for_row(row, option_name):
            option_type, option_data = sv_scatter_options[option_name]

            if option_type == "stat":
                return row[option_data]
            elif option_type == "sv":
                return row["SVD"][option_data]
            elif option_type == "derived":
                return compute_derived_sv_stat_s7(row["SVD"], option_data)

        # Generate colors for each layer (same as Section 5)
        all_colors = px.colors.qualitative.Plotly
        layer_0_color = all_colors[0]
        other_colors = all_colors[1:]

        def get_layer_color(layer_idx):
            if layer_idx == 0:
                return layer_0_color
            else:
                return other_colors[(layer_idx - 1) % len(other_colors)]

        # Create scatter plot
        fig = go.Figure()

        for layer_idx in range(n_layers):
            df_layer = df.query(f"layer == {layer_idx}")

            x_vals = []
            y_vals = []
            for _, row in df_layer.iterrows():
                x_vals.append(get_value_for_row(row, x_option))
                y_vals.append(get_value_for_row(row, y_option))

            # Create hover text showing layer and head info
            hover_text = [f"Layer {layer_idx}, Head {h}<br>{x_option}: {x:.4f}<br>{y_option}: {y:.4f}"
                          for h, x, y in zip(df_layer["head"].values, x_vals, y_vals)]

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    name=f"Layer {layer_idx}",
                    marker=dict(
                        color=get_layer_color(layer_idx),
                        size=8,
                        opacity=0.7
                    ),
                    hovertext=hover_text,
                    hoverinfo="text"
                )
            )

        fig.update_layout(
            xaxis_title=x_option,
            yaxis_title=y_option,
            hovermode="closest",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            height=600
        )

        st.plotly_chart(fig, width="content", key="scatter_sv_plot")

        # Snapshot button for Section 7
        create_snapshot_button(
            fig=fig,
            metadata={
                "section": "section_7_compare_statistics_sv",
                "campaign": campaign_name,
                "model": model_selected,
                "weight_type": weight_selected,
                "x_option": x_option,
                "y_option": y_option,
                "n_layers": int(n_layers),
                "n_heads": int(n_heads),
                "d_model": d_model,
            },
            section_name="weights_section_7",
            key="snapshot_weights_s7"
        )


def render():
    weights_dashboard_app()


if __name__ == "__main__":
    weights_dashboard_app()