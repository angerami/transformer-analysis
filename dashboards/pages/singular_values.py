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
)


def singular_values_app():
    """Singular Value Analysis Dashboard - dedicated page for SVD metrics and visualization."""

    merge_key = 'merged'

    # Load data
    if is_HF_environment():
        campaign_name = "ana-004"
    else:
        available_datasets = get_available_campaigns("ana-")
        if not available_datasets:
            st.error("No datasets found.")
            st.stop()
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

    # Filter to W_QK only (singular values only available for W_QK)
    weight_types = get_unique_values(df_full, "weight_type")
    if "W_QK" not in weight_types:
        st.error("No W_QK data found. Singular value analysis requires W_QK matrices.")
        st.stop()

    weight_selected = "W_QK"

    df = df_full.query(f"model == '{model_selected}' and weight_type == '{weight_selected}'")

    # Get architecture dimensions
    d_model = metadata["d_model"]
    n_layers = df["layer"].max() + 1
    n_heads = df["head"].max() + 1

    # Sort dataframe for consistent plotting
    df_sorted = df.sort_values(by=["layer", "head"]).reset_index(drop=True)

    ########
    st.title("Singular Value Analysis Dashboard")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Info")
    st.sidebar.markdown(f"Model: {model_selected}")
    st.sidebar.markdown(f"Weight Type: W_QK (d_head × d_head)")
    st.sidebar.markdown(f"Model Dimension: {d_model}")
    st.sidebar.markdown(f"Heads: {n_heads}")
    st.sidebar.markdown(f"Layers: {n_layers}")
    st.sidebar.markdown(f"Head Dimension: {d_model // n_heads}")

    st.sidebar.markdown("---")
    use_eigenvalues = st.sidebar.checkbox(
        "Plot eigenvalues (λ²)",
        value=True,
        help="Square singular values to show eigenvalues. "
             "Eigenvalues are the natural RMT quantity and enhance bulk/outlier separation."
    )
    spec_label = r"\lambda^2" if use_eigenvalues else r"\lambda"
    spec_name = "Eigenvalue" if use_eigenvalues else "Singular Value"

    def to_plot_space(sv_array):
        """Convert SVD array to plot space (eigenvalues if toggle is on)."""
        sv = np.array(sv_array)
        return sv ** 2 if use_eigenvalues else sv

    ########################################################################
    # Section 1: Single Head Singular Value Visualization
    ########################################################################

    st.header(f"Section 1: Single Head {spec_name}s")
    st.markdown(f"Visualize the {spec_name.lower()} spectrum of W_QK for a specific attention head.")

    col1, col2 = st.columns(2)
    layer_s1 = col1.slider("Layer", 0, n_layers - 1, 0, key="s1_layer")
    head_s1 = col2.slider("Head", 0, n_heads - 1, 0, key="s1_head")

    entry_s1 = df.query(f"layer == {layer_s1} and head == {head_s1}")
    svd_s1 = entry_s1["SVD"].iloc[0]
    plot_vals_s1 = to_plot_space(svd_s1)

    col1, col2 = st.columns(2)
    plot_type_s1 = col1.selectbox("Plot Type", [f"{spec_name}s", f"P({spec_label})"], key="s1_plot_type")
    use_log_s1 = col2.checkbox("Log scale", key="s1_log")

    if spec_name in plot_type_s1:
        fig_s1 = go.Figure()
        fig_s1.add_trace(
            go.Scatter(
                x=list(range(len(plot_vals_s1))),
                y=plot_vals_s1,
                mode="lines+markers",
                name=f"{spec_name}s",
            )
        )
        fig_s1.update_layout(
            xaxis_title="Index",
            yaxis_title=spec_name,
            yaxis_type="log" if use_log_s1 else "linear",
        )
    else:
        # Histogram from raw SVD array (re-binned for eigenvalue space)
        d_head = d_model // n_heads
        vals = plot_vals_s1[:d_head]
        fig_s1 = go.Figure()
        fig_s1.add_trace(go.Histogram(
            x=vals, nbinsx=40, histnorm="probability density",
            name=f"P({spec_label})"
        ))
        fig_s1.update_layout(
            xaxis_title=spec_name,
            yaxis_title="Density",
            yaxis_type="log" if use_log_s1 else "linear",
        )

    st.plotly_chart(fig_s1, use_container_width=True, key="s1_plot")

    # Display computed metrics
    st.subheader("Computed Metrics")
    col1, col2, col3 = st.columns(3)

    # Check if metrics exist in the dataframe (from post-processing)
    metrics_to_display = {
        "Mean": "sv_mean",
        "Std Dev": "std",
        "Participation Ratio": "participation_ratio",
        "Spectral Entropy": "spectral_entropy",
        "Condition Number": "condition_number",
        "Stable Rank": "stable_rank",
    }

    for idx, (display_name, col_name) in enumerate(metrics_to_display.items()):
        if col_name in entry_s1.columns:
            value = entry_s1[col_name].iloc[0]
            if idx % 3 == 0:
                col1.metric(display_name, f"{value:.4f}")
            elif idx % 3 == 1:
                col2.metric(display_name, f"{value:.4f}")
            else:
                col3.metric(display_name, f"{value:.4f}")

    ########################################################################
    # Section 2: Singular Value Metrics Across Architecture
    ########################################################################

    st.header(f"Section 2: {spec_name} Metrics Across Architecture")
    st.markdown(f"Compare {spec_name.lower()} statistics across all attention heads in the model.")

    # Get the number of singular values from the first entry
    first_svd = df_sorted["SVD"].iloc[0]
    n_svs = len(first_svd)

    # Build options: pre-computed stats + derived SV stats + individual SVs
    sv_options = {}

    # Add pre-computed statistics (if they exist from post-processing)
    precomputed_sv_metrics = [
        ("SV Mean", "sv_mean"),
        ("SV Variance", "sv_variance"),
        ("SV Skewness", "sv_skewness"),
        ("SV Kurtosis", "sv_kurtosis"),
        ("Σσ", "sv_sum"),
        ("Σσ²", "sv_sum_squares"),
        ("Participation Ratio", "participation_ratio"),
        ("Normalized Participation Ratio", "normalized_participation_ratio"),
        ("Spectral Entropy", "spectral_entropy"),
        ("Condition Number", "condition_number"),
        ("Stable Rank", "stable_rank"),
    ]

    for display_name, col_name in precomputed_sv_metrics:
        if col_name in df_sorted.columns:
            sv_options[display_name] = ("stat", col_name)

    # Add standard stats
    for display_name, col_name in stat_display.items():
        if col_name in df_sorted.columns:
            sv_options[display_name] = ("stat", col_name)

    # If pre-computed metrics don't exist, add derived computation options
    if not any(col_name in df_sorted.columns for _, col_name in precomputed_sv_metrics):
        st.info("Note: Singular value metrics not found in dataset. Computing on-the-fly. "
                "Run with `--reprocess-metrics` to pre-compute and store these metrics.")

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

    # Add individual singular values / eigenvalues
    sv_prefix = "EV" if use_eigenvalues else "SV"
    for k in range(min(n_svs, 20)):
        sv_options[f"{sv_prefix}[{k}]"] = ("sv", k)

    # Multi-select for what to plot
    default_metrics = ["Participation Ratio"] if "Participation Ratio" in sv_options else [list(sv_options.keys())[0]]
    selected_sv_stats = st.multiselect(
        "Select Statistics/Singular Values to Plot",
        options=list(sv_options.keys()),
        default=default_metrics,
        key="sv_stats_select"
    )

    if not selected_sv_stats:
        st.warning("Please select at least one statistic or singular value")
    else:
        # Options
        col1, col2 = st.columns(2)
        use_separate_axes_sv = col1.checkbox(
            "Use separate Y-axes",
            value=False,
            help="Each statistic gets its own Y-axis with matching colors",
            key="separate_axes_sv"
        )
        show_layer_avg_sv = col2.checkbox(
            "Show layer averages",
            value=False,
            help="Display the mean value across all heads in each layer",
            key="layer_avg_sv"
        )

        # Color palette
        colors_sv = px.colors.qualitative.Plotly[: len(selected_sv_stats)]

        # Helper function to compute derived SV statistic
        def compute_derived_sv_stat(svd_array, stat_type):
            """Compute derived statistics from singular values."""
            sv = np.array(svd_array)
            d_head = d_model // n_heads

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
                    p = p[p > 0]
                    return -np.sum(p * np.log(p))
                return 0
            elif stat_type == "condition_number":
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
                    sv_val = row["SVD"][option_data]
                    values.append(sv_val ** 2 if use_eigenvalues else sv_val)
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
                    avg_values = []
                    option_type, option_data = sv_options[option_name]
                    for layer_idx in range(n_layers):
                        df_layer = df.query(f"layer == {layer_idx}")

                        layer_values = []
                        for _, row in df_layer.iterrows():
                            if option_type == "stat":
                                layer_values.append(row[option_data])
                            elif option_type == "sv":
                                layer_values.append(row["SVD"][option_data])
                            elif option_type == "derived":
                                layer_values.append(compute_derived_sv_stat(row["SVD"], option_data))

                        layer_avg = np.mean(layer_values)
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
                    avg_values = []
                    option_type, option_data = sv_options[option_name]
                    for layer_idx in range(n_layers):
                        df_layer = df.query(f"layer == {layer_idx}")

                        layer_values = []
                        for _, row in df_layer.iterrows():
                            if option_type == "stat":
                                layer_values.append(row[option_data])
                            elif option_type == "sv":
                                layer_values.append(row["SVD"][option_data])
                            elif option_type == "derived":
                                layer_values.append(compute_derived_sv_stat(row["SVD"], option_data))

                        layer_avg = np.mean(layer_values)
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

        st.plotly_chart(fig, use_container_width=True, key="section2_plot")

    ########################################################################
    # Section 3: Scatter Plot - Compare Two SV Statistics
    ########################################################################

    st.header(f"Section 3: Compare Two {spec_name} Statistics")
    st.markdown(f"Scatter plot comparing any two {spec_name.lower()} metrics across all attention heads.")

    col1, col2 = st.columns(2)
    x_option = col1.selectbox(
        "X-axis Statistic/SV",
        options=list(sv_options.keys()),
        key="scatter_sv_x"
    )
    y_option = col2.selectbox(
        "Y-axis Statistic/SV",
        options=list(sv_options.keys()),
        index=min(1, len(sv_options) - 1),
        key="scatter_sv_y"
    )

    # Helper function to extract value for a single row
    def get_value_for_row(row, option_name):
        option_type, option_data = sv_options[option_name]

        if option_type == "stat":
            return row[option_data]
        elif option_type == "sv":
            sv_val = row["SVD"][option_data]
            return sv_val ** 2 if use_eigenvalues else sv_val
        elif option_type == "derived":
            return compute_derived_sv_stat(row["SVD"], option_data)
        return 0

    # Extract x and y values
    x_vals = []
    y_vals = []
    for _, row in df_sorted.iterrows():
        x_vals.append(get_value_for_row(row, x_option))
        y_vals.append(get_value_for_row(row, y_option))

    # Create scatter plot
    # Generate colors for each layer
    all_colors = px.colors.qualitative.Plotly
    layer_0_color = all_colors[0]
    other_colors = all_colors[1:]

    def get_layer_color(layer_idx):
        if layer_idx == 0:
            return layer_0_color
        else:
            return other_colors[(layer_idx - 1) % len(other_colors)]

    fig_scatter = go.Figure()

    for layer_idx in range(n_layers):
        df_layer = df_sorted.query(f"layer == {layer_idx}")
        layer_x = []
        layer_y = []
        hover_text = []

        for _, row in df_layer.iterrows():
            x_val = get_value_for_row(row, x_option)
            y_val = get_value_for_row(row, y_option)
            layer_x.append(x_val)
            layer_y.append(y_val)
            hover_text.append(
                f"Layer {layer_idx}, Head {int(row['head'])}<br>"
                f"{x_option}: {x_val:.4f}<br>{y_option}: {y_val:.4f}"
            )

        fig_scatter.add_trace(
            go.Scatter(
                x=layer_x,
                y=layer_y,
                mode="markers",
                name=f"Layer {layer_idx}",
                marker=dict(size=8, color=get_layer_color(layer_idx)),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig_scatter.update_layout(
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
    )

    st.plotly_chart(fig_scatter, use_container_width=True, key="section3_plot")


def render():
    """Render the singular values dashboard page."""
    singular_values_app()
