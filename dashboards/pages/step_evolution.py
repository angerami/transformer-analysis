"""
Step Evolution Dashboard
Visualizes how statistics evolve across training checkpoints (steps)
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy import stats as scipy_stats
from dashboard_utils import (
    stat_display,
    get_available_datasets,
    load_dataset_with_metadata,
    get_unique_values,
    get_data_path,
    is_HF_environment,
)


def step_evolution_app():
    st.title("Training Step Evolution Analysis")
    st.markdown("Visualize how model statistics evolve across training checkpoints")
    st.set_page_config(page_title="Step Evolution")

    # Sidebar: Model and weight type selection
    st.sidebar.header("Dataset Selection")

    ####
    hf_version = "weight_evolution"  # dropped _002
    if is_HF_environment():
        available_datasets = get_available_datasets(hf_version)
    else:
        campaign = "step-analysis_002"
        available_datasets = get_available_datasets(campaign)
        campaign = st.sidebar.selectbox("Campaign", [campaign])

    if not available_datasets:
        st.error(f"No datasets found.")
        st.stop()

    ds_name = st.sidebar.selectbox(
        "Dataset",
        available_datasets,
        index=available_datasets.index("pythia-1.4b-deduped") if "pythia-1.4b-deduped" in available_datasets else 0,
    )

    df_full, metadata = load_dataset_with_metadata(
        ds_name=ds_name,
        campaign=campaign if not is_HF_environment() else None,
        hf_version=hf_version,
    )
    st.success(f"Loaded: {ds_name}")

    model_names = get_unique_values(df_full, "model")
    # model_selected = st.sidebar.selectbox("Model", model_names)
    model_selected = model_names[0]

    weight_types = get_unique_values(df_full, "weight_type")
    weight_selected = st.sidebar.selectbox("Weight Type", weight_types)

    # Filter by model and weight type (all steps)
    df = df_full.query(
        f"model == '{model_selected}' and weight_type == '{weight_selected}'"
    )

    # Get architecture dimensions
    d_model = metadata.get("d_model")
    n_layers = df["layer"].max() + 1
    n_heads = df["head"].max() + 1

    # Get available steps and sort them
    steps_available = sorted(df["step"].unique())
    st.sidebar.markdown(f"**Available steps:** {len(steps_available)}")
    st.sidebar.markdown(f"Range: {steps_available[0]} - {steps_available[-1]}")

    st.sidebar.markdown("---")
    use_eigenvalues = st.sidebar.checkbox(
        "Plot eigenvalues (λ²)",
        value=True,
        help="Square singular values to show eigenvalues.",
        key="step_use_eigenvalues"
    )

    def to_plot_space(sv_array):
        sv = np.array(sv_array)
        return sv ** 2 if use_eigenvalues else sv

    def compute_derived_sv_stat(svd_array, stat_type):
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
        elif stat_type == "leading_sv":
            return float(sv[0])
        return 0

    # Build extended statistics options including SVD-based stats
    extended_stat_display = dict(stat_display)
    sv_stat_options = {}

    if weight_selected == "W_QK":
        sv_stat_options["SV Mean"] = ("derived", "mean")
        sv_stat_options["SV Variance"] = ("derived", "variance")
        sv_stat_options["SV Skewness"] = ("derived", "skewness")
        sv_stat_options["SV Kurtosis"] = ("derived", "kurtosis")
        sv_stat_options["Σσ"] = ("derived", "sum")
        sv_stat_options["Σσ²"] = ("derived", "sum_squares")
        sv_stat_options["Participation Ratio"] = ("derived", "participation_ratio")
        sv_stat_options["Normalized Participation Ratio"] = ("derived", "normalized_participation_ratio")
        sv_stat_options["Spectral Entropy"] = ("derived", "spectral_entropy")
        sv_stat_options["Condition Number"] = ("derived", "condition_number")
        sv_stat_options["Stable Rank"] = ("derived", "stable_rank")
        sv_stat_options["Max SV"] = ("derived", "leading_sv")

        for display_name in sv_stat_options.keys():
            extended_stat_display[display_name] = display_name

    def get_stat_value(row, stat_display_name):
        if stat_display_name in sv_stat_options:
            _, stat_type = sv_stat_options[stat_display_name]
            return compute_derived_sv_stat(row["SVD"], stat_type)
        else:
            stat_name = stat_display[stat_display_name]
            return row[stat_name]

    # ============================================================================
    # SECTION 1: Single Layer/Head Evolution
    # ============================================================================
    st.header("Section 1: Statistic Evolution for Single Layer/Head")
    st.markdown(
        "Track how a specific statistic evolves over training steps for a chosen layer and head"
    )

    col1, col2 = st.columns(2)
    layer_selected = col1.slider("Layer", 0, n_layers - 1, 0, key="layer_s1")
    head_selected = col2.slider("Head", 0, n_heads - 1, 0, key="head_s1")

    stat_display_name = st.selectbox(
        "Statistic", list(extended_stat_display.keys()), key="stat_s1"
    )

    # Filter for the specific layer/head across all steps
    df_filtered = df.query(f"layer == {layer_selected} and head == {head_selected}")
    df_filtered = df_filtered.sort_values("step")

    # Compute stat values
    stat_values = np.array([get_stat_value(row, stat_display_name) for _, row in df_filtered.iterrows()])

    # Create line plot
    fig_s1 = go.Figure()
    fig_s1.add_trace(go.Scatter(
        x=df_filtered["step"].values,
        y=stat_values,
        mode="lines+markers",
        marker=dict(size=5),
    ))

    # Log scale option
    use_log_s1 = st.checkbox("Use log scale (x-axis)", key="log_s1")

    fig_s1.update_layout(
        xaxis_title="Training Step",
        yaxis_title=stat_display_name,
        title=f"{stat_display_name} vs Training Step (Layer {layer_selected}, Head {head_selected})",
        xaxis_type="log" if use_log_s1 else "linear"
    )

    st.plotly_chart(fig_s1, width="stretch")

    # Display some statistics about the evolution
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Initial Value", f"{stat_values[0]:.4f}")
    col2.metric("Final Value", f"{stat_values[-1]:.4f}")
    col3.metric("Change", f"{stat_values[-1] - stat_values[0]:.4f}")
    col4.metric("Max Value", f"{stat_values.max():.4f}")

    # ============================================================================
    # SECTION 2: 2D Heatmap (Step vs Layer/Head)
    # ============================================================================
    st.header("Section 2: Evolution Heatmap Across Architecture")
    st.markdown(
        "Visualize how statistics evolve across both training steps and model architecture"
    )

    stat_display_name_2d = st.selectbox(
        "Statistic", list(extended_stat_display.keys()), key="stat_s2"
    )

    # Prepare data for heatmap
    # Sort by layer and head to get consistent ordering
    df_for_heatmap = df.sort_values(["step", "layer", "head"]).copy()

    # Create a composite index for layer/head
    df_for_heatmap["layer_head_idx"] = (
        df_for_heatmap["layer"] * n_heads + df_for_heatmap["head"]
    )

    # Compute stat values
    df_for_heatmap["stat_value"] = df_for_heatmap.apply(
        lambda row: get_stat_value(row, stat_display_name_2d), axis=1
    )

    # Pivot the data
    heatmap_data = df_for_heatmap.pivot(
        index="layer_head_idx", columns="step", values="stat_value"
    )

    # Get the step values for x-axis
    step_values = heatmap_data.columns.values

    # Create heatmap
    use_log_2d = st.checkbox("Log scale (color)", key="log_2d")

    z_data = heatmap_data.values
    if use_log_2d:
        z_data = np.log10(np.abs(z_data) + 1e-10)  # Add small epsilon to avoid log(0)

    fig_s2 = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=step_values,
            y=heatmap_data.index.values,
            colorscale="Viridis",
            colorbar=dict(title="Log10 Value" if use_log_2d else "Value"),
        )
    )

    # Add vertical lines to separate layers
    if n_layers < 16:
        for layer_boundary in range(n_heads, n_layers * n_heads, n_heads):
            fig_s2.add_hline(
                y=layer_boundary - 0.5, line=dict(color="white", width=0.5, dash="dot")
            )

    fig_s2.update_layout(
        xaxis_title="Training Step",
        yaxis_title="Layer/Head Index",
        title=f"{stat_display_name_2d} Evolution Heatmap",
        height=600,
        xaxis=dict(
            type="log"
            if st.checkbox("Log scale (x-axis)", key="log_x_s2")
            else "linear"
        ),
    )

    st.plotly_chart(fig_s2, width="stretch")

    # Add interpretation help
    with st.expander("Understanding the heatmap"):
        st.markdown(f"""
        - **Y-axis**: Concatenated layer/head index (0 to {n_layers * n_heads - 1})
        - Heads 0-{n_heads - 1}: Layer 0
        - Heads {n_heads}-{2 * n_heads - 1}: Layer 1
        - And so on...
        - **X-axis**: Training step (checkpoint)
        - **Color**: Value of {stat_display_name_2d}
        - **White horizontal lines**: Layer boundaries
        
        Look for patterns like:
        - Vertical bands: Changes affecting all layers/heads at specific steps
        - Horizontal bands: Specific layers/heads behaving differently
        - Gradients: Gradual evolution patterns
        """)

    # ============================================================================
    # Footer information
    # ============================================================================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Info")
    # st.sidebar.markdown(f"Total rows: {len(df_full):,}")
    # st.sidebar.markdown(f"Filtered rows: {len(df):,}")
    st.sidebar.markdown(f"Model Dimension: {d_model}")
    st.sidebar.markdown(f"Layers: {n_layers}")
    st.sidebar.markdown(f"Heads per layer: {n_heads}")

    # ============================================================================
    # SECTION 3: Per-Head Comparison for Selected Layer
    # ============================================================================
    st.header("Section 3: Multi-Head Comparison Within Layer")
    st.markdown("Compare evolution of a statistic across all heads in a selected layer")

    col1, col2 = st.columns(2)
    layer_selected_s3 = col1.slider("Layer", 0, n_layers - 1, 0, key="layer_s3")
    stat_display_name_s3 = col2.selectbox(
        "Statistic", list(extended_stat_display.keys()), key="stat_s3"
    )

    # Option to choose between separate panels or same panel
    view_mode = st.radio(
        "Display mode",
        ["Separate panels (subplots)", "Same panel (overlaid)"],
        key="view_mode_s3",
    )

    use_log_s3 = st.checkbox("Use log scale (x-axis)", key="log_s3")

    # Filter for the selected layer across all heads and steps
    df_layer = df.query(f"layer == {layer_selected_s3}")

    if view_mode == "Separate panels (subplots)":
        # Create subplot grid
        n_cols = 4
        n_rows = int(np.ceil(n_heads / n_cols))

        from plotly.subplots import make_subplots

        fig_s3 = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"Head {i}" for i in range(n_heads)],
            x_title="Training Step",
            y_title=stat_display_name_s3,
        )

        for head_idx in range(n_heads):
            df_head = df_layer.query(f"head == {head_idx}").sort_values("step")

            row = head_idx // n_cols + 1
            col = head_idx % n_cols + 1

            y_values = [get_stat_value(r, stat_display_name_s3) for _, r in df_head.iterrows()]

            fig_s3.add_trace(
                go.Scatter(
                    x=df_head["step"],
                    y=y_values,
                    mode="lines+markers",
                    marker=dict(size=3),
                    name=f"Head {head_idx}",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        if use_log_s3:
            fig_s3.update_xaxes(type="log")

        fig_s3.update_layout(
            height=200 * n_rows,
            title_text=f"Layer {layer_selected_s3}: {stat_display_name_s3} Evolution Across Heads",
        )

    else:  # Same panel (overlaid)
        fig_s3 = go.Figure()

        # Use a color scale for different heads
        colors = px.colors.sample_colorscale(
            "Viridis", [i / (n_heads - 1) for i in range(n_heads)]
        )

        for head_idx in range(n_heads):
            df_head = df_layer.query(f"head == {head_idx}").sort_values("step")

            y_values = [get_stat_value(r, stat_display_name_s3) for _, r in df_head.iterrows()]

            fig_s3.add_trace(
                go.Scatter(
                    x=df_head["step"],
                    y=y_values,
                    mode="lines+markers",
                    marker=dict(size=4),
                    name=f"Head {head_idx}",
                    line=dict(color=colors[head_idx]),
                    opacity=0.7,
                )
            )

        # Compute and plot layer average
        layer_steps = sorted(df_layer["step"].unique())
        layer_avg_values = []
        for step in layer_steps:
            df_step = df_layer.query(f"step == {step}")
            step_values = [get_stat_value(r, stat_display_name_s3) for _, r in df_step.iterrows()]
            layer_avg_values.append(np.mean(step_values))

        fig_s3.add_trace(
            go.Scatter(
                x=layer_steps,
                y=layer_avg_values,
                mode="lines",
                line=dict(color="red", width=3, dash="dash"),
                name="Layer average",
            )
        )

        fig_s3.update_layout(
            xaxis_title="Training Step",
            yaxis_title=stat_display_name_s3,
            title=f"Layer {layer_selected_s3}: {stat_display_name_s3} Evolution Across All Heads",
            height=600,
            xaxis_type="log" if use_log_s3 else "linear",
            hovermode="x unified",
        )

    st.plotly_chart(fig_s3, width="stretch")

    # Summary statistics for this layer
    with st.expander("Layer statistics summary"):
        st.markdown(f"**Layer {layer_selected_s3} - {stat_display_name_s3}**")

        # Compute statistics across heads at final step
        final_step = steps_available[-1]
        df_final = df_layer.query(f"step == {final_step}")

        final_values = np.array([get_stat_value(r, stat_display_name_s3) for _, r in df_final.iterrows()])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean across heads", f"{final_values.mean():.4f}")
        col2.metric("Std across heads", f"{final_values.std():.4f}")
        col3.metric("Min head value", f"{final_values.min():.4f}")
        col4.metric("Max head value", f"{final_values.max():.4f}")

    # ============================================================================
    # SECTION 4: Multi-Layer Comparison with Head Averages
    # ============================================================================
    st.header("Section 4: Layer-by-Layer Evolution with Averages")
    st.markdown("Compare individual heads and layer averages across multiple layers")

    stat_display_name_s4 = st.selectbox(
        "Statistic", list(extended_stat_display.keys()), key="stat_s4"
    )

    use_log_s4 = st.checkbox("Use log scale (x-axis)", key="log_s4")

    # Create subplot grid - one panel per layer
    n_cols_s4 = min(3, n_layers)  # Max 3 columns
    n_rows_s4 = int(np.ceil(n_layers / n_cols_s4))

    from plotly.subplots import make_subplots

    fig_s4 = make_subplots(
        rows=n_rows_s4,
        cols=n_cols_s4,
        subplot_titles=[f"Layer {i}" for i in range(n_layers)],
        x_title="Training Step",
        y_title=stat_display_name_s4,
    )

    # Use a color scale for different heads
    colors = px.colors.sample_colorscale(
        "Viridis", [i / (n_heads - 1) for i in range(n_heads)]
    )

    for layer_idx in range(n_layers):
        df_layer = df.query(f"layer == {layer_idx}")

        row = layer_idx // n_cols_s4 + 1
        col = layer_idx % n_cols_s4 + 1

        # Plot individual heads with thin lines
        for head_idx in range(n_heads):
            df_head = df_layer.query(f"head == {head_idx}").sort_values("step")

            y_values = [get_stat_value(r, stat_display_name_s4) for _, r in df_head.iterrows()]

            fig_s4.add_trace(
                go.Scatter(
                    x=df_head["step"],
                    y=y_values,
                    mode="lines",
                    line=dict(color=colors[head_idx], width=1),
                    name=f"L{layer_idx}H{head_idx}",
                    showlegend=False,
                    opacity=0.5,
                ),
                row=row,
                col=col,
            )

        # Compute and plot layer average (mean across heads)
        layer_steps = sorted(df_layer["step"].unique())
        layer_avg_values = []
        for step in layer_steps:
            df_step = df_layer.query(f"step == {step}")
            step_values = [get_stat_value(r, stat_display_name_s4) for _, r in df_step.iterrows()]
            layer_avg_values.append(np.mean(step_values))

        fig_s4.add_trace(
            go.Scatter(
                x=layer_steps,
                y=layer_avg_values,
                mode="lines",
                line=dict(color="red", width=3, dash="dash"),
                name="Layer average",
                showlegend=(layer_idx == 0),  # Only show legend for first layer
            ),
            row=row,
            col=col,
        )

    if use_log_s4:
        fig_s4.update_xaxes(type="log")

    fig_s4.update_layout(
        height=300 * n_rows_s4,
        title_text=f"{stat_display_name_s4} Evolution: Individual Heads (thin) vs Layer Average (red dashed)",
    )

    st.plotly_chart(fig_s4, width="stretch")

    with st.expander("Understanding this view"):
        st.markdown("""
        - **Thin colored lines**: Individual head trajectories
        - **Thick red dashed line**: Average across all heads in that layer
        - Each panel represents one layer
        - Useful for seeing:
        - Head diversity within layers
        - Whether heads converge or diverge during training
        - How average layer behavior differs across layers
        """)

        # ============================================================================
    # SECTION 5: Distribution Evolution Heatmaps
    # ============================================================================
    st.header("Section 5: Distribution Evolution Over Training")
    st.markdown(
        "Visualize how weight/singular value distributions change across training steps"
    )

    col1, col2, col3 = st.columns(3)
    layer_selected_s5 = col1.slider("Layer", 0, n_layers - 1, 0, key="layer_s5")
    head_selected_s5 = col2.slider("Head", 0, n_heads - 1, 0, key="head_s5")

    # Determine available distributions based on weight type
    available_dists_s5 = ["P(W)"]
    if weight_selected == "W_QK":
        available_dists_s5.extend(["P(λ)", "SVD"])

    dist_type_s5 = col3.selectbox("Distribution", available_dists_s5, key="dist_s5")

    # Map display names to column names
    dist_map = {"P(W)": "P_w", "P(λ)": "P_sv", "SVD": "SVD"}
    dist_col = dist_map[dist_type_s5]

    # Get appropriate bins
    ev_label = "Eigenvalue" if use_eigenvalues else "Singular Value"
    if dist_col == "P_w":
        bins = np.array(metadata["w_bins"])
        xlabel = "Weight Value"
    elif dist_col == "P_sv":
        bins = np.array(metadata["sv_bins"])
        xlabel = ev_label
    else:  # SVD
        xlabel = f"{ev_label} Index"

    # Filter data for selected layer/head across all steps
    df_filtered_s5 = df.query(
        f"layer == {layer_selected_s5} and head == {head_selected_s5}"
    )
    df_filtered_s5 = df_filtered_s5.sort_values("step")

    # Stack distributions into 2D array
    if dist_col == "SVD":
        dist_stack = np.array([to_plot_space(row[dist_col]) for _, row in df_filtered_s5.iterrows()])
    else:
        dist_stack = np.array([row[dist_col] for _, row in df_filtered_s5.iterrows()])
    step_values_s5 = df_filtered_s5["step"].values

    # Compute bin centers
    if dist_col == "SVD":
        bin_centers = np.arange(dist_stack.shape[1])
    else:
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Log scale options
    use_log_color_s5 = st.checkbox("Log scale (color)", key="log_color_s5")
    use_log_x_s5 = st.checkbox("Log scale (x-axis, steps)", key="log_x_s5")

    # Calculate minimum color value based on d_model
    d_model = metadata.get("d_model")
    zmin = None
    zmax = None

    if d_model is not None and dist_col != "SVD":
        min_prob = 0.5 / (d_model ** 2)
        if use_log_color_s5:
            zmin = np.log10(min_prob)
        else:
            zmin = min_prob

    z_data_s5 = dist_stack
    if use_log_color_s5:
        z_data_s5 = np.log10(np.abs(dist_stack) + 1e-10)

    fig_s5 = go.Figure(
        data=go.Heatmap(
            z=z_data_s5,
            x=bin_centers,
            y=step_values_s5,
            colorscale="Viridis",
            colorbar=dict(title="Log10" if use_log_color_s5 else "Value"),
            zmin=zmin,
            zmax=zmax,
        )
    )

    # Set x-axis range dynamically for SVD plots
    xaxis_range = None
    if dist_col == "SVD":
        if d_model is not None:
            xaxis_range = [0, d_model // n_heads - 1]

    fig_s5.update_layout(
        xaxis_title=xlabel,
        yaxis_title="Training Step",
        title=f"{dist_type_s5} Evolution: Layer {layer_selected_s5}, Head {head_selected_s5}",
        height=600,
        yaxis_type="log" if use_log_x_s5 else "linear",
        xaxis_range=xaxis_range,
    )

    st.plotly_chart(fig_s5, width="stretch")

    with st.expander("Interpretation guide"):
        st.markdown(f"""
        - **X-axis**: {xlabel}
        - **Y-axis**: Training step (checkpoint)
        - **Color**: Distribution value at that bin/index

        Look for:
        - **Horizontal bands**: Consistent distribution shape across training
        - **Vertical evolution**: Changes in peak location or spread
        - **Emergence/disappearance of features**: New modes or tail behavior
        """)

    # ============================================================================
    # SECTION 6: Overlaid Distributions Across Selected Steps
    # ============================================================================
    st.header("Section 6: Distribution Evolution - Overlaid View")
    st.markdown(
        "Compare distributions across multiple training steps on the same plot"
    )

    col1, col2, col3 = st.columns(3)
    layer_selected_s6 = col1.slider("Layer", 0, n_layers - 1, 0, key="layer_s6")
    head_selected_s6 = col2.slider("Head", 0, n_heads - 1, 0, key="head_s6")

    # Determine available distributions based on weight type
    available_dists_s6 = ["P(W)"]
    if weight_selected == "W_QK":
        available_dists_s6.extend(["P(λ)", "SVD"])

    dist_type_s6 = col3.selectbox("Distribution", available_dists_s6, key="dist_s6")

    # Map display names to column names
    dist_map = {"P(W)": "P_w", "P(λ)": "P_sv", "SVD": "SVD"}
    dist_col_s6 = dist_map[dist_type_s6]

    # Step selection with multiselect
    st.markdown("**Select training steps to overlay:**")

    # Suggest some default steps (first, middle, last)
    default_indices = [0, len(steps_available) // 2, len(steps_available) - 1]
    default_steps = [steps_available[i] for i in default_indices if i < len(steps_available)]

    selected_steps_s6 = st.multiselect(
        "Training steps",
        options=steps_available,
        default=default_steps,
        key="steps_s6",
        help="Select multiple steps to overlay their distributions"
    )

    if not selected_steps_s6:
        st.warning("Please select at least one training step to visualize.")
    else:
        # Filter data for selected layer/head and selected steps
        df_filtered_s6 = df.query(
            f"layer == {layer_selected_s6} and head == {head_selected_s6} and step in @selected_steps_s6"
        )

        # Get appropriate bins
        ev_label_s6 = "Eigenvalue" if use_eigenvalues else "Singular Value"
        if dist_col_s6 == "P_w":
            bins = np.array(metadata["w_bins"])
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            xlabel_s6 = "Weight Value"
        elif dist_col_s6 == "P_sv":
            bins = np.array(metadata["sv_bins"])
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            xlabel_s6 = ev_label_s6
        else:  # SVD
            bin_centers = None  # Will be set per trace
            xlabel_s6 = f"{ev_label_s6} Index"

        # Create overlaid line plot
        fig_s6 = go.Figure()

        # Use distinct colors from a qualitative palette
        colors = px.colors.qualitative.Plotly
        if len(selected_steps_s6) > len(colors):
            # Cycle through colors if we have more steps than colors
            colors = colors * (len(selected_steps_s6) // len(colors) + 1)

        # Add trace for each selected step
        for idx, step in enumerate(sorted(selected_steps_s6)):
            df_step = df_filtered_s6.query(f"step == {step}")

            if len(df_step) == 0:
                continue

            dist_data = df_step.iloc[0][dist_col_s6]

            # For SVD, x-axis is index and apply transformation
            if dist_col_s6 == "SVD":
                x_data = np.arange(len(dist_data))
                y_data = to_plot_space(dist_data)
            else:
                x_data = bin_centers
                y_data = dist_data

            fig_s6.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode="lines",
                    name=f"Step {step:,}",
                    line=dict(color=colors[idx], width=2),
                    hovertemplate=(
                        f"<b>Step {step:,}</b><br>"
                        f"{xlabel_s6}: %{{x:.4f}}<br>"
                        "Value: %{y:.4e}<br>"
                        "<extra></extra>"
                    )
                )
            )

        # Layout options
        col_a, col_b = st.columns(2)
        use_log_y_s6 = col_a.checkbox("Log scale (y-axis)", key="log_y_s6")
        use_auto_xrange_s6 = col_b.checkbox("Auto x-axis range", key="auto_xrange_s6", value=False)

        # Calculate minimum y-value based on d_model
        d_model = metadata.get("d_model")
        yaxis_range = None
        if d_model is not None:
            min_prob = 0.5 / (d_model ** 2)
            if use_log_y_s6:
                yaxis_range = [np.log10(min_prob), None]
            else:
                yaxis_range = [min_prob, None]

        # Set x-axis range
        xaxis_range = None
        if not use_auto_xrange_s6:
            if dist_col_s6 == "SVD" and d_model is not None:
                xaxis_range = [0, d_model]
            elif dist_col_s6 != "SVD":
                # Fixed range for P(W) and P(λ)
                xaxis_range = [-0.15, 0.15]

        # Create a unique revision key based on layer, head, and distribution type
        # This preserves zoom when steps change but resets when layer/head/dist changes
        uirevision_key = f"s6_{layer_selected_s6}_{head_selected_s6}_{dist_type_s6}"

        fig_s6.update_layout(
            xaxis_title=xlabel_s6,
            yaxis_title="Probability Density" if dist_col_s6 != "SVD" else ev_label_s6,
            title=f"{dist_type_s6} Evolution: Layer {layer_selected_s6}, Head {head_selected_s6}",
            height=600,
            yaxis_type="log" if use_log_y_s6 else "linear",
            yaxis_range=yaxis_range,
            xaxis_range=xaxis_range,
            hovermode="x unified",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            uirevision=uirevision_key  # Preserve zoom state
        )

        st.plotly_chart(fig_s6, width="stretch")

        # Summary statistics
        with st.expander("Distribution statistics"):
            st.markdown(f"**Layer {layer_selected_s6}, Head {head_selected_s6}**")

            cols = st.columns(min(len(selected_steps_s6), 4))
            for idx, step in enumerate(sorted(selected_steps_s6)):
                df_step = df_filtered_s6.query(f"step == {step}")
                if len(df_step) > 0:
                    with cols[idx % 4]:
                        st.markdown(f"**Step {step:,}**")
                        if dist_col_s6 == "SVD":
                            # For SVD, show some basic stats
                            dist_data = to_plot_space(df_step.iloc[0][dist_col_s6])
                            st.metric(f"Max {ev_label_s6}", f"{dist_data[0]:.4f}")
                            st.metric(f"Min {ev_label_s6}", f"{dist_data[-1]:.4e}")
                        else:
                            # Use mean, sigma, entropy, KL divergence from dataframe
                            st.metric("μ (Mean)", f"{df_step.iloc[0]['mean']:.4f}")
                            st.metric("σ (Std Dev)", f"{df_step.iloc[0]['std']:.4f}")
                            st.metric("Entropy", f"{df_step.iloc[0]['entropy']:.4f}")
                            st.metric("D_KL", f"{df_step.iloc[0]['kl_vs_empirical_normal']:.4f}")

        with st.expander("Understanding this view"):
            st.markdown(f"""
            - **Different colored lines**: {dist_type_s6} distributions at different training steps
            - **X-axis**: {xlabel_s6}
            - **Y-axis**: {"Probability density" if dist_col_s6 != "SVD" else ev_label_s6 + " magnitude"}
            - **Zoom persistence**: Your zoom level is preserved when adding/removing steps

            Look for:
            - **Narrowing distributions**: Indicates regularization or convergence
            - **Shifting peaks**: Changes in typical value magnitude
            - **Changing tails**: Evolution of extreme values
            - **Multi-modal behavior**: Emergence or disappearance of multiple peaks
            """)

    # ============================================================================
    # SECTION 7: Singular Value Index Heatmap (Step vs Layer/Head)
    # ============================================================================
    if weight_selected == "W_QK":
        ev_label_s7 = "Eigenvalue" if use_eigenvalues else "Singular Value"
        st.header(f"Section 7: {ev_label_s7} Index Heatmap")
        st.markdown(
            f"Visualize how a specific {ev_label_s7.lower()} index evolves across training steps and model architecture"
        )

        # Get max SV index based on d_model
        d_model = metadata.get("d_model")
        max_sv_index = (d_model // n_heads - 1) if d_model is not None else 63

        sv_index_s7 = st.slider(
            f"{ev_label_s7} Index",
            0,
            max_sv_index,
            0,
            key="sv_index_s7",
            help=f"Select index from 0 to {max_sv_index} (d_model/n_heads - 1)"
        )

        # Prepare data for heatmap
        # Extract SV[index] for each row
        df_for_heatmap_sv = df.copy()
        df_for_heatmap_sv["sv_value"] = df_for_heatmap_sv["SVD"].apply(
            lambda svd_array: to_plot_space(svd_array)[sv_index_s7] if sv_index_s7 < len(svd_array) else np.nan
        )

        # Sort by layer and head to get consistent ordering
        df_for_heatmap_sv = df_for_heatmap_sv.sort_values(["step", "layer", "head"])

        # Create a composite index for layer/head
        df_for_heatmap_sv["layer_head_idx"] = (
            df_for_heatmap_sv["layer"] * n_heads + df_for_heatmap_sv["head"]
        )

        # Pivot the data
        heatmap_data_sv = df_for_heatmap_sv.pivot(
            index="layer_head_idx", columns="step", values="sv_value"
        )

        # Get the step values for x-axis
        step_values_s7 = heatmap_data_sv.columns.values

        # Create heatmap
        use_log_2d_sv = st.checkbox("Log scale (color)", key="log_2d_sv")

        z_data_sv = heatmap_data_sv.values
        if use_log_2d_sv:
            z_data_sv = np.log10(np.abs(z_data_sv) + 1e-10)

        fig_s7 = go.Figure(
            data=go.Heatmap(
                z=z_data_sv,
                x=step_values_s7,
                y=heatmap_data_sv.index.values,
                colorscale="Viridis",
                colorbar=dict(title="Log10 Value" if use_log_2d_sv else "Value"),
            )
        )

        # Add horizontal lines to separate layers
        if n_layers < 16:
            for layer_boundary in range(n_heads, n_layers * n_heads, n_heads):
                fig_s7.add_hline(
                    y=layer_boundary - 0.5, line=dict(color="white", width=0.5, dash="dot")
                )

        fig_s7.update_layout(
            xaxis_title="Training Step",
            yaxis_title="Layer/Head Index",
            title=f"{ev_label_s7}[{sv_index_s7}] Evolution Heatmap",
            height=600,
            xaxis=dict(
                type="log"
                if st.checkbox("Log scale (x-axis)", key="log_x_s7_heatmap")
                else "linear"
            ),
        )

        st.plotly_chart(fig_s7, width="stretch")

        # Add interpretation help
        with st.expander("Understanding the heatmap"):
            st.markdown(f"""
            - **Y-axis**: Concatenated layer/head index (0 to {n_layers * n_heads - 1})
            - Heads 0-{n_heads - 1}: Layer 0
            - Heads {n_heads}-{2 * n_heads - 1}: Layer 1
            - And so on...
            - **X-axis**: Training step (checkpoint)
            - **Color**: Value of {ev_label_s7}[{sv_index_s7}]
            - **White horizontal lines**: Layer boundaries

            Look for patterns like:
            - **Vertical bands**: Changes affecting all layers/heads at specific steps
            - **Horizontal bands**: Specific layers/heads behaving differently
            - **Gradients**: Gradual evolution patterns
            - **Layer-specific behavior**: Different layers having different {ev_label_s7.lower()} evolution
            """)

    # ============================================================================
    # SECTION 8: Singular Value Index Evolution (Multi-Layer Grid)
    # ============================================================================
    if weight_selected == "W_QK":
        ev_label_s8 = "Eigenvalue" if use_eigenvalues else "Singular Value"
        st.header(f"Section 8: {ev_label_s8} Evolution - Multi-Layer Grid")
        st.markdown(f"Track a specific {ev_label_s8.lower()} index across layers and heads over training")

        # Get max SV index based on d_model
        d_model = metadata.get("d_model")
        max_sv_index = (d_model // n_heads - 1) if d_model is not None else 63

        sv_index_s8 = st.slider(
            f"{ev_label_s8} Index",
            0,
            max_sv_index,
            0,
            key="sv_index_s8",
            help=f"Select index from 0 to {max_sv_index} (d_model/n_heads - 1)"
        )

        use_log_s8 = st.checkbox("Use log scale (x-axis)", key="log_s8")

        # Create subplot grid - one panel per layer
        n_cols_s8 = min(3, n_layers)
        n_rows_s8 = int(np.ceil(n_layers / n_cols_s8))

        from plotly.subplots import make_subplots

        fig_s8 = make_subplots(
            rows=n_rows_s8,
            cols=n_cols_s8,
            subplot_titles=[f"Layer {i}" for i in range(n_layers)],
            x_title="Training Step",
            y_title=f"{ev_label_s8}[{sv_index_s8}]",
        )

        # Use a color scale for different heads
        colors = px.colors.sample_colorscale(
            "Viridis", [i / (n_heads - 1) for i in range(n_heads)]
        )

        for layer_idx in range(n_layers):
            df_layer = df.query(f"layer == {layer_idx}")

            row = layer_idx // n_cols_s8 + 1
            col = layer_idx % n_cols_s8 + 1

            # Plot individual heads with thin lines
            for head_idx in range(n_heads):
                df_head = df_layer.query(f"head == {head_idx}").sort_values("step")

                # Extract SVD[index] for each step
                sv_values = []
                steps = []
                for _, row_data in df_head.iterrows():
                    svd_array = to_plot_space(row_data["SVD"])
                    if sv_index_s8 < len(svd_array):
                        sv_values.append(svd_array[sv_index_s8])
                        steps.append(row_data["step"])

                if len(sv_values) > 0:
                    fig_s8.add_trace(
                        go.Scatter(
                            x=steps,
                            y=sv_values,
                            mode="lines",
                            line=dict(color=colors[head_idx], width=1),
                            name=f"L{layer_idx}H{head_idx}",
                            showlegend=False,
                            opacity=0.5,
                        ),
                        row=row,
                        col=col,
                    )

            # Compute and plot layer average (mean across heads)
            layer_avg_sv = []
            layer_steps = sorted(df_layer["step"].unique())
            for step in layer_steps:
                df_step = df_layer.query(f"step == {step}")
                sv_values_at_step = []
                for _, row_data in df_step.iterrows():
                    svd_array = to_plot_space(row_data["SVD"])
                    if sv_index_s8 < len(svd_array):
                        sv_values_at_step.append(svd_array[sv_index_s8])
                if len(sv_values_at_step) > 0:
                    layer_avg_sv.append(np.mean(sv_values_at_step))
                else:
                    layer_avg_sv.append(np.nan)

            fig_s8.add_trace(
                go.Scatter(
                    x=layer_steps,
                    y=layer_avg_sv,
                    mode="lines",
                    line=dict(color="red", width=3, dash="dash"),
                    name="Layer average",
                    showlegend=(layer_idx == 0),
                ),
                row=row,
                col=col,
            )

        if use_log_s8:
            fig_s8.update_xaxes(type="log")

        fig_s8.update_layout(
            height=300 * n_rows_s8,
            title_text=f"{ev_label_s8}[{sv_index_s8}] Evolution: Individual Heads (thin) vs Layer Average (red dashed)",
        )

        st.plotly_chart(fig_s8, width="stretch")

        with st.expander("Understanding this view"):
            st.markdown(f"""
            - **Thin colored lines**: Individual head trajectories for {ev_label_s8}[{sv_index_s8}]
            - **Thick red dashed line**: Average across all heads in that layer
            - Each panel represents one layer
            - Useful for seeing:
                - How specific {ev_label_s8.lower()}s evolve during training
                - Head diversity in {ev_label_s8.lower()} structure
                - Layer-specific patterns in {ev_label_s8.lower()} evolution
            """)

    # ============================================================================
    # SECTION 9: Singular Value Index Evolution (Overlaid View)
    # ============================================================================
    if weight_selected == "W_QK":
        ev_label_s9 = "Eigenvalue" if use_eigenvalues else "Singular Value"
        st.header(f"Section 9: {ev_label_s9} Evolution - Overlaid View")
        st.markdown(f"Compare specific {ev_label_s9.lower()} indices over training for a selected layer/head")

        col1, col2 = st.columns(2)
        layer_selected_s9 = col1.slider("Layer", 0, n_layers - 1, 0, key="layer_s9")
        head_selected_s9 = col2.slider("Head", 0, n_heads - 1, 0, key="head_s9")

        # Get max SV index
        d_model = metadata.get("d_model")
        max_sv_index = (d_model // n_heads - 1) if d_model is not None else 63

        st.markdown(f"**Select {ev_label_s9.lower()} indices to overlay:**")

        # Suggest some default indices (0, middle, last)
        default_sv_indices = [0, max_sv_index // 2, max_sv_index]

        selected_sv_indices_s9 = st.multiselect(
            f"{ev_label_s9} Indices",
            options=list(range(max_sv_index + 1)),
            default=default_sv_indices,
            key="sv_indices_s9",
            help=f"Select indices from 0 to {max_sv_index}"
        )

        if not selected_sv_indices_s9:
            st.warning(f"Please select at least one {ev_label_s9.lower()} index to visualize.")
        else:
            # Filter data for selected layer/head
            df_filtered_s9 = df.query(
                f"layer == {layer_selected_s9} and head == {head_selected_s9}"
            ).sort_values("step")

            # Create overlaid line plot
            fig_s9 = go.Figure()

            # Use distinct colors
            colors = px.colors.qualitative.Plotly
            if len(selected_sv_indices_s9) > len(colors):
                colors = colors * (len(selected_sv_indices_s9) // len(colors) + 1)

            # Add trace for each selected index
            for idx, sv_index in enumerate(sorted(selected_sv_indices_s9)):
                sv_values = []
                steps = []
                for _, row_data in df_filtered_s9.iterrows():
                    svd_array = to_plot_space(row_data["SVD"])
                    if sv_index < len(svd_array):
                        sv_values.append(svd_array[sv_index])
                        steps.append(row_data["step"])

                if len(sv_values) > 0:
                    fig_s9.add_trace(
                        go.Scatter(
                            x=steps,
                            y=sv_values,
                            mode="lines+markers",
                            name=f"{ev_label_s9}[{sv_index}]",
                            line=dict(color=colors[idx], width=2),
                            marker=dict(size=4),
                            hovertemplate=(
                                f"<b>{ev_label_s9}[{sv_index}]</b><br>"
                                "Step: %{x:,}<br>"
                                "Value: %{y:.4f}<br>"
                                "<extra></extra>"
                            )
                        )
                    )

            use_log_x_s9 = st.checkbox("Log scale (x-axis)", key="log_x_s9")
            use_log_y_s9 = st.checkbox("Log scale (y-axis)", key="log_y_s9")

            # Create uirevision key to preserve zoom
            uirevision_key_s9 = f"s9_{layer_selected_s9}_{head_selected_s9}"

            fig_s9.update_layout(
                xaxis_title="Training Step",
                yaxis_title=ev_label_s9,
                title=f"{ev_label_s9} Evolution: Layer {layer_selected_s9}, Head {head_selected_s9}",
                height=600,
                xaxis_type="log" if use_log_x_s9 else "linear",
                yaxis_type="log" if use_log_y_s9 else "linear",
                hovermode="x unified",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                ),
                uirevision=uirevision_key_s9
            )

            st.plotly_chart(fig_s9, width="stretch")

            # Summary statistics
            with st.expander(f"{ev_label_s9} statistics"):
                st.markdown(f"**Layer {layer_selected_s9}, Head {head_selected_s9}**")

                # Show stats at first and last step
                first_step = df_filtered_s9.iloc[0]["step"]
                last_step = df_filtered_s9.iloc[-1]["step"]

                col_first, col_last = st.columns(2)

                with col_first:
                    st.markdown(f"**Step {first_step:,} (Initial)**")
                    svd_first = to_plot_space(df_filtered_s9.iloc[0]["SVD"])
                    for sv_idx in selected_sv_indices_s9:
                        if sv_idx < len(svd_first):
                            st.metric(f"{ev_label_s9}[{sv_idx}]", f"{svd_first[sv_idx]:.4f}")

                with col_last:
                    st.markdown(f"**Step {last_step:,} (Final)**")
                    svd_last = to_plot_space(df_filtered_s9.iloc[-1]["SVD"])
                    for sv_idx in selected_sv_indices_s9:
                        if sv_idx < len(svd_last):
                            st.metric(f"{ev_label_s9}[{sv_idx}]", f"{svd_last[sv_idx]:.4f}")

            with st.expander("Understanding this view"):
                st.markdown(f"""
                - **Different colored lines**: Different {ev_label_s9.lower()} indices
                - **X-axis**: Training step
                - **Y-axis**: {ev_label_s9} magnitude
                - **Zoom persistence**: Your zoom level is preserved when adding/removing indices

                Look for:
                - **Rank changes**: How the relative importance of different {ev_label_s9.lower()}s changes
                - **Convergence patterns**: Whether {ev_label_s9.lower()}s stabilize or continue evolving
                - **Gaps**: Large differences between consecutive {ev_label_s9.lower()}s indicating rank structure
                """)

    # Sections 10-11 (animated visualizations) moved to the Animations tab.



def render():
    step_evolution_app()


if __name__ == "__main__":
    step_evolution_app()