"""
Step Evolution Dashboard
Visualizes how statistics evolve across training checkpoints (steps)
"""
import json
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datasets import load_from_disk

# Same data path structure as weights_dashboard_app
DATA_PATH = "histos_4/pythia-70m-deduped_all_checkpoints"


@st.cache_data
def load_dataset_with_metadata(path):
    """Load dataset and associated metadata file"""
    dataset = load_from_disk(path)
    metadata_file = Path(path) / dataset.info.description
    metadata = json.load(open(metadata_file))
    return dataset.to_pandas(), metadata


@st.cache_data
def get_unique_values(df, column):
    """Get sorted unique values from a column"""
    return sorted(df[column].unique())


# Display name mappings (same as weights_dashboard_app)
stat_display = {
    "σ (Std Dev)": "std",
    "σ (fit)": "fit_sigma",
    "Entropy": "entropy",
    "μ (Mean)": "mean",
    "μ (fit)": "fit_mu",
    "sum": "sum",
    "max": "max",
    "min": "min",
    "skew": "skew",
    "kurtosis": "kurtosis",
    "D_KL(P || N(0,1)": "kl_vs_standard_normal",
    "D_KL(P || N(μ,σ)": "kl_vs_empirical_normal",
    "D_KL( N(μ,σ) || N(0,1))": "kl_normal_vs_standard",
}

st.title("Training Step Evolution Analysis")
st.markdown("Visualize how model statistics evolve across training checkpoints")

# Load data
df_full, metadata = load_dataset_with_metadata(DATA_PATH)

# Sidebar: Model and weight type selection
st.sidebar.header("Data Selection")
model_names = get_unique_values(df_full, "model")
model_selected = st.sidebar.selectbox("Model", model_names)

weight_types = get_unique_values(df_full, "weight_type")
weight_selected = st.sidebar.selectbox("Weight Type", weight_types)

# Filter by model and weight type (all steps)
df = df_full.query(f"model == '{model_selected}' and weight_type == '{weight_selected}'")

# Get architecture dimensions
n_layers = df["layer"].max() + 1
n_heads = df["head"].max() + 1

# Get available steps and sort them
steps_available = sorted(df["step"].unique())
st.sidebar.markdown(f"**Available steps:** {len(steps_available)}")
st.sidebar.markdown(f"Range: {steps_available[0]} - {steps_available[-1]}")

# ============================================================================
# SECTION 1: Single Layer/Head Evolution
# ============================================================================
st.header("Section 1: Statistic Evolution for Single Layer/Head")
st.markdown("Track how a specific statistic evolves over training steps for a chosen layer and head")

col1, col2 = st.columns(2)
layer_selected = col1.slider("Layer", 0, n_layers - 1, 0, key="layer_s1")
head_selected = col2.slider("Head", 0, n_heads - 1, 0, key="head_s1")

stat_display_name = st.selectbox(
    "Statistic", 
    list(stat_display.keys()),
    key="stat_s1"
)
stat_name = stat_display[stat_display_name]

# Filter for the specific layer/head across all steps
df_filtered = df.query(f"layer == {layer_selected} and head == {head_selected}")
df_filtered = df_filtered.sort_values("step")

# Create line plot
fig_s1 = px.line(
    df_filtered,
    x="step",
    y=stat_name,
    labels={"step": "Training Step", stat_name: stat_display_name},
    title=f"{stat_display_name} vs Training Step (Layer {layer_selected}, Head {head_selected})"
)

# Add markers for better visibility
fig_s1.update_traces(mode='lines+markers', marker=dict(size=5))

# Log scale option
use_log_s1 = st.checkbox("Use log scale (x-axis)", key="log_s1")
if use_log_s1:
    fig_s1.update_xaxes(type="log")

st.plotly_chart(fig_s1, use_container_width=True)

# Display some statistics about the evolution
col1, col2, col3, col4 = st.columns(4)
stat_values = df_filtered[stat_name].values
col1.metric("Initial Value", f"{stat_values[0]:.4f}")
col2.metric("Final Value", f"{stat_values[-1]:.4f}")
col3.metric("Change", f"{stat_values[-1] - stat_values[0]:.4f}")
col4.metric("Max Value", f"{stat_values.max():.4f}")

# ============================================================================
# SECTION 2: 2D Heatmap (Step vs Layer/Head)
# ============================================================================
st.header("Section 2: Evolution Heatmap Across Architecture")
st.markdown("Visualize how statistics evolve across both training steps and model architecture")

stat_display_name_2d = st.selectbox(
    "Statistic", 
    list(stat_display.keys()),
    key="stat_s2"
)
stat_name_2d = stat_display[stat_display_name_2d]

# Prepare data for heatmap
# Sort by layer and head to get consistent ordering
df_for_heatmap = df.sort_values(["step", "layer", "head"])

# Create pivot-like structure: rows = layer/head combinations, columns = steps
# Create a composite index for layer/head
df_for_heatmap["layer_head_idx"] = df_for_heatmap["layer"] * n_heads + df_for_heatmap["head"]

# Pivot the data
heatmap_data = df_for_heatmap.pivot(
    index="layer_head_idx",
    columns="step",
    values=stat_name_2d
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
        colorbar=dict(title="Log10 Value" if use_log_2d else "Value")
    )
)

# Add vertical lines to separate layers
for layer_boundary in range(n_heads, n_layers * n_heads, n_heads):
    fig_s2.add_hline(
        y=layer_boundary - 0.5,
        line=dict(color="white", width=1, dash="dot")
    )

fig_s2.update_layout(
    xaxis_title="Training Step",
    yaxis_title="Layer/Head Index",
    title=f"{stat_display_name_2d} Evolution Heatmap",
    height=600,
    xaxis=dict(type="log" if st.checkbox("Log scale (x-axis)", key="log_x_s2") else "linear")
)

st.plotly_chart(fig_s2, use_container_width=True)

# Add interpretation help
with st.expander("Understanding the heatmap"):
    st.markdown(f"""
    - **Y-axis**: Concatenated layer/head index (0 to {n_layers * n_heads - 1})
      - Heads 0-{n_heads-1}: Layer 0
      - Heads {n_heads}-{2*n_heads-1}: Layer 1
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
st.sidebar.markdown(f"Total rows: {len(df_full):,}")
st.sidebar.markdown(f"Filtered rows: {len(df):,}")
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
    "Statistic", 
    list(stat_display.keys()),
    key="stat_s3"
)
stat_name_s3 = stat_display[stat_display_name_s3]

# Option to choose between separate panels or same panel
view_mode = st.radio(
    "Display mode",
    ["Separate panels (subplots)", "Same panel (overlaid)"],
    key="view_mode_s3"
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
        y_title=stat_display_name_s3
    )
    
    for head_idx in range(n_heads):
        df_head = df_layer.query(f"head == {head_idx}").sort_values("step")
        
        row = head_idx // n_cols + 1
        col = head_idx % n_cols + 1
        
        fig_s3.add_trace(
            go.Scatter(
                x=df_head["step"],
                y=df_head[stat_name_s3],
                mode='lines+markers',
                marker=dict(size=3),
                name=f"Head {head_idx}",
                showlegend=False
            ),
            row=row,
            col=col
        )
    
    if use_log_s3:
        fig_s3.update_xaxes(type="log")
    
    fig_s3.update_layout(
        height=200 * n_rows,
        title_text=f"Layer {layer_selected_s3}: {stat_display_name_s3} Evolution Across Heads"
    )
    
else:  # Same panel (overlaid)
    fig_s3 = go.Figure()
    
    # Use a color scale for different heads
    colors = px.colors.sample_colorscale("Viridis", [i/(n_heads-1) for i in range(n_heads)])
    
    for head_idx in range(n_heads):
        df_head = df_layer.query(f"head == {head_idx}").sort_values("step")
        
        fig_s3.add_trace(
            go.Scatter(
                x=df_head["step"],
                y=df_head[stat_name_s3],
                mode='lines+markers',
                marker=dict(size=4),
                name=f"Head {head_idx}",
                line=dict(color=colors[head_idx])
            )
        )
    
    fig_s3.update_layout(
        xaxis_title="Training Step",
        yaxis_title=stat_display_name_s3,
        title=f"Layer {layer_selected_s3}: {stat_display_name_s3} Evolution Across All Heads",
        height=600,
        xaxis_type="log" if use_log_s3 else "linear",
        hovermode='x unified'
    )

st.plotly_chart(fig_s3, use_container_width=True)

# Summary statistics for this layer
with st.expander("Layer statistics summary"):
    st.markdown(f"**Layer {layer_selected_s3} - {stat_display_name_s3}**")
    
    # Compute statistics across heads at final step
    final_step = steps_available[-1]
    df_final = df_layer.query(f"step == {final_step}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean across heads", f"{df_final[stat_name_s3].mean():.4f}")
    col2.metric("Std across heads", f"{df_final[stat_name_s3].std():.4f}")
    col3.metric("Min head value", f"{df_final[stat_name_s3].min():.4f}")
    col4.metric("Max head value", f"{df_final[stat_name_s3].max():.4f}")

# ============================================================================
# SECTION 4: Multi-Layer Comparison with Head Averages
# ============================================================================
st.header("Section 4: Layer-by-Layer Evolution with Averages")
st.markdown("Compare individual heads and layer averages across multiple layers")

stat_display_name_s4 = st.selectbox(
    "Statistic", 
    list(stat_display.keys()),
    key="stat_s4"
)
stat_name_s4 = stat_display[stat_display_name_s4]

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
    y_title=stat_display_name_s4
)

# Use a color scale for different heads
colors = px.colors.sample_colorscale("Viridis", [i/(n_heads-1) for i in range(n_heads)])

for layer_idx in range(n_layers):
    df_layer = df.query(f"layer == {layer_idx}")
    
    row = layer_idx // n_cols_s4 + 1
    col = layer_idx % n_cols_s4 + 1
    
    # Plot individual heads with thin lines
    for head_idx in range(n_heads):
        df_head = df_layer.query(f"head == {head_idx}").sort_values("step")
        
        fig_s4.add_trace(
            go.Scatter(
                x=df_head["step"],
                y=df_head[stat_name_s4],
                mode='lines',
                line=dict(color=colors[head_idx], width=1),
                name=f"L{layer_idx}H{head_idx}",
                showlegend=False,
                opacity=0.5
            ),
            row=row,
            col=col
        )
    
    # Compute and plot layer average (mean across heads)
    df_layer_avg = df_layer.groupby("step")[stat_name_s4].mean().reset_index()
    df_layer_avg = df_layer_avg.sort_values("step")
    
    fig_s4.add_trace(
        go.Scatter(
            x=df_layer_avg["step"],
            y=df_layer_avg[stat_name_s4],
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name=f"Layer {layer_idx} avg",
            showlegend=(layer_idx == 0)  # Only show legend for first layer
        ),
        row=row,
        col=col
    )

if use_log_s4:
    fig_s4.update_xaxes(type="log")

fig_s4.update_layout(
    height=300 * n_rows_s4,
    title_text=f"{stat_display_name_s4} Evolution: Individual Heads (thin) vs Layer Average (red dashed)"
)

st.plotly_chart(fig_s4, use_container_width=True)

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