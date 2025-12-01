import json
from pathlib import Path
import subprocess
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datasets import load_from_disk

def model_size_from_name(ds_name: str) -> float:
    """Extract model size for sorting (in millions of parameters)."""
    import re
    
    # Extract size like "70m", "1.4b", "12b"
    match = re.search(r'(\d+\.?\d*)([mb])', ds_name.lower())
    if not match:
        return 0
    
    size, unit = match.groups()
    size = float(size)
    
    # Convert to millions for consistent comparison
    if unit == 'b':
        size *= 1000
    
    return size

def ensure_offline_available(path: Path):
    """Pin files for offline access via Google Drive."""
    real_path = path.resolve()
    
    try:
        subprocess.run(
            ["find", str(real_path), "-type", "f", 
             "-exec", "xattr", "-w", "com.google.drivefs.pinned", "true", "{}", ";"],
            check=True,
            capture_output=True,
            timeout=30
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        st.warning(f"Could not pin files: {e}")
        return False

def get_available_datasets(campaign: str = "step-analysis_001") -> list[str]:
    """Scan Drive for available datasets matching pattern."""
    drive_path = Path("Drive") / campaign
    
    if not drive_path.exists():
        return []
    
    datasets = []
    for item in drive_path.iterdir():
        if item.is_dir() and item.name.endswith("_all_checkpoints"):
            # Extract DS_NAME by removing suffix
            ds_name = item.name.replace("_all_checkpoints", "")
            datasets.append(ds_name)
    return sorted(datasets, key=model_size_from_name)

@st.cache_data
def load_dataset_with_metadata(ds_name: str, campaign: str = "step-analysis_001"):
    """Load dataset after ensuring offline availability."""
    dataset_path = Path("Drive") / campaign / f"{ds_name}_all_checkpoints"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Ensure files are downloaded
    with st.spinner("Ensuring files are available offline..."):
        ensure_offline_available(dataset_path)
    
    # Load dataset
    with st.spinner("Loading dataset..."):
        df = load_from_disk(str(dataset_path))
    # Load metdata
    metadata_path = dataset_path / df.info.description
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    return df.to_pandas(), metadata

#@st.cache_data
def get_unique_values(_df, column):
    """Get sorted unique values from a column"""
    return sorted(_df[column].unique())


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
    "D_KL(P || N(μ,σ)": "kl_vs_empirical_normal",
}
plot_display = {"P(W)": "P_w", "P(λ)": "P_sv", "SVD": "SVD"}
st.title("Training Step Evolution Analysis")



# Sidebar: Model and weight type selection
st.sidebar.header("Dataset Selection")

# Campaign selector (if you have multiple campaigns)
campaign = st.sidebar.selectbox(
    "Campaign",
    ["step-analysis_001","ana-002"]  # Add more as needed
)

# Get available datasets
available_datasets = get_available_datasets(campaign)

if not available_datasets:
    st.error(f"No datasets found in Drive/{campaign}/")
    st.stop()

# Dataset dropdown
ds_name = st.sidebar.selectbox(
    "Dataset",
    available_datasets,
    index=0 if "pythia-1.4b-deduped" in available_datasets else 0
)

# Load data
df_full, metadata = load_dataset_with_metadata(ds_name, campaign)
st.success(f"Loaded: {ds_name}")


model_names = get_unique_values(df_full, "model")
# model_selected = st.sidebar.selectbox("Model", model_names)
model_selected = model_names[0]

weight_types = get_unique_values(df_full, "weight_type")
weight_selected = st.sidebar.selectbox("Weight Type", weight_types)
query_str = f"model == '{model_selected}' and weight_type == '{weight_selected}'"
if 'step' in df_full.columns:
    n_steps = df_full["step"].max()
    query_str += f" and step == {n_steps}"
df = df_full.query(query_str)

# Get architecture dimensions
n_layers = df["layer"].max() + 1
n_heads = df["head"].max() + 1

########
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.markdown(f"Layers: {n_layers}")
st.sidebar.markdown(f"Heads per layer: {n_heads}")


st.title("Transformer Weight Analysis")
# Sidebar
###
# Load once
st.header("Distribution Analysis")

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
#show_fit = st.checkbox("Show Gaussian fit", key="fit_1")

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
y_min = np.min(np.abs(h)) * 0.5
dist_centers = h_centers
if plot_type == "SVD":
    dist_centers = np.arange(len(h))
y_vals = np.log10(np.maximum(h, y_min)) if use_log_1 else h
fig.add_trace(go.Bar(x=dist_centers, y=y_vals, name=plot_type))

fig.update_layout(xaxis_title=xtitle, yaxis_title=ytitle)


# if show_fit:
#     mu = entry["fit_mu"].iloc[0]
#     sigma = entry["fit_sigma"].iloc[0]
#     st.write(f"mu = {mu}, sigma={sigma}")
#     # Gaussian curve
#     from scipy.stats import norm

#     fit_curve = norm.pdf(h_centers, mu, sigma)
#     # Scale to match histogram
#     fit_curve *= np.sum(h) * (h_centers[1] - h_centers[0])
#     fig.add_trace(
#         go.Scatter(
#             x=h_centers,
#             y=fit_curve,
#             mode="lines",
#             name="Fit",
#             line=dict(color="red", width=2),
#         )
#     )

st.plotly_chart(fig, width="stretch")


# Statistics
st.subheader("Statistics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Standard Deviation", f"{entry['std'].iloc[0]:.4f}")
col2.metric("$\sigma$ (Gaussian fit)", f"{entry['fit_sigma'].iloc[0]:.4f}")
col3.metric("Entropy", f"{entry['entropy'].iloc[0]:.4f}")
col4.metric("Max", f"{entry['max'].iloc[0]:.4f}")
col5.metric("Min", f"{entry['min'].iloc[0]:.4f}")

# Section 2: Across layers/heads
st.header("Statistics Across Architecture")
stat_display_name = st.selectbox("Statistic", list(stat_display.keys()))
stat_name = stat_display[stat_display_name]

# # Compute stat array
df_sorted = df.sort_values(["layer", "head"])
stats = df_sorted[stat_name].values


# # 1D plot
fig = px.line(y=stats.flatten(), labels={"y": stat_display_name, "index": "Head Index"})
fig.update_xaxes(
    tickmode='array',
    tickvals=[i * n_heads for i in range(n_layers)],
    ticktext=[str(i) for i in range(n_layers)],
    title="Layer"
)
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
st.plotly_chart(fig, width="stretch", key="section1_plot")
# Section 3: 2D Probability Distribution Stack
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
prob_min = np.min(prob_stack)

use_log_2d = st.checkbox("Log scale", key="log_2d")

fig = go.Figure(
    data=go.Heatmap(
        z=np.where(prob_stack > 0, np.log10(prob_stack), prob_min)
        if use_log_2d
        else prob_stack,
        x=h_centers,
        y=np.arange(n_layers * n_heads),
        colorscale="Viridis",
    )
)
fig.update_yaxes(
    tickmode='array',
    tickvals=[i * n_heads for i in range(n_layers)],
    ticktext=[str(i) for i in range(n_layers)],
    title="Layer"
)
fig.update_layout(xaxis_title=xtitle_2d, yaxis_title=ytitle_2d, height=600)
st.plotly_chart(fig, width="stretch")
# Section 4: Per-Layer Head Grid
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
    h = df.query(f"layer == {layer_grid} and head == {head}")[plot_type_grid_name].iloc[
        0
    ]

    row = head // n_cols + 1
    col = head % n_cols + 1
    pmin = 1e-5
    dist_centers = h_centers
    if plot_type_grid == "SVD":
        dist_centers = np.arange(len(h))
        pmin = 1e-1
    if show_fit_grid and plot_type_grid != "SVD":
        from scipy.stats import norm

        mu = 0  # hb.stats_values['fit_mu']
        sigma = 1  # hb.stats_values['fit_sigma']

        fit_curve = norm.pdf(h_centers, mu, sigma)
        fit_curve *= np.sum(h) * (h_centers[1] - h_centers[0])

        y_vals_fit = (
            np.where(fit_curve > 0, np.log10(fit_curve), pmin)
            if use_log_grid
            else fit_curve
        )
        fig.add_trace(
            go.Scatter(
                x=dist_centers,
                y=y_vals_fit,
                mode="lines",
                line=dict(color="red", width=1),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    y_vals = np.where(h > 0, np.log10(h), 1e-1) if use_log_grid else h
    fig.add_trace(
        go.Bar(x=dist_centers, y=y_vals, name=f"Head {head}", showlegend=False),
        row=row,
        col=col,
    )


fig.update_layout(height=200 * n_rows, title=f"Layer {layer_grid} - {plot_type_grid}")

st.plotly_chart(fig, width="stretch", key="section1_sv")
