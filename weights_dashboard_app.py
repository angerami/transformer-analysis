import streamlit as st
from histogram_utils import load_group_from_file, normality_metrics, stats_config_standard, extract_metrics_
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# Display name mappings
model_display = {
    'gpt2-small (124M)': 'small',
    'gpt2-medium (355M)': 'medium', 
    'gpt2-large (774M)': 'large',
    'gpt2-xl (1558M)': 'xl'
}

plot_display = {
    'P(W)': 'h',
    'P(λ)': 'SVD_prob',
    'SVD': 'SVD'
}
stat_display = {
    'σ (Std Dev)': 'std',
    'μ (Mean)': 'mean',
    'sum' : 'sum',
    'max' : 'max', 'min' : 'min',
    'skew' : 'skew', 'kurtosis' : 'kurtosis'
}

@st.cache_data
def load_weight_data(model_name, weight_name):
    return load_group_from_file()

st.title("Transformer Weight Analysis")

# Sidebar
model_display_name = st.sidebar.selectbox("Model", list(model_display.keys()))
model_name = model_display[model_display_name]
weight_name = st.sidebar.selectbox("Weight", ["W_q", "W_k", "W_v", "W_QK"])

# Load data
fname = f"parquet/{model_name}.{weight_name}.parquet"
df = pd.read_parquet(fname)

n_layers = max(df['layer'])
n_heads = max(df['head'])
bins = df.attrs['bins'] 
available_plots = [h for h in df.attrs['histos'] if h != 'bins']
# Section 1: Single head analysis
st.header("Distribution Analysis")
col1, col2 = st.columns(2)
layer = col1.slider("Layer", 0, n_layers-1, 0)
head = col2.slider("Head", 0, n_heads-1, 0)
plot_type = st.selectbox("Plot", available_plots)

entry = df.query(f'layer == {layer} and head == {head}')
h = entry[plot_type]
h_centers = [ 0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]

# Histogram plot
use_log_1 = st.checkbox("Log scale", key='log_1')
show_fit = st.checkbox("Show Gaussian fit", key='fit_1')

fig = go.Figure()
y_min = np.min(h)
y_vals = np.where(h > 0, np.log10(h), 0.5*y_min) if use_log_1 else h
fig.add_trace(go.Bar(x=h_centers, y=y_vals, name=plot_type))

if show_fit and 'fit_normal' in hb.stats_values:
    mu = entry['fit_mu'].iloc[0]
    sigma = entry['fit_sigma'].iloc[0]
    # Gaussian curve
    from scipy.stats import norm
    fit_curve = norm.pdf(h_centers, mu, sigma)
    # Scale to match histogram
    fit_curve *= np.sum(h) * (h_centers[1] - h_centers[0])
    fig.add_trace(go.Scatter(x=h_centers, y=fit_curve, 
                             mode='lines', name='Fit',
                             line=dict(color='red', width=2)))

fig.update_layout(xaxis_title="Value", yaxis_title="Count")
st.plotly_chart(fig, use_container_width=True)

# Statistics
st.subheader("Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Mean", f"{entry['mean'].iloc[0]:.4f}")
col2.metric("Std", f"{entry['std'].iloc[0]:.4f}")
col3.metric("Max", f"{entry['max'].iloc[0]:.4f}")

# Section 2: Across layers/heads
st.header("Statistics Across Architecture")
stat_display_name = st.selectbox("Statistic", list(stat_display.keys()))
stat_name = stat_display[stat_display_name]

# # Compute stat array
df_sorted = df.sort_values(['layer', 'head'])
stats = df_sorted[stat_name].values

# # 1D plot
fig = px.line(y=stats.flatten(), labels={'y': stat_display_name, 'index': 'Head Index'})
st.plotly_chart(fig, use_container_width=True, key='section1_plot')

# 2D heatmap
fig = px.imshow(stats, labels=dict(x="Head", y="Layer", color=stat_display_name),
                aspect="auto", color_continuous_scale='Viridis')
st.plotly_chart(fig, use_container_width=True)

# Section 3: 2D Probability Distribution Stack
st.header("Stacked Probability Distributions")

plot_display_name_2d = st.selectbox("Distribution", list(available_plots.keys()), key='2d_plot')
plot_type_2d = available_plots[plot_display_name_2d]

# Stack histograms into 2D array
n_bins = len(hb.get_bin_centers())
prob_stack = np.zeros((n_layers * n_heads, n_bins))

for (l, h), val in hgroup.items():
    idx = l * n_heads + h
    prob_stack[idx, :] = val.histograms[plot_type_2d]

prob_min = np.min(prob_stack)

use_log_2d = st.checkbox("Log scale", key='log_2d')

fig = go.Figure(data=go.Heatmap(
    z=np.where(prob_stack > 0, np.log10(prob_stack), prob_min) if use_log_2d else prob_stack,

    x=hb.get_bin_centers(),
    y=np.arange(n_layers * n_heads),
    colorscale='Viridis'
))

fig.update_layout(
    xaxis_title="Weight Value",
    yaxis_title="Layer/Head Index",
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# Section 4: Per-Layer Head Grid
st.header("Distribution Grid by Layer")

col1, col2 = st.columns(2)
layer_grid = col1.selectbox("Layer", range(n_layers), key='grid_layer')
plot_display_name_grid = col2.selectbox("Distribution", list(available_plots.keys()), key='grid_plot')
plot_type_grid = available_plots[plot_display_name_grid]
show_fit_grid = st.checkbox("Show Gaussian fits", key='fit_grid')
use_log_grid = st.checkbox("Log scale", key='log_grid')
# Create subplot grid
from plotly.subplots import make_subplots

n_cols = 4
n_rows = int(np.ceil(n_heads / n_cols))

fig = make_subplots(
    rows=n_rows, 
    cols=n_cols,
    subplot_titles=[f'Head {i}' for i in range(n_heads)]
)
for head in range(n_heads):
    
    hb = hgroup[layer_grid, head]
    h = hb.histograms[plot_type_grid]
    h_centers = hb.get_bin_centers()
    
    row = head // n_cols + 1
    col = head % n_cols + 1
    pmin = 1./ (hb.n_entries * hb.bin_dx)
    if show_fit_grid and 'fit_normal' in hb.stats_values:
        from scipy.stats import norm
        mu = 0#hb.stats_values['fit_mu']
        sigma = 1#hb.stats_values['fit_sigma']
        
        fit_curve = norm.pdf(h_centers, mu, sigma)
        fit_curve *= np.sum(h) * (h_centers[1] - h_centers[0])
        
        y_vals_fit = np.where(fit_curve > 0, np.log10(fit_curve), pmin) if use_log_grid else fit_curve
        fig.add_trace(
            go.Scatter(x=h_centers, y=y_vals_fit, mode='lines',
                      line=dict(color='red', width=1), showlegend=False),
            row=row, col=col
        )
    y_vals = np.where(h > 0, np.log10(h), pmin) if use_log_grid else h
    fig.add_trace(
        go.Bar(x=h_centers, y=y_vals, name=f'Head {head}', showlegend=False),
        row=row, col=col
    )


fig.update_layout(
    height=200 * n_rows,
    title=f"Layer {layer_grid} - {plot_display_name_grid}"
)

st.plotly_chart(fig, use_container_width=True, key='section1_sv')