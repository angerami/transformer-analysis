import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datasets import load_from_disk
import json
from pathlib import Path


def load_dataset_with_metadata(path):
    dataset = load_from_disk(path)
    metadata_file = Path(path) / dataset.info.description
    metadata = json.load(open(metadata_file))
    return dataset, metadata

@st.cache_data
def load_data(model, weight_type):
    dataset, metadata = load_dataset_with_metadata(f'histos/{model}')
    # print(dataset.unique('model'))
    # print(dataset.unique('weight_type'))
    df = dataset.filter(lambda x: x['model'] == model and x['weight_type'] == weight_type).to_pandas()
    return df, metadata

plot_display = {
    'P(W)': 'h',
    'P(λ)': 'SVD_prob',
    'SVD': 'SVD'
}
stat_display = {
    'σ (Std Dev)': 'std',
    'σ (fit)' : 'fit_sigma',
    'Entropy' : 'entropy',
    'μ (Mean)': 'mean',
    'μ (fit)':  'fit_mu',
    'sum' : 'sum',
    'max' : 'max', 'min' : 'min',
    'skew' : 'skew', 'kurtosis' : 'kurtosis',
    'D_KL(P || N(0,1)' : 'kl_vs_standard_normal',
    'D_KL(P || N(μ,σ)' : 'kl_vs_empirical_normal',
    'D_KL( N(μ,σ) || N(0,1))' : 'kl_normal_vs_standard'
}

st.title("Transformer Weight Analysis")
# Sidebar
model_display_name = st.sidebar.selectbox("Model", ["pythia-70m-deduped","pythia-2.8b-deduped"])
model_name = model_display_name#model_display[model_display_name]
weight_name = st.sidebar.selectbox("Weight", ["W_Q", "W_K", "W_V", "W_QK"])

df, metadata = load_data(model_name, weight_name)

bins = metadata['w_bins']
sv_bins = metadata['sv_bins']
available_plots = [ "P_W" ]
if weight_name == "W_QK":
    available_plots.extend(['P_l','SVD'])
n_layers = max(df['layer'])
n_heads = max(df['head'])
st.header("Distribution Analysis")
col1, col2 = st.columns(2)
layer = col1.slider("Layer", 0, n_layers - 1, 0)
head = col2.slider("Head", 0, n_heads - 1, 0)
plot_type = st.selectbox("Plot", available_plots)

entry = df.query(f'layer == {layer} and head == {head}')
print(entry.columns)
print(available_plots)
h = entry[plot_type].iloc[0]
h_centers = [ 0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]
use_log_1 = st.checkbox("Log scale", key='log_1')
show_fit = st.checkbox("Show Gaussian fit", key='fit_1')

fig = go.Figure()
y_min = np.min(np.abs(h))*0.5
dist_centers = h_centers
if plot_type == 'SVD':
    dist_centers = np.arange(len(h))
if plot_type == 'P_l':
    dist_centers = [ 0.5 * (sv_bins[i] + sv_bins[i + 1]) for i in range(len(sv_bins) - 1)]
y_vals = np.log10(np.maximum(h, y_min)) if use_log_1 else h
fig.add_trace(go.Bar(x=dist_centers, y=y_vals, name=plot_type))
xtitle, ytitle = 'Weight', 'Probability'
if plot_type == 'SVD':
    xtitle, ytitle = 'Index', 'Singular Value'


fig.update_layout(xaxis_title=xtitle, yaxis_title=ytitle)


if show_fit:
    mu = entry['fit_mu'].iloc[0]
    sigma = entry['fit_sigma'].iloc[0]
    st.write(f"mu = {mu}, sigma={sigma}")
    # Gaussian curve
    from scipy.stats import norm
    fit_curve = norm.pdf(h_centers, mu, sigma)
    # Scale to match histogram
    fit_curve *= np.sum(h) * (h_centers[1] - h_centers[0])
    fig.add_trace(go.Scatter(x=h_centers, y=fit_curve, 
                             mode='lines', name='Fit',
                             line=dict(color='red', width=2)))

st.plotly_chart(fig, width='stretch')


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
df_sorted = df.sort_values(['layer', 'head'])
stats = df_sorted[stat_name].values

# # 1D plot
fig = px.line(y=stats.flatten(), labels={'y': stat_display_name, 'index': 'Head Index'})
for xpos in range(n_heads, n_layers * n_heads, n_heads):
    fig.add_shape(
        type="line",
        x0=xpos, x1=xpos,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="lightgray", width=1, dash="dot")
    )
st.plotly_chart(fig, width='stretch', key='section1_plot')
# Section 3: 2D Probability Distribution Stack
st.header("Stacked Probability Distributions")

plot_type_2d = st.selectbox("Distribution", available_plots, key='2d_plot')
# Stack histograms into 2D array
n_bins = len(bins) - 1
prob_stack = np.array([row[plot_type_2d] for _, row in df_sorted.iterrows()])
prob_min = np.min(prob_stack)

use_log_2d = st.checkbox("Log scale", key='log_2d')

fig = go.Figure(data=go.Heatmap(
    z=np.where(prob_stack > 0, np.log10(prob_stack), prob_min) if use_log_2d else prob_stack,
    x=h_centers,
    y=np.arange(n_layers * n_heads),
    colorscale='Viridis'
))

fig.update_layout(
    xaxis_title="Weight Value",
    yaxis_title="Layer/Head Index",
    height=600
)
st.plotly_chart(fig, width='stretch')
# Section 4: Per-Layer Head Grid
st.header("Distribution Grid by Layer")

col1, col2 = st.columns(2)
layer_grid = col1.selectbox("Layer", range(n_layers), key='grid_layer')
plot_type_grid = col2.selectbox("Distribution", available_plots, key='grid_plot')
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
    
    h = df.query(f'layer == {layer_grid} and head == {head}')[plot_type_grid].iloc[0]
    
    row = head // n_cols + 1
    col = head % n_cols + 1
    pmin = 1e-5
    dist_centers = h_centers
    if plot_type_grid == 'SVD':
        dist_centers = np.arange(len(h))
        pmin = 1e-1
    if show_fit_grid and plot_type_grid != 'SVD':
        from scipy.stats import norm
        mu = 0#hb.stats_values['fit_mu']
        sigma = 1#hb.stats_values['fit_sigma']
        
        fit_curve = norm.pdf(h_centers, mu, sigma)
        fit_curve *= np.sum(h) * (h_centers[1] - h_centers[0])
        
        y_vals_fit = np.where(fit_curve > 0, np.log10(fit_curve), pmin) if use_log_grid else fit_curve
        fig.add_trace(
            go.Scatter(x=dist_centers, y=y_vals_fit, mode='lines',
                      line=dict(color='red', width=1), showlegend=False),
            row=row, col=col
        )
    y_vals = np.where(h > 0, np.log10(h), 1) if use_log_grid else h
    fig.add_trace(
        go.Bar(x=dist_centers, y=y_vals, name=f'Head {head}', showlegend=False),
        row=row, col=col
    )


fig.update_layout(
    height=200 * n_rows,
    title=f"Layer {layer_grid} - {plot_type_grid}"
)

st.plotly_chart(fig, width='stretch', key='section1_sv')
