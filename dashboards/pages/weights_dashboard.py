import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy import stats as scipy_stats
from plotly.subplots import make_subplots
from dashboard_utils import (
    stat_display,
    get_available_campaigns,
    load_dataset_with_metadata,
    get_unique_values,
    is_HF_environment,
    model_size_from_name,
)


def weights_dashboard_app():
    plot_display = {"P(W)": "P_w", "P(λ)": "P_sv", "SVD": "SVD"}
    merge_key = 'merged'

    if is_HF_environment():
        df_full, metadata = load_dataset_with_metadata(
            ds_name=None, campaign=None,
            hf_repo_id="angerami/transformer_weights_cross_model"
        )
    else:
        available_datasets = get_available_campaigns("ana-")
        if not available_datasets:
            st.error("No datasets found.")
            st.stop()
        campaign_name = st.sidebar.selectbox("Campaign", available_datasets, index=0)
        df_full, metadata = load_dataset_with_metadata(
            ds_name="weight_study", campaign=campaign_name, hf_version="ana-003"
        )

    model_names = sorted(
        get_unique_values(df_full, "model"),
        key=lambda x: (x.split('-')[0], model_size_from_name(x))
    )
    model_selected = st.sidebar.selectbox("Model", model_names)
    if merge_key in metadata and model_selected in metadata[merge_key]:
        metadata = metadata[merge_key][model_selected]

    weight_types = get_unique_values(df_full, "weight_type")
    default_wt_idx = weight_types.index("W_QK") if "W_QK" in weight_types else 0
    weight_selected = st.sidebar.selectbox("Weight Type", weight_types, index=default_wt_idx)

    df = df_full.query(
        f"model == '{model_selected}' and weight_type == '{weight_selected}'"
    )
    d_model = metadata["d_model"]
    n_layers = df["layer"].max() + 1
    n_heads = df["head"].max() + 1
    df_sorted = df.sort_values(["layer", "head"])

    st.title("Weight Dashboard")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Info")
    st.sidebar.markdown(f"Model Dimension: {d_model}")
    st.sidebar.markdown(f"Heads: {n_heads}")
    st.sidebar.markdown(f"Layers: {n_layers}")

    st.sidebar.markdown("---")
    use_eigenvalues = st.sidebar.checkbox(
        "Plot eigenvalues (λ²)",
        value=True,
        help="Square singular values to show eigenvalues.",
        key="weights_use_eigenvalues"
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
            pr = (sum_sv**2) / sum_sv2 if sum_sv2 > 0 else 0
            return pr / d_head if d_head > 0 else 0
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

    ev_label = "Eigenvalue" if use_eigenvalues else "Singular Value"
    xtitle, ytitle = "Weight", "Probability"
    bins = metadata["w_bins"]

    if plot_type_name == "P_sv":
        d_head = d_model // n_heads
        svd_raw = np.array(entry["SVD"].iloc[0])[:d_head]
        plot_vals = to_plot_space(svd_raw)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=plot_vals, nbinsx=40, histnorm="probability density",
            name=f"P({ev_label})"
        ))
        if use_log_1:
            fig.update_yaxes(type="log")
        fig.update_layout(xaxis_title=ev_label, yaxis_title="Density")
    elif plot_type_name == "SVD":
        plot_vals = to_plot_space(h)
        nsvs = len(plot_vals)
        dist_centers = np.arange(nsvs)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dist_centers, y=plot_vals, name=plot_type))
        if use_log_1:
            fig.update_yaxes(type="log")
        max_sv_index = d_model // n_heads - 1
        fig.update_layout(
            xaxis_title="Index", yaxis_title=ev_label,
            xaxis_range=[0, max_sv_index]
        )
    else:
        h_centers = [0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=h_centers, y=h, name=plot_type))
        if use_log_1:
            fig.update_yaxes(type="log")
        fig.update_layout(xaxis_title=xtitle, yaxis_title=ytitle)

    if show_fit and plot_type == "P(W)":
        mu = entry["fit_mu"].iloc[0]
        sigma = entry["fit_sigma"].iloc[0]
        st.write(f"μ = {mu:.4f}, σ ={sigma:.4f}")
        from scipy.stats import norm
        fit_curve = norm.pdf(h_centers, mu, sigma)
        fit_curve *= np.sum(h) * (h_centers[1] - h_centers[0])
        fig.add_trace(
            go.Scatter(
                x=h_centers,
                y=fit_curve,
                mode="lines",
                name="Fit",
                line=dict(color="red", width=2),
            )
        )

    st.plotly_chart(fig, width="content")

    st.subheader("Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Standard Deviation", f"{entry['std'].iloc[0]:.4f}")
    col2.metric("$\sigma$ (Gaussian fit)", f"{entry['fit_sigma'].iloc[0]:.4f}")
    col3.metric("Entropy", f"{entry['entropy'].iloc[0]:.4f}")
    col4.metric("Max", f"{entry['max'].iloc[0]:.4f}")
    col5.metric("Min", f"{entry['min'].iloc[0]:.4f}")

    ########################################################################
    # Section 2: Stacked Probability Distributions
    ########################################################################

    st.header("Section 2: Stacked Probability Distributions")

    plot_type_2d = st.selectbox("Distribution", available_plots, key="2d_plot")
    plot_type_name_2d = plot_display[plot_type_2d]
    ev_label_2d = "Eigenvalue" if use_eigenvalues else "Singular Value"
    d_head = d_model // n_heads
    use_log_2d = st.checkbox("Log scale", key="log_2d")

    if plot_type_name_2d == "SVD":
        prob_stack = np.array([to_plot_space(row["SVD"])[:d_head]
                               for _, row in df_sorted.iterrows()])
        h_centers_2d = np.arange(d_head)
        xtitle_2d = "Index"
    elif plot_type_name_2d == "P_sv":
        all_vals = np.concatenate([to_plot_space(row["SVD"])[:d_head]
                                   for _, row in df_sorted.iterrows()])
        eig_max = np.percentile(all_vals[all_vals > 0], 99.5)
        n_bins_2d = 80
        bin_edges_2d = np.linspace(0, eig_max, n_bins_2d + 1)
        h_centers_2d = 0.5 * (bin_edges_2d[:-1] + bin_edges_2d[1:])
        prob_stack = np.zeros((len(df_sorted), n_bins_2d))
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            vals = to_plot_space(row["SVD"])[:d_head]
            h_row, _ = np.histogram(vals, bins=bin_edges_2d, density=True)
            prob_stack[i] = h_row
        xtitle_2d = ev_label_2d
    else:
        bins = metadata["w_bins"]
        h_centers_2d = [0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]
        prob_stack = np.array([row[plot_type_name_2d] for _, row in df_sorted.iterrows()])
        xtitle_2d = "Weight"

    prob_floor = prob_stack[prob_stack > 0].min() * 0.5 if np.any(prob_stack > 0) else 1e-10
    prob_plot = np.maximum(prob_stack, prob_floor)

    fig = go.Figure(
        data=go.Heatmap(
            z=np.log10(prob_plot) if use_log_2d else prob_stack,
            x=h_centers_2d,
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
        fig.update_xaxes(range=[0, d_head - 1])
    fig.update_layout(xaxis_title=xtitle_2d, yaxis_title="Layer", height=600)
    st.plotly_chart(fig, width="content")

    ########################################################################
    # Section 3: Distribution Grid by Layer
    ########################################################################

    st.header("Section 3: Distribution Grid by Layer")

    col1, col2 = st.columns(2)
    layer_grid = col1.selectbox("Layer", range(n_layers), key="grid_layer")
    plot_type_grid = col2.selectbox("Distribution", available_plots, key="grid_plot")
    plot_type_grid_name = plot_display[plot_type_grid]
    show_fit_grid = st.checkbox("Show Gaussian fits", key="fit_grid")
    use_log_grid = st.checkbox("Log scale", key="log_grid")

    n_cols = 4
    n_rows = int(np.ceil(n_heads / n_cols))

    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=[f"Head {i}" for i in range(n_heads)]
    )
    d_head_grid = d_model // n_heads
    for head in range(n_heads):
        df_h = df.query(f"layer == {layer_grid} and head == {head}")
        h = df_h[plot_type_grid_name].iloc[0]

        row = head // n_cols + 1
        col = head % n_cols + 1

        if plot_type_grid == "SVD":
            plot_vals = to_plot_space(h)[:d_head_grid]
            dist_centers = np.arange(len(plot_vals))
            y_vals = plot_vals
        elif plot_type_grid == "P(λ)":
            svd_raw = np.array(df_h["SVD"].iloc[0])[:d_head_grid]
            plot_vals = to_plot_space(svd_raw)
            fig.add_trace(
                go.Histogram(x=plot_vals, nbinsx=20, histnorm="probability density",
                             name=f"Head {head}", showlegend=False),
                row=row, col=col,
            )
            if use_log_grid:
                fig.update_yaxes(type="log", row=row, col=col)
            continue
        else:
            w_bins = metadata["w_bins"]
            dist_centers = [0.5 * (w_bins[i] + w_bins[i + 1]) for i in range(len(w_bins) - 1)]
            y_vals = h

        if show_fit_grid and plot_type_grid == "P(W)":
            from scipy.stats import norm
            mu = df_h["fit_mu"].iloc[0]
            sigma = df_h["fit_sigma"].iloc[0]
            w_bins = metadata["w_bins"]
            h_centers_fit = [0.5 * (w_bins[i] + w_bins[i + 1]) for i in range(len(w_bins) - 1)]
            fit_curve = norm.pdf(h_centers_fit, mu, sigma)
            fit_curve *= np.sum(h) * (h_centers_fit[1] - h_centers_fit[0])
            fig.add_trace(
                go.Scatter(x=dist_centers, y=fit_curve, mode="lines",
                           line=dict(color="red", width=1), showlegend=False),
                row=row, col=col,
            )

        fig.add_trace(
            go.Bar(x=dist_centers, y=y_vals, name=f"Head {head}", showlegend=False),
            row=row, col=col,
        )
        if use_log_grid:
            fig.update_yaxes(type="log", row=row, col=col)

    if plot_type_grid == "SVD":
        fig.update_xaxes(range=[0, d_head_grid - 1])

    fig.update_layout(
        height=200 * n_rows, title=f"Layer {layer_grid} - {plot_type_grid}"
    )
    st.plotly_chart(fig, width="stretch", key="section3_grid")

    ########################################################################
    # Section 4: Statistics Across Architecture
    ########################################################################

    st.header("Section 4: Statistics Across Architecture")

    sv_options = {}
    for display_name, col_name in stat_display.items():
        sv_options[display_name] = ("stat", col_name)

    if weight_selected == "W_QK":
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
        sv_options["Max SV"] = ("derived", "leading_sv")

    selected_sv_stats = st.multiselect(
        "Select Statistics to Plot",
        options=list(sv_options.keys()),
        default=[list(sv_options.keys())[0]],
        key="sv_stats_select"
    )

    if not selected_sv_stats:
        st.warning("Please select at least one statistic")
    else:
        use_separate_axes_sv = st.checkbox(
            "Use separate Y-axes for each statistic",
            value=False,
            help="Each statistic gets its own Y-axis with matching colors",
            key="separate_axes_sv"
        )
        show_layer_avg_sv = st.checkbox(
            "Show layer averages",
            value=False,
            help="Display the mean value across all heads in each layer",
            key="layer_avg_sv"
        )

        colors_sv = px.colors.qualitative.Plotly[:len(selected_sv_stats)]

        def get_values(option_name):
            option_type, option_data = sv_options[option_name]
            values = []
            for _, row in df_sorted.iterrows():
                if option_type == "stat":
                    values.append(row[option_data])
                elif option_type == "derived":
                    values.append(compute_derived_sv_stat(row["SVD"], option_data))
            return np.array(values)

        def get_layer_avg_values(option_name):
            option_type, option_data = sv_options[option_name]
            avg_values = []
            for layer_idx in range(n_layers):
                df_layer = df.query(f"layer == {layer_idx}")
                layer_values = []
                for _, row in df_layer.iterrows():
                    if option_type == "stat":
                        layer_values.append(row[option_data])
                    elif option_type == "derived":
                        layer_values.append(compute_derived_sv_stat(row["SVD"], option_data))
                avg_values.extend([np.mean(layer_values)] * n_heads)
            return avg_values

        if use_separate_axes_sv:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            for idx, option_name in enumerate(selected_sv_stats):
                stats = get_values(option_name)
                use_secondary = idx > 0
                fig.add_trace(
                    go.Scatter(
                        y=stats, mode="lines", name=option_name,
                        line=dict(color=colors_sv[idx]),
                        yaxis="y2" if use_secondary else "y",
                    ),
                    secondary_y=use_secondary,
                )
                if show_layer_avg_sv:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(df_sorted))),
                            y=get_layer_avg_values(option_name),
                            mode="lines",
                            line=dict(color="red", width=3, dash="dash"),
                            name="Layer average" if idx == 0 else None,
                            showlegend=(idx == 0),
                            yaxis="y2" if use_secondary else "y",
                        ),
                        secondary_y=use_secondary,
                    )
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
            fig = go.Figure()
            for idx, option_name in enumerate(selected_sv_stats):
                stats = get_values(option_name)
                fig.add_trace(
                    go.Scatter(
                        y=stats, mode="lines", name=option_name,
                        line=dict(color=colors_sv[idx]),
                    )
                )
                if show_layer_avg_sv:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(df_sorted))),
                            y=get_layer_avg_values(option_name),
                            mode="lines",
                            line=dict(color="red", width=3, dash="dash"),
                            name="Layer average" if idx == 0 else None,
                            showlegend=(idx == 0),
                        )
                    )

        fig.update_xaxes(
            tickmode="array",
            tickvals=[i * n_heads for i in range(n_layers)],
            ticktext=[str(i) for i in range(n_layers)],
            title="Layer",
        )
        for xpos in range(n_heads, n_layers * n_heads, n_heads):
            fig.add_shape(
                type="line", x0=xpos, x1=xpos, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="lightgray", width=1, dash="dot"),
            )
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, width="content", key="section4_plot")

    ########################################################################
    # Section 5: Statistics by Layer and Head
    ########################################################################

    st.header("Section 5: Statistics by Layer and Head")

    stat_display_name_2b = st.selectbox(
        "Statistic", list(stat_display.keys()), key="stat_2b"
    )
    stat_name_2b = stat_display[stat_display_name_2b]

    z_pivot = df_sorted.pivot(index="layer", columns="head", values=stat_name_2b)
    z0 = z_pivot.values.astype(float)
    _finite_z0 = z0[np.isfinite(z0)]
    if len(_finite_z0) < z0.size:
        z0 = np.where(np.isfinite(z0), z0, _finite_z0.min() if len(_finite_z0) > 0 else 0)
    layer_avgs_2b = z0.mean(axis=1)
    head_avgs_2b = z0.mean(axis=0)

    def pad_range(lo, hi, pct=0.05):
        span = max(hi - lo, 1e-8)
        return [lo - span * pct, hi + span * pct]

    layer_y_range = [-0.5, n_layers - 0.5]
    head_x_range = [-0.5, n_heads - 0.5]
    layer_avg_range = pad_range(float(layer_avgs_2b.min()), float(layer_avgs_2b.max()))
    head_avg_range = pad_range(float(head_avgs_2b.min()), float(head_avgs_2b.max()))

    col_w = [0.82, 0.18]
    row_h = [0.70, 0.30]
    v_gap = 0.10
    top_row_len = row_h[0] * (1.0 - v_gap)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=col_w,
        row_heights=row_h,
        horizontal_spacing=0.04,
        vertical_spacing=v_gap,
    )
    fig.add_trace(go.Heatmap(
        z=z0, x=list(range(n_heads)), y=list(range(n_layers)),
        colorscale="Viridis",
        colorbar=dict(title=stat_display_name_2b, x=1.02, thickness=15,
                      len=top_row_len, y=1.0, yanchor="top"),
        hovertemplate="Layer %{y}<br>Head %{x}<br>%{z:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=layer_avgs_2b.tolist(), y=list(range(n_layers)),
        mode="lines+markers", line=dict(color="steelblue", width=2), marker=dict(size=4),
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=list(range(n_heads)), y=head_avgs_2b.tolist(),
        mode="lines+markers", line=dict(color="steelblue", width=2), marker=dict(size=4),
        showlegend=False,
    ), row=2, col=1)

    fft_coeffs = np.fft.rfft(head_avgs_2b)
    fft_power = np.abs(fft_coeffs) ** 2
    fft_freqs = np.fft.rfftfreq(n_heads)
    fig.add_trace(go.Scatter(
        x=fft_freqs, y=fft_power,
        mode="lines+markers", line=dict(color="steelblue", width=2), marker=dict(size=4),
        showlegend=False,
    ), row=2, col=2)

    fig.update_xaxes(title_text="Head", range=head_x_range, tickmode="linear", tick0=0, dtick=1, row=1, col=1)
    fig.update_yaxes(title_text="Layer", range=layer_y_range, tickmode="linear", tick0=0, dtick=1, row=1, col=1)
    fig.update_xaxes(title_text=stat_display_name_2b, range=layer_avg_range, tickformat=".3g", row=1, col=2)
    fig.update_yaxes(range=layer_y_range, tickmode="linear", tick0=0, dtick=1, showticklabels=False, row=1, col=2)
    fig.update_xaxes(range=head_x_range, tickmode="linear", tick0=0, dtick=1, showticklabels=False, row=2, col=1)
    fig.update_yaxes(title_text=stat_display_name_2b, range=head_avg_range, tickformat=".3g", row=2, col=1)
    fig.update_xaxes(title_text="Frequency", tickformat=".2g", row=2, col=2)
    fig.update_yaxes(title_text="Power", tickformat=".3g", row=2, col=2)
    fig.update_layout(
        height=min(max(800, 55 * n_layers + 350), 780),
        margin=dict(b=40, r=110, t=60, l=80),
    )
    st.plotly_chart(fig, width="content", key="section5_plot")

    ########################################################################
    # Section 6: Compare Statistics
    ########################################################################

    st.header("Section 6: Compare Statistics")

    sv_scatter_options = {}
    for display_name, col_name in stat_display.items():
        sv_scatter_options[display_name] = ("stat", col_name)

    if weight_selected == "W_QK":
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
        sv_scatter_options["Max SV"] = ("derived", "leading_sv")

    col1, col2 = st.columns(2)
    x_option = col1.selectbox(
        "X-axis Statistic", options=list(sv_scatter_options.keys()), key="scatter_sv_x"
    )
    y_option = col2.selectbox(
        "Y-axis Statistic", options=list(sv_scatter_options.keys()), index=1, key="scatter_sv_y"
    )

    def get_value_for_row(row, option_name):
        option_type, option_data = sv_scatter_options[option_name]
        if option_type == "stat":
            return row[option_data]
        elif option_type == "derived":
            return compute_derived_sv_stat(row["SVD"], option_data)

    all_colors = px.colors.qualitative.Plotly
    layer_0_color = all_colors[0]
    other_colors = all_colors[1:]

    def get_layer_color(layer_idx):
        if layer_idx == 0:
            return layer_0_color
        return other_colors[(layer_idx - 1) % len(other_colors)]

    fig = go.Figure()
    for layer_idx in range(n_layers):
        df_layer = df.query(f"layer == {layer_idx}")
        x_vals = []
        y_vals = []
        for _, row in df_layer.iterrows():
            x_vals.append(get_value_for_row(row, x_option))
            y_vals.append(get_value_for_row(row, y_option))
        hover_text = [
            f"Layer {layer_idx}, Head {h}<br>{x_option}: {x:.4f}<br>{y_option}: {y:.4f}"
            for h, x, y in zip(df_layer["head"].values, x_vals, y_vals)
        ]
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals, mode="markers",
                name=f"Layer {layer_idx}",
                marker=dict(color=get_layer_color(layer_idx), size=8, opacity=0.7),
                hovertext=hover_text, hoverinfo="text",
            )
        )

    fig.update_layout(
        xaxis_title=x_option,
        yaxis_title=y_option,
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        height=600,
    )
    st.plotly_chart(fig, width="content", key="section6_scatter")


def render():
    weights_dashboard_app()


if __name__ == "__main__":
    weights_dashboard_app()
