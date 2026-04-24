"""
Animations Dashboard
Animated visualizations of weight distribution and architecture evolution over training steps.
"""

import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy import stats as scipy_stats
from dashboard_utils import (
    stat_display,
    get_available_datasets,
    load_dataset_with_metadata,
    get_unique_values,
    is_HF_environment,
)


def animations_app():
    st.title("Weight Evolution Animations")
    st.markdown("Animated visualizations of distribution and architecture evolution over training steps")

    st.sidebar.header("Dataset Selection")

    hf_version = "weight_evolution"
    if is_HF_environment():
        available_datasets = get_available_datasets(hf_version)
    else:
        campaign = "step-analysis_002"
        available_datasets = get_available_datasets(campaign)
        campaign = st.sidebar.selectbox("Campaign", [campaign])

    if not available_datasets:
        st.error("No datasets found.")
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

    model_selected = get_unique_values(df_full, "model")[0]
    weight_types = get_unique_values(df_full, "weight_type")
    weight_selected = st.sidebar.selectbox("Weight Type", weight_types)

    df = df_full.query(
        f"model == '{model_selected}' and weight_type == '{weight_selected}'"
    )

    d_model = metadata.get("d_model")
    n_layers = df["layer"].max() + 1
    n_heads = df["head"].max() + 1
    steps_available = sorted(df["step"].unique())
    st.sidebar.markdown(f"**Available steps:** {len(steps_available)}")
    st.sidebar.markdown(f"Range: {steps_available[0]} - {steps_available[-1]}")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Model Dimension: {d_model}")
    st.sidebar.markdown(f"Layers: {n_layers}")
    st.sidebar.markdown(f"Heads per layer: {n_heads}")

    st.sidebar.markdown("---")
    use_eigenvalues = st.sidebar.checkbox(
        "Plot eigenvalues (λ²)",
        value=True,
        help="Square singular values to show eigenvalues.",
        key="anim_use_eigenvalues"
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
        for display_name in sv_stat_options:
            extended_stat_display[display_name] = display_name

    def get_stat_value(row, stat_display_name):
        if stat_display_name in sv_stat_options:
            _, stat_type = sv_stat_options[stat_display_name]
            return compute_derived_sv_stat(row["SVD"], stat_type)
        return row[stat_display[stat_display_name]]

    # ============================================================================
    # SECTION 1: Animated Distribution Evolution
    # ============================================================================
    st.header("Section 1: Animated Distribution Evolution")
    st.markdown("Visualize distribution evolution over training as an animation")

    col1, col2, col3 = st.columns(3)
    layer_selected_a1 = col1.slider("Layer", 0, n_layers - 1, 0, key="layer_a1")
    head_selected_a1 = col2.slider("Head", 0, n_heads - 1, 0, key="head_a1")

    available_dists_a1 = ["P(W)"]
    if weight_selected == "W_QK":
        available_dists_a1.extend(["P(λ)", "SVD"])
    dist_type_a1 = col3.selectbox("Distribution", available_dists_a1, key="dist_a1")

    dist_map = {"P(W)": "P_w", "P(λ)": "P_sv", "SVD": "SVD"}
    dist_col_a1 = dist_map[dist_type_a1]

    df_filtered_a1 = df.query(
        f"layer == {layer_selected_a1} and head == {head_selected_a1}"
    ).sort_values("step")

    if len(df_filtered_a1) == 0:
        st.warning("No data available for selected layer/head")
    else:
        col_speed, col_log = st.columns(2)
        frame_duration_a1 = col_speed.select_slider(
            "Animation speed",
            options=[50, 100, 200, 500, 1000],
            value=200,
            format_func=lambda x: f"{x} ms/frame",
            key="frame_duration_a1"
        )
        use_log_y_a1 = col_log.checkbox("Log scale (y-axis)", key="log_y_a1")

        ev_label_a1 = "Eigenvalue" if use_eigenvalues else "Singular Value"
        if dist_col_a1 == "P_w":
            bins = np.array(metadata["w_bins"])
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            xlabel_a1 = "Weight Value"
            ylabel_a1 = "Probability Density"
        elif dist_col_a1 == "P_sv":
            bins = np.array(metadata["sv_bins"])
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            xlabel_a1 = ev_label_a1
            ylabel_a1 = "Probability Density"
        else:
            xlabel_a1 = f"{ev_label_a1} Index"
            ylabel_a1 = ev_label_a1

        steps_list_a1 = df_filtered_a1["step"].values
        frames_a1 = []
        for _, row in df_filtered_a1.iterrows():
            step = row["step"]
            if dist_col_a1 == "SVD":
                plot_vals = to_plot_space(row["SVD"])
                x_data = np.arange(len(plot_vals))
                y_data = np.array(plot_vals, dtype=float)
            else:
                x_data = bin_centers
                y_data = np.array(row[dist_col_a1], dtype=float)

            if use_log_y_a1:
                y_data = np.log10(np.maximum(y_data, 1e-10))

            frames_a1.append(go.Frame(
                data=[go.Scatter(x=x_data, y=y_data, mode="lines", line=dict(color="blue", width=2))],
                name=str(step),
                layout=go.Layout(title_text=f"{dist_type_a1} at Step {step:,}")
            ))

        # Initial frame
        if dist_col_a1 == "SVD":
            y_init = np.array(to_plot_space(df_filtered_a1.iloc[0]["SVD"]), dtype=float)
            x_init = np.arange(len(y_init))
        else:
            x_init = bin_centers
            y_init = np.array(df_filtered_a1.iloc[0][dist_col_a1], dtype=float)
        if use_log_y_a1:
            y_init = np.log10(np.maximum(y_init, 1e-10))

        # Y-axis range across all frames
        all_y = np.concatenate([f.data[0].y for f in frames_a1])
        finite_y = all_y[np.isfinite(all_y)]
        ymin_a1 = float(np.min(finite_y)) - (0.5 if use_log_y_a1 else 0)
        ymax_a1 = float(np.max(finite_y)) * (1.0 if use_log_y_a1 else 1.1) + (0.5 if use_log_y_a1 else 0)

        if dist_col_a1 == "SVD" and d_model is not None:
            xmin_a1, xmax_a1 = 0, d_model // n_heads - 1
        elif dist_col_a1 != "SVD":
            xmin_a1, xmax_a1 = float(bin_centers[0]), float(bin_centers[-1])
        else:
            xmin_a1, xmax_a1 = None, None

        fig_a1 = go.Figure(
            data=[go.Scatter(x=x_init, y=y_init, mode="lines", line=dict(color="blue", width=2))],
            frames=frames_a1
        )
        fig_a1.update_layout(
            xaxis=dict(title=xlabel_a1, range=[xmin_a1, xmax_a1]),
            yaxis=dict(
                title=("Log₁₀ " if use_log_y_a1 else "") + ylabel_a1,
                range=[ymin_a1, ymax_a1]
            ),
            title=f"{dist_type_a1} Evolution: Layer {layer_selected_a1}, Head {head_selected_a1}",
            height=600,
            margin=dict(b=100),
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {"label": "▶ Play", "method": "animate", "args": [None, {
                        "frame": {"duration": frame_duration_a1, "redraw": True},
                        "fromcurrent": True, "mode": "immediate",
                        "transition": {"duration": frame_duration_a1 // 2}
                    }]},
                    {"label": "⏸ Pause", "method": "animate", "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate", "transition": {"duration": 0}
                    }]},
                ],
                "x": 0.1, "y": 1.12
            }],
            sliders=[{
                "active": 0,
                "steps": [
                    {
                        "args": [[str(step)], {
                            "frame": {"duration": frame_duration_a1, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": frame_duration_a1 // 2}
                        }],
                        "label": "",
                        "method": "animate"
                    }
                    for step in steps_list_a1
                ],
                "x": 0.1, "len": 0.85, "xanchor": "left",
                "y": -0.05, "yanchor": "top",
                "pad": {"t": 30},
                "currentvalue": {
                    "visible": True, "prefix": "Step: ",
                    "xanchor": "center", "font": {"size": 14}
                },
            }],
        )
        st.plotly_chart(fig_a1, width="stretch")

    # ============================================================================
    # SECTION 2: Animated Architecture Heatmap
    # ============================================================================
    st.header("Section 2: Animated Architecture Evolution")
    st.markdown("Visualize how statistics evolve across the architecture over training")

    stat_display_name_a2 = st.selectbox(
        "Statistic", list(extended_stat_display.keys()), key="stat_a2"
    )

    col_speed2, col_log2 = st.columns(2)
    frame_duration_a2 = col_speed2.select_slider(
        "Animation speed",
        options=[50, 100, 200, 500, 1000],
        value=200,
        format_func=lambda x: f"{x} ms/frame",
        key="frame_duration_a2"
    )
    use_log_color_a2 = col_log2.checkbox("Log color scale", key="log_color_a2")

    steps_list_a2 = sorted(df["step"].unique())

    # Precompute all stat values to determine global zmin/zmax
    all_stat_values = []
    for step in steps_list_a2:
        df_step = df.query(f"step == {step}").sort_values(["layer", "head"])
        for _, row in df_step.iterrows():
            all_stat_values.append(get_stat_value(row, stat_display_name_a2))
    all_stat_values = np.array(all_stat_values)

    if use_log_color_a2:
        all_z = np.log10(np.abs(all_stat_values) + 1e-10)
    else:
        all_z = all_stat_values
    zmin_a2, zmax_a2 = float(np.min(all_z)), float(np.max(all_z))
    colorbar_title_a2 = ("Log₁₀ " if use_log_color_a2 else "") + stat_display_name_a2

    frames_a2 = []
    for step in steps_list_a2:
        df_step = df.query(f"step == {step}").sort_values(["layer", "head"])
        z_vals = np.array([get_stat_value(row, stat_display_name_a2) for _, row in df_step.iterrows()])
        z_plot = np.log10(np.abs(z_vals) + 1e-10) if use_log_color_a2 else z_vals
        frames_a2.append(go.Frame(
            data=[go.Heatmap(
                z=z_plot.reshape(n_layers, n_heads),
                x=list(range(n_heads)),
                y=list(range(n_layers)),
                colorscale="Viridis",
                zmin=zmin_a2, zmax=zmax_a2,
                colorbar=dict(title=colorbar_title_a2)
            )],
            name=str(step),
            layout=go.Layout(title_text=f"{stat_display_name_a2} at Step {step:,}")
        ))

    df_init_a2 = df.query(f"step == {steps_list_a2[0]}").sort_values(["layer", "head"])
    z_init_vals = np.array([get_stat_value(row, stat_display_name_a2) for _, row in df_init_a2.iterrows()])
    z_init_a2 = np.log10(np.abs(z_init_vals) + 1e-10) if use_log_color_a2 else z_init_vals

    fig_a2 = go.Figure(
        data=[go.Heatmap(
            z=z_init_a2.reshape(n_layers, n_heads),
            x=list(range(n_heads)),
            y=list(range(n_layers)),
            colorscale="Viridis",
            zmin=zmin_a2, zmax=zmax_a2,
            colorbar=dict(title=colorbar_title_a2)
        )],
        frames=frames_a2
    )
    fig_a2.update_layout(
        xaxis=dict(title="Head", tickmode="linear", tick0=0, dtick=1),
        yaxis=dict(title="Layer", tickmode="linear", tick0=0, dtick=1),
        title=f"{stat_display_name_a2} Evolution Across Architecture",
        height=max(500, 40 * n_layers + 200),
        margin=dict(b=100),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {
                    "frame": {"duration": frame_duration_a2, "redraw": True},
                    "fromcurrent": True, "mode": "immediate",
                    "transition": {"duration": frame_duration_a2 // 2}
                }]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate", "transition": {"duration": 0}
                }]},
            ],
            "x": 0.1, "y": 1.12
        }],
        sliders=[{
            "active": 0,
            "steps": [
                {
                    "args": [[str(step)], {
                        "frame": {"duration": frame_duration_a2, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": frame_duration_a2 // 2}
                    }],
                    "label": "",
                    "method": "animate"
                }
                for step in steps_list_a2
            ],
            "x": 0.1, "len": 0.85, "xanchor": "left",
            "y": -0.05, "yanchor": "top",
            "pad": {"t": 30},
            "currentvalue": {
                "visible": True, "prefix": "Step: ",
                "xanchor": "center", "font": {"size": 14}
            },
        }],
    )
    st.plotly_chart(fig_a2, width="stretch")


def render():
    animations_app()


if __name__ == "__main__":
    animations_app()
