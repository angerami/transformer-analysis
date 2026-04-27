"""
Animations Dashboard
Animated visualizations of weight distribution and architecture evolution over training steps.
"""

import io
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def _generate_section2_gif(all_z_plots, steps_list, n_layers, n_heads,
                            global_zmin, global_zmax, layer_avg_range, head_avg_range,
                            fix_scale, colorbar_title, stat_name,
                            frame_ms=100, progress_cb=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from PIL import Image

    frames = []
    for i, step in enumerate(steps_list):
        z = np.array(all_z_plots[step])
        local_vmin = global_zmin if fix_scale else float(np.min(z))
        local_vmax = global_zmax if fix_scale else float(np.max(z))
        layer_avgs = z.mean(axis=1)
        head_avgs = z.mean(axis=0)

        fig = plt.figure(figsize=(13, 7), facecolor="white", dpi=90)
        gs = gridspec.GridSpec(2, 2, width_ratios=[4.5, 1], height_ratios=[7, 3],
                               hspace=0.18, wspace=0.08,
                               left=0.07, right=0.90, top=0.93, bottom=0.07)

        ax_heat = fig.add_subplot(gs[0, 0])
        ax_layer = fig.add_subplot(gs[0, 1])
        ax_head = fig.add_subplot(gs[1, 0])

        im = ax_heat.imshow(z, aspect="auto", origin="lower",
                            vmin=local_vmin, vmax=local_vmax, cmap="viridis",
                            extent=[-0.5, n_heads - 0.5, -0.5, n_layers - 0.5])
        ax_heat.set_xlabel("Head", fontsize=10)
        ax_heat.set_ylabel("Layer", fontsize=10)
        ax_heat.set_title(f"{stat_name} — Step {step:,}", fontsize=11)
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.03)
        cbar.set_label(colorbar_title, fontsize=9)

        ax_layer.plot(layer_avgs, np.arange(n_layers), "o-",
                      color="steelblue", markersize=3, linewidth=1.5)
        ax_layer.set_xlim(layer_avg_range)
        ax_layer.set_ylim(-0.5, n_layers - 0.5)
        ax_layer.set_xlabel(colorbar_title, fontsize=9)
        ax_layer.yaxis.set_visible(False)
        ax_layer.tick_params(axis="x", labelsize=8)

        ax_head.plot(np.arange(n_heads), head_avgs, "o-",
                     color="steelblue", markersize=3, linewidth=1.5)
        ax_head.set_xlim(-0.5, n_heads - 0.5)
        ax_head.set_ylim(head_avg_range)
        ax_head.set_ylabel(colorbar_title, fontsize=9)
        ax_head.xaxis.set_visible(False)
        ax_head.tick_params(axis="y", labelsize=8)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

        if progress_cb:
            progress_cb((i + 1) / len(steps_list))

    gif_buf = io.BytesIO()
    frames[0].save(gif_buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=frame_ms, loop=0, optimize=False)
    gif_buf.seek(0)
    return gif_buf.read()


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
        ctrl_cols = st.columns(4)
        frame_duration_a1 = ctrl_cols[0].select_slider(
            "Animation speed",
            options=[50, 100, 200, 500, 1000],
            value=200,
            format_func=lambda x: f"{x} ms/frame",
            key="frame_duration_a1"
        )
        use_log_y_a1 = ctrl_cols[1].checkbox("Log scale (y-axis)", key="log_y_a1")
        show_t0_a1 = ctrl_cols[2].checkbox("Overlay t=0", key="show_t0_a1")
        show_final_a1 = ctrl_cols[3].checkbox("Overlay final", key="show_final_a1")

        ev_label_a1 = "Eigenvalue" if use_eigenvalues else "Singular Value"
        is_dist_a1 = dist_col_a1 in ("P_w", "P_sv")

        if dist_col_a1 == "P_w":
            bins = np.array(metadata["w_bins"])
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_width = float(bins[1] - bins[0])
            xlabel_a1 = "Weight Value"
            ylabel_a1 = "Probability Density"
        elif dist_col_a1 == "P_sv":
            bins = np.array(metadata["sv_bins"])
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_width = float(bins[1] - bins[0])
            xlabel_a1 = ev_label_a1
            ylabel_a1 = "Probability Density"
        else:
            xlabel_a1 = f"{ev_label_a1} Index"
            ylabel_a1 = ev_label_a1

        count_min = 1.0 / (d_model ** 2) if d_model else 1e-6

        def prepare_y_a1(y_raw, log_scale):
            y = np.array(y_raw, dtype=float)
            if is_dist_a1:
                y = np.maximum(y, count_min)
            if log_scale:
                y = np.log10(np.maximum(y, 1e-10))
            return y

        steps_list_a1 = df_filtered_a1["step"].values

        # Compute overlay reference data before frame loop so frames can include them
        row0 = df_filtered_a1.iloc[0]
        rowf = df_filtered_a1.iloc[-1]
        if dist_col_a1 == "SVD":
            x_ref = np.arange(len(to_plot_space(row0["SVD"])))
            y_t0 = prepare_y_a1(to_plot_space(row0["SVD"]), use_log_y_a1)
            y_final = prepare_y_a1(to_plot_space(rowf["SVD"]), use_log_y_a1)
            bw_init = 0.8
        else:
            x_ref = bin_centers
            y_t0 = prepare_y_a1(row0[dist_col_a1], use_log_y_a1)
            y_final = prepare_y_a1(rowf[dist_col_a1], use_log_y_a1)
            bw_init = bin_width
        x_init, y_init = x_ref, y_t0.copy()

        # Each frame carries all traces so they stay visible regardless of redraw mode
        frames_a1 = []
        for _, row in df_filtered_a1.iterrows():
            step = row["step"]
            if dist_col_a1 == "SVD":
                plot_vals = to_plot_space(row["SVD"])
                x_data = np.arange(len(plot_vals))
                y_data = prepare_y_a1(plot_vals, use_log_y_a1)
                bw = 0.8
            else:
                x_data = bin_centers
                y_data = prepare_y_a1(row[dist_col_a1], use_log_y_a1)
                bw = bin_width

            frame_data = [go.Bar(x=x_data, y=y_data, width=bw, marker_color="steelblue", showlegend=False)]
            if show_t0_a1:
                frame_data.append(go.Scatter(x=x_ref, y=y_t0, mode="lines",
                                             line=dict(color="limegreen", width=2),
                                             name=f"t={steps_list_a1[0]:,}"))
            if show_final_a1:
                frame_data.append(go.Scatter(x=x_ref, y=y_final, mode="lines",
                                             line=dict(color="tomato", width=2),
                                             name=f"t={steps_list_a1[-1]:,}"))
            frames_a1.append(go.Frame(
                data=frame_data,
                traces=list(range(len(frame_data))),
                name=str(step),
                layout=go.Layout(title_text=f"{dist_type_a1} at Step {step:,}")
            ))

        all_y = np.concatenate([f.data[0].y for f in frames_a1])
        finite_y = all_y[np.isfinite(all_y)]
        if use_log_y_a1 and is_dist_a1:
            ymin_a1 = np.log10(count_min) - 0.3
        elif use_log_y_a1:
            ymin_a1 = float(np.min(finite_y)) - 0.5
        elif is_dist_a1:
            ymin_a1 = count_min
        else:
            ymin_a1 = 0.0
        ymax_a1 = float(np.max(finite_y)) * (1.0 if use_log_y_a1 else 1.1) + (0.5 if use_log_y_a1 else 0)

        if dist_col_a1 == "SVD" and d_model is not None:
            xmin_a1, xmax_a1 = 0, d_model // n_heads - 1
        elif dist_col_a1 != "SVD":
            xmin_a1, xmax_a1 = float(bin_centers[0]), float(bin_centers[-1])
        else:
            xmin_a1, xmax_a1 = None, None

        fig_data_a1 = [go.Bar(
            x=x_init, y=y_init,
            width=bw_init,
            marker_color="steelblue",
            name="Current",
            showlegend=False,
        )]
        if show_t0_a1:
            fig_data_a1.append(go.Scatter(
                x=x_ref, y=y_t0,
                mode="lines",
                line=dict(color="limegreen", width=2),
                name=f"t={steps_list_a1[0]:,}",
            ))
        if show_final_a1:
            fig_data_a1.append(go.Scatter(
                x=x_ref, y=y_final,
                mode="lines",
                line=dict(color="tomato", width=2),
                name=f"t={steps_list_a1[-1]:,}",
            ))

        fig_a1 = go.Figure(data=fig_data_a1, frames=frames_a1)
        fig_a1.update_layout(
            xaxis=dict(title=xlabel_a1, range=[xmin_a1, xmax_a1]),
            yaxis=dict(
                title=("Log₁₀ " if use_log_y_a1 else "") + ylabel_a1,
                range=[ymin_a1, ymax_a1]
            ),
            title=f"{dist_type_a1} Evolution: Layer {layer_selected_a1}, Head {head_selected_a1}",
            height=600,
            margin=dict(b=100),
            showlegend=(show_t0_a1 or show_final_a1),
            bargap=0,
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {"label": "▶ Play", "method": "animate", "args": [None, {
                        "frame": {"duration": frame_duration_a1, "redraw": True},
                        "fromcurrent": True, "mode": "immediate",
                        "transition": {"duration": 0}
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
                            "transition": {"duration": 0}
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
        html_a1 = fig_a1.to_html(include_plotlyjs="cdn", full_html=True).encode()
        st.download_button(
            "⬇ Download Interactive HTML",
            html_a1,
            file_name=f"dist_evolution_L{layer_selected_a1}_H{head_selected_a1}.html",
            mime="text/html",
            key="dl_s1_html",
        )

    # ============================================================================
    # SECTION 2: Architecture Heatmap — 2×2 corner-plot
    # Driven by a self-contained HTML component so animation runs entirely in JS
    # via Plotly.react (in-place diff, no canvas clear, no strobing).
    # ============================================================================
    st.header("Section 2: Animated Architecture Evolution")
    st.markdown("Visualize how statistics evolve across the architecture over training")

    stat_display_name_a2 = st.selectbox(
        "Statistic", list(extended_stat_display.keys()), key="stat_a2"
    )

    ctrl_cols2 = st.columns(2)
    use_log_color_a2 = ctrl_cols2[0].checkbox("Log color scale", key="log_color_a2")
    fix_scale_a2 = ctrl_cols2[1].checkbox("Fix color scale", value=True, key="fix_scale_a2")

    steps_list_a2 = sorted(df["step"].unique())

    # Precompute all stat grids and z_plots once — serialized to JS, no re-runs needed
    all_stat_grids = {}
    for step in steps_list_a2:
        df_step = df.query(f"step == {step}").sort_values(["layer", "head"])
        vals = np.array([get_stat_value(row, stat_display_name_a2) for _, row in df_step.iterrows()])
        all_stat_grids[step] = vals.reshape(n_layers, n_heads)

    def to_z(grid):
        return np.log10(np.abs(grid) + 1e-10) if use_log_color_a2 else grid

    all_z_plots = {step: to_z(g) for step, g in all_stat_grids.items()}

    all_z_flat = np.concatenate([z.ravel() for z in all_z_plots.values()])
    global_zmin, global_zmax = float(np.min(all_z_flat)), float(np.max(all_z_flat))

    all_layer_avgs = np.concatenate([z.mean(axis=1) for z in all_z_plots.values()])
    all_head_avgs = np.concatenate([z.mean(axis=0) for z in all_z_plots.values()])

    def pad_range(lo, hi, pct=0.05):
        span = max(hi - lo, 1e-8)
        return [lo - span * pct, hi + span * pct]

    layer_avg_range = pad_range(float(np.min(all_layer_avgs)), float(np.max(all_layer_avgs)))
    head_avg_range = pad_range(float(np.min(all_head_avgs)), float(np.max(all_head_avgs)))
    colorbar_title_a2 = ("Log₁₀ " if use_log_color_a2 else "") + stat_display_name_a2

    # Cap height to screen-friendly size before building the figure
    fig_height = min(max(800, 55 * n_layers + 350), 780)

    # Build the initial (first step) Plotly figure — layout only, data updated via JS
    z0 = all_z_plots[steps_list_a2[0]]
    zmin0 = global_zmin if fix_scale_a2 else float(np.min(z0))
    zmax0 = global_zmax if fix_scale_a2 else float(np.max(z0))
    layer_y_range = [-0.5, n_layers - 0.5]
    head_x_range = [-0.5, n_heads - 0.5]

    # Layout proportions — wide left column so heatmap cells are wider than tall;
    # colorbar len/y anchored to match the top-row height exactly.
    col_w = [0.82, 0.18]
    row_h = [0.70, 0.30]
    v_gap = 0.10
    top_row_len = row_h[0] * (1.0 - v_gap)   # colorbar length = top-row paper height

    fig_a2 = make_subplots(
        rows=2, cols=2,
        column_widths=col_w,
        row_heights=row_h,
        horizontal_spacing=0.04,
        vertical_spacing=v_gap,
    )
    fig_a2.add_trace(go.Heatmap(
        z=z0, x=list(range(n_heads)), y=list(range(n_layers)),
        colorscale="Viridis", zmin=zmin0, zmax=zmax0,
        colorbar=dict(title=colorbar_title_a2, x=1.02, thickness=15,
                      len=top_row_len, y=1.0, yanchor="top"),
    ), row=1, col=1)
    fig_a2.add_trace(go.Scatter(
        x=z0.mean(axis=1).tolist(), y=list(range(n_layers)),
        mode="lines+markers", line=dict(color="steelblue", width=2), marker=dict(size=4),
        showlegend=False,
    ), row=1, col=2)
    fig_a2.add_trace(go.Scatter(
        x=list(range(n_heads)), y=z0.mean(axis=0).tolist(),
        mode="lines+markers", line=dict(color="steelblue", width=2), marker=dict(size=4),
        showlegend=False,
    ), row=2, col=1)

    fig_a2.update_xaxes(title_text="Head", range=head_x_range, tickmode="linear", tick0=0, dtick=1, row=1, col=1)
    fig_a2.update_yaxes(title_text="Layer", range=layer_y_range, tickmode="linear", tick0=0, dtick=1, row=1, col=1)
    fig_a2.update_xaxes(title_text=colorbar_title_a2, range=layer_avg_range, tickformat=".3g", row=1, col=2)
    fig_a2.update_yaxes(range=layer_y_range, tickmode="linear", tick0=0, dtick=1, showticklabels=False, row=1, col=2)
    fig_a2.update_xaxes(range=head_x_range, tickmode="linear", tick0=0, dtick=1, showticklabels=False, row=2, col=1)
    fig_a2.update_yaxes(title_text=colorbar_title_a2, range=head_avg_range, tickformat=".3g", row=2, col=1)
    fig_a2.update_layout(
        title=f"{stat_display_name_a2} — Step {steps_list_a2[0]:,}",
        height=fig_height,  # matches the capped CSS height set in the component
        margin=dict(b=40, r=110, t=60, l=80),
        uirevision="constant",
    )

    # Serialize figure + per-frame data for the JS component
    fig_json_str = fig_a2.to_json()
    frames_data = [
        {
            "step": int(step),
            "z": all_z_plots[step].tolist(),
            "layer_avgs": all_z_plots[step].mean(axis=1).tolist(),
            "head_avgs": all_z_plots[step].mean(axis=0).tolist(),
            "zmin": global_zmin if fix_scale_a2 else float(np.min(all_z_plots[step])),
            "zmax": global_zmax if fix_scale_a2 else float(np.max(all_z_plots[step])),
        }
        for step in steps_list_a2
    ]
    frames_json_str = json.dumps(frames_data)
    title_prefix_js = json.dumps(f"{stat_display_name_a2} \u2014 Step ")
    n_frames = len(steps_list_a2)
    ctrl_height = 50
    total_height = fig_height + ctrl_height

    html_component = f"""<!DOCTYPE html>
<html style="height:{total_height}px; overflow:hidden;">
<head>
<script src="https://cdn.plot.ly/plotly-latest.min.js" charset="utf-8"></script>
<style>
  body {{ margin:0; padding:0; font-family:sans-serif; background:white;
          height:{total_height}px; overflow:hidden; }}
  #controls {{ padding:6px 14px; height:{ctrl_height}px; box-sizing:border-box;
               display:flex; align-items:center; gap:14px; }}
  #fig_div {{ width:100%; height:{fig_height}px; }}
  button {{ padding:3px 12px; cursor:pointer; font-size:13px; }}
  .lbl {{ font-size:12px; color:#555; white-space:nowrap; }}
  input[type=range] {{ cursor:pointer; }}
</style>
</head>
<body>
<div id="controls">
  <button id="play-btn" onclick="togglePlay()">&#9654; Play</button>
  <input type="range" id="step-slider" min="0" max="{n_frames - 1}" value="0"
         style="flex:1" oninput="onSlider(this.value)">
  <span class="lbl" id="step-display">Step: {steps_list_a2[0]:,} (1/{n_frames})</span>
  <span class="lbl">Speed:</span>
  <input type="range" id="speed-slider" min="50" max="500" value="100" step="50"
         style="width:80px" oninput="onSpeed(this.value)">
  <span class="lbl" id="speed-display">100 ms</span>
</div>
<div id="fig_div"></div>
<script>
(function() {{
  var figData = {fig_json_str};
  var framesData = {frames_json_str};
  var titlePrefix = {title_prefix_js};
  var frameIdx = 0, playing = false, timer = null, speed = 100;
  var figDiv = document.getElementById('fig_div');
  var iData = figData.data;
  var iLayout = figData.layout;

  Plotly.newPlot(figDiv, iData, iLayout, {{staticPlot: false}});

  function applyFrame(idx) {{
    var f = framesData[idx];
    Plotly.react(figDiv,
      [
        Object.assign({{}}, iData[0], {{z: f.z, zmin: f.zmin, zmax: f.zmax}}),
        Object.assign({{}}, iData[1], {{x: f.layer_avgs}}),
        Object.assign({{}}, iData[2], {{y: f.head_avgs}})
      ],
      Object.assign({{}}, iLayout, {{title: {{text: titlePrefix + f.step.toLocaleString()}}}})
    );
    document.getElementById('step-display').textContent =
      'Step: ' + f.step.toLocaleString() + ' (' + (idx+1) + '/{n_frames})';
    document.getElementById('step-slider').value = idx;
    frameIdx = idx;
  }}

  window.togglePlay = function() {{
    if (playing) {{
      clearInterval(timer); playing = false;
      document.getElementById('play-btn').innerHTML = '&#9654; Play';
    }} else {{
      playing = true;
      document.getElementById('play-btn').innerHTML = '&#9646;&#9646; Pause';
      timer = setInterval(function() {{
        applyFrame((frameIdx + 1) % framesData.length);
      }}, speed);
    }}
  }};

  window.onSlider = function(val) {{
    if (playing) {{
      clearInterval(timer); playing = false;
      document.getElementById('play-btn').innerHTML = '&#9654; Play';
    }}
    applyFrame(parseInt(val));
  }};

  window.onSpeed = function(val) {{
    speed = parseInt(val);
    document.getElementById('speed-display').textContent = val + ' ms';
    if (playing) {{
      clearInterval(timer);
      timer = setInterval(function() {{
        applyFrame((frameIdx + 1) % framesData.length);
      }}, speed);
    }}
  }};
}})();
</script>
</body>
</html>"""

    st.components.v1.html(html_component, height=total_height, scrolling=True)

    exp_cols = st.columns(2)
    html_s2_bytes = html_component.encode()
    exp_cols[0].download_button(
        "⬇ Download Interactive HTML",
        html_s2_bytes,
        file_name=f"architecture_evolution_{stat_display_name_a2}.html",
        mime="text/html",
        key="dl_s2_html",
    )
    if exp_cols[1].button("🎞 Generate GIF", key="gen_s2_gif"):
        prog = st.progress(0.0, text="Generating GIF frames…")
        gif_bytes = _generate_section2_gif(
            all_z_plots, steps_list_a2, n_layers, n_heads,
            global_zmin, global_zmax, layer_avg_range, head_avg_range,
            fix_scale_a2, colorbar_title_a2, stat_display_name_a2,
            frame_ms=100, progress_cb=lambda p: prog.progress(p, text=f"Rendering frame {int(p * len(steps_list_a2))}/{len(steps_list_a2)}…"),
        )
        st.session_state["s2_gif_bytes"] = gif_bytes
        st.session_state["s2_gif_name"] = f"architecture_evolution_{stat_display_name_a2}.gif"
        prog.empty()

    if "s2_gif_bytes" in st.session_state:
        st.download_button(
            "⬇ Download GIF",
            st.session_state["s2_gif_bytes"],
            file_name=st.session_state.get("s2_gif_name", "animation.gif"),
            mime="image/gif",
            key="dl_s2_gif",
        )


def render():
    animations_app()


if __name__ == "__main__":
    animations_app()
