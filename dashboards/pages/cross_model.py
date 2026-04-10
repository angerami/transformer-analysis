import plotly.graph_objects as go
import streamlit as st
import numpy as np
from dashboard_utils import (
    stat_display,
    get_available_campaigns,
    load_dataset_with_metadata,
    get_unique_values,
    is_HF_environment,
    model_size_from_name,
    create_snapshot_button,
)

MODEL_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

SV_DERIVED_OPTIONS = {
    "Leading SV": "leading_sv",
    "SV Mean": "sv_mean",
    "Σσ": "sv_sum",
    "Σσ²": "sv_sum2",
    "Participation Ratio": "participation_ratio",
    "Normalized PR": "normalized_participation_ratio",
    "Spectral Entropy": "spectral_entropy",
    "Condition Number": "condition_number",
    "Stable Rank": "stable_rank",
}


def _model_family(name):
    n = name.lower()
    if "pythia" in n:
        return "pythia"
    if "gpt" in n:
        return "gpt"
    if "llama" in n:
        return "llama"
    if "mistral" in n:
        return "mistral"
    return n.split("-")[0]


def _compute_sv_stats(sv_array, d_head):
    sv = np.asarray(sv_array, dtype=float)[:d_head]
    sv_sum = np.sum(sv)
    sv_sum2 = np.sum(sv ** 2)
    pr = sv_sum ** 2 / sv_sum2 if sv_sum2 > 0 else 0.0
    if sv_sum2 > 0:
        p = sv ** 2 / sv_sum2
        p = p[p > 0]
        spectral_entropy = float(-np.sum(p * np.log(p)))
    else:
        spectral_entropy = 0.0
    sv_nonzero = sv[sv > 0]
    condition_number = float(sv_nonzero[0] / sv_nonzero[-1]) if len(sv_nonzero) > 1 else 1.0
    stable_rank = float(sv_sum2 / sv[0] ** 2) if sv[0] > 0 else 0.0
    return {
        "leading_sv": float(sv[0]) if len(sv) > 0 else 0.0,
        "sv_mean": float(np.mean(sv)),
        "sv_sum": float(sv_sum),
        "sv_sum2": float(sv_sum2),
        "participation_ratio": float(pr),
        "normalized_participation_ratio": float(pr / d_head) if d_head > 0 else 0.0,
        "spectral_entropy": spectral_entropy,
        "condition_number": condition_number,
        "stable_rank": stable_rank,
    }


def _get_d_head(metadata, model_name, df_model):
    meta = metadata.get("merged", {}).get(model_name, metadata)
    d_model = meta.get("d_model", metadata.get("d_model", 768))
    n_heads = int(df_model["head"].max()) + 1
    return d_model // n_heads


def _extract_values(df_sorted, option_display, metadata, model_name):
    if option_display in stat_display:
        return df_sorted[stat_display[option_display]].values.astype(float)

    stat_key = SV_DERIVED_OPTIONS[option_display]
    d_head = _get_d_head(metadata, model_name, df_sorted)

    # Try precomputed column first
    if stat_key in df_sorted.columns and df_sorted[stat_key].notna().any():
        vals = df_sorted[stat_key].values.astype(float)
        if stat_key == "normalized_participation_ratio":
            pr_key = "participation_ratio"
            if pr_key in df_sorted.columns and df_sorted[pr_key].notna().any():
                vals = df_sorted[pr_key].values.astype(float) / d_head
        return vals

    return np.array([
        _compute_sv_stats(row["SVD"], d_head)[stat_key]
        for _, row in df_sorted.iterrows()
    ])


def _architecture_preset(model_names, metadata):
    """One model per family, each closest to the median d_model across all models."""
    merge_meta = metadata.get("merged", {})
    d_models = {m: merge_meta.get(m, {}).get("d_model", 0) for m in model_names}
    vals = [v for v in d_models.values() if v > 0]
    target = sorted(vals)[len(vals) // 2] if vals else 768

    families: dict[str, list[str]] = {}
    for m in model_names:
        families.setdefault(_model_family(m), []).append(m)

    return [
        min(models, key=lambda m: abs(d_models.get(m, 0) - target))
        for models in families.values()
    ]


def cross_model_app():
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

    all_model_names = sorted(
        get_unique_values(df_full, "model"),
        key=lambda x: (_model_family(x), model_size_from_name(x)),
    )
    weight_types = get_unique_values(df_full, "weight_type")
    weight_selected = st.sidebar.selectbox("Weight Type", weight_types)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Selection")

    preset = st.sidebar.selectbox(
        "Preset",
        ["All models", "All GPT", "All Pythia", "Architecture", "Custom"],
    )

    if preset == "All models":
        default_models = all_model_names
    elif preset == "All GPT":
        default_models = [m for m in all_model_names if _model_family(m) == "gpt"]
    elif preset == "All Pythia":
        default_models = [m for m in all_model_names if _model_family(m) == "pythia"]
    elif preset == "Architecture":
        default_models = _architecture_preset(all_model_names, metadata)
    else:
        default_models = all_model_names[:min(3, len(all_model_names))]

    # Key tied to preset forces re-initialization when preset changes
    models_selected = st.sidebar.multiselect(
        "Models",
        options=all_model_names,
        default=default_models,
        key=f"models_{preset}",
    )

    if not models_selected:
        st.warning("Select at least one model.")
        st.stop()

    # Stable color assignment: index into full model list so colors don't shift as selection changes
    model_colors = {
        m: MODEL_COLORS[i % len(MODEL_COLORS)]
        for i, m in enumerate(all_model_names)
    }

    st.title("Cross-Model Comparison")

    sv_options = list(SV_DERIVED_OPTIONS.keys()) if weight_selected == "W_QK" else []
    all_stat_options = list(stat_display.keys()) + sv_options

    ########################################################################
    # Section 1: Statistic across architecture (layer × head), multi-model
    ########################################################################
    st.header("Section 1: Statistic Across Architecture")

    col1, col2 = st.columns(2)
    stat_s1 = col1.selectbox("Statistic", options=all_stat_options, key="s1_stat")
    view_mode = col2.radio(
        "View",
        ["Per head", "Layer means only"],
        horizontal=True,
        key="s1_view_mode",
    )
    show_layer_avg = (
        st.checkbox(
            "Overlay layer means",
            value=False,
            help="Dashed line at each layer's mean, in the matching model color",
            key="s1_layer_avg",
        )
        if view_mode == "Per head"
        else False
    )

    fig_s1 = go.Figure()

    for model_name in models_selected:
        df_m = df_full.query(
            f"model == '{model_name}' and weight_type == '{weight_selected}'"
        )
        if df_m.empty:
            continue

        df_m_sorted = df_m.sort_values(["layer", "head"])
        n_layers_m = int(df_m_sorted["layer"].max()) + 1
        n_heads_m = int(df_m_sorted["head"].max()) + 1
        color = model_colors[model_name]

        if view_mode == "Layer means only":
            layer_idxs, layer_means = [], []
            for layer_idx in range(n_layers_m):
                df_layer = df_m.query(f"layer == {layer_idx}").sort_values("head")
                try:
                    layer_vals = _extract_values(df_layer, stat_s1, metadata, model_name)
                    layer_means.append(float(np.mean(layer_vals)))
                    layer_idxs.append(layer_idx)
                except Exception:
                    pass

            fig_s1.add_trace(go.Scatter(
                x=layer_idxs,
                y=layer_means,
                mode="lines+markers",
                name=model_name,
                line=dict(color=color),
                marker=dict(color=color, size=5),
                legendgroup=model_name,
            ))
        else:
            try:
                values = _extract_values(df_m_sorted, stat_s1, metadata, model_name)
            except Exception:
                continue

            fig_s1.add_trace(go.Scatter(
                x=np.arange(len(values)).tolist(),
                y=values.tolist(),
                mode="lines",
                name=model_name,
                line=dict(color=color),
                legendgroup=model_name,
            ))

            if show_layer_avg:
                avg_values = []
                for layer_idx in range(n_layers_m):
                    df_layer = df_m.query(f"layer == {layer_idx}").sort_values("head")
                    try:
                        layer_vals = _extract_values(df_layer, stat_s1, metadata, model_name)
                        avg = float(np.mean(layer_vals))
                    except Exception:
                        avg = 0.0
                    avg_values.extend([avg] * n_heads_m)

                fig_s1.add_trace(go.Scatter(
                    x=list(range(len(avg_values))),
                    y=avg_values,
                    mode="lines",
                    name=f"{model_name} (layer avg)",
                    line=dict(color=color, width=3, dash="dash"),
                    legendgroup=model_name,
                    showlegend=False,
                ))

    xaxis_title = "Layer" if view_mode == "Layer means only" else "Head index (layer × n_heads + head)"
    fig_s1.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=stat_s1,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_s1, width="content", key="cross_model_s1")

    create_snapshot_button(
        fig=fig_s1,
        metadata={
            "section": "cross_model_section_1",
            "campaign": campaign_name,
            "models": models_selected,
            "weight_type": weight_selected,
            "statistic": stat_s1,
            "view_mode": view_mode,
            "show_layer_avg": show_layer_avg,
        },
        section_name="cross_model_s1",
        key="snapshot_cm_s1",
    )

    ########################################################################
    # Section 2: 2D scatter colored by model
    ########################################################################
    st.header("Section 2: Cross-Model Correlation")

    col1, col2 = st.columns(2)
    x_opt = col1.selectbox("X-axis", options=all_stat_options, key="cm_scatter_x")
    y_opt = col2.selectbox(
        "Y-axis",
        options=all_stat_options,
        index=min(1, len(all_stat_options) - 1),
        key="cm_scatter_y",
    )

    fig_s2 = go.Figure()

    for model_name in models_selected:
        df_m = df_full.query(
            f"model == '{model_name}' and weight_type == '{weight_selected}'"
        )
        if df_m.empty:
            continue

        df_m_sorted = df_m.sort_values(["layer", "head"])
        color = model_colors[model_name]

        try:
            x_vals = _extract_values(df_m_sorted, x_opt, metadata, model_name)
            y_vals = _extract_values(df_m_sorted, y_opt, metadata, model_name)
        except Exception:
            continue

        hover_texts = [
            f"{model_name} L{row['layer']} H{row['head']}<br>"
            f"{x_opt}: {x:.4f}<br>{y_opt}: {y:.4f}"
            for (_, row), x, y in zip(df_m_sorted.iterrows(), x_vals, y_vals)
        ]

        fig_s2.add_trace(go.Scatter(
            x=x_vals.tolist(),
            y=y_vals.tolist(),
            mode="markers",
            name=model_name,
            marker=dict(color=color, size=6, opacity=0.7),
            hovertext=hover_texts,
            hoverinfo="text",
        ))

    fig_s2.update_layout(
        xaxis_title=x_opt,
        yaxis_title=y_opt,
        hovermode="closest",
        height=600,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    st.plotly_chart(fig_s2, width="content", key="cross_model_s2")

    create_snapshot_button(
        fig=fig_s2,
        metadata={
            "section": "cross_model_section_2",
            "campaign": campaign_name,
            "models": models_selected,
            "weight_type": weight_selected,
            "x_statistic": x_opt,
            "y_statistic": y_opt,
        },
        section_name="cross_model_s2",
        key="snapshot_cm_s2",
    )


    ########################################################################
    # Section 3: Marginal distribution per model (KDE or histogram)
    ########################################################################
    st.header("Section 3: Statistic Distribution by Model")

    col1, col2 = st.columns(2)
    stat_s3 = col1.selectbox("Statistic", options=all_stat_options, key="s3_stat")
    dist_mode = col2.radio("Display", ["KDE", "Histogram"], horizontal=True, key="s3_mode")

    fig_s3 = go.Figure()

    for model_name in models_selected:
        df_m = df_full.query(
            f"model == '{model_name}' and weight_type == '{weight_selected}'"
        )
        if df_m.empty:
            continue

        df_m_sorted = df_m.sort_values(["layer", "head"])
        color = model_colors[model_name]

        try:
            values = _extract_values(df_m_sorted, stat_s3, metadata, model_name)
        except Exception:
            continue

        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue

        if dist_mode == "KDE":
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 300)
            h = color.lstrip("#")
            fill_rgba = f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},0.10)"
            fig_s3.add_trace(go.Scatter(
                x=x_range.tolist(),
                y=kde(x_range).tolist(),
                mode="lines",
                name=model_name,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=fill_rgba,
            ))
        else:
            fig_s3.add_trace(go.Histogram(
                x=values.tolist(),
                name=model_name,
                marker_color=color,
                opacity=0.5,
                histnorm="probability density",
                nbinsx=40,
            ))

    if dist_mode == "Histogram":
        fig_s3.update_layout(barmode="overlay")

    fig_s3.update_layout(
        xaxis_title=stat_s3,
        yaxis_title="Density",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_s3, width="content", key="cross_model_s3")

    create_snapshot_button(
        fig=fig_s3,
        metadata={
            "section": "cross_model_section_3",
            "campaign": campaign_name,
            "models": models_selected,
            "weight_type": weight_selected,
            "statistic": stat_s3,
            "display_mode": dist_mode,
        },
        section_name="cross_model_s3",
        key="snapshot_cm_s3",
    )


    ########################################################################
    # Section 4: Corner plot
    ########################################################################
    st.header("Section 4: Corner Plot")

    from plotly.subplots import make_subplots
    from scipy.stats import gaussian_kde

    N = 4
    corner_cols = st.columns(N)
    vars_s4 = [
        corner_cols[k].selectbox(
            f"Variable {k + 1}",
            options=all_stat_options,
            index=min(k, len(all_stat_options) - 1),
            key=f"s4_var_{k}",
        )
        for k in range(N)
    ]
    corner_diag_mode = st.radio(
        "Diagonal display", ["KDE", "Histogram"], horizontal=True, key="s4_diag_mode"
    )

    specs = [[{"type": "xy"}] * N for _ in range(N)]
    fig_s4 = make_subplots(
        rows=N, cols=N, specs=specs,
        horizontal_spacing=0.05, vertical_spacing=0.05,
    )

    legend_added: set[str] = set()

    for i in range(N):
        for j in range(N):
            if j > i:
                # Upper triangle: blank out axes
                fig_s4.update_xaxes(visible=False, showgrid=False, zeroline=False, row=i + 1, col=j + 1)
                fig_s4.update_yaxes(visible=False, showgrid=False, zeroline=False, row=i + 1, col=j + 1)
                continue

            var_x = vars_s4[j]
            var_y = vars_s4[i]

            for model_name in models_selected:
                df_m = df_full.query(
                    f"model == '{model_name}' and weight_type == '{weight_selected}'"
                )
                if df_m.empty:
                    continue

                df_m_sorted = df_m.sort_values(["layer", "head"])
                color = model_colors[model_name]
                show_leg = model_name not in legend_added

                try:
                    x_vals = _extract_values(df_m_sorted, var_x, metadata, model_name)
                except Exception:
                    continue

                if i == j:
                    x_clean = x_vals[np.isfinite(x_vals)]
                    if len(x_clean) == 0:
                        continue

                    if corner_diag_mode == "KDE":
                        kde = gaussian_kde(x_clean)
                        x_range = np.linspace(x_clean.min(), x_clean.max(), 200)
                        h = color.lstrip("#")
                        fill_rgba = f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},0.10)"
                        fig_s4.add_trace(
                            go.Scatter(
                                x=x_range.tolist(),
                                y=kde(x_range).tolist(),
                                mode="lines",
                                name=model_name,
                                line=dict(color=color, width=1.5),
                                fill="tozeroy",
                                fillcolor=fill_rgba,
                                legendgroup=model_name,
                                showlegend=show_leg,
                            ),
                            row=i + 1, col=j + 1,
                        )
                    else:
                        fig_s4.add_trace(
                            go.Histogram(
                                x=x_clean.tolist(),
                                name=model_name,
                                marker_color=color,
                                opacity=0.5,
                                histnorm="probability density",
                                nbinsx=30,
                                legendgroup=model_name,
                                showlegend=show_leg,
                            ),
                            row=i + 1, col=j + 1,
                        )
                else:
                    try:
                        y_vals = _extract_values(df_m_sorted, var_y, metadata, model_name)
                    except Exception:
                        continue

                    fig_s4.add_trace(
                        go.Scatter(
                            x=x_vals.tolist(),
                            y=y_vals.tolist(),
                            mode="markers",
                            name=model_name,
                            marker=dict(color=color, size=3, opacity=0.5),
                            legendgroup=model_name,
                            showlegend=show_leg,
                        ),
                        row=i + 1, col=j + 1,
                    )

                if show_leg:
                    legend_added.add(model_name)

    # Axis labels: x on bottom row, x on diagonal, y on left column (off-diagonal only)
    for k in range(N):
        fig_s4.update_xaxes(title_text=vars_s4[k], row=N, col=k + 1)
        fig_s4.update_xaxes(title_text=vars_s4[k], row=k + 1, col=k + 1)
        if k > 0:
            fig_s4.update_yaxes(title_text=vars_s4[k], row=k + 1, col=1)

    if corner_diag_mode == "Histogram":
        fig_s4.update_layout(barmode="overlay")

    fig_s4.update_layout(
        height=750,
        hovermode="closest",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.78,  # sits in the empty upper-right triangle
        ),
    )
    st.plotly_chart(fig_s4, width="content", key="cross_model_s4")

    create_snapshot_button(
        fig=fig_s4,
        metadata={
            "section": "cross_model_section_4",
            "campaign": campaign_name,
            "models": models_selected,
            "weight_type": weight_selected,
            "variables": vars_s4,
            "diagonal_mode": corner_diag_mode,
        },
        section_name="cross_model_s4",
        key="snapshot_cm_s4",
    )


def render():
    cross_model_app()


if __name__ == "__main__":
    cross_model_app()
