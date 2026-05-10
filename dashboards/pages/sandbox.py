"""Eigenvalue staircase sandbox — standalone, not wired into streamlit_app.py.

Run with: streamlit run dashboards/pages/sandbox.py

Implements RMT-style eigenvalue unfolding and spacing analysis for a single
attention head. Three plots:
  A. Staircase N(λ) — cumulative count of eigenvalues ≤ λ
  B. Unfolded eigenvalues ξᵢ = N̄(λᵢ) — polynomial smooth background
  C. Spacing distribution P(s), sᵢ = ξᵢ₊₁ - ξᵢ
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))
from dashboard_utils import (
    get_available_campaigns,
    get_unique_values,
    is_HF_environment,
    load_dataset_with_metadata,
    model_size_from_name,
)

# ---------------------------------------------------------------------------
# Pure numpy helpers (importable by tests without Streamlit)
# ---------------------------------------------------------------------------


def staircase(ev):
    """Cumulative eigenvalue count N(λ).

    Returns (sorted_ev, N) where N[i] = number of eigenvalues ≤ sorted_ev[i].
    """
    sorted_ev = np.sort(ev)
    N = np.arange(1, len(sorted_ev) + 1, dtype=float)
    return sorted_ev, N


def unfold(sorted_ev, degree=4):
    """Polynomial unfolding of eigenvalue spectrum.

    Fits a degree-`degree` polynomial to the staircase (N vs λ) and evaluates
    it at each eigenvalue to produce the smooth background N̄(λ).

    Returns ξ = N̄(sorted_ev), the unfolded eigenvalues.
    """
    N = np.arange(1, len(sorted_ev) + 1, dtype=float)
    coeffs = np.polyfit(sorted_ev, N, deg=degree)
    xi = np.polyval(coeffs, sorted_ev)
    return xi


def spacings(xi):
    """Nearest-neighbour spacings sᵢ = ξᵢ₊₁ - ξᵢ.

    Returns a (len(xi) - 1,) array of spacings, normalized so that <s> = 1.
    """
    s = np.diff(xi)
    mean_s = s.mean()
    if mean_s > 0:
        s = s / mean_s
    return s


def wigner_dyson(s):
    """Wigner-Dyson (GOE) level-spacing distribution P(s) = (π/2) s exp(-πs²/4)."""
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)


def poisson(s):
    """Poisson (uncorrelated) level-spacing distribution P(s) = exp(-s)."""
    return np.exp(-s)


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------


def sandbox_app():
    st.set_page_config(page_title="Eigenvalue Staircase Sandbox", layout="wide")
    st.title("Eigenvalue Staircase Sandbox")
    st.markdown(
        "RMT-style eigenvalue unfolding and spacing analysis for a single attention head. "
        "This page is experimental and runs standalone — not part of the main dashboard."
    )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    if is_HF_environment():
        df_full, metadata = load_dataset_with_metadata(
            ds_name=None, campaign=None,
            hf_repo_id="angerami/transformer_weights_cross_model"
        )
    else:
        available_datasets = get_available_campaigns("ana-")
        if not available_datasets:
            st.error("No datasets found. Run the analysis pipeline first.")
            st.stop()
        campaign_name = st.sidebar.selectbox("Campaign", available_datasets, index=0)
        df_full, metadata = load_dataset_with_metadata(
            ds_name="weight_study", campaign=campaign_name, hf_version="ana-003"
        )

    model_names = sorted(
        get_unique_values(df_full, "model"),
        key=lambda x: (x.split("-")[0], model_size_from_name(x)),
    )
    model_selected = st.sidebar.selectbox("Model", model_names)

    weight_types = get_unique_values(df_full, "weight_type")
    sv_types = [wt for wt in weight_types if "QK" in wt or "gram" in wt.lower()]
    if not sv_types:
        sv_types = weight_types
    default_wt = "W_QK" if "W_QK" in sv_types else sv_types[0]
    weight_selected = st.sidebar.selectbox(
        "Weight Type", sv_types, index=sv_types.index(default_wt)
    )

    df = df_full.query(
        f"model == '{model_selected}' and weight_type == '{weight_selected}'"
    )
    if df.empty or df["SVD"].isnull().all():
        st.error(f"No SVD data for {model_selected} / {weight_selected}.")
        st.stop()

    d_model = metadata.get("d_model", None)
    n_layers = int(df["layer"].max()) + 1
    n_heads = int(df["head"].max()) + 1

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Layers:** {n_layers}  |  **Heads:** {n_heads}")

    use_eigenvalues = st.sidebar.checkbox(
        "Plot eigenvalues (λ²)", value=True,
        help="Square the singular values to work with eigenvalues."
    )
    poly_degree = st.sidebar.slider("Polynomial degree for unfolding", 2, 6, 4)
    overlay_rmt = st.sidebar.checkbox("Overlay RMT reference curves", value=True)

    col1, col2 = st.columns(2)
    layer_sel = col1.slider("Layer", 0, n_layers - 1, 0)
    head_sel = col2.slider("Head", 0, n_heads - 1, 0)

    entry = df.query(f"layer == {layer_sel} and head == {head_sel}")
    if entry.empty or entry["SVD"].iloc[0] is None:
        st.warning("No SVD data for this head.")
        st.stop()

    sv_raw = np.array(entry["SVD"].iloc[0])
    ev = sv_raw**2 if use_eigenvalues else sv_raw
    spec_label = "λ (eigenvalue)" if use_eigenvalues else "σ (singular value)"

    # ------------------------------------------------------------------
    # Computations
    # ------------------------------------------------------------------
    sorted_ev, N = staircase(ev)
    xi = unfold(sorted_ev, degree=poly_degree)

    # polynomial smooth for overlay on staircase plot
    ev_fine = np.linspace(sorted_ev[0], sorted_ev[-1], 300)
    coeffs = np.polyfit(sorted_ev, N, deg=poly_degree)
    N_smooth = np.polyval(coeffs, ev_fine)

    s = spacings(xi)

    # ------------------------------------------------------------------
    # Plot A: Staircase N(λ)
    # ------------------------------------------------------------------
    fig_a = go.Figure()
    # Step function for actual staircase
    fig_a.add_trace(go.Scatter(
        x=np.repeat(sorted_ev, 2)[1:],
        y=np.repeat(N, 2)[:-1],
        mode="lines", name="N(λ) staircase",
        line=dict(color="steelblue", width=1.5),
    ))
    fig_a.add_trace(go.Scatter(
        x=ev_fine, y=N_smooth,
        mode="lines", name=f"N̄(λ) poly deg={poly_degree}",
        line=dict(color="crimson", width=2, dash="dash"),
    ))
    fig_a.update_layout(
        title="A — Eigenvalue staircase N(λ)",
        xaxis_title=spec_label, yaxis_title="N(λ)",
        legend=dict(orientation="h", y=1.08),
    )

    # ------------------------------------------------------------------
    # Plot B: Unfolded eigenvalues ξᵢ vs index i
    # ------------------------------------------------------------------
    fig_b = go.Figure()
    indices = np.arange(len(xi))
    fig_b.add_trace(go.Scatter(
        x=indices, y=xi,
        mode="markers+lines", name="ξᵢ (unfolded)",
        marker=dict(size=4), line=dict(color="steelblue", width=1),
    ))
    fig_b.add_trace(go.Scatter(
        x=indices, y=indices.astype(float),
        mode="lines", name="ideal (ξᵢ = i)",
        line=dict(color="gray", width=1, dash="dot"),
    ))
    fig_b.update_layout(
        title="B — Unfolded eigenvalues ξᵢ",
        xaxis_title="Index i", yaxis_title="ξᵢ",
        legend=dict(orientation="h", y=1.08),
    )

    # ------------------------------------------------------------------
    # Plot C: Spacing distribution P(s)
    # ------------------------------------------------------------------
    fig_c = go.Figure()
    if len(s) > 2:
        s_max = min(s.max(), 4.0)
        counts, edges = np.histogram(s, bins=max(5, len(s) // 3), range=(0, s_max), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        fig_c.add_trace(go.Bar(
            x=centers, y=counts,
            name="P(s) observed",
            marker_color="steelblue", opacity=0.7,
        ))
        if overlay_rmt:
            s_grid = np.linspace(0, s_max, 200)
            fig_c.add_trace(go.Scatter(
                x=s_grid, y=wigner_dyson(s_grid),
                mode="lines", name="Wigner-Dyson (GOE)",
                line=dict(color="crimson", width=2),
            ))
            fig_c.add_trace(go.Scatter(
                x=s_grid, y=poisson(s_grid),
                mode="lines", name="Poisson",
                line=dict(color="seagreen", width=2, dash="dash"),
            ))
    else:
        fig_c.add_annotation(text="Not enough spacings to plot", showarrow=False)

    fig_c.update_layout(
        title="C — Spacing distribution P(s)",
        xaxis_title="s (normalized spacing)", yaxis_title="P(s)",
        legend=dict(orientation="h", y=1.08),
    )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    st.plotly_chart(fig_a, use_container_width=True)
    st.plotly_chart(fig_b, use_container_width=True)
    st.plotly_chart(fig_c, use_container_width=True)

    with st.expander("Raw values"):
        col_a, col_b, col_c = st.columns(3)
        col_a.write("Sorted eigenvalues")
        col_a.write(sorted_ev)
        col_b.write("Unfolded ξᵢ")
        col_b.write(xi)
        col_c.write("Spacings sᵢ")
        col_c.write(s)


if __name__ == "__main__":
    sandbox_app()
