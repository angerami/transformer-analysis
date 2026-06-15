"""Ultrametricity view — a page of the correlations app.

Explores hierarchical (Parisi/RSB) structure in the head-head overlap matrices
Q_{hh'}: the distances themselves, which heads drive the ultrametricity, how they
relate across layers, whether a tree is even the right model, and which specific
head pairs sit closer than the inferred tree predicts.

`render(exp_dir)` takes an experiment directory (…/outputs/<experiment>) holding
the `ultrametricity/` and `correlations/` outputs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform

from transformer_analysis.ultrametricity import (
    to_distance, synthetic_ultrametric, null_distance_samples)

METRICS = {"Frobenius cosine": "frob_cosine",
           "Jensen-Shannon": "hist_jensen_shannon"}

SUFFIX = "_W_QK_ultrametricity_summary.json"

# Above this many heads, head-resolved plots are too heavy for the browser:
# the dendrogram is truncated and the matrix heatmaps aggregate to layers.
HEAD_LIMIT = 256


def _discover_runs(ultra_dir: Path):
    if not ultra_dir.is_dir():
        return []
    return sorted(p.name[: -len(SUFFIX)] for p in ultra_dir.glob(f"*{SUFFIX}"))


@st.cache_data
def _load(exp_dir: str, run_key: str, metric: str):
    exp = Path(exp_dir)
    prefix = f"{run_key}_W_QK"
    summary = json.loads((exp / "ultrametricity" / f"{prefix}_ultrametricity_summary.json").read_text())
    meta = json.loads((exp / "correlations" / f"{prefix}_metadata.json").read_text())
    ultra = np.load(exp / "ultrametricity" / f"{prefix}_{metric}_ultra.npz")
    Q = np.load(exp / "correlations" / f"{prefix}_Q.npz")[f"Q_{metric}"]
    D = to_distance(Q, metric)
    dims = (meta["head_dim"], meta["d_model"])
    return summary, {k: ultra[k] for k in ultra.files}, D, dims


@st.cache_data
def _null(d_head: int, d_model: int, metric: str):
    return null_distance_samples(d_head, d_model, metric)


def _grid(values, keys, n_layers, n_heads):
    g = np.full((n_layers, n_heads), np.nan)
    for v, (l, h) in zip(values, keys):
        g[int(l), int(h)] = v
    return g


def _layer_blocks(M, keys, n_layers, how):
    """Aggregate an N×N head matrix to layer×layer (mean or sum)."""
    la = np.array([k[0] for k in keys])
    idx = [np.where(la == l)[0] for l in range(n_layers)]
    out = np.full((n_layers, n_layers), np.nan)
    for a, ia in enumerate(idx):
        for b, ib in enumerate(idx):
            if len(ia) and len(ib):
                sub = M[np.ix_(ia, ib)]
                out[a, b] = sub.sum() if how == "sum" else sub.mean()
    return out


def render(exp_dir: Path):
    exp_dir = Path(exp_dir)
    st.title("Ultrametricity")
    st.caption("Strong triangle inequality d(i,k) ≤ max(d(i,j), d(j,k)) on the "
               "head overlap distances — hierarchical / RSB structure.")

    runs = _discover_runs(exp_dir / "ultrametricity")
    if not runs:
        st.warning(f"No ultrametricity outputs in `{exp_dir / 'ultrametricity'}`. "
                   "Run `scripts/run_ultrametricity.py` first.")
        return

    c0, c1 = st.columns(2)
    run_key = c0.selectbox("Model", runs)
    metric_label = c1.selectbox("Distance", list(METRICS.keys()))
    metric = METRICS[metric_label]

    try:
        summary, U, D, dims = _load(str(exp_dir), run_key, metric)
    except (FileNotFoundError, KeyError):
        st.error(f"Metric '{metric}' not available for this run.")
        return

    m = summary["metrics"][metric]
    keys = [tuple(k) for k in U["head_index"]]
    n_layers, n_heads = summary["n_layers"], summary["n_heads"]
    d_head, d_model = dims
    n = len(keys)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cophenetic corr", f"{m['cophenetic_corr']:.3f}",
              help="Global ultrametricity: 1 = perfect tree")
    k2.metric("Mean Rammal u", f"{m['mean_u']:.3f}", help="0 = perfectly ultrametric")
    k3.metric("Frac. ultrametric", f"{m['frac_ultrametric']:.1%}")
    k4.metric("Triangle violations", f"{m['triangle_violation_frac']:.1%}",
              help="≈0 confirms a proper metric")
    if m["sampled"]:
        st.caption(f"Triples sampled ({m['n_triples']:,}).")

    # ── Distances: d (data) and C (tree) ──
    st.subheader("Distances: d (data) vs C (tree)")
    st.caption("d_ij is the metric distance between heads; C_ij is the cophenetic "
               "distance from the average-linkage tree (the merge height of the two "
               "heads' lowest common ancestor, detailed below). The tree is "
               "ultrametric ⇔ C = d. d = chordal √(2(1−cos)) for Frobenius, √JSD for JS.")
    iu = np.triu_indices(n, k=1)
    Cmat = squareform(cophenet(U["linkage"], squareform(D, checks=False))[1])
    dvals, cvals = D[iu], Cmat[iu]

    # heatmaps: d and C share a color scale for comparison
    if n > HEAD_LIMIT:
        zD = _layer_blocks(D, keys, n_layers, "mean")
        zC = _layer_blocks(Cmat, keys, n_layers, "mean")
        ax, note = "layer", f" (layer means, {n} heads)"
    else:
        zD, zC, ax, note = D, Cmat, "head index", ""
    vmax = float(max(np.nanmax(zD), np.nanmax(zC)))
    h1, h2 = st.columns(2)
    for col, z, title in ((h1, zD, "d_ij (data)"), (h2, zC, "C_ij (tree)")):
        fh = go.Figure(go.Heatmap(z=z, colorscale="Viridis", zmin=0, zmax=vmax,
                                  colorbar_title="dist"))
        fh.update_layout(title=title + note, xaxis_title=ax, yaxis_title=ax,
                         height=400, yaxis=dict(autorange="reversed"))
        col.plotly_chart(fh, use_container_width=True)

    # corner plot of (d, C): marginals on the diagonal, joint below-left
    st.caption("Corner plot of every head pair: P(d) (top) and P(C) (right) marginals "
               "with the joint C-vs-d density (lower-left). A perfectly ultrametric "
               "tree lies on C = d; spread around it is the cophenetic shortfall. "
               "Ranges are clipped to the support; the untrained null overlays P(d).")
    log1d = st.checkbox("log density (1D marginals)", value=True, key="corner_log")
    null = _null(int(d_head), int(d_model), metric)

    dmin, dmax = float(dvals.min()), float(dvals.max())
    cmin, cmax = float(cvals.min()), float(cvals.max())
    dr = [dmin - 0.02 * (dmax - dmin + 1e-9), dmax + 0.02 * (dmax - dmin + 1e-9)]
    cr = [cmin - 0.02 * (cmax - cmin + 1e-9), cmax + 0.02 * (cmax - cmin + 1e-9)]
    H, xe, ye = np.histogram2d(dvals, cvals, bins=60, range=[dr, cr])
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])

    fig = make_subplots(rows=2, cols=2, column_widths=[0.78, 0.22],
                        row_heights=[0.22, 0.78], horizontal_spacing=0.015,
                        vertical_spacing=0.015, shared_xaxes=True, shared_yaxes=True,
                        specs=[[{}, None], [{}, {}]])
    fig.add_histogram(x=dvals, nbinsx=60, histnorm="probability density", name="P(d)",
                      marker_color="#00CC96", opacity=0.75, row=1, col=1)
    if null is not None and len(null):
        hn, en = np.histogram(null, bins=60, density=True)
        fig.add_scatter(x=0.5 * (en[:-1] + en[1:]), y=hn, mode="lines",
                        name="untrained null", line=dict(color="#FF6692", width=2),
                        row=1, col=1)
    fig.add_trace(go.Heatmap(x=xc, y=yc, z=np.log10(H.T + 1.0), colorscale="Inferno",
                             colorbar=dict(title="log₁₀(pairs+1)", len=0.55, y=0.36,
                                           x=1.02)),
                  row=2, col=1)
    lo, hi = max(dr[0], cr[0]), min(dr[1], cr[1])
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                    line=dict(color="#fff", dash="dash"), row=2, col=1)
    fig.add_histogram(y=cvals, nbinsy=60, histnorm="probability density",
                      marker_color="#AB63FA", showlegend=False, row=2, col=2)

    if metric == "frob_cosine":
        for r in (1, 2):
            fig.add_vline(x=np.sqrt(2), line_dash="dash", line_color="#888", row=r, col=1)
    fig.update_xaxes(range=dr, row=2, col=1, title_text="d (data)")
    fig.update_yaxes(range=cr, row=2, col=1, title_text="C (tree)")
    fig.update_yaxes(type="log" if log1d else "linear", row=1, col=1, title_text="P(d)")
    fig.update_xaxes(type="log" if log1d else "linear", row=2, col=2, title_text="P(C)")
    fig.update_layout(height=600, bargap=0,
                      title=f"d, C corner  (cophenetic corr {m['cophenetic_corr']:.3f})",
                      legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                                  bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-head ultrametricity map ──
    st.subheader("Per-head ultrametricity")
    st.caption("Mean Rammal u over triples containing each head — which heads "
               "participate in ultrametric structure (low = strongly hierarchical).")
    g = _grid(U["per_head_u"], keys, n_layers, n_heads)
    fig = go.Figure(go.Heatmap(z=g, colorscale="Viridis_r", colorbar_title="u",
                               hovertemplate="layer %{y}, head %{x}<br>u=%{z:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="head", yaxis_title="layer", height=420,
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    # ── P(u) and u vs layer span ──
    cA, cB = st.columns(2)
    edges, counts = U["u_hist_edges"], U["u_hist_counts"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    figu = go.Figure(go.Bar(x=centers, y=counts, marker_color="#636EFA"))
    figu.update_layout(title="P(u)", xaxis_title="Rammal index u",
                       yaxis_title="triples", height=340, bargap=0)
    cA.plotly_chart(figu, use_container_width=True)

    span, span_u = U["span_grid"], U["span_mean_u"]
    figs = go.Figure(go.Scatter(x=span, y=span_u, mode="lines+markers",
                                marker_color="#EF553B"))
    figs.update_layout(title="⟨u⟩ vs layer span",
                       xaxis_title="layer span (max − min layer of the triple)",
                       yaxis_title="⟨u⟩", height=340)
    cB.plotly_chart(figs, use_container_width=True)
    st.caption("**P(u)**: distribution of the Rammal index over head triples "
               "(u=0 ⇒ the two largest sides are equal ⇒ perfectly ultrametric). "
               "**⟨u⟩ vs layer span**: triples are binned by how many layers they "
               "straddle (span 0 = all three heads in one layer; large span = "
               "early+late mix). A trend means the hierarchy is organized by depth.")
    strat = m.get("stratum_mean_u", {})
    if strat:
        st.caption("⟨u⟩ by number of distinct layers in the triple — " +
                   ", ".join(f"{k}-layer: {v:.3f}" for k, v in strat.items()))

    # ── Dendrogram ──
    labels = [f"L{l}H{h}" for (l, h) in keys]
    st.subheader("Head hierarchy (average linkage)")
    # NOTE: the per-head dendrogram is disabled for large models — even truncated it
    # is too heavy, and a per-head tree is the wrong view at that scale. Needs a
    # coarse-grained (layer-level) tree before re-enabling. See git history for the
    # truncated version.
    if n > HEAD_LIMIT:
        st.caption(f"Dendrogram disabled for large models ({n} heads) — a per-head tree "
                   "is not the right representation at this scale (TODO: layer-level "
                   "tree). Use the layer-resolved maps, spectra, and corner plot instead.")
    else:
        st.caption("Agglomerative clustering of heads on the distance matrix: leaves "
                   "are heads (L{layer}H{head}), join height = the average distance at "
                   "which two groups merge. This is the ultrametric tree the cophenetic "
                   "correlation scores against; tick labels are colored by layer.")
        fig_d, axd = plt.subplots(figsize=(12, 4))
        dendrogram(U["linkage"], ax=axd, labels=labels, leaf_font_size=6,
                   color_threshold=0, above_threshold_color="#444")
        norm = plt.Normalize(0, max(n_layers - 1, 1))
        for t in axd.get_xticklabels():
            try:
                ly = int(t.get_text()[1:].split("H")[0])
                t.set_color(cm.viridis(norm(ly)))
            except (ValueError, IndexError):
                pass
        axd.set_ylabel("linkage distance")
        st.pyplot(fig_d)
        plt.close(fig_d)

    with st.expander("Reference: perfect ultrametric tree"):
        ref_noise = st.slider("Noise", 0.0, 1.0, 0.0, 0.05,
                              help="0 = perfectly ultrametric (cophenetic corr = 1)")
        Dref, kref = synthetic_ultrametric(levels=4, noise=ref_noise)
        Zref = linkage(squareform(Dref, checks=False), "average")
        fig_r, axr = plt.subplots(figsize=(10, 3.2))
        dendrogram(Zref, ax=axr, color_threshold=0, above_threshold_color="#444",
                   labels=[f"{b}.{i}" for (b, i) in kref], leaf_font_size=6)
        axr.set_ylabel("linkage distance")
        st.pyplot(fig_r)
        plt.close(fig_r)
        st.caption("At noise 0 every clade merges in clean nested blocks at identical "
                   "heights — cophenetic corr = 1. Raise noise to watch the ideal "
                   "degrade toward a real model's tree.")

    # ── Is it a tree? Model comparison ──
    st.subheader("Is it a tree? Model comparison")
    st.caption("Ultrametric (3-point) vs additive tree (4-point) vs flat Euclidean. "
               "Lower u = more ultrametric; lower u4 = more additive. If u4 ≪ u the "
               "topology is tree-like but the equal-depth (clock) assumption is wrong.")
    mc = st.columns(4)
    mc[0].metric("mean u (3-pt)", f"{m['mean_u']:.3f}")
    mc[1].metric("mean u4 (4-pt)", f"{m.get('mean_u4', float('nan')):.3f}")
    mc[2].metric("frac additive", f"{m.get('frac_additive', float('nan')):.1%}")
    mc[3].metric("δ / diam", f"{m.get('delta_rel_mean', float('nan')):.3f}",
                 help="Gromov 4-point hyperbolicity; small = tree-like")

    cC, cD = st.columns(2)
    if "u4_hist_counts" in U:
        e = U["u_hist_edges"]; cu = 0.5 * (e[:-1] + e[1:])
        e4 = U["u4_hist_edges"]; cu4 = 0.5 * (e4[:-1] + e4[1:])
        hu = U["u_hist_counts"] / max(U["u_hist_counts"].sum(), 1)
        hu4 = U["u4_hist_counts"] / max(U["u4_hist_counts"].sum(), 1)
        figc = go.Figure()
        figc.add_scatter(x=cu, y=hu, name="u (3-pt)", line_shape="hv",
                         fill="tozeroy", line=dict(color="#636EFA"),
                         fillcolor="rgba(99,110,250,0.35)")
        figc.add_scatter(x=cu4, y=hu4, name="u4 (4-pt)", line_shape="hv",
                         fill="tozeroy", line=dict(color="#00CC96"),
                         fillcolor="rgba(0,204,150,0.35)")
        figc.update_layout(title="P(u) vs P(u4)", height=320,
                           xaxis_title="index", yaxis_title="fraction",
                           legend=dict(x=0.6, y=0.98))
        cC.plotly_chart(figc, use_container_width=True)
    lkd = m.get("linkage_cophenetic", {})
    if lkd:
        figl = go.Figure(go.Bar(x=list(lkd.keys()), y=list(lkd.values()),
                                marker_color="#AB63FA"))
        figl.update_layout(title="cophenetic corr by linkage rule", height=320,
                           yaxis_title="corr", yaxis_range=[0, 1])
        cD.plotly_chart(figl, use_container_width=True)
        st.caption("Average linkage above the others means the ultrametric shortfall "
                   "is real, not a greedy-algorithm artifact (single = subdominant "
                   "ultrametric).")

    # ── Tree-residual specific pairs ──
    st.subheader("Specific pairs (closer than the tree)")
    st.caption("Most negative D−C residuals: head pairs more correlated than the "
               "hierarchy explains — specific correlations layered on top of RSB.")
    rp = pd.DataFrame(m["residual_pairs"])
    cR1, cR2 = st.columns([3, 2])
    cR1.dataframe(rp[["layer_1", "head_1", "layer_2", "head_2",
                      "delta_layer", "distance", "residual"]].head(25),
                  use_container_width=True, height=360)
    figr = go.Figure(go.Scatter(
        x=rp["delta_layer"], y=rp["residual"], mode="markers",
        marker=dict(color=rp["distance"], colorscale="Plasma", showscale=True,
                    colorbar_title="d"),
        text=[f"L{a}H{b}–L{c}H{d}" for a, b, c, d in
              zip(rp.layer_1, rp.head_1, rp.layer_2, rp.head_2)],
        hovertemplate="%{text}<br>Δlayer=%{x}<br>residual=%{y:.3f}<extra></extra>"))
    figr.update_layout(title="residual vs Δlayer", xaxis_title="Δlayer",
                       yaxis_title="D−C", height=360)
    cR2.plotly_chart(figr, use_container_width=True)

    # ── Residual structure: D = C + R ──
    st.subheader("Residual structure (D = C + R)")
    rs = m.get("residual")
    if rs is None:
        st.caption(f"Skipped — {n} heads exceeds the dense-analysis limit.")
    else:
        b = st.columns(4)
        b[0].metric("residual energy", f"{rs['res_energy_frac']:.1%}", help="‖R‖ / ‖D‖")
        b[1].metric("effective rank", f"{rs['eff_rank']:.1f}")
        b[2].metric("Gini(|R|)", f"{rs['gini']:.2f}")
        b[3].metric("verdict", rs["verdict"])

        R = D - Cmat
        rvals = R[iu]
        rabs = float(np.abs(rvals).max()) or 1.0

        # R heatmap and P(R)
        rh, rp = st.columns([3, 2])
        zR = _layer_blocks(R, keys, n_layers, "mean") if n > HEAD_LIMIT else R
        figR = go.Figure(go.Heatmap(z=zR, colorscale="RdBu", zmid=0,
                                    colorbar_title="R"))
        figR.update_layout(title="R = D − C" + (f" (layer means, {n} heads)"
                                                if n > HEAD_LIMIT else ""),
                           xaxis_title="head index", yaxis_title="head index",
                           height=380, yaxis=dict(autorange="reversed"))
        rh.plotly_chart(figR, use_container_width=True)
        logR = rp.checkbox("log y", value=True, key="pr_logy")
        figpr = go.Figure(go.Histogram(x=rvals, nbinsx=60, marker_color="#FF6692"))
        figpr.update_layout(title="P(R)", xaxis_title="residual R", yaxis_title="pairs",
                            height=380, yaxis_type="log" if logR else "linear")
        rp.plotly_chart(figpr, use_container_width=True)

        # scree of R eigenvalues
        spec = U.get("resid_eigvals_sorted", np.array([]))
        if spec.size:
            figsc = go.Figure(go.Scatter(x=list(range(1, len(spec) + 1)),
                                         y=np.abs(spec), mode="lines+markers",
                                         marker_color="#FF6692"))
            figsc.add_vline(x=rs["eff_rank"], line_dash="dash", line_color="#888",
                            annotation_text="eff. rank")
            figsc.update_layout(title="R eigenvalue scree (|λ| by rank)", height=320,
                                xaxis_title="index", yaxis_title="|λ|",
                                yaxis_type="log")
            st.plotly_chart(figsc, use_container_width=True)

        # top-N residual eigenvectors as head-space maps (N = effective rank)
        vecs = U.get("resid_top_eigvecs", np.zeros((0, 0)))
        if vecs.size:
            n_show = int(min(max(round(rs["eff_rank"]), 1), vecs.shape[1]))
            st.caption(f"Top {n_show} residual eigenvectors (R modes) on the layer×head "
                       "grid — the head-space patterns of the beyond-tree structure. "
                       "These are the vectors to later project onto the Q_hh eigenmodes.")
            cols = st.columns(n_show)
            for j in range(n_show):
                v = vecs[:, j]
                gv = _grid(v, keys, n_layers, n_heads)
                vmaxv = float(np.nanmax(np.abs(gv))) or 1.0
                figv = go.Figure(go.Heatmap(z=gv, colorscale="RdBu", zmid=0,
                                            zmin=-vmaxv, zmax=vmaxv, showscale=False))
                figv.update_layout(title=f"v{j+1} (λ={U['resid_top_eigs'][j]:+.3f})",
                                   height=300, margin=dict(l=8, r=8, t=30, b=8),
                                   yaxis=dict(autorange="reversed"))
                cols[j].plotly_chart(figv, use_container_width=True)

        st.caption("R is what the ultrametric tree leaves out. Low effective rank with "
                   "energy in a few eigenvalues ⇒ a handful of global factors on top of "
                   "the tree; high Gini/kurtosis ⇒ sparse specific pairs; otherwise noise.")

    # ── Flat geometry alternative: classical MDS ──
    st.subheader("Euclidean alternative (classical MDS)")
    em = m.get("embedding")
    if em is None:
        st.caption("Skipped for large N.")
    else:
        d = st.columns(4)
        d[0].metric("var in 2D", f"{em['var_frac_2d']:.1%}")
        d[1].metric("var in 3D", f"{em['var_frac_3d']:.1%}")
        d[2].metric("eff. dim", f"{em['eff_dim']:.1f}")
        d[3].metric("non-Euclidean", f"{em['neg_energy_frac']:.1%}",
                    help="negative-eigenvalue energy; large ⇒ not flat-Euclidean")
        mds = U.get("mds_eigs", np.array([]))
        if mds.size:
            figm = go.Figure(go.Bar(x=list(range(1, len(mds) + 1)), y=mds,
                                    marker_color="#19D3F3"))
            figm.update_layout(title="MDS eigenvalues (scree)", height=320,
                               xaxis_title="dimension", yaxis_title="eigenvalue")
            st.plotly_chart(figm, use_container_width=True)

        # leading principal coordinates as head-space maps
        mvecs = U.get("mds_top_eigvecs", np.zeros((0, 0)))
        if mvecs.size:
            n_show = int(min(6, mvecs.shape[1], max(round(em["eff_dim"]), 1)))
            st.caption(f"Leading {n_show} MDS principal coordinates on the layer×head "
                       "grid — the flat-embedding axes of the head geometry.")
            cols = st.columns(n_show)
            for j in range(n_show):
                gv = _grid(mvecs[:, j], keys, n_layers, n_heads)
                vmaxv = float(np.nanmax(np.abs(gv))) or 1.0
                figv = go.Figure(go.Heatmap(z=gv, colorscale="RdBu", zmid=0,
                                            zmin=-vmaxv, zmax=vmaxv, showscale=False))
                figv.update_layout(title=f"v{j+1} (λ={U['mds_top_eigs'][j]:+.2g})",
                                   height=300, margin=dict(l=8, r=8, t=30, b=8),
                                   yaxis=dict(autorange="reversed"))
                cols[j].plotly_chart(figv, use_container_width=True)

        st.caption("High variance in 2–3 dims with a low non-Euclidean fraction ⇒ a flat "
                   "low-dim embedding explains D better than any tree.")

    # ── Tight-pair co-occurrence ──
    st.subheader("Tight-pair co-occurrence")
    if n > HEAD_LIMIT:
        z = _layer_blocks(U["cooccur"], keys, n_layers, "sum")
        axis_title = "layer"
        st.caption("How often head pairs form the closest (sibling) edge of a "
                   f"near-ultrametric triple, aggregated to layer×layer ({n} heads).")
    else:
        z = U["cooccur"]
        axis_title = "head index"
        st.caption("How often each head pair forms the closest (sibling) edge of a "
                   "near-ultrametric triple.")
    fig2 = go.Figure(go.Heatmap(z=z, colorscale="Inferno", colorbar_title="count"))
    fig2.update_layout(xaxis_title=axis_title, yaxis_title=axis_title,
                       height=480, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Per-head drill-down ──
    st.subheader("Head drill-down")
    sel = st.selectbox("Head", labels)
    si = labels.index(sel)
    order = np.argsort(D[si])
    near = [(labels[j], float(D[si, j])) for j in order if j != si][:12]
    nn = pd.DataFrame(near, columns=["head", "distance"])
    st.caption(f"Nearest heads to {sel} (per-head u = {U['per_head_u'][si]:.3f}).")
    st.dataframe(nn, use_container_width=True, height=300)
