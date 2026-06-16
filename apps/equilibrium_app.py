"""Streamlit app for QK/OV equilibrium observables.

Treats the QK circuit as an interaction (W_QK = symmetric Hamiltonian S +
antisymmetric circulating part A) and the OV circuit as a state
(rho_OV = W_OV W_OV^T / Tr), and visualizes how far each head sits from
equilibrium. Three sections mirror the observable families:

  1. Cross-circuit equilibrium — the commutator C = [rho_OV, W_QK] (Theta).
  2. QK irreversibility — antisymmetric fraction f_A and non-normality nu.
  3. Spectra & element statistics of S and A.

Discovers runs automatically from the output directory:
  - Output root: $DATA_PATH if set, otherwise ./outputs
  - Runs: any `*_equilibrium_spectra.npz` under the root (with a sibling
    `*_scalars.csv`), produced by `run_correlations.py --equilibrium`.

Usage:
    streamlit run apps/equilibrium_app.py
    DATA_PATH=eq_outputs streamlit run apps/equilibrium_app.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import streamlit as st

st.set_page_config(
    page_title="QK/OV Equilibrium",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_OUTPUT_DIR = Path(os.environ.get("DATA_PATH", "outputs"))
SUFFIX = "_equilibrium_spectra.npz"

# Plotly default qualitative palette (matches the other study apps)
BLUE, RED, GREEN, PURPLE, ORANGE, CYAN, PINK = (
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692")


# ── Discovery & loading ───────────────────────────────────────────────

def discover_runs(root: Path):
    return sorted(root.rglob(f"*{SUFFIX}")) if root.is_dir() else []


@st.cache_data(show_spinner="Loading equilibrium outputs...")
def load_run(npz_path: str):
    p = Path(npz_path)
    run_key = p.name[: -len(SUFFIX)]
    z = np.load(p)
    arrays = {k: z[k] for k in z.files}
    df = (pd.read_csv(p.parent / f"{run_key}_scalars.csv")
          .sort_values(["layer", "head"]).reset_index(drop=True))
    return run_key, df, arrays


# ── Plot helpers ──────────────────────────────────────────────────────

def _grid(df, col, n_layers, n_heads):
    g = df.pivot_table(index="layer", columns="head", values=col)
    return g.reindex(index=range(n_layers), columns=range(n_heads)).values


def _heatmap(z, title, colorscale="Viridis", zmid=None, cbar=""):
    fig = go.Figure(go.Heatmap(
        z=z, colorscale=colorscale, zmid=zmid, colorbar_title=cbar,
        hovertemplate="layer %{y}, head %{x}<br>%{z:.4g}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="head", yaxis_title="layer",
                      height=380, yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _hist(vals, title, xtitle, color, vlines=(), log_y=False):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    fig = go.Figure(go.Histogram(x=vals, nbinsx=40, marker_color=color))
    for x in vlines:
        fig.add_vline(x=x, line_dash="dash", line_color="#888")
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title="heads",
                      height=340, bargap=0.02, margin=dict(l=10, r=10, t=40, b=10),
                      yaxis_type="log" if log_y else "linear")
    return fig


def _spectral_hist(vals, title, xtitle, color):
    """Histogram of a spectrum; clip the y-scale to the non-zero bins and
    annotate the pile-up bin (the small/near-zero modes otherwise wipe out
    the scale of the modes that carry the structure)."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    fig = go.Figure()
    if vals.size == 0:
        return fig.update_layout(title=title, height=320)
    counts, edges = np.histogram(vals, bins=40)
    centers = 0.5 * (edges[:-1] + edges[1:])
    zbin = int(np.argmin(np.abs(centers)))           # bin nearest zero
    others = np.delete(counts, zbin)
    ymax = float(others.max()) if others.size and others.max() > 0 else float(counts.max())
    fig.add_bar(x=centers, y=counts, width=edges[1] - edges[0], marker_color=color)
    fig.add_vline(x=0, line_dash="dash", line_color="#aaa")
    if counts[zbin] > ymax:
        fig.add_annotation(x=centers[zbin], y=ymax, ay=-26,
                           text=f"{int(counts[zbin])} @≈{centers[zbin]:.2g}",
                           font=dict(size=11, color=color), arrowhead=2)
        fig.update_yaxes(range=[0, ymax * 1.15])
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title="modes",
                      height=320, bargap=0.02, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# Spectral stats reuse head_metrics' singular-value battery; for S the values
# fed in are |λ(S)| and for A they are ω_k (rotation rates = singular values of
# A). The `sv_*` columns are those magnitudes — relabel for clarity.
SPEC_BASE = {"S": "|λ|", "A": "ω"}
SUFFIX_NICE = {
    "sv_mean": "mean", "sv_variance": "variance", "sv_skewness": "skewness",
    "sv_kurtosis": "kurtosis", "sv_sum": "sum", "sv_sum_squares": "sum of squares",
    "participation_ratio": "participation ratio",
    "normalized_participation_ratio": "norm. participation ratio",
    "spectral_entropy": "spectral entropy", "condition_number": "condition number",
    "stable_rank": "stable rank",
    "sum": "element sum", "mean": "element mean", "std": "element std",
    "max": "element max", "min": "element min", "skew": "element skew",
    "kurtosis": "element kurtosis", "differential_entropy": "element diff. entropy",
    "entropy": "element entropy", "fit_mu": "fit μ", "fit_sigma": "fit σ",
    "kl_vs_empirical_normal": "KL vs normal",
}


def _suffix_label(suf):
    return SUFFIX_NICE.get(suf, suf.replace("_", " "))


def _overlay_label(suf):
    base = _suffix_label(suf)
    return f"{base} of |λ|/ω" if suf.startswith("sv_") else base


def _stat_label(col):
    part, suf = col[0], col[2:]
    base = _suffix_label(suf)
    return f"{part} · {base} of {SPEC_BASE[part]}" if suf.startswith("sv_") \
        else f"{part} · {base}"


SCALAR_LABEL = {
    "Theta": "Θ", "Theta_supp": "Θ_supp", "Theta_curr": "Θ_curr",
    "Theta_full": "Θ_full", "f_A": "f_A", "nu": "ν", "nu_norm": "ν_norm",
    "nu_align": "ν_align", "norm_C": "‖C‖", "norm_C_anti": "‖C_anti‖",
    "norm_C_sym": "‖C_sym‖", "norm_rho_OV": "‖ρ_OV‖", "norm_W_QK": "‖W_QK‖",
    "norm_S": "‖S‖", "norm_A": "‖A‖", "norm_PSP": "‖PSP‖",
    "decomp_within": "decomp within", "decomp_cross": "decomp cross",
}


def _var_label(col):
    return _stat_label(col) if col.startswith(("S_", "A_")) else SCALAR_LABEL.get(col, col)


# ── Sections ──────────────────────────────────────────────────────────

def section_overview(run_key, df, arrays):
    n_layers, n_heads = int(df.layer.max()) + 1, int(df["head"].max()) + 1
    d_model = arrays["lambda_S"].shape[1]
    st.title("QK/OV equilibrium")
    st.caption(
        "Per head, QK is read as an interaction (W_QK = S + A, a symmetric "
        "Hamiltonian S plus an antisymmetric circulating part A) and OV as a state "
        "(rho_OV = W_OV W_OVᵀ / Tr). Distance from equilibrium is the failure of "
        "rho_OV to commute with the QK energy. All quantities are weight-only.")
    c = st.columns(5)
    c[0].metric("Heads", f"{len(df)}")
    c[1].metric("Layers × heads", f"{n_layers} × {n_heads}")
    c[2].metric("d_model", f"{d_model}")
    c[3].metric("median Θ", f"{np.nanmedian(df['Theta']):.3f}",
                help="Energy mismatch ‖[ρ_OV,S]‖ / (‖ρ_OV‖‖S‖); 0 = equilibrium")
    c[4].metric("median f_A", f"{np.nanmedian(df['f_A']):.3f}",
                help="Antisymmetric (non-reciprocal) fraction of W_QK")
    return n_layers, n_heads, d_model


def section_commutator(df, n_layers, n_heads):
    st.header("1 · Cross-circuit equilibrium")
    st.caption(
        "C = [ρ_OV, W_QK], split by symmetry into C_anti = [ρ_OV, S] (energy "
        "mismatch) and C_sym = [ρ_OV, A] (current alignment). **Θ** = ‖C_anti‖ / "
        "(‖ρ_OV‖‖S‖) is distance from equilibrium: Θ = 0 means ρ_OV and the energy S "
        "are simultaneously diagonalizable (Gibbs). But Θ is dominated by the trivial "
        "**support↔null leakage** — ρ_OV is rank ≤ head_dim, so S mostly couples OV's "
        "tiny used subspace to the huge unused complement (a random S gives the same "
        "≈0.98 leakage fraction). **Θ_supp** restricts to ρ_OV's support — equilibrium "
        "*within the subspace OV actually uses* — and is the part that carries real, "
        "weights-specific signal. NaN denotes a head with ‖S‖ or ‖A‖ ≈ 0.")

    df = df.copy()
    denom = df["decomp_within"] + df["decomp_cross"]
    df["null_leak_frac"] = np.where(denom > 0, df["decomp_cross"] / denom, np.nan)
    has_supp = "Theta_supp" in df.columns

    k = st.columns(4)
    k[0].metric("median Θ", f"{np.nanmedian(df['Theta']):.3f}",
                help="Full energy mismatch; rank-mismatch dominated")
    if has_supp:
        k[1].metric("median Θ_supp", f"{np.nanmedian(df['Theta_supp']):.3f}",
                    help="Equilibrium within ρ_OV's support — the meaningful part")
    k[2].metric("median Θ_full", f"{np.nanmedian(df['Theta_full']):.3f}")
    k[3].metric("median null-leak", f"{np.nanmedian(df['null_leak_frac']):.3f}",
                help="Fraction of ‖[ρ_OV,S]‖² leaking to OV's null space (≈0.98 "
                     "is the generic rank-deficiency baseline)")

    choices = {"Θ — energy mismatch (full)": ("Theta", "Viridis", None)}
    if has_supp:
        choices["Θ_supp — equilibrium within OV's support"] = ("Theta_supp", "Viridis", None)
    choices.update({
        "Θ_curr — current alignment": ("Theta_curr", "Viridis", None),
        "Θ_full — full commutator": ("Theta_full", "Viridis", None),
        "Null-leakage fraction of ‖[ρ_OV,S]‖²": ("null_leak_frac", "Magma", None),
    })
    default_idx = 1 if has_supp else 0
    label = st.selectbox("Map", list(choices), index=default_idx, key="f1_map")
    col, cs, zmid = choices[label]

    a, b = st.columns([3, 2])
    a.plotly_chart(_heatmap(_grid(df, col, n_layers, n_heads), label,
                            colorscale=cs, zmid=zmid, cbar=col),
                   use_container_width=True)
    b.plotly_chart(_hist(df[col], f"P({col})", col, BLUE),
                   use_container_width=True)
    if col == "null_leak_frac":
        st.caption("Splits ‖[ρ_OV,S]‖² into reorganization *within* the directions OV "
                   "uses vs. leakage into OV's unused null directions. ≈0.98 for every "
                   "head — and a random S reproduces it — so this is rank-deficiency "
                   "geometry, not learned structure. Θ_supp isolates the within part.")

    st.plotly_chart(
        go.Figure(go.Scatter(
            x=df["layer"], y=df["Theta"], mode="markers",
            marker=dict(color=df["f_A"], colorscale="Plasma", showscale=True,
                        colorbar_title="f_A", size=7),
            text=[f"L{l}H{h}" for l, h in zip(df.layer, df["head"])],
            hovertemplate="%{text}<br>layer=%{x}<br>Θ=%{y:.4f}<extra></extra>"))
        .update_layout(title="Θ vs layer (colored by f_A)", xaxis_title="layer",
                       yaxis_title="Θ", height=340, margin=dict(l=10, r=10, t=40, b=10)),
        use_container_width=True)


def section_irreversibility(df, n_layers, n_heads):
    st.header("2 · QK irreversibility (weight-only)")
    st.caption(
        "**f_A** = ‖A‖²/‖W_QK‖² is the non-reciprocal fraction of the interaction: "
        "0 = fully reciprocal, ½ = equipartition, 1 = pure circulation. **ν** = "
        "‖[W_QK, W_QKᵀ]‖ is non-normality. Since [W_QK, W_QKᵀ] = 2[A, S] (the S² and "
        "A² parts cancel), ν = 2‖[A, S]‖ is purely an S–A interaction: it needs both "
        "parts present *and* non-commuting. So **ν_align** = ‖[A, S]‖/(‖A‖‖S‖) is the "
        "magnitude-free piece independent of f_A — as f_A → 1, S → 0 and ν itself "
        "collapses (the pure-rotation limit), so over the full range ν is "
        "non-monotonic in f_A, not simply correlated.")

    df = df.copy()
    if "nu_align" not in df.columns:
        denom = 2 * df["norm_A"] * df["norm_S"]
        df["nu_align"] = np.where(denom > 0, df["nu"] / denom, np.nan)

    k = st.columns(3)
    k[0].metric("median f_A", f"{np.nanmedian(df['f_A']):.3f}")
    k[1].metric("median ν_norm", f"{np.nanmedian(df['nu_norm']):.3f}")
    k[2].metric("median ν_align", f"{np.nanmedian(df['nu_align']):.3f}",
                help="‖[A,S]‖/(‖A‖‖S‖); 0 ⇔ A and S commute (W_QK normal)")

    nu_choices = {"ν_align — A↔S alignment (magnitude-free)": "nu_align",
                  "ν / ‖W_QK‖² — non-normality": "nu_norm"}
    nu_label = st.selectbox("Non-normality measure", list(nu_choices), key="f2_nu")
    nu_col = nu_choices[nu_label]

    a, b = st.columns(2)
    a.plotly_chart(_heatmap(_grid(df, "f_A", n_layers, n_heads),
                            "f_A — antisymmetric fraction", colorscale="RdBu",
                            zmid=0.5, cbar="f_A"), use_container_width=True)
    b.plotly_chart(_heatmap(_grid(df, nu_col, n_layers, n_heads), nu_label,
                            colorscale="Cividis", cbar=nu_col),
                   use_container_width=True)

    a, b = st.columns(2)
    a.plotly_chart(
        go.Figure(go.Scatter(
            x=df["f_A"], y=df[nu_col], mode="markers",
            marker=dict(color=df["layer"], colorscale="Turbo", showscale=True,
                        colorbar_title="layer", size=7),
            text=[f"L{l}H{h}" for l, h in zip(df.layer, df["head"])],
            hovertemplate="%{text}<br>f_A=%{x:.3f}<br>"
                          + nu_col + "=%{y:.3f}<extra></extra>"))
        .update_layout(title=f"f_A vs {nu_col} (independence check)",
                       xaxis_title="f_A", yaxis_title=nu_col, height=360,
                       margin=dict(l=10, r=10, t=40, b=10)),
        use_container_width=True)
    b.plotly_chart(_hist(df["f_A"], "P(f_A)", "f_A", GREEN, vlines=(0.5,)),
                   use_container_width=True)


def section_spectra(df, arrays, n_layers, n_heads):
    st.header("3 · Spectra & element statistics of S and A")
    st.caption(
        "S is symmetric indefinite — its **signed** eigenvalues are the energy "
        "density of states (asymmetry about 0 is the element-mean/skew of W_QK, which "
        "lives entirely in S). A is antisymmetric — its spectrum is the set of "
        "rotation rates ω_k. Element distributions: A has zero diagonal and zero "
        "mean by construction.")

    labels = [f"L{int(l)}H{int(h)}" for l, h in zip(df.layer, df["head"])]
    arr_idx = {(int(l), int(h)): i for i, (l, h) in enumerate(arrays["head_index"])}
    sel = st.selectbox("Head", labels, key="f3_head")
    l, h = (int(x) for x in sel[1:].split("H"))
    i = arr_idx[(l, h)]

    lam = arrays["lambda_S"][i]
    omega = arrays["omega_A"][i]
    tol = 1e-6 * max(np.abs(lam).max(), 1e-30)
    lam_nz = lam[np.abs(lam) > tol]
    omega_nz = omega[omega > 1e-6 * max(omega.max(), 1e-30)]
    st.caption(f"{sel}: rank(S) ≈ {len(lam_nz)}, active current loops ≈ "
               f"{len(omega_nz) // 2} (of d_model = {len(lam)}). Spectral histograms "
               "below clip the near-zero pile-up bin (its count is annotated) so the "
               "rest of the spectrum is visible.")

    c = st.columns(3)
    c[0].plotly_chart(_spectral_hist(lam_nz, "signed eig(S) — energy DOS", "λ", RED),
                      use_container_width=True)
    c[1].plotly_chart(_spectral_hist(omega_nz, "ω_k(A) — rotation rates", "ω", PURPLE),
                      use_container_width=True)

    bins = arrays["w_bins"]
    centers = 0.5 * (bins[:-1] + bins[1:])
    figE = go.Figure()
    figE.add_scatter(x=centers, y=arrays["P_w_S"][i], name="S", line_shape="hv",
                     fill="tozeroy", line=dict(color=RED),
                     fillcolor="rgba(239,85,59,0.30)")
    figE.add_scatter(x=centers, y=arrays["P_w_A"][i], name="A", line_shape="hv",
                     fill="tozeroy", line=dict(color=BLUE),
                     fillcolor="rgba(99,110,250,0.30)")
    figE.update_layout(title="element density P(w)", xaxis_title="matrix element",
                       yaxis_title="density", height=320, yaxis_type="log",
                       margin=dict(l=10, r=10, t=40, b=10),
                       legend=dict(x=0.01, y=0.99))
    c[2].plotly_chart(figE, use_container_width=True)

    # ── Per-head statistic map (2D) ──
    st.subheader("Per-head statistic map")
    stat_cols = [c for c in df.columns if c.startswith(("S_", "A_"))]
    default = stat_cols.index("S_spectral_entropy") if "S_spectral_entropy" in stat_cols else 0
    metric = st.selectbox("Statistic", stat_cols, index=default, key="f3_stat",
                          format_func=_stat_label)
    st.plotly_chart(_heatmap(_grid(df, metric, n_layers, n_heads), _stat_label(metric),
                             colorscale="Viridis", cbar=metric),
                    use_container_width=True)

    # ── Unrolled head×layer views ──
    st.subheader("Unrolled (head × layer)")
    st.caption("Heads in a single sequence ordered by (layer, head); dashed lines mark "
               "layer boundaries. Lets you compare element distributions across every "
               "head at once, and overlay an S vs A statistic at fixed head.")
    layer_starts = list(range(0, len(df), n_heads))
    layer_ticks = [f"L{k // n_heads}" for k in layer_starts]

    part = st.radio("P(w) matrix", ["S", "A"], horizontal=True, key="f3_pw_part")
    Pw = arrays[f"P_w_{part}"]
    figpw = go.Figure(go.Heatmap(
        x=centers, y=list(range(len(Pw))), z=np.log10(Pw + 1e-6),
        colorscale="Inferno", colorbar_title="log₁₀ P(w)",
        hovertemplate="w=%{x:.3f}<br>head×layer %{y}<br>log₁₀P=%{z:.2f}<extra></extra>"))
    for k in layer_starts[1:]:
        figpw.add_hline(y=k - 0.5, line_color="rgba(255,255,255,0.18)", line_width=1)
    figpw.update_yaxes(tickvals=layer_starts, ticktext=layer_ticks,
                       autorange="reversed")
    figpw.update_layout(title=f"P(w) for {part} across heads", height=460,
                        xaxis_title="matrix element w", yaxis_title="head × layer",
                        margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(figpw, use_container_width=True)

    suffixes = sorted({c[2:] for c in df.columns if c.startswith("S_")}
                      & {c[2:] for c in df.columns if c.startswith("A_")})
    sd = suffixes.index("stable_rank") if "stable_rank" in suffixes else 0
    suf = st.selectbox("Statistic (S vs A overlay)", suffixes, index=sd,
                       key="f3_overlay", format_func=_overlay_label)
    figov = go.Figure()
    figov.add_scatter(x=list(range(len(df))), y=df[f"S_{suf}"], mode="lines",
                      name=_stat_label(f"S_{suf}"), line=dict(color=RED))
    figov.add_scatter(x=list(range(len(df))), y=df[f"A_{suf}"], mode="lines",
                      name=_stat_label(f"A_{suf}"), line=dict(color=BLUE))
    for k in layer_starts[1:]:
        figov.add_vline(x=k - 0.5, line_color="rgba(128,128,128,0.30)", line_width=1)
    figov.update_xaxes(tickvals=layer_starts, ticktext=layer_ticks)
    figov.update_layout(title=f"S vs A · {_overlay_label(suf)} (unrolled)",
                        xaxis_title="head × layer", yaxis_title=_overlay_label(suf),
                        height=380, margin=dict(l=10, r=10, t=40, b=10),
                        legend=dict(x=0.01, y=0.99))
    st.plotly_chart(figov, use_container_width=True)


def section_corner(df):
    st.header("4 · Corner plot")
    st.caption(
        "Pairwise relationships across heads for any four observables: lower "
        "triangle is the per-head scatter (colored by layer), the diagonal is the "
        "marginal distribution. Use it to read off correlations and outliers — e.g. "
        "Θ vs f_A vs ν_align, or a spectral statistic against the commutator.")

    num_cols = [c for c in df.columns if c not in ("layer", "head")
                and np.issubdtype(df[c].dtype, np.number)]
    defaults = [c for c in ("Theta", "f_A", "nu_align", "Theta_full") if c in num_cols]
    while len(defaults) < 4:
        defaults.append(num_cols[len(defaults) % len(num_cols)])

    N = 4
    sel_cols = st.columns(N)
    vars_ = [sel_cols[k].selectbox(f"Variable {k + 1}", num_cols,
                                   index=num_cols.index(defaults[k]),
                                   format_func=_var_label, key=f"corner_var_{k}")
             for k in range(N)]
    diag_mode = st.radio("Diagonal", ["KDE", "Histogram"], horizontal=True,
                         key="corner_diag")

    labels = [f"L{int(l)}H{int(h)}" for l, h in zip(df.layer, df["head"])]
    layer = df["layer"].values

    fig = make_subplots(rows=N, cols=N, horizontal_spacing=0.04, vertical_spacing=0.04)
    for i in range(N):
        for j in range(N):
            if j > i:
                fig.update_xaxes(visible=False, row=i + 1, col=j + 1)
                fig.update_yaxes(visible=False, row=i + 1, col=j + 1)
                continue
            if i == j:
                x = df[vars_[i]].values.astype(float)
                x = x[np.isfinite(x)]
                if len(x) == 0:
                    continue
                if diag_mode == "KDE" and np.ptp(x) > 0:
                    xr = np.linspace(x.min(), x.max(), 200)
                    fig.add_trace(go.Scatter(
                        x=xr, y=gaussian_kde(x)(xr), mode="lines", showlegend=False,
                        line=dict(color=BLUE, width=1.5), fill="tozeroy",
                        fillcolor="rgba(99,110,250,0.15)"), row=i + 1, col=j + 1)
                else:
                    fig.add_trace(go.Histogram(
                        x=x, marker_color=BLUE, opacity=0.7, nbinsx=30,
                        histnorm="probability density", showlegend=False),
                        row=i + 1, col=j + 1)
            else:
                fig.add_trace(go.Scatter(
                    x=df[vars_[j]].values, y=df[vars_[i]].values, mode="markers",
                    marker=dict(color=layer, coloraxis="coloraxis", size=5, opacity=0.6),
                    text=labels, showlegend=False,
                    hovertemplate=f"%{{text}}<br>{_var_label(vars_[j])}=%{{x:.3g}}<br>"
                                  f"{_var_label(vars_[i])}=%{{y:.3g}}<extra></extra>"),
                    row=i + 1, col=j + 1)

    for k in range(N):
        fig.update_xaxes(title_text=_var_label(vars_[k]), row=N, col=k + 1)
        if k > 0:
            fig.update_yaxes(title_text=_var_label(vars_[k]), row=k + 1, col=1)
    if diag_mode == "Histogram":
        fig.update_layout(barmode="overlay")
    fig.update_layout(
        height=760, hovermode="closest",
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis=dict(colorscale="Turbo",
                       colorbar=dict(title="layer", x=0.82, y=0.80, len=0.45,
                                     thickness=12)))
    st.plotly_chart(fig, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    runs = discover_runs(DEFAULT_OUTPUT_DIR)
    run_map = {p.name[: -len(SUFFIX)]: p for p in runs}
    with st.sidebar:
        st.title("QK/OV Equilibrium")
        if not run_map:
            st.warning(f"No `*{SUFFIX}` under `{DEFAULT_OUTPUT_DIR}`. Run "
                       "`run_correlations.py --equilibrium`, or set `DATA_PATH`.")
            return
        sel = st.selectbox("Run", list(run_map))
        npz = run_map[sel]
        st.caption(f"`{npz.parent}`")

    run_key, df, arrays = load_run(str(npz))
    n_layers, n_heads, _ = section_overview(run_key, df, arrays)
    st.divider()
    section_commutator(df, n_layers, n_heads)
    st.divider()
    section_irreversibility(df, n_layers, n_heads)
    st.divider()
    section_spectra(df, arrays, n_layers, n_heads)
    st.divider()
    section_corner(df)


if __name__ == "__main__":
    main()
