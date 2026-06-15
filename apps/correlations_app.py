"""Streamlit viewer for transformer weight correlation figures.

Discovers experiments automatically from the output directory:
  - Output root: $DATA_PATH if set, otherwise ./outputs
  - Experiments: any subdirectory of the root that contains figures/pairs/
  - Figures: {root}/{experiment}/figures/pairs/{run_key}/*.png

Usage:
    streamlit run apps/correlations_app.py
    DATA_PATH=/mnt/data/outputs streamlit run apps/correlations_app.py
"""

import os
import re
from pathlib import Path

import streamlit as st

import ultrametricity_view

# ── Page config (must be first st call) ───────────────────────────────

st.set_page_config(
    page_title="Transformer Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path(os.environ.get("DATA_PATH", "outputs"))

COMPONENTS = ["W_QK", "W_OV", "b_Q", "b_K", "b_V", "b_O"]
_COMP_RE = "|".join(re.escape(c) for c in COMPONENTS)

PLOT_TYPES = [
    "corr_vs_layer_distance", "dominant_eigenvectors", "MP_overlay",
    "Q_heatmap", "Q_eigenvalues", "block_means", "P_Q",
    "scalars_vs_layer", "scalars_heatmap", "scalar_correlations",
]

PLOT_TYPE_DISPLAY = {
    "Q_heatmap":              "Correlation heatmap",
    "block_means":            "Block means",
    "P_Q":                    "Overlap distribution P(Q)",
    "Q_eigenvalues":          "Eigenvalue spectrum",
    "corr_vs_layer_distance": "Correlation vs distance",
    "MP_overlay":             "Marchenko-Pastur",
    "dominant_eigenvectors":  "Dominant eigenvectors",
    "cross_heatmap":          "Cross-correlation heatmap",
    "cross_diagonal":         "Cross-correlation diagonal",
    "scalars_vs_layer":       "Scalars vs layer",
    "scalars_heatmap":        "Scalar heatmaps",
    "scalar_correlations":    "Scalar correlations",
}

METRIC_DISPLAY = {
    "frob_cosine":         "Frobenius cosine",
    "pearson_corr":        "Pearson correlation",
    "two_point":           "Two-point function",
    "connected_corr":      "Connected correlation",
    "hist_symmetric_kl":   "Symmetric KL (hist)",
    "hist_jensen_shannon": "Jensen-Shannon (hist)",
    "symmetric_kl":        "Symmetric KL (KDE)",
    "jensen_shannon":      "Jensen-Shannon (KDE)",
}

COMPONENT_COLORS = {
    "weight":  ("#3a1a5a", "#c06aff"),
    "bias":    ("#5a3a1a", "#ffb06a"),
    "cross":   ("#1a5a5a", "#6affff"),
    "scalar":  ("#3a5a1a", "#b0ff6a"),
}


# ── Filename parser (mirrors generate_viewer_index.parse_filename) ────

_CROSS_DIAG_RE = re.compile(
    rf"^(.+?)_cross_diagonal_({_COMP_RE})_vs_({_COMP_RE})$"
)
_CROSS_RE = re.compile(
    rf"^(.+?)_cross_({_COMP_RE})_vs_({_COMP_RE})_(.+)$"
)
_SELF_RE = re.compile(rf"^(.+?)_({_COMP_RE})_(.+)$")


def parse_filename(filename):
    name = filename.removesuffix(".png")
    result = {
        "model": "", "component": "", "plot_type": "",
        "metric": "", "cross_pair": "",
    }

    m = _CROSS_DIAG_RE.match(name)
    if m:
        result["model"] = m.group(1)
        result["component"] = f"{m.group(2)} vs {m.group(3)}"
        result["cross_pair"] = result["component"]
        result["plot_type"] = "cross_diagonal"
        return result

    m = _CROSS_RE.match(name)
    if m:
        result["model"] = m.group(1)
        result["component"] = f"{m.group(2)} vs {m.group(3)}"
        result["cross_pair"] = result["component"]
        result["plot_type"] = "cross_heatmap"
        result["metric"] = m.group(4)
        return result

    for pt in ("scalars_vs_layer", "scalars_heatmap", "scalar_correlations"):
        tag = f"_{pt}"
        if name.endswith(tag):
            result["model"] = name[:-len(tag)]
            result["component"] = "scalars"
            result["plot_type"] = pt
            return result

    m = _SELF_RE.match(name)
    if m:
        result["model"] = m.group(1)
        result["component"] = m.group(2)
        rest = m.group(3)
        for pt in PLOT_TYPES:
            if rest == pt:
                result["plot_type"] = pt
                return result
            if rest.startswith(pt + "_"):
                result["plot_type"] = pt
                result["metric"] = rest[len(pt) + 1:]
                return result
        result["plot_type"] = rest
        return result

    result["model"] = name
    return result


def component_kind(comp):
    if not comp:
        return "weight"
    if comp == "scalars":
        return "scalar"
    if " vs " in comp:
        return "cross"
    if comp.startswith("b_"):
        return "bias"
    return "weight"


# ── Data loading ──────────────────────────────────────────────────────

def detect_experiments(output_dir: Path) -> list[str]:
    """Return experiment names: subdirs with figures/pairs/ or ultrametricity/."""
    if not output_dir.is_dir():
        return []
    return sorted(
        p.name for p in output_dir.iterdir()
        if p.is_dir() and ((p / "figures" / "pairs").is_dir()
                           or (p / "ultrametricity").is_dir())
    )


@st.cache_data(show_spinner="Loading figures...")
def load_catalog(pairs_dir: str) -> list[dict]:
    """Recursively scan pairs_dir/{run_key}/*.png → catalog."""
    entries = []
    for png in sorted(Path(pairs_dir).rglob("*.png")):
        meta = parse_filename(png.name)
        meta["path"] = str(png)
        meta["filename"] = png.name
        meta["run"] = png.parent.name   # run_key subdirectory
        entries.append(meta)
    return entries


# ── CSS ───────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    .figure-card {
        background: #1a1a2a;
        border: 1px solid #3a3a4a;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .card-title {
        font-size: 0.9rem;
        font-weight: 500;
        color: #e0e0e0;
        margin: 0.4rem 0 0.3rem 0;
    }
    .card-tags { display: flex; flex-wrap: wrap; gap: 4px; }
    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.72rem;
        font-weight: 500;
    }
    .tag-model     { background: #1a3a5a; color: #6ab0ff; }
    .tag-weight    { background: #3a1a5a; color: #c06aff; }
    .tag-bias      { background: #5a3a1a; color: #ffb06a; }
    .tag-cross     { background: #1a5a5a; color: #6affff; }
    .tag-scalar    { background: #3a5a1a; color: #b0ff6a; }
    .tag-metric    { background: #1a5a3a; color: #6affc0; }
    .tag-plot-type { background: #2a2a3a; color: #aab0ff; }

    /* tighten Streamlit column gaps */
    [data-testid="stHorizontalBlock"] { gap: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)


# ── Rendering ─────────────────────────────────────────────────────────

def render_tags_html(entry):
    kind = component_kind(entry["component"])
    parts = [f'<span class="tag tag-model">{entry["run"]}</span>']
    if entry["component"]:
        parts.append(
            f'<span class="tag tag-{kind}">{entry["component"]}</span>'
        )
    if entry["metric"]:
        label = METRIC_DISPLAY.get(entry["metric"], entry["metric"])
        parts.append(f'<span class="tag tag-metric">{label}</span>')
    return "".join(parts)


def render_card(entry):
    title = PLOT_TYPE_DISPLAY.get(entry["plot_type"], entry["plot_type"])
    tags_html = render_tags_html(entry)
    st.markdown('<div class="figure-card">', unsafe_allow_html=True)
    st.image(entry["path"], use_container_width=True)
    st.markdown(
        f'<div class="card-title">{title}</div>'
        f'<div class="card-tags">{tags_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def display_name(key, display_map):
    """Format for multiselect: 'internal_key (Display Name)'."""
    d = display_map.get(key)
    return f"{key}  —  {d}" if d else key


# ── Main ──────────────────────────────────────────────────────────────

def main():
    inject_css()

    experiments = detect_experiments(DEFAULT_OUTPUT_DIR)
    if not experiments:
        st.warning(
            f"No experiments found under `{DEFAULT_OUTPUT_DIR}`. "
            "Run the pipeline first, or set `DATA_PATH` to the correct output directory."
        )
        return

    with st.sidebar:
        st.title("Transformer Analysis")
        view = st.radio("View", ["Correlation figures", "Ultrametricity"])
        experiment = st.selectbox("Experiment", experiments)

    exp_dir = DEFAULT_OUTPUT_DIR / experiment
    if view == "Ultrametricity":
        ultrametricity_view.render(exp_dir)
    else:
        render_gallery(exp_dir)


def render_gallery(exp_dir: Path):
    pairs_dir = exp_dir / "figures" / "pairs"
    catalog = load_catalog(str(pairs_dir))

    if not catalog:
        st.warning(f"No figures found in `{pairs_dir}`.")
        return

    # Unique values for filters
    all_runs       = sorted({e["run"] for e in catalog})
    all_components = sorted({e["component"] for e in catalog if e["component"]})
    all_plot_types = sorted({e["plot_type"] for e in catalog if e["plot_type"]})
    all_metrics    = sorted({e["metric"] for e in catalog if e["metric"]})

    with st.sidebar:
        st.caption(f"`{pairs_dir}` · {len(catalog)} figures")

        sel_runs = st.multiselect(
            "Run", all_runs,
            placeholder="All runs",
        )
        sel_components = st.multiselect(
            "Component", all_components,
            placeholder="All components",
        )
        sel_plot_types = st.multiselect(
            "Plot type", all_plot_types,
            format_func=lambda k: PLOT_TYPE_DISPLAY.get(k, k),
            placeholder="All plot types",
        )
        sel_metrics = st.multiselect(
            "Metric", all_metrics,
            format_func=lambda k: METRIC_DISPLAY.get(k, k),
            placeholder="All metrics",
        )

        st.divider()
        search = st.text_input("Search filenames", "")
        n_cols = st.slider("Columns", 1, 5, 3)
        per_page = st.slider("Per page", 6, 60, 18, step=6)

    # ── Filter ────────────────────────────────────────────────────
    filtered = catalog
    if sel_runs:
        filtered = [e for e in filtered if e["run"] in sel_runs]
    if sel_components:
        filtered = [e for e in filtered if e["component"] in sel_components]
    if sel_plot_types:
        filtered = [e for e in filtered if e["plot_type"] in sel_plot_types]
    if sel_metrics:
        filtered = [e for e in filtered if e["metric"] in sel_metrics]
    if search:
        q = search.lower()
        filtered = [e for e in filtered if q in e["filename"].lower()]

    # ── Stats bar ─────────────────────────────────────────────────
    stat_cols = st.columns(4)
    stat_cols[0].metric("Total", len(catalog))
    stat_cols[1].metric("Filtered", len(filtered))
    stat_cols[2].metric("Runs", len(all_runs))
    stat_cols[3].metric("Components", len(all_components))

    if not filtered:
        st.info("No figures match the current filters.")
        return

    # ── Pagination ────────────────────────────────────────────────
    n_pages = max(1, (len(filtered) + per_page - 1) // per_page)

    if "gallery_page" not in st.session_state:
        st.session_state.gallery_page = 0
    # Reset page when filters change
    filter_key = (
        tuple(sel_runs), tuple(sel_components),
        tuple(sel_plot_types), tuple(sel_metrics), search,
    )
    if st.session_state.get("_filter_key") != filter_key:
        st.session_state.gallery_page = 0
        st.session_state["_filter_key"] = filter_key

    page = st.session_state.gallery_page
    page = max(0, min(page, n_pages - 1))

    nav_cols = st.columns([1, 3, 1])
    with nav_cols[0]:
        if st.button("← Prev", disabled=(page == 0)):
            st.session_state.gallery_page = page - 1
            st.rerun()
    with nav_cols[1]:
        st.markdown(
            f"<div style='text-align:center; color:#888; padding-top:6px'>"
            f"Page {page + 1} / {n_pages}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with nav_cols[2]:
        if st.button("Next →", disabled=(page >= n_pages - 1)):
            st.session_state.gallery_page = page + 1
            st.rerun()

    # ── Gallery grid ──────────────────────────────────────────────
    start = page * per_page
    page_items = filtered[start : start + per_page]

    for row_start in range(0, len(page_items), n_cols):
        cols = st.columns(n_cols)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx >= len(page_items):
                break
            with col:
                with st.container():
                    render_card(page_items[idx])

    # ── Detail view (click filename to expand) ────────────────────
    with st.sidebar:
        st.divider()
        filenames = [e["filename"] for e in filtered]
        sel = st.selectbox(
            "Detail view",
            [""] + filenames,
            format_func=lambda x: x if x else "Select a figure...",
        )
        if sel:
            entry = next(e for e in filtered if e["filename"] == sel)
            st.image(entry["path"], use_container_width=True)
            st.caption(entry["filename"])
            for k in ("run", "component", "plot_type", "metric"):
                if entry.get(k):
                    st.text(f"{k}: {entry[k]}")


if __name__ == "__main__":
    main()
