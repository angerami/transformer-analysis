#!/usr/bin/env python3
"""
Generate publication-quality figures for blog post 2:
"Singular Value Structure of Transformer Attention Heads"

Usage:
    # Generate all figures to default output directory
    python scripts/generate_blog_figures_post2.py

    # Export directly to website repo
    python scripts/generate_blog_figures_post2.py --website ../angerami.github.io

    # Generate only specific figures
    python scripts/generate_blog_figures_post2.py --figures 1 2 3

    # Use HuggingFace datasets instead of local data
    python scripts/generate_blog_figures_post2.py --source hf

    # Include MC baseline overlays
    python scripts/generate_blog_figures_post2.py --mc-baseline mc_baseline.json

Data sources:
    Local:  $DATA_PATH/ana-004/weight_study  (or ana-003)
    HF:     angerami/weight_study_ana-003

Requires: matplotlib, datasets, numpy, scipy
    pip install matplotlib datasets numpy scipy
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
from scipy import stats as scipy_stats

# ── Configuration ───────────────────────────────────────────────────────

FONT_SIZE = 11
TITLE_FONT_SIZE = 13
MPL_DPI = 200

MODEL_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

HEAD_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#1F77B4", "#FF7F0E",
]

MC_COLOR = "#EF553B"
MC_STYLE = dict(color=MC_COLOR, linestyle="--", linewidth=1.5, alpha=0.8)
MC_HLINE_STYLE = dict(color=MC_COLOR, linestyle="--", linewidth=1.5, alpha=0.6)

FIGURE_MAP = {
    # MC baseline illustration (Figure 1)
    1: "sv_mc_baseline.png",
    # Combined single-head SV spectra (Figure 2)
    2: "sv_single_head_comparison.png",
    # Full-model heatmaps (Figures 3-4)
    3: "sv_heatmap_svd.png",
    4: "sv_heatmap_plambda.png",
    # Per-head overlays at fixed layer (Figures 5-6)
    5: "sv_layer_grid_svd.png",
    6: "sv_layer_grid_plambda.png",
    # SV statistics vs layer (Figures 7-10)
    7: "sv_leading_lambda_vs_layer.png",
    8: "sv_npr_vs_layer.png",
    9: "sv_spectral_entropy_vs_layer.png",
    10: "sv_condition_number_vs_layer.png",
    # Scatter: spectral entropy vs NPR (Figure 11)
    11: "sv_entropy_vs_npr.png",
    # Correlations with element-wise statistics (Figure 12)
    12: "sv_stats_vs_sigma.png",
    # Cross-model comparisons (Figures 13-16)
    13: "sv_cross_model_leading_lambda.png",
    14: "sv_cross_model_npr.png",
    15: "sv_cross_model_spectral_entropy.png",
    16: "sv_cross_model_condition_number.png",
}


# ── Data Loading (shared with post 1) ──────────────────────────────────

def load_data_local(data_path, campaign="ana-004"):
    from datasets import load_from_disk

    dataset_path = Path(data_path) / campaign / "weight_study"
    if not dataset_path.exists():
        for alt in ["ana-003", "ana-002"]:
            alt_path = Path(data_path) / alt / "weight_study"
            if alt_path.exists():
                dataset_path = alt_path
                campaign = alt
                break
        else:
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                f"Set DATA_PATH env var or use --source hf"
            )

    print(f"Loading from local: {dataset_path}")
    ds = load_from_disk(str(dataset_path))
    df = ds.to_pandas()

    metadata_path = dataset_path / ds.info.description
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return df, metadata


def load_data_hf(hf_version="ana-003"):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    repo_id = f"angerami/weight_study_{hf_version}"
    print(f"Loading from HuggingFace: {repo_id}")

    ds = load_dataset(repo_id, split="train")
    df = ds.to_pandas()

    metadata_path = hf_hub_download(
        repo_id=repo_id, filename="metadata.json", repo_type="dataset"
    )
    with open(metadata_path) as f:
        metadata = json.load(f)

    return df, metadata


def load_data(source="local", data_path=None, campaign="ana-004", hf_version="ana-003"):
    if source == "hf":
        return load_data_hf(hf_version)
    else:
        if data_path is None:
            data_path = os.environ.get("DATA_PATH", "Drive")
        return load_data_local(data_path, campaign)


def load_mc_baseline(path):
    """Load MC baseline from an NPZ file (mp_statistics.npz format).

    The file contains one entry per model (e.g. 'gpt2', 'gpt2-medium'),
    each a dict with:
        d_model, d_head, n_heads, n_layers,
        singular_values  — flat array, reshape to (n_draws, d_head)
        stats            — dict of arrays (100 MC draws each):
                           max, participation_ratio, normalized_participation_ratio,
                           spectral_entropy, condition_number, stable_rank
        mp_predictions   — dict of MP analytical scalars
        exact_predictions — dict of exact moment-method scalars
        mp_density_lambda, mp_density_rho — MP density curve

    Returns: dict keyed by model name, each value a normalized dict with:
        svd_mean, svd_std           — mean/std SV curve (d_head,)
        mp_density_lambda, mp_density_rho — MP density curve
        leading_sv, npr, spectral_entropy, condition_number — MC mean scalars
        exact_*                     — exact moment-method predictions
        mp_*                        — MP analytical predictions
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: MC baseline file not found: {p}")
        return None

    raw = np.load(p, allow_pickle=True)
    mc_all = {}
    for model_key in raw.keys():
        obj = raw[model_key].item()
        d_head = obj["d_head"]

        # Reshape flat singular_values → (n_draws, d_model), keep first d_head (nonzero)
        d_model = obj["d_model"]
        sv_flat = np.asarray(obj["singular_values"])
        n_draws = len(sv_flat) // d_model
        sv_full = sv_flat.reshape(n_draws, d_model)
        sv_2d = sv_full[:, :d_head]  # only the d_head nonzero SVs

        stats = obj["stats"]
        mc_entry = {
            "d_model": obj["d_model"],
            "d_head": d_head,
            "n_heads": obj["n_heads"],
            # Full MC draws for histograms (n_draws, d_head)
            "sv_2d": sv_2d,
            # Mean SV curve for overlay on per-head plots
            "svd_mean": np.mean(sv_2d, axis=0),
            "svd_std": np.std(sv_2d, axis=0),
            # MP density curve for P(lambda) overlay
            "mp_density_lambda": np.asarray(obj["mp_density_lambda"]),
            "mp_density_rho": np.asarray(obj["mp_density_rho"]),
            # MC mean scalar statistics (for hlines)
            "leading_sv": float(np.mean(stats["max"])),
            "npr": float(np.mean(stats["normalized_participation_ratio"])),
            "spectral_entropy": float(np.mean(stats["spectral_entropy"])),
            "condition_number": float(np.mean(stats["condition_number"])),
        }

        # Store analytical predictions (handle both key names)
        pred_map = [
            ("pme", "pme_predictions"),
            ("pme", "exact_predictions"),   # older NPZ format
            ("mp", "mp_predictions"),
        ]
        for prefix, src_key in pred_map:
            if src_key in obj and isinstance(obj[src_key], dict):
                for k, v in obj[src_key].items():
                    key = f"{prefix}_{k}"
                    if key not in mc_entry:
                        mc_entry[key] = float(np.asarray(v))

        # Store raw stats arrays for histogramming
        mc_entry["stats_raw"] = {k: np.asarray(v) for k, v in stats.items()}

        mc_all[model_key] = mc_entry

    print(f"  Loaded MC baselines for: {list(mc_all.keys())}")
    return mc_all


def get_mc_for_model(mc_all, model):
    """Look up the MC baseline entry for a given model name."""
    if mc_all is None:
        return None
    if model in mc_all:
        return mc_all[model]
    # Try without suffixes
    for k in mc_all:
        if model.startswith(k):
            return mc_all[k]
    return None


# ── Plot Utilities ──────────────────────────────────────────────────────

def apply_blog_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.color": "#eeeeee",
        "grid.linewidth": 0.5,
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": MPL_DPI,
        "savefig.dpi": MPL_DPI,
        "savefig.bbox": "tight",
    })

apply_blog_style()


def save_mpl(fig, output_dir, filename, dpi=None):
    filepath = Path(output_dir) / filename
    fig.savefig(filepath, dpi=dpi or MPL_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def get_model_metadata(metadata, model_name):
    if "merged" in metadata and model_name in metadata["merged"]:
        return metadata["merged"][model_name]
    return metadata


def add_layer_separators(ax, n_layers, n_heads):
    for xpos in range(n_heads, n_layers * n_heads, n_heads):
        ax.axvline(xpos, color="#dddddd", linewidth=0.8, linestyle=":")


def display_name(model_name):
    """Clean up model name for display: strip '-deduped' from Pythia models."""
    return model_name.replace("-deduped", "")


def get_ordered_models(df_wt):
    models = sorted(df_wt["model"].unique())
    pythia = [m for m in models if "pythia" in m.lower()]
    non_pythia = [m for m in models if "pythia" not in m.lower()]
    return non_pythia + pythia


def get_wqk(df, model):
    return df.query(f"model == '{model}' and weight_type == 'W_QK'").sort_values(["layer", "head"])


def add_layer_averages(ax, values, n_layers, n_heads, color="red"):
    """Overlay layer-averaged values as a dashed step line."""
    avg_x, avg_y = [], []
    for layer_idx in range(n_layers):
        start = layer_idx * n_heads
        end = start + n_heads
        layer_vals = values[start:end]
        layer_mean = np.mean(layer_vals)
        avg_x.extend([start, end - 1])
        avg_y.extend([layer_mean, layer_mean])
    ax.plot(avg_x, avg_y, color=color, linewidth=2.5, linestyle="--",
            alpha=0.7, label="Layer avg", zorder=5)


def set_plambda_log(ax):
    """Switch P(lambda) axis to log scale to tame the zero-bin peak."""
    ax.set_yscale("log")


# ── SV Statistics Helpers ───────────────────────────────────────────────

def compute_sv_stats(sv):
    sv = np.asarray(sv, dtype=float)
    sv_sum = np.sum(sv)
    sv_sum2 = np.sum(sv**2)

    pr = sv_sum**2 / sv_sum2 if sv_sum2 > 0 else 0.0

    sv2 = sv**2
    if sv_sum2 > 0:
        p = sv2 / sv_sum2
        p = p[p > 0]
        spectral_entropy = -np.sum(p * np.log(p))
    else:
        spectral_entropy = 0.0

    # Condition number: lambda_max / lambda_min (nonzero SVs only)
    sv_nonzero = sv[sv > 0]
    if len(sv_nonzero) > 1:
        condition_number = sv_nonzero[0] / sv_nonzero[-1]
    else:
        condition_number = 1.0

    return {
        "leading_sv": sv[0] if len(sv) > 0 else 0.0,
        "sv_sum": sv_sum,
        "sv_sum2": sv_sum2,
        "participation_ratio": pr,
        "spectral_entropy": spectral_entropy,
        "condition_number": condition_number,
    }


def extract_sv_stat(df_sorted, stat_name, d_head=None):
    """Extract a precomputed SV stat column, or compute from SVD arrays.

    If the stored column is missing or fully null, falls back to recomputing
    from SVD so a broken merge doesn't silently produce empty plots.
    """
    if stat_name in df_sorted.columns and df_sorted[stat_name].notna().any():
        return df_sorted[stat_name].values
    values = []
    for _, row in df_sorted.iterrows():
        stats = compute_sv_stats(row["SVD"])
        if stat_name == "normalized_participation_ratio":
            denom = d_head or len(row["SVD"])
            values.append(stats["participation_ratio"] / denom)
        elif stat_name in stats:
            values.append(stats[stat_name])
        else:
            values.append(0.0)
    return np.array(values)


# ── Figure 1: MC Baseline Illustration ────────────────────────────────

def figure_1_mc_baseline(mc, output_dir):
    """MC baseline: eigenvalues vs index (left), eigenvalue density (right).

    Plots λ² (eigenvalues) throughout. MP density overlaid on both panels.
    PME bounds shown as vertical lines on the density panel.
    """
    if mc is None:
        print("Figure 1: skipped (no MC data)")
        return
    print("Figure 1: MC baseline illustration...")

    sv_2d = mc["sv_2d"]       # (n_draws, d_head) singular values
    d_head = mc["d_head"]

    # Eigenvalue spectrum: λ² sorted descending per draw
    eig_2d = sv_2d ** 2
    eig_mean = eig_2d.mean(axis=0)   # already sorted descending (from SVD)
    eigenvalues = eig_2d.ravel()

    # MP ordered spectrum from quantiles of the MP density
    mp_lam = np.array(mc["mp_density_lambda"])
    mp_rho = np.array(mc["mp_density_rho"])
    dlam = np.diff(mp_lam)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (mp_rho[:-1] + mp_rho[1:]) * dlam)])
    quantiles = np.linspace(0.5 / d_head, 1 - 0.5 / d_head, d_head)
    mp_ordered = np.interp(quantiles, cdf, mp_lam)[::-1]

    pme_max = mc.get("pme_max", None)
    pme_min = mc.get("pme_lam_min", None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: eigenvalues vs index
    idx = np.arange(d_head)
    ax1.plot(idx, eig_mean, "o-", color="#636EFA", markersize=3,
             linewidth=1.5, label="MC")
    ax1.plot(idx, mp_ordered, "-", color="#EF553B", linewidth=1.8,
             label="Marchenko–Pastur")
    ax1.set_xlabel("Index")
    ax1.set_ylabel(r"$\lambda^2_k$")
    ax1.set_xlim(-0.5, d_head - 0.5)
    ax1.legend(framealpha=0.9, edgecolor="#cccccc")

    # Right: eigenvalue density
    ax2.hist(eigenvalues, bins=120, density=True, color="#636EFA",
             edgecolor="none", alpha=0.7, label="MC")
    ax2.plot(mp_lam, mp_rho, color="#EF553B", linewidth=1.8,
             label="Marchenko–Pastur")
    if pme_max is not None:
        ax2.axvline(pme_max, color="#00CC96", linestyle="--", linewidth=1.5,
                     label=r"PME $\lambda^2_{\max}$")
    if pme_min is not None:
        ax2.axvline(pme_min, color="#00CC96", linestyle="--", linewidth=1.5,
                     label=r"PME $\lambda^2_{\min}$")
    ax2.set_xlabel(r"$\lambda^2$")
    ax2.set_ylabel("Density")
    set_plambda_log(ax2)
    ax2.legend(framealpha=0.9, edgecolor="#cccccc", fontsize=FONT_SIZE - 2)

    fig.tight_layout()
    save_mpl(fig, output_dir, FIGURE_MAP[1])


# ── Figure 2: Combined Single-Head Eigenvalue Spectrum ───────────────────

def figure_2_combined_single_head(df, metadata, model, selections, output_dir):
    """Two-panel figure comparing head selections on the same axes.

    selections: list of (layer, head, label, color) tuples.
    Left panel: eigenvalues (λ²) vs index, right panel: P(λ²) histogram (log scale).
    Legend on left panel only.
    """
    print(f"Figure 2: {model} combined single-head comparison...")

    model_meta = get_model_metadata(metadata, model)
    d_model = model_meta["d_model"]
    n_heads = df.query(f"model == '{model}' and weight_type == 'W_QK'")["head"].max() + 1
    d_head = d_model // n_heads

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for layer, head, label, color in selections:
        row = df.query(
            f"model == '{model}' and weight_type == 'W_QK' "
            f"and layer == {layer} and head == {head}"
        )
        if row.empty:
            print(f"  WARNING: no data for {model} L{layer} H{head}")
            continue
        entry = row.iloc[0]
        svd = np.array(entry["SVD"])[:d_head]
        eig = svd ** 2

        ax1.plot(np.arange(d_head), eig, "o-", color=color, markersize=3,
                 linewidth=1.5, label=label)
        ax2.hist(eig, bins=30, density=True, color=color,
                 edgecolor="none", alpha=0.5, label=label)

    ax1.set_xlabel("Index")
    ax1.set_ylabel(r"$\lambda^2_k$")
    ax1.set_xlim(-0.5, d_head - 0.5)
    ax1.legend(framealpha=0.9, edgecolor="#cccccc")

    ax2.set_xlabel(r"$\lambda^2$")
    ax2.set_ylabel(r"$P(\lambda^2)$")
    set_plambda_log(ax2)

    fig.tight_layout()
    save_mpl(fig, output_dir, FIGURE_MAP[2])


# ── Figure 3-4: SV Heatmaps ────────────────────────────────────────────

def figure_3_sv_heatmap(df, metadata, model, output_dir):
    """Heatmap: eigenvalues (λ²) by index, log color scale."""
    print(f"Figure 3: {model} eigenvalue index heatmap...")

    model_meta = get_model_metadata(metadata, model)
    d_model = model_meta["d_model"]
    df_wqk = get_wqk(df, model)

    n_layers = df_wqk["layer"].max() + 1
    n_heads = df_wqk["head"].max() + 1
    d_head = d_model // n_heads

    svd_stack = np.array([np.array(row["SVD"]) for _, row in df_wqk.iterrows()])
    eig_stack = svd_stack[:, :d_head] ** 2

    eig_floor = eig_stack[eig_stack > 0].min() * 0.5 if np.any(eig_stack > 0) else 1e-10
    eig_plot = np.maximum(eig_stack, eig_floor)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        np.log10(eig_plot), aspect="auto", origin="lower",
        extent=[0, d_head, 0, n_layers * n_heads],
        cmap="viridis", interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(\lambda^2_k)$")
    ax.set_xlabel(r"Index $k$")
    ax.set_ylabel("Layer")
    ax.set_yticks([i * n_heads + n_heads / 2 for i in range(n_layers)])
    ax.set_yticklabels([str(i) for i in range(n_layers)])
    ax.grid(False)
    fig.tight_layout()
    save_mpl(fig, output_dir, FIGURE_MAP[3], dpi=300)


def figure_4_plambda_heatmap(df, metadata, model, output_dir):
    """Heatmap: P(λ²) across all heads, log color scale. Re-histogrammed from SVD arrays."""
    print(f"Figure 4: {model} P(λ²) heatmap...")

    model_meta = get_model_metadata(metadata, model)
    d_model = model_meta["d_model"]
    df_wqk = get_wqk(df, model)

    n_layers = df_wqk["layer"].max() + 1
    n_heads = df_wqk["head"].max() + 1
    d_head = d_model // n_heads

    # Collect all eigenvalues to determine global bin edges
    all_eig = []
    for _, row in df_wqk.iterrows():
        eig = np.array(row["SVD"])[:d_head] ** 2
        all_eig.append(eig)
    all_eig_flat = np.concatenate(all_eig)
    n_bins = 80
    eig_max = np.percentile(all_eig_flat[all_eig_flat > 0], 99.5)
    bin_edges = np.linspace(0, eig_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Build histogram stack
    psv_stack = np.zeros((len(df_wqk), n_bins))
    for i, (_, row) in enumerate(df_wqk.iterrows()):
        eig = np.array(row["SVD"])[:d_head] ** 2
        h, _ = np.histogram(eig, bins=bin_edges, density=True)
        psv_stack[i] = h

    psv_floor = psv_stack[psv_stack > 0].min() * 0.5 if np.any(psv_stack > 0) else 1e-10
    psv_plot = np.maximum(psv_stack, psv_floor)
    z = np.log10(psv_plot)

    # Color limits: exclude first bin (near zero)
    bin_width = bin_centers[1] - bin_centers[0]
    zero_mask = bin_centers > bin_width * 2
    psv_for_clim = psv_stack[:, zero_mask]
    if np.any(psv_for_clim > 0):
        vmax = np.log10(psv_for_clim.max())
        vmin = np.log10(psv_for_clim[psv_for_clim > 0].min() * 0.5)
    else:
        vmin, vmax = z.min(), z.max()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        z, aspect="auto", origin="lower",
        extent=[bin_centers[0], bin_centers[-1], 0, n_layers * n_heads],
        cmap="viridis", interpolation="bilinear",
        vmin=vmin, vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}[P(\lambda^2)]$")
    ax.set_xlabel(r"$\lambda^2$")
    ax.set_ylabel("Layer")
    ax.set_yticks([i * n_heads + n_heads / 2 for i in range(n_layers)])
    ax.set_yticklabels([str(i) for i in range(n_layers)])
    ax.grid(False)
    fig.tight_layout()
    save_mpl(fig, output_dir, FIGURE_MAP[4], dpi=300)


# ── Figures 5-6: Per-Head Grid at Fixed Layer ───────────────────────────

def figure_5_6_layer_grid(df, metadata, model, layer, output_dir,
                          plot_type="SVD", fig_num=5, shared_range=True):
    """Grid of per-head eigenvalue distributions for one layer.

    plot_type: "SVD" → eigenvalues vs index, "P_sv" → P(λ²) histogram.
    shared_range: if True, all subplots share the same y-axis limits.
    """
    label = r"$\lambda^2$" if plot_type == "SVD" else r"$P(\lambda^2)$"
    print(f"Figure {fig_num}: {model} W_QK layer {layer} {label} grid...")

    model_meta = get_model_metadata(metadata, model)
    d_model = model_meta["d_model"]
    df_wqk = df.query(f"model == '{model}' and weight_type == 'W_QK'")
    n_heads = df_wqk["head"].max() + 1
    d_head = d_model // n_heads

    n_cols = 4
    n_rows = int(np.ceil(n_heads / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows), squeeze=False)

    # First pass: collect eigenvalue data and track global range
    head_data = {}
    y_max = 0
    for head in range(n_heads):
        row_data = df_wqk.query(f"layer == {layer} and head == {head}")
        if row_data.empty:
            continue
        eig = np.array(row_data["SVD"].iloc[0])[:d_head] ** 2
        head_data[head] = eig
        if plot_type == "SVD":
            y_max = max(y_max, eig.max())

    for head in range(n_heads):
        r, c = divmod(head, n_cols)
        ax = axes[r][c]

        if head not in head_data:
            ax.set_visible(False)
            continue

        head_color = HEAD_COLORS[head % len(HEAD_COLORS)]
        eig = head_data[head]

        if plot_type == "SVD":
            ax.plot(np.arange(d_head), eig, "o-",
                    color=head_color, markersize=2, linewidth=1.2)
            ax.set_xlim(-0.5, d_head - 0.5)
            if shared_range:
                ax.set_ylim(0, y_max * 1.05)
        else:
            ax.hist(eig, bins=20, density=True, color=head_color,
                    edgecolor="none")
            set_plambda_log(ax)

        ax.grid(False)
        ax.set_title(f"Head {head}", fontsize=FONT_SIZE)

    for idx in range(n_heads, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Layer {layer} — {label}", fontsize=TITLE_FONT_SIZE, y=1.01)
    fig.tight_layout()
    save_mpl(fig, output_dir, FIGURE_MAP[fig_num])


# ── Figures 7-10: SV Statistics vs Layer ────────────────────────────────
# Shared function for the four per-layer stat plots

SV_STAT_CONFIGS = {
    7:  ("leading_sv",                    r"$\lambda_0$",                 r"$\lambda_0$",                        "leading_sv"),
    8:  ("normalized_participation_ratio", "Normalized Participation Ratio", "NPR",                              "npr"),
    9:  ("spectral_entropy",              r"Spectral Entropy $S_\lambda$", r"$S_\lambda$",                      "spectral_entropy"),
    10: ("condition_number",              r"Condition Number $\kappa$",    r"$\kappa$",                          "condition_number"),
}


def figure_sv_stat_vs_layer(df, metadata, model, output_dir, fig_num=7, mc=None):
    """Generic: one SV statistic across all heads, ordered by layer, with layer averages."""
    stat_name, ylabel, legend_label, mc_key = SV_STAT_CONFIGS[fig_num]
    print(f"Figure {fig_num}: {model} {stat_name} vs layer...")

    model_meta = get_model_metadata(metadata, model)
    d_model = model_meta["d_model"]
    df_wqk = get_wqk(df, model)
    n_layers = df_wqk["layer"].max() + 1
    n_heads = df_wqk["head"].max() + 1
    d_head = d_model // n_heads

    values = extract_sv_stat(df_wqk, stat_name, d_head=d_head)
    x = np.arange(len(df_wqk))

    use_log = (fig_num == 10)  # log scale for condition number
    use_log = False
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, values, color="#636EFA", linewidth=1.2, alpha=0.8, label=legend_label)
    add_layer_averages(ax, values, n_layers, n_heads)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    if use_log:
        ax.set_yscale("log")
    elif fig_num == 8:
        ax.set_ylim(0, 1.05)
    ax.set_xticks([i * n_heads for i in range(n_layers)])
    ax.set_xticklabels([str(i) for i in range(n_layers)])
    add_layer_separators(ax, n_layers, n_heads)

    # MC baselines removed: MC stats computed from eigenvalues (λ²),
    # trained model stats from singular values (λ). Deferred to next post.

    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc")
    fig.tight_layout()
    save_mpl(fig, output_dir, FIGURE_MAP[fig_num])


# ── Figure 11: Spectral Entropy vs NPR Scatter ─────────────────────────

def figure_11_entropy_vs_npr(df, metadata, model, output_dir):
    """Scatter of spectral entropy vs NPR, colored by layer."""
    print(f"Figure 11: {model} spectral entropy vs NPR...")

    model_meta = get_model_metadata(metadata, model)
    d_model = model_meta["d_model"]
    df_wqk = get_wqk(df, model)
    n_heads = df_wqk["head"].max() + 1
    d_head = d_model // n_heads

    npr = extract_sv_stat(df_wqk, "normalized_participation_ratio", d_head=d_head)
    s_lambda = extract_sv_stat(df_wqk, "spectral_entropy")
    layers = df_wqk["layer"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(npr, s_lambda, c=layers, cmap="viridis",
                         s=20, alpha=0.7, edgecolors="none")
    cbar = fig.colorbar(scatter, ax=ax, label="Layer")
    ax.set_xlabel("Normalized Participation Ratio")
    ax.set_ylabel(r"Spectral Entropy $S_\lambda$")

    fig.tight_layout()
    save_mpl(fig, output_dir, FIGURE_MAP[11])


# ── Figure 12: SV Statistics vs Element-Wise Sigma ─────────────────────

def figure_12_sv_vs_sigma(df, metadata, output_dir):
    """2x2 scatter: SV statistics vs element-wise sigma, all models."""
    print("Figure 12: SV stats vs sigma...")

    df_wqk = df.query("weight_type == 'W_QK'").sort_values(["model", "layer", "head"])
    models = get_ordered_models(df_wqk)

    stat_configs = [
        ("leading_sv",       r"$\lambda_0$",       r"$\lambda_0$"),
        ("sv_sum",           r"$\Sigma\lambda$",    r"$\Sigma\lambda$"),
        ("sv_sum2",          r"$\Sigma\lambda^2$",  r"$\Sigma\lambda^2$"),
        ("spectral_entropy", r"$S_\lambda$",        "Spectral Entropy"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i_model, model_name in enumerate(models):
        df_m = df_wqk.query(f"model == '{model_name}'").sort_values(["layer", "head"])
        if df_m.empty:
            continue

        model_meta = get_model_metadata(metadata, model_name)
        d_model = model_meta["d_model"]
        n_heads_m = df_m["head"].max() + 1
        d_head = d_model // n_heads_m
        sigma = df_m["std"].values
        color = MODEL_COLORS[i_model % len(MODEL_COLORS)]

        for j, (stat_name, ylabel, title) in enumerate(stat_configs):
            vals = extract_sv_stat(df_m, stat_name, d_head=d_head)
            axes[j].scatter(sigma, vals, s=6, alpha=0.4, color=color,
                            edgecolors="none", label=display_name(model_name))
            axes[j].set_xlabel(r"$\sigma$")
            axes[j].set_ylabel(ylabel)
            axes[j].set_title(title)

    # Legend in row 1, col 0 (axes[2]), upper left, two columns
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="upper left", framealpha=0.9,
                   edgecolor="#cccccc", fontsize=FONT_SIZE - 2,
                   ncol=2, borderaxespad=0.3)

    fig.tight_layout()
    save_mpl(fig, output_dir, FIGURE_MAP[12])


# ── Figures 13-16: Cross-Model Comparisons ──────────────────────────────
# One per SV statistic, matching the four per-layer stats

CROSS_MODEL_CONFIGS = {
    13: ("leading_sv",                    r"$\lambda_0$ (layer average)",  "leading_sv"),
    14: ("normalized_participation_ratio", "NPR (layer average)",          "npr"),
    15: ("spectral_entropy",              r"$S_\lambda$ (layer average)",  "spectral_entropy"),
    16: ("condition_number",              r"$\kappa$ (layer average)",     "condition_number"),
}


def figure_cross_model(df, metadata, output_dir, fig_num=14, mc=None):
    """Cross-model layer-averaged comparison for one SV statistic."""
    stat_name, ylabel, mc_key = CROSS_MODEL_CONFIGS[fig_num]
    print(f"Figure {fig_num}: cross-model {stat_name}...")

    df_wqk = df.query("weight_type == 'W_QK'")
    models = get_ordered_models(df_wqk)

    use_log = (fig_num == 16)  # log scale for condition number

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, model_name in enumerate(models):
        df_m = df_wqk.query(f"model == '{model_name}'").sort_values(["layer", "head"])
        if df_m.empty:
            continue

        model_meta = get_model_metadata(metadata, model_name)
        d_model = model_meta["d_model"]
        n_heads_m = df_m["head"].max() + 1
        d_head = d_model // n_heads_m

        vals = extract_sv_stat(df_m, stat_name, d_head=d_head)
        df_m = df_m.copy()
        df_m["_stat"] = vals
        layer_avg = df_m.groupby("layer")["_stat"].mean()

        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(layer_avg.index, layer_avg.values,
                marker="o", markersize=3, linewidth=1.8, color=color,
                label=display_name(model_name))

    # MC baseline horizontal line — only for scale-invariant stats (figs 15, 16)
    if fig_num in (15, 16):
        mc_entry = get_mc_for_model(mc, "gpt2") if isinstance(mc, dict) else mc
        if mc_entry is not None and mc_key in mc_entry:
            mc_val = mc_entry[mc_key]
            ax.axhline(mc_val, label="MC baseline (GPT-2)", **MC_HLINE_STYLE)

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    if use_log:
        ax.set_yscale("log")
    elif fig_num == 14:
        ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend_loc = "upper right" if fig_num == 13 else "lower right"
    ax.legend(loc=legend_loc, framealpha=0.9, edgecolor="#cccccc",
              fontsize=FONT_SIZE - 2, ncol=2)

    fig.tight_layout()
    save_mpl(fig, output_dir, FIGURE_MAP[fig_num])


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate blog figures for post 2: SV structure"
    )
    parser.add_argument("--website", type=Path, default=None,
                        help="Path to Jekyll repo root (exports to images/transformer-analysis/)")
    parser.add_argument("--output", type=str, default="figures/post2",
                        help="Output directory (default: figures/post2)")
    parser.add_argument("--figures", type=int, nargs="+", default=None,
                        help="Generate only specific figure numbers")
    parser.add_argument("--source", choices=["local", "hf"], default="local",
                        help="Data source (default: local)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Local data path (default: $DATA_PATH)")
    parser.add_argument("--campaign", type=str, default="ana-004",
                        help="Campaign name for local data (default: ana-004)")
    parser.add_argument("--mc-baseline", type=str, default=None,
                        help="Path to MC baseline JSON file")
    args = parser.parse_args()

    if args.website:
        output_dir = Path(args.website) / "images" / "transformer-analysis"
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    mc_all = load_mc_baseline(args.mc_baseline)

    figures_to_generate = set(args.figures) if args.figures else set(FIGURE_MAP.keys())

    # Reference model for single-model figures
    ref_model = "gpt2-large"

    print("\nLoading model data...")
    df, metadata = load_data(
        source=args.source,
        data_path=args.data_path,
        campaign=args.campaign,
    )
    print(f"  Loaded {len(df)} rows, models: {sorted(df['model'].unique())}")

    mc_ref = get_mc_for_model(mc_all, ref_model)

    # Figure 1: MC baseline illustration (uses MC data only)
    if 1 in figures_to_generate:
        mc_fig1 = get_mc_for_model(mc_all, ref_model) or get_mc_for_model(mc_all, "gpt2")
        figure_1_mc_baseline(mc_fig1, output_dir)

    # Figure 2: combined single-head comparison (layer 0, three heads)
    if 2 in figures_to_generate:
        figure_2_combined_single_head(
            df, metadata, ref_model,
            selections=[
                (0, 0,  "L0 H0",  "#636EFA"),
                (0, 1,  "L0 H1",  "#EF553B"),
                (0, 11, "L0 H11", "#00CC96"),
            ],
            output_dir=output_dir,
        )

    # Figures 3-4: heatmaps
    if 3 in figures_to_generate:
        figure_3_sv_heatmap(df, metadata, ref_model, output_dir)
    if 4 in figures_to_generate:
        figure_4_plambda_heatmap(df, metadata, ref_model, output_dir)

    # Figures 5-6: per-head grids at layer 0
    if 5 in figures_to_generate:
        figure_5_6_layer_grid(df, metadata, ref_model, layer=0,
                              output_dir=output_dir, plot_type="SVD", fig_num=5)
    if 6 in figures_to_generate:
        figure_5_6_layer_grid(df, metadata, ref_model, layer=0,
                              output_dir=output_dir, plot_type="P_sv", fig_num=6)

    # Figures 7-10: SV statistics vs layer
    for fig_n in [7, 8, 9, 10]:
        if fig_n in figures_to_generate:
            figure_sv_stat_vs_layer(df, metadata, ref_model, output_dir,
                                    fig_num=fig_n, mc=mc_ref)

    # Figure 11: entropy vs NPR scatter
    if 11 in figures_to_generate:
        figure_11_entropy_vs_npr(df, metadata, ref_model, output_dir)

    # Figure 12: SV stats vs element-wise sigma (all models)
    if 12 in figures_to_generate:
        figure_12_sv_vs_sigma(df, metadata, output_dir)

    # Figures 13-16: cross-model comparisons
    for fig_n in [13, 14, 15, 16]:
        if fig_n in figures_to_generate:
            figure_cross_model(df, metadata, output_dir, fig_num=fig_n, mc=mc_all)

    print(f"\nDone! Generated {len(figures_to_generate)} figures in {output_dir}")


if __name__ == "__main__":
    main()
