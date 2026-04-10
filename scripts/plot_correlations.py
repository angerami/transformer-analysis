#!/usr/bin/env python3
"""Plot head-head correlation results from saved .npz / .npy files.

Usage:
    python scripts/plot_correlations.py --data corr_out --model gpt2
    python scripts/plot_correlations.py --data corr_out --model gpt2 --metrics frob_cosine jensen_shannon
    python scripts/plot_correlations.py --data corr_out --model gpt2 --out figures/correlations
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── Style ──────────────────────────────────────────────────────────────

FONT_SIZE = 11
TITLE_SIZE = 13
DPI = 200
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "figure.dpi": DPI,
})


# ── Data loading ───────────────────────────────────────────────────────

def load_results(data_dir, model, revision="main", weight_type="W_QK"):
    """Load all saved correlation data for a model run."""
    prefix = f"{model}_{revision}_{weight_type}"

    with open(os.path.join(data_dir, f"{prefix}_metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(data_dir, f"{prefix}_summary.json")) as f:
        summary = json.load(f)

    Q_data = np.load(os.path.join(data_dir, f"{prefix}_Q.npz"))
    Q = {k.replace("Q_", ""): Q_data[k] for k in Q_data.files}

    eigenvalues = {}
    P_Q = {}
    block_means = {}
    for m in metadata["metrics"]:
        eig_path = os.path.join(data_dir, f"{prefix}_{m}_eigenvalues.npy")
        if os.path.exists(eig_path):
            eigenvalues[m] = np.load(eig_path)
        pq_path = os.path.join(data_dir, f"{prefix}_{m}_P_Q.npy")
        if os.path.exists(pq_path):
            P_Q[m] = np.load(pq_path)
        bm_path = os.path.join(data_dir, f"{prefix}_{m}_block_means.npy")
        if os.path.exists(bm_path):
            block_means[m] = np.load(bm_path)

    keys = [tuple(k) for k in metadata["head_index"]]
    return {
        "Q": Q, "summary": summary, "eigenvalues": eigenvalues,
        "P_Q": P_Q, "block_means": block_means,
        "metadata": metadata, "keys": keys,
    }


# ── Plot functions ─────────────────────────────────────────────────────

def _layer_boundaries(keys):
    layers = [k[0] for k in keys]
    bounds = []
    for i in range(1, len(layers)):
        if layers[i] != layers[i - 1]:
            bounds.append(i)
    return bounds


def _metric_display(name):
    return {
        "frob_cosine": "Frobenius cosine similarity",
        "symmetric_kl": "Symmetric KL divergence (KDE)",
        "jensen_shannon": "Jensen-Shannon divergence (KDE)",
        "hist_symmetric_kl": "Symmetric KL divergence (histogram)",
        "hist_jensen_shannon": "Jensen-Shannon divergence (histogram)",
        "two_point": "Two-point function $\\langle W_1 W_2 \\rangle$",
        "connected_corr": "Connected correlation $\\langle W_1 W_2 \\rangle - \\langle W_1 \\rangle \\langle W_2 \\rangle$",
        "pearson_corr": "Pearson correlation (normalized connected)",
    }.get(name, name)


def _is_divergence(name):
    return name in ("symmetric_kl", "jensen_shannon",
                     "hist_symmetric_kl", "hist_jensen_shannon")


def _is_correlation_metric(name):
    """Metrics where a diverging (RdBu) colormap centered on 0 is appropriate."""
    return name in ("frob_cosine", "connected_corr", "pearson_corr", "two_point")


# Canonical 2×3 metric ordering: cosine + Pearson (similar shape),
# symmetric KL + connected corr, JS + two-point.
_METRIC_ORDER = [
    "frob_cosine", "pearson_corr",
    "symmetric_kl", "connected_corr",
    "jensen_shannon", "two_point",
]
_METRIC_ALT = {
    "symmetric_kl": "hist_symmetric_kl",
    "jensen_shannon": "hist_jensen_shannon",
}


def _order_metrics(available):
    """Return metrics in canonical 2×3 order, falling back to hist variants."""
    ordered = []
    for slot in _METRIC_ORDER:
        if slot in available:
            ordered.append(slot)
        elif slot in _METRIC_ALT and _METRIC_ALT[slot] in available:
            ordered.append(_METRIC_ALT[slot])
    for m in available:
        if m not in ordered:
            ordered.append(m)
    return ordered


def plot_heatmap(Q, keys, metric_name, model_name, out_dir):
    """Single Q_{hh'} heatmap."""
    fig, ax = plt.subplots(figsize=(10, 9))
    bounds = _layer_boundaries(keys)
    n = Q.shape[0]

    if _is_divergence(metric_name):
        cmap = "viridis_r"
        im = ax.imshow(Q, cmap=cmap, aspect="equal")
    else:
        vmax = np.percentile(np.abs(Q), 98)
        cmap = "RdBu_r"
        im = ax.imshow(Q, cmap=cmap, aspect="equal", vmin=-vmax, vmax=vmax)

    for b in bounds:
        ax.axhline(b - 0.5, color="white", linewidth=0.5, alpha=0.8)
        ax.axvline(b - 0.5, color="white", linewidth=0.5, alpha=0.8)

    # layer labels at midpoints
    layers = sorted(set(k[0] for k in keys))
    n_per = len(keys) // len(layers)
    tick_pos = [l * n_per + n_per // 2 for l in range(len(layers))]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(l) for l in layers], fontsize=9)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([str(l) for l in layers], fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")

    ax.set_title(f"{model_name}  —  $Q_{{hh'}}$ ({_metric_display(metric_name)})")
    fig.colorbar(im, ax=ax, shrink=0.8, label=_metric_display(metric_name))
    fig.tight_layout()
    fpath = os.path.join(out_dir, f"{model_name}_Q_heatmap_{metric_name}.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fpath}")
    return fpath


def plot_P_Q(P_Q_dict, summary, model_name, out_dir):
    """P(Q) overlap distributions in canonical 2×3 grid."""
    ordered = _order_metrics(P_Q_dict.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flat

    for idx, m in enumerate(ordered[:6]):
        ax = axes[idx]
        vals = P_Q_dict[m]
        ax.hist(vals, bins=60, density=True, alpha=0.7, color="#636EFA",
                edgecolor="white", linewidth=0.3)
        mu = summary[m]["mean_offdiag"]
        sigma = summary[m].get("std_offdiag", np.std(vals))
        ax.axvline(mu, color="#EF553B", linestyle="--", linewidth=1.2,
                   label=f"$\\mu$ = {mu:.3f},  $\\sigma$ = {sigma:.3f}")
        ax.set_title(_metric_display(m), fontsize=10)
        ax.set_xlabel("$Q$ value")
        ax.set_ylabel("density")
        ax.set_yscale("log")
        ax.legend(fontsize=9)

    for idx in range(len(ordered[:6]), 6):
        axes[idx].set_visible(False)

    fig.suptitle(f"{model_name}  —  Overlap distributions $P(Q)$", fontsize=TITLE_SIZE)
    fig.tight_layout()
    fpath = os.path.join(out_dir, f"{model_name}_P_Q.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fpath}")
    return fpath


def plot_eigenvalues(eig_dict, model_name, out_dir):
    """Eigenvalue spectra of Q in canonical 2×3 grid."""
    ordered = _order_metrics(eig_dict.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flat

    for idx, m in enumerate(ordered[:6]):
        ax = axes[idx]
        eigvals = eig_dict[m]
        abs_eig = np.sort(np.abs(eigvals))[::-1]
        ax.plot(abs_eig, "o-", markersize=3, color="#EF553B")
        ax.set_yscale("log")
        ax.set_ylabel("$|\\lambda|$")
        ax.set_title(_metric_display(m), fontsize=10)
        ax.set_xlabel("Index")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for idx in range(len(ordered[:6]), 6):
        axes[idx].set_visible(False)

    fig.suptitle(f"{model_name}  —  Eigenvalues of $Q_{{hh'}}$", fontsize=TITLE_SIZE)
    fig.tight_layout()
    fpath = os.path.join(out_dir, f"{model_name}_Q_eigenvalues.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fpath}")
    return fpath


def plot_block_means(block_dict, metadata, model_name, out_dir):
    """Layer × layer block-mean heatmaps."""
    n_layers = metadata["n_layers"]
    layers = list(range(n_layers))
    metrics = list(block_dict.keys())

    for m in metrics:
        block = block_dict[m]
        fig, ax = plt.subplots(figsize=(7, 6))

        if _is_divergence(m):
            im = ax.imshow(block, cmap="viridis_r", aspect="equal")
        else:
            vmax = np.max(np.abs(block))
            im = ax.imshow(block, cmap="RdBu_r", aspect="equal",
                           vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(n_layers))
        ax.set_yticks(range(n_layers))
        tick_fs = 7 if n_layers > 20 else 9
        tick_rot = 90 if n_layers > 20 else 0
        ax.set_xticklabels(layers, fontsize=tick_fs, rotation=tick_rot)
        ax.set_yticklabels(layers, fontsize=tick_fs)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        ax.set_title(f"{model_name}  —  Layer-block means\n{_metric_display(m)}")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fpath = os.path.join(out_dir, f"{model_name}_block_means_{m}.png")
        fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fpath}")


def plot_correlation_vs_layer_distance(P_Q_dict, keys, Q_dict, model_name, out_dir):
    """Mean |Q| as a function of layer distance |l - l'|, for each metric.

    Fixed 2×3 grid: top row frob_cosine, KL, JS; bottom row two_point,
    connected_corr, pearson_corr.  Unused panels hidden.
    """
    layers = np.array([k[0] for k in keys])
    n = len(keys)

    ordered = _order_metrics(Q_dict.keys())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flat

    for idx, m in enumerate(ordered[:6]):
        ax = axes[idx]
        Q = Q_dict[m]
        triu_i, triu_j = np.triu_indices(n, k=1)
        dists = np.abs(layers[triu_i] - layers[triu_j])
        vals = Q[triu_i, triu_j]

        unique_d = np.unique(dists)
        means = [np.mean(np.abs(vals[dists == d])) for d in unique_d]
        stds = [np.std(vals[dists == d]) for d in unique_d]

        ax.errorbar(unique_d, means, yerr=stds, fmt="o-", markersize=4,
                    capsize=3, color="#636EFA")
        ax.set_xlabel("Layer distance $|\\ell - \\ell'|$")
        ax.set_ylabel("Mean $|Q|$" if not _is_divergence(m) else "Mean $Q$")
        ax.set_title(_metric_display(m), fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for idx in range(len(ordered[:6]), 6):
        axes[idx].set_visible(False)

    fig.suptitle(f"{model_name}  —  Correlation vs. layer distance", fontsize=TITLE_SIZE)
    fig.tight_layout()
    fpath = os.path.join(out_dir, f"{model_name}_corr_vs_layer_distance.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fpath}")


# ── Marchenko-Pastur overlay ───────────────────────────────────────────

def _mp_density(lam, gamma):
    """Marchenko-Pastur density for aspect ratio gamma = N/p."""
    lam_m = (1 - np.sqrt(gamma)) ** 2
    lam_p = (1 + np.sqrt(gamma)) ** 2
    mask = (lam >= lam_m) & (lam <= lam_p)
    density = np.zeros_like(lam)
    density[mask] = (np.sqrt((lam_p - lam[mask]) * (lam[mask] - lam_m))
                     / (2 * np.pi * gamma * lam[mask]))
    return density


def compute_Q_eigen_stats(eigvals, gamma):
    """Condition number, NPR, and stable rank from Q eigenvalues + MP predictions."""
    eigvals = np.sort(np.real(eigvals))[::-1]
    lam_max = eigvals[0]
    lam_min = eigvals[-1]

    # Measured
    cond = lam_max / max(lam_min, 1e-12)
    npr = (np.sum(eigvals) ** 2) / (len(eigvals) * np.sum(eigvals ** 2))
    srank = np.sum(eigvals ** 2) / max(lam_max ** 2, 1e-12)

    # MP predictions
    sg = np.sqrt(gamma)
    mp_lam_plus = (1 + sg) ** 2
    mp_lam_minus = (1 - sg) ** 2
    mp_cond = mp_lam_plus / max(mp_lam_minus, 1e-12)
    mp_npr = 1 / (1 + gamma)
    N = len(eigvals)
    mp_srank = N * (1 + gamma) / (1 + sg) ** 4  # E[tr(Q²)] / E[λ_max]²

    return {
        "condition_number": cond,
        "npr": npr,
        "stable_rank": srank,
        "mp_condition_number": mp_cond,
        "mp_npr": mp_npr,
        "mp_stable_rank": mp_srank,
        "gamma": gamma,
        "N": N,
        "lam_max": lam_max,
        "lam_min": lam_min,
    }


def plot_mp_overlay(Q_frob, metadata, model_name, out_dir):
    """Eigenvalue spectrum of Q^(Frob) with Marchenko-Pastur prediction.

    Robust version: clamps axis limits, constrains annotations to the
    visible canvas, and caps the figure aspect ratio.
    """
    N = metadata["n_layers"] * metadata["n_heads"]
    d_head = metadata["head_dim"]
    p = d_head ** 2
    gamma = N / p

    lam_minus = (1 - np.sqrt(gamma)) ** 2
    lam_plus = (1 + np.sqrt(gamma)) ** 2

    eigvals = np.linalg.eigvalsh(Q_frob)[::-1]

    n_outliers = int(np.sum(eigvals > lam_plus))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: ordered eigenvalues with MP band ──
    idx = np.arange(len(eigvals))
    ax1.semilogy(idx, np.maximum(eigvals, 1e-8), "o-", markersize=3,
                 color="#EF553B", label=f"{model_name} (trained)", zorder=3)
    ax1.axhspan(lam_minus, lam_plus, alpha=0.15, color="#636EFA",
                label=f"MP bulk [{lam_minus:.2f}, {lam_plus:.2f}]", zorder=1)
    ax1.axhline(lam_plus, color="#636EFA", linestyle="--", linewidth=1, alpha=0.7)
    ax1.axhline(lam_minus, color="#636EFA", linestyle="--", linewidth=1, alpha=0.7)

    if n_outliers > 0:
        # Arrow points at the last (smallest) outlier eigenvalue —
        # index n_outliers-1 in the descending-sorted array.
        last_outlier_idx = n_outliers - 1
        text_x = min(last_outlier_idx + 8, N * 0.4)
        text_y = eigvals[last_outlier_idx] * 1.5
        ax1.annotate(
            f"{n_outliers} outlier{'s' if n_outliers > 1 else ''} "
            f"above MP edge ($\\lambda_{{max}}$={eigvals[0]:.1f})",
            xy=(last_outlier_idx, eigvals[last_outlier_idx]),
            xytext=(text_x, text_y),
            fontsize=9, color="#636EFA",
            arrowprops=dict(arrowstyle="->", color="#636EFA", lw=1),
            annotation_clip=True,
        )

    ax1.set_xlim(-0.5, N + 0.5)
    ax1.set_xlabel("Index")
    ax1.set_ylabel("$\\lambda$")
    ax1.set_title(f"Eigenvalue spectrum of $Q_{{hh'}}^{{\\mathrm{{(Frob)}}}}$\n"
                  f"({model_name}, {N} heads, $\\gamma$ = {gamma:.4f})")
    ax1.legend(fontsize=9, loc="upper right")

    # ── Right: histogram with MP density overlay ──
    # Clamp histogram range: focus on the MP bulk + modest outlier range
    hist_max = min(lam_plus * 4, eigvals.max() * 1.1)
    # Keep at least 90 % of eigenvalues visible
    sorted_eig = np.sort(eigvals)
    p90 = sorted_eig[int(0.9 * len(sorted_eig))] if len(sorted_eig) else 1.0
    hist_max = max(hist_max, p90 * 1.5)

    bins = np.linspace(0, hist_max, 60)
    ax2.hist(eigvals[eigvals <= hist_max * 1.1], bins=bins, density=True,
             alpha=0.6, color="#EF553B", edgecolor="white", linewidth=0.3,
             label=f"{model_name} eigenvalues")

    lam_grid = np.linspace(0.01, lam_plus * 1.5, 500)
    mp_curve = _mp_density(lam_grid, gamma)
    ax2.plot(lam_grid, mp_curve, "-", color="#636EFA", linewidth=2.5,
             label=f"MP density ($\\gamma$ = {gamma:.4f})")

    # Annotate outliers: list all values as text (they often exceed hist range)
    outlier_vals = eigvals[eigvals > lam_plus]
    if len(outlier_vals) > 0:
        # Build a compact label listing the outlier eigenvalues
        if len(outlier_vals) <= 5:
            val_strs = [f"{v:.1f}" for v in outlier_vals]
        else:
            val_strs = [f"{v:.1f}" for v in outlier_vals[:4]] + ["..."]
        ax2.text(
            0.97, 0.95,
            f"{len(outlier_vals)} outlier{'s' if len(outlier_vals) > 1 else ''}"
            f" > $\\lambda_+$\n$\\lambda$ = {', '.join(val_strs)}",
            transform=ax2.transAxes, fontsize=8, color="#636EFA",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2a", ec="#636EFA",
                      alpha=0.8),
        )

    ax2.axvline(lam_plus, color="#636EFA", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Eigenvalue $\\lambda$")
    ax2.set_ylabel("Density")
    ax2.set_title("Eigenvalue distribution vs. MP prediction")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, hist_max)

    # ── Stats inset on left panel ──
    stats = compute_Q_eigen_stats(eigvals, gamma)
    stats_text = (
        f"{'':>12s} {'Meas':>8s} {'MP':>8s}\n"
        f"{'C':>12s} {stats['condition_number']:>8.1f} {stats['mp_condition_number']:>8.2f}\n"
        f"{'NPR':>12s} {stats['npr']:>8.3f} {stats['mp_npr']:>8.3f}\n"
        f"{'stable rank':>12s} {stats['stable_rank']:>8.1f} {stats['mp_stable_rank']:>8.1f}"
    )
    ax1.text(
        0.97, 0.45, stats_text,
        transform=ax1.transAxes, fontsize=7.5, fontfamily="monospace",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2a", ec="#888888",
                  alpha=0.85),
        color="#e0e0e0",
    )

    fig.tight_layout()
    fpath = os.path.join(out_dir, f"{model_name}_MP_overlay.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fpath}")
    return fpath, stats


# ── Dominant eigenvector visualization ─────────────────────────────────

def plot_dominant_eigenvector(Q_frob, metadata, model_name, out_dir, n_modes=3):
    """Dominant eigenvectors of Q^(Frob) as layer×head heatmaps with projections.

    Top row: mode heatmaps (layer × head) + layer loading bar chart.
    Bottom row: head loading bar charts aligned under each mode + blank cells.
    """
    n_layers = metadata["n_layers"]
    n_heads = metadata["n_heads"]
    N = n_layers * n_heads

    eigvals, eigvecs = np.linalg.eigh(Q_frob)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    d_head = metadata["head_dim"]
    gamma = N / (d_head ** 2)
    lam_plus = (1 + np.sqrt(gamma)) ** 2
    n_outliers = int(np.sum(eigvals > lam_plus))
    n_show = max(1, min(n_modes, n_outliers, 3))

    n_cols = n_show + 1
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(5 * n_cols, 9),
        gridspec_kw={
            "width_ratios": [1] * n_show + [0.6],
            "height_ratios": [1.2, 1],
        },
    )
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    mode_colors = ["#636EFA", "#EF553B", "#00CC96"]

    # ── Top row: mode heatmaps + layer loading ──
    for k in range(n_show):
        ax = axes[0, k]
        v = eigvecs[:, k]
        v_grid = v.reshape(n_layers, n_heads)
        vmax = np.max(np.abs(v_grid))
        im = ax.imshow(v_grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"Mode {k + 1}: $\\lambda_{{{k + 1}}}$ = {eigvals[k]:.1f}")
        fig.colorbar(im, ax=ax, shrink=0.7)

    # Top-right: layer loading
    ax = axes[0, -1]
    layers_arr = np.arange(n_layers)
    width = 0.8 / n_show
    for k in range(n_show):
        v = eigvecs[:, k]
        layer_loading = np.array([np.sum(v[l * n_heads:(l + 1) * n_heads] ** 2)
                                  for l in range(n_layers)])
        ax.barh(layers_arr + k * width, layer_loading, height=width,
                color=mode_colors[k % len(mode_colors)], alpha=0.7,
                label=f"Mode {k + 1}")
    ax.set_ylabel("Layer")
    ax.set_xlabel("$\\sum_h v^2_{(\\ell,h)}$")
    ax.set_title("Layer loading")
    ax.invert_yaxis()
    ax.legend(fontsize=8)

    # ── Bottom row: head loading per mode ──
    heads_arr = np.arange(n_heads)
    for k in range(n_show):
        ax = axes[1, k]
        v = eigvecs[:, k]
        head_loading = np.array([np.sum(v[h::n_heads] ** 2)
                                 for h in range(n_heads)])
        ax.bar(heads_arr, head_loading,
               color=mode_colors[k % len(mode_colors)], alpha=0.7)
        ax.set_xlabel("Head")
        ax.set_ylabel("$\\sum_\\ell v^2_{(\\ell,h)}$")
        ax.set_title(f"Head loading — Mode {k + 1}")

    # Bottom-right: blank
    axes[1, -1].axis("off")

    fig.suptitle(f"{model_name}  —  Dominant eigenvectors of "
                 f"$Q_{{hh'}}^{{\\mathrm{{(Frob)}}}}$\n"
                 f"({n_outliers} outlier{'s' if n_outliers != 1 else ''} "
                 f"above MP edge at $\\lambda$ = {lam_plus:.2f})",
                 fontsize=TITLE_SIZE)
    fig.tight_layout()
    fpath = os.path.join(out_dir, f"{model_name}_dominant_eigenvectors.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fpath}")
    return fpath


# ── Cross-correlation heatmap ─────────────────────────────────────────

def plot_cross_heatmap(Q, keys, metric_name, label, model_name, out_dir):
    """Heatmap for a cross-correlation matrix (not necessarily symmetric).

    The diagonal shows intra-head cross-circuit coupling.
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    bounds = _layer_boundaries(keys)
    n = Q.shape[0]

    if _is_divergence(metric_name):
        cmap = "viridis_r"
        im = ax.imshow(Q, cmap=cmap, aspect="equal")
    else:
        vmax = np.percentile(np.abs(Q), 98)
        cmap = "RdBu_r"
        im = ax.imshow(Q, cmap=cmap, aspect="equal", vmin=-vmax, vmax=vmax)

    for b in bounds:
        ax.axhline(b - 0.5, color="white", linewidth=0.5, alpha=0.8)
        ax.axvline(b - 0.5, color="white", linewidth=0.5, alpha=0.8)

    layers = sorted(set(k[0] for k in keys))
    n_per = len(keys) // len(layers)
    tick_pos = [l * n_per + n_per // 2 for l in range(len(layers))]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(l) for l in layers], fontsize=9)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([str(l) for l in layers], fontsize=9)

    parts = label.split("_vs_")
    ax.set_xlabel(f"Layer ({parts[1] if len(parts) > 1 else 'B'})")
    ax.set_ylabel(f"Layer ({parts[0] if len(parts) > 0 else 'A'})")

    ax.set_title(f"{model_name}  —  Cross-correlation ({label})\n"
                 f"{_metric_display(metric_name)}")
    fig.colorbar(im, ax=ax, shrink=0.8, label=_metric_display(metric_name))
    fig.tight_layout()
    safe_label = label.replace("/", "_")
    fpath = os.path.join(out_dir,
                         f"{model_name}_cross_{safe_label}_{metric_name}.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fpath}")
    return fpath


def plot_cross_diagonal(cross_Q_dict, keys, label, model_name, out_dir):
    """Plot the diagonal of cross-correlation matrices (intra-head cross-circuit).

    Shows how correlated QK and OV (or W and b) are for the same head,
    as a function of layer.
    """
    metrics = list(cross_Q_dict.keys())
    layers = np.array([k[0] for k in keys])
    heads = np.array([k[1] for k in keys])

    fig, axes = plt.subplots(1, len(metrics),
                              figsize=(6 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        diag = np.diag(cross_Q_dict[m])
        unique_layers = np.unique(layers)
        layer_means = [np.mean(diag[layers == l]) for l in unique_layers]
        layer_stds = [np.std(diag[layers == l]) for l in unique_layers]

        # scatter all heads
        ax.scatter(layers, diag, alpha=0.3, s=15, color="#636EFA", zorder=2)
        # layer means
        ax.errorbar(unique_layers, layer_means, yerr=layer_stds,
                     fmt="o-", markersize=6, capsize=3, color="#EF553B",
                     linewidth=2, zorder=3, label="layer mean")
        ax.set_xlabel("Layer")
        ax.set_ylabel(_metric_display(m))
        ax.set_title(f"Intra-head {label}", fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8)

    fig.suptitle(f"{model_name}  —  Same-head cross-circuit coupling",
                 fontsize=TITLE_SIZE)
    fig.tight_layout()
    safe_label = label.replace("/", "_")
    fpath = os.path.join(out_dir,
                         f"{model_name}_cross_diagonal_{safe_label}.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fpath}")
    return fpath


# ── Multi-model comparison plots ──────────────────────────────────────

MODEL_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#E45756",
]


def plot_eigenvalue_comparison(data_dir, models, revision="main",
                               weight_type="W_QK", metric="frob_cosine",
                               out_dir="."):
    """Overlay eigenvalue spectra of Q for multiple models on one plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        try:
            r = load_results(data_dir, model, revision, weight_type)
        except FileNotFoundError:
            continue
        if metric not in r["eigenvalues"]:
            continue
        eigvals = r["eigenvalues"][metric]
        abs_eig = np.sort(np.abs(eigvals))[::-1]
        N = r["metadata"]["n_layers"] * r["metadata"]["n_heads"]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(abs_eig, "o-", markersize=3, color=color,
                label=f"{model} ({N} heads)", alpha=0.8)

    ax.set_yscale("log")
    ax.set_xlabel("Index")
    ax.set_ylabel("$|\\lambda|$")
    ax.set_title(f"Eigenvalue spectra of $Q_{{hh'}}$ ({_metric_display(metric)})\n"
                 f"Component: {weight_type}")
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    fpath = os.path.join(out_dir,
                         f"all_models_{weight_type}_eigenvalues_{metric}.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fpath}")
    return fpath


def plot_eigen_stats_comparison(data_dir, models, revision="main",
                                weight_type="W_QK", out_dir="."):
    """Bar chart comparing condition number, NPR, stable rank across models."""
    names, conds, nprs, sranks = [], [], [], []
    mp_conds, mp_nprs, mp_sranks = [], [], []

    for model in models:
        try:
            r = load_results(data_dir, model, revision, weight_type)
        except FileNotFoundError:
            continue
        if "frob_cosine" not in r["Q"]:
            continue

        N = r["metadata"]["n_layers"] * r["metadata"]["n_heads"]
        d_head = r["metadata"]["head_dim"]
        gamma = N / (d_head ** 2)
        eigvals = np.linalg.eigvalsh(r["Q"]["frob_cosine"])[::-1]
        s = compute_Q_eigen_stats(eigvals, gamma)

        names.append(model)
        conds.append(s["condition_number"])
        nprs.append(s["npr"])
        sranks.append(s["stable_rank"])
        mp_conds.append(s["mp_condition_number"])
        mp_nprs.append(s["mp_npr"])
        mp_sranks.append(s["mp_stable_rank"])

    if not names:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(names))
    w = 0.35

    for ax, meas, pred, title, ylabel in [
        (axes[0], conds, mp_conds, "Condition number $C$", "$\\lambda_{max}/\\lambda_{min}$"),
        (axes[1], nprs, mp_nprs, "NPR", "$(\\Sigma\\lambda)^2 / (N \\Sigma\\lambda^2)$"),
        (axes[2], sranks, mp_sranks, "Stable rank", "$\\Sigma\\lambda^2 / \\lambda_{max}^2$"),
    ]:
        ax.bar(x - w / 2, meas, w, label="Measured", color="#EF553B", alpha=0.8)
        ax.bar(x + w / 2, pred, w, label="MP prediction", color="#636EFA", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    # Log scale for condition number (huge dynamic range)
    axes[0].set_yscale("log")

    fig.suptitle(f"Q eigenvalue statistics vs. Marchenko-Pastur ({weight_type})",
                 fontsize=TITLE_SIZE)
    fig.tight_layout()
    fpath = os.path.join(out_dir,
                         f"all_models_{weight_type}_eigen_stats.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fpath}")
    return fpath


# ── Auto-discovery helpers ────────────────────────────────────────────

def discover_weight_types(data_dir, model, revision="main"):
    """Find all weight types (W_QK, W_OV, etc.) with saved data for a model."""
    import glob
    pattern = os.path.join(data_dir, f"{model}_{revision}_*_metadata.json")
    weight_types = []
    cross_labels = []
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        suffix = "_metadata.json"
        prefix = f"{model}_{revision}_"
        mid = fname[len(prefix):-len(suffix)]
        # Check if it's a cross-correlation (contains "_vs_")
        if "_vs_" in mid:
            cross_labels.append(mid)
        else:
            weight_types.append(mid)
    return weight_types, cross_labels


def load_cross_results(data_dir, model, label, revision="main"):
    """Load saved cross-correlation data."""
    prefix = f"{model}_{revision}_{label}"

    with open(os.path.join(data_dir, f"{prefix}_metadata.json")) as f:
        metadata = json.load(f)

    Q_data = np.load(os.path.join(data_dir, f"{prefix}_Q.npz"))
    Q = {k.replace("Q_", ""): Q_data[k] for k in Q_data.files}

    keys = [tuple(k) for k in metadata["head_index"]]
    return {"Q": Q, "metadata": metadata, "keys": keys}


# ── Main ───────────────────────────────────────────────────────────────

def plot_weight_type(data_dir, model, revision, weight_type, out_dir, metrics_filter=None):
    """Generate all standard plots for one weight type."""
    print(f"\n{'='*50}")
    print(f"Weight type: {weight_type}")
    print(f"{'='*50}")

    r = load_results(data_dir, model, revision, weight_type)
    metrics = metrics_filter or r["metadata"]["metrics"]
    Q = {m: r["Q"][m] for m in metrics if m in r["Q"]}
    P_Q = {m: r["P_Q"][m] for m in metrics if m in r["P_Q"]}
    eig = {m: r["eigenvalues"][m] for m in metrics if m in r["eigenvalues"]}
    blk = {m: r["block_means"][m] for m in metrics if m in r["block_means"]}
    summary = {m: r["summary"][m] for m in metrics if m in r["summary"]}

    # Always include component in filename for consistency
    name_tag = f"{model}_{weight_type}"

    print(f"Generating figures for: {list(Q.keys())}")

    print("Heatmaps:")
    for m in Q:
        plot_heatmap(Q[m], r["keys"], m, name_tag, out_dir)

    if P_Q:
        print("P(Q) distributions:")
        plot_P_Q(P_Q, summary, name_tag, out_dir)

    if eig:
        print("Eigenvalue spectra:")
        plot_eigenvalues(eig, name_tag, out_dir)

    if blk:
        print("Block means:")
        plot_block_means(blk, r["metadata"], name_tag, out_dir)

    if Q:
        print("Correlation vs. layer distance:")
        plot_correlation_vs_layer_distance(P_Q, r["keys"], Q, name_tag, out_dir)

    if "frob_cosine" in Q:
        print("MP overlay:")
        _, eigen_stats = plot_mp_overlay(Q["frob_cosine"], r["metadata"],
                                         name_tag, out_dir)
        # Save eigenvalue stats as JSON
        stats_path = os.path.join(out_dir, f"{name_tag}_eigen_stats.json")
        with open(stats_path, "w") as f:
            json.dump({k: float(v) for k, v in eigen_stats.items()}, f, indent=2)
        print(f"  {stats_path}")

        print("Dominant eigenvectors:")
        plot_dominant_eigenvector(Q["frob_cosine"], r["metadata"],
                                  name_tag, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Plot head-head correlations from saved data")
    parser.add_argument("--data", type=str, default="corr_out",
                        help="Directory with saved correlation outputs")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--weight-type", type=str, default=None,
                        help="Specific weight type (default: auto-discover all)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output figure directory (default: {data}/figures)")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Subset of metrics to plot (default: all)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.data, "figures")
    os.makedirs(out_dir, exist_ok=True)

    if args.weight_type:
        # Single weight type (backward compat)
        plot_weight_type(args.data, args.model, args.revision,
                         args.weight_type, out_dir, args.metrics)
    else:
        # Auto-discover all weight types and cross-correlations
        weight_types, cross_labels = discover_weight_types(
            args.data, args.model, args.revision)

        if not weight_types and not cross_labels:
            # Fallback: try W_QK (old naming)
            weight_types = ["W_QK"]

        print(f"Found weight types: {weight_types}")
        if cross_labels:
            print(f"Found cross-correlations: {cross_labels}")

        # Self-correlations
        for wt in weight_types:
            try:
                plot_weight_type(args.data, args.model, args.revision,
                                 wt, out_dir, args.metrics)
            except Exception as e:
                print(f"  *** Error plotting {wt}: {e}")

        # Cross-correlations
        for label in cross_labels:
            try:
                print(f"\n{'='*50}")
                print(f"Cross-correlation: {label}")
                print(f"{'='*50}")
                cr = load_cross_results(args.data, args.model,
                                         label, args.revision)
                name_tag = f"{args.model}"
                for m, Q_mat in cr["Q"].items():
                    plot_cross_heatmap(Q_mat, cr["keys"], m, label,
                                       name_tag, out_dir)
                if cr["Q"]:
                    plot_cross_diagonal(cr["Q"], cr["keys"], label,
                                        name_tag, out_dir)
            except Exception as e:
                print(f"  *** Error plotting {label}: {e}")

    print(f"\nDone. Figures in {out_dir}")


if __name__ == "__main__":
    main()
