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
    """P(Q) overlap distributions, one panel per metric."""
    metrics = list(P_Q_dict.keys())
    n_m = len(metrics)
    ncols = min(n_m, 2)
    nrows = (n_m + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_m == 1:
        axes = [axes]
    else:
        axes = axes.flat

    for ax, m in zip(axes, metrics):
        vals = P_Q_dict[m]
        ax.hist(vals, bins=60, density=True, alpha=0.7, color="#636EFA",
                edgecolor="white", linewidth=0.3)
        mu = summary[m]["mean_offdiag"]
        ax.axvline(mu, color="#EF553B", linestyle="--", linewidth=1.2,
                   label=f"mean = {mu:.4f}")
        ax.set_title(_metric_display(m), fontsize=10)
        ax.set_xlabel("$Q$ value")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    # hide unused axes
    for i in range(n_m, len(list(axes))):
        axes[i].set_visible(False) if hasattr(axes, '__len__') else None

    fig.suptitle(f"{model_name}  —  Overlap distributions $P(Q)$", fontsize=TITLE_SIZE)
    fig.tight_layout()
    fpath = os.path.join(out_dir, f"{model_name}_P_Q.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fpath}")
    return fpath


def plot_eigenvalues(eig_dict, model_name, out_dir):
    """Eigenvalue spectra of Q, one panel per metric."""
    metrics = list(eig_dict.keys())
    n_m = len(metrics)
    ncols = min(n_m, 2)
    nrows = (n_m + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_m == 1:
        axes = [axes]
    else:
        axes = axes.flat

    for ax, m in zip(axes, metrics):
        eigvals = eig_dict[m]
        # all metrics: plot |λ| on log scale
        abs_eig = np.sort(np.abs(eigvals))[::-1]
        ax.plot(abs_eig, "o-", markersize=3, color="#EF553B")
        ax.set_yscale("log")
        ax.set_ylabel("$|\\lambda|$")
        ax.set_title(_metric_display(m), fontsize=10)
        ax.set_xlabel("Index")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

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
        ax.set_xticklabels(layers)
        ax.set_yticklabels(layers)
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
    """Mean |Q| as a function of layer distance |l - l'|, for each metric."""
    layers = np.array([k[0] for k in keys])
    n = len(keys)

    metrics = list(Q_dict.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
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


def plot_mp_overlay(Q_frob, metadata, model_name, out_dir):
    """Eigenvalue spectrum of Q^(Frob) with Marchenko-Pastur prediction.

    Computes MP parameters from model dimensions (N_heads, d_head^2).
    Produces two panels: ordered eigenvalues with MP band, and
    eigenvalue histogram with MP density curve.
    """
    N = metadata["n_layers"] * metadata["n_heads"]
    d_head = metadata["head_dim"]
    p = d_head ** 2  # dimension of each flattened W_QK
    gamma = N / p

    lam_minus = (1 - np.sqrt(gamma)) ** 2
    lam_plus = (1 + np.sqrt(gamma)) ** 2

    # eigendecomposition
    eigvals = np.linalg.eigvalsh(Q_frob)[::-1]

    n_outliers = int(np.sum(eigvals > lam_plus))
    n_below = int(np.sum(eigvals < lam_minus))

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
        ax1.annotate(f"{n_outliers} outlier{'s' if n_outliers > 1 else ''} above MP edge",
                     xy=(n_outliers, lam_plus),
                     xytext=(min(n_outliers + 10, N // 3), lam_plus * 3),
                     fontsize=9, color="#636EFA",
                     arrowprops=dict(arrowstyle="->", color="#636EFA", lw=1))

    ax1.set_xlabel("Index")
    ax1.set_ylabel("$\\lambda$")
    ax1.set_title(f"Eigenvalue spectrum of $Q_{{hh'}}^{{\\mathrm{{(Frob)}}}}$\n"
                  f"({model_name}, {N} heads, $\\gamma$ = {gamma:.4f})")
    ax1.legend(fontsize=9, loc="upper right")

    # ── Right: histogram with MP density overlay ──
    bulk_cutoff = lam_plus * 2.5
    bins = np.linspace(0, min(bulk_cutoff, eigvals.max() * 1.1), 60)
    ax2.hist(eigvals, bins=bins, density=True, alpha=0.6, color="#EF553B",
             edgecolor="white", linewidth=0.3, label=f"{model_name} eigenvalues")

    lam_grid = np.linspace(0.01, lam_plus * 1.5, 500)
    mp_curve = _mp_density(lam_grid, gamma)
    ax2.plot(lam_grid, mp_curve, "-", color="#636EFA", linewidth=2.5,
             label=f"MP density ($\\gamma$ = {gamma:.4f})")

    # annotate outliers
    outlier_vals = eigvals[eigvals > lam_plus]
    for i, ov in enumerate(outlier_vals[:5]):
        x_arrow = min(ov, bins[-1] * 0.95)
        ax2.annotate(f"$\\lambda$={ov:.1f}",
                     xy=(x_arrow, 0),
                     xytext=(bins[-1] * 0.55, 0.4 + i * 0.35),
                     fontsize=8, color="#333",
                     arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax2.axvline(lam_plus, color="#636EFA", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Eigenvalue $\\lambda$")
    ax2.set_ylabel("Density")
    ax2.set_title("Eigenvalue distribution vs. MP prediction")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, bulk_cutoff)

    fig.tight_layout()
    fpath = os.path.join(out_dir, f"{model_name}_MP_overlay.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fpath}")
    return fpath


# ── Dominant eigenvector visualization ─────────────────────────────────

def plot_dominant_eigenvector(Q_frob, metadata, model_name, out_dir, n_modes=3):
    """Visualize the dominant eigenvector(s) of Q^(Frob) as layer×head heatmaps.

    Shows which heads participate in each collective mode.
    """
    n_layers = metadata["n_layers"]
    n_heads = metadata["n_heads"]
    N = n_layers * n_heads

    eigvals, eigvecs = np.linalg.eigh(Q_frob)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # MP edge for outlier identification
    d_head = metadata["head_dim"]
    gamma = N / (d_head ** 2)
    lam_plus = (1 + np.sqrt(gamma)) ** 2
    n_outliers = int(np.sum(eigvals > lam_plus))
    n_show = max(1, min(n_modes, n_outliers, 3))

    fig, axes = plt.subplots(1, n_show + 1, figsize=(5 * (n_show + 1), 5),
                              gridspec_kw={"width_ratios": [1] * n_show + [0.6]})

    for k in range(n_show):
        ax = axes[k]
        v = eigvecs[:, k]
        v_grid = v.reshape(n_layers, n_heads)
        vmax = np.max(np.abs(v_grid))
        im = ax.imshow(v_grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"Mode {k + 1}: $\\lambda_{{{k + 1}}}$ = {eigvals[k]:.1f}")
        fig.colorbar(im, ax=ax, shrink=0.7)

    # rightmost panel: layer loading for all shown modes
    ax = axes[-1]
    layers_arr = np.arange(n_layers)
    width = 0.8 / n_show
    colors = ["#636EFA", "#EF553B", "#00CC96"]
    for k in range(n_show):
        v = eigvecs[:, k]
        layer_loading = np.array([np.sum(v[l * n_heads:(l + 1) * n_heads] ** 2)
                                  for l in range(n_layers)])
        ax.barh(layers_arr + k * width, layer_loading, height=width,
                color=colors[k % len(colors)], alpha=0.7,
                label=f"Mode {k + 1}")
    ax.set_ylabel("Layer")
    ax.set_xlabel("$\\sum_h v^2_{(\\ell,h)}$")
    ax.set_title("Layer loading")
    ax.invert_yaxis()
    ax.legend(fontsize=8)

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


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot head-head correlations from saved data")
    parser.add_argument("--data", type=str, default="corr_out",
                        help="Directory with saved correlation outputs")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--weight-type", type=str, default="W_QK")
    parser.add_argument("--out", type=str, default=None,
                        help="Output figure directory (default: {data}/figures)")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Subset of metrics to plot (default: all)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.data, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.model} from {args.data} ...")
    r = load_results(args.data, args.model, args.revision, args.weight_type)

    metrics = args.metrics or r["metadata"]["metrics"]
    # filter to available
    Q = {m: r["Q"][m] for m in metrics if m in r["Q"]}
    P_Q = {m: r["P_Q"][m] for m in metrics if m in r["P_Q"]}
    eig = {m: r["eigenvalues"][m] for m in metrics if m in r["eigenvalues"]}
    blk = {m: r["block_means"][m] for m in metrics if m in r["block_means"]}
    summary = {m: r["summary"][m] for m in metrics if m in r["summary"]}

    print(f"\nGenerating figures for: {list(Q.keys())}")
    print(f"Output: {out_dir}\n")

    # 1. Heatmaps
    print("Heatmaps:")
    for m in Q:
        plot_heatmap(Q[m], r["keys"], m, args.model, out_dir)

    # 2. P(Q)
    if P_Q:
        print("P(Q) distributions:")
        plot_P_Q(P_Q, summary, args.model, out_dir)

    # 3. Eigenvalues
    if eig:
        print("Eigenvalue spectra:")
        plot_eigenvalues(eig, args.model, out_dir)

    # 4. Block means
    if blk:
        print("Block means:")
        plot_block_means(blk, r["metadata"], args.model, out_dir)

    # 5. Correlation vs layer distance
    if Q:
        print("Correlation vs. layer distance:")
        plot_correlation_vs_layer_distance(P_Q, r["keys"], Q, args.model, out_dir)

    # 6. MP overlay (frob_cosine only — needs the Q matrix, not just eigenvalues)
    if "frob_cosine" in Q:
        print("MP overlay:")
        plot_mp_overlay(Q["frob_cosine"], r["metadata"], args.model, out_dir)

    # 7. Dominant eigenvectors (frob_cosine)
    if "frob_cosine" in Q:
        print("Dominant eigenvectors:")
        plot_dominant_eigenvector(Q["frob_cosine"], r["metadata"],
                                  args.model, out_dir)

    print(f"\nDone. Figures in {out_dir}")


if __name__ == "__main__":
    main()
