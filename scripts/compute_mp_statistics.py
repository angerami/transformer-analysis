#!/usr/bin/env python3
"""Compute Marchenko-Pastur statistics and distributions for all models.

Generates ground truth / null hypothesis reference data for eigenvalue statistics
of random W_QK matrices at various (d_model, d_head) scales.

Outputs:
    mp_statistics.npz - consolidated data file with:
        - Statistics distributions (max, PR, NPR, entropy, cond, stable_rank) for each model
        - Eigenvalue density curves (MP and exact product ensemble)
        - Analytical predictions (MP and exact)
        - Model metadata (d_model, d_head, n_heads, n_layers)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL_DIMS = {
    "gpt2":            (12, 12,  64),
    "gpt2-medium":     (24, 16,  64),
    "gpt2-large":      (36, 20,  64),
    "gpt2-xl":         (48, 25,  64),
    "pythia-70m-deduped":   (6,  8,  64),
    "pythia-160m-deduped":  (12, 12, 64),
    "pythia-410m-deduped":  (24, 16, 64),
    "pythia-1b-deduped":    (16, 8, 128),
    "pythia-1.4b-deduped":  (24, 16, 128),
    "pythia-2.8b-deduped":  (32, 32, 80),
    "pythia-6.9b-deduped":  (32, 32, 128),
    "pythia-12b-deduped":   (36, 40, 128),
    "llama-3.1-8b":    (32, 32, 128),
    "mistral-7b-v0.3": (32, 32, 128),
}


def generate_low_rank(d_n, d_m, mu=0, sigma=1):
    W_Q = np.random.normal(loc=mu, scale=sigma, size=d_n * d_m).reshape(d_n, d_m)
    W_K = np.random.normal(loc=mu, scale=sigma, size=d_n * d_m).reshape(d_m, d_n)
    W_QK = W_Q @ W_K / np.sqrt(d_n)
    _, S, _ = np.linalg.svd(W_QK)
    return S


def compute_eig_stat(eigenvalues, stat_type, d_model, d_head):
    lam = eigenvalues
    if stat_type == "max":
        return np.max(lam)
    elif stat_type == "condition_number":
        lam_nz = lam[:d_head]
        if len(lam_nz) > 0 and lam_nz[-1] > 0:
            return lam_nz[0] / lam_nz[-1]
        return 0
    elif stat_type == "participation_ratio":
        return np.sum(lam)**2 / np.sum(lam**2) if np.sum(lam**2) > 0 else 0
    elif stat_type == "normalized_participation_ratio":
        pr = np.sum(lam)**2 / np.sum(lam**2) if np.sum(lam**2) > 0 else 0
        return pr / d_head if d_head > 0 else 0
    elif stat_type == "spectral_entropy":
        s = np.sum(lam)
        if s > 0:
            p = lam / s
            p = p[p > 0]
            return -np.sum(p * np.log(p))
        return 0
    elif stat_type == "stable_rank":
        return np.sum(lam**2) / np.max(lam)**2 if np.max(lam) > 0 else 0
    return 0


def collect_eig_stats(d_model, d_head, n_samples=500, sigma=1.0):
    stat_names = [
        "max", "participation_ratio", "normalized_participation_ratio",
        "spectral_entropy", "condition_number", "stable_rank"
    ]
    stats = {k: np.zeros(n_samples) for k in stat_names}
    singular_values = []
    for i in range(n_samples):
        S = generate_low_rank(d_model, d_head, mu=0, sigma=sigma)
        singular_values.extend(S)
        eigenvalues = S**2
        for k in stat_names:
            stats[k][i] = compute_eig_stat(eigenvalues, k, d_model, d_head)
    return stats, np.array(singular_values)


def marchenko_pastur_density(lam, gamma, sigma_hat_sq, n_points=1000):
    lam_min = sigma_hat_sq * (1 - np.sqrt(gamma))**2
    lam_max = sigma_hat_sq * (1 + np.sqrt(gamma))**2
    lam_grid = np.linspace(lam_min + 1e-10, lam_max - 1e-10, n_points)
    rho = np.zeros_like(lam_grid)
    mask = (lam_grid >= lam_min) & (lam_grid <= lam_max)
    l = lam_grid[mask]
    rho[mask] = np.sqrt((lam_max - l) * (l - lam_min)) / (2 * np.pi * sigma_hat_sq * gamma * l)
    return lam_grid, rho


def product_ensemble_edges(d, d_h, sigma=1.0):
    gamma = d_h / d
    alpha = np.sqrt(1 + 8/gamma)
    z_plus  = (-1 + alpha) / 4
    z_minus = (-1 - alpha) / 4
    def inv_psi(z):
        return (1 + z) * (1 + gamma*z)**2 / z
    scale = d * sigma**4
    return scale * inv_psi(z_minus), scale * inv_psi(z_plus)


def pme_predictions(d, d_h, sigma=1.0):
    """Product Multiplicative Ensemble predictions using S-transform edges."""
    gamma = d_h / d
    s2 = d * sigma**4
    E_lam = s2
    E_lam2_exact = sigma**8 * (d**2 + 2*d*d_h + 2*d + d_h + 3)
    sum_lam = E_lam * d_h
    sum_lam2 = E_lam2_exact * d_h
    pr = sum_lam**2 / sum_lam2 if sum_lam2 > 0 else 0
    npr = pr / d_h if d_h > 0 else 0
    lam_min, lam_max = product_ensemble_edges(d, d_h, sigma)
    stable_rank = sum_lam2 / lam_max**2 if lam_max > 0 else 0
    return {
        "max": lam_max,
        "condition_number": lam_max / lam_min if lam_min > 0 else np.inf,
        "participation_ratio": pr,
        "normalized_participation_ratio": npr,
        "stable_rank": stable_rank,
        "lam_min": lam_min,
        "E_lam": E_lam,
        "E_lam2": E_lam2_exact,
    }


def mp_predictions_direct(d, d_h, sigma=1.0):
    gamma = d_h / d
    s2 = d * sigma**4
    lam_max = s2 * (1 + np.sqrt(gamma))**2
    lam_min = s2 * (1 - np.sqrt(gamma))**2
    cond = lam_max / lam_min if lam_min > 0 else np.inf
    E_lam = s2
    E_lam2 = (1 + gamma) * s2**2
    pr = d_h * E_lam**2 / E_lam2
    npr = pr / d_h
    stable_rank = d_h * E_lam2 / lam_max**2
    return {
        "max": lam_max, "condition_number": cond,
        "participation_ratio": pr, "normalized_participation_ratio": npr,
        "stable_rank": stable_rank,
        "lam_min": lam_min, "E_lam": E_lam, "E_lam2": E_lam2,
    }


def plot_stat_distributions(stats, d_model, d_head, n_bins=40):
    import matplotlib.pyplot as plt

    nice_names = {
        "max": r"$\lambda_{\max}$",
        "participation_ratio": r"PR $= (\sum \lambda)^2 / \sum \lambda^2$",
        "normalized_participation_ratio": r"Normalized PR / $d_h$",
        "spectral_entropy": r"Spectral entropy $-\sum p_i \ln p_i$",
        "condition_number": r"Condition number $\lambda_1 / \lambda_{d_h}$",
        "stable_rank": r"Stable rank $\sum \lambda^2 / \lambda_{\max}^2$",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for idx, (key, vals) in enumerate(stats.items()):
        ax = axes[idx]
        ax.hist(vals, bins=n_bins, density=True, alpha=0.7, edgecolor='k', linewidth=0.3)
        ax.set_xlabel(nice_names.get(key, key), fontsize=10)
        ax.set_ylabel("Density")
        mean, std = np.mean(vals), np.std(vals)
        ax.axvline(mean, color='red', ls='--', lw=1.2, label=f"mean={mean:.2f}")
        ax.legend(fontsize=8)
        ax.set_title(f"d={d_model}, d_h={d_head}", fontsize=9)
    plt.tight_layout()
    return fig


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    parser = argparse.ArgumentParser(description="Compute MP statistics for all models")
    parser.add_argument("--out", type=str, default="mp_statistics",
                        help="Output directory")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of MC samples per model")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only compute for these models")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots for each model")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    models_to_run = args.models if args.models else list(MODEL_DIMS.keys())

    all_data = {}

    for model_name in models_to_run:
        if model_name not in MODEL_DIMS:
            logging.warning(f"Unknown model {model_name}, skipping")
            continue

        n_layers, n_heads, d_head = MODEL_DIMS[model_name]
        d_model = n_heads * d_head

        logging.info(f"Processing {model_name}: d_model={d_model}, d_head={d_head}, n_heads={n_heads}")

        stats, singular_values = collect_eig_stats(d_model, d_head, n_samples=args.n_samples)

        mp_pred = mp_predictions_direct(d_model, d_head)
        pme_pred = pme_predictions(d_model, d_head)

        mp_lam, mp_density = marchenko_pastur_density(
            None, d_head / d_model, d_model * 1.0**4, n_points=1000
        )

        all_data[model_name] = {
            "d_model": d_model,
            "d_head": d_head,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "stats": stats,
            "singular_values": singular_values,
            "mp_predictions": mp_pred,
            "pme_predictions": pme_pred,
            "mp_density_lambda": mp_lam,
            "mp_density_rho": mp_density,
        }

        if args.plot:
            fig = plot_stat_distributions(stats, d_model, d_head)
            plot_path = os.path.join(args.out, f"{model_name}_stats.png")
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Saved plot to {plot_path}")

    output_file = os.path.join(args.out, "mp_statistics.npz")
    np.savez_compressed(output_file, **all_data)
    logging.info(f"Saved consolidated data to {output_file}")

    logging.info(f"\nSummary:")
    logging.info(f"Processed {len(all_data)} models")
    logging.info(f"Data file: {output_file}")
    if args.plot:
        logging.info(f"Plots saved to {args.out}/*.png")


if __name__ == "__main__":
    main()
