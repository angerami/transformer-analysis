"""Metrics for comparing pairs of attention heads.

Two interpretations:
  - Matrix: Frobenius inner products, cosine similarity on flattened W
  - Ensemble: treat matrix elements as samples from a density,
              compute KL divergence and connected correlations between heads
"""

import numpy as np
from scipy.stats import gaussian_kde, entropy


# ── Matrix interpretation ──────────────────────────────────────────────

def frobenius_cosine(W1_flat, W2_flat):
    """Cosine similarity between two flattened weight matrices."""
    dot = np.dot(W1_flat, W2_flat)
    n1 = np.linalg.norm(W1_flat)
    n2 = np.linalg.norm(W2_flat)
    return dot / (n1 * n2) if (n1 > 0 and n2 > 0) else 0.0


def frobenius_inner_product(W1_flat, W2_flat):
    """Raw Frobenius inner product Tr(W1^T W2) via flattened vectors."""
    return np.dot(W1_flat, W2_flat)


# ── Ensemble interpretation (KDE-based) ───────────────────────────────

def _build_kde(x, bw_method="scott"):
    """Build a Gaussian KDE from a 1-d sample."""
    return gaussian_kde(x, bw_method=bw_method)


def kde_kl_divergence(x1, x2, n_eval=2048, bw_method="scott"):
    """KL divergence D_KL(p1 || p2) estimated via KDEs.

    Args:
        x1, x2: 1-d arrays of matrix elements (flattened weights).
        n_eval: number of evaluation points on a shared grid.
        bw_method: bandwidth selection for scipy gaussian_kde.

    Returns:
        D_KL(p1 || p2) in nats.
    """
    kde1 = _build_kde(x1, bw_method)
    kde2 = _build_kde(x2, bw_method)

    lo = min(x1.min(), x2.min())
    hi = max(x1.max(), x2.max())
    margin = 0.1 * (hi - lo)
    grid = np.linspace(lo - margin, hi + margin, n_eval)

    p = kde1(grid)
    q = kde2(grid)
    # floor to avoid log(0)
    eps = 1e-30
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    # normalize to proper pmf on the grid
    p /= p.sum()
    q /= q.sum()
    return entropy(p, q)


def symmetric_kl(x1, x2, **kwargs):
    """Symmetrized KL: 0.5 * (D_KL(p1||p2) + D_KL(p2||p1))."""
    return 0.5 * (kde_kl_divergence(x1, x2, **kwargs) +
                  kde_kl_divergence(x2, x1, **kwargs))


def jensen_shannon(x1, x2, n_eval=2048, bw_method="scott"):
    """Jensen-Shannon divergence (symmetric, bounded)."""
    kde1 = _build_kde(x1, bw_method)
    kde2 = _build_kde(x2, bw_method)

    lo = min(x1.min(), x2.min())
    hi = max(x1.max(), x2.max())
    margin = 0.1 * (hi - lo)
    grid = np.linspace(lo - margin, hi + margin, n_eval)

    p = kde1(grid)
    q = kde2(grid)
    eps = 1e-30
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def histogram_kl(x1, x2, bins=None, n_bins=200, density=True):
    """KL divergence between histograms of two weight arrays.

    Much faster than KDE-based KL: O(n_bins) per pair instead of
    O(n_data * n_eval).  Uses a shared bin range.
    """
    if bins is None:
        lo = min(x1.min(), x2.min())
        hi = max(x1.max(), x2.max())
        bins = np.linspace(lo, hi, n_bins + 1)
    h1, _ = np.histogram(x1, bins=bins, density=density)
    h2, _ = np.histogram(x2, bins=bins, density=density)
    eps = 1e-30
    h1 = np.maximum(h1, eps)
    h2 = np.maximum(h2, eps)
    h1 /= h1.sum()
    h2 /= h2.sum()
    return entropy(h1, h2)


def histogram_symmetric_kl(x1, x2, **kwargs):
    """Symmetrized histogram-based KL divergence."""
    return 0.5 * (histogram_kl(x1, x2, **kwargs) +
                  histogram_kl(x2, x1, **kwargs))


def histogram_jensen_shannon(x1, x2, bins=None, n_bins=200, density=True):
    """Jensen-Shannon divergence from histograms."""
    if bins is None:
        lo = min(x1.min(), x2.min())
        hi = max(x1.max(), x2.max())
        bins = np.linspace(lo, hi, n_bins + 1)
    h1, _ = np.histogram(x1, bins=bins, density=density)
    h2, _ = np.histogram(x2, bins=bins, density=density)
    eps = 1e-30
    h1 = np.maximum(h1, eps)
    h2 = np.maximum(h2, eps)
    h1 /= h1.sum()
    h2 /= h2.sum()
    m = 0.5 * (h1 + h2)
    return 0.5 * (entropy(h1, m) + entropy(h2, m))


def two_point_function(x1, x2):
    """Two-point function: <W1*W2> (raw, unnormalized).

    Same-index pairing of flattened weight elements.
    """
    return np.mean(x1 * x2)


def connected_correlation(x1, x2):
    """Connected (co)variance: <W1*W2> - <W1><W2>.

    Treats the element-wise product of two heads' flattened weights
    as a joint sample (same-index pairing).
    """
    return np.mean(x1 * x2) - np.mean(x1) * np.mean(x2)


def pearson_correlation(x1, x2):
    """Pearson correlation: (<W1*W2> - <W1><W2>) / sqrt(<W1^2><W2^2>).

    Normalized connected correlation.  The denominator uses the
    second moments (not the variances), so this is
    cov(W1,W2) / sqrt(E[W1^2] * E[W2^2]).
    """
    cov = np.mean(x1 * x2) - np.mean(x1) * np.mean(x2)
    e1sq = np.mean(x1 ** 2)
    e2sq = np.mean(x2 ** 2)
    denom = np.sqrt(e1sq * e2sq)
    return cov / denom if denom > 0 else 0.0


# ── Convenience: compute all pair metrics at once ─────────────────────

def compute_pair_metrics(W1_flat, W2_flat, n_eval=2048, bw_method="scott"):
    """Compute the full battery of pair metrics between two flattened W arrays.

    Returns a dict of scalar metrics.
    """
    return {
        "frob_cosine": frobenius_cosine(W1_flat, W2_flat),
        "frob_inner": frobenius_inner_product(W1_flat, W2_flat),
        "symmetric_kl": symmetric_kl(W1_flat, W2_flat,
                                     n_eval=n_eval, bw_method=bw_method),
        "jensen_shannon": jensen_shannon(W1_flat, W2_flat,
                                         n_eval=n_eval, bw_method=bw_method),
        "two_point": two_point_function(W1_flat, W2_flat),
        "connected_corr": connected_correlation(W1_flat, W2_flat),
        "pearson_corr": pearson_correlation(W1_flat, W2_flat),
    }
