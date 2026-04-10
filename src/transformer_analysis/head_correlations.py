"""Head-head correlation analysis for transformer attention weights.

Builds the N_heads x N_heads correlation matrix Q_{hh'} using multiple
correlation measures.  Manages memory by extracting and caching only the
flattened W_QK vectors per head (d_head^2 floats each), iterating over
all N(N-1)/2 unique pairs without holding the full model in memory.

Usage:
    from transformer_analysis.head_correlations import (
        HeadStore, compute_correlation_matrices, correlation_summary,
    )
"""

import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformer_analysis.pair_metrics import (
    frobenius_cosine,
    symmetric_kl,
    jensen_shannon,
    histogram_symmetric_kl,
    histogram_jensen_shannon,
    two_point_function,
    connected_correlation,
    pearson_correlation,
)


# ── Lightweight per-head cache ────────────────────────────────────────

class HeadStore:
    """Holds flattened weight vectors for all heads.

    Each head is identified by (layer, head_idx).  Stores only the
    numpy-flattened W_QK (or any single weight type) to keep memory
    manageable: for N_heads heads with d_head^2 elements each, total
    storage is N_heads * d_head^2 floats.

    For a 32-layer / 32-head model with d_head=128:
        1024 heads * 16384 floats * 4 bytes = 64 MB.
    """

    def __init__(self):
        self._data = {}   # (layer, head) -> np.ndarray (1-d)
        self._index = []  # ordered list of (layer, head) tuples

    def add(self, layer_idx, head_idx, W_flat):
        """Store a flattened weight vector for one head."""
        key = (layer_idx, head_idx)
        self._data[key] = np.ascontiguousarray(W_flat, dtype=np.float32)
        if key not in set(self._index):
            self._index.append(key)

    def get(self, layer_idx, head_idx):
        return self._data[(layer_idx, head_idx)]

    @property
    def keys(self):
        return list(self._index)

    @property
    def n_heads(self):
        return len(self._index)

    def memory_mb(self):
        total_bytes = sum(v.nbytes for v in self._data.values())
        return total_bytes / (1024 ** 2)


# ── Correlation matrix construction ───────────────────────────────────

# Available metric functions: name -> (func, needs_kde)
# "needs_kde" flags which metrics require the raw arrays (for KDE)
# vs. those that can work on pre-normalized vectors.

# Available metric functions: name -> (func, needs_kde, requires_equal_len)
# "needs_kde" flags which metrics require the raw arrays (for KDE).
# "requires_equal_len" flags metrics that need len(w_a) == len(w_b).
#   False = works with any pair of 1-d arrays (e.g. histogram divergences).

METRIC_REGISTRY = {
    "frob_cosine":        (frobenius_cosine,        False, True),
    "symmetric_kl":       (symmetric_kl,            True,  False),
    "jensen_shannon":     (jensen_shannon,           True,  False),
    "hist_symmetric_kl":  (histogram_symmetric_kl,   False, False),
    "hist_jensen_shannon":(histogram_jensen_shannon,  False, False),
    "two_point":          (two_point_function,        False, True),
    "connected_corr":     (connected_correlation,     False, True),
    "pearson_corr":       (pearson_correlation,        False, True),
}


def compute_correlation_matrices(
    store,
    metrics=("frob_cosine",),
    kde_kwargs=None,
    show_progress=True,
):
    """Compute N_heads x N_heads correlation matrices.

    Args:
        store: HeadStore with flattened weights for all heads.
        metrics: tuple of metric names from METRIC_REGISTRY.
        kde_kwargs: dict passed to KDE-based metrics (n_eval, bw_method).
        show_progress: show tqdm progress bar.

    Returns:
        dict of {metric_name: np.ndarray of shape (N, N)}.
        Index order matches store.keys.
    """
    if kde_kwargs is None:
        kde_kwargs = {}

    n = store.n_heads
    keys = store.keys
    results = {m: np.zeros((n, n), dtype=np.float32) for m in metrics}

    # diagonal
    for i, k in enumerate(keys):
        w = store.get(*k)
        for m in metrics:
            if m in ("frob_cosine", "pearson_corr"):
                results[m][i, i] = 1.0
            elif m in ("symmetric_kl", "jensen_shannon",
                        "hist_symmetric_kl", "hist_jensen_shannon"):
                results[m][i, i] = 0.0
            elif m == "connected_corr":
                results[m][i, i] = np.var(w)
            elif m == "two_point":
                results[m][i, i] = np.mean(w ** 2)

    n_pairs = n * (n - 1) // 2
    pair_iter = itertools.combinations(range(n), 2)
    if show_progress:
        pair_iter = tqdm(pair_iter, total=n_pairs, desc="Head pairs", leave=False)

    for i, j in pair_iter:
        w_i = store.get(*keys[i])
        w_j = store.get(*keys[j])
        for m in metrics:
            func, needs_kde, _ = METRIC_REGISTRY[m]
            if needs_kde:
                val = func(w_i, w_j, **kde_kwargs)
            else:
                val = func(w_i, w_j)
            results[m][i, j] = val
            results[m][j, i] = val

    return results


# ── Cross-correlation between two weight types ───────────────────────

def compute_cross_correlation_matrices(
    store_a,
    store_b,
    metrics=("frob_cosine",),
    kde_kwargs=None,
    show_progress=True,
):
    """Compute N_heads × N_heads cross-correlation matrices between two weight types.

    This is the entry point for all cross-term analyses:
      - QK ↔ OV  (how do attention routing and value circuits co-vary?)
      - W  ↔ b   (how do weight matrices relate to their biases?)
      - Any future pairings of HeadStores.

    Q^{cross}_{hh'} measures the relationship between the weight of head h
    in store_a and head h' in store_b.  Unlike self-correlations, this matrix
    is *not* symmetric in general (Q[i,j] ≠ Q[j,i] when the stores differ),
    though many metrics are symmetric in their arguments.

    The diagonal Q[h,h] gives the *intra-head* cross-circuit correlation
    (e.g., how aligned are QK and OV for the same head).

    Args:
        store_a: HeadStore for first weight type (e.g. W_QK)
        store_b: HeadStore for second weight type (e.g. W_OV)
        metrics: tuple of metric names from METRIC_REGISTRY.
            For cross-circuit analysis of matrices with different shapes,
            use metrics that operate on raw arrays (frob_cosine, pearson_corr,
            two_point, connected_corr).
        kde_kwargs: dict passed to KDE-based metrics.
        show_progress: show tqdm progress bar.

    Returns:
        dict of {metric_name: np.ndarray of shape (N, N)}.
        Index order: rows from store_a.keys, columns from store_b.keys.

    Notes:
        - store_a and store_b must have the same set of (layer, head) keys.
        - Metric functions receive (w_a[i], w_b[j]) — they need not have
          the same dimensionality, but many existing metrics assume equal-
          length vectors.  For W↔b cross-correlations, you may need
          dedicated metrics (to be added to METRIC_REGISTRY as needed).
    """
    if kde_kwargs is None:
        kde_kwargs = {}

    keys_a = store_a.keys
    keys_b = store_b.keys
    assert keys_a == keys_b, (
        "store_a and store_b must have the same head keys "
        f"(got {len(keys_a)} vs {len(keys_b)} heads)"
    )

    n = len(keys_a)

    # Check dimension compatibility and filter metrics
    sample_a = store_a.get(*keys_a[0])
    sample_b = store_b.get(*keys_b[0])
    same_dim = (len(sample_a) == len(sample_b))

    usable_metrics = []
    for m in metrics:
        _, _, requires_equal_len = METRIC_REGISTRY[m]
        if requires_equal_len and not same_dim:
            import logging
            logging.warning(
                f"Skipping metric '{m}' for cross-correlation: "
                f"requires equal-length vectors but got "
                f"{len(sample_a)} vs {len(sample_b)}. "
                f"Use distribution-based metrics (hist_symmetric_kl, "
                f"hist_jensen_shannon) for W↔b comparisons."
            )
        else:
            usable_metrics.append(m)

    if not usable_metrics:
        import logging
        logging.warning("No compatible metrics for this cross-correlation pair")
        return {}

    results = {m: np.zeros((n, n), dtype=np.float32) for m in usable_metrics}

    n_pairs = n * n  # full matrix, not just upper triangle
    pair_iter = ((i, j) for i in range(n) for j in range(n))
    if show_progress:
        pair_iter = tqdm(pair_iter, total=n_pairs,
                         desc="Cross-correlation pairs", leave=False)

    for i, j in pair_iter:
        w_i = store_a.get(*keys_a[i])
        w_j = store_b.get(*keys_b[j])
        for m in usable_metrics:
            func, needs_kde, _ = METRIC_REGISTRY[m]
            if needs_kde:
                val = func(w_i, w_j, **kde_kwargs)
            else:
                val = func(w_i, w_j)
            results[m][i, j] = val

    return results


# ── Summary statistics and P(Q) ──────────────────────────────────────

def correlation_summary(Q, keys):
    """Compute summary statistics of a correlation matrix Q.

    Returns a dict with:
      - mean_offdiag, std_offdiag: mean/std of upper-triangle
      - P_Q_values: the off-diagonal values (for histogramming)
      - intra_layer_mean: mean of pairs within same layer
      - inter_layer_mean: mean of pairs across layers
      - eigenvalues: sorted eigenvalues of Q (descending)
    """
    n = Q.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    offdiag = Q[triu_idx]

    # intra vs inter layer
    layers = np.array([k[0] for k in keys])
    intra_mask = layers[triu_idx[0]] == layers[triu_idx[1]]
    intra_vals = offdiag[intra_mask]
    inter_vals = offdiag[~intra_mask]

    # eigendecomposition
    eigvals = np.linalg.eigvalsh(Q)[::-1]  # descending

    return {
        "mean_offdiag": float(np.mean(offdiag)),
        "std_offdiag": float(np.std(offdiag)),
        "P_Q_values": offdiag,
        "intra_layer_mean": float(np.mean(intra_vals)) if len(intra_vals) > 0 else np.nan,
        "intra_layer_std": float(np.std(intra_vals)) if len(intra_vals) > 0 else np.nan,
        "inter_layer_mean": float(np.mean(inter_vals)) if len(inter_vals) > 0 else np.nan,
        "inter_layer_std": float(np.std(inter_vals)) if len(inter_vals) > 0 else np.nan,
        "eigenvalues": eigvals,
        "n_heads": n,
    }


def layer_block_means(Q, keys):
    """Compute the mean correlation per (layer_a, layer_b) block.

    Returns a 2-d array indexed by (layer_a, layer_b).
    """
    layers = np.array([k[0] for k in keys])
    unique_layers = np.unique(layers)
    n_layers = len(unique_layers)
    block = np.zeros((n_layers, n_layers))

    for a_idx, la in enumerate(unique_layers):
        for b_idx, lb in enumerate(unique_layers):
            mask_a = layers == la
            mask_b = layers == lb
            sub = Q[np.ix_(mask_a, mask_b)]
            if a_idx == b_idx:
                # exclude diagonal for intra-layer
                n_h = sub.shape[0]
                if n_h > 1:
                    triu = sub[np.triu_indices(n_h, k=1)]
                    block[a_idx, b_idx] = np.mean(triu) if len(triu) > 0 else 0.0
                else:
                    block[a_idx, b_idx] = 0.0
            else:
                block[a_idx, b_idx] = np.mean(sub)
    return block, unique_layers


def correlation_to_dataframe(Q, keys, metric_name="frob_cosine"):
    """Convert a correlation matrix to a long-form DataFrame.

    Includes only upper triangle (i < j).
    """
    rows = []
    n = Q.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            l1, h1 = keys[i]
            l2, h2 = keys[j]
            rows.append({
                "layer_1": l1, "head_1": h1,
                "layer_2": l2, "head_2": h2,
                "same_layer": l1 == l2,
                "layer_distance": abs(l1 - l2),
                "metric": metric_name,
                "value": float(Q[i, j]),
            })
    return pd.DataFrame(rows)
