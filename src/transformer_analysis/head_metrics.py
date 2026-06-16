import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import entropy, kurtosis, norm, skew, differential_entropy

# ---------------------------------------------------------------------------
# Bin configurations
# ---------------------------------------------------------------------------

weight_bins_default = np.linspace(-1.5, 1.5, 1201)
sv_bins_default = np.linspace(0, 100, 100)

bins_dict_default = {
    "w_bins": weight_bins_default,
    "sv_bins": sv_bins_default,
}


def _adaptive_bins(strategy, data, value_range):
    if strategy not in ("scott", "fd"):
        raise ValueError(f"Unknown binning strategy: '{strategy}'. Choose 'fixed', 'scott', or 'fd'.")
    if data is None:
        raise ValueError(f"strategy='{strategy}' requires data to be provided")
    data = np.asarray(data).ravel()
    if strategy == "scott":
        h = 3.49 * np.std(data) * len(data) ** (-1 / 3)
    else:  # fd
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        h = 2.0 * iqr * len(data) ** (-1 / 3) if iqr > 0 else 3.49 * np.std(data) * len(data) ** (-1 / 3)
    lo, hi = value_range if value_range else (data.min(), data.max())
    return np.arange(lo, hi + h, h)


def make_weight_bins(strategy="fixed", data=None, n_bins=1200, value_range=(-1.5, 1.5)):
    if strategy == "fixed":
        return np.linspace(value_range[0], value_range[1], n_bins + 1)
    return _adaptive_bins(strategy, data, value_range)


def make_sv_bins(strategy="fixed", data=None, n_bins=100, value_range=(0, 100)):
    if strategy == "fixed":
        return np.linspace(value_range[0], value_range[1], n_bins)
    return _adaptive_bins(strategy, data, value_range)

# ---------------------------------------------------------------------------
# Scalar stats applied to raw weight arrays
# ---------------------------------------------------------------------------

stats_config_default = {
    "sum": np.sum,
    "mean": np.mean,
    "std": np.std,
    "max": np.max,
    "min": np.min,
    "skew": skew,
    "kurtosis": kurtosis,
    "differential_entropy": differential_entropy,
}


def fast_histogram(x, bins, density=False):
    """O(n) histogram for uniform bins via direct indexing + bincount.

    Equivalent to np.histogram(x, bins, density=density) when `bins` is uniform
    (the project's fixed strategy), but avoids the O(n log B) searchsorted.
    Out-of-range values are excluded, matching np.histogram."""
    lo, hi = float(bins[0]), float(bins[-1])
    B = len(bins) - 1
    width = (hi - lo) / B
    idx = ((x - lo) / width).astype(np.intp)
    inside = (idx >= 0) & (idx < B)
    counts = np.bincount(idx[inside], minlength=B).astype(float)
    if density:
        n_in = int(inside.sum())
        counts /= (n_in * width) if n_in else 1.0
    return counts


def _hist_differential_entropy(P_w, bins, density):
    """Plug-in differential entropy H = -∫ f ln f dx from a (density) histogram.
    O(B); reuses the histogram instead of scipy's O(n log n) Vasicek sort."""
    width = float(bins[1] - bins[0])
    f = np.asarray(P_w, dtype=float)
    if not density:
        total = f.sum()
        f = f / (total * width) if total else f
    f = f[f > 0]
    return float(-width * np.sum(f * np.log(f)))


# Stats served from a single fused pass (shared central moments) instead of the
# independent, per-call scipy/numpy implementations. Entropy is taken from the
# histogram. Keys outside this set fall back to stats_config_default.
_FUSED_STAT_KEYS = {"sum", "mean", "std", "max", "min", "skew",
                    "kurtosis", "differential_entropy"}


def element_stats(x, bins, density=True, stat_keys=None, histo=True):
    """Histogram + scalar element stats for a flattened weight array, computed
    in one fused pass. Histogram and moments are numerically identical to the
    np.histogram / scipy implementations; differential_entropy is the histogram
    plug-in estimator (O(n) rather than scipy's O(n log n) sort)."""
    keys = list(stat_keys) if stat_keys is not None else list(_FUSED_STAT_KEYS)
    out = {}

    P_w = fast_histogram(x, bins, density=density) if histo else None
    if histo:
        out["P_w"] = P_w

    if "sum" in keys:
        out["sum"] = float(x.sum())
    if "max" in keys:
        out["max"] = float(x.max())
    if "min" in keys:
        out["min"] = float(x.min())

    if {"mean", "std", "skew", "kurtosis"} & set(keys):
        mean = float(x.mean())
        if "mean" in keys:
            out["mean"] = mean
        if {"std", "skew", "kurtosis"} & set(keys):
            d = x - mean
            d2 = d * d
            m2 = float(d2.mean())
            if "std" in keys:
                out["std"] = float(np.sqrt(m2))
            if m2 > 0:
                if "skew" in keys:
                    out["skew"] = float((d2 * d).mean() / m2 ** 1.5)
                if "kurtosis" in keys:
                    out["kurtosis"] = float((d2 * d2).mean() / m2 ** 2 - 3)
            else:
                out.update({k: 0.0 for k in ("skew", "kurtosis") if k in keys})

    if "differential_entropy" in keys:
        hist = P_w if P_w is not None else fast_histogram(x, bins, density=density)
        out["differential_entropy"] = _hist_differential_entropy(hist, bins, density)

    for k in keys:                      # fallback for any non-fused stat
        if k not in out:
            out[k] = stats_config_default[k](x)
    return out

# ---------------------------------------------------------------------------
# Normality metrics — called as fn(h, centers) where h is a dict accumulator
# ---------------------------------------------------------------------------


def entropy_stat(h, centers):
    p = h["P_w"]
    h.update({"entropy": entropy(p) + np.log(centers[1] - centers[0])})


def kl_vs_standard_normal(h, centers):
    p = h["P_w"]
    q = norm.pdf(centers, 0, 1)
    h.update({"kl_vs_standard_normal": entropy(p, q)})


def kl_vs_empirical_normal(h, centers):
    mu, sigma = h["mean"], h["std"]
    p = h["P_w"]
    q = norm.pdf(centers, mu, sigma)
    h.update({"kl_vs_empirical_normal": entropy(p, q)})


def kl_normal_vs_standard(h, centers):
    mu, sigma = h["mean"], h["std"]
    h.update(
        {"kl_vs_empirical_normal": 0.5 * (sigma**2 + mu**2 - 1 - np.log(sigma**2))}
    )


def fit_normal(h, centers, n_sigma=1.5):
    p = h["P_w"]
    mu, sigma = h["mean"], h["std"]
    if np.isnan(mu) or np.isnan(sigma):
        h.update({"fit_mu": np.nan, "fit_sigma": np.nan})
    mask = np.abs(centers - mu) <= n_sigma * sigma

    def gaussian(x, mu, sigma):
        return norm.pdf(x, mu, sigma)

    try:
        if mask.sum() > 1:
            popt, _ = curve_fit(gaussian, centers[mask], p[mask], p0=[mu, sigma])
        else:
            popt = [np.nan, np.nan]
    except RuntimeError:
        popt = [np.nan, np.nan]
    h.update({"fit_mu": popt[0], "fit_sigma": popt[1]})


normality_metrics = {
    "entropy": entropy_stat,
    "fit_normal": fit_normal,
    "kl_vs_empirical_normal": kl_vs_empirical_normal,
}

# ---------------------------------------------------------------------------
# Singular value metrics — called as fn(h, svd_array)
# ---------------------------------------------------------------------------


def sv_mean(h, svd_array):
    h.update({"sv_mean": np.mean(svd_array)})


def sv_variance(h, svd_array):
    h.update({"sv_variance": np.var(svd_array)})


def sv_skewness(h, svd_array):
    h.update({"sv_skewness": skew(svd_array)})


def sv_kurtosis_stat(h, svd_array):
    h.update({"sv_kurtosis": kurtosis(svd_array)})


def sv_sum(h, svd_array):
    h.update({"sv_sum": np.sum(svd_array)})


def sv_sum_squares(h, svd_array):
    h.update({"sv_sum_squares": np.sum(svd_array**2)})


def participation_ratio(h, svd_array):
    sum_sv = np.sum(svd_array)
    sum_sv2 = np.sum(svd_array**2)
    pr = (sum_sv**2) / sum_sv2 if sum_sv2 > 0 else np.nan
    h.update({"participation_ratio": pr})


def normalized_participation_ratio(h, svd_array, d_head=None):
    sum_sv = np.sum(svd_array)
    sum_sv2 = np.sum(svd_array**2)
    pr = (sum_sv**2) / sum_sv2 if sum_sv2 > 0 else np.nan
    if d_head is None:
        d_head = h.get("d_head", len(svd_array))
    npr = pr / d_head if d_head > 0 and not np.isnan(pr) else np.nan
    h.update({"normalized_participation_ratio": npr})


def spectral_entropy(h, svd_array):
    sv2 = svd_array**2
    sum_sv2 = np.sum(sv2)
    if sum_sv2 > 0:
        p = sv2 / sum_sv2
        p = p[p > 0]
        se = -np.sum(p * np.log(p))
    else:
        se = np.nan
    h.update({"spectral_entropy": se})


def condition_number(h, svd_array):
    sv_nonzero = svd_array[svd_array > 1e-10]
    if len(sv_nonzero) > 0:
        cn = sv_nonzero[0] / sv_nonzero[-1] if sv_nonzero[-1] > 0 else np.nan
    else:
        cn = np.nan
    h.update({"condition_number": cn})


def stable_rank(h, svd_array):
    sum_sv2 = np.sum(svd_array**2)
    sv_max = svd_array[0] if len(svd_array) > 0 else 0
    sr = sum_sv2 / (sv_max**2) if sv_max > 0 else np.nan
    h.update({"stable_rank": sr})


singular_value_metrics = {
    "sv_mean": sv_mean,
    "sv_variance": sv_variance,
    "sv_skewness": sv_skewness,
    "sv_kurtosis": sv_kurtosis_stat,
    "sv_sum": sv_sum,
    "sv_sum_squares": sv_sum_squares,
    "participation_ratio": participation_ratio,
    "normalized_participation_ratio": normalized_participation_ratio,
    "spectral_entropy": spectral_entropy,
    "condition_number": condition_number,
    "stable_rank": stable_rank,
}
