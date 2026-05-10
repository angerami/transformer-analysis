import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import entropy, kurtosis, norm, skew, differential_entropy

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
        # np.linspace(0, 100, 100) matches sv_bins_default (100 edges = 99 bins)
        return np.linspace(value_range[0], value_range[1], n_bins)
    return _adaptive_bins(strategy, data, value_range)


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


def sv_mean(h, svd_array):
    """Compute mean of singular values."""
    h.update({"sv_mean": np.mean(svd_array)})


def sv_variance(h, svd_array):
    """Compute variance of singular values."""
    h.update({"sv_variance": np.var(svd_array)})


def sv_skewness(h, svd_array):
    """Compute skewness of singular values."""
    h.update({"sv_skewness": skew(svd_array)})


def sv_kurtosis_stat(h, svd_array):
    """Compute kurtosis of singular values."""
    h.update({"sv_kurtosis": kurtosis(svd_array)})


def sv_sum(h, svd_array):
    """Compute sum of singular values (Σσ)."""
    h.update({"sv_sum": np.sum(svd_array)})


def sv_sum_squares(h, svd_array):
    """Compute sum of squared singular values (Σσ²)."""
    h.update({"sv_sum_squares": np.sum(svd_array**2)})


def participation_ratio(h, svd_array):
    """Compute participation ratio: PR = (Σσ)² / Σσ²."""
    sum_sv = np.sum(svd_array)
    sum_sv2 = np.sum(svd_array**2)
    pr = (sum_sv**2) / sum_sv2 if sum_sv2 > 0 else np.nan
    h.update({"participation_ratio": pr})


def normalized_participation_ratio(h, svd_array, d_head=None):
    """
    Compute normalized participation ratio: PR / d_head.

    If d_head is not provided in h, it will be inferred from the length
    of the singular value array.
    """
    sum_sv = np.sum(svd_array)
    sum_sv2 = np.sum(svd_array**2)
    pr = (sum_sv**2) / sum_sv2 if sum_sv2 > 0 else np.nan

    # Get d_head from h if available, otherwise use length of svd_array
    if d_head is None:
        d_head = h.get("d_head", len(svd_array))

    npr = pr / d_head if d_head > 0 and not np.isnan(pr) else np.nan
    h.update({"normalized_participation_ratio": npr})


def spectral_entropy(h, svd_array):
    """
    Compute spectral entropy: -Σ(p_i * log(p_i)) where p_i = σ_i² / Σσ².
    """
    sv2 = svd_array**2
    sum_sv2 = np.sum(sv2)

    if sum_sv2 > 0:
        p = sv2 / sum_sv2
        p = p[p > 0]  # Remove zeros to avoid log(0)
        se = -np.sum(p * np.log(p))
    else:
        se = np.nan

    h.update({"spectral_entropy": se})


def condition_number(h, svd_array):
    """
    Compute condition number: σ_max / σ_min.

    Only considers non-zero singular values.
    """
    sv_nonzero = svd_array[svd_array > 1e-10]  # Filter out numerical zeros

    if len(sv_nonzero) > 0:
        cn = sv_nonzero[0] / sv_nonzero[-1] if sv_nonzero[-1] > 0 else np.nan
    else:
        cn = np.nan

    h.update({"condition_number": cn})


def stable_rank(h, svd_array):
    """Compute stable rank: Σσ² / σ_max²."""
    sum_sv2 = np.sum(svd_array**2)
    sv_max = svd_array[0] if len(svd_array) > 0 else 0

    sr = sum_sv2 / (sv_max**2) if sv_max > 0 else np.nan
    h.update({"stable_rank": sr})


normality_metrics = {
    "entropy": entropy_stat,
    "fit_normal": fit_normal,
    "kl_vs_empirical_normal": kl_vs_empirical_normal,
}

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

PYTHIA_REVISIONS = [
    "step0",
    "step1",
    "step2",
    "step4",
    "step8",
    "step16",
    "step32",
    "step64",
    "step128",
    "step256",
    "step512",
] + [f"step{step}" for step in range(1000, 144000, 1000)]

PYTHIA_MODELS = [
    "pythia-70m-deduped",
    "pythia-160m-deduped",
    "pythia-410m-deduped",
    "pythia-1b-deduped",
    "pythia-1.4b-deduped",
    "pythia-2.8b-deduped",
    "pythia-6.9b-deduped",
    "pythia-12b-deduped",
]


def get_model_versions(model_name):
    if model_name in PYTHIA_MODELS:
        return PYTHIA_REVISIONS
    return []
