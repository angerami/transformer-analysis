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
    "differential_entropy" : differential_entropy
}

weight_bins_default = np.linspace(-1.5, 1.5, 1201)
sv_bins_default = np.linspace(0, 100, 100)

bins_dict_default = {
    "w_bins": weight_bins_default,
    "sv_bins": sv_bins_default,
}


def entropy_stat(h, centers):
    p = h["P_w"]
    h.update({"entropy": entropy(p)+np.log(centers[1]-centers[0])})


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
    'pythia-70m-deduped',
    'pythia-160m-deduped',
    'pythia-410m-deduped',
    'pythia-1b-deduped',
    'pythia-1.4b-deduped',
    'pythia-2.8b-deduped',
    'pythia-6.9b-deduped',
    'pythia-12b-deduped'
]

def get_model_versions(model_name):
    if model_name in PYTHIA_MODELS:
        return PYTHIA_REVISIONS
    return []
