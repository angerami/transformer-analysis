"""Unit tests for metric functions in metrics.py."""

import numpy as np
import pytest

from transformer_analysis.metrics import (
    weight_bins_default,
    entropy_stat,
    kl_vs_empirical_normal,
    fit_normal,
    normality_metrics,
    normalized_participation_ratio,
    spectral_entropy,
    condition_number,
    stable_rank,
    singular_value_metrics,
)


def _uniform_h(n_bins=1200):
    """Build a uniform P_w histogram accumulator."""
    p = np.ones(n_bins) / n_bins
    centers = (weight_bins_default[:-1] + weight_bins_default[1:]) / 2
    return {"P_w": p, "mean": 0.0, "std": 1.0}, centers


def _gaussian_sv(d=8, scale=1.0):
    """Singular values for a random Gaussian matrix (all equal to scale)."""
    return np.full(d, scale)


# ── Normality metrics ──────────────────────────────────────────────────────────

def test_entropy_stat_uniform():
    h, centers = _uniform_h()
    entropy_stat(h, centers)
    bin_width = centers[1] - centers[0]
    expected = np.log(len(centers)) + np.log(bin_width)
    assert np.isclose(h["entropy"], expected, rtol=1e-4)


def test_entropy_stat_populates_key():
    h, centers = _uniform_h()
    entropy_stat(h, centers)
    assert "entropy" in h


def test_kl_vs_empirical_normal_identity():
    # If P_w is already N(0,1) evaluated at bin centers, KL should be ~0.
    centers = (weight_bins_default[:-1] + weight_bins_default[1:]) / 2
    from scipy.stats import norm
    bin_width = centers[1] - centers[0]
    p = norm.pdf(centers, 0, 1) * bin_width
    p = p / p.sum()
    h = {"P_w": p, "mean": 0.0, "std": 1.0}
    kl_vs_empirical_normal(h, centers)
    assert h["kl_vs_empirical_normal"] >= 0
    assert h["kl_vs_empirical_normal"] < 0.05


def test_fit_normal_returns_params():
    h, centers = _uniform_h()
    h["mean"] = 0.0
    h["std"] = 0.3
    fit_normal(h, centers)
    assert "fit_mu" in h
    assert "fit_sigma" in h


def test_normality_metrics_keys():
    h, centers = _uniform_h()
    for fn in normality_metrics.values():
        fn(h, centers)
    for key in ["entropy", "fit_mu", "fit_sigma", "kl_vs_empirical_normal"]:
        assert key in h, f"Missing key: {key}"


# ── Singular value metrics ─────────────────────────────────────────────────────

def test_normalized_participation_ratio_identity():
    # For uniform singular values, PR = d, NPR = 1.
    d = 8
    sv = np.ones(d)
    h = {"d_head": d}
    normalized_participation_ratio(h, sv)
    assert np.isclose(h["normalized_participation_ratio"], 1.0, atol=1e-6)


def test_normalized_participation_ratio_rank1():
    # For rank-1 matrix (one nonzero sv), PR = 1, NPR = 1/d.
    d = 8
    sv = np.zeros(d)
    sv[0] = 5.0
    h = {"d_head": d}
    normalized_participation_ratio(h, sv)
    assert np.isclose(h["normalized_participation_ratio"], 1.0 / d, atol=1e-6)


def test_spectral_entropy_uniform():
    d = 8
    sv = np.ones(d)
    h = {}
    spectral_entropy(h, sv)
    assert np.isclose(h["spectral_entropy"], np.log(d), atol=1e-6)


def test_spectral_entropy_rank1():
    d = 8
    sv = np.zeros(d)
    sv[0] = 1.0
    h = {}
    spectral_entropy(h, sv)
    assert h["spectral_entropy"] == 0.0


def test_condition_number_identity():
    sv = np.array([3.0, 2.0, 1.0])
    h = {}
    condition_number(h, sv)
    assert np.isclose(h["condition_number"], 3.0, atol=1e-6)


def test_stable_rank_identity():
    sv = np.ones(4) * 2.0
    h = {}
    stable_rank(h, sv)
    assert np.isclose(h["stable_rank"], 4.0, atol=1e-6)


def test_all_sv_metrics_populate():
    d = 8
    sv = np.random.default_rng(42).random(d) + 0.1
    h = {"d_head": d}
    for fn in singular_value_metrics.values():
        fn(h, sv)
    expected_keys = [
        "sv_mean", "sv_variance", "sv_skewness", "sv_kurtosis",
        "sv_sum", "sv_sum_squares",
        "participation_ratio", "normalized_participation_ratio",
        "spectral_entropy", "condition_number", "stable_rank",
    ]
    for key in expected_keys:
        assert key in h, f"Missing key: {key}"
