import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "dashboards"))
from dashboard_utils import kde_from_histogram


def _make_histogram(mu=0.0, sigma=0.3, n_samples=5000, n_bins=200):
    rng = np.random.default_rng(42)
    data = rng.normal(mu, sigma, n_samples)
    counts, edges = np.histogram(data, bins=n_bins, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def test_output_length():
    centers, counts = _make_histogram()
    x, y = kde_from_histogram(centers, counts, n_points=300)
    assert len(x) == 300
    assert len(y) == 300


def test_peak_near_true_mean():
    centers, counts = _make_histogram(mu=0.0, sigma=0.3)
    x, y = kde_from_histogram(centers, counts)
    peak_x = x[np.argmax(y)]
    assert abs(peak_x) < 0.1, f"KDE peak at {peak_x}, expected near 0"


def test_kde_is_nonnegative():
    centers, counts = _make_histogram()
    _, y = kde_from_histogram(centers, counts)
    assert np.all(y >= 0), "KDE values should be non-negative"


def test_zero_counts_no_error():
    centers = np.linspace(-1, 1, 50)
    counts = np.zeros(50)
    x, y = kde_from_histogram(centers, counts)
    assert len(x) == 300
    assert np.all(y == 0)


def test_grid_spans_bin_range():
    centers, counts = _make_histogram()
    x, _ = kde_from_histogram(centers, counts)
    assert x[0] == pytest.approx(centers[0])
    assert x[-1] == pytest.approx(centers[-1])
