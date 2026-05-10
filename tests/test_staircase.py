import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "dashboards" / "pages"))
from sandbox import staircase, unfold, spacings, wigner_dyson, poisson


def test_staircase_is_nondecreasing():
    ev = np.array([3.0, 1.0, 4.0, 1.5, 2.0])
    sorted_ev, N = staircase(ev)
    assert np.all(np.diff(N) >= 0), "N(λ) must be non-decreasing"


def test_staircase_reaches_n():
    ev = np.random.default_rng(0).uniform(0, 10, 32)
    sorted_ev, N = staircase(ev)
    assert N[-1] == len(ev), "N(λ_max) must equal number of eigenvalues"


def test_staircase_starts_at_one():
    ev = np.array([2.0, 5.0, 1.0])
    sorted_ev, N = staircase(ev)
    assert N[0] == 1.0


def test_staircase_sorted():
    ev = np.array([5.0, 1.0, 3.0])
    sorted_ev, N = staircase(ev)
    assert list(sorted_ev) == [1.0, 3.0, 5.0]


def test_unfold_uniform_spectrum_is_linear():
    ev = np.linspace(0, 10, 20)
    xi = unfold(ev, degree=3)
    # For a uniform spectrum, unfolded values should be approximately linear
    residuals = xi - np.arange(1, len(xi) + 1)
    assert np.max(np.abs(residuals)) < 2.0, f"Unfolded residuals too large: {residuals}"


def test_unfold_preserves_length():
    ev = np.sort(np.random.default_rng(1).uniform(0, 5, 16))
    xi = unfold(ev, degree=4)
    assert len(xi) == len(ev)


def test_spacings_positive():
    ev = np.sort(np.random.default_rng(2).uniform(0, 10, 32))
    xi = unfold(ev, degree=4)
    s = spacings(xi)
    assert np.all(s > 0), "All spacings must be positive (monotone unfolded spectrum)"


def test_spacings_mean_is_one():
    ev = np.sort(np.random.default_rng(3).uniform(0, 10, 64))
    xi = unfold(ev, degree=4)
    s = spacings(xi)
    assert abs(s.mean() - 1.0) < 0.05, f"Normalized spacings should have mean ≈ 1, got {s.mean():.3f}"


def test_spacings_length():
    ev = np.sort(np.random.default_rng(4).uniform(0, 5, 20))
    xi = unfold(ev)
    s = spacings(xi)
    assert len(s) == len(xi) - 1


def test_wigner_dyson_nonneg():
    s = np.linspace(0, 4, 100)
    assert np.all(wigner_dyson(s) >= 0)


def test_poisson_at_zero_is_one():
    assert poisson(0.0) == pytest.approx(1.0)
