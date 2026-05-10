import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transformer_analysis.histogram_utils import (
    make_weight_bins,
    make_sv_bins,
    weight_bins_default,
    sv_bins_default,
)


def test_fixed_weight_bins_match_default():
    bins = make_weight_bins("fixed")
    np.testing.assert_array_equal(bins, weight_bins_default)


def test_fixed_sv_bins_match_default():
    bins = make_sv_bins("fixed")
    np.testing.assert_array_equal(bins, sv_bins_default)


def test_scott_weight_bins_monotone():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 0.3, 5000)
    bins = make_weight_bins("scott", data=data)
    assert np.all(np.diff(bins) > 0), "Scott bins are not strictly monotone"


def test_fd_weight_bins_monotone():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 0.5, 5000)
    bins = make_weight_bins("fd", data=data)
    assert np.all(np.diff(bins) > 0), "FD bins are not strictly monotone"


def test_scott_sv_bins_monotone():
    rng = np.random.default_rng(2)
    data = np.abs(rng.normal(5, 2, 1000))
    bins = make_sv_bins("scott", data=data)
    assert np.all(np.diff(bins) > 0)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown binning strategy"):
        make_weight_bins("unknown_strategy")


def test_unknown_sv_strategy_raises():
    with pytest.raises(ValueError):
        make_sv_bins("banana")


def test_adaptive_without_data_raises():
    with pytest.raises(ValueError, match="requires data"):
        make_weight_bins("scott", data=None)


def test_adaptive_sv_without_data_raises():
    with pytest.raises(ValueError, match="requires data"):
        make_sv_bins("fd", data=None)


def test_fixed_bins_cover_range():
    bins = make_weight_bins("fixed", value_range=(-2.0, 2.0), n_bins=400)
    assert bins[0] == pytest.approx(-2.0)
    assert bins[-1] == pytest.approx(2.0)
    assert len(bins) == 401
