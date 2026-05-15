"""
Regression tests for the primary analysis stage (LayerHeadContainer).

These tests verify that the core computation (weight extraction → stats →
SVD → histogram) produces consistent output for fixed-seed inputs.
They do not download any models from HuggingFace.
"""

import numpy as np
import pytest

from transformer_analysis.head_analyzer import LayerHeadContainer


EXPECTED_COLUMNS = {
    "weight_type", "mean", "std", "skew", "kurtosis", "differential_entropy",
    "P_w", "SVD", "P_sv",
    "entropy", "fit_mu", "fit_sigma", "kl_vs_empirical_normal",
    "normalized_participation_ratio", "spectral_entropy", "condition_number",
    "head", "layer",
}

EXPECTED_WEIGHT_TYPES = {"W_Q", "W_K", "W_QK", "W_Q_gram", "W_K_gram", "QK_alignment"}


def test_output_shape(tiny_config, tiny_weights):
    lhc = LayerHeadContainer(0, tiny_config)
    lhc.analyze_layer(tiny_weights)
    lhc.post_process()
    df = lhc.to_pandas()

    # n_heads × n_weight_types rows
    assert len(df) == tiny_config.n_heads * len(tiny_config.weight_type)


def test_output_columns(tiny_config, tiny_weights):
    lhc = LayerHeadContainer(0, tiny_config)
    lhc.analyze_layer(tiny_weights)
    lhc.post_process()
    df = lhc.to_pandas()

    missing = EXPECTED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_weight_types_present(tiny_config, tiny_weights):
    lhc = LayerHeadContainer(0, tiny_config)
    lhc.analyze_layer(tiny_weights)
    lhc.post_process()
    df = lhc.to_pandas()

    assert set(df["weight_type"].unique()) == EXPECTED_WEIGHT_TYPES


def test_histogram_sums_to_one(tiny_config, tiny_weights):
    lhc = LayerHeadContainer(0, tiny_config)
    lhc.analyze_layer(tiny_weights)
    df = lhc.to_pandas()

    # Gram matrices have higher-variance entries that fall outside the standard
    # weight bins; only check raw weight types where the bins are appropriate.
    raw_types = {"W_Q", "W_K", "W_QK"}
    bin_width = tiny_config.w_bins[1] - tiny_config.w_bins[0]
    for _, row in df[df["weight_type"].isin(raw_types)].iterrows():
        total = np.sum(row["P_w"]) * bin_width
        assert np.isclose(total, 1.0, atol=0.01), f"P_w doesn't integrate to 1: {total}"


def test_svd_length(tiny_config, tiny_weights):
    lhc = LayerHeadContainer(0, tiny_config)
    lhc.analyze_layer(tiny_weights)
    df = lhc.to_pandas()

    wqk_rows = df[df["weight_type"] == "W_QK"]
    # W_QK is d_model x d_model with rank <= head_dim. The factored SVD path
    # produces head_dim singular values, then pads to d_model with zeros.
    for _, row in wqk_rows.iterrows():
        svd = row["SVD"]
        assert len(svd) == tiny_config.d_model
        sorted_desc = np.sort(svd)[::-1]
        assert np.allclose(sorted_desc[tiny_config.head_dim:], 0.0, atol=1e-6)


def test_mean_std_consistent(tiny_config, tiny_weights):
    """mean and std from stats should be finite and in a reasonable range for N(0,1) inputs."""
    lhc = LayerHeadContainer(0, tiny_config)
    lhc.analyze_layer(tiny_weights)
    df = lhc.to_pandas()

    assert df["mean"].notna().all()
    assert df["std"].notna().all()
    assert (df["std"] > 0).all()


def test_reproducible_output(tiny_config, tiny_weights):
    """Same inputs → identical output DataFrame (regression check)."""
    import pandas as pd

    lhc1 = LayerHeadContainer(0, tiny_config)
    lhc1.analyze_layer(tiny_weights)
    lhc1.post_process()
    df1 = lhc1.to_pandas()

    lhc2 = LayerHeadContainer(0, tiny_config)
    lhc2.analyze_layer(tiny_weights)
    lhc2.post_process()
    df2 = lhc2.to_pandas()

    for col in ["mean", "std", "skew", "kurtosis", "entropy", "normalized_participation_ratio"]:
        assert np.allclose(df1[col].values, df2[col].values, equal_nan=True), f"Column {col} not reproducible"
