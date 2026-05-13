"""
Tests for the transform stage (reprocess_metrics logic).

These tests exercise the metric re-computation on an existing on-disk dataset
without downloading any model weights from HuggingFace.
"""

import os

import numpy as np
import pandas as pd
import pytest
from datasets import load_from_disk

from transformer_analysis.head_metrics import normality_metrics, singular_value_metrics


NORMALITY_KEYS = list(normality_metrics.keys())
SV_KEYS = list(singular_value_metrics.keys())


def _recompute_on_dataset(dataset_dir):
    """
    Core of reprocess_metrics logic, extracted for direct testing without
    hitting the model_name / out_dir path resolution in the full function.
    """
    import json
    import numpy as np

    ds = load_from_disk(str(dataset_dir))
    df = ds.to_pandas()

    metadata_path = os.path.join(str(dataset_dir), "metadata.json")
    with open(metadata_path) as f:
        meta = json.load(f)

    w_bins = np.array(meta["w_bins"])
    centers = (w_bins[:-1] + w_bins[1:]) / 2

    new_columns = {}
    for idx, row in df.iterrows():
        h = row.to_dict()
        for fn in normality_metrics.values():
            fn(h, centers)
        svd = h.get("SVD")
        if svd is not None and hasattr(svd, "__len__") and len(svd) > 0:
            for fn in singular_value_metrics.values():
                fn(h, svd)
        for key, value in h.items():
            if key not in new_columns:
                new_columns[key] = [None] * len(df)
            new_columns[key][idx] = value

    for col, vals in new_columns.items():
        df[col] = vals

    return df


def test_transform_adds_normality_columns(tiny_dataset_dir):
    df = _recompute_on_dataset(tiny_dataset_dir)
    for key in ["entropy", "fit_mu", "fit_sigma", "kl_vs_empirical_normal"]:
        assert key in df.columns, f"Missing column: {key}"


def test_transform_adds_sv_columns(tiny_dataset_dir):
    df = _recompute_on_dataset(tiny_dataset_dir)
    wqk_rows = df[df["weight_type"] == "W_QK"]
    assert len(wqk_rows) > 0
    for key in ["normalized_participation_ratio", "spectral_entropy", "condition_number"]:
        assert key in df.columns, f"Missing column: {key}"
        assert wqk_rows[key].notna().all(), f"NaN values in {key} for W_QK rows"


def test_transform_preserves_row_count(tiny_dataset_dir):
    ds_before = load_from_disk(str(tiny_dataset_dir))
    df_after = _recompute_on_dataset(tiny_dataset_dir)
    assert len(df_after) == len(ds_before)


def test_transform_reproducible(tiny_dataset_dir):
    df1 = _recompute_on_dataset(tiny_dataset_dir)
    df2 = _recompute_on_dataset(tiny_dataset_dir)
    for col in ["entropy", "normalized_participation_ratio", "spectral_entropy"]:
        assert np.allclose(df1[col].values, df2[col].values, equal_nan=True), \
            f"Column {col} not reproducible across runs"
