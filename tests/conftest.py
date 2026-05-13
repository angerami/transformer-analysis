"""Shared fixtures for all tests."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from transformer_analysis.metrics import (
    stats_config_default,
    weight_bins_default,
    sv_bins_default,
    normality_metrics,
    singular_value_metrics,
)


@pytest.fixture(scope="session")
def tiny_config():
    """Minimal analysis config: 1 layer, 2 heads, d_model=16, head_dim=8."""
    cfg = SimpleNamespace()
    cfg.weight_type = ["W_Q", "W_K", "W_QK"]
    cfg.stats = stats_config_default.copy()
    cfg.w_bins = weight_bins_default.copy()
    cfg.sv_bins = sv_bins_default.copy()
    cfg.use_density = True
    cfg.n_heads = 2
    cfg.d_model = 16
    cfg.head_dim = 8
    cfg.n_layers = 1
    cfg.low_rank_svd_approximation = False
    cfg.top_k_svd = -1
    return cfg


@pytest.fixture(scope="session")
def tiny_weights(tiny_config):
    """Fixed-seed random weights matching tiny_config dimensions."""
    rng = torch.Generator()
    rng.manual_seed(42)
    W_Q = torch.randn(tiny_config.n_heads, tiny_config.head_dim, tiny_config.d_model, generator=rng)
    W_K = torch.randn(tiny_config.n_heads, tiny_config.head_dim, tiny_config.d_model, generator=rng)
    return {"W_Q": W_Q, "W_K": W_K}


@pytest.fixture()
def tiny_dataset_dir(tmp_path, tiny_config, tiny_weights):
    """Build a minimal on-disk HF Dataset + metadata.json for transform tests."""
    from datasets import Dataset
    from transformer_analysis.attn_head_analysis import LayerHeadContainer

    lhc = LayerHeadContainer(0, tiny_config)
    lhc.analyze_layer(tiny_weights)
    lhc.post_process()
    df = lhc.to_pandas()
    df["model"] = "test-model"
    df["job_uuid"] = "test"
    df["job_id"] = "test"

    ds = Dataset.from_pandas(df)
    ds.save_to_disk(str(tmp_path))

    cfg_dict = vars(tiny_config).copy()
    cfg_dict["stats"] = {k: v.__name__ for k, v in cfg_dict["stats"].items()}
    cfg_dict["w_bins"] = cfg_dict["w_bins"].tolist()
    cfg_dict["sv_bins"] = cfg_dict["sv_bins"].tolist()
    with open(os.path.join(str(tmp_path), "metadata.json"), "w") as f:
        json.dump(cfg_dict, f)

    return tmp_path
