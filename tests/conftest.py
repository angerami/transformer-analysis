"""Shared fixtures for all tests."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Make tests pick up this worktree's `src/` ahead of any installed package, so
# changes in the worktree are exercised even when `transformer_analysis` is
# installed editable against another path.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
import torch

from transformer_analysis.head_metrics import (
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
    cfg.weight_type = ["W_Q", "W_K", "W_QK", "W_Q_gram", "W_K_gram", "QK_alignment"]
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


@pytest.fixture
def tiny_hf_model_factory(tmp_path):
    """Build a tiny HF model with controlled weights, save in the on-disk format
    that `model_registry.extract_*_qkv` expects, and return enough to drive a
    round-trip test.

    Usage:
        path, model, info = tiny_hf_model_factory("gpt2", weight_init="randn", seed=0)

    weight_init:
      - "randn": fixed-seed torch.randn for all params
      - "arange": Q/K/V projection weights set to torch.arange(numel).reshape(...).float()
                  (other params seeded randn)
    """
    def _build(model_registry_name, weight_init="randn", seed=0):
        torch.manual_seed(seed)
        sub_dir = tmp_path / model_registry_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        if model_registry_name == "gpt2":
            # Use the base GPT2Model (not LMHead) — published openai-community/gpt2
            # safetensors uses `h.X.attn.c_attn.weight` keys (no `transformer.` prefix),
            # which is what extract_gpt2_qkv reads.
            from transformers import GPT2Config, GPT2Model
            cfg = GPT2Config(
                n_layer=1, n_head=2, n_embd=16,
                n_positions=32, vocab_size=64,
                attn_pdrop=0.0, resid_pdrop=0.0, embd_pdrop=0.0,
            )
            model = GPT2Model(cfg)
            qkv_param_names = ["h.0.attn.c_attn.weight"]
            info = {
                "n_heads": cfg.n_head, "d_model": cfg.n_embd,
                "head_dim": cfg.n_embd // cfg.n_head, "n_layers": cfg.n_layer,
                "save_kwargs": {"safe_serialization": True},
            }
        elif model_registry_name == "pythia-70m-deduped":
            from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
            cfg = GPTNeoXConfig(
                num_hidden_layers=1, num_attention_heads=2, hidden_size=16,
                intermediate_size=32, max_position_embeddings=32, vocab_size=64,
                hidden_dropout=0.0, attention_dropout=0.0,
            )
            model = GPTNeoXForCausalLM(cfg)
            qkv_param_names = ["gpt_neox.layers.0.attention.query_key_value.weight"]
            info = {
                "n_heads": cfg.num_attention_heads, "d_model": cfg.hidden_size,
                "head_dim": cfg.hidden_size // cfg.num_attention_heads,
                "n_layers": cfg.num_hidden_layers,
                "save_kwargs": {"safe_serialization": False},  # extractor reads pytorch_model.bin
            }
        else:
            raise ValueError(f"Unknown model_registry_name: {model_registry_name}")

        if weight_init == "arange":
            with torch.no_grad():
                state = dict(model.named_parameters())
                for name in qkv_param_names:
                    p = state[name]
                    p.copy_(torch.arange(p.numel(), dtype=torch.float32).reshape(p.shape))
        elif weight_init != "randn":
            raise ValueError(f"Unknown weight_init: {weight_init}")

        model.eval()
        model.save_pretrained(str(sub_dir), **info["save_kwargs"])
        info["qkv_param_names"] = qkv_param_names
        return str(sub_dir), model, info

    return _build


@pytest.fixture()
def tiny_dataset_dir(tmp_path, tiny_config, tiny_weights):
    """Build a minimal on-disk HF Dataset + metadata.json for transform tests."""
    from datasets import Dataset
    from transformer_analysis.head_analyzer import LayerHeadContainer

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
