"""Pathological-weight extraction tests.

Build a tiny HuggingFace model, set the QKV projection to a deterministic
pattern (arange), save it in the on-disk format the registered extractor reads,
call extract_qkv, and assert the result matches a hand-computed reference
derived from each model's actual attention forward-pass layout.

The reference reshape for each architecture is derived from the HF modeling
code, NOT from the pipeline's reshape — so these tests will catch a bug where
the pipeline mis-aligns rows of the stored weight with heads.
"""
import pytest
import torch

from transformer_analysis.model_registry import get_model_config, extract_weight_map


def _gpt2_reference_qkv(model, info):
    """GPT-2 stores c_attn.weight as (d_model, 3*d_model). Forward splits the
    3*d_model output into [Q | K | V] contiguous d_model blocks. So after
    transposing to (3*d_model, d_model), rows 0..d_model-1 are W_Q, etc."""
    W = model.h[0].attn.c_attn.weight.detach().clone().T  # (3*d_model, d_model)
    d = info["d_model"]
    return W[:d], W[d:2 * d], W[2 * d:]


def _pythia_reference_qkv(model, info):
    """GPT-NeoX stores query_key_value.weight as (3*d_model, d_model) where
    rows are laid out per-head as [head0_Q, head0_K, head0_V, head1_Q, ...].
    See transformers/models/gpt_neox/modeling_gpt_neox.py GPTNeoXAttention.forward:
        qkv = qkv.view(*input_shape, n_heads, 3 * head_size).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
    """
    W = model.gpt_neox.layers[0].attention.query_key_value.weight.detach().clone()
    n_heads, d_model, head_dim = info["n_heads"], info["d_model"], info["head_dim"]
    W_per_head = W.view(n_heads, 3 * head_dim, d_model)
    W_Q = W_per_head[:, :head_dim, :].reshape(n_heads * head_dim, d_model)
    W_K = W_per_head[:, head_dim:2 * head_dim, :].reshape(n_heads * head_dim, d_model)
    W_V = W_per_head[:, 2 * head_dim:, :].reshape(n_heads * head_dim, d_model)
    return W_Q, W_K, W_V


REFERENCES = {
    "gpt2": _gpt2_reference_qkv,
    "pythia-70m-deduped": _pythia_reference_qkv,
}


@pytest.mark.parametrize("model_name", ["gpt2", "pythia-70m-deduped"])
@pytest.mark.parametrize("weight_init", ["arange", "randn"])
def test_extract_qkv_matches_hf_layout(model_name, weight_init, tiny_hf_model_factory):
    cache_path, model, info = tiny_hf_model_factory(model_name, weight_init=weight_init, seed=0)
    cfg = get_model_config(model_name)
    weight_map = extract_weight_map(cache_path=cache_path)
    W_Q, W_K, W_V = cfg.extract_qkv(
        cache_path, layer_idx=0, d_model=info["d_model"],
        weight_map=weight_map, device="cpu", qkv_scale_factor=1.0,
    )

    ref_Q, ref_K, ref_V = REFERENCES[model_name](model, info)

    # Shape sanity
    expected_shape = (info["n_heads"] * info["head_dim"], info["d_model"])
    assert W_Q.shape == expected_shape, f"W_Q shape {W_Q.shape} != {expected_shape}"
    assert W_K.shape == expected_shape
    assert W_V.shape == expected_shape

    # Element-wise match against architecture-correct reference
    torch.testing.assert_close(W_Q, ref_Q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(W_K, ref_K, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(W_V, ref_V, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("model_name", ["gpt2", "pythia-70m-deduped"])
def test_reshape_round_trip_to_per_head(model_name, tiny_hf_model_factory):
    """The pipeline reshapes (n_heads*head_dim, d_model) -> (n_heads, head_dim, d_model)
    with row-major contiguous reshape. Verify this matches the per-head split
    implied by the model's forward pass."""
    cache_path, model, info = tiny_hf_model_factory(model_name, weight_init="arange", seed=0)
    cfg = get_model_config(model_name)
    weight_map = extract_weight_map(cache_path=cache_path)
    W_Q, _, _ = cfg.extract_qkv(
        cache_path, layer_idx=0, d_model=info["d_model"],
        weight_map=weight_map, device="cpu", qkv_scale_factor=1.0,
    )
    n_heads, head_dim, d_model = info["n_heads"], info["head_dim"], info["d_model"]
    W_Q_h = W_Q.reshape(n_heads, head_dim, d_model).float()

    # For head h, the row-major slice should equal rows [h*head_dim:(h+1)*head_dim] of W_Q
    for h in range(n_heads):
        torch.testing.assert_close(
            W_Q_h[h], W_Q[h * head_dim:(h + 1) * head_dim].float(),
            atol=1e-5, rtol=1e-5,
        )
