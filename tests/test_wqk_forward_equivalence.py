"""Architecture-agnostic W_QK correctness test.

For each model, build a tiny HF model with random weights, run the pipeline
(extract_qkv -> reshape -> W_QK = W_Q^T @ W_K), then compare against the
ground-truth attention score that HF's own projection layer would compute on
the same inputs.

For random embedding vectors x_i, x_j and head h:
    s_ref[h]  = (W_Q^h x_i) · (W_K^h x_j)         # via HF projection layer
    s_ours[h] = x_i^T @ W_QK[h] @ x_j             # via our pipeline
These must agree to floating-point tolerance for every head.

Biases are zeroed and RoPE is skipped — W_QK is the pre-RoPE linear bilinear
form, which is what our analysis pipeline computes.
"""
import pytest
import torch
import torch.nn.functional as F

from transformer_analysis.model_registry import get_model_config, extract_weight_map


def _gpt2_reference_qk(model, x_i, x_j, info):
    """Compute per-head (q, k) projections for GPT-2 using the model's own
    c_attn weight. Bias zeroed."""
    W = model.h[0].attn.c_attn.weight  # (d_model, 3*d_model)
    d_model = info["d_model"]
    n_heads = info["n_heads"]
    head_dim = info["head_dim"]
    # GPT-2's c_attn is a Conv1D: y = x @ W + b, so projection of x is x @ W
    qkv_i = x_i @ W  # (3*d_model,)
    qkv_j = x_j @ W
    q_all_i = qkv_i[:d_model].view(n_heads, head_dim)
    k_all_j = qkv_j[d_model:2 * d_model].view(n_heads, head_dim)
    return q_all_i, k_all_j


def _pythia_reference_qk(model, x_i, x_j, info):
    """Compute per-head (q, k) for GPT-NeoX using its query_key_value layout.
    See modeling_gpt_neox.GPTNeoXAttention.forward:
        qkv = W x; reshape to (n_heads, 3*head_dim); chunk(3, dim=-1) -> Q, K, V
    """
    W = model.gpt_neox.layers[0].attention.query_key_value.weight  # (3*d_model, d_model)
    n_heads = info["n_heads"]
    head_dim = info["head_dim"]
    qkv_i = F.linear(x_i, W)  # (3*d_model,)
    qkv_j = F.linear(x_j, W)
    q_per_head_i = qkv_i.view(n_heads, 3 * head_dim)[:, :head_dim]
    k_per_head_j = qkv_j.view(n_heads, 3 * head_dim)[:, head_dim:2 * head_dim]
    return q_per_head_i, k_per_head_j


HF_QK_REFERENCES = {
    "gpt2": _gpt2_reference_qk,
    "pythia-70m-deduped": _pythia_reference_qk,
}


@pytest.mark.parametrize("model_name", ["gpt2", "pythia-70m-deduped"])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_wqk_matches_hf_attention_scores(model_name, seed, tiny_hf_model_factory):
    cache_path, model, info = tiny_hf_model_factory(model_name, weight_init="randn", seed=seed)

    # Extract via the pipeline
    cfg = get_model_config(model_name)
    weight_map = extract_weight_map(cache_path=cache_path)
    W_Q, W_K, _ = cfg.extract_qkv(
        cache_path, layer_idx=0, d_model=info["d_model"],
        weight_map=weight_map, device="cpu", qkv_scale_factor=1.0,
    )
    n_heads, head_dim, d_model = info["n_heads"], info["head_dim"], info["d_model"]
    W_Q_h = W_Q.reshape(n_heads, head_dim, d_model).float()
    W_K_h = W_K.reshape(n_heads, head_dim, d_model).float()

    # Our W_QK per head: (d_model, d_model) bilinear form
    W_QK_h = torch.bmm(W_Q_h.transpose(1, 2), W_K_h)  # (n_heads, d_model, d_model)

    # Several random (x_i, x_j) pairs
    g = torch.Generator().manual_seed(seed + 100)
    for _ in range(3):
        x_i = torch.randn(d_model, generator=g)
        x_j = torch.randn(d_model, generator=g)

        q_ref, k_ref = HF_QK_REFERENCES[model_name](model, x_i, x_j, info)
        # Ground-truth per-head scores
        s_ref = (q_ref * k_ref).sum(dim=-1)  # (n_heads,)

        # Pipeline-computed scores
        s_ours = torch.einsum("d,hde,e->h", x_i, W_QK_h, x_j)

        torch.testing.assert_close(s_ours, s_ref, atol=1e-4, rtol=1e-4,
            msg=f"{model_name}: attention score mismatch")
