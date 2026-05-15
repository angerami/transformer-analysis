"""Verify the factored SVD path in head_analyzer.analyze_layer matches the full
d_model x d_model SVD of W_QK = W_Q^T @ W_K.

The factored path computes SVs of M = diag(S_Q) @ U_Q^T @ U_K @ diag(S_K), a
d_head x d_head matrix. This is mathematically equivalent to SVs of W_QK
restricted to its non-trivial subspace, but the equivalence was never asserted.
"""
import numpy as np
import torch
from transformer_analysis.head_analyzer import LayerHeadContainer


def _full_wqk_svs(W_Q, W_K):
    W_QK = torch.bmm(W_Q.transpose(1, 2), W_K)  # (n_heads, d_model, d_model)
    return torch.linalg.svdvals(W_QK)


def test_factored_svd_matches_full(tiny_config, tiny_weights):
    lhc = LayerHeadContainer(0, tiny_config, device="cpu")
    lhc.analyze_layer(tiny_weights)

    W_Q, W_K = tiny_weights["W_Q"], tiny_weights["W_K"]
    full_svs = _full_wqk_svs(W_Q, W_K).numpy()  # (n_heads, d_model)
    d_head = tiny_config.head_dim
    d_model = tiny_config.d_model

    for head_idx in range(tiny_config.n_heads):
        stored = lhc.data[head_idx].data["W_QK"]["SVD"]
        # Stored is padded to d_model with trailing zeros after the d_head leading SVs
        assert len(stored) == d_model
        stored_sorted = np.sort(stored)[::-1]
        full_sorted = np.sort(full_svs[head_idx])[::-1]

        # Top d_head SVs must match
        np.testing.assert_allclose(
            stored_sorted[:d_head], full_sorted[:d_head], atol=1e-4,
            err_msg=f"top-{d_head} SVs of W_QK disagree for head {head_idx}",
        )
        # Trailing entries from the factored path should be exactly zero (padding)
        np.testing.assert_allclose(stored_sorted[d_head:], 0.0, atol=1e-12)
        # And the true W_QK should also have ~zero SVs past rank d_head
        np.testing.assert_allclose(full_sorted[d_head:], 0.0, atol=1e-4)


def test_factored_svd_rank_bounded_by_d_head():
    """Sanity: rank(W_Q^T @ W_K) <= min(rank(W_Q), rank(W_K)) <= d_head."""
    torch.manual_seed(0)
    n_heads, d_head, d_model = 3, 4, 32
    W_Q = torch.randn(n_heads, d_head, d_model)
    W_K = torch.randn(n_heads, d_head, d_model)
    full_svs = _full_wqk_svs(W_Q, W_K)
    # Trailing d_model - d_head SVs should be ~0
    assert torch.allclose(full_svs[:, d_head:], torch.zeros_like(full_svs[:, d_head:]), atol=1e-4)
