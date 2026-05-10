import sys
import numpy as np
import torch
import pytest
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transformer_analysis.attn_head_analysis import LayerHeadContainer


@pytest.fixture
def config():
    cfg = SimpleNamespace()
    cfg.weight_type = ["W_Q", "W_K", "W_QK", "W_Q_gram", "W_K_gram", "QK_alignment"]
    cfg.stats = {"mean": np.mean, "std": np.std}
    cfg.w_bins = np.linspace(-2, 2, 201)
    cfg.sv_bins = np.linspace(0, 10, 50)
    cfg.use_density = False
    cfg.n_heads = 4
    cfg.d_model = 64
    cfg.head_dim = 16
    cfg.low_rank_svd_approximation = False
    cfg.top_k_svd = -1
    return cfg


@pytest.fixture
def layer_output(config):
    torch.manual_seed(42)
    n_heads, d_head, d_model = config.n_heads, config.head_dim, config.d_model
    W_Q = torch.randn(n_heads, d_head, d_model)
    W_K = torch.randn(n_heads, d_head, d_model)
    lhc = LayerHeadContainer(0, config, device="cpu")
    lhc.analyze_layer({"W_Q": W_Q, "W_K": W_K})
    lhc.post_process()
    df = lhc.to_pandas()
    return df, W_Q, W_K


def test_gram_columns_present(layer_output):
    df, _, _ = layer_output
    wt = set(df["weight_type"].unique())
    assert "W_Q_gram" in wt
    assert "W_K_gram" in wt
    assert "QK_alignment" in wt


def test_gram_svd_matches_matrix_svd(layer_output, config):
    df, W_Q, W_K = layer_output
    tol = 1e-4
    for head_idx in range(config.n_heads):
        # Reference: true singular values of W_Q[head_idx]
        true_svd_q = torch.linalg.svdvals(W_Q[head_idx]).numpy()
        # Stored: sqrt of gram eigenvalues
        row = df[(df["weight_type"] == "W_Q_gram") & (df["head"] == head_idx)].iloc[0]
        stored_svd_q = np.sort(row["SVD"])[::-1]
        np.testing.assert_allclose(np.sort(stored_svd_q)[::-1], np.sort(true_svd_q)[::-1], atol=tol,
                                   err_msg=f"W_Q_gram SVD mismatch for head {head_idx}")

        true_svd_k = torch.linalg.svdvals(W_K[head_idx]).numpy()
        row_k = df[(df["weight_type"] == "W_K_gram") & (df["head"] == head_idx)].iloc[0]
        stored_svd_k = row_k["SVD"]
        np.testing.assert_allclose(np.sort(stored_svd_k)[::-1], np.sort(true_svd_k)[::-1], atol=tol,
                                   err_msg=f"W_K_gram SVD mismatch for head {head_idx}")


def test_alignment_cosines_in_unit_interval(layer_output, config):
    df, _, _ = layer_output
    for head_idx in range(config.n_heads):
        row = df[(df["weight_type"] == "QK_alignment") & (df["head"] == head_idx)].iloc[0]
        cosines = row["SVD"]
        assert cosines is not None
        assert np.all(cosines >= -1e-6), f"Cosine below 0 for head {head_idx}"
        assert np.all(cosines <= 1 + 1e-6), f"Cosine above 1 for head {head_idx}"


def test_gram_svd_shape(layer_output, config):
    df, _, _ = layer_output
    d_head = config.head_dim
    for head_idx in range(config.n_heads):
        row_q = df[(df["weight_type"] == "W_Q_gram") & (df["head"] == head_idx)].iloc[0]
        assert len(row_q["SVD"]) == d_head, f"W_Q_gram SVD wrong length for head {head_idx}"
        row_a = df[(df["weight_type"] == "QK_alignment") & (df["head"] == head_idx)].iloc[0]
        assert len(row_a["SVD"]) == d_head, f"QK_alignment cosines wrong length for head {head_idx}"


def test_backward_compat_without_new_types(config):
    config.weight_type = ["W_Q", "W_K", "W_QK"]
    torch.manual_seed(0)
    W_Q = torch.randn(config.n_heads, config.head_dim, config.d_model)
    W_K = torch.randn(config.n_heads, config.head_dim, config.d_model)
    lhc = LayerHeadContainer(0, config, device="cpu")
    lhc.analyze_layer({"W_Q": W_Q, "W_K": W_K})
    df = lhc.to_pandas()
    wt = set(df["weight_type"].unique())
    assert "W_QK" in wt
    assert "W_Q_gram" not in wt
    assert "QK_alignment" not in wt
