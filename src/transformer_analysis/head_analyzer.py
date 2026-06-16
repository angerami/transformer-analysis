import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformer_analysis.head_metrics import element_stats


def _svd(W):
    """Thin SVD with numpy fallback for ill-conditioned matrices.

    torch.linalg.svd uses LAPACK gesdd (divide-and-conquer) which occasionally
    fails to converge on ill-conditioned weight matrices (e.g. OLMo 2).
    numpy.linalg.svd defaults to gesvd which is slower but always converges.
    """
    try:
        return torch.linalg.svd(W, full_matrices=False)
    except torch._C._LinAlgError:
        U, S, Vh = np.linalg.svd(W.float().cpu().numpy(), full_matrices=False)
        dev, dt = W.device, W.dtype
        return (torch.from_numpy(U).to(dev, dt),
                torch.from_numpy(S).to(dev, dt),
                torch.from_numpy(Vh).to(dev, dt))


def _svdvals(W):
    """Singular values only. Delegates to _svd so we inherit the numpy fallback
    and so we avoid torch.linalg.svdvals, which is not implemented on MPS."""
    return _svd(W)[1]


def _us_from_gram(W):
    """Left singular vectors U and singular values S of W (…, d_head, d_model),
    via eigendecomposition of the small Gram matrix W Wᵀ (d_head × d_head).

    W Wᵀ = U diag(S²) Uᵀ, so U,S come from a d_head×d_head eigh instead of a
    wide d_head×d_model SVD. The nonzero eigenvalues of W Wᵀ are exactly the
    squared singular values, and since d_head ≤ d_model none are lost. Returns S
    descending to match torch.linalg.svd. Vh is not produced — callers that need
    the right singular vectors must use _svd. Avoids the MPS→CPU svd fallback."""
    dev, dt = W.device, W.dtype
    # Form and diagonalize the Gram in float64: WWᵀ squares the dynamic range, so
    # float32 would lose precision on small singular values. eigh is not on MPS and
    # the matrix is only d_head×d_head, so run it on CPU — still far cheaper than
    # the wide SVD it replaces.
    Wd = W.cpu().double()
    G = torch.bmm(Wd, Wd.transpose(-1, -2))
    evals, U = torch.linalg.eigh(G)             # ascending
    S = evals.flip(-1).clamp(min=0).sqrt()
    U = U.flip(-1)
    return U.to(dev, dt), S.to(dev, dt)


class HeadAnalyzer:
    def __init__(self, config, low_rank_svd_approximation=False, top_k_svd=-1, device="cpu"):
        self.config = config
        self.stats_functions = dict(config.stats)
        self.w_bins = config.w_bins
        self.sv_bins = config.sv_bins
        self.use_density = config.use_density
        self.device = torch.device(device)

        # SVD configuration
        self.low_rank_svd_approximation = low_rank_svd_approximation
        self.top_k_svd = top_k_svd

        # initialize data
        self.data = {
            weight_type: {"weight_type": weight_type}
            for weight_type in config.weight_type
        }

    def analyze_head(self, head):
        W_Q_h, W_K_h, W_QK_h = head["W_Q"], head["W_K"], head["W_QK"]
        S_wqk = head.get("S_WQK")
        self.fill_WW(W_Q_h, W_K_h, W_QK_h, S_wqk=S_wqk)
        if "W_Q_gram" in self.data and "W_Q_gram" in head:
            self.fill_gram("W_Q_gram", head["W_Q_gram"])
        if "W_K_gram" in self.data and "W_K_gram" in head:
            self.fill_gram("W_K_gram", head["W_K_gram"])
        if "QK_alignment" in self.data and "QK_alignment" in head:
            self.fill_alignment("QK_alignment", head["QK_alignment"])

    # tensors are for a given head, thus are matrices
    def fill_WW(self, W_Q_h, W_K_h, W_QK, ov=False, S_wqk=None):
        W_Q_key, W_K_key, W_QK_key = "W_Q", "W_K", "W_QK"
        if ov:
            W_Q_key, W_K_key, W_QK_key = "W_O", "W_V", "W_OV"

        W_Q_vector = W_Q_h.flatten().detach().cpu().numpy()
        self.fill_vector(W_Q_key, W_Q_vector)

        W_K_vector = W_K_h.flatten().detach().cpu().numpy()
        self.fill_vector(W_K_key, W_K_vector)

        if S_wqk is not None:
            self.fill_matrix_with_svd(W_QK_key, W_QK, S_wqk)
        else:
            self.fill_matrix(W_QK_key, W_QK)

    def fill_stats(self, weight_name, x_arr):
        self.data[weight_name].update(
            {k: f(x_arr) for k, f in self.stats_functions.items()}
        )

    def fill_scalar(self, weight_name, v):
        self.data[weight_name].update({weight_name: v})

    def fill_vector(self, weight_name, x_arr, histo=True, copy=False):
        self.data[weight_name].update(
            element_stats(x_arr, self.w_bins, density=self.use_density,
                          stat_keys=self.stats_functions.keys(), histo=histo)
        )
        if copy:
            self.data[weight_name].update({"x": x_arr.to_numpy()})

    def fill_gram(self, weight_name, W_gram_tensor):
        x_arr = W_gram_tensor.flatten().detach().cpu().numpy()
        self.fill_vector(weight_name, x_arr, histo=True, copy=False)
        try:
            W_gpu = W_gram_tensor.to(self.device)
            # gram eigenvalues = σᵢ(W)²; take sqrt to store σᵢ(W)
            sv2 = _svdvals(W_gpu)
            svd = torch.sqrt(sv2.clamp(min=0)).detach().cpu().numpy()
            self.data[weight_name].update({"SVD": svd})
            P_sv, _ = np.histogram(svd, bins=self.sv_bins, density=self.use_density)
            self.data[weight_name].update({"P_sv": P_sv})
        except Exception as e:
            print(f"Warning: SVD computation failed for {weight_name}: {e}")
            self.data[weight_name].update({"SVD": None, "P_sv": None})

    def fill_alignment(self, weight_name, cosines_arr):
        # cosines_arr: numpy (d_head,), principal-angle cosines between W_Q and W_K col-spaces
        self.fill_stats(weight_name, cosines_arr)
        self.data[weight_name].update({"SVD": cosines_arr, "P_sv": None})

    def fill_matrix(self, weight_name, W_tensor):
        x_arr = W_tensor.flatten().detach().cpu().numpy()
        self.fill_vector(weight_name, x_arr, histo=True, copy=False)
        try:
            W_gpu = W_tensor.to(self.device)
            if self.low_rank_svd_approximation:
                q = min(self.top_k_svd + max(self.top_k_svd // 2, 10), min(W_gpu.shape))
                _, S, _ = torch.svd_lowrank(W_gpu, q=q, niter=4)
                S = S[:self.top_k_svd]
                d = W_gpu.shape[0]
                if len(S) < d:
                    S_padded = torch.zeros(d, dtype=S.dtype, device=S.device)
                    S_padded[:len(S)] = S
                    S = S_padded
            else:
                _, S, _ = torch.linalg.svd(W_gpu)

            svd = S.detach().cpu().numpy()
            self.data[weight_name].update({"SVD": svd})
            P_sv, _ = np.histogram(svd, bins=self.sv_bins, density=self.use_density)
            self.data[weight_name].update({"P_sv": P_sv})
        except (RuntimeError, Exception) as e:
            print(f"Warning: SVD computation failed for {weight_name}: {e}")
            self.data[weight_name].update({"SVD": None, "P_sv": None})

    def fill_matrix_with_svd(self, weight_name, W_tensor, S_precomputed):
        x_arr = W_tensor.flatten().detach().cpu().numpy()
        self.fill_vector(weight_name, x_arr, histo=True, copy=False)
        try:
            d = W_tensor.shape[0]
            S = S_precomputed
            if len(S) < d:
                S_padded = torch.zeros(d, dtype=S.dtype, device=S.device)
                S_padded[:len(S)] = S
                S = S_padded
            svd = S.detach().cpu().numpy()
            self.data[weight_name].update({"SVD": svd})
            P_sv, _ = np.histogram(svd, bins=self.sv_bins, density=self.use_density)
            self.data[weight_name].update({"P_sv": P_sv})
        except Exception as e:
            print(f"Warning: SVD computation failed for {weight_name}: {e}")
            self.data[weight_name].update({"SVD": None, "P_sv": None})

    def to_pandas(self):
        df = pd.DataFrame([v for v in self.data.values()])
        return df


class LayerHeadContainer:
    def __init__(self, layer_idx, config, low_rank_svd_approximation=False, top_k_svd=-1,
                 svd_via_gram=False, device="cpu"):
        self.layer_idx = layer_idx
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.device = device
        self.svd_via_gram = svd_via_gram

        # SVD configuration
        self.low_rank_svd_approximation = low_rank_svd_approximation
        if low_rank_svd_approximation and top_k_svd == -1:
            self.top_k_svd = self.head_dim
        else:
            self.top_k_svd = top_k_svd

        # Create HeadAnalyzer instances with SVD configuration
        self.data = [
            HeadAnalyzer(config, low_rank_svd_approximation=self.low_rank_svd_approximation,
                        top_k_svd=self.top_k_svd, device=device)
            for _ in range(self.n_heads)
        ]

    def analyze_layer(self, input_dict):
        # expected shape for W is n_heads, d_head, d_model
        weight_types = set(self.config.weight_type)

        W_Q_h = input_dict["W_Q"]
        W_K_h = input_dict["W_K"]
        # W_QK = W_Q^T @ W_K is (d_model, d_model) per head — the bilinear form
        # x^T W_QK x' for attention scores. It is only used for the per-head
        # element histogram (computed on CPU), so we build it one head at a time
        # inside the loop below rather than materializing the full
        # (n_heads, d_model, d_model) tensor — a ~32× smaller peak footprint that
        # avoids swap on memory-constrained machines.

        compute_grams = "W_Q_gram" in weight_types or "W_K_gram" in weight_types
        compute_alignment = "QK_alignment" in weight_types
        compute_factored_wqk = "W_QK" in weight_types

        if compute_grams:
            W_Q_gram_all = torch.bmm(W_Q_h, W_Q_h.transpose(1, 2)).to(self.device)
            W_K_gram_all = torch.bmm(W_K_h, W_K_h.transpose(1, 2)).to(self.device)

        # Thin SVD of W_Q and W_K is shared between alignment and factored W_QK SVD.
        # SVs of W_QK = W_Q^T W_K equal SVs of diag(S_Q) @ U_Q^T @ U_K @ diag(S_K),
        # a d_head×d_head matrix — exact and cheap vs full d_model×d_model SVD.
        if compute_alignment or compute_factored_wqk:
            # The factored W_QK SVD needs only U,S of W_Q/W_K. When alignment is
            # not requested (which needs Vh), get U,S from the cheap d_head×d_head
            # Gram eigendecomposition instead of a wide MPS→CPU-fallback SVD.
            if self.svd_via_gram and not compute_alignment:
                U_Q, S_Q = _us_from_gram(W_Q_h.to(self.device))
                U_K, S_K = _us_from_gram(W_K_h.to(self.device))
            else:
                U_Q, S_Q, Vh_Q = _svd(W_Q_h.to(self.device))
                U_K, S_K, Vh_K = _svd(W_K_h.to(self.device))

            if compute_alignment:
                M_align = torch.bmm(Vh_Q, Vh_K.transpose(1, 2))  # (n_heads, d_head, d_head)
                cosines_all = _svdvals(M_align).clamp(0, 1).detach().cpu().numpy()

            if compute_factored_wqk:
                # M = diag(S_Q) @ U_Q^T @ U_K @ diag(S_K)
                M_wqk = S_Q.unsqueeze(2) * torch.bmm(U_Q.transpose(1, 2), U_K) * S_K.unsqueeze(1)
                S_WQK_all = _svdvals(M_wqk)  # (n_heads, d_head)

        for head_idx in tqdm(range(self.n_heads), desc=f"  Layer {self.layer_idx} heads", leave=False):
            head_data = {
                "W_Q": W_Q_h[head_idx],
                "W_K": W_K_h[head_idx],
                # (d_model, d_model) for this head only
                "W_QK": W_Q_h[head_idx].transpose(0, 1) @ W_K_h[head_idx],
            }
            if compute_grams:
                head_data["W_Q_gram"] = W_Q_gram_all[head_idx]
                head_data["W_K_gram"] = W_K_gram_all[head_idx]
            if compute_alignment:
                head_data["QK_alignment"] = cosines_all[head_idx]
            if compute_factored_wqk:
                head_data["S_WQK"] = S_WQK_all[head_idx]
            self.data[head_idx].analyze_head(head_data)

    def post_process(self, weight_metrics=None, sv_metrics=None):
        """
        Post-process analysis by computing additional metrics.

        Args:
            weight_metrics: Dictionary of metric functions for weight histograms.
                           Each function should have signature f(h, centers).
                           If None, uses normality_metrics.
            sv_metrics: Dictionary of metric functions for singular values.
                       Each function should have signature f(h, svd_array).
                       If None, uses singular_value_metrics.
        """
        if weight_metrics is None:
            from transformer_analysis.head_metrics import normality_metrics
            weight_metrics = normality_metrics

        if sv_metrics is None:
            from transformer_analysis.head_metrics import singular_value_metrics
            sv_metrics = singular_value_metrics

        for head in self.data:  # loop on heads
            centers = (head.w_bins[:-1] + head.w_bins[1:]) / 2
            for h in head.data.values():  # loop on weights associated with head
                if h.get("P_w") is not None:
                    for f_m in weight_metrics.values():
                        f_m(h, centers)

                if "SVD" in h and h["SVD"] is not None:
                    svd_array = h["SVD"]
                    for f_m in sv_metrics.values():
                        f_m(h, svd_array)

    def to_pandas(self):
        df_list = []
        for head_idx in range(self.n_heads):
            head_df = self.data[head_idx].to_pandas()
            head_df["head"] = head_idx
            df_list.append(head_df)

        df = pd.concat(df_list, ignore_index=True)
        df["layer"] = self.layer_idx
        return df


if __name__ == "__main__":
    from types import SimpleNamespace

    import numpy as np
    import torch

    test_single_head = False
    test_layer = True

    config = SimpleNamespace()
    config.weight_type = ["W_Q", "W_K", "W_QK", "W_Q_gram", "W_K_gram", "QK_alignment"]
    config.stats = {"mean": np.mean, "std": np.std}
    config.w_bins = np.linspace(
        -2, 2, 201
    )  # low number of bins for easy visual inspection
    config.use_density = False
    config.n_heads = 32
    config.d_model = 1024
    config.head_dim = 12

    # testing a single head
    if test_single_head:
        print("Testing single head functionality")
        W_Q = torch.randn(config.head_dim, config.d_model)
        W_K = torch.randn(config.head_dim, config.d_model)
        W_QK = torch.randn(config.d_model, config.d_model)

        ha = HeadAnalyzer(config)
        head_data = {"W_Q": W_Q, "W_K": W_K, "W_QK": W_QK}
        ha.analyze_head(head_data)
        df = ha.to_pandas()
        print("Mean and variance should be consistent with N(0,1)")
        print(df[["mean", "std"]])

    if test_layer:
        layer_idx = 37  # random choice
        layer = LayerHeadContainer(layer_idx, config)
        W_Q = torch.randn(config.n_heads, config.head_dim, config.d_model)
        W_K = torch.randn(config.n_heads, config.head_dim, config.d_model)

        layer_input = {"W_Q": W_Q, "W_K": W_K}
        layer.analyze_layer(layer_input)
        df = layer.to_pandas()

        print(df.columns)
        print(df[["layer", "head", "weight_type", "std", "P_w"]])

        import matplotlib.pyplot as plt

        plt.plot(df["SVD"][2])
        # plt.show()
        plt.savefig("test.png", dpi=150, bbox_inches="tight")
        plt.close()


# %%
