import numpy as np
import pandas as pd
import torch


class HeadAnalyzer:
    def __init__(self, config):
        # if config is None:
        #     config = config_default
        self.config = config  # for passing around
        # some unpacking
        self.stats_functions = dict(config.stats)
        self.w_bins = config.w_bins
        self.sv_bins = config.sv_bins
        self.use_density = config.use_density

        # initialize data
        self.data = {
            weight_type: {"weight_type": weight_type}
            for weight_type in config.weight_type
        }

    def analyze_head(self, head):
        W_Q_h, W_K_h, W_QK_h = head["W_Q"], head["W_K"], head["W_QK"]
        self.fill_WW(W_Q_h, W_K_h, W_QK_h)
        # add BB and BW

        # placeholder for OV
        # W_O_h, W_V_h = head['W_O_h'], head['W_V_h']
        # self.fill_WW(W_O_h, Q_V_h)

    # tensors are for a given head, thus are matrices
    def fill_WW(self, W_Q_h, W_K_h, W_QK, ov=False):
        W_Q_key, W_K_key, W_QK_key = "W_Q", "W_K", "W_QK"
        if ov:
            W_Q_key, W_K_key, W_QK_key = "W_O", "W_V", "W_OV"

        W_Q_vector = W_Q_h.flatten().detach().cpu().numpy()
        self.fill_vector(W_Q_key, W_Q_vector)

        W_K_vector = W_K_h.flatten().detach().cpu().numpy()
        self.fill_vector(W_K_key, W_K_vector)

        self.fill_matrix(W_QK_key, W_QK)

    def fill_stats(self, weight_name, x_arr):
        self.data[weight_name].update(
            {k: f(x_arr) for k, f in self.stats_functions.items()}
        )

    def fill_scalar(self, weight_name, v):
        self.data[weight_name].update({weight_name: v})

    def fill_vector(self, weight_name, x_arr, histo=True, copy=False):
        if histo:
            h, _ = np.histogram(x_arr, bins=self.w_bins, density=self.use_density)
            self.data[weight_name].update({"P_w": h})
        if copy:
            self.data[weight_name].update({"x": x_arr.to_numpy()})
        self.fill_stats(weight_name, x_arr)

    def fill_matrix(self, weight_name, W_tensor):
        x_arr = W_tensor.flatten().detach().cpu().numpy()
        self.fill_vector(weight_name, x_arr, histo=True, copy=False)
        _, S, _ = torch.linalg.svd(W_tensor)
        svd = S.detach().cpu().numpy()
        self.data[weight_name].update({"SVD": svd})
        P_sv, _ = np.histogram(svd, bins=self.sv_bins, density=self.use_density)
        self.data[weight_name].update({"P_sv": P_sv})

    def to_pandas(self):
        df = pd.DataFrame([v for v in self.data.values()])
        return df


class LayerHeadContainer:
    def __init__(self, layer_idx, config):
        self.layer_idx = layer_idx
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.data = [HeadAnalyzer(config) for _ in range(self.n_heads)]

    def analyze_layer(self, input_dict):
        # expected shape for W is n_heads, d_head, d_model

        W_Q_h = input_dict["W_Q"]
        W_K_h = input_dict["W_K"]
        # W_QK = W_Q^T @ W_K: (d_model, d_head) @ (d_head, d_model) = (d_model, d_model)
        W_QK_h = W_Q_h.transpose(1, 2) @ W_K_h

        for head_idx in range(self.n_heads):
            head_data = {
                "W_Q": W_Q_h[head_idx],
                "W_K": W_K_h[head_idx],
                "W_QK": W_QK_h[head_idx],
            }
            self.data[head_idx].analyze_head(head_data)

    def post_process(self, metrics=None):
        if metrics is None:
            from transformer_analysis.histogram_utils import normality_metrics

            metrics = normality_metrics

        for head in self.data:  # loop on heads
            centers = (head.w_bins[:-1] + head.w_bins[1:]) / 2
            for h in head.data.values():  # loop on weights associated with head
                # h is a dictionary
                for f_m in metrics.values():
                    f_m(h, centers)

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
    config.weight_type = ["W_Q", "W_K", "W_QK"]
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
