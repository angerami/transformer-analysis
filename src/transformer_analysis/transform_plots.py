"""Summary plots logged as MLflow artifacts in the transform stage."""

import json
import logging
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from datasets import load_from_disk
from matplotlib.colors import LogNorm


def _load_refined(refined_path):
    df = load_from_disk(refined_path).to_pandas()
    with open(os.path.join(refined_path, "metadata.json")) as f:
        metadata = json.load(f)
    return df, metadata


def _plot_p_w_single(df, w_bins, model_name):
    centers = 0.5 * (w_bins[:-1] + w_bins[1:])
    width = w_bins[1] - w_bins[0]
    row = df.query("layer == 0 and head == 0").iloc[0]
    p_w = np.asarray(row["P_w"], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(centers, p_w, width=width, align="center")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Probability")
    ax.set_title(f"{model_name} L0 H0 — P(W) [W_QK]")
    fig.tight_layout()
    return fig


def _plot_p_w_stacked(df_sorted, w_bins, n_layers, n_heads, model_name):
    Z = np.array([np.asarray(row["P_w"], dtype=float)
                  for _, row in df_sorted.iterrows()]).T
    floor = max(Z[Z > 0].min() * 0.5, 1e-12) if np.any(Z > 0) else 1e-12
    Z_plot = np.maximum(Z, floor)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        Z_plot,
        aspect="auto",
        origin="lower",
        norm=LogNorm(vmin=floor, vmax=Z_plot.max()),
        extent=[0, n_layers * n_heads, w_bins[0], w_bins[-1]],
        cmap="viridis",
    )
    ax.set_xticks([i * n_heads for i in range(n_layers)])
    ax.set_xticklabels([str(i) for i in range(n_layers)])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Weight")
    ax.set_title(f"{model_name} — P(W) stacked across heads [W_QK]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(W) (log)")
    fig.tight_layout()
    return fig


def _plot_std_heatmap(df_sorted, n_layers, n_heads, model_name):
    z = df_sorted.pivot(index="layer", columns="head", values="std").values.astype(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(z, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"{model_name} — std(W_QK) per head")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("std")
    fig.tight_layout()
    return fig


def log_transform_plots(refined_path, model_name=None):
    """Build and log summary plots for the refined dataset to the active MLflow run."""
    if not os.path.isdir(refined_path):
        logging.warning("Refined dataset not found at %s; skipping transform plots.", refined_path)
        return
    df, metadata = _load_refined(refined_path)
    df = df.query("weight_type == 'W_QK'")
    if df.empty:
        logging.warning("No W_QK rows found in %s; skipping transform plots.", refined_path)
        return

    if model_name is None:
        model_name = df["model"].iloc[0] if "model" in df.columns else ""

    if "merged" in metadata and model_name in metadata["merged"]:
        meta_for_model = metadata["merged"][model_name]
    else:
        meta_for_model = metadata
    w_bins = np.array(meta_for_model["w_bins"])

    df_sorted = df.sort_values(["layer", "head"]).reset_index(drop=True)
    n_layers = int(df_sorted["layer"].max()) + 1
    n_heads = int(df_sorted["head"].max()) + 1

    figs = {
        "plots/p_w_layer0_head0.png": _plot_p_w_single(df_sorted, w_bins, model_name),
        "plots/p_w_stacked.png": _plot_p_w_stacked(df_sorted, w_bins, n_layers, n_heads, model_name),
        "plots/std_heatmap.png": _plot_std_heatmap(df_sorted, n_layers, n_heads, model_name),
    }
    for path, fig in figs.items():
        mlflow.log_figure(fig, path)
        plt.close(fig)
