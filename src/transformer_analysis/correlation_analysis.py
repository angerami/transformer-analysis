"""Top-level correlation analysis: extract W_QK per head, build HeadStore,
compute Q_{hh'}, and write results.

This module is designed to run standalone or be called from the main
weight_analysis pipeline.  It reuses the model_registry infrastructure
for model loading and weight extraction.

Usage (standalone):
    python -m transformer_analysis.correlation_analysis --model gpt2 --out corr_out

Usage (from code):
    from transformer_analysis.correlation_analysis import run_correlation_analysis
    results = run_correlation_analysis(model_name="gpt2", ...)
"""

import json
import os
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoConfig
from datasets import Dataset

from transformer_analysis.model_registry import get_model_config, extract_weight_map
from transformer_analysis.device_utils import get_device
from transformer_analysis.head_correlations import (
    HeadStore,
    compute_correlation_matrices,
    correlation_summary,
    layer_block_means,
    correlation_to_dataframe,
)


def extract_head_store(
    model_name,
    weight_type="W_QK",
    revision=None,
    cache_dir="./model_data",
    device=None,
    resume_download=True,
    max_workers=4,
):
    """Download model and extract flattened W per head into a HeadStore.

    Args:
        model_name: registered model name (e.g. "gpt2", "pythia-70m-deduped")
        weight_type: "W_QK" (default), "W_Q", or "W_K"
        revision: checkpoint revision string (Pythia), or None for main
        cache_dir: download cache location
        device: torch device for intermediate computation
        resume_download, max_workers: passed to snapshot_download

    Returns:
        (store, config) where store is a populated HeadStore and config
        holds model dimensions.
    """
    device_str = str(get_device(device))
    model_config = get_model_config(model_name)
    revision_string = revision if revision else "main"

    cache_path = snapshot_download(
        repo_id=model_config.repo_id,
        revision=revision,
        cache_dir=f"{cache_dir}/{model_name}/{revision_string}",
        allow_patterns=model_config.allow_patterns,
        resume_download=resume_download,
        max_workers=max_workers,
    )
    hf_config = AutoConfig.from_pretrained(cache_path)

    n_heads = model_config.get_config_value(hf_config.__dict__, "n_heads")
    d_model = model_config.get_config_value(hf_config.__dict__, "d_model")
    n_layers = model_config.get_config_value(hf_config.__dict__, "n_layers")
    head_dim = d_model // n_heads

    weight_map = extract_weight_map(cache_path=cache_path)
    store = HeadStore()

    for layer_idx in tqdm(range(n_layers), desc="Extracting heads"):
        W_Q, W_K, _ = model_config.extract_qkv(
            cache_path, layer_idx, d_model, weight_map,
            device=device_str,
            qkv_scale_factor=model_config.qkv_scale_factor,
        )
        W_Q_h = W_Q.reshape(n_heads, head_dim, d_model).float()
        W_K_h = W_K.reshape(n_heads, head_dim, d_model).float()

        if weight_type == "W_QK":
            W_all = torch.bmm(W_Q_h, W_K_h.transpose(1, 2))  # (n_heads, head_dim, head_dim)
        elif weight_type == "W_Q":
            W_all = W_Q_h
        elif weight_type == "W_K":
            W_all = W_K_h
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

        for h_idx in range(n_heads):
            flat = W_all[h_idx].detach().cpu().numpy().flatten()
            store.add(layer_idx, h_idx, flat)

        del W_Q, W_K, W_Q_h, W_K_h, W_all

    cfg = SimpleNamespace(
        n_heads=n_heads, d_model=d_model, n_layers=n_layers,
        head_dim=head_dim, model_name=model_name, revision=revision,
        weight_type=weight_type,
    )
    return store, cfg


def run_correlation_analysis(
    model_name="gpt2",
    weight_type="W_QK",
    revision=None,
    metrics=("frob_cosine", "symmetric_kl", "jensen_shannon", "two_point", "connected_corr", "pearson_corr"),
    cache_dir="./model_data",
    out_dir="corr_out",
    device=None,
    kde_n_eval=2048,
    kde_bw="scott",
    save=True,
    resume_download=True,
    max_workers=4,
):
    """Full correlation analysis pipeline.

    Returns:
        dict with keys:
          - "Q": dict of metric_name -> np.ndarray (N x N)
          - "summary": dict of metric_name -> summary stats dict
          - "block_means": dict of metric_name -> (block_array, layer_ids)
          - "store": the HeadStore (for further analysis)
          - "config": model config namespace
    """
    logging.info(f"Starting correlation analysis: {model_name} ({weight_type})")

    store, cfg = extract_head_store(
        model_name=model_name,
        weight_type=weight_type,
        revision=revision,
        cache_dir=cache_dir,
        device=device,
        resume_download=resume_download,
        max_workers=max_workers,
    )
    logging.info(f"HeadStore: {store.n_heads} heads, {store.memory_mb():.1f} MB")

    Q_matrices = compute_correlation_matrices(
        store,
        metrics=metrics,
        kde_kwargs={"n_eval": kde_n_eval, "bw_method": kde_bw},
        show_progress=True,
    )

    summaries = {}
    blocks = {}
    for m in metrics:
        summaries[m] = correlation_summary(Q_matrices[m], store.keys)
        blocks[m] = layer_block_means(Q_matrices[m], store.keys)

    results = {
        "Q": Q_matrices,
        "summary": summaries,
        "block_means": blocks,
        "store": store,
        "config": cfg,
    }

    if save:
        _save_results(results, out_dir, cfg)

    return results


def _save_results(results, out_dir, cfg):
    """Persist correlation matrices, summaries, and long-form DataFrames."""
    os.makedirs(out_dir, exist_ok=True)
    rev = cfg.revision or "main"
    prefix = f"{cfg.model_name}_{rev}_{cfg.weight_type}"

    # save Q matrices as .npz
    Q_dict = {f"Q_{m}": Q for m, Q in results["Q"].items()}
    np.savez_compressed(f"{out_dir}/{prefix}_Q.npz", **Q_dict)

    # save summaries as JSON (strip non-serializable arrays)
    json_summaries = {}
    for m, s in results["summary"].items():
        js = {k: v for k, v in s.items()
              if not isinstance(v, np.ndarray)}
        json_summaries[m] = js
    with open(f"{out_dir}/{prefix}_summary.json", "w") as f:
        json.dump(json_summaries, f, indent=2)

    # save eigenvalues
    for m, s in results["summary"].items():
        np.save(f"{out_dir}/{prefix}_{m}_eigenvalues.npy", s["eigenvalues"])
        np.save(f"{out_dir}/{prefix}_{m}_P_Q.npy", s["P_Q_values"])

    # save block means
    for m, (block, layers) in results["block_means"].items():
        np.save(f"{out_dir}/{prefix}_{m}_block_means.npy", block)

    # save long-form dataframe (frob_cosine only, to avoid huge files)
    if "frob_cosine" in results["Q"]:
        df = correlation_to_dataframe(
            results["Q"]["frob_cosine"],
            results["store"].keys,
            "frob_cosine",
        )
        ds = Dataset.from_pandas(df)
        ds.save_to_disk(f"{out_dir}/{prefix}_pairs_dataset")

    # metadata
    meta = {
        "model_name": cfg.model_name,
        "revision": cfg.revision,
        "weight_type": cfg.weight_type,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "d_model": cfg.d_model,
        "head_dim": cfg.head_dim,
        "metrics": list(results["Q"].keys()),
        "head_index": results["store"].keys,
    }
    with open(f"{out_dir}/{prefix}_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logging.info(f"Results saved to {out_dir}/{prefix}_*")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    parser = argparse.ArgumentParser(description="Head-head correlation analysis")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--weight-type", type=str, default="W_QK",
                        choices=["W_QK", "W_Q", "W_K"])
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--out", type=str, default="corr_out")
    parser.add_argument("--cache", type=str, default="./model_data")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "mps", "cpu"])
    parser.add_argument("--metrics", nargs="+",
                        default=["frob_cosine", "symmetric_kl",
                                 "jensen_shannon", "two_point",
                                 "connected_corr", "pearson_corr"])
    parser.add_argument("--kde-n-eval", type=int, default=2048)
    parser.add_argument("--kde-bw", type=str, default="scott")
    parser.add_argument("--resume-download", action="store_true", default=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    run_correlation_analysis(
        model_name=args.model,
        weight_type=args.weight_type,
        revision=args.revision,
        metrics=tuple(args.metrics),
        cache_dir=args.cache,
        out_dir=args.out,
        device=args.device,
        kde_n_eval=args.kde_n_eval,
        kde_bw=args.kde_bw,
        resume_download=args.resume_download,
        max_workers=args.max_workers,
    )
