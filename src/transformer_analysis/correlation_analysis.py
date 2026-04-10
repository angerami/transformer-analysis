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
import sys
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
    compute_cross_correlation_matrices,
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
    stores, cfg, _scalars = extract_head_stores(
        model_name=model_name,
        circuits=["QK"] if weight_type in ("W_QK", "W_Q", "W_K") else ["OV"],
        include_bias=False,
        revision=revision,
        cache_dir=cache_dir,
        device=device,
        resume_download=resume_download,
        max_workers=max_workers,
    )
    # Back-compat: return the single relevant store
    if weight_type in ("W_QK", "W_Q", "W_K"):
        store = stores["W_QK"]
    else:
        store = stores["W_OV"]
    cfg.weight_type = weight_type
    return store, cfg


def extract_head_stores(
    model_name,
    circuits=("QK",),
    include_bias=False,
    revision=None,
    cache_dir="./model_data",
    device=None,
    resume_download=True,
    max_workers=4,
):
    """Download model and extract multiple weight types into HeadStores.

    This is the general entry point for extracting any combination of
    (W, bias) × (QK, OV) per head.

    Args:
        model_name: registered model name
        circuits: tuple/list from {"QK", "OV"}
        include_bias: if True, also extract bias vectors for selected circuits
        revision: checkpoint revision (Pythia), or None
        cache_dir: download cache location
        device: torch device
        resume_download, max_workers: passed to snapshot_download

    Returns:
        (stores, config) where:
          stores: dict mapping weight-type names to HeadStore instances.
              Possible keys: "W_QK", "W_OV", "b_Q", "b_K", "b_V", "b_O"
          config: SimpleNamespace with model dimensions and extraction info.

    The stores dict only contains entries for the requested circuits.
    Bias stores are included only if include_bias=True and the model
    actually has biases (e.g. LLaMA has no attention biases).
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

    want_qk = "QK" in circuits
    want_ov = "OV" in circuits

    if want_ov and model_config.extract_o is None:
        raise ValueError(
            f"OV circuit requested but {model_name} has no extract_o registered"
        )

    # Initialize stores
    stores = {}
    if want_qk:
        stores["W_QK"] = HeadStore()
    if want_ov:
        stores["W_OV"] = HeadStore()

    bias_stores = {}  # filled only if include_bias
    has_biases = False

    # Per-head scalar observables (from magnetic-field-notes.md Section 4).
    # Accumulated as lists during extraction, converted to arrays after.
    scalars = {}  # name -> list of (layer, head, value)

    for layer_idx in tqdm(range(n_layers), desc="Extracting heads"):
        # ── QKV extraction (needed for both QK circuit and biases) ──
        W_Q, W_K, W_V = model_config.extract_qkv(
            cache_path, layer_idx, d_model, weight_map,
            device=device_str,
            qkv_scale_factor=model_config.qkv_scale_factor,
        )
        W_Q_h = W_Q.reshape(n_heads, head_dim, d_model).float()
        W_K_h = W_K.reshape(n_heads, head_dim, d_model).float()
        W_V_h = W_V.reshape(n_heads, head_dim, d_model).float()

        # ── QK circuit ──
        if want_qk:
            W_QK = torch.bmm(W_Q_h, W_K_h.transpose(1, 2))
            for h_idx in range(n_heads):
                flat = W_QK[h_idx].detach().cpu().numpy().flatten()
                stores["W_QK"].add(layer_idx, h_idx, flat)
            del W_QK

        # ── OV circuit ──
        if want_ov:
            W_O = model_config.extract_o(
                cache_path, layer_idx, d_model, weight_map, device=device_str,
            )
            # W_O is (d_model, d_model) full matrix;
            # per-head slice: W_O_h[h] = W_O[h*head_dim : (h+1)*head_dim, :]
            # W_OV_h = W_V_h @ W_O_h^T -> (n_heads, head_dim, head_dim)
            W_O_h = W_O.reshape(n_heads, head_dim, d_model).float()
            W_OV = torch.bmm(W_V_h, W_O_h.transpose(1, 2))
            for h_idx in range(n_heads):
                flat = W_OV[h_idx].detach().cpu().numpy().flatten()
                stores["W_OV"].add(layer_idx, h_idx, flat)
            del W_O, W_O_h, W_OV

        # ── Biases ──
        if include_bias and model_config.extract_biases is not None:
            layer_biases = model_config.extract_biases(
                cache_path, layer_idx, d_model, weight_map, device=device_str,
            )
            if layer_biases:
                has_biases = True
                for bname, bvec in layer_biases.items():
                    if bname not in bias_stores:
                        bias_stores[bname] = HeadStore()
                    # Per-head bias slice
                    bvec_np = bvec.detach().cpu().float().numpy()
                    if bname in ("b_Q", "b_K", "b_V"):
                        # shape: (d_model,) -> (n_heads, head_dim)
                        b_per_head = bvec_np.reshape(n_heads, head_dim)
                        for h_idx in range(n_heads):
                            bias_stores[bname].add(layer_idx, h_idx, b_per_head[h_idx])
                    elif bname == "b_O":
                        b_per_head = bvec_np.reshape(n_heads, head_dim)
                        for h_idx in range(n_heads):
                            bias_stores[bname].add(layer_idx, h_idx, b_per_head[h_idx])

                # ── Scalar observables from bias+weight combinations ──
                # These require the per-head W_Q_h, W_K_h and bias tensors
                # to be simultaneously available, which is only true here.
                #
                # From magnetic-field-notes.md:
                #   E_QK     = <b_Q | b_K>                (field overlap, scalar)
                #   |b_Q|    = ||b_Q||                    (bias norm)
                #   |b_K|    = ||b_K||                    (bias norm)
                #   h_QK     = W_Q^T b_K                  (query field vector)
                #   h_KQ     = W_K^T b_Q                  (key field vector)
                #   |h_QK|   = ||W_Q^T b_K||              (query field magnitude)
                #   |h_KQ|   = ||W_K^T b_Q||              (key field magnitude)
                #   h_KQ·h_QK = (W_K^T b_Q)^T (W_Q^T b_K)  (field overlap through QK)
                #   <b_Q|W_QK|b_K> = b_Q^T W_Q^T W_K b_K   (bias sandwich)

                b_Q_t = layer_biases.get("b_Q")
                b_K_t = layer_biases.get("b_K")

                if b_Q_t is not None and b_K_t is not None:
                    # Per-head bias slices as tensors (head_dim,)
                    b_Q_h = b_Q_t.reshape(n_heads, head_dim).float()
                    b_K_h = b_K_t.reshape(n_heads, head_dim).float()

                    for h_idx in range(n_heads):
                        bq = b_Q_h[h_idx]         # (head_dim,)
                        bk = b_K_h[h_idx]         # (head_dim,)
                        Wq = W_Q_h[h_idx]         # (head_dim, d_model)
                        Wk = W_K_h[h_idx]         # (head_dim, d_model)

                        # E_QK = <b_Q | b_K>
                        E_QK = torch.dot(bq, bk).item()

                        # Bias norms
                        norm_bQ = torch.linalg.norm(bq).item()
                        norm_bK = torch.linalg.norm(bk).item()

                        # h_QK = W_Q^T b_K  (d_model,)
                        # Per head: W_Q_h is (head_dim, d_model), b_K is (head_dim,)
                        # so h_QK = Wq^T @ bk = (d_model, head_dim) @ (head_dim,) = (d_model,)
                        h_QK = Wq.T @ bk          # (d_model,)
                        norm_h_QK = torch.linalg.norm(h_QK).item()

                        # h_KQ = W_K^T b_Q  (d_model,)
                        h_KQ = Wk.T @ bq          # (d_model,)
                        norm_h_KQ = torch.linalg.norm(h_KQ).item()

                        # h_KQ^T h_QK  (scalar: field overlap through QK operator)
                        h_KQ_dot_h_QK = torch.dot(h_KQ, h_QK).item()

                        # <b_Q | W_QK | b_K> = b_Q^T (W_Q^T W_K) b_K
                        # = (W_Q b_Q)^T (W_K b_K) ... but simpler:
                        # = h_KQ^T @ h_QK  ... wait, that's (Wk^T bq)^T (Wq^T bk)
                        # Actually <b_Q|W_QK|b_K> = bq^T Wq^T Wk bk
                        #   = (Wq bq)^T (Wk bk)
                        # No: Wq is (head_dim, d_model), bq is (head_dim,)
                        # W_QK = W_Q^T W_K, so <bQ|W_QK|bK> = bQ^T W_Q^T W_K bK
                        # = (W_Q bQ)^T (W_K bK)
                        # W_Q bQ: (head_dim, d_model) @ (head_dim,) -- dimension mismatch!
                        # Actually W_Q is (head_dim, d_model) so W_Q^T is (d_model, head_dim)
                        # W_QK = W_Q^T W_K is (d_model, d_model)
                        # <bQ|W_QK|bK> needs bQ in d_model space, but bQ is (head_dim,)
                        # The per-head W_QK is (head_dim, head_dim), and bQ, bK are (head_dim,)
                        # So the natural per-head observable is: bQ^T @ W_QK_h @ bK
                        W_QK_h = Wq @ Wk.T        # (head_dim, head_dim)
                        bQ_WQK_bK = (bq @ W_QK_h @ bk).item()

                        # Also: <bQ|W_QK^T W_QK|bQ> = ||W_QK bQ||^2  (bias "energy")
                        WQK_bq = W_QK_h @ bq      # (head_dim,)
                        bQ_WQKWQK_bQ = torch.dot(WQK_bq, WQK_bq).item()

                        # <bK|W_QK^T W_QK|bK>
                        WQK_bk = W_QK_h @ bk
                        bK_WQKWQK_bK = torch.dot(WQK_bk, WQK_bk).item()

                        key = (layer_idx, h_idx)
                        for name, val in [
                            ("E_QK", E_QK),
                            ("norm_bQ", norm_bQ),
                            ("norm_bK", norm_bK),
                            ("norm_h_QK", norm_h_QK),
                            ("norm_h_KQ", norm_h_KQ),
                            ("h_KQ_dot_h_QK", h_KQ_dot_h_QK),
                            ("bQ_WQK_bK", bQ_WQK_bK),
                            ("bQ_WQKWQK_bQ", bQ_WQKWQK_bQ),
                            ("bK_WQKWQK_bK", bK_WQKWQK_bK),
                        ]:
                            if name not in scalars:
                                scalars[name] = []
                            scalars[name].append((layer_idx, h_idx, val))

        del W_Q, W_K, W_V, W_Q_h, W_K_h, W_V_h

    # Merge bias stores
    if include_bias and has_biases:
        stores.update(bias_stores)

    # Convert scalar lists to a DataFrame
    scalars_df = None
    if scalars:
        import pandas as pd
        rows = []
        first_key = next(iter(scalars))
        for i, (layer, head, _) in enumerate(scalars[first_key]):
            row = {"layer": layer, "head": head}
            for name in scalars:
                row[name] = scalars[name][i][2]
            rows.append(row)
        scalars_df = pd.DataFrame(rows)

    cfg = SimpleNamespace(
        n_heads=n_heads, d_model=d_model, n_layers=n_layers,
        head_dim=head_dim, model_name=model_name, revision=revision,
        circuits=list(circuits),
        include_bias=include_bias,
        weight_types=list(stores.keys()),
        has_scalars=scalars_df is not None,
    )
    return stores, cfg, scalars_df


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


def run_multi_circuit_analysis(
    model_name="gpt2",
    circuits=("QK", "OV"),
    include_bias=False,
    self_correlations=True,
    cross_correlations=("QKOV",),
    metrics=("frob_cosine", "two_point", "connected_corr", "pearson_corr",
             "hist_symmetric_kl", "hist_jensen_shannon"),
    cross_metrics=("frob_cosine", "pearson_corr"),
    wb_cross_metrics=("hist_symmetric_kl", "hist_jensen_shannon"),
    revision=None,
    cache_dir="./model_data",
    out_dir="corr_out",
    device=None,
    kde_n_eval=2048,
    kde_bw="scott",
    save=True,
    resume_download=True,
    max_workers=4,
):
    """Multi-circuit correlation analysis pipeline.

    Extracts all requested weight types, computes self-correlations for each,
    and optionally computes cross-correlations between specified pairs.

    Args:
        model_name: registered model name
        circuits: which circuits to process, from {"QK", "OV"}
        include_bias: also extract bias vectors
        self_correlations: compute Q_{hh'} within each weight type
        cross_correlations: cross-correlation pairs to compute.
            Supported: "QKOV" (W_QK ↔ W_OV), "WB" (weight ↔ bias).
            Empty tuple to skip cross-correlations.
        metrics: metric names for self-correlations
        cross_metrics: metric names for QKOV cross-correlations
            (same-dimension: frob_cosine, pearson_corr work here)
        wb_cross_metrics: metric names for W↔b cross-correlations.
            Defaults to distribution-based metrics since W and b have
            different dimensionality (d_head² vs d_head).
        revision, cache_dir, out_dir, device: as before
        save: write results to disk

    Returns:
        dict with:
          "stores": dict of weight_type -> HeadStore
          "self_Q": dict of weight_type -> {metric -> Q matrix}
          "cross_Q": dict of "A_vs_B" -> {metric -> Q matrix}
          "summaries": nested dict
          "config": model config namespace
    """
    logging.info(f"Multi-circuit analysis: {model_name}, circuits={circuits}, "
                 f"bias={include_bias}, cross={cross_correlations}")

    stores, cfg, scalars_df = extract_head_stores(
        model_name=model_name,
        circuits=circuits,
        include_bias=include_bias,
        revision=revision,
        cache_dir=cache_dir,
        device=device,
        resume_download=resume_download,
        max_workers=max_workers,
    )
    if scalars_df is not None:
        logging.info(f"  Scalars: {len(scalars_df)} heads × "
                     f"{len([c for c in scalars_df.columns if c not in ('layer','head')])} observables")
    for wt, st in stores.items():
        logging.info(f"  {wt}: {st.n_heads} heads, {st.memory_mb():.1f} MB")

    kde_kwargs = {"n_eval": kde_n_eval, "bw_method": kde_bw}

    # ── Self-correlations ──
    self_Q = {}
    summaries = {}
    blocks = {}
    if self_correlations:
        # For weight matrices (W_QK, W_OV): full metric set
        # For biases (b_Q, b_K, ...): fast metrics only
        matrix_types = [wt for wt in stores if wt.startswith("W_")]
        bias_types = [wt for wt in stores if wt.startswith("b_")]
        fast_metrics = tuple(m for m in metrics
                             if m not in ("symmetric_kl", "jensen_shannon"))

        for wt in matrix_types:
            logging.info(f"Self-correlation: {wt} ({len(metrics)} metrics)")
            self_Q[wt] = compute_correlation_matrices(
                stores[wt], metrics=metrics,
                kde_kwargs=kde_kwargs, show_progress=True,
            )
            summaries[wt] = {}
            blocks[wt] = {}
            for m in metrics:
                summaries[wt][m] = correlation_summary(
                    self_Q[wt][m], stores[wt].keys)
                blocks[wt][m] = layer_block_means(
                    self_Q[wt][m], stores[wt].keys)

        for wt in bias_types:
            logging.info(f"Self-correlation: {wt} ({len(fast_metrics)} metrics)")
            self_Q[wt] = compute_correlation_matrices(
                stores[wt], metrics=fast_metrics,
                kde_kwargs=kde_kwargs, show_progress=True,
            )
            summaries[wt] = {}
            blocks[wt] = {}
            for m in fast_metrics:
                summaries[wt][m] = correlation_summary(
                    self_Q[wt][m], stores[wt].keys)
                blocks[wt][m] = layer_block_means(
                    self_Q[wt][m], stores[wt].keys)

    # ── Cross-correlations ──
    cross_Q = {}
    if cross_correlations:
        for cross_type in cross_correlations:
            if cross_type == "QKOV" and "W_QK" in stores and "W_OV" in stores:
                label = "W_QK_vs_W_OV"
                logging.info(f"Cross-correlation: {label}")
                cross_Q[label] = compute_cross_correlation_matrices(
                    stores["W_QK"], stores["W_OV"],
                    metrics=cross_metrics,
                    kde_kwargs=kde_kwargs,
                    show_progress=True,
                )
            elif cross_type == "WB":
                # Cross-correlate each weight with its bias counterpart.
                # W and b have different dimensions (d_head² vs d_head),
                # so use distribution-based metrics by default.
                wb_pairs = [
                    ("W_QK", "b_Q"), ("W_QK", "b_K"),
                    ("W_OV", "b_V"), ("W_OV", "b_O"),
                ]
                for wt, bt in wb_pairs:
                    if wt in stores and bt in stores:
                        label = f"{wt}_vs_{bt}"
                        logging.info(f"Cross-correlation: {label}")
                        cross_Q[label] = compute_cross_correlation_matrices(
                            stores[wt], stores[bt],
                            metrics=wb_cross_metrics,
                            kde_kwargs=kde_kwargs,
                            show_progress=True,
                        )

    results = {
        "stores": stores,
        "self_Q": self_Q,
        "cross_Q": cross_Q,
        "summaries": summaries,
        "block_means": blocks,
        "scalars": scalars_df,
        "config": cfg,
    }

    if save:
        _save_multi_results(results, out_dir, cfg)

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


def _save_multi_results(results, out_dir, cfg):
    """Persist multi-circuit analysis results.

    File naming: {model}_{rev}_{weight_type}_*.{ext}
    Cross-correlations: {model}_{rev}_{label}_*.{ext}
    """
    os.makedirs(out_dir, exist_ok=True)
    rev = cfg.revision or "main"

    # ── Self-correlations (per weight type) ──
    for wt, Q_dict in results["self_Q"].items():
        prefix = f"{cfg.model_name}_{rev}_{wt}"

        # Q matrices
        npz_dict = {f"Q_{m}": Q for m, Q in Q_dict.items()}
        np.savez_compressed(f"{out_dir}/{prefix}_Q.npz", **npz_dict)

        # summaries
        wt_summary = results["summaries"].get(wt, {})
        json_summaries = {}
        for m, s in wt_summary.items():
            js = {k: v for k, v in s.items()
                  if not isinstance(v, np.ndarray)}
            json_summaries[m] = js
        with open(f"{out_dir}/{prefix}_summary.json", "w") as f:
            json.dump(json_summaries, f, indent=2)

        # eigenvalues and P(Q)
        for m, s in wt_summary.items():
            np.save(f"{out_dir}/{prefix}_{m}_eigenvalues.npy", s["eigenvalues"])
            np.save(f"{out_dir}/{prefix}_{m}_P_Q.npy", s["P_Q_values"])

        # block means
        wt_blocks = results["block_means"].get(wt, {})
        for m, (block, layers) in wt_blocks.items():
            np.save(f"{out_dir}/{prefix}_{m}_block_means.npy", block)

        # metadata
        store = results["stores"][wt]
        meta = {
            "model_name": cfg.model_name,
            "revision": cfg.revision,
            "weight_type": wt,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "head_dim": cfg.head_dim,
            "metrics": list(Q_dict.keys()),
            "head_index": store.keys,
        }
        with open(f"{out_dir}/{prefix}_metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logging.info(f"  Self-corr saved: {prefix}_*")

    # ── Cross-correlations ──
    for label, Q_dict in results["cross_Q"].items():
        prefix = f"{cfg.model_name}_{rev}_{label}"
        npz_dict = {f"Q_{m}": Q for m, Q in Q_dict.items()}
        np.savez_compressed(f"{out_dir}/{prefix}_Q.npz", **npz_dict)

        meta = {
            "model_name": cfg.model_name,
            "revision": cfg.revision,
            "cross_label": label,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "head_dim": cfg.head_dim,
            "metrics": list(Q_dict.keys()),
            "head_index": list(results["stores"].values())[0].keys,
        }
        with open(f"{out_dir}/{prefix}_metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logging.info(f"  Cross-corr saved: {prefix}_*")

    # ── Scalar observables ──
    scalars_df = results.get("scalars")
    if scalars_df is not None and len(scalars_df) > 0:
        prefix = f"{cfg.model_name}_{rev}_scalars"
        scalars_df.to_csv(f"{out_dir}/{prefix}.csv", index=False)
        logging.info(f"  Scalars saved: {prefix}.csv")


# ── CLI ───────────────────────────────────────────────────────────────

def find_cached_models(cache_dir: str) -> list[str]:
    """Find which registered models have cached weights in cache_dir.

    Returns list of model names that exist in both the cache directory
    and the MODEL_CONFIGS registry.
    """
    from .model_registry import MODEL_CONFIGS

    found = []
    if not os.path.isdir(cache_dir):
        return found

    # The cache structure is: {cache_dir}/{model_name}/{revision}/...
    for entry in os.listdir(cache_dir):
        entry_path = os.path.join(cache_dir, entry)
        if os.path.isdir(entry_path) and entry in MODEL_CONFIGS:
            found.append(entry)
    return sorted(found)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    parser = argparse.ArgumentParser(description="Head-head correlation analysis")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model name (default: process all cached models)")

    # Legacy single-circuit mode
    parser.add_argument("--weight-type", type=str, default=None,
                        choices=["W_QK", "W_Q", "W_K"],
                        help="(legacy) Single weight type for backward compat")

    # Multi-circuit mode
    parser.add_argument("--circuits", nargs="+", default=None,
                        choices=["QK", "OV"],
                        help="Circuits to process (default: QK)")
    parser.add_argument("--include-bias", action="store_true", default=False,
                        help="Also extract and analyze bias vectors")
    parser.add_argument("--cross", nargs="*", default=None,
                        choices=["QKOV", "WB"],
                        help="Cross-correlations to compute")

    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--out", type=str, default="corr_out")
    parser.add_argument("--cache", type=str, default="./model_data")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "mps", "cpu"])
    parser.add_argument("--metrics", nargs="+",
                        default=["frob_cosine", "two_point",
                                 "connected_corr", "pearson_corr",
                                 "hist_symmetric_kl", "hist_jensen_shannon"])
    parser.add_argument("--cross-metrics", nargs="+",
                        default=["frob_cosine", "pearson_corr"])
    parser.add_argument("--wb-cross-metrics", nargs="+",
                        default=["hist_symmetric_kl", "hist_jensen_shannon"],
                        help="Metrics for W↔b cross-correlations (distribution-based)")
    parser.add_argument("--kde-n-eval", type=int, default=2048)
    parser.add_argument("--kde-bw", type=str, default="scott")
    parser.add_argument("--resume-download", action="store_true", default=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Models to skip when processing all cached models")
    args = parser.parse_args()

    # Determine which models to process
    if args.model:
        models_to_process = [args.model]
    else:
        # Auto-discover from cache
        models_to_process = find_cached_models(args.cache)
        models_to_process = [m for m in models_to_process if m not in args.skip]
        if not models_to_process:
            logging.error(f"No cached models found in {args.cache}")
            sys.exit(1)
        logging.info(f"Found {len(models_to_process)} cached models: {models_to_process}")

    # Process each model
    for model_idx, model_name in enumerate(models_to_process):
        if len(models_to_process) > 1:
            logging.info(f"\n{'='*70}")
            logging.info(f"Processing model [{model_idx+1}/{len(models_to_process)}]: {model_name}")
            logging.info(f"{'='*70}")

        # Decide which mode to use
        if args.weight_type is not None and args.circuits is None:
            # Legacy single-circuit mode
            run_correlation_analysis(
                model_name=model_name,
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
        else:
            # Multi-circuit mode
            circuits = tuple(args.circuits) if args.circuits else ("QK",)
            cross = tuple(args.cross) if args.cross else ()
            run_multi_circuit_analysis(
                model_name=model_name,
                circuits=circuits,
                include_bias=args.include_bias,
                cross_correlations=cross,
                metrics=tuple(args.metrics),
                cross_metrics=tuple(args.cross_metrics),
                wb_cross_metrics=tuple(args.wb_cross_metrics),
                revision=args.revision,
                cache_dir=args.cache,
                out_dir=args.out,
                device=args.device,
                kde_n_eval=args.kde_n_eval,
                kde_bw=args.kde_bw,
                resume_download=args.resume_download,
                max_workers=args.max_workers,
            )

    if len(models_to_process) > 1:
        logging.info(f"\n{'='*70}")
        logging.info(f"Completed processing {len(models_to_process)} models")
        logging.info(f"{'='*70}")
