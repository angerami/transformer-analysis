#!/usr/bin/env python3
"""Full production run: correlation analysis + plotting for all cached models.

Runs multi-circuit analysis (QK, OV, biases, cross-correlations) for every
model found in the cache directory, then generates all plots including
multi-model comparisons.

Usage:
    # Dry run — show what would be processed
    python scripts/run_all_correlations.py \
        --cache /Volumes/Flux/Projects/transformer-analysis/downloads \
        --dry-run

    # Full production run
    python scripts/run_all_correlations.py \
        --cache /Volumes/Flux/Projects/transformer-analysis/downloads \
        --out corr_out

    # Fast metrics only (skip KDE), no biases
    python scripts/run_all_correlations.py \
        --cache /path/to/downloads --fast --no-bias

    # Add new metrics to an existing run without full re-extraction
    python scripts/run_all_correlations.py \
        --cache /path/to/downloads --out corr_out \
        --add-metrics hist_jensen_shannon hist_symmetric_kl
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformer_analysis.model_registry import MODEL_CONFIGS
from transformer_analysis.correlation_analysis import (
    run_multi_circuit_analysis,
    find_cached_models,
    extract_head_store,
)
from transformer_analysis.head_correlations import (
    compute_correlation_matrices,
    correlation_summary,
    layer_block_means,
)

# (n_layers, n_heads, head_dim)
MODEL_DIMS = {
    "gpt2":            (12, 12,  64),
    "gpt2-medium":     (24, 16,  64),
    "gpt2-large":      (36, 20,  64),
    "gpt2-xl":         (48, 25,  64),
    "pythia-70m-deduped":   (6,  8,  64),
    "pythia-160m-deduped":  (12, 12, 64),
    "pythia-410m-deduped":  (24, 16, 64),
    "pythia-1b-deduped":    (16, 8, 128),
    "pythia-1.4b-deduped":  (24, 16, 128),
    "pythia-2.8b-deduped":  (32, 32, 80),
    "pythia-6.9b-deduped":  (32, 32, 128),
    "pythia-12b-deduped":   (36, 40, 128),
    "llama-3.1-8b":    (32, 32, 128),
    "mistral-7b-v0.3": (32, 32, 128),
}

FAST_METRICS = ["frob_cosine", "two_point", "connected_corr", "pearson_corr"]
HIST_METRICS = ["hist_symmetric_kl", "hist_jensen_shannon"]
DEFAULT_METRICS = FAST_METRICS + HIST_METRICS


def recompute_metrics_for_model(model_name, new_metrics, out_dir, cache_dir, device=None):
    """Add new metrics to all existing weight-type outputs for a model.

    Discovers weight types from metadata files in out_dir, skips metrics that
    are already present, re-extracts heads from cache, and merges new results
    into existing .npz and summary files.
    """
    meta_paths = sorted(
        p for p in glob.glob(os.path.join(out_dir, f"{model_name}_*_metadata.json"))
        if "_vs_" not in os.path.basename(p)
    )
    if not meta_paths:
        logging.warning(f"  No existing outputs found for {model_name} in {out_dir}")
        return

    for meta_path in meta_paths:
        with open(meta_path) as f:
            meta = json.load(f)

        fname = os.path.basename(meta_path).replace("_metadata.json", "")
        model_rev_prefix = f"{model_name}_main_"
        if not fname.startswith(model_rev_prefix):
            logging.warning(f"  Skipping {os.path.basename(meta_path)}: non-main revision")
            continue
        weight_type = fname[len(model_rev_prefix):]

        to_add = [m for m in new_metrics if m not in meta.get("metrics", [])]
        if not to_add:
            logging.info(f"  {model_name}/{weight_type}: metrics already present, skipping")
            continue

        logging.info(f"  {model_name}/{weight_type}: adding {to_add}")

        store, _ = extract_head_store(
            model_name=model_name,
            weight_type=weight_type,
            revision=None,
            cache_dir=cache_dir,
            device=device,
        )

        Q_new = compute_correlation_matrices(
            store,
            metrics=tuple(to_add),
            kde_kwargs={"n_eval": 2048, "bw_method": "scott"},
            show_progress=True,
        )

        # Merge into existing .npz
        npz_path = os.path.join(out_dir, f"{fname}_Q.npz")
        existing = dict(np.load(npz_path)) if os.path.exists(npz_path) else {}
        for m, Q in Q_new.items():
            existing[f"Q_{m}"] = Q
        np.savez_compressed(npz_path, **existing)

        # Update summary + per-metric arrays
        summary_path = os.path.join(out_dir, f"{fname}_summary.json")
        summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
        for m, Q in Q_new.items():
            s = correlation_summary(Q, store.keys)
            summary[m] = {k: v for k, v in s.items() if not isinstance(v, np.ndarray)}
            np.save(os.path.join(out_dir, f"{fname}_{m}_eigenvalues.npy"), s["eigenvalues"])
            np.save(os.path.join(out_dir, f"{fname}_{m}_P_Q.npy"), s["P_Q_values"])
            block, _ = layer_block_means(Q, store.keys)
            np.save(os.path.join(out_dir, f"{fname}_{m}_block_means.npy"), block)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Update metadata
        for m in to_add:
            meta.setdefault("metrics", []).append(m)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)


def estimate_time_minutes(model_name, n_circuits=2, include_bias=True, fast_only=True):
    """Rough runtime estimate in minutes."""
    if model_name not in MODEL_DIMS:
        return None
    n_l, n_h, d_h = MODEL_DIMS[model_name]
    N = n_l * n_h
    n_pairs = N * (N - 1) // 2
    ref_pairs = 144 * 143 // 2
    d2_ratio = (d_h ** 2) / (64 ** 2)

    # Extraction: ~2 min for gpt2-scale, scales with depth and width
    extract = 2.0 * (n_l / 12.0) * (d_h / 64.0)

    # Pair loop: ~0.5 min for gpt2 fast metrics per circuit
    pair = 0.5 * (n_pairs / ref_pairs) * d2_ratio
    if not fast_only:
        pair += 17.0 * (n_pairs / ref_pairs) * d2_ratio

    total = extract + pair * n_circuits
    if include_bias:
        total += pair * 4  # b_Q, b_K, b_V, b_O are cheaper but there are 4
    return total


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Full production run: correlations + plots for all models")
    parser.add_argument("--cache", type=str, required=True,
                        help="Model cache directory")
    parser.add_argument("--out", type=str, default="corr_out",
                        help="Output directory for correlation results")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Models to skip")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only run these models (default: all cached)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "mps", "cpu"])
    parser.add_argument("--fast", action="store_true",
                        help="Fast metrics only (skip histogram divergences)")
    parser.add_argument("--no-bias", action="store_true",
                        help="Skip bias correlations")
    parser.add_argument("--no-cross", action="store_true",
                        help="Skip cross-correlations")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--config", type=str, default=None,
                        help="Read detailed options from config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and time estimates without running")
    parser.add_argument("--add-metrics", nargs="+", default=None,
                        help="Add metrics to existing outputs without full re-extraction")
    args = parser.parse_args()

    metrics = FAST_METRICS if args.fast else DEFAULT_METRICS
    circuits = ("QK", "OV")
    cross = () if args.no_cross else ("QKOV", "WB")
    include_bias = not args.no_bias

    to_run = None
    if args.config is not None:
        with open(args.config, "r") as f:
            config_opts = json.load(f)
            if 'metrics' in config_opts:
                metrics = config_opts['metrics']
            if 'circuits' in config_opts:
                circuits = tuple(config_opts['circuits'])
            if 'cross' in config_opts:
                cross = tuple(config_opts['cross'])
            if 'models' in config_opts:
                to_run = config_opts['models']

    if not to_run:
        # Discover models
        cached = find_cached_models(args.cache)
        if args.models:
            to_run = [m for m in args.models if m in cached]
            missing = [m for m in args.models if m not in cached]
            if missing:
                print(f"Warning: not cached: {missing}")
        else:
            to_run = [m for m in cached if m not in args.skip]

    if not to_run:
        print(f"No models to run. Cached: {cached}")
        return 1

    # Summary table
    print("\n" + "=" * 70)
    print("PRODUCTION RUN PLAN")
    print(f"  Circuits:  {', '.join(circuits)}")
    print(f"  Biases:    {'yes' if include_bias else 'no'}")
    print(f"  Cross:     {', '.join(cross) if cross else 'none'}")
    print(f"  Metrics:   {', '.join(metrics)}")
    print(f"  Output:    {os.path.abspath(args.out)}")
    print("-" * 70)

    header = "{:<28} {:>8} {:>8} {:>12}".format("Model", "N_heads", "d_head", "Est. time")
    print(header)
    print("-" * 70)

    total_est = 0.0
    for model in to_run:
        dims = MODEL_DIMS.get(model)
        if dims:
            n_l, n_h, d_h = dims
            N = n_l * n_h
            est = estimate_time_minutes(model, len(circuits), include_bias, args.fast)
            total_est += est
            time_str = "{:.0f} min".format(est) if est < 60 else "{:.1f} hr".format(est / 60)
            print("  {:<26} {:>8} {:>8} {:>12}".format(model, N, d_h, time_str))
        else:
            print("  {:<26} {:>8} {:>8} {:>12}".format(model, "?", "?", "unknown"))

    print("-" * 70)
    total_str = "{:.0f} min".format(total_est) if total_est < 60 else "{:.1f} hr".format(total_est / 60)
    print("  {:<26} {:>8} {:>8} {:>12}".format("TOTAL", "", "", total_str))
    print("=" * 70 + "\n")

    if args.dry_run:
        print("Dry run — exiting.")
        return 0

    # Run analysis
    os.makedirs(args.out, exist_ok=True)
    results = {}

    for i, model in enumerate(to_run):
        print("\n" + "=" * 60)
        print("[{}/{}] {}".format(i + 1, len(to_run), model))
        print("=" * 60)

        t0 = time.time()
        try:
            if args.add_metrics:
                recompute_metrics_for_model(
                    model_name=model,
                    new_metrics=args.add_metrics,
                    out_dir=args.out,
                    cache_dir=args.cache,
                    device=args.device,
                )
            else:
                run_multi_circuit_analysis(
                    model_name=model,
                    circuits=circuits,
                    include_bias=include_bias,
                    cross_correlations=cross,
                    metrics=tuple(metrics),
                    cache_dir=args.cache,
                    out_dir=args.out,
                    device=args.device,
                )
            elapsed = (time.time() - t0) / 60
            results[model] = ("OK", elapsed)
            print("  Completed in {:.1f} min".format(elapsed))
        except Exception as e:
            elapsed = (time.time() - t0) / 60
            results[model] = ("FAILED", elapsed)
            logging.error("FAILED: {} — {}".format(model, e))
            import traceback
            traceback.print_exc()

    # Plot all
    if not args.no_plot:
        print("\n" + "=" * 60)
        print("PLOTTING")
        print("=" * 60)
        subprocess.run([
            sys.executable, os.path.join(os.path.dirname(__file__), "plot_corr_figures.py"),
            "--data", args.out,
        ])

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("-" * 60)
    for model, (status, elapsed) in results.items():
        print("  {:<30} {:>8} {:>8.1f} min".format(model, status, elapsed))
    n_ok = sum(1 for s, _ in results.values() if s == "OK")
    n_fail = sum(1 for s, _ in results.values() if s != "OK")
    print("-" * 60)
    print("  {} succeeded, {} failed".format(n_ok, n_fail))
    print("  Output: {}".format(os.path.abspath(args.out)))
    print("=" * 60)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
