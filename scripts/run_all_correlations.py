#!/usr/bin/env python3
"""Run correlation analysis + plotting for all models found in a cache directory.

Discovers which registered models have cached weights, estimates runtime,
and runs the full pipeline for each.

Usage:
    python scripts/run_all_correlations.py \
        --cache /Volumes/Flux/Projects/transformer-analysis/downloads \
        --out corr_out \
        --skip gpt2

Time estimates are based on:
  - GPT-2 (12 layers, 12 heads, d_head=64) took ~20 min
  - Scaling: extraction ~ O(n_layers), pair loop ~ O(N_heads^2) with KDE overhead
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformer_analysis.model_registry import MODEL_CONFIGS, get_model_config
from transformer_analysis.correlation_analysis import run_correlation_analysis

# ── Model dimensions for time estimation ──────────────────────────────
# (n_layers, n_heads, head_dim) — from model configs
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
    "llama-3.1-70b":   (80, 64, 128),
    "llama-3.2-1b":    (16, 32,  64),
    "llama-3.2-3b":    (28, 24, 128),
    "mistral-7b-v0.3": (32, 32, 128),
    "mixtral-8x7b-v0.1": (32, 32, 128),
    "mixtral-8x22b-v0.1": (56, 48, 128),
}

# GPT-2 reference: 144 heads, ~20 min total
GPT2_HEADS = 144
GPT2_TIME_MIN = 20.0


FAST_METRICS = ["frob_cosine", "two_point", "connected_corr", "pearson_corr"]
HIST_METRICS = ["hist_symmetric_kl", "hist_jensen_shannon"]
KDE_METRICS = ["symmetric_kl", "jensen_shannon"]
# Default: fast metrics + histogram divergences (all cheap)
DEFAULT_METRICS = FAST_METRICS + HIST_METRICS
# Full: adds KDE-based divergences (expensive)
ALL_METRICS = DEFAULT_METRICS + KDE_METRICS


def estimate_time_minutes(model_name, fast_only=False):
    """Estimate runtime in minutes based on model dimensions.

    Two cost components:
      1. Extraction: ~linear in n_layers × model_size (IO-bound)
      2. Pair loop:
         - Fast metrics (dot products): O(N_heads^2 * d^2), very cheap
         - KDE metrics: O(N_heads^2 * d^2), ~100x more expensive per pair

    Reference: GPT-2 (12L, 12H, d_h=64) = ~20 min full, ~3 min fast.
    """
    if model_name not in MODEL_DIMS:
        return None
    n_layers, n_heads, d_head = MODEL_DIMS[model_name]
    N = n_layers * n_heads
    d2 = d_head ** 2
    n_pairs = N * (N - 1) // 2

    ref_N = GPT2_HEADS
    ref_pairs = ref_N * (ref_N - 1) // 2
    ref_d2 = 64 ** 2

    # extraction: scales with n_layers and model file size
    # gpt2 (12 layers, 0.5GB) ~ 2 min extraction
    size_factor = (n_layers / 12.0) * (d_head / 64.0)
    extract_min = 2.0 * size_factor

    # fast pair loop: gpt2 ~ 0.5 min for dot-product metrics
    fast_pair = 0.5 * (n_pairs / ref_pairs) * (d2 / ref_d2)

    if fast_only:
        return extract_min + fast_pair

    # KDE pair loop: gpt2 ~ 17 min for KDE metrics
    kde_pair = 17.0 * (n_pairs / ref_pairs) * (d2 / ref_d2)

    return extract_min + fast_pair + kde_pair


def find_cached_models(cache_dir):
    """Find which registered models have cached weights in cache_dir."""
    found = []
    if not os.path.isdir(cache_dir):
        return found

    # The cache structure is: {cache_dir}/{model_name}/{revision}/...
    for entry in os.listdir(cache_dir):
        entry_path = os.path.join(cache_dir, entry)
        if os.path.isdir(entry_path) and entry in MODEL_CONFIGS:
            found.append(entry)
    return sorted(found)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Run correlation analysis for all cached models")
    parser.add_argument("--cache", type=str, required=True,
                        help="Model cache directory (e.g. /Volumes/.../downloads)")
    parser.add_argument("--out", type=str, default="corr_out",
                        help="Output directory for correlation results")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Models to skip (e.g. --skip gpt2)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "mps", "cpu"])
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Explicit metric list (default: depends on --fast)")
    parser.add_argument("--fast", action="store_true",
                        help="Only compute dot-product metrics (no KDE). ~50x faster.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print time estimates without running")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Generate plots after each model (default: True)")
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    args = parser.parse_args()

    # Resolve metrics
    if args.metrics is not None:
        metrics = args.metrics
    elif args.fast:
        metrics = FAST_METRICS
    else:
        metrics = DEFAULT_METRICS  # histogram divergences, no KDE
    fast_only = all(m not in KDE_METRICS for m in metrics)

    # Discover models
    cached = find_cached_models(args.cache)
    to_run = [m for m in cached if m not in args.skip]

    if not to_run:
        print(f"No models found in {args.cache} (or all skipped).")
        print(f"Available in cache: {cached}")
        print(f"Skipping: {args.skip}")
        return

    # Print table of estimates
    print("\n" + "=" * 70)
    print(f"{'Model':<28} {'N_heads':>8} {'d_head²':>8} {'Est. time':>12}")
    print("-" * 70)

    total_est = 0.0
    for model in to_run:
        dims = MODEL_DIMS.get(model)
        if dims:
            n_l, n_h, d_h = dims
            N = n_l * n_h
            est = estimate_time_minutes(model, fast_only=fast_only)
            total_est += est
            if est < 60:
                time_str = f"{est:.0f} min"
            else:
                time_str = f"{est / 60:.1f} hr"
            print(f"  {model:<26} {N:>8} {d_h**2:>8} {time_str:>12}")
        else:
            print(f"  {model:<26} {'?':>8} {'?':>8} {'unknown':>12}")

    print("-" * 70)
    if total_est < 60:
        print(f"  {'TOTAL':<26} {'':>8} {'':>8} {total_est:.0f} min")
    else:
        print(f"  {'TOTAL':<26} {'':>8} {'':>8} {total_est / 60:.1f} hr")
    print("=" * 70 + "\n")

    if args.dry_run:
        print("Dry run — exiting.")
        return

    # Run each model
    os.makedirs(args.out, exist_ok=True)

    for i, model in enumerate(to_run):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(to_run)}] {model}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            run_correlation_analysis(
                model_name=model,
                weight_type="W_QK",
                metrics=tuple(metrics),
                cache_dir=args.cache,
                out_dir=args.out,
                device=args.device,
                save=True,
            )
        except Exception as e:
            logging.error(f"FAILED: {model} — {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = (time.time() - t0) / 60
        print(f"  Completed in {elapsed:.1f} min")

        # Plot
        if args.plot:
            try:
                from scripts_plot import plot_model
            except ImportError:
                # inline plot call
                import subprocess
                subprocess.run([
                    sys.executable, "scripts/plot_correlations.py",
                    "--data", args.out,
                    "--model", model,
                    "--out", os.path.join(args.out, "figures"),
                ], check=True)

    print(f"\n{'='*60}")
    print(f"All done. Results in {args.out}/")
    print(f"Figures in {args.out}/figures/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
