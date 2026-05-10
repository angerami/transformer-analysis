#!/usr/bin/env python3
"""
Weight Analysis Script - Flexible harness for transformer weight analysis

This script supports three modes of operation:
1. Single model, single revision (e.g., gpt2 at main)
2. Single model, all revisions (e.g., pythia-70m-deduped across all checkpoints)
3. Config file mode (process multiple models/revisions from JSON config)

Examples:
    # Single model, single revision
    python run_weight_analysis.py --model gpt2 --revision main

    # Single model, all revisions
    python run_weight_analysis.py --model pythia-70m-deduped --all-revisions

    # Process all revisions and merge into single dataset
    python run_weight_analysis.py --model pythia-70m-deduped --all-revisions --merge

    # Config file mode
    python run_weight_analysis.py --config sweep_config.json

    # List available models
    python run_weight_analysis.py --list-models
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

from tqdm import tqdm
from transformers import logging as hf_logging
import warnings

from transformer_analysis.weight_analysis import process_model, create_campaign, merge_versions
from transformer_analysis.model_registry import MODEL_CONFIGS, list_supported_models, get_model_config


def process_single_model(
    model_name: str,
    revision: Optional[str] = None,
    all_revisions: bool = False,
    out_dir: str = "outputs",
    cache_dir: str = "./model_data",
    clobber: bool = False,
    cleanup_downloads: bool = False,
    low_rank_svd: bool = False,
    top_k_svd: int = -1,
    resume_download: bool = True,
    max_workers: int = 4,
    device: Optional[str] = None,
    quiet: bool = False,
    merge: bool = False,
    merge_suffix: str = "all_checkpoints",
    skip_postprocess: bool = False,
    binning_strategy: str = "fixed",
):
    if not quiet:
        print("\n" + "=" * 80)
        print(f"Processing Model: {model_name}")
        print("=" * 80)

    try:
        model_config = get_model_config(model_name)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    # Determine which revisions to process
    if all_revisions:
        revisions = model_config.revisions
        if not revisions:
            print(f"Model {model_name} has no revisions defined. Processing main branch only.")
            revisions = [None]
    elif revision:
        revisions = [revision]
    else:
        revisions = [None]  # Main/latest branch

    if not quiet:
        print(f"Revisions to process: {len(revisions)}")
        if len(revisions) <= 10:
            print(f"  {revisions}")

    # Process each revision
    for rev in tqdm(revisions, desc=f"Processing {model_name}", disable=quiet):
        revision_str = rev if rev else "main"

        # Check if output already exists
        if rev:
            target_dir = os.path.join(out_dir, f"{model_name}_{revision_str}")
        else:
            target_dir = os.path.join(out_dir, model_name)

        if os.path.exists(target_dir) and not clobber:
            if not quiet:
                print(f"  Skipping {model_name} @ {revision_str} (output exists)")
            continue

        if not quiet:
            print(f"\n  Processing: {model_name} @ {revision_str}")

        try:
            process_model(
                model_name=model_name,
                revision=rev,
                out_dir=out_dir,
                cache_dir=cache_dir,
                cleanup_downloads=cleanup_downloads,
                low_rank_svd_approximation=low_rank_svd,
                top_k_svd=top_k_svd,
                resume_download=resume_download,
                max_workers=max_workers,
                device=device,
                skip_postprocess=skip_postprocess,
                binning_strategy=binning_strategy,
            )
        except Exception as e:
            print(f"  ERROR processing {model_name} @ {revision_str}: {e}")
            continue

    if not quiet:
        print("\n" + "=" * 80)
        print(f"Completed: {model_name}")
        print("=" * 80 + "\n")

    # Merge datasets if requested and multiple revisions were processed
    if merge and len(revisions) > 1:
        if not quiet:
            print("\n" + "=" * 80)
            print(f"Merging datasets for: {model_name}")
            print("=" * 80)

        try:
            merge_versions(
                model_name=model_name,
                path=out_dir,
                suffix=merge_suffix,
            )
            if not quiet:
                print(f"Successfully merged to: {out_dir}/{model_name}_{merge_suffix}")
                print("=" * 80 + "\n")
        except Exception as e:
            print(f"ERROR merging datasets: {e}")
            if not quiet:
                print("=" * 80 + "\n")


def process_from_config(
    config_path: str,
    out_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    clobber: Optional[bool] = None,
    cleanup_downloads: Optional[bool] = None,
    low_rank_svd: Optional[bool] = None,
    top_k_svd: Optional[int] = None,
    resume_download: Optional[bool] = None,
    max_workers: Optional[int] = None,
    device: Optional[str] = None,
    quiet: bool = False,
):
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in config file: {e}")
        return

    # Extract global settings (with CLI overrides)
    global_out_dir = out_dir or config.get("output_dir", "outputs")
    global_cache_dir = cache_dir or config.get("cache_dir", "./model_data")
    global_clobber = clobber if clobber is not None else config.get("clobber", False)
    global_cleanup = cleanup_downloads if cleanup_downloads is not None else config.get("cleanup_downloads", False)
    global_low_rank_svd = low_rank_svd if low_rank_svd is not None else config.get("low_rank_svd", False)
    global_top_k_svd = top_k_svd if top_k_svd is not None else config.get("top_k_svd", -1)
    global_resume_download = resume_download if resume_download is not None else config.get("resume_download", True)
    global_max_workers = max_workers if max_workers is not None else config.get("max_workers", 4)
    global_device = device if device is not None else config.get("device", None)

    if not quiet:
        print("\n" + "=" * 80)
        print(f"Processing from config: {config_path}")
        print("=" * 80)
        print(f"Output directory: {global_out_dir}")
        print(f"Cache directory: {global_cache_dir}")
        print(f"Clobber: {global_clobber}")
        print(f"Cleanup downloads: {global_cleanup}")
        print(f"Low-rank SVD: {global_low_rank_svd}")
        if global_low_rank_svd:
            print(f"Top-k SVD: {global_top_k_svd}")
        print(f"Resume downloads: {global_resume_download}")
        print(f"Max workers: {global_max_workers}")
        print("=" * 80 + "\n")

    # Process models
    models = config.get("models", [])
    if not models:
        print("ERROR: No models specified in config file")
        return

    for model_spec in models:
        if isinstance(model_spec, str):
            # Simple format: just model name
            model_name = model_spec
            revisions = None
            all_revs = False
            model_low_rank = global_low_rank_svd
            model_top_k = global_top_k_svd
        elif isinstance(model_spec, dict):
            # Detailed format
            model_name = model_spec.get("name")
            if not model_name:
                print("ERROR: Model specification missing 'name' field")
                continue

            revisions = model_spec.get("revisions")
            all_revs = model_spec.get("all_revisions", False)
            model_low_rank = model_spec.get("low_rank_svd", global_low_rank_svd)
            model_top_k = model_spec.get("top_k_svd", global_top_k_svd)
        else:
            print(f"ERROR: Invalid model specification: {model_spec}")
            continue

        # Process based on specification
        if revisions:
            # Specific revisions listed
            for rev in revisions:
                process_single_model(
                    model_name=model_name,
                    revision=rev,
                    all_revisions=False,
                    out_dir=global_out_dir,
                    cache_dir=global_cache_dir,
                    clobber=global_clobber,
                    cleanup_downloads=global_cleanup,
                    low_rank_svd=model_low_rank,
                    top_k_svd=model_top_k,
                    resume_download=global_resume_download,
                    max_workers=global_max_workers,
                    device=global_device,
                    quiet=quiet,
                )
        else:
            # Process with all_revisions flag
            process_single_model(
                model_name=model_name,
                revision=None,
                all_revisions=all_revs,
                out_dir=global_out_dir,
                cache_dir=global_cache_dir,
                clobber=global_clobber,
                cleanup_downloads=global_cleanup,
                low_rank_svd=model_low_rank,
                top_k_svd=model_top_k,
                resume_download=global_resume_download,
                max_workers=global_max_workers,
                device=global_device,
                quiet=quiet,
            )


def model_size_from_name(model_name: str) -> float:
    """Extract model size for sorting (in millions of parameters)."""
    import re

    # Extract size like "70m", "1.4b", "12b"
    match = re.search(r"(\d+\.?\d*)([mb])", model_name.lower())
    if not match:
        return 0

    size, unit = match.groups()
    size = float(size)

    # Convert to millions for consistent comparison
    if unit == "b":
        size *= 1000

    return size


def list_models():
    models = list_supported_models()
    print("\nAvailable models:")
    print("=" * 80)

    # Group by model family
    families = {}
    for model in models:
        if "pythia" in model:
            family = "Pythia"
        elif "gpt2" in model:
            family = "GPT-2"
        elif "llama" in model:
            family = "LLaMA"
        elif "mistral" in model or "mixtral" in model:
            family = "Mistral/Mixtral"
        else:
            family = "Other"

        if family not in families:
            families[family] = []
        families[family].append(model)

    for family, family_models in sorted(families.items()):
        print(f"\n{family}:")
        # Sort models within family by size
        for model in sorted(family_models, key=model_size_from_name):
            model_config = get_model_config(model)
            n_revisions = len(model_config.revisions)
            rev_info = f"({n_revisions} revisions)" if n_revisions > 0 else "(no revisions)"
            print(f"  - {model} {rev_info}")

    print("\n" + "=" * 80)
    print(f"Total models: {len(models)}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Weight Analysis Script - Flexible harness for transformer weight analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model, main revision
  %(prog)s --model gpt2

  # Single model, specific revision
  %(prog)s --model pythia-70m-deduped --revision step1000

  # Single model, all revisions
  %(prog)s --model pythia-70m-deduped --all-revisions

  # Process all revisions and merge into single dataset
  %(prog)s --model pythia-70m-deduped --all-revisions --merge

  # Config file mode
  %(prog)s --config sweep_config.json

  # With low-rank SVD
  %(prog)s --model gpt2 --low-rank-svd --top-k-svd 64

  # List available models
  %(prog)s --list-models
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--model", type=str, help="Model name to process")
    mode_group.add_argument("--config", type=str, help="Path to JSON config file")
    mode_group.add_argument("--list-models", action="store_true", help="List all available models")

    # Model-specific options
    parser.add_argument("--revision", type=str, help="Specific revision/checkpoint to process")
    parser.add_argument("--all-revisions", action="store_true",
                        help="Process all available revisions for the model")

    # Directory options
    parser.add_argument("--out", type=str, default="outputs",
                        help="Output directory (default: outputs)")
    parser.add_argument("--cache", type=str, default="./model_data",
                        help="Cache directory for downloads (default: ./model_data)")

    # Processing options
    parser.add_argument("--clobber", action="store_true",
                        help="Overwrite existing outputs")
    parser.add_argument("--cleanup", action="store_true",
                        help="Clean up downloaded models after processing")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all revisions into a single dataset after processing (only with --all-revisions)")
    parser.add_argument("--merge-suffix", type=str, default="all_checkpoints",
                        help="Suffix for merged dataset (default: all_checkpoints)")

    parser.add_argument("--low-rank-svd", action="store_true")
    parser.add_argument("--top-k-svd", type=int, default=-1)
    parser.add_argument("--binning-strategy", type=str, default="fixed",
                        choices=["fixed", "scott", "fd"],
                        help="Histogram binning strategy: fixed linspace (default), Scott's rule, or Freedman-Diaconis")
    parser.add_argument("--resume-download", action="store_true", default=True, dest="resume_download")
    parser.add_argument("--no-resume-download", action="store_false", dest="resume_download")
    parser.add_argument("--max-workers", type=int, default=4, dest="max_workers")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--quiet", "-q", action="store_true")

    # Post-processing options
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Skip post-processing metrics computation")
    parser.add_argument("--reprocess-metrics", action="store_true",
                        help="Reprocess existing datasets to update/add metrics without re-running full analysis")

    args = parser.parse_args()

    # Suppress warnings unless verbose
    if args.quiet:
        hf_logging.set_verbosity_error()
        warnings.filterwarnings("ignore")

    # Handle list-models mode
    if args.list_models:
        list_models()
        return

    # Handle config file mode
    if args.config:
        process_from_config(
            config_path=args.config,
            out_dir=args.out if args.out != "outputs" else None,
            cache_dir=args.cache if args.cache != "./model_data" else None,
            clobber=args.clobber if args.clobber else None,
            cleanup_downloads=args.cleanup if args.cleanup else None,
            low_rank_svd=args.low_rank_svd if args.low_rank_svd else None,
            top_k_svd=args.top_k_svd if args.top_k_svd != -1 else None,
            resume_download=args.resume_download,
            max_workers=args.max_workers,
            device=args.device,
            quiet=args.quiet,
        )
        return

    # Handle single model mode
    if args.model:
        # Check for reprocess-metrics mode
        if args.reprocess_metrics:
            # Import the reprocessing function
            from transformer_analysis.weight_analysis import reprocess_metrics
            reprocess_metrics(
                model_name=args.model,
                revision=args.revision,
                all_revisions=args.all_revisions,
                out_dir=args.out,
                quiet=args.quiet,
            )
            return

        # Suppress warnings for cleaner output
        hf_logging.set_verbosity_error()
        warnings.filterwarnings("ignore")

        process_single_model(
            model_name=args.model,
            revision=args.revision,
            all_revisions=args.all_revisions,
            out_dir=args.out,
            cache_dir=args.cache,
            clobber=args.clobber,
            cleanup_downloads=args.cleanup,
            low_rank_svd=args.low_rank_svd,
            top_k_svd=args.top_k_svd,
            resume_download=args.resume_download,
            max_workers=args.max_workers,
            device=args.device,
            quiet=args.quiet,
            merge=args.merge,
            merge_suffix=args.merge_suffix,
            skip_postprocess=args.no_postprocess,
            binning_strategy=args.binning_strategy,
        )


if __name__ == "__main__":
    main()
