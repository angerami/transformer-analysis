#!/usr/bin/env python3
"""
Dataset Merging Script

Lightweight wrapper around transformer_analysis.weight_analysis merge functions.

This script supports two modes:
1. Merge all versions/checkpoints of a single model (--model flag)
2. Merge multiple different models into a combined dataset (no --model flag)

Examples:
    # Merge all checkpoints of pythia-70m-deduped
    python merge_datasets.py --model pythia-70m-deduped --path outputs

    # Merge multiple models from a directory
    python merge_datasets.py --path outputs --out-name cross_model_study
"""

from pathlib import Path
from transformer_analysis.weight_analysis import merge_versions, merge_datasets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge weight analysis datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model name to merge all versions/checkpoints",
    )
    parser.add_argument(
        "--path", type=str, default="histos", help="Base directory containing datasets"
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="weight_study",
        help="Output name for merged dataset",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="all_checkpoints",
        help="Suffix for single-model merge output",
    )
    args = parser.parse_args()

    if args.model is not None:
        # Mode 1: Merge all versions of a single model
        merge_versions(model_name=args.model, path=args.path, suffix=args.suffix)
    else:
        # Mode 2: Merge multiple models
        path = Path(args.path)
        model_list = []
        for d in path.glob("*/"):
            if args.out_name in d.name or "logs" in d.name:
                continue
            model_list.append(d.name)
            print(d.name)
        merge_datasets(model_list, path=args.path, out_name=args.out_name)
