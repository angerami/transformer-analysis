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

    # Archive inputs before deleting them
    python merge_datasets.py --model pythia-70m-deduped --path outputs --archive --delete
"""

import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from transformer_analysis.weight_analysis import merge_versions, merge_datasets


def archive_datasets(dataset_paths: list[Path], archive_name: str, base_path: Path) -> Path:
    """
    Bundle dataset directories into a compressed tar archive.

    Args:
        dataset_paths: List of dataset directory paths to archive
        archive_name: Base name for the archive file
        base_path: Base directory where archive will be created

    Returns:
        Path to the created archive file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = base_path / f"{archive_name}_inputs_{timestamp}.tar.gz"

    print(f"\nCreating archive: {archive_path}")
    with tarfile.open(archive_path, "w:gz") as tar:
        for dataset_path in dataset_paths:
            if dataset_path.exists():
                print(f"  Adding: {dataset_path.name}")
                tar.add(dataset_path, arcname=dataset_path.name)

    print(f"Archive created successfully: {archive_path}")
    return archive_path


def delete_datasets(dataset_paths: list[Path]) -> None:
    """
    Delete dataset directories.

    Args:
        dataset_paths: List of dataset directory paths to delete
    """
    print("\nDeleting input datasets:")
    for dataset_path in dataset_paths:
        if dataset_path.exists():
            print(f"  Deleting: {dataset_path}")
            shutil.rmtree(dataset_path)
    print("Input datasets deleted successfully")


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
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Bundle input datasets into a tar.gz archive before deletion",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete input datasets after merging (use with --archive to preserve)",
    )
    args = parser.parse_args()

    base_path = Path(args.path)
    input_datasets = []

    if args.model is not None:
        # Mode 1: Merge all versions of a single model
        # The output directory will be named: {model_name}_{suffix}
        output_dir_name = f"{args.model}_{args.suffix}"

        # Collect input datasets (all versions/checkpoints of the model)
        # BEFORE merging to avoid including the output directory
        for d in base_path.glob("*/"):
            if d.name.startswith(args.model) and d.name != output_dir_name:
                input_datasets.append(d)

        merge_versions(model_name=args.model, path=args.path, suffix=args.suffix)
        archive_name = output_dir_name
    else:
        # Mode 2: Merge multiple models
        # The output directory will be named: {out_name}
        output_dir_name = args.out_name

        # Collect input datasets (all model directories except output and logs)
        model_list = []
        for d in base_path.glob("*/"):
            # Exclude the output directory, logs, and any archive directories
            if d.name == output_dir_name or "logs" in d.name:
                continue
            model_list.append(d.name)
            input_datasets.append(d)
            print(d.name)

        merge_datasets(model_list, path=args.path, out_name=args.out_name)
        archive_name = output_dir_name

    # Post-merge operations: archive and/or delete
    if args.archive and input_datasets:
        archive_datasets(input_datasets, archive_name, base_path)

    if args.delete and input_datasets:
        if not args.archive:
            print("\nWARNING: Deleting inputs without archiving!")
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                print("Deletion cancelled")
                exit(0)
        delete_datasets(input_datasets)
