#!/usr/bin/env python3
"""Build a HuggingFace dataset from generated figures.

Each row stores the PNG image + parsed metadata (model, component,
plot_type, metric).  The dataset can be pushed to the Hub for use
by the Streamlit viewer on HF Spaces.

Usage:
    python scripts/build_hf_dataset.py --figures corr_out/figures
    python scripts/build_hf_dataset.py --figures corr_out/figures --push user/transformer-analysis-figures
    python scripts/build_hf_dataset.py --figures corr_out/figures --save-local dataset/
"""

import argparse
import os
from pathlib import Path

from datasets import Dataset, Features, Image, Value

from generate_viewer_index import parse_filename


def build_dataset(figures_dir):
    """Build a HF Dataset from PNGs in figures_dir."""
    figures_dir = Path(figures_dir)
    paths = sorted(figures_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNGs found in {figures_dir}")

    filenames, models, components, plot_types, metrics, cross_pairs = (
        [], [], [], [], [], [],
    )
    str_paths = []

    for p in paths:
        meta = parse_filename(p.name)
        str_paths.append(str(p))
        filenames.append(p.name)
        models.append(meta["model"])
        components.append(meta["component"])
        plot_types.append(meta["plot_type"])
        metrics.append(meta["metric"])
        cross_pairs.append(meta.get("cross_pair", ""))

    ds = Dataset.from_dict(
        {
            "image": str_paths,
            "filename": filenames,
            "model": models,
            "component": components,
            "plot_type": plot_types,
            "metric": metrics,
            "cross_pair": cross_pairs,
        },
        features=Features({
            "image": Image(),
            "filename": Value("string"),
            "model": Value("string"),
            "component": Value("string"),
            "plot_type": Value("string"),
            "metric": Value("string"),
            "cross_pair": Value("string"),
        }),
    )

    print(f"Built dataset: {len(ds)} images")
    print(f"  Models:     {sorted(set(models))}")
    print(f"  Components: {sorted(set(components))}")
    print(f"  Plot types: {sorted(set(plot_types))}")
    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Build HF dataset from generated figures")
    parser.add_argument("--figures", type=str, default="corr_out/figures",
                        help="Directory containing PNG figures")
    parser.add_argument("--push", type=str, default=None,
                        help="Push to HF Hub repo (e.g. user/transformer-analysis-figures)")
    parser.add_argument("--save-local", type=str, default=None,
                        help="Save dataset locally (arrow format)")
    args = parser.parse_args()

    ds = build_dataset(args.figures)

    if args.save_local:
        os.makedirs(args.save_local, exist_ok=True)
        ds.save_to_disk(args.save_local)
        print(f"Saved to {args.save_local}")

    if args.push:
        print(f"Pushing to {args.push} ...")
        ds.push_to_hub(args.push)
        print(f"Done: https://huggingface.co/datasets/{args.push}")

    if not args.save_local and not args.push:
        print("\nDry run — pass --push REPO or --save-local DIR to persist.")
        print(f"Example: python {__file__} --figures {args.figures} "
              f"--push your-username/transformer-analysis-figures")


if __name__ == "__main__":
    main()
