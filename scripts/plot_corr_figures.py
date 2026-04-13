#!/usr/bin/env python3
"""Plot correlations for all models found in a data directory.

Automatically discovers models from metadata files and generates plots for each.
Similar to run_all_correlations.py but for plotting instead of analysis.

Usage:
    python scripts/plot_all_models.py --data corr_out
    python scripts/plot_all_models.py --data corr_out --skip gpt2
    python scripts/plot_all_models.py --data corr_out --models gpt2 gpt2-large
    python scripts/plot_all_models.py --data corr_out --components weights  # or biases
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from collections import defaultdict


def find_models_and_components(data_dir):
    """Find all models and their components from metadata files.

    Returns:
        dict: {model_name: [(revision, component), ...]}
              where component is like 'W_QK', 'W_OV', 'b_Q', etc.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return {}

    models = defaultdict(list)

    # Pattern: {model}_{revision}_{component}_metadata.json
    # Examples:
    #   gpt2_main_W_QK_metadata.json
    #   gpt2_main_b_Q_metadata.json
    #   pythia-70m-deduped_main_W_OV_metadata.json

    for metadata_file in data_path.glob("*_metadata.json"):
        filename = metadata_file.name

        # Skip cross-correlation files (contain _vs_)
        if "_vs_" in filename:
            continue

        # Parse filename
        parts = filename.replace("_metadata.json", "").split("_")

        # Find the component (W_QK, W_OV, b_Q, etc.)
        component = None
        for i, part in enumerate(parts):
            if part in ["W", "b"] and i + 1 < len(parts):
                component = f"{part}_{parts[i + 1]}"
                component_idx = i
                break

        if not component:
            continue

        # Everything before component is model + revision
        model_revision = "_".join(parts[:component_idx])

        # Last part before component is usually revision
        if parts[component_idx - 1] in ["main", "step0", "step1000"]:
            revision = parts[component_idx - 1]
            model = "_".join(parts[:component_idx - 1])
        else:
            revision = "main"
            model = model_revision

        models[model].append((revision, component))

    return dict(models)


def categorize_component(component):
    """Categorize component as 'weight' or 'bias'."""
    if component.startswith("W_"):
        return "weight"
    elif component.startswith("b_"):
        return "bias"
    return "unknown"


def plot_model_component(data_dir, model, revision, component, out_dir, quiet=False):
    """Run plot_correlations.py for a specific model/component."""
    # Determine weight_type parameter (legacy parameter name)
    weight_type = component

    cmd = [
        sys.executable,
        "scripts/plot_correlations.py",
        "--data", data_dir,
        "--model", model,
        "--revision", revision,
        "--weight-type", weight_type,
        "--out", out_dir,
    ]

    if not quiet:
        print(f"  Plotting: {model} @ {revision} - {component}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=quiet,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        if not quiet:
            print(f"    ERROR: {e}")
            if e.stderr:
                print(f"    {e.stderr}")
        return False
    except Exception as e:
        if not quiet:
            print(f"    ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Plot correlations for all models in data directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all models
  python scripts/plot_all_models.py --data corr_out

  # Plot specific models only
  python scripts/plot_all_models.py --data corr_out --models gpt2 gpt2-large

  # Skip certain models
  python scripts/plot_all_models.py --data corr_out --skip gpt2

  # Plot only weights (no biases)
  python scripts/plot_all_models.py --data corr_out --components weights

  # Plot only biases
  python scripts/plot_all_models.py --data corr_out --components biases

  # Quiet mode (less output)
  python scripts/plot_all_models.py --data corr_out --quiet
        """
    )

    parser.add_argument(
        "--data", type=str, default="corr_out",
        help="Data directory containing correlation results (default: corr_out)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output directory for figures (default: {data}/figures)"
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Specific models to plot (default: all found models)"
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Models to skip"
    )
    parser.add_argument(
        "--components", choices=["weights", "biases", "all"], default="all",
        help="Which components to plot (default: all)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress detailed output"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be plotted without plotting"
    )
    parser.add_argument(
        "--build-dataset", type=str, default=None, metavar="REPO",
        help="After plotting, build and push HF dataset "
             "(e.g. user/transformer-analysis-figures)"
    )

    args = parser.parse_args()

    # Default output directory
    out_dir = args.out or os.path.join(args.data, "figures")

    # Find models
    if not args.quiet:
        print(f"Scanning directory: {args.data}")

    models_components = find_models_and_components(args.data)

    if not models_components:
        print(f"No models found in {args.data}")
        print("Make sure the directory contains *_metadata.json files")
        return 1

    # Filter models
    if args.models:
        models_to_plot = {
            m: c for m, c in models_components.items()
            if m in args.models
        }
    else:
        models_to_plot = models_components

    # Skip models
    if args.skip:
        models_to_plot = {
            m: c for m, c in models_to_plot.items()
            if m not in args.skip
        }

    if not models_to_plot:
        print("No models to plot after filtering")
        return 1

    # Count components
    total_components = sum(len(components) for components in models_to_plot.values())

    # Filter by component type
    if args.components != "all":
        # Map plural to singular
        component_type_map = {
            "weights": "weight",
            "biases": "bias"
        }
        component_type = component_type_map.get(args.components, args.components)

        filtered_models = {}
        for model, components in models_to_plot.items():
            filtered = [
                (rev, comp) for rev, comp in components
                if categorize_component(comp) == component_type
            ]
            if filtered:
                filtered_models[model] = filtered
        models_to_plot = filtered_models

    # Recount after filtering
    filtered_components = sum(len(components) for components in models_to_plot.values())

    # Print summary
    print("\n" + "=" * 70)
    print(f"Found {len(models_to_plot)} models with {filtered_components} components:")
    print("-" * 70)

    for model, components in sorted(models_to_plot.items()):
        if components:
            comp_strs = []
            for rev, comp in sorted(set(components)):
                comp_type = "W" if comp.startswith("W_") else "b"
                comp_strs.append(f"{comp}")

            print(f"  {model:<30} {len(components):>2} components: {', '.join(sorted(set(comp_strs)))}")

    print("=" * 70)

    if args.dry_run:
        print("\nDry run - exiting without plotting")
        return 0

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Plot each model/component
    print(f"\nOutput directory: {out_dir}\n")

    success_count = 0
    fail_count = 0

    for i, (model, components) in enumerate(sorted(models_to_plot.items()), 1):
        if not components:
            continue

        print(f"[{i}/{len(models_to_plot)}] {model}")

        for revision, component in sorted(set(components)):
            if plot_model_component(
                args.data, model, revision, component, out_dir, args.quiet
            ):
                success_count += 1
            else:
                fail_count += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"Plotting complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Output:  {out_dir}")
    print("=" * 70)

    # Multi-model comparison plots
    if success_count > 1:
        print("\nGenerating multi-model comparison plots...")
        try:
            from plot_correlations import (plot_eigenvalue_comparison,
                                            plot_eigen_stats_comparison)
            model_list = sorted(models_to_plot.keys())
            # One comparison plot per weight type that all models share
            all_wts = set()
            for components in models_to_plot.values():
                for _, comp in components:
                    all_wts.add(comp)
            for wt in sorted(all_wts):
                try:
                    plot_eigenvalue_comparison(
                        args.data, model_list, weight_type=wt,
                        out_dir=out_dir)
                except Exception as e:
                    print(f"  *** Error on {wt} eigenvalues: {e}")
                try:
                    plot_eigen_stats_comparison(
                        args.data, model_list, weight_type=wt,
                        out_dir=out_dir)
                except Exception as e:
                    print(f"  *** Error on {wt} eigen stats: {e}")
        except ImportError as e:
            print(f"  *** Could not import comparison plotter: {e}")

    # Build HF dataset if requested
    if args.build_dataset and success_count > 0:
        print(f"\nBuilding HF dataset → {args.build_dataset}")
        try:
            from build_hf_dataset import build_dataset
            ds = build_dataset(out_dir)
            ds.push_to_hub(args.build_dataset)
            print(f"Pushed: https://huggingface.co/datasets/{args.build_dataset}")
        except Exception as e:
            print(f"  *** Dataset build failed: {e}")

    # Reminder to regenerate viewer
    if success_count > 0:
        print("\nTo view in browser, regenerate the viewer index:")
        print(f"  python scripts/generate_viewer_index.py --out {args.data} --serve")
        print("=" * 70)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
