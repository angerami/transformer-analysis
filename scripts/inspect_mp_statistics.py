#!/usr/bin/env python3
"""Inspect and summarize MP statistics data file."""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Inspect MP statistics data")
    parser.add_argument("datafile", help="Path to mp_statistics.npz file")
    parser.add_argument("--model", help="Show detailed stats for a specific model")
    parser.add_argument("--compare", nargs="+", help="Compare predictions across models")
    args = parser.parse_args()

    data = np.load(args.datafile, allow_pickle=True)
    model_names = [k for k in data.files]

    print(f"MP Statistics Data File: {args.datafile}")
    print(f"Models: {len(model_names)}")
    print()

    if args.model:
        if args.model not in model_names:
            print(f"Model {args.model} not found. Available: {', '.join(model_names)}")
            return

        model = data[args.model].item()
        print(f"Model: {args.model}")
        print(f"  d_model: {model['d_model']}")
        print(f"  d_head: {model['d_head']}")
        print(f"  n_heads: {model['n_heads']}")
        print(f"  n_layers: {model['n_layers']}")
        print(f"  gamma: {model['d_head'] / model['d_model']:.4f}")
        print()
        print("MC Statistics (mean ± std):")
        for stat_name, values in model['stats'].items():
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {stat_name:>35s}: {mean:10.3f} ± {std:8.3f}")
        print()
        print("MP Predictions:")
        for k, v in model['mp_predictions'].items():
            print(f"  {k:>35s}: {v:10.3f}")
        print()
        print("PME (Product Multiplicative Ensemble) Predictions:")
        for k, v in model['pme_predictions'].items():
            print(f"  {k:>35s}: {v:10.3f}")

    elif args.compare:
        models_to_compare = [m for m in args.compare if m in model_names]
        if not models_to_compare:
            print(f"None of the requested models found. Available: {', '.join(model_names)}")
            return

        print(f"{'Model':<20s} {'d_model':>8s} {'d_head':>7s} {'gamma':>7s} " +
              f"{'PR (MC)':>10s} {'PR (MP)':>10s} {'PR (PME)':>10s}")
        print("-" * 93)
        for model_name in models_to_compare:
            model = data[model_name].item()
            pr_mc = np.mean(model['stats']['participation_ratio'])
            pr_mp = model['mp_predictions']['participation_ratio']
            pr_pme = model['pme_predictions']['participation_ratio']
            gamma = model['d_head'] / model['d_model']
            print(f"{model_name:<20s} {model['d_model']:8d} {model['d_head']:7d} {gamma:7.4f} " +
                  f"{pr_mc:10.3f} {pr_mp:10.3f} {pr_pme:10.3f}")

    else:
        print(f"{'Model':<20s} {'d_model':>8s} {'d_head':>7s} {'n_heads':>8s} " +
              f"{'n_layers':>9s} {'gamma':>7s} {'n_samples':>10s}")
        print("-" * 90)
        for model_name in sorted(model_names):
            model = data[model_name].item()
            n_samples = len(model['stats']['max'])
            gamma = model['d_head'] / model['d_model']
            print(f"{model_name:<20s} {model['d_model']:8d} {model['d_head']:7d} " +
                  f"{model['n_heads']:8d} {model['n_layers']:9d} {gamma:7.4f} {n_samples:10d}")


if __name__ == "__main__":
    main()
