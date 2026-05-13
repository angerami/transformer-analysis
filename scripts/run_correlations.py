#!/usr/bin/env python3
"""
Correlation analysis stage: compute head-head correlation matrices from model weights.
Outputs .npz files per circuit to --out-dir.

Examples:
    python scripts/run_correlations.py --model gpt2 --out-dir correlations
    python scripts/run_correlations.py --model pythia-70m-deduped --revision step8000 \
        --circuits QK OV --out-dir correlations
"""

import argparse

from transformer_analysis.pair_pipeline import run_multi_circuit_analysis


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="")
    p.add_argument("--out-dir", default="correlations")
    p.add_argument("--cache-dir", default="./model_data")
    p.add_argument("--circuits", nargs="+", default=["QK"])
    p.add_argument("--metrics", nargs="+", default=["frob_cosine", "pearson_corr", "hist_jensen_shannon"])
    p.add_argument("--device", default=None)
    p.add_argument("--max-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    revision = args.revision or None

    run_multi_circuit_analysis(
        model_name=args.model,
        revision=revision,
        circuits=tuple(args.circuits),
        metrics=tuple(args.metrics),
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        device=args.device,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
