#!/usr/bin/env python3
"""Thin CLI wrapper for eval_metrics. Called by rules/eval.smk."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transformer_analysis.eval_metrics import evaluate_model, to_long_format


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--corpus", type=str, default="wikitext103",
                        choices=["wikitext103", "pile"])
    parser.add_argument("--pile-tokens", type=int, default=204800)
    parser.add_argument("--pile-cache", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--cache", type=str, default="model_cache")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])

    args = parser.parse_args()

    result = evaluate_model(
        model_name=args.model, revision=args.revision or None,
        corpus=args.corpus, pile_tokens=args.pile_tokens,
        cache_dir=args.cache, device_str=args.device,
        stride=args.stride, max_tokens=args.max_tokens,
        pile_cache=args.pile_cache,
    )
    print(f"  ppl={result['perplexity']:.2f}  bpb={result['bpb']:.4f}")

    df = pd.DataFrame(to_long_format(result))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"  Saved {len(df)} rows → {args.out}")


if __name__ == "__main__":
    main()
