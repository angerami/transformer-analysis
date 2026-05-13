#!/usr/bin/env python3
"""Thin CLI wrapper for eval_metrics. Called by rules/eval.smk."""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transformer_analysis.model_registry import MODEL_CONFIGS, get_model_config
from transformer_analysis.eval_metrics import evaluate_model, to_long_format, append_to_parquet


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all-models", action="store_true")

    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--all-revisions", action="store_true")
    parser.add_argument("--corpus", type=str, default="wikitext103",
                        choices=["wikitext103", "pile"])
    parser.add_argument("--pile-tokens", type=int, default=204800)
    parser.add_argument("--pile-cache", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--out", type=str, default="outputs/eval_metrics/eval_metrics.parquet")
    parser.add_argument("--cache", type=str, default="model_cache")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])

    args = parser.parse_args()

    models = list(MODEL_CONFIGS.keys()) if args.all_models else [args.model]

    for model_name in models:
        try:
            model_config = get_model_config(model_name)
        except ValueError as e:
            print(f"Skipping {model_name}: {e}")
            continue

        if args.all_revisions:
            revisions = model_config.revisions or [None]
        elif args.revision:
            revisions = [args.revision]
        else:
            revisions = [None]

        print(f"\n{'='*60}\n{model_name} — {len(revisions)} revision(s)\n{'='*60}")
        rows = []
        for rev in tqdm(revisions, desc=model_name):
            try:
                result = evaluate_model(
                    model_name=model_name, revision=rev,
                    corpus=args.corpus, pile_tokens=args.pile_tokens,
                    cache_dir=args.cache, device_str=args.device,
                    stride=args.stride, max_tokens=args.max_tokens,
                    pile_cache=args.pile_cache,
                )
                print(f"  ppl={result['perplexity']:.2f}  bpb={result['bpb']:.4f}")
                rows.extend(to_long_format(result))
            except Exception as e:
                print(f"  ERROR {model_name} @ {rev}: {e}")

        if rows:
            append_to_parquet(rows, args.out)


if __name__ == "__main__":
    main()
