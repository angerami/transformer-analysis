#!/usr/bin/env python3
"""Thin CLI wrapper for eval_metrics. Called by rules/eval.smk."""

import argparse
import os
import sys
import time
from pathlib import Path

import hashlib
import json

import mlflow
from mlflow.entities import Dataset as DatasetEntity, DatasetInput
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
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "bf16", "fp16", "fp32"],
                        help="Weight dtype. 'auto' uses the model's saved dtype (bf16/fp16 for most modern models, fp32 for GPT-2). Override with fp32 for a numerics reference.")
    parser.add_argument("--mlflow-uri", default="file:./mlruns")
    parser.add_argument("--mlflow-experiment", default="production")

    args = parser.parse_args()
    revision = args.revision or None
    rev_label = revision or "main"

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name=f"{args.model}-{rev_label}-eval"):
        mlflow.set_tag("mlflow.note.content", "Evaluate model perplexity on corpus → parquet")
        mlflow.set_tag("model", args.model)
        mlflow.log_params({
            "model": args.model,
            "revision": rev_label,
            "corpus": args.corpus,
            "pile_tokens": args.pile_tokens,
            "stride": args.stride,
            "dtype": args.dtype,
            "device": args.device or "auto",
            "out": args.out,
        })

        t0 = time.time()
        result = evaluate_model(
            model_name=args.model, revision=revision,
            corpus=args.corpus, pile_tokens=args.pile_tokens,
            cache_dir=args.cache, device_str=args.device,
            stride=args.stride, max_tokens=args.max_tokens,
            pile_cache=args.pile_cache, dtype=args.dtype,
        )
        wall_time = time.time() - t0

        print(f"  ppl={result['perplexity']:.2f}  bpb={result['bpb']:.4f}")
        mlflow.log_metrics({
            "perplexity": result["perplexity"],
            "nll": result["nll"],
            "bpb": result["bpb"],
            "wall_time_s": wall_time,
        })

        df = pd.DataFrame(to_long_format(result))
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"  Saved {len(df)} rows → {args.out}")
        digest = hashlib.md5(args.out.encode()).hexdigest()[:8]
        entity = DatasetEntity(
            name=f"{args.model}-{rev_label}-eval",
            digest=digest,
            source_type="local",
            source=json.dumps({"uri": args.out}),
        )
        mlflow.MlflowClient().log_inputs(
            mlflow.active_run().info.run_id,
            [DatasetInput(dataset=entity, tags=[])],
        )


if __name__ == "__main__":
    main()
