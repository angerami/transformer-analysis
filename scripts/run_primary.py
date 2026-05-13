#!/usr/bin/env python3
"""
Primary pipeline stage: download model → extract weights → compute stats → HF Dataset.
Logs parameters, timing, and artifact path to MLflow.

Examples:
    python scripts/run_primary.py --model gpt2 --out-dir outputs
    python scripts/run_primary.py --model pythia-70m-deduped --revision step8000 --out-dir outputs
"""

import argparse
import os
import time

import mlflow

from transformer_analysis.head_pipeline import process_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--cache-dir", default="./model_data")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--cleanup-downloads", action="store_true")
    p.add_argument("--mlflow-uri", default="file:./mlruns")
    p.add_argument("--mlflow-experiment", default="production")
    return p.parse_args()


def main():
    args = parse_args()
    revision = args.revision or None

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    rev_label = revision or "main"
    run_name = f"{args.model}-{rev_label}-primary"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": args.model,
            "revision": rev_label,
            "max_workers": args.max_workers,
            "device": args.device or "auto",
            "out_dir": args.out_dir,
        })

        t0 = time.time()
        process_model(
            model_name=args.model,
            revision=revision,
            out_dir=args.out_dir,
            cache_dir=args.cache_dir,
            cleanup_downloads=args.cleanup_downloads,
            max_workers=args.max_workers,
            device=args.device,
        )
        wall_time = time.time() - t0

        out_key = f"{args.model}_{revision}" if revision else args.model
        dataset_path = os.path.join(args.out_dir, out_key)

        mlflow.log_metric("wall_time_s", wall_time)
        mlflow.log_param("dataset_path", dataset_path)


if __name__ == "__main__":
    main()
