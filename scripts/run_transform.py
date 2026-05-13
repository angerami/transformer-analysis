#!/usr/bin/env python3
"""
Transform pipeline stage: re-run derived metrics on an existing primary HF Dataset.
Safe to re-run repeatedly ("patch") without re-downloading model weights.
Each run gets its own MLflow entry so different metric versions can be compared.

Examples:
    python scripts/run_transform.py --model gpt2 --dataset-dir outputs/gpt2
    python scripts/run_transform.py --model pythia-70m-deduped --revision step8000 \
        --dataset-dir outputs/pythia-70m-deduped_step8000
"""

import argparse
import subprocess
import time

import mlflow

from transformer_analysis.head_pipeline import reprocess_metrics
from transformer_analysis.head_metrics import normality_metrics, singular_value_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out-dir", default=None,
                   help="Output path for refined dataset (default: {dataset-dir}_refined)")
    p.add_argument("--drop-columns", nargs="*", default=[],
                   help="Column names to drop from the refined dataset")
    p.add_argument("--mlflow-uri", default="file:./mlruns")
    p.add_argument("--mlflow-experiment", default="production")
    return p.parse_args()


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    args = parse_args()
    revision = args.revision or None

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    rev_label = revision or "main"
    run_name = f"{args.model}-{rev_label}-transform"

    metrics_applied = sorted(list(normality_metrics.keys()) + list(singular_value_metrics.keys()))

    import os
    dataset_dir = args.dataset_dir
    out_dir = args.out_dir or f"{dataset_dir}_refined"
    in_dir = os.path.dirname(dataset_dir) or "."

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": args.model,
            "revision": rev_label,
            "dataset_dir": dataset_dir,
            "out_dir": out_dir,
            "metrics_applied": ",".join(metrics_applied),
            "drop_columns": ",".join(args.drop_columns) if args.drop_columns else "",
            "git_sha": _git_sha(),
        })

        t0 = time.time()
        reprocess_metrics(
            model_name=args.model,
            revision=revision,
            in_dir=in_dir,
            out_dir=out_dir,
            drop_columns=args.drop_columns or None,
        )
        wall_time = time.time() - t0

        mlflow.log_metric("wall_time_s", wall_time)


if __name__ == "__main__":
    main()
