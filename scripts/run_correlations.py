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
import subprocess
import time

import hashlib
import json

import mlflow
from mlflow.entities import Dataset as DatasetEntity, DatasetInput

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
    run_name = f"{args.model}-{rev_label}-correlations"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("mlflow.note.content", "Compute head-head correlation matrices from model weights → .npz")
        mlflow.set_tag("model", args.model)
        mlflow.log_params({
            "model": args.model,
            "revision": rev_label,
            "circuits": ",".join(args.circuits),
            "metrics": ",".join(args.metrics),
            "max_workers": args.max_workers,
            "out_dir": args.out_dir,
            "git_sha": _git_sha(),
        })

        t0 = time.time()
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
        mlflow.log_metric("wall_time_s", time.time() - t0)
        digest = hashlib.md5(args.out_dir.encode()).hexdigest()[:8]
        entity = DatasetEntity(
            name=f"{args.model}-{rev_label}-correlations",
            digest=digest,
            source_type="local",
            source=json.dumps({"uri": args.out_dir}),
        )
        mlflow.MlflowClient().log_inputs(
            mlflow.active_run().info.run_id,
            [DatasetInput(dataset=entity, tags=[])],
        )


if __name__ == "__main__":
    main()
