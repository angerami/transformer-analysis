#!/usr/bin/env python3
"""
Fetch model-card evaluation benchmarks for Pythia models from HuggingFace and
compare against any computed eval_metrics.parquet entries.

Two modes:
  1. Fetch HF model card eval results (LAMBADA, PIQA zero-shot) for each model
  2. Compare computed perplexity against paper_reference_loss.csv

Usage:
    python extract_paper_loss.py --fetch-hf
    python extract_paper_loss.py --compare --metrics outputs/eval_metrics/eval_metrics.parquet
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transformer_analysis.model_registry import MODEL_CONFIGS, get_model_config
from transformer_analysis.histogram_utils import PYTHIA_MODELS

REFERENCE_CSV = Path(__file__).parent / "paper_reference_loss.csv"

# HuggingFace evaluation results endpoint for a model card
HF_EVAL_URL = "https://huggingface.co/api/models/{repo_id}?full=true"

BENCHMARK_KEYS = ["harness|lambada_openai|0", "harness|piqa|0", "harness|arc_easy|0"]


def fetch_hf_evals(model_name: str) -> dict:
    """Fetch zero-shot benchmark scores from the HuggingFace model card."""
    cfg = get_model_config(model_name)
    url = HF_EVAL_URL.format(repo_id=cfg.repo_id)
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        results = {}
        for entry in data.get("cardData", {}).get("model-index", []):
            for result in entry.get("results", []):
                task = result.get("task", {}).get("name", "")
                for metric in result.get("metrics", []):
                    key = f"{task}_{metric.get('name', '')}"
                    results[key] = metric.get("value")
        # Also check eval_results field
        for er in data.get("evalResults", []):
            task_id = er.get("task", {}).get("taskId", "")
            results[task_id] = er.get("metrics", {}).get("acc", None)
        return results
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-hf", action="store_true",
                        help="Fetch HF model card benchmark results for all Pythia models")
    parser.add_argument("--compare", action="store_true",
                        help="Compare computed eval_metrics.parquet against paper_reference_loss.csv")
    parser.add_argument("--metrics", type=str,
                        default="outputs/eval_metrics/eval_metrics.parquet")
    args = parser.parse_args()

    if args.fetch_hf:
        print("Fetching HuggingFace model card eval results for Pythia models...\n")
        rows = []
        for model_name in PYTHIA_MODELS:
            try:
                get_model_config(model_name)
            except ValueError:
                continue
            results = fetch_hf_evals(model_name)
            print(f"{model_name}: {results}")
            for k, v in results.items():
                if v is not None:
                    rows.append({"model": model_name, "revision": "main",
                                 "step": 143000, "metric": k, "value": v,
                                 "source": "hf_model_card", "corpus": None})
        if rows:
            out_path = "outputs/eval_metrics/hf_benchmarks.parquet"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            print(f"\nSaved {len(rows)} benchmark entries → {out_path}")

    if args.compare:
        if not os.path.exists(args.metrics):
            print(f"No eval_metrics file found at {args.metrics}")
            return
        if not REFERENCE_CSV.exists():
            print(f"No reference CSV at {REFERENCE_CSV}")
            return

        computed = pd.read_parquet(args.metrics)
        reference = pd.read_csv(REFERENCE_CSV)

        print("=== Comparison: computed vs. paper reference ===\n")
        print(f"{'Model':<30} {'Step':>8} {'Ref loss':>10} {'Computed NLL':>13} {'Δ':>8}")
        print("-" * 75)
        for _, ref_row in reference.iterrows():
            comp_rows = computed[
                (computed["model"] == ref_row["model"]) &
                (computed["step"] == ref_row["step"]) &
                (computed["metric"] == "nll")
            ]
            if comp_rows.empty:
                comp_nll = float("nan")
                delta = float("nan")
            else:
                comp_nll = comp_rows["value"].iloc[0]
                delta = comp_nll - ref_row["value"]
            print(f"{ref_row['model']:<30} {int(ref_row['step']):>8} "
                  f"{ref_row['value']:>10.4f} {comp_nll:>13.4f} {delta:>8.4f}")


if __name__ == "__main__":
    main()
