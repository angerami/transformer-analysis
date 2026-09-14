#!/usr/bin/env python3
"""
One-time prefetch: download llama3.2, gemma2, and smollm2 families to the
cache volume before the pipeline runs them.

Usage:
    HF_CACHE_PATH=/Volumes/Flux/Projects/transformer-analysis/downloads \
        python scripts/prefetch_models.py

The cache layout mirrors head_pipeline.py: {cache_dir}/{model_key}/main/
Gemma 2 and Llama 3.2 are gated — requires a valid HF token in
~/.cache/huggingface/token or HUGGING_FACE_HUB_TOKEN env var.
"""

import os
import sys
from huggingface_hub import snapshot_download

CACHE_DIR = os.environ.get(
    "HF_CACHE_PATH",
    "/Volumes/Flux/Projects/transformer-analysis/downloads",
)

MODELS = {
    # key: (repo_id, allow_patterns)
    "llama3.2-1b": ("meta-llama/Llama-3.2-1B",              ["*.safetensors", "model.safetensors.index.json", "config.json"]),
    "llama3.2-3b": ("meta-llama/Llama-3.2-3B",              ["*.safetensors", "model.safetensors.index.json", "config.json"]),
    "gemma2-2b":   ("google/gemma-2-2b",                     ["*.safetensors", "model.safetensors.index.json", "config.json"]),
    "gemma2-9b":   ("google/gemma-2-9b",                     ["*.safetensors", "model.safetensors.index.json", "config.json"]),
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M-Instruct", ["*.safetensors", "model.safetensors.index.json", "config.json"]),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M-Instruct", ["*.safetensors", "model.safetensors.index.json", "config.json"]),
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", ["*.safetensors", "model.safetensors.index.json", "config.json"]),
}

ok, failed = [], []

for key, (repo_id, allow_patterns) in MODELS.items():
    cache_path = os.path.join(CACHE_DIR, key, "main")
    print(f"\n{'='*60}")
    print(f"  {key}  ({repo_id})")
    print(f"  → {cache_path}")
    print(f"{'='*60}")
    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_path,
            allow_patterns=allow_patterns,
        )
        print(f"  OK")
        ok.append(key)
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        failed.append((key, e))

print(f"\n\nDone: {len(ok)} succeeded, {len(failed)} failed.")
if failed:
    for key, e in failed:
        print(f"  {key}: {e}", file=sys.stderr)
    sys.exit(1)
