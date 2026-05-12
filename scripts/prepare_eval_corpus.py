#!/usr/bin/env python3
"""
One-time download of a fixed text sample from The Pile test split.
Run this once; subsequent perplexity evaluations load from the saved file.

Usage:
    python scripts/prepare_eval_corpus.py --size-mb 500
    python scripts/prepare_eval_corpus.py --size-mb 2000 --out outputs/eval_corpus/pile_2gb.jsonl.gz

The output is a gzipped JSONL file (one {"text": "..."} per line).
compute_perplexity.py will load this automatically when --pile-cache is set.
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=float, default=500,
                        help="Approximate uncompressed target size in MB (default: 500)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (default: outputs/eval_corpus/pile_{size}mb.jsonl.gz)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    size_bytes = int(args.size_mb * 1024 * 1024)
    out_path = args.out or f"outputs/eval_corpus/pile_{int(args.size_mb)}mb.jsonl.gz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        existing_size = os.path.getsize(out_path) / 1024**2
        print(f"Already exists: {out_path} ({existing_size:.1f} MB compressed). Delete to re-download.")
        return

    print(f"Streaming Pile test split, collecting ~{args.size_mb:.0f} MB of text ...")
    ds = load_dataset("EleutherAI/pile", split="test", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=2000)

    collected_bytes = 0
    n_docs = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        with tqdm(unit="MB", unit_scale=True, desc="Collecting") as pbar:
            for example in ds:
                text = example["text"]
                line = json.dumps({"text": text}) + "\n"
                f.write(line)
                chunk = len(text.encode("utf-8"))
                collected_bytes += chunk
                n_docs += 1
                pbar.update(chunk / 1024**2)
                if collected_bytes >= size_bytes:
                    break

    compressed_mb = os.path.getsize(out_path) / 1024**2
    print(f"\nSaved {n_docs:,} documents ({collected_bytes/1024**2:.1f} MB raw text, "
          f"{compressed_mb:.1f} MB compressed) → {out_path}")
    print(f"\nUse with: python scripts/compute_perplexity.py --corpus pile --pile-cache {out_path}")


if __name__ == "__main__":
    main()
