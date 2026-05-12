#!/usr/bin/env python3
"""
Compute perplexity for transformer models on WikiText-103 or The Pile test split.
Outputs a parquet side table (eval_metrics.parquet) joinable onto weight analysis
datasets on (model, revision).

Usage:
    python compute_perplexity.py --model gpt2
    python compute_perplexity.py --model pythia-70m-deduped --all-revisions --corpus pile --pile-tokens 51200
    python compute_perplexity.py --all-models --corpus wikitext103

Output schema: model, revision, step, metric, value, source, corpus
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transformer_analysis.model_registry import MODEL_CONFIGS, get_model_config
from transformer_analysis.device_utils import get_device


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus_tokens(corpus: str, tokenizer, pile_tokens: int = 204800,
                       pile_seed: int = 42) -> torch.Tensor:
    if corpus == "wikitext103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
    elif corpus == "pile":
        ds = load_dataset("EleutherAI/pile", split="test", streaming=True)
        tokens_collected = []
        for example in ds.shuffle(seed=pile_seed, buffer_size=1000):
            enc = tokenizer(example["text"], return_tensors="pt",
                            truncation=False, add_special_tokens=False)
            tokens_collected.append(enc.input_ids[0])
            if sum(len(t) for t in tokens_collected) >= pile_tokens:
                break
        all_tokens = torch.cat(tokens_collected)[:pile_tokens]
        return all_tokens
    else:
        raise ValueError(f"Unknown corpus: {corpus!r}. Choose 'wikitext103' or 'pile'.")

    encodings = tokenizer(text, return_tensors="pt", truncation=False,
                          add_special_tokens=False)
    return encodings.input_ids[0]


# ---------------------------------------------------------------------------
# Forward pass and collector pattern
# ---------------------------------------------------------------------------

def run_inference(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  output_hidden_states: bool = False,
                  output_attentions: bool = False):
    """
    Single forward pass. output_hidden_states / output_attentions are the hooks
    for future data-weighted statistics (e.g. <W_QK>_{data}):
      - output_hidden_states=True → out.hidden_states[layer] for dressed operators
      - output_attentions=True   → out.attentions[layer][head] for <A>_{data}
    """
    with torch.no_grad():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )


class NLLCollector:
    """Accumulates per-token negative log-likelihood for perplexity computation."""
    name = "nll"

    def __init__(self):
        self._total_nll = 0.0
        self._n_tokens = 0

    def update(self, out, n_tokens: int):
        # out.loss is mean NLL over non-masked tokens in the window
        self._total_nll += out.loss.item() * n_tokens
        self._n_tokens += n_tokens

    def result(self):
        if self._n_tokens == 0:
            return float("nan")
        return self._total_nll / self._n_tokens


def eval_loop(model, input_ids: torch.Tensor, device, stride: int = 512,
              max_tokens: Optional[int] = None,
              collectors=None) -> dict:
    """
    Sliding-window perplexity loop (canonical HuggingFace approach).
    Collectors accumulate statistics over all windows; extend by adding new
    Collector subclasses (e.g. DressedWQKCollector for <W_QK>_{data}).
    """
    if collectors is None:
        collectors = [NLLCollector()]

    max_length = model.config.max_position_embeddings
    seq_len = min(len(input_ids), max_tokens or len(input_ids))
    input_ids = input_ids[:seq_len].unsqueeze(0).to(device)

    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end
        window = input_ids[:, begin:end]
        mask = torch.ones_like(window)

        out = run_inference(model, window, mask)
        for collector in collectors:
            collector.update(out, target_len)

        prev_end = end
        if end == seq_len:
            break

    return {c.name: c.result() for c in collectors}


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_name: str, revision: Optional[str],
                   corpus: str, pile_tokens: int, cache_dir: str,
                   device_str: Optional[str], stride: int = 512,
                   max_tokens: Optional[int] = None) -> dict:
    model_config = get_model_config(model_name)
    revision_str = revision or "main"

    print(f"  Downloading {model_name} @ {revision_str} ...")
    cache_path = snapshot_download(
        repo_id=model_config.repo_id,
        revision=revision,
        cache_dir=f"{cache_dir}/{model_name}/{revision_str}",
        allow_patterns=["*.safetensors", "*.bin", "*.json", "tokenizer*"],
        resume_download=True,
    )

    device = torch.device(device_str or get_device())
    print(f"  Loading model on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(cache_path)
    model = AutoModelForCausalLM.from_pretrained(cache_path, torch_dtype=torch.float32)
    model = model.to(device).eval()

    print(f"  Loading corpus ({corpus}) ...")
    tokens = load_corpus_tokens(corpus, tokenizer, pile_tokens=pile_tokens)
    print(f"  Evaluating on {len(tokens):,} tokens ...")

    results = eval_loop(model, tokens, device, stride=stride, max_tokens=max_tokens)
    nll = results["nll"]
    ppl = math.exp(nll)
    bpb = nll / math.log(2)

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    step = None
    if revision and revision.startswith("step"):
        try:
            step = int(revision[4:])
        except ValueError:
            pass

    return {
        "model": model_name,
        "revision": revision_str,
        "step": step,
        "perplexity": ppl,
        "nll": nll,
        "bpb": bpb,
        "corpus": corpus,
        "n_tokens": len(tokens),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def to_long_format(row: dict) -> list[dict]:
    """Convert one evaluation result row into long-format (model, revision, step, metric, value, source, corpus)."""
    base = {"model": row["model"], "revision": row["revision"],
            "step": row["step"], "source": "eval_pass", "corpus": row["corpus"]}
    return [
        {**base, "metric": "perplexity", "value": row["perplexity"]},
        {**base, "metric": "nll", "value": row["nll"]},
        {**base, "metric": "bpb", "value": row["bpb"]},
    ]


def append_to_parquet(rows: list[dict], out_path: str):
    new_df = pd.DataFrame(rows)
    if os.path.exists(out_path):
        existing = pd.read_parquet(out_path)
        # Drop any existing rows for (model, revision, corpus) we're replacing
        key = ["model", "revision", "corpus"]
        mask = existing[key].apply(tuple, axis=1).isin(
            new_df[key].apply(tuple, axis=1).unique()
        )
        existing = existing[~mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"  Saved {len(new_df)} rows → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute perplexity for transformer models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all-models", action="store_true")

    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--all-revisions", action="store_true")
    parser.add_argument("--corpus", type=str, default="wikitext103",
                        choices=["wikitext103", "pile"])
    parser.add_argument("--pile-tokens", type=int, default=204800,
                        help="Token count for Pile subset (default: 200K)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Cap total tokens evaluated (default: all)")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--out", type=str, default="outputs/eval_metrics/eval_metrics.parquet")
    parser.add_argument("--cache", type=str, default="./model_data")
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
                )
                print(f"  ppl={result['perplexity']:.2f}  bpb={result['bpb']:.4f}")
                rows.extend(to_long_format(result))
            except Exception as e:
                print(f"  ERROR {model_name} @ {rev}: {e}")

        if rows:
            append_to_parquet(rows, args.out)


if __name__ == "__main__":
    main()
