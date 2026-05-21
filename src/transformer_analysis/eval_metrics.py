"""Eval-pass metrics: perplexity, NLL, BPB.

Structured around a collector pattern so future data-weighted statistics
(dressed operators, attention statistics, etc.) can be added as new
Collector subclasses without changing the eval loop.

Public API:
    evaluate_model(...)      → raw result dict
    to_long_format(row)      → list of long-format dicts for the parquet
    append_to_parquet(rows, path)
"""

import math
import os
import shutil
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from transformer_analysis.model_registry import MODEL_CONFIGS, get_model_config
from transformer_analysis.device_utils import get_device


# Map CLI dtype strings to torch dtypes. "auto" is passed through to
# from_pretrained, which picks the dtype the model was saved in.
_DTYPE_MAP = {
    "auto": "auto",
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def resolve_dtype(name: str):
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype {name!r}. Choose from {list(_DTYPE_MAP)}.")
    return _DTYPE_MAP[name]


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_pile_cache(pile_cache: str, tokenizer, pile_tokens: int) -> torch.Tensor:
    import gzip, json as _json
    tokens_collected = []
    n = 0
    opener = gzip.open if pile_cache.endswith(".gz") else open
    with opener(pile_cache, "rt", encoding="utf-8") as f:
        for line in f:
            text = _json.loads(line)["text"]
            enc = tokenizer(text, return_tensors="pt",
                            truncation=False, add_special_tokens=False)
            tokens_collected.append(enc.input_ids[0])
            n += len(tokens_collected[-1])
            if n >= pile_tokens:
                break
    if not tokens_collected:
        raise ValueError(f"pile_cache file appears empty: {pile_cache}")
    return torch.cat(tokens_collected)[:pile_tokens]


def load_corpus_tokens(corpus: str, tokenizer, pile_tokens: int = 204800,
                       pile_seed: int = 42, pile_cache: Optional[str] = None) -> torch.Tensor:
    if corpus == "wikitext103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
        encodings = tokenizer(text, return_tensors="pt", truncation=False,
                              add_special_tokens=False)
        return encodings.input_ids[0]
    elif corpus == "pile":
        if pile_cache:
            print(f"  Loading Pile corpus from cache: {pile_cache}")
            return load_pile_cache(pile_cache, tokenizer, pile_tokens)
        print("  No --pile-cache set; streaming from HuggingFace (slow for repeated runs).")
        print("  Run prepare_eval_corpus.py once to create a local cache.")
        ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        tokens_collected = []
        for example in ds.shuffle(seed=pile_seed, buffer_size=1000):
            enc = tokenizer(example["text"], return_tensors="pt",
                            truncation=False, add_special_tokens=False)
            tokens_collected.append(enc.input_ids[0])
            if sum(len(t) for t in tokens_collected) >= pile_tokens:
                break
        return torch.cat(tokens_collected)[:pile_tokens]
    else:
        raise ValueError(f"Unknown corpus: {corpus!r}. Choose 'wikitext103' or 'pile'.")


# ---------------------------------------------------------------------------
# Forward pass and collector pattern
# ---------------------------------------------------------------------------

def run_inference(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  output_hidden_states: bool = False,
                  output_attentions: bool = False):
    """Single forward pass. output_hidden_states / output_attentions are hooks
    for future data-weighted statistics (dressed operators, <A>_{data}, etc.)."""
    with torch.no_grad():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )


class NLLCollector:
    """Accumulates per-token NLL for perplexity computation."""
    name = "nll"

    def __init__(self):
        self._total_nll = 0.0
        self._n_tokens = 0

    def update(self, out, n_tokens: int):
        self._total_nll += out.loss.item() * n_tokens
        self._n_tokens += n_tokens

    def result(self):
        return self._total_nll / self._n_tokens if self._n_tokens else float("nan")


def eval_loop(model, input_ids: torch.Tensor, device, stride: int = 512,
              max_tokens: Optional[int] = None, collectors=None) -> dict:
    """Sliding-window eval loop. Extend by adding Collector subclasses."""
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
        out = run_inference(model, window, torch.ones_like(window))
        for collector in collectors:
            collector.update(out, target_len)
        prev_end = end
        if end == seq_len:
            break

    return {c.name: c.result() for c in collectors}


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def _device_map_key(device: torch.device):
    """Return the key accelerate's max_memory dict expects for this device."""
    if device.type == "cuda":
        return device.index or 0
    if device.type == "mps":
        return "mps"
    return "cpu"


def _load_with_offload(cache_path, torch_dtype, device, max_memory, offload_folder):
    """Layer-streamed load via accelerate: empty-init then dispatch with a
    memory budget that forces excess layers to spill to offload_folder."""
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch

    cfg = AutoConfig.from_pretrained(cache_path)
    # from_config doesn't accept "auto" the way from_pretrained does — it
    # tries getattr(torch, "auto"). Resolve "auto" against the saved config
    # before constructing the empty model.
    resolved_dtype = torch_dtype
    if torch_dtype == "auto":
        cfg_dtype = getattr(cfg, "torch_dtype", None) or getattr(cfg, "dtype", None)
        if isinstance(cfg_dtype, str):
            resolved_dtype = getattr(torch, cfg_dtype, torch.float32)
        elif cfg_dtype is not None:
            resolved_dtype = cfg_dtype
        else:
            resolved_dtype = torch.float32
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=resolved_dtype)

    if max_memory is None:
        # Conservative defaults: cap accelerator at 6GiB so big models spill.
        max_memory = {_device_map_key(device): "6GiB", "cpu": "2GiB"}
    else:
        # Caller may use the literal "__gpu__" placeholder so they don't have
        # to know the runtime device. Translate now.
        if "__gpu__" in max_memory:
            max_memory = {**max_memory}
            max_memory[_device_map_key(device)] = max_memory.pop("__gpu__")

    return load_checkpoint_and_dispatch(
        model, cache_path,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=offload_folder,
        offload_state_dict=True,
        dtype=resolved_dtype,
    )


def evaluate_model(model_name: str, revision: Optional[str],
                   corpus: str, pile_tokens: int, cache_dir: str,
                   device_str: Optional[str], stride: int = 512,
                   max_tokens: Optional[int] = None,
                   pile_cache: Optional[str] = None,
                   dtype: str = "auto",
                   offload: bool = False,
                   offload_folder: Optional[str] = None,
                   max_memory: Optional[dict] = None) -> dict:
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

    device = get_device(device_str)
    torch_dtype = resolve_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(cache_path)

    # If offloading, prepare the spill folder. We own its lifetime when the
    # caller didn't pass one in — clean it up on exit, even on failure.
    cleanup_offload = False
    if offload and offload_folder is None:
        offload_folder = tempfile.mkdtemp(prefix="eval_offload_")
        cleanup_offload = True
    if offload and offload_folder is not None:
        os.makedirs(offload_folder, exist_ok=True)

    try:
        if offload:
            print(f"  Loading model with layer offload on {device} "
                  f"(dtype={dtype}, offload_folder={offload_folder}) ...")
            model = _load_with_offload(cache_path, torch_dtype, device,
                                       max_memory, offload_folder)
            model.eval()
        else:
            print(f"  Loading model on {device} (dtype={dtype}) ...")
            model = AutoModelForCausalLM.from_pretrained(
                cache_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            model = model.to(device).eval()

        print(f"  Loading corpus ({corpus}) ...")
        tokens = load_corpus_tokens(corpus, tokenizer, pile_tokens=pile_tokens,
                                    pile_cache=pile_cache)
        print(f"  Evaluating on {len(tokens):,} tokens ...")

        # For offloaded models the embedding device varies; let accelerate's
        # hooks route inputs. For non-offloaded models, send to the resident
        # device.
        loop_device = device if not offload else torch.device("cpu")
        results = eval_loop(model, tokens, loop_device, stride=stride,
                            max_tokens=max_tokens)
        nll = results["nll"]
        ppl = math.exp(nll)
        bpb = nll / math.log(2)
    finally:
        # Free model + offload spill before returning so the caller sees
        # memory released even on failure.
        try:
            del model
        except UnboundLocalError:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if cleanup_offload and offload_folder:
            shutil.rmtree(offload_folder, ignore_errors=True)

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
    base = {"model": row["model"], "revision": row["revision"],
            "step": row["step"], "source": "eval_pass", "corpus": row["corpus"]}
    return [
        {**base, "metric": "perplexity", "value": row["perplexity"]},
        {**base, "metric": "nll",        "value": row["nll"]},
        {**base, "metric": "bpb",        "value": row["bpb"]},
    ]
