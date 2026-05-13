"""
Inspect config.json fields and attention weight key names for new model families.
Run this before modifying model_registry.py to confirm architecture parameters.

Usage:
    conda run -n llm-work python scripts/inspect_new_models.py
"""

from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
import json
import os

MODELS_TO_CHECK = [
    # OLMo 2
    "allenai/OLMo-2-0425-1B",
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-13B",
    "allenai/OLMo-2-0325-32B",
    # Llama 3.2 (gated — requires HUGGING_FACE_HUB_TOKEN)
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    # Gemma 2 (gated — requires HUGGING_FACE_HUB_TOKEN)
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    # SmolLM2
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]

CONFIG_KEYS = [
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "attention_bias",
]

print("=" * 70)
print("Step 1: config.json fields")
print("=" * 70)
for slug in MODELS_TO_CHECK:
    try:
        path = hf_hub_download(repo_id=slug, filename="config.json")
        with open(path) as f:
            cfg = json.load(f)
        print(f"\n{slug}")
        for key in CONFIG_KEYS:
            print(f"  {key:30s}: {cfg.get(key, 'NOT PRESENT')}")
    except Exception as e:
        print(f"\n{slug}  ERROR: {e}")

print("\n\n" + "=" * 70)
print("Step 2: attention weight key names (one family at a time)")
print("=" * 70)

# Inspect one model per family — edit SLUG_TO_INSPECT and re-run as needed
SLUG_TO_INSPECT = "allenai/OLMo-2-0425-1B"

try:
    cache = snapshot_download(SLUG_TO_INSPECT)
    st_files = sorted(f for f in os.listdir(cache) if f.endswith(".safetensors"))
    if not st_files:
        print(f"No safetensors found in {cache}")
    else:
        print(f"\nWeight keys for {SLUG_TO_INSPECT} — layer 0 attention:")
        with safe_open(os.path.join(cache, st_files[0]), framework="pt", device="cpu") as f:
            all_keys = list(f.keys())

        layer0_attn_keys = [
            k for k in all_keys
            if ("layers.0" in k or "h.0" in k or "blocks.0" in k)
            and any(x in k for x in ["attn", "attention", "q_proj", "k_proj",
                                      "query", "key", "qkv", "self_attn"])
        ]
        if not layer0_attn_keys:
            layer0_attn_keys = [k for k in all_keys if ".0." in k]

        with safe_open(os.path.join(cache, st_files[0]), framework="pt", device="cpu") as f:
            for k in sorted(layer0_attn_keys):
                try:
                    t = f.get_tensor(k)
                    print(f"  {k:70s}  {tuple(t.shape)}")
                except Exception:
                    print(f"  {k:70s}  (not in this shard)")
except Exception as e:
    print(f"ERROR inspecting {SLUG_TO_INSPECT}: {e}")
