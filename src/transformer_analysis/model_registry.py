from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, TYPE_CHECKING
import os
import json

if TYPE_CHECKING:
    import torch


@dataclass
class ModelConfig:
    """Configuration for model-specific weight extraction."""

    repo_id: str
    config_fields: Dict[str, str]
    extract_qkv: Callable[
        [str, int, int, Dict], Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]
    ]
    revisions: List[str]
    allow_patterns: List[str]
    qkv_scale_factor: float = 1.0  # Scaling factor applied to W_Q, W_K, W_V after extraction

    def get_config_value(self, config_dict: Dict, standard_name: str) -> int:
        """Extract a config value using the model-specific field name."""
        model_field = self.config_fields[standard_name]
        return config_dict[model_field]


# ============================================================================
# Extraction Functions
# ============================================================================
def extract_shard_path(
    cache_path: str, key: str, weight_map: Dict = None, binfile_name="pytorch_model.bin"
):
    if weight_map:
        shard = weight_map[key]
        return os.path.join(cache_path, shard)
    else:
        return os.path.join(cache_path, binfile_name)


def extract_weight_map(cache_path: str, index_file_name="pytorch_model.bin.index.json"):
    index_file = os.path.join(cache_path, index_file_name)
    if os.path.exists(index_file):
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        weight_map = None  # Single file model
    return weight_map


def get_safetensors_path(cache_path, key):
    """Find which shard contains the key."""
    index_path = os.path.join(cache_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shard = weight_map[key]
        return os.path.join(cache_path, shard)
    else:
        # Single file
        return os.path.join(cache_path, "model.safetensors")


def extract_pythia_qkv(
    cache_path: str, layer_idx: int, d_model: int, weight_map: Dict = None, device: str = "cpu", qkv_scale_factor: float = 1.0
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Extract Q, K, V weights for Pythia/GPT-NeoX models with memory-mapped loading."""
    import torch

    key = f"gpt_neox.layers.{layer_idx}.attention.query_key_value.weight"

    shard_path = extract_shard_path(
        cache_path=cache_path,
        key=key,
        weight_map=weight_map,
        binfile_name="pytorch_model.bin",
    )

    state_dict = torch.load(shard_path, map_location=device, mmap=True)
    qkv = state_dict[key].clone()
    del state_dict

    W_Q, W_K, W_V = qkv.chunk(3, dim=0)
    return W_Q * qkv_scale_factor, W_K * qkv_scale_factor, W_V * qkv_scale_factor


def extract_gpt2_qkv(
    cache_path: str, layer_idx: int, d_model: int, weight_map: Dict = None, device: str = "cpu", qkv_scale_factor: float = 1.0
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Extract Q, K, V weights for GPT-2 models using safetensors."""
    from safetensors import safe_open
    import os

    # GPT-2 uses safetensors (single file, not sharded)
    safetensors_path = os.path.join(cache_path, "model.safetensors")

    key = f"h.{layer_idx}.attn.c_attn.weight"

    with safe_open(safetensors_path, framework="pt", device=device) as f:
        c_attn = f.get_tensor(key).T.clone()

    W_Q, W_K, W_V = c_attn.chunk(3, dim=0)
    return W_Q * qkv_scale_factor, W_K * qkv_scale_factor, W_V * qkv_scale_factor


def extract_llama_qkv(
    cache_path: str, layer_idx: int, d_model: int, weight_map: Dict = None, device: str = "cpu", qkv_scale_factor: float = 1.0
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Extract Q, K, V for LLaMA models."""
    from safetensors import safe_open

    q_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
    k_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
    v_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
    q_path = get_safetensors_path(cache_path, q_key)
    k_path = get_safetensors_path(cache_path, k_key)
    v_path = get_safetensors_path(cache_path, v_key)

    with safe_open(q_path, framework="pt", device=device) as f:
        W_Q = f.get_tensor(q_key).clone()
    with safe_open(k_path, framework="pt", device=device) as f:
        W_K = f.get_tensor(k_key).clone()
    with safe_open(v_path, framework="pt", device=device) as f:
        W_V = f.get_tensor(v_key).clone()

    # n_kv_heads = W_K.shape[0] // (W_Q.shape[0] // W_K.shape[0])
    repeat_factor = W_Q.shape[0] // W_K.shape[0]
    W_K = W_K.repeat_interleave(repeat_factor, dim=0)
    W_V = W_V.repeat_interleave(repeat_factor, dim=0)
    return W_Q * qkv_scale_factor, W_K * qkv_scale_factor, W_V * qkv_scale_factor


def extract_mistral_qkv(
    cache_path: str, layer_idx: int, d_model: int, weight_map: Dict = None, device: str = "cpu", qkv_scale_factor: float = 1.0
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Extract Q, K, V for mistral models.

    Supports both consolidated and sharded safetensors formats.
    Prefers consolidated if available, otherwise uses sharded format.
    """
    from safetensors import safe_open
    import os

    # Try consolidated format first (if it exists)
    consolidated_path = os.path.join(cache_path, "consolidated.safetensors")
    if os.path.exists(consolidated_path):
        safetensors_path = consolidated_path
        q_key = f"layers.{layer_idx}.attention.wq.weight"
        k_key = f"layers.{layer_idx}.attention.wk.weight"
        v_key = f"layers.{layer_idx}.attention.wv.weight"

        with safe_open(safetensors_path, framework="pt", device=device) as f:
            W_Q = f.get_tensor(q_key).clone()
            W_K = f.get_tensor(k_key).clone()
            W_V = f.get_tensor(v_key).clone()
    else:
        # Use sharded format with index
        q_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        k_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        v_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"

        q_path = get_safetensors_path(cache_path, q_key)
        k_path = get_safetensors_path(cache_path, k_key)
        v_path = get_safetensors_path(cache_path, v_key)

        with safe_open(q_path, framework="pt", device=device) as f:
            W_Q = f.get_tensor(q_key).clone()
        with safe_open(k_path, framework="pt", device=device) as f:
            W_K = f.get_tensor(k_key).clone()
        with safe_open(v_path, framework="pt", device=device) as f:
            W_V = f.get_tensor(v_key).clone()

    # uses grouped-query attention
    # fewer independent heads for kv than q
    # W_QK = W_Q W_K uses a different W_Q per head
    # but reuses W_K repeat factor times
    # n_kv_heads = W_K.shape[0] // (W_Q.shape[0] // W_K.shape[0])
    repeat_factor = W_Q.shape[0] // W_K.shape[0]
    W_K = W_K.repeat_interleave(repeat_factor, dim=0)
    W_V = W_V.repeat_interleave(repeat_factor, dim=0)

    return W_Q * qkv_scale_factor, W_K * qkv_scale_factor, W_V * qkv_scale_factor


# ============================================================================
# Model Registry
# ============================================================================

#
# Pythia models to register
PYTHIA_MODELS = [
    "pythia-70m-deduped",
    "pythia-160m-deduped",
    "pythia-410m-deduped",
    "pythia-1b-deduped",
    "pythia-1.4b-deduped",
    "pythia-2.8b-deduped",
    "pythia-6.9b-deduped",
    "pythia-12b-deduped",
]


PYTHIA_REVISIONS = [
    "step0",
    "step1",
    "step2",
    "step4",
    "step8",
    "step16",
    "step32",
    "step64",
    "step128",
    "step256",
    "step512",
] + [f"step{step}" for step in range(1000, 144000, 1000)]


# Common config for all Pythia models
PYTHIA_CONFIG_FIELDS = {
    "n_layers": "num_hidden_layers",
    "d_model": "hidden_size",
    "n_heads": "num_attention_heads",
}

# Build MODEL_CONFIGS registry
MODEL_CONFIGS = {}

# Add Pythia models
for pythia_model in PYTHIA_MODELS:
    MODEL_CONFIGS[pythia_model] = ModelConfig(
        repo_id=f"EleutherAI/{pythia_model}",
        config_fields=PYTHIA_CONFIG_FIELDS,
        extract_qkv=extract_pythia_qkv,
        revisions=PYTHIA_REVISIONS,
        allow_patterns=["*.bin", "*.json"],
    )

# Add GPT-2 models
GPT2_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
GPT2_CONFIG_FIELDS = {
    "n_layers": "n_layer",
    "d_model": "n_embd",
    "n_heads": "n_head",
}

for gpt2_model in GPT2_MODELS:
    MODEL_CONFIGS[gpt2_model] = ModelConfig(
        repo_id=f"openai-community/{gpt2_model}",
        config_fields=GPT2_CONFIG_FIELDS,
        extract_qkv=extract_gpt2_qkv,
        revisions=[],
        allow_patterns=["*.safetensors", "config.json"],
    )
# Add LLaMA models
LLAMA_MODELS = ["llama-3.1-8b", "llama-3.1-70b", "llama-3.2-1b", "llama-3.2-3b"]
LLAMA_CONFIG_FIELDS = {
    "n_layers": "num_hidden_layers",
    "d_model": "hidden_size",
    "n_heads": "num_attention_heads",
}

for llama_model in LLAMA_MODELS:
    MODEL_CONFIGS[llama_model] = ModelConfig(
        repo_id=f"meta-llama/{llama_model}",
        config_fields=LLAMA_CONFIG_FIELDS,
        extract_qkv=extract_llama_qkv,
        revisions=[],
        allow_patterns=["*.safetensors", "model.safetensors.index.json", "config.json"],
    )

# Add Mistral models

MISTRAL_MODELS = ["mistral-7b-v0.3", "mixtral-8x7b-v0.1", "mixtral-8x22b-v0.1"]
MISTRAL_CONFIG_FIELDS = {
    "n_layers": "num_hidden_layers",
    "d_model": "hidden_size",
    "n_heads": "num_attention_heads",
}

for mistral_model in MISTRAL_MODELS:
    MODEL_CONFIGS[mistral_model] = ModelConfig(
        repo_id=f"mistralai/{mistral_model}",
        config_fields=MISTRAL_CONFIG_FIELDS,
        extract_qkv=extract_mistral_qkv,
        revisions=[],
        # Exclude consolidated.safetensors to avoid downloading duplicate weights
        # The model-*-of-*.safetensors files contain the same weights in sharded format
        allow_patterns=["model-*.safetensors", "model.safetensors.index.json", "config.json"],
        qkv_scale_factor=5.66,  # Suspected factor of sqrt(n_heads) for Mistral models
    )


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get the configuration for a given model.

    Args:
        model_name: Name of the model (e.g., 'pythia-70m-deduped', 'gpt2')

    Returns:
        ModelConfig object containing repo_id, config_fields, and extract_qkv

    Raises:
        ValueError: If model_name is not in registry
    """
    if model_name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {model_name}\nAvailable models: {available}")
    return MODEL_CONFIGS[model_name]


def list_supported_models() -> list[str]:
    """Return list of all supported model names."""
    return sorted(MODEL_CONFIGS.keys())


def is_model_supported(model_name: str) -> bool:
    """Check if a model is supported."""
    return model_name in MODEL_CONFIGS


def get_model_versions(model_name: str):
    mc = get_model_config(model_name=model_name)
    return mc.revisions
