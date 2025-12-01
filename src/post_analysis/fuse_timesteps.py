from model_registry import get_model_config, get_model_versions, extract_weight_map

import pandas as pd
from huggingface_hub import snapshot_download
from datasets import Dataset, load_from_disk
from transformers import AutoConfig
import numpy as np

def process_model(model_name="pythia-70m-deduped", revision='step3000', cache_dir='.', out_dir='timesteps'):
    model_config = get_model_config(model_name)
    cache_path = snapshot_download(
        repo_id=model_config.repo_id,
        revision=revision,
        cache_dir=f"{cache_dir}/{model_name}/{revision}",
        allow_patterns=model_config.allow_patterns
    )
    hf_config = AutoConfig.from_pretrained(cache_path)
    n_heads = model_config.get_config_value(hf_config.__dict__, 'n_heads')
    d_model = model_config.get_config_value(hf_config.__dict__, 'd_model')
    n_layers = model_config.get_config_value(hf_config.__dict__, 'n_layers')    
    head_dim = d_model // n_heads
    weight_map = extract_weight_map(cache_path=cache_path)
###    self.data[weight_name].update({"x": x_arr.to_numpy()})

    df_in = []
    for layer_idx in range(n_layers):
        # qkv = state_dict[key].clone()
        W_Q, W_K, _ = model_config.extract_qkv(cache_path, layer_idx, d_model, weight_map)
        # For per-head analysis:
        W_Q_h = W_Q.reshape(n_heads, head_dim, d_model).float()
        W_K_h = W_K.reshape(n_heads, head_dim, d_model).float()
        W_QK_h = W_Q_h @ W_Q_h.transpose(1, 2)
        for head_idx in range(n_heads):
            df_in.append({
                "step" : int(revision.strip("step")), 
                "layer" : layer_idx,
                "head" : head_idx,
                "W_QK" : W_QK_h[head_idx].flatten().cpu().numpy()
                })
    df = pd.DataFrame(df_in)
    ds = Dataset.from_pandas(df)
    ds.save_to_disk(f'{out_dir}/{revision}')

def merge(model_name="pythia-70m-deduped", out_dir='timesteps'):
    model_config = get_model_config(model_name=model_name)
    revisions = model_config.revisions
    dfs = []
    for revision in revisions:
        ds = load_from_disk(f"{out_dir}/{revision}")
        dfs.append(ds.to_pandas())
    # Concatenate all time dataframes
    df_all = pd.concat(dfs, ignore_index=True)

    # Group by layer and head
    grouped = df_all.groupby(['layer', 'head'])

    rows = []
    for (layer, head), group in grouped:
        # Stack v vectors (one per timestep)
        v_matrix = np.stack(group.sort_values('step')['W_QK'].values)
        
        # Transpose: now rows are vector indices, columns are timesteps
        for idx in range(v_matrix.shape[1]):
            rows.append({
                'index': idx,
                'layer': layer,
                'head': head,
                'W_t': v_matrix[:, idx]
            })

    new_df = pd.DataFrame(rows)
    ds = Dataset.from_pandas(new_df)
    ds.save_to_disk(f'{out_dir}/all_checkpoints')

if __name__ == "__main__":

    # model_name="pythia-70m-deduped"
    # model_config = get_model_config(model_name=model_name)
    # revisions = model_config.revisions
    # for revision in revisions:
    #     process_model(model_name=model_name, revision=revision, cache_dir='.', out_dir='timesteps')
    merge()