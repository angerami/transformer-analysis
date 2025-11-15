# `run_weight_analysis.py`
# Analysis main for weight analysis using GPTModel 
from transformers import GPTNeoXForCausalLM
from histogram_utils import build_group_standard
import numpy as np
import psutil
import os
import sys




def main(model_name="pythia-70m-deduped", revision="step3000", idx_max=-1):

    # try:
    #     model_long = model_shorts[model_name]

    # except KeyError:
    #     print(f"Model [{model_long}] is not a valid option. Choose from")
    #     for k in model_shorts.keys():
    #         print(k)
    #     exit()
   
    # 

    process = psutil.Process(os.getpid())
    print(f"Loading {model_name}, revision {revision}")
    vm = psutil.virtual_memory()
    print(f"CPU: {process.cpu_percent():.1f}%")
    print(f"Threads: {process.num_threads()}")
    print(f"Virtual Memory: {vm.percent:.1f}% ({vm.used/1024**3:.1f}/{vm.total/1024**3:.1f} GB)")
    print(f"Memory: {process.memory_info().rss / 1024**2:.1f} MB")
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/{model_name}",
        revision=revision,
        cache_dir=f"./{model_name}/{revision}",
    )

    config = model.config
    n_heads = config.num_attention_heads
    n_layers = config.num_hidden_layers
    d_model = config.hidden_size
    head_dim = d_model // n_heads
    vocab_size = config.vocab_size

    print(f"Loading complete")
    print(f"Virtual Memory: {vm.percent:.1f}% ({vm.used/1024**3:.1f}/{vm.total/1024**3:.1f} GB)")
    print(f"Memory: {process.memory_info().rss / 1024**2:.1f} MB")

    h_qk = build_group_standard(n_layers=n_layers, n_heads=n_heads, svd=True, prefix="W_QK")
    h_q = build_group_standard(n_layers=n_layers, n_heads=n_heads, svd=False, prefix="W_Q")
    h_k = build_group_standard(n_layers=n_layers, n_heads=n_heads, svd=False, prefix="W_K")
    h_v = build_group_standard(n_layers=n_layers, n_heads=n_heads, svd=False, prefix="W_V")

    n_hl = n_heads * n_layers
    if idx_max == -1:
        idx_max = n_hl
    else:
        idx_max = min(abs(idx_max), n_hl)


    print(f"Processing {idx_max} / {n_hl}")
        
    idx = 0
    for layer_idx, layer in enumerate(model.gpt_neox.layers):
        qkv = layer.attention.query_key_value.weight  # (1536, 512)
        W_q, W_k, W_v = qkv.chunk(3, dim=0)  # each (512, 512)
        # For per-head analysis:
        W_q_h = W_q.reshape(n_heads, head_dim, d_model)
        W_k_h = W_k.reshape(n_heads, head_dim, d_model)
        W_v_h = W_v.reshape(n_heads, head_dim, d_model)

        W_q_h = W_q_h.transpose(0, 1)
        W_k_h = W_k_h.permute(1, 2, 0)
        W_qk = W_q_h @ W_k_h

        for head_idx in range(n_heads):
            vals_qk = W_qk[head_idx].flatten().detach().cpu().numpy()
            h_qk[(layer_idx, head_idx)].fill(vals_qk)
            h_qk[(layer_idx, head_idx)].fill_SVD(W_qk[head_idx])

            vals_q = W_q_h[head_idx].flatten().detach().cpu().numpy()
            h_q[(layer_idx, head_idx)].fill(vals_q)

            vals_k = W_k_h[head_idx].flatten().detach().cpu().numpy()
            h_k[(layer_idx, head_idx)].fill(vals_k)

            vals_v = W_v_h[head_idx].flatten().detach().cpu().numpy()
            h_v[(layer_idx, head_idx)].fill(vals_v)

            idx += 1
            del vals_qk, vals_q, vals_k, vals_v
            if idx % 10 == 0:
                print(f"Processing {idx} / {n_hl}")
                print(f"\t\tMemory: {process.memory_info().rss / 1024**2:.1f} MB")
            if idx >= idx_max:
                break #head loop
        del W_k_h, W_q_h, W_qk
        if idx >= idx_max:
            break


    print(f"Processed {idx}")

    h_qk.save(f"histos/{model_name}.W_QK.histos.pkl")
    h_q.save(f"histos/{model_name}.W_Q.histos.pkl")
    h_k.save(f"histos/{model_name}.W_K.histos.pkl")
    h_v.save(f"histos/{model_name}.W_V.histos.pkl")

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='small')
    parser.add_argument('--n', type=int, default=-1)
    args = parser.parse_args()
    
    # model_name = arg.model
    #model_name = "pythia-70m-deduped"
    model_name = "pythia-2.8b-deduped"
    revision="step3000"

    main(model_name=model_name, revision=revision, idx_max=args.n)