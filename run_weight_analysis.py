# `run_weight_analysis.py`
# Analysis main for weight analysis using GPTModel 

from gpt import GPTModel, load_weights_into_gpt, get_model_dict
from gpt_download import download_and_load_gpt2
from histogram_utils import build_group_standard
import numpy as np
import psutil
import os
import sys

model_shorts = {
    'small': 'gpt2-small (124M)',
    'medium': 'gpt2-medium (355M)',
    'large': 'gpt2-large (774M)',
    'xl': 'gpt2-xl (1558M)'}

def main(model_name="small", test=False):

    print('tf')
    try:
        model_long = model_shorts[model_name]

    except KeyError:
        print(f"Model [{model_long}] is not a valid option. Choose from")
        for k in model_shorts.keys():
            print(k)
        exit()
   
    n_layers_max = sys.maxsize
    n_heads_max = sys.maxsize

    if test:
        n_layers_max = 1
        n_heads_max = 1
        model_long = "gpt2-small (124M)"
        model_name = 'test'
        print("TEST MODE: model will be gpt2-small (124M)")
        
    MODEL_CONFIG = get_model_dict(model_long)
    print(f"Configuration\nMODEL : {model_long}")
    for k,v in MODEL_CONFIG.items():
        print(f"\t{k} : {v}")

    ## Get Data
    print("Loading Data ...")

    model_size = model_long.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="/Users/angerami/Desktop/Materials/gpt2")

    model = GPTModel(MODEL_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval();

    print("Model successfully loaded")
    print('\n\n' + '-' * 80)
    print(model)
    print('-' * 80 + '\n\n')

    n_layers = len(model.trf_blocks)
    n_heads = model.trf_blocks[0].att.n_heads
    d_out = model.trf_blocks[0].att.d_out
    head_dim = model.trf_blocks[0].att.head_dim

    print("Configuring Analysis ... ")
    ## Configure Analysis Objects

    process = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    print(f"CPU: {process.cpu_percent():.1f}%")
    print(f"Threads: {process.num_threads()}")
    print(f"Virtual Memory: {vm.percent:.1f}% ({vm.used/1024**3:.1f}/{vm.total/1024**3:.1f} GB)")
    print(f"Memory: {process.memory_info().rss / 1024**2:.1f} MB")

    print("Event Loop ... ")
    bins = np.linspace(-1.6, 1.6, 1024)
    # hg = HistogramGroup(bins=bins, n_layers=n_layers, n_heads=n_heads)
    h_qk = build_group_standard(n_layers=n_layers, n_heads=n_heads, svd=True, prefix="W_QK")
    h_q = build_group_standard(n_layers=n_layers, n_heads=n_heads, svd=False, prefix="W_Q")
    h_k = build_group_standard(n_layers=n_layers, n_heads=n_heads, svd=False, prefix="W_K")
    ## Event Loop Fill 
    idx = 0
    for layer_idx in range(min(n_layers,n_layers_max)):
        print(f">>> Layer {layer_idx + 1} of {n_layers}")
        mha = model.trf_blocks[layer_idx].att
        W_k_h = mha.W_key.weight.view(d_out, n_heads, head_dim)
        W_q_h = mha.W_query.weight.view(d_out, n_heads, head_dim)
        W_q_h = W_q_h.transpose(0, 1)
        W_k_h = W_k_h.permute(1, 2, 0)
        print(W_q_h.shape,W_k_h.shape)
        W_qk = W_q_h @ W_k_h
        for head_idx in range(min(n_heads, n_heads_max)):
            vals_qk = W_qk[head_idx].flatten().detach().cpu().numpy()
            h_qk[(layer_idx, head_idx)].fill(vals_qk)
            h_qk[(layer_idx, head_idx)].fill_SVD(W_qk[head_idx])

            vals_q = W_q_h[head_idx].flatten().detach().cpu().numpy()
            h_q[(layer_idx, head_idx)].fill(vals_q)

            vals_k = W_k_h[head_idx].flatten().detach().cpu().numpy()
            h_k[(layer_idx, head_idx)].fill(vals_k)


            idx += 1
            if idx % 10 == 0:
                print(f"\t\tMemory: {process.memory_info().rss / 1024**2:.1f} MB")
            del vals_qk, vals_q, vals_k
        del W_k_h, W_q_h, W_qk

    h_qk.save(f"histos/{model_name}.W_QK.histos.pkl")
    h_q.save(f"histos/{model_name}.W_Q.histos.pkl")
    h_k.save(f"histos/{model_name}.W_K.histos.pkl")
    
import argparse
if __name__ == '__main__':
    print('qwfqr')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='small')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    main(model_name=args.model, test=args.test)