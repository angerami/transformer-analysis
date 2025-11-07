# `run_weight_analysis.py`
# Analysis main for weight analysis using GPTModel 

from gpt import GPTModel, load_weights_into_gpt, get_model_dict
from gpt_download import download_and_load_gpt2
from histogram_tools import HistogramBase, HistogramGroup
import numpy as np
import psutil
import os

def main():

    ## Specify Configuration
    #CHOOSE_MODEL = "gpt2-large (774M)"
    CHOOSE_MODEL = "gpt2-small (124M)"
    MODEL_CONFIG = get_model_dict(CHOOSE_MODEL)
    print("Configuration\nMODEL : {CHOOSE_MODEL}")
    for k,v in MODEL_CONFIG.items():
        print(k, v)

    ## Get Data
    print("Loading Data ...")

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
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
    bins = np.linspace(-2.5, 2.5, 11)

    process = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    print(f"CPU: {process.cpu_percent():.1f}%")
    print(f"Threads: {process.num_threads()}")
    print(f"Virtual Memory: {vm.percent:.1f}% ({vm.used/1024**3:.1f}/{vm.total/1024**3:.1f} GB)")
    print(f"Memory: {process.memory_info().rss / 1024**2:.1f} MB")

    print("Event Loop ... ")

    hg = HistogramGroup(bins=bins, n_layers=n_layers, n_heads=n_heads)
    ## Event Loop Fill 
    for layer_idx in range(2): #range(n_layers):
        print("Begin ... ", f"Layer {layer_idx + 1} of {n_layers}")
        mha = model.trf_blocks[layer_idx].att
        W_k_h = mha.W_key.weight.view(d_out, n_heads, head_dim)
        W_q_h = mha.W_query.weight.view(d_out, n_heads, head_dim)
        W_q_h = W_q_h.transpose(0, 1)
        W_k_h = W_k_h.permute(1, 2, 0)
        W_qk = W_q_h @ W_k_h
        h_L = HistogramBase(bins=bins, name=f"h_L{layer_idx:03d}")
        for head_idx in range(n_heads):
            h = HistogramBase(bins=bins, name=f"h_L{layer_idx:03d}_H{head_idx:03d}")
            vals = W_qk[head_idx].flatten().detach().cpu().numpy()
            h.fill(vals)
            h.save(filename=f"{h.name}.pkl")
            h_L.fill(vals)
            del vals
        h_L.save(filename=f"{h_L.name}.pkl")
        del W_k_h, W_q_h, W_qk
        print(" ... End", f"Layer {layer_idx + 1} of {n_layers}")
        print(f"Memory: {process.memory_info().rss / 1024**2:.1f} MB")

if __name__ == '__main__':
    print("run_weight_analysis.py")
    main()