# `run_weight_analysis.py`
# Analysis main for weight analysis using GPTModel 
from transformers import GPTNeoXForCausalLM
from histogram_tools import HistogramGroup
from datasets import Dataset, concatenate_datasets
import numpy as np
import psutil
import os
import sys
import json





def main(model_name="pythia-70m-deduped", revision="step3000", idx_max=-1, out_dir='histos'):

    # try:
    #     model_long = model_shorts[model_name]

    # except KeyError:
    #     print(f"Model [{model_long}] is not a valid option. Choose from")
    #     for k in model_shorts.keys():
    #         print(k)
    #     exit()
   
    # 

    process = psutil.Process(os.getpid())
    import uuid
    from datetime import datetime
    job_id = str(uuid.uuid4())[:8]
    job_date = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

# e.g., '20241115_143022_a3f2'
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

    weight_names = ['W_QK', 'W_Q', 'W_K', 'W_V']
    hg_dict = {}
    for wt in weight_names:
        hg_dict[wt] =  HistogramGroup.standard(weight_type=wt, n_layers=n_layers, n_heads=n_heads)

    #set max number of layers/heads in loops
    #default idx_max = -1 loops over all
    n_hl = n_heads * n_layers
    if idx_max == -1:
        idx_max = n_hl
    else:
        idx_max = min(abs(idx_max), n_hl)
    print(f"Processing {idx_max} / {n_hl}")


    icount = 0
    for layer_idx, layer in enumerate(model.gpt_neox.layers):
        qkv = layer.attention.query_key_value.weight  # (1536, 512)
        W_Q, W_K, W_V = qkv.chunk(3, dim=0)  # each (512, 512)
        # For per-head analysis:
        W_Q_h = W_Q.reshape(n_heads, head_dim, d_model)
        W_K_h = W_K.reshape(n_heads, head_dim, d_model)
        W_V_h = W_V.reshape(n_heads, head_dim, d_model)

        W_Q_h = W_Q_h.transpose(0, 1)
        W_K_h = W_K_h.permute(1, 2, 0)
        W_QK = W_Q_h @ W_K_h

        for head_idx in range(n_heads):
            idx = (layer_idx, head_idx)
            hg_dict['W_QK'][idx].fill(W_QK[head_idx])
            hg_dict['W_Q'][idx].fill(W_Q[head_idx])
            hg_dict['W_K'][idx].fill(W_K[head_idx])
            hg_dict['W_V'][idx].fill(W_V[head_idx])

            icount += 1
            if icount % 10 == 0:
                print(f"Processing (layer, head) = {idx}. {icount} / {n_hl}")
                print(f"\t\tMemory: {process.memory_info().rss / 1024**2:.1f} MB")

            if icount >= idx_max:
                break #head loop
        if icount >= idx_max:
            break



    print(f"Processed {icount}")

    print('Producing output files...')
    datasets = []
    for k,v in hg_dict.items():
        v.post_process()
        df, metadata = v.to_pandas()
        df['model'] = model_name
        df['revision'] = revision
        df['weight_type'] = k
        df['job_id'] = job_id
        df['date'] = job_date
        datasets.append(Dataset.from_pandas(df))
    combined = concatenate_datasets(datasets)
    combined.info.description = "metadata.json"
    combined.save_to_disk(f'{out_dir}/{model_name}')
    json.dump(metadata, open(f'{out_dir}/{model_name}/metadata.json', 'w'))


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

    main(model_name=model_name, revision=revision,  out_dir='histos', idx_max=args.n)