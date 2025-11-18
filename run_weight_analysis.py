# `run_weight_analysis.py`
# Analysis main for weight analysis using GPTModel 
from transformers import GPTNeoXForCausalLM
from datasets import Dataset
import pandas as pd
import json
import logging
from perf_logger import PerfLogger
import uuid
from datetime import datetime
from types import SimpleNamespace
import numpy as np



from attn_head_analysis import LayerHeadContainer

def main(model_name="pythia-70m-deduped", revision="step3000", idx_max=-1, out_dir='histos'):

    job_uuid = str(uuid.uuid4())[:8]
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{out_dir}/logs/{job_id}.log'),
            logging.StreamHandler()
        ]
    )

    perf = PerfLogger(job_id)

    logging.info(f"Starting job {job_id} {job_uuid}")

    with perf.phase('load_model'):
        logging.info("Loading model...")   
        model = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/{model_name}",
            revision=revision,
            cache_dir=f"./{model_name}/{revision}",
        )
    logging.info(perf.log_report(context=model_name))

    # Phase 2: Configuration
    with perf.phase('configure'):
        logging.info("Configuring analysis...")
        model_config = model.config
        config = SimpleNamespace()
        config.weight_type = ['W_Q', 'W_K', 'W_QK']
        config.stats = {'mean' : np.mean, 'std' : np.std}
        config.w_bins = np.linspace(-2, 2, 201 ) #low number of bins for easy visual inspection
        config.use_density = False
        config.n_heads = model_config.num_attention_heads
        config.d_model = model_config.hidden_size        
        config.head_dim = config.d_model // config.n_heads
        config.n_layers = model_config.num_hidden_layers

        n_layers, n_heads, head_dim = config.n_layers, config.n_heads, config.head_dim
        d_model = config.d_model
    logging.info(perf.log_report())

    # Phase 3: Loop with conditional logging
    layer_data = []
    with perf.phase('loop'):
        n_hl = n_heads * n_layers
        if idx_max == -1:
            idx_max = n_hl
        else:
            idx_max = min(abs(idx_max), n_hl)
        logging.info(f"Processing {idx_max} / {n_hl}")

        for layer_idx, layer in enumerate(model.gpt_neox.layers):
            with perf.loop_item(layer_idx, log_every=1):
                qkv = layer.attention.query_key_value.weight  # (1536, 512)
                W_Q, W_K, W_V = qkv.chunk(3, dim=0)  # each (512, 512)
                # For per-head analysis:
                W_Q_h = W_Q.reshape(n_heads, head_dim, d_model)
                W_K_h = W_K.reshape(n_heads, head_dim, d_model)
                hc = LayerHeadContainer(layer_idx, config)
                layer_input = {'W_Q' : W_Q_h, 'W_K' : W_K_h }
                hc.analyze_layer(layer_input)
                layer_data.append(hc)
    logging.info(perf.log_report())

    # Phase 4: Finalization
    dfs = []
    with perf.phase('finalize'):
        logging.info("Aggregating results...")
        for l in layer_data:
            l.post_process()
            dfs.append(l.to_pandas())
        df = pd.concat(dfs, ignore_index=True)
        
    logging.info(perf.log_report())
    
    # Phase 5: Write output
    with perf.phase('write_output'):
        logging.info("Writing outputs...")
       
        ds = Dataset.from_pandas(df)
        ds.info.description = "metadata.json"
        ds.save_to_disk(f'{out_dir}/{model_name}')
        # with open(f'{out_dir}/{model_name}/metadata.json', 'w') as f:
        #     json.dump(metadata, f)
        with open(f'{out_dir}/logs/perf_{job_id}.json', 'w') as f:
            json.dump(perf.to_metadata(), f, indent=2)
    logging.info(perf.log_report())

      
    # Summary
    logging.info("\n" + "="*60)
    logging.info("Performance Summary:")
    for phase_name in perf.phases.keys():
        logging.info(perf.log_report(phase=phase_name))
    logging.info("="*60)    
    logging.info(f"Performance saved to {out_dir}/logs/perf_{job_id}.json")


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='small')
    parser.add_argument('--n', type=int, default=-1)
    args = parser.parse_args()
    
    # model_name = arg.model
    model_name = "pythia-70m-deduped"
    # model_name = "pythia-2.8b-deduped"
    revision="step3000"

    main(model_name=model_name, revision=revision,  out_dir='histos_1', idx_max=args.n)