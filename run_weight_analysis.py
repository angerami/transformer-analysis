# `run_weight_analysis.py`
# Analysis main for weight analysis using GPTModel 
from transformers import GPTNeoXForCausalLM
from histogram_tools import HistogramGroup
from datasets import Dataset, concatenate_datasets
import json
import logging
from perf_logger import PerfLogger
import uuid
from datetime import datetime


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
        config = model.config
        n_heads = config.num_attention_heads
        n_layers = config.num_hidden_layers
        d_model = config.hidden_size
        head_dim = d_model // n_heads
        vocab_size = config.vocab_size


        weight_names = ['W_QK', 'W_Q', 'W_K', 'W_V']
        hg_dict = {}
        for wt in weight_names:
            hg_dict[wt] =  HistogramGroup.standard(weight_type=wt, n_layers=n_layers, n_heads=n_heads)

        #set max number of layers/heads in loops
        #default idx_max = -1 loops over all
    logging.info(perf.log_report())

    # Phase 3: Loop with conditional logging
    with perf.phase('loop'):
        n_hl = n_heads * n_layers
        if idx_max == -1:
            idx_max = n_hl
        else:
            idx_max = min(abs(idx_max), n_hl)
        logging.info(f"Processing {idx_max} / {n_hl}")

        icount = 0
        for layer_idx, layer in enumerate(model.gpt_neox.layers):
            with perf.loop_item(layer_idx, log_every=1):
                print(layer_idx, icount)
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
                    if icount >= idx_max:
                        break #head loop
            if icount >= idx_max:
                break
    logging.info(perf.log_report())

    # Phase 4: Finalization
    with perf.phase('finalize'):
        logging.info("Aggregating results...")
        logging.info(f"Processed {icount} total")
    logging.info(perf.log_report())
    
    # Phase 5: Write output
    with perf.phase('write_output'):
        logging.info("Writing outputs...")
        datasets = []
        for k,v in hg_dict.items():
            v.post_process()
            df, metadata = v.to_pandas()
            df['model'] = model_name
            df['revision'] = revision
            df['weight_type'] = k
            df['job_uuid'] = job_uuid
            df['date'] = job_id
            datasets.append(Dataset.from_pandas(df))
        combined = concatenate_datasets(datasets)
        combined.info.description = "metadata.json"
        combined.save_to_disk(f'{out_dir}/{model_name}')
        json.dump(metadata, open(f'{out_dir}/{model_name}/metadata.json', 'w'))
        perf_metadata = perf.to_metadata()
        with open(f'{out_dir}/logs/perf_{job_id}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
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