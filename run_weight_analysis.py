# `run_weight_analysis.py`
# Analysis main for weight analysi
import json
import logging
import uuid
import os
import shutil
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
from datasets import Dataset
from transformers import GPTNeoXForCausalLM

from perf_logger import PerfLogger
from attn_head_analysis import LayerHeadContainer
from histogram_utils import stats_config_default, weight_bins_default, sv_bins_default


def main(
    model_name="pythia-70m-deduped", revision="step3000", idx_max=-1, out_dir="histos", cache_dir = '.'
):
    job_uuid = str(uuid.uuid4())[:8]
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{out_dir}/logs/{job_id}.log"),
            logging.StreamHandler(),
        ],
    )

    perf = PerfLogger(job_id)

    logging.info(f"Starting job {job_id} {job_uuid}")
  

    with perf.phase("load_model"):
        logging.info("Loading model...")
        model = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/{model_name}",
            revision=revision,
            cache_dir=f"{cache_dir}/{model_name}/{revision}",
        )
    logging.info(perf.log_report(context=model_name))

    # Phase 2: Configuration
    with perf.phase("configure"):
        logging.info("Configuring analysis...")
        model_config = model.config
        config = SimpleNamespace()
        config.weight_type = ["W_Q", "W_K", "W_QK"]
        config.stats = stats_config_default.copy()
        config.w_bins = weight_bins_default.copy()
        config.sv_bins = sv_bins_default.copy()
        config.use_density = True
        config.n_heads = model_config.num_attention_heads
        config.d_model = model_config.hidden_size
        config.head_dim = config.d_model // config.n_heads
        config.n_layers = model_config.num_hidden_layers

        n_layers, n_heads, head_dim = config.n_layers, config.n_heads, config.head_dim
        d_model = config.d_model
    logging.info(perf.log_report())

    # Phase 3: Loop with conditional logging
    layer_data = []
    with perf.phase("loop"):
        n_hl = n_heads * n_layers
        if idx_max == -1:
            idx_max = n_hl
        else:
            idx_max = min(abs(idx_max), n_hl)
        logging.info(f"Processing {idx_max} / {n_hl}")

        # model-dependent extraction code
        for layer_idx, layer in enumerate(model.gpt_neox.layers):
            qkv = layer.attention.query_key_value.weight  # (1536, 512)
            W_Q, W_K, _ = qkv.chunk(3, dim=0)  # each (512, 512)
            # For per-head analysis:
            W_Q_h = W_Q.reshape(n_heads, head_dim, d_model)
            W_K_h = W_K.reshape(n_heads, head_dim, d_model)

            hc = LayerHeadContainer(layer_idx, config)
            layer_input = {"W_Q": W_Q_h, "W_K": W_K_h}
            hc.analyze_layer(layer_input)
            layer_data.append(hc)
    logging.info(perf.log_report())

    # Phase 4: Finalization
    dfs = []
    with perf.phase("finalize"):
        logging.info("Aggregating results...")
        for lhc in layer_data:
            lhc.post_process()
            dfs.append(lhc.to_pandas())
        df = pd.concat(dfs, ignore_index=True)
        df["model"] = model_name
        df["revision"] = revision
        df["step"] = int(revision.strip("step"))
        df["job_uuid"] = job_uuid
        df["job_id"] = job_id
    logging.info(perf.log_report())

    # Phase 5: Write output
    with perf.phase("write_output"):
        logging.info("Writing outputs...")

        ds = Dataset.from_pandas(df)
        ds.info.description = "metadata.json"
        out_prefix = f"{out_dir}/{model_name}_{revision}"
        ds.save_to_disk(out_prefix)
        logging.info(f"Saving dataset: {out_prefix}")
        # for metadata we need to do some coversions to make the objects JSON serializable
        config_dict = vars(config).copy()
        config_dict["stats"] = {k: v.__name__ for k, v in config_dict["stats"].items()}
        config_dict["w_bins"] = config_dict["w_bins"].tolist()
        config_dict["sv_bins"] = config_dict["sv_bins"].tolist()
        with open(f"{out_prefix}/metadata.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        with open(f"{out_dir}/logs/perf_{job_id}.json", "w") as f:
            json.dump(perf.to_metadata(), f, indent=2)
    logging.info(perf.log_report())

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("Performance Summary:")
    for phase_name in perf.phases.keys():
        logging.info(perf.log_report(phase=phase_name))
    logging.info("=" * 60)
    logging.info(f"Performance saved to {out_dir}/logs/perf_{job_id}.json")


def create_versioned_dir(path, name, time=False, clobber=False):
    """Create directory with timestamp, or numeric suffix if it exists."""
    if time:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = os.path.join(path, f"{name}_{timestamp}")
    else:
        base_dir = os.path.join(path, name)
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return base_dir

    elif clobber:
        shutil.rmtree(base_dir)
        os.makedirs(base_dir)
        return base_dir

    
    # Try numeric suffixes
    for i in range(1, 1000):
        versioned_dir = f"{base_dir}_{i:03d}"
        if not os.path.exists(versioned_dir):
            os.makedirs(versioned_dir)
            return versioned_dir
    
    raise RuntimeError("Could not find available directory suffix")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pythia-70m-deduped")
    parser.add_argument("--out", type=str, default='histos')
    parser.add_argument("--clobber", type=bool, default=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()
    if args.test:
        print('='*20 + 'Test option selected' + '='*20)
        print('\t\t' + 'output and clobber options will be overwritten')
        args.out, args.clobber = 'test', True

    cwd = os.getcwd()
    out_dir = create_versioned_dir(path=cwd, name=args.out, clobber=args.clobber)
    log_dir = create_versioned_dir(path=out_dir, name='logs', clobber=True)
    model_name = args.model

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

    if args.test:
        PYTHIA_REVISIONS = PYTHIA_REVISIONS[-1:]
    for revision in PYTHIA_REVISIONS:
        main(model_name=model_name, revision=revision, out_dir=out_dir)
