# `run_weight_analysis.py`
# Analysis main for weight analysis
import json
import logging
import uuid
import os
import shutil
from datetime import datetime
from types import SimpleNamespace

from tqdm import tqdm
import pandas as pd
from huggingface_hub import snapshot_download
from datasets import Dataset
from transformers import AutoConfig

from transformer_analysis.perf_logger import PerfLogger
from transformer_analysis.attn_head_analysis import LayerHeadContainer
from transformer_analysis.histogram_utils import (
    stats_config_default,
    weight_bins_default,
    sv_bins_default,
)
from transformer_analysis.model_registry import (
    get_model_config,
    extract_weight_map,
)


def process_model(
    model_name="pythia-70m-deduped",
    revision=None,
    idx_max=-1,
    out_dir="histos",
    cache_dir="./model_data",
    cleanup_downloads=False,
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
        model_config = get_model_config(model_name)

        revision_string = revision if revision else "main"

        cache_path = snapshot_download(
            repo_id=model_config.repo_id,
            revision=revision,
            cache_dir=f"{cache_dir}/{model_name}/{revision_string}",
            allow_patterns=model_config.allow_patterns,
        )
        hf_config = AutoConfig.from_pretrained(cache_path)

    logging.info(perf.log_report(context=model_name))

    # Phase 2: Configuration
    with perf.phase("configure"):
        logging.info("Configuring analysis...")

        config = SimpleNamespace()
        config.weight_type = ["W_Q", "W_K", "W_QK"]
        config.stats = stats_config_default.copy()
        config.w_bins = weight_bins_default.copy()
        config.sv_bins = sv_bins_default.copy()
        config.use_density = True
        config.n_heads = model_config.get_config_value(hf_config.__dict__, "n_heads")
        config.d_model = model_config.get_config_value(hf_config.__dict__, "d_model")
        config.n_layers = model_config.get_config_value(hf_config.__dict__, "n_layers")
        config.head_dim = config.d_model // config.n_heads

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

        # Get weight_map, needed if safetensors format unavailable and bin files are sharded
        weight_map = extract_weight_map(cache_path=cache_path)

        for layer_idx in range(n_layers):
            # qkv = state_dict[key].clone()
            W_Q, W_K, _ = model_config.extract_qkv(
                cache_path, layer_idx, d_model, weight_map
            )

            # For per-head analysis:
            W_Q_h = W_Q.reshape(n_heads, head_dim, d_model).float()
            W_K_h = W_K.reshape(n_heads, head_dim, d_model).float()

            hc = LayerHeadContainer(layer_idx, config)
            layer_input = {"W_Q": W_Q_h, "W_K": W_K_h}
            hc.analyze_layer(layer_input)
            layer_data.append(hc)
            del W_Q, W_K, W_Q_h, W_K_h
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
        if revision:
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
        out_prefix = f"{out_dir}/{model_name}_{revision_string}"
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

    # Phase 6: Cleanup
    with perf.phase("cleanup"):
        if cleanup_downloads:
            logging.info("Cleaning up downloads...")
            cache_path = f"{cache_dir}/{model_name}/{revision}"
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)

    disk_usage = shutil.disk_usage(cache_dir)
    logging.info(
        f"Disk usage: {disk_usage.used / (1024**3):.2f} GB / {disk_usage.total / (1024**3):.2f} GB"
    )
    logging.info(perf.log_report())

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("Performance Summary:")
    for phase_name in perf.phases.keys():
        logging.info(perf.log_report(phase=phase_name))
    logging.info("=" * 60)
    logging.info(f"Performance saved to {out_dir}/logs/perf_{job_id}.json")


def create_campaign(path, name, clobber=False, logs=True):
    base_dir = os.path.join(path, name)
    log_dir = os.path.join(base_dir, "logs")

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        os.makedirs(log_dir)
        return base_dir

    elif clobber:
        shutil.rmtree(base_dir)
        os.makedirs(base_dir)
        os.makedirs(log_dir)
        return base_dir

    return base_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")  # "pythia-70m-deduped")
    parser.add_argument("--out", type=str, default="Drive/ana-002")
    parser.add_argument("--cache", type=str, default="./model_data")
    parser.add_argument("--clobber", type=bool, default=False)
    parser.add_argument("--test", action="store_true", default=False)

    args = parser.parse_args()
    if args.test:
        print("=" * 20 + "Test option selected" + "=" * 20)
        print("\t\t" + "output and clobber options will be overwritten")
        args.out, args.clobber = "test", True
    cwd = os.getcwd()
    out_dir = create_campaign(path=cwd, name=args.out, clobber=args.clobber, logs=True)
    model_name = args.model

    model_config = get_model_config(args.model)
    revisions = model_config.revisions
    if args.test:
        revisions = revisions[-1:] if revisions else None
    else:
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        import warnings

        warnings.filterwarnings("ignore")

    # loop on checkpoints
    if revisions:
        for revision in tqdm(revisions):
            process_model(
                model_name=model_name,
                revision=revision,
                out_dir=out_dir,
                cache_dir=args.cache,
            )
    else:
        process_model(
            model_name=model_name, revision=None, out_dir=out_dir, cache_dir=args.cache
        )
