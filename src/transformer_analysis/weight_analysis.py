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
from transformer_analysis.device_utils import get_device


def process_model(
    model_name="pythia-70m-deduped",
    revision=None,
    idx_max=-1,
    out_dir="histos",
    cache_dir="./model_data",
    cleanup_downloads=False,
    low_rank_svd_approximation=False,
    top_k_svd=-1,
    resume_download=True,
    max_workers=4,
    device=None,
    skip_postprocess=False,
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

    device_str = str(get_device(device))
    logging.info(f"Using device: {device_str}")

    with perf.phase("load_model"):
        logging.info("Loading model...")
        model_config = get_model_config(model_name)

        revision_string = revision if revision else "main"

        cache_path = snapshot_download(
            repo_id=model_config.repo_id,
            revision=revision,
            cache_dir=f"{cache_dir}/{model_name}/{revision_string}",
            allow_patterns=model_config.allow_patterns,
            resume_download=resume_download,
            max_workers=max_workers,
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
        # SVD configuration options (passed from function parameters)
        config.low_rank_svd_approximation = low_rank_svd_approximation
        config.top_k_svd = top_k_svd

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

        for layer_idx in tqdm(range(n_layers), desc="Processing layers", leave=True):
            W_Q, W_K, _ = model_config.extract_qkv(
                cache_path, layer_idx, d_model, weight_map, device=device_str, qkv_scale_factor=model_config.qkv_scale_factor
            )

            # For per-head analysis:
            W_Q_h = W_Q.reshape(n_heads, head_dim, d_model).float()
            W_K_h = W_K.reshape(n_heads, head_dim, d_model).float()

            hc = LayerHeadContainer(
                layer_idx,
                config,
                low_rank_svd_approximation=config.low_rank_svd_approximation,
                top_k_svd=config.top_k_svd,
                device=device_str
            )
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
            if not skip_postprocess:
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


def reprocess_metrics(
    model_name: str,
    revision=None,
    all_revisions: bool = False,
    out_dir: str = "outputs",
    quiet: bool = False,
):
    """
    Reprocess existing datasets to update/add metrics without re-running full analysis.

    This function loads existing datasets, recomputes metrics (both weight histogram
    and singular value metrics), and overwrites the dataset with updated columns.

    Args:
        model_name: Name of the model to reprocess
        revision: Specific revision to reprocess (or None for main)
        all_revisions: Whether to process all available revisions
        out_dir: Output directory containing existing datasets
        quiet: Whether to suppress output
    """
    from datasets import load_from_disk, Dataset
    from transformer_analysis.histogram_utils import (
        normality_metrics,
        singular_value_metrics,
        get_model_versions,
    )
    import numpy as np

    if not quiet:
        print("\n" + "=" * 80)
        print(f"Reprocessing Metrics: {model_name}")
        print("=" * 80)

    # Determine which revisions to process
    if all_revisions:
        revisions = get_model_versions(model_name)
        if not revisions:
            print(f"Model {model_name} has no revisions defined. Processing main branch only.")
            revisions = [None]
    elif revision:
        revisions = [revision]
    else:
        revisions = [None]

    if not quiet:
        print(f"Revisions to process: {len(revisions)}")

    for rev in tqdm(revisions, desc=f"Reprocessing {model_name}", disable=quiet):
        revision_str = rev if rev else "main"

        # Determine dataset path
        if rev:
            dataset_path = os.path.join(out_dir, f"{model_name}_{revision_str}")
        else:
            dataset_path = os.path.join(out_dir, model_name)

        if not os.path.exists(dataset_path):
            print(f"  WARNING: Dataset not found at {dataset_path}, skipping...")
            continue

        if not quiet:
            print(f"\n  Reprocessing: {model_name} @ {revision_str}")

        try:
            # Load existing dataset
            ds = load_from_disk(dataset_path)
            df = ds.to_pandas()

            # Load metadata to get bin information
            metadata_path = os.path.join(dataset_path, "metadata.json")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            w_bins = np.array(metadata["w_bins"])
            centers = (w_bins[:-1] + w_bins[1:]) / 2

            if not quiet:
                print(f"    Loaded dataset with {len(df)} rows")

            # Process each row
            new_columns = {}
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Processing rows", disable=quiet, leave=False):
                # Create a dictionary for this row (simulating the 'h' dict)
                h = row.to_dict()

                # Apply weight histogram metrics
                for metric_func in normality_metrics.values():
                    metric_func(h, centers)

                # Apply singular value metrics if SVD data exists
                if "SVD" in h and h["SVD"] is not None and not pd.isna(h["SVD"]).all():
                    svd_array = h["SVD"]
                    if isinstance(svd_array, (list, np.ndarray)) and len(svd_array) > 0:
                        for metric_func in singular_value_metrics.values():
                            metric_func(h, svd_array)

                # Store new metric values
                for key, value in h.items():
                    if key not in row or row[key] != value:
                        if key not in new_columns:
                            new_columns[key] = [None] * len(df)
                        new_columns[key][idx] = value

            # Add new columns to dataframe
            for col_name, col_values in new_columns.items():
                df[col_name] = col_values
                if not quiet:
                    print(f"    Added/updated column: {col_name}")

            # Save updated dataset
            updated_ds = Dataset.from_pandas(df)
            updated_ds.info.description = "metadata.json"
            updated_ds.save_to_disk(dataset_path)

            if not quiet:
                print(f"    ✓ Saved updated dataset to {dataset_path}")

        except Exception as e:
            print(f"  ERROR reprocessing {model_name} @ {revision_str}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not quiet:
        print("\n" + "=" * 80)
        print(f"Completed reprocessing: {model_name}")
        print("=" * 80 + "\n")


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


def write_dataset_and_metadata(ds_list, metadata, ds_name):
    """Write combined dataset and metadata to disk.

    Args:
        ds_list: List of datasets to concatenate
        metadata: Metadata dictionary to write
        ds_name: Output directory name for the dataset
    """
    from datasets import concatenate_datasets

    combined_ds = concatenate_datasets(ds_list)
    combined_ds.info.description = "metadata.json"
    combined_ds.save_to_disk(ds_name)
    with open(f"{ds_name}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def merge_versions(
    model_name="pythia-70m-deduped", path="histos", suffix="all_checkpoints"
):
    """Merge all checkpoint versions of a single model into one dataset.

    Args:
        model_name: Name of the model to merge
        path: Base directory containing the datasets
        suffix: Suffix for the output merged dataset
    """
    from datasets import load_from_disk
    from transformer_analysis.histogram_utils import get_model_versions

    ds_list = []
    metadata = None
    for rev in tqdm(get_model_versions(model_name), desc=f"Processing {model_name}"):
        pattern = f"{model_name}_{rev}"
        ds = load_from_disk(f"{path}/{pattern}")
        ds_list.append(ds)
        if metadata is None:
            with open(f"{path}/{pattern}/{ds.info.description}") as f:
                metadata = json.load(f)
    write_dataset_and_metadata(ds_list, metadata, f"{path}/{model_name}_{suffix}")


def merge_datasets(model_name_list, path="histos", out_name="merged", suffix=None):
    """Merge multiple model datasets into a single combined dataset.

    Args:
        model_name_list: List of model names (or patterns) to merge
        path: Base directory containing the datasets
        out_name: Name for the output merged dataset
        suffix: Optional suffix to append to each model name pattern
    """
    from datasets import load_from_disk

    META_MERGE_KEY = "merged"

    ds_list = []
    combined_metadata = None
    merged_dict = {}
    for model_name in tqdm(model_name_list, desc="Processing models"):
        pattern = model_name
        if suffix is not None and isinstance(str, suffix):
            pattern += "_" + suffix
        ds = load_from_disk(f"{path}/{pattern}")
        ds_list.append(ds)

        # now the metadata
        mf = f"{path}/{pattern}/{ds.info.description}"
        with open(mf) as f:
            metadata = json.load(f)
        if combined_metadata is None:
            combined_metadata = {
                k: v for k, v in metadata.items() if k != META_MERGE_KEY
            }
        model_name = model_name.rstrip("_main")
        if META_MERGE_KEY in metadata:  # Flatten
            for k, v in metadata[META_MERGE_KEY].items():
                key = k
                while key in merged_dict:  # make key name unique
                    key = f"{model_name}_{key}"
                merged_dict[key] = v
        else:
            merged_dict[model_name] = metadata

    combined_metadata.update({META_MERGE_KEY: merged_dict})
    write_dataset_and_metadata(ds_list, combined_metadata, f"{path}/{out_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")  # "pythia-70m-deduped")
    parser.add_argument("--out", type=str, default="Drive/ana-002")
    parser.add_argument("--cache", type=str, default="./model_data")
    parser.add_argument("--clobber", type=bool, default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--low-rank-svd", action="store_true", default=False, dest="low_rank_svd")
    parser.add_argument("--top-k-svd", type=int, default=-1, dest="top_k_svd")
    parser.add_argument("--resume-download", action="store_true", default=True, dest="resume_download")
    parser.add_argument("--no-resume-download", action="store_false", dest="resume_download")
    parser.add_argument("--max-workers", type=int, default=4, dest="max_workers")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])

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
                low_rank_svd_approximation=args.low_rank_svd,
                top_k_svd=args.top_k_svd,
                resume_download=args.resume_download,
                max_workers=args.max_workers,
                device=args.device,
            )
    else:
        process_model(
            model_name=model_name,
            revision=None,
            out_dir=out_dir,
            cache_dir=args.cache,
            low_rank_svd_approximation=args.low_rank_svd,
            top_k_svd=args.top_k_svd,
            resume_download=args.resume_download,
            max_workers=args.max_workers,
            device=args.device,
        )
