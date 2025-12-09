import json
from tqdm import tqdm
from datasets import concatenate_datasets, load_from_disk
from transformer_analysis.histogram_utils import get_model_versions

META_FILE = "metadata.json"
META_MERGE_KEY = "merged"


def write_dataset_and_metadata(ds_list, metadata, ds_name):
    combined_ds = concatenate_datasets(ds_list)
    combined_ds.info.description = META_FILE
    combined_ds.save_to_disk(ds_name)
    with open(f"{ds_name}/{META_FILE}", "w") as f:
        json.dump(metadata, f, indent=2)


def merge_versions(
    model_name="pythia-70m-deduped", path="histos", suffix="all_checkpoints"
):
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

    combined_metadata.update({META_MERGE_KEY : merged_dict})
    write_dataset_and_metadata(ds_list, combined_metadata, f"{path}/{out_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--path", type=str, default="histos")
    parser.add_argument("--out-name", type=str, default="weight_study")
    parser.add_argument("--suffix", type=str, default="all_checkpoints")
    args = parser.parse_args()

    if args.model is not None:
        merge_versions(model_name=args.model, path=args.path, suffix=args.suffix)

    else:
        from pathlib import Path

        path = Path(args.path)
        model_list = []
        for d in path.glob("*/"):
            if args.out_name in d.name or 'logs' in d.name:
                continue
            model_list.append(d.name)
        merge_datasets(model_list, path=args.path, out_name=args.out_name)
