import json
from tqdm import tqdm
from datasets import concatenate_datasets, load_from_disk
from transformer_analaysis.histogram_utils import get_model_versions

SUFFIX = "all_checkpoints"
META_FILE = "metadata.json"

def merge_versions(model_name = 'pythia-70m-deduped', path = 'histos'):
    ds_list = []
    metadata_list = []
    for rev in tqdm(get_model_versions(model_name), desc=f'Processing {model_name}'):
        pattern = f"{model_name}_{rev}"
        ds = load_from_disk(f"{path}/{pattern}")
        ds_list.append(ds)
        mf = f"{path}/{pattern}/{ds.info.description}"
        with open(mf) as f:
            x = json.load(f)
            metadata_list.append(x)
    return ds_list, metadata_list

def write_dataset_and_metadata(ds_list, metadata_list, ds_name):
    combined_ds = concatenate_datasets(ds_list)
    combined_ds.info.description = META_FILE
    combined_ds.save_to_disk(ds_name)
    with open(f"{ds_name}/{META_FILE}", "w") as f:
        json.dump(metadata_list[0], f, indent=2)



if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pythia-70m-deduped")
    parser.add_argument("--path", type=str, default='histos')
    args = parser.parse_args()
    
    ds_list, metadata_list = merge_versions(model_name=args.model, path=args.path)
    write_dataset_and_metadata(ds_list, metadata_list, f"{args.path}/{args.model}_{SUFFIX}")
