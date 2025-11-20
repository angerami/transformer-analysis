from datasets import load_from_disk, concatenate_datasets
import json

MODEL_NAME = "pythia-70m-deduped"
PREFIX = 'histos_4'
SUFFIX = 'all_checkpoints'
META_FILE = 'metadata.json'
PYTHIA_REVISIONS = [
    "step0", "step1", "step2", "step4", "step8", "step16", "step32", 
    "step64", "step128", "step256", "step512",
] + [f"step{step}" for step in range(1000, 144000, 1000)]

ds_list = []
metadata_list = []
for rev in PYTHIA_REVISIONS:
    pattern =  f'{MODEL_NAME}_{rev}'
    print(f'Merging {pattern}')
    ds = load_from_disk(f'{PREFIX}/{pattern}')
    ds_list.append(ds)
    mf = f'{PREFIX}/{pattern}/{ds.info.description}'
    with open(mf) as f:
        x = json.load(f)
        #x.update({"model_name" : MODEL_NAME})#, "revision" : rev})
        metadata_list.append(x)

combined_ds = concatenate_datasets(ds_list)
combined_ds.info.description = META_FILE
combined_ds.save_to_disk(f'{PREFIX}/{MODEL_NAME}_{SUFFIX}')

# with open(f'{PREFIX}/{MODEL_NAME}_{SUFFIX}/{META_FILE}', 'w') as f:
#     json.dump(metadata_list, f, indent=2)
#For now only need to copy not combine
with open(f'{PREFIX}/{MODEL_NAME}_{SUFFIX}/{META_FILE}', 'w') as f:
    json.dump(metadata_list[0], f, indent=2)

