from histogram_utils import load_group_from_file, extract_metrics_, to_dataframe
from datasets import Dataset, concatenate_datasets
import glob
import json

datasets = []
bins = None
sv_bins = None
hnames = set()

#dataset_name = 'gpt2_histos'
dataset_name = 'HFDS'

for f_in in glob.glob("histos/pythia-*.pkl"):
    print(f"Processing {f_in} ... " )
    # if 'W_QK' not in f_in:
    #     continue
    # if 'small' not in f_in:
    #     continue
    histo_group = load_group_from_file(f_in)
    for idx, histo_base_obj in histo_group:
        # print("INDEX ", idx)
        extract_metrics_(histo_base_obj)

    df = to_dataframe(histo_group)
    
    if bins is None:
        bins = df.attrs['bins']
    if sv_bins is None:
        sv_bins = df.attrs['sv_bins']

    hnames.update(df.attrs['histos'])
    
    parts = f_in.split('/')[-1].split('.')
    df['model'] = parts[0]
    df['weight_type'] = parts[1]
    ddf = Dataset.from_pandas(df)
    datasets.append(ddf)
combined = concatenate_datasets(datasets)
combined.info.description = "metadata.json"
combined.save_to_disk(dataset_name)
json.dump({'bins': bins, 'sv_bins' : sv_bins, 'histos' : list(hnames)}, open(f'{dataset_name}/metadata.json', 'w'))