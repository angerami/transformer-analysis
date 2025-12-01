import os
from tqdm import tqdm
from transformers import logging as hf_logging
from transformer_analaysisrun_weight_analysis import process_model, create_versioned_dir
from transformer_analaysismodel_registry import MODEL_CONFIGS

def main(out_dir='Drive/ana-002', clobber=False):
    cwd = os.getcwd()

    hf_logging.set_verbosity_error()
    import warnings
    warnings.filterwarnings('ignore')

    #for now get mistral and llama models
    models=[k for k in MODEL_CONFIGS.keys() if 'tral-' in k] #mistral and mixtral
    models.extend([k for k in MODEL_CONFIGS.keys() if 'llama-' in k])

    for model_name in tqdm(models):
        target_dir = os.path.join(out_dir, model_name)
        if os.path.exists(target_dir):
            if clobber:
                print(f'Model = {model_name} output exists as {target_dir}. OVERWRITING.')
            else:
                print(f'Model = {model_name} output exists as {target_dir}. SKIPPING.')
                continue
        process_model(model_name=model_name, cache_dir='./downloads', revision=None, out_dir=out_dir,cleanup_downloads=True)

if __name__ == "__main__":
    main()