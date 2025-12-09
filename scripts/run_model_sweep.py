import os
from transformer_analysis.run_weight_analysis import process_model, create_versioned_dir
from transformer_analysis.model_registry import MODEL_CONFIGS

def model_sweep(model_list, out_dir='Drive/ana-002', clobber=False):
    from tqdm import tqdm
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
    import warnings
    warnings.filterwarnings('ignore')

    for model_name in tqdm(model_list):

        print('\n' + '-'*40 + '\n')
        print(f'Processing Model = {model_name}')
        print('\n' + '-'*40 + '\n')

        target_dir = os.path.join(out_dir, model_name)
        if os.path.exists(target_dir):
            if clobber:
                print(f'Model = {model_name} output exists as {target_dir}. OVERWRITING.')
            else:
                print(f'Model = {model_name} output exists as {target_dir}. SKIPPING.')
                continue
        process_model(model_name=model_name, cache_dir='./downloads', revision=None, out_dir=out_dir,cleanup_downloads=True)
        print('\n' + '-'*40 + '\n')

if __name__ == "__main__":

    models=[k for k in MODEL_CONFIGS.keys() if 'tral-' in k] #mistral and mixtral
    models.extend([k for k in MODEL_CONFIGS.keys() if 'llama-' in k])
    model_sweep(model_list=models, out_dir='Drive/ana-002', clobber=False):