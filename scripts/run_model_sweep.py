import os
from transformer_analysis.run_weight_analysis import process_model, create_versioned_dir
from transformer_analysis.model_registry import MODEL_CONFIGS
from tqdm import tqdm
from transformers import logging as hf_logging
import warnings


def model_sweep(model_list, out_dir='Drive/ana-002', clobber=False, cache_dir = './downloads'):
    hf_logging.set_verbosity_error()
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
        process_model(model_name=model_name, cache_dir=cache_dir, revision=None, out_dir=out_dir, cleanup_downloads=True)
        print('\n' + '-'*40 + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default='Drive/ana-002')
    parser.add_argument("--cache", type=str, default='./model_data')
    parser.add_argument("--clobber", type=bool, default=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()

    models=[k for k in MODEL_CONFIGS.keys() if 'tral-' in k] #mistral and mixtral
    models.extend([k for k in MODEL_CONFIGS.keys() if 'llama-' in k])

    out_dir = create_campaign(args.out, clobber=args.clobber)
    model_sweep(model_list=models, out_dir=out_dir, clobber=args.clobber, cache_dir=args.cache):