from datasets import load_from_disk
import json
import numpy as np
import pandas as pd
from deepdiff import DeepDiff
import argparse

def diff_datasets(name1, name2, path=None, topN=20):
    pd.set_option('display.precision', 4)
    if isinstance(path, str):
        name1 = f'{path}/{name1}'
        name2 = f'{path}/{name2}'

    # Load datasets and metadata
    ds1 = load_from_disk(name1)
    with open(f'{name1}/{ds1.info.description}') as f:
        meta1 = json.load(f)

    ds2 = load_from_disk(name2)
    with open(f'{name2}/{ds2.info.description}') as f:
        meta2 = json.load(f)

    df1 = ds1.to_pandas()
    df2 = ds2.to_pandas()

    # Track columns to exclude from comparison
    exclude_cols = {'job_uuid', 'job_id', 'SVD'}

    # Check for identical job ids (unexpected)
    if df1['job_uuid'].equals(df2['job_uuid']):
        print("WARNING: job_uuid columns are identical - verify this is intended")

    # Compare w_bins
    bins1 = np.array(meta1.get("w_bins", []))
    bins2 = np.array(meta2.get("w_bins", []))
    if not np.array_equal(bins1, bins2):
        print("w_bins DIFFER:")
        print(f"  1: n={len(bins1)-1}, range=[{bins1[0]:.4g}, {bins1[-1]:.4g}]")
        print(f"  2: n={len(bins2)-1}, range=[{bins2[0]:.4g}, {bins2[-1]:.4g}]")
        exclude_cols.add('P_w')

    # Compare sv_bins
    sv_bins1 = np.array(meta1.get("sv_bins", []))
    sv_bins2 = np.array(meta2.get("sv_bins", []))
    if not np.array_equal(sv_bins1, sv_bins2):
        print("sv_bins DIFFER:")
        print(f"  1: n={len(sv_bins1)-1}, range=[{sv_bins1[0]:.4g}, {sv_bins1[-1]:.4g}]")
        print(f"  2: n={len(sv_bins2)-1}, range=[{sv_bins2[0]:.4g}, {sv_bins2[-1]:.4g}]")
        exclude_cols.add('P_sv')

    # Compare stats keys (column presence)
    stats1 = set(meta1.get("stats", {}).keys())
    stats2 = set(meta2.get("stats", {}).keys())
    common_stats = stats1 & stats2
    
    if stats1 != stats2:
        print(f"stats columns DIFFER:")
        if stats1 - stats2:
            print(f"  Only in 1: {stats1 - stats2}")
        if stats2 - stats1:
            print(f"  Only in 2: {stats2 - stats1}")
        exclude_cols.update(stats1 ^ stats2)  # exclude non-common stats columns

    # Compare remaining metadata
    meta1_rest = {k: v for k, v in meta1.items() if k not in ("w_bins", "sv_bins", "stats")}
    meta2_rest = {k: v for k, v in meta2.items() if k not in ("w_bins", "sv_bins", "stats")}
    meta_diff = DeepDiff(meta1_rest, meta2_rest, ignore_order=True)
    if meta_diff:
        print("Other metadata differences:")
        print(meta_diff.to_json(indent=2))

    # Compare dataframes
    cols1 = set(df1.columns) - exclude_cols
    cols2 = set(df2.columns) - exclude_cols
    common_cols = sorted(cols1 & cols2)

    if cols1 != cols2:
        print(f"DataFrame columns differ (excluding ignored):")
        if cols1 - cols2:
            print(f"  Only in 1: {cols1 - cols2}")
        if cols2 - cols1:
            print(f"  Only in 2: {cols2 - cols1}")

    # Numerical comparison on common columns
    df1_cmp = df1[common_cols].reset_index(drop=True)
    df2_cmp = df2[common_cols].reset_index(drop=True)

    if df1_cmp.shape != df2_cmp.shape:
        print(f"Shape mismatch: {df1_cmp.shape} vs {df2_cmp.shape}")
        return

    if df1_cmp.equals(df2_cmp):
        print(f"DataFrames match on {len(common_cols)} compared columns.")
    else:
        ###
        ### Main diff command
        diff = df1_cmp.compare(df2_cmp, result_names=("reference", "target"))
        print(f"DataFrame differences ({len(diff)} rows differ):")
        if topN < 0:
            print(diff.head(20))
        else:
            print(diff.head(topN))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diff two HuggingFace datasets",
        epilog="""Examples:
  %(prog)s ds_v1 ds_v2 --path /data/results
  %(prog)s /full/path/ds_v1 /full/path/ds_v2
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reference", help="Reference dataset name")
    parser.add_argument("target", help="Target dataset name")
    parser.add_argument("--path", "-p", default=None, help="Base path for datasets")
    parser.add_argument("--num-print", "-n", default=50, help="Max number of rows to print. -1 prints all")
    
    args = parser.parse_args()
    diff_datasets(args.reference, args.target, path=args.path,topN=args.num_print)