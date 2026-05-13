#!/usr/bin/env python3
"""
Merge pipeline stage.  Two modes:

  cross-model  Incrementally append new models to a shared merged dataset.
               Models already present in the output are skipped unless listed
               in --refresh.  Safe to re-run when a new model is added.

  checkpoints  Collapse all per-revision datasets for one model into a single
               dataset (e.g. all 154 Pythia steps → {model}_all_checkpoints).
               Delegates to weight_analysis.merge_versions().

Examples:
    # Add any new models from config runs to outputs/all_models
    python scripts/run_merge.py cross-model \
        --dataset-dirs outputs/gpt2 outputs/gpt2-medium outputs/pythia-70m-deduped_step143000 \
        --out-path outputs/all_models

    # Replace gpt2 entry (re-transform was run) and add a new model
    python scripts/run_merge.py cross-model \
        --dataset-dirs outputs/gpt2 outputs/gpt2-medium outputs/new-model \
        --out-path outputs/all_models \
        --refresh gpt2

    # Merge all Pythia-70m checkpoints
    python scripts/run_merge.py checkpoints \
        --model pythia-70m-deduped --out-dir outputs
"""

import argparse
import json
import os
import sys


# ── helpers ───────────────────────────────────────────────────────────────────

def _present_keys(merged_path):
    """Return set of (model, revision) tuples already in the merged dataset."""
    from datasets import load_from_disk
    if not os.path.exists(merged_path):
        return set()
    ds = load_from_disk(merged_path)
    df = ds.to_pandas()[["model"] + (["revision"] if "revision" in ds.column_names else [])]
    if "revision" in df.columns:
        return set(zip(df["model"], df["revision"].fillna("main")))
    return set((m, "main") for m in df["model"].unique())


def _run_key_from_dir(dataset_dir):
    """Infer (model, revision) from the dataset's own rows."""
    from datasets import load_from_disk
    ds = load_from_disk(dataset_dir)
    model = ds[0]["model"]
    revision = ds[0].get("revision", None) or "main"
    return model, revision


# ── cross-model incremental merge ─────────────────────────────────────────────

def cross_model_merge(dataset_dirs, out_path, refresh=()):
    """Append new (model, revision) entries to an existing merged dataset.

    dataset_dirs : list of per-run HF Dataset directories
    out_path     : path to the merged output dataset (created if absent)
    refresh      : model names whose existing rows should be replaced
    """
    from datasets import load_from_disk, concatenate_datasets

    refresh = set(refresh)
    present = _present_keys(out_path)

    # Load existing merged dataset, dropping any models that need refreshing
    existing_ds = None
    if os.path.exists(out_path):
        existing_ds = load_from_disk(out_path)
        if refresh:
            existing_df = existing_ds.to_pandas()
            existing_df = existing_df[~existing_df["model"].isin(refresh)]
            from datasets import Dataset
            existing_ds = Dataset.from_pandas(existing_df, preserve_index=False)
            # Update present set to reflect removed rows
            present = {(m, r) for m, r in present if m not in refresh}

    to_add = []
    for d in dataset_dirs:
        if not os.path.isdir(d):
            print(f"  SKIP (not found): {d}")
            continue
        model, revision = _run_key_from_dir(d)
        key = (model, revision)
        if key in present:
            print(f"  SKIP (already merged): {model} @ {revision}")
            continue
        print(f"  ADDING: {model} @ {revision}")
        to_add.append(load_from_disk(d))

    if not to_add:
        print("Nothing new to merge.")
        return

    parts = ([existing_ds] if existing_ds is not None else []) + to_add
    merged = concatenate_datasets(parts)
    merged.info.description = "metadata.json"
    merged.save_to_disk(out_path)

    # Write/update metadata from first new dataset's metadata.json
    first_new_meta_path = os.path.join(to_add[0].info.description or "outputs",
                                       "metadata.json") if to_add else None
    src_meta = os.path.join(dataset_dirs[0], "metadata.json")
    dst_meta = os.path.join(out_path, "metadata.json")
    if os.path.exists(src_meta) and not os.path.exists(dst_meta):
        import shutil
        shutil.copy(src_meta, dst_meta)

    print(f"  Saved merged dataset ({len(merged)} rows) → {out_path}")


# ── checkpoint merge ───────────────────────────────────────────────────────────

def checkpoint_merge(model_name, out_dir):
    """Collapse all per-revision datasets for model_name into one dataset."""
    from transformer_analysis.weight_analysis import merge_versions
    print(f"Merging checkpoints for {model_name} …")
    merge_versions(model_name=model_name, path=out_dir)
    print(f"  Done → {out_dir}/{model_name}_all_checkpoints")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    cm = sub.add_parser("cross-model")
    cm.add_argument("--dataset-dirs", nargs="+", required=True,
                    help="Per-run HF Dataset directories to merge")
    cm.add_argument("--out-path", required=True,
                    help="Path for the merged output dataset")
    cm.add_argument("--refresh", nargs="*", default=[],
                    help="Model names to force-replace even if already present")

    cp = sub.add_parser("checkpoints")
    cp.add_argument("--model", required=True)
    cp.add_argument("--out-dir", default="outputs")

    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "cross-model":
        cross_model_merge(args.dataset_dirs, args.out_path, args.refresh)
    elif args.mode == "checkpoints":
        checkpoint_merge(args.model, args.out_dir)


if __name__ == "__main__":
    main()
