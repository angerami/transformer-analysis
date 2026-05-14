#!/usr/bin/env python3
"""
Compare full SVD (nominal, from disk) vs d_head-truncated SVD (recomputed).

For each head, evaluates:
  - Compute time: loop phase elapsed seconds from perf logs
  - Singular value accuracy: RMSE and relative error of the top d_head SVs
    against the nominal result

Usage:
    python scripts/experiment_svd_truncation.py \
        --models gpt2 pythia-2.8b-deduped \
        --nominal-dirs outputs/ana-005-open-production/gpt2_main \
                       outputs/ana-005-open-production/pythia-2.8b-deduped_step143000 \
        --out-dir outputs/experiment_svd_truncation
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from datasets import load_from_disk

from transformer_analysis.head_pipeline import process_model


def find_perf_log(dataset_dir):
    """Return the perf JSON for a dataset, matching by job_id stored in the data."""
    meta_path = os.path.join(dataset_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    logs_dir = os.path.join(os.path.dirname(dataset_dir), "..", "logs")
    logs_dir = os.path.normpath(logs_dir)
    if not os.path.isdir(logs_dir):
        return None, None

    # Try to find by job_id in dataset rows
    ds = load_from_disk(dataset_dir)
    job_ids = set(r["job_id"] for r in ds if r.get("job_id"))
    if not job_ids:
        return None, None

    for job_id in job_ids:
        candidate = os.path.join(logs_dir, f"perf_{job_id}.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f), job_id

    return None, None


def compare_svd(nominal_ds, trunc_ds, d_head):
    """
    For each (layer, head, weight_type) compare top-d_head singular values.
    Returns a DataFrame with per-head RMSE and relative error.
    """
    def index_ds(ds):
        idx = {}
        for row in ds:
            if row["SVD"] is None:
                continue
            key = (row["layer"], row["head"], row["weight_type"])
            idx[key] = np.array(row["SVD"])
        return idx

    nom_idx = index_ds(nominal_ds)
    trunc_idx = index_ds(trunc_ds)

    records = []
    for key in nom_idx:
        if key not in trunc_idx:
            continue
        layer, head, wt = key
        sv_nom = nom_idx[key][:d_head]   # top d_head from full SVD
        sv_trunc = trunc_idx[key][:d_head]

        # pad shorter side if needed (shouldn't be, but defensive)
        n = min(len(sv_nom), len(sv_trunc))
        sv_nom, sv_trunc = sv_nom[:n], sv_trunc[:n]

        rmse = float(np.sqrt(np.mean((sv_nom - sv_trunc) ** 2)))
        denom = float(np.mean(np.abs(sv_nom))) if np.any(sv_nom) else 1.0
        rel_err = rmse / denom if denom > 0 else float("nan")

        records.append({
            "layer": layer,
            "head": head,
            "weight_type": wt,
            "rmse": rmse,
            "rel_err": rel_err,
            "sv_nom_max": float(sv_nom[0]) if len(sv_nom) > 0 else float("nan"),
            "sv_trunc_max": float(sv_trunc[0]) if len(sv_trunc) > 0 else float("nan"),
        })

    return pd.DataFrame(records)


def run_one(model, nominal_dir, out_dir, cache_dir, device, max_workers):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Model: {model}")
    print(f"{'=' * 60}")

    # Load nominal dataset and metadata
    print(f"Loading nominal dataset from {nominal_dir}")
    nominal_ds = load_from_disk(nominal_dir)
    with open(os.path.join(nominal_dir, "metadata.json")) as f:
        nominal_meta = json.load(f)

    d_head = nominal_meta.get("d_head") or nominal_meta.get("head_dim")
    d_model = nominal_meta.get("d_model")
    n_heads = nominal_meta.get("n_heads")
    print(f"  d_model={d_model}  n_heads={n_heads}  d_head={d_head}")

    # Match the revision used in the nominal run so weights are identical
    nominal_revision = None
    if "revision" in nominal_ds.column_names:
        revisions = [r for r in nominal_ds["revision"] if r]
        nominal_revision = revisions[0] if revisions else None
    if nominal_revision:
        print(f"  Nominal revision: {nominal_revision}")

    # Recover nominal loop time from perf log
    nominal_perf, nominal_job_id = find_perf_log(nominal_dir)
    nominal_loop_s = nominal_perf.get("loop_elapsed_sec") if nominal_perf else None
    if nominal_loop_s:
        print(f"  Nominal loop time: {nominal_loop_s:.1f}s (job {nominal_job_id})")
    else:
        print("  Nominal loop time: not found in perf logs")

    # Run truncated pipeline using same revision as nominal
    rev_label = nominal_revision or "main"
    trunc_out = os.path.join(out_dir, f"{model}_{rev_label}")
    print(f"\nRunning d_head-truncated pipeline → {trunc_out}")
    t0 = time.time()
    process_model(
        model_name=model,
        revision=nominal_revision,
        out_dir=out_dir,
        cache_dir=cache_dir,
        top_k_svd_d_head=True,
        max_workers=max_workers,
        device=device,
        skip_postprocess=False,
    )
    trunc_wall_s = time.time() - t0

    # Load truncated result
    trunc_ds = load_from_disk(trunc_out)
    trunc_perf, trunc_job_id = find_perf_log(trunc_out)
    trunc_loop_s = trunc_perf.get("loop_elapsed_sec") if trunc_perf else trunc_wall_s

    # Compare
    print("\nComparing singular values...")
    cmp_df = compare_svd(nominal_ds, trunc_ds, d_head)
    cmp_df["model"] = model

    results = {
        "model": model,
        "d_head": d_head,
        "d_model": d_model,
        "n_heads": n_heads,
        "nominal_loop_s": nominal_loop_s,
        "trunc_loop_s": trunc_loop_s,
        "speedup": (nominal_loop_s / trunc_loop_s) if nominal_loop_s and trunc_loop_s else None,
    }

    if not cmp_df.empty:
        for wt, grp in cmp_df.groupby("weight_type"):
            results[f"{wt}_rmse_mean"] = float(grp["rmse"].mean())
            results[f"{wt}_rmse_max"] = float(grp["rmse"].max())
            results[f"{wt}_rel_err_mean"] = float(grp["rel_err"].mean())

    # Print summary
    print(f"  Nominal loop:   {nominal_loop_s:.1f}s" if nominal_loop_s else "  Nominal loop:   n/a")
    print(f"  Truncated loop: {trunc_loop_s:.1f}s")
    if results.get("speedup"):
        print(f"  Speedup:        {results['speedup']:.2f}x")
    if not cmp_df.empty:
        print(cmp_df.groupby("weight_type")[["rmse", "rel_err"]].mean().to_string())

    return results, cmp_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--nominal-dirs", nargs="+", required=True, dest="nominal_dirs",
                   help="Paths to existing full-SVD datasets on disk, one per model")
    p.add_argument("--out-dir", default="outputs/experiment_svd_truncation")
    p.add_argument("--cache-dir", default="./model_data")
    p.add_argument("--device", default=None)
    p.add_argument("--max-workers", type=int, default=4)
    args = p.parse_args()

    if len(args.models) != len(args.nominal_dirs):
        p.error("--models and --nominal-dirs must have the same number of entries")

    all_results = []
    all_cmp = []

    for model, nominal_dir in zip(args.models, args.nominal_dirs):
        results, cmp_df = run_one(
            model=model,
            nominal_dir=nominal_dir,
            out_dir=args.out_dir,
            cache_dir=args.cache_dir,
            device=args.device,
            max_workers=args.max_workers,
        )
        all_results.append(results)
        all_cmp.append(cmp_df)

    # Save combined outputs
    out_json = os.path.join(args.out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(all_results if len(all_results) > 1 else all_results[0], f, indent=2)
    print(f"\nResults written to {out_json}")

    combined_cmp = pd.concat(all_cmp, ignore_index=True)
    out_csv = os.path.join(args.out_dir, "per_head_comparison.csv")
    combined_cmp.to_csv(out_csv, index=False)
    print(f"Per-head comparison written to {out_csv}")


if __name__ == "__main__":
    main()
