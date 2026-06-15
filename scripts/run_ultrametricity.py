#!/usr/bin/env python3
"""Ultrametricity stage: triple + clustering diagnostics on the Q_{hh'} matrices.

Pure post-processing on the correlations-stage .npz outputs (no model download).

Examples:
    python scripts/run_ultrametricity.py --model gpt2 \
        --corr-dir outputs/ana-006-open-production/correlations \
        --out-dir outputs/ana-006-open-production/ultrametricity
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import mlflow
from mlflow.entities import Dataset as DatasetEntity, DatasetInput

from transformer_analysis.ultrametricity import (
    to_distance,
    triple_statistics,
    cluster_analysis,
    quadruple_statistics,
    residual_decomposition,
    embedding_analysis,
    linkage_comparison,
    tight_pair_scores,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="")
    p.add_argument("--weight-type", default="W_QK")
    p.add_argument("--corr-dir", required=True)
    p.add_argument("--out-dir", default="ultrametricity")
    p.add_argument("--metrics", nargs="+",
                   default=["frob_cosine", "hist_jensen_shannon"])
    p.add_argument("--sample-cap", type=int, default=2_000_000)
    p.add_argument("--quad-sample-cap", type=int, default=1_000_000)
    p.add_argument("--max-dense", type=int, default=3000,
                   help="skip O(N^3) residual/MDS analyses above this many heads")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--mlflow-uri", default="file:./mlruns")
    p.add_argument("--mlflow-experiment", default="production")
    return p.parse_args()


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    args = parse_args()
    rev_label = args.revision or "main"
    prefix = f"{args.model}_{rev_label}_{args.weight_type}"
    corr_dir = Path(args.corr_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Q_npz = np.load(corr_dir / f"{prefix}_Q.npz")
    meta = json.loads((corr_dir / f"{prefix}_metadata.json").read_text())
    keys = [tuple(k) for k in meta["head_index"]]
    layers = [k[0] for k in keys]

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    run_name = f"{args.model}-{rev_label}-ultrametricity"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("mlflow.note.content",
                       "Ultrametricity (triple + clustering) on head-head overlaps Q_hh")
        mlflow.set_tag("model", args.model)
        mlflow.log_params({
            "model": args.model,
            "revision": rev_label,
            "weight_type": args.weight_type,
            "metrics": ",".join(args.metrics),
            "sample_cap": args.sample_cap,
            "tol": args.tol,
            "seed": args.seed,
            "out_dir": str(out_dir),
            "git_sha": _git_sha(),
        })

        t0 = time.time()
        summary = {"model": args.model, "revision": rev_label,
                   "weight_type": args.weight_type, "n_heads": meta["n_heads"],
                   "n_layers": meta["n_layers"], "metrics": {}}

        for metric in args.metrics:
            key = f"Q_{metric}"
            if key not in Q_npz:
                continue
            D = to_distance(Q_npz[key], metric)
            stats = triple_statistics(D, layers, sample_cap=args.sample_cap,
                                      seed=args.seed, tol=args.tol)
            clust = cluster_analysis(D, keys)
            quad = quadruple_statistics(D, sample_cap=args.quad_sample_cap,
                                        seed=args.seed, tol=args.tol)
            lk = linkage_comparison(D)
            tight = tight_pair_scores(D, keys, method="smallest", k_smallest=300,
                                      seed=args.seed)

            dense = D.shape[0] <= args.max_dense
            resid = residual_decomposition(D, clust["linkage"]) if dense else None
            emb = embedding_analysis(D) if dense else None

            np.savez_compressed(
                out_dir / f"{prefix}_{metric}_ultra.npz",
                head_index=np.array(keys),
                per_head_u=stats["per_head_u"],
                u_hist_edges=stats["u_hist_edges"],
                u_hist_counts=stats["u_hist_counts"],
                span_grid=stats["span_grid"],
                span_mean_u=stats["span_mean_u"],
                span_count=stats["span_count"],
                cooccur=stats["cooccur"],
                linkage=clust["linkage"],
                u4_hist_edges=quad["u4_hist_edges"],
                u4_hist_counts=quad["u4_hist_counts"],
                resid_top_eigs=resid["top_eigs"] if resid else np.array([]),
                resid_top_eigvecs=resid["top_eigvecs"] if resid else np.zeros((0, 0)),
                resid_eigvals_sorted=resid["eigvals_sorted"] if resid else np.array([]),
                mds_eigs=emb["eigenvalues"] if emb else np.array([]),
                mds_top_eigvecs=emb["top_eigvecs"] if emb else np.zeros((0, 0)),
                mds_top_eigs=emb["top_eigs"] if emb else np.array([]),
                tight_scores=tight["scores"],
                tight_null_scores=tight["null_scores"],
                tight_pair_idx=tight["pair_idx"],
            )

            summary["metrics"][metric] = {
                "mean_u": stats["mean_u"],
                "median_u": stats["median_u"],
                "frac_ultrametric": stats["frac_ultrametric"],
                "triangle_violation_frac": stats["triangle_violation_frac"],
                "cophenetic_corr": clust["cophenetic_corr"],
                "n_triples": stats["n_triples"],
                "sampled": stats["sampled"],
                "stratum_mean_u": stats["stratum_mean_u"],
                "residual_pairs": clust["residual_pairs"],
                "mean_u4": quad["mean_u4"],
                "median_u4": quad["median_u4"],
                "frac_additive": quad["frac_additive"],
                "delta_rel_mean": quad["delta_rel_mean"],
                "delta_rel_p95": quad["delta_rel_p95"],
                "n_quads": quad["n_quads"],
                "linkage_cophenetic": lk,
                "residual": resid and {k: resid[k] for k in
                    ("res_energy_frac", "eff_rank", "frac_top10",
                     "gini", "kurtosis", "verdict")},
                "embedding": emb and {k: emb[k] for k in
                    ("neg_energy_frac", "var_frac_2d", "var_frac_3d", "eff_dim")},
                "tight": {k: tight[k] for k in
                    ("method", "n_pairs", "score_median", "null_median", "low_flagged")},
            }
            log = {
                f"{metric}.mean_u": stats["mean_u"],
                f"{metric}.frac_ultrametric": stats["frac_ultrametric"],
                f"{metric}.cophenetic_corr": clust["cophenetic_corr"],
                f"{metric}.triangle_violation_frac": stats["triangle_violation_frac"],
                f"{metric}.mean_u4": quad["mean_u4"],
                f"{metric}.frac_additive": quad["frac_additive"],
                f"{metric}.tight_score_median": tight["score_median"],
            }
            if resid:
                log[f"{metric}.res_energy_frac"] = resid["res_energy_frac"]
            if emb:
                log[f"{metric}.mds_neg_energy_frac"] = emb["neg_energy_frac"]
            mlflow.log_metrics(log)

        (out_dir / f"{prefix}_ultrametricity_summary.json").write_text(
            json.dumps(summary, indent=2))
        mlflow.log_artifact(str(out_dir / f"{prefix}_ultrametricity_summary.json"))
        mlflow.log_metric("wall_time_s", time.time() - t0)

        digest = hashlib.md5(str(out_dir).encode()).hexdigest()[:8]
        entity = DatasetEntity(
            name=f"{args.model}-{rev_label}-ultrametricity",
            digest=digest, source_type="local",
            source=json.dumps({"uri": str(out_dir)}),
        )
        mlflow.MlflowClient().log_inputs(
            mlflow.active_run().info.run_id,
            [DatasetInput(dataset=entity, tags=[])],
        )


if __name__ == "__main__":
    main()
