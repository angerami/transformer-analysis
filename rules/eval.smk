# Eval rule: compute perplexity for each (model, revision) → per-run parquet.
# A merge_eval rule then concatenates all per-run files into the final artifact.
# Safe to run in parallel (-j8); no shared write during the eval stage.
#
# Run everything (eval + merge) with:
#   snakemake eval_all -j8
# Single run:
#   snakemake outputs/eval/gpt2.parquet
# Merge only (re-run after adding models):
#   snakemake outputs/eval_metrics/eval_metrics.parquet --forcerun merge_eval


def _eval_params(wildcards):
    for run in config["runs"]:
        model = run["model"]
        rev = run.get("revision") or None
        key = f"{model}_{rev}" if rev else model
        if key == wildcards.run_key:
            device = config.get("device") or ""
            eval_cfg = config.get("eval", {})
            corpus = eval_cfg.get("corpus", "wikitext103")
            pile_cache = eval_cfg.get("pile_cache") or ""
            pile_tokens = eval_cfg.get("pile_tokens", 204800)
            return {
                "model": model,
                "revision": rev or "",
                "corpus": corpus,
                "pile_tokens": pile_tokens,
                "device_flag": f"--device {device}" if device else "",
                "pile_cache_flag": f"--pile-cache {pile_cache}" if pile_cache else "",
            }
    raise ValueError(f"No run config found for run_key={wildcards.run_key!r}")


rule eval:
    input:
        "done/{run_key}.transform.done",
    output:
        config["output_dir"] + "/eval/{run_key}.parquet",
    params:
        p=_eval_params,
        cache_dir=config["cache_dir"],
        mlflow_uri=config["mlflow_uri"],
        mlflow_experiment=config["mlflow_experiment"],
    shell:
        """
        python scripts/run_eval.py \
            --model {params.p[model]} \
            --revision "{params.p[revision]}" \
            --corpus {params.p[corpus]} \
            --pile-tokens {params.p[pile_tokens]} \
            --out {output} \
            --cache {params.cache_dir} \
            --mlflow-uri {params.mlflow_uri} \
            --mlflow-experiment {params.mlflow_experiment} \
            {params.p[device_flag]} \
            {params.p[pile_cache_flag]}
        """


rule merge_eval:
    input:
        expand(config["output_dir"] + "/eval/{run_key}.parquet",
               run_key=[_run_key(r) for r in config["runs"]]),
    output:
        config.get("eval", {}).get("out", "outputs/eval_metrics/eval_metrics.parquet"),
    run:
        import pandas as pd
        pd.concat([pd.read_parquet(f) for f in input], ignore_index=True).to_parquet(
            output[0], index=False
        )
        print(f"Merged {len(input)} runs → {output[0]}")
