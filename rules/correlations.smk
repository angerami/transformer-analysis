# Correlation analysis rule: head-head correlation matrices → .npz intermediates.
# This is an optional downstream stage; not part of the default `all` target.
# Run all correlations with:
#   snakemake correlations_all
# or a single run with:
#   snakemake done/<run_key>.correlations.done


def _corr_params(wildcards):
    run_key = wildcards.run_key
    for run in config["runs"]:
        model = run["model"]
        rev = run.get("revision") or None
        key = f"{model}_{rev}" if rev else model
        if key == run_key:
            return {"model": model, "revision": rev or ""}
    raise ValueError(f"No run config found for run_key={run_key!r}")


rule correlations:
    input:
        "done/{run_key}.transform.done",
    output:
        touch("done/{run_key}.correlations.done"),
    params:
        p=_corr_params,
        circuits=" ".join(config.get("correlation", {}).get("circuits", ["QK"])),
        metrics=" ".join(config.get("correlation", {}).get("metrics", ["frob_cosine"])),
        cache_dir=config["cache_dir"],
        mlflow_uri=config["mlflow_uri"],
        mlflow_experiment=config["mlflow_experiment"],
    shell:
        """
        python scripts/run_correlations.py \
            --model {params.p[model]} \
            --revision "{params.p[revision]}" \
            --out-dir correlations \
            --cache-dir {params.cache_dir} \
            --circuits {params.circuits} \
            --metrics {params.metrics} \
            --mlflow-uri {params.mlflow_uri} \
            --mlflow-experiment {params.mlflow_experiment}
        """
