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
    """Compute head-head correlation matrices from model weights → .npz. Optional stage."""
    input:
        config["experiment_dir"] + "/done/{run_key}.transform.done",
    output:
        touch(config["experiment_dir"] + "/done/{run_key}.correlations.done"),
    params:
        p=_corr_params,
        circuits=" ".join(config.get("correlation", {}).get("circuits", ["QK"])),
        metrics=" ".join(config.get("correlation", {}).get("metrics", ["frob_cosine"])),
        equilibrium="--equilibrium" if config.get("correlation", {}).get("equilibrium", False) else "",
        corr_dir=config["experiment_dir"] + "/correlations",
        cache_dir=config["cache_dir"],
        mlflow_uri=config["mlflow_uri"],
        mlflow_experiment=config["mlflow_experiment"],
    shell:
        """
        python scripts/run_correlations.py \
            --model {params.p[model]} \
            --revision "{params.p[revision]}" \
            --out-dir {params.corr_dir} \
            --cache-dir {params.cache_dir} \
            --circuits {params.circuits} \
            --metrics {params.metrics} \
            {params.equilibrium} \
            --mlflow-uri {params.mlflow_uri} \
            --mlflow-experiment {params.mlflow_experiment}
        """
