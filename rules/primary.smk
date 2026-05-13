# Primary analysis rule: download model weights → extract → compute stats → HF Dataset.
# Uses a sentinel touch file in done/ to signal completion (avoids Snakemake
# directory-output conflicts with the transform stage).


def _primary_params(wildcards):
    for run in config["runs"]:
        model = run["model"]
        rev = run.get("revision") or None
        key = f"{model}_{rev}" if rev else model
        if key == wildcards.run_key:
            device = config.get("device") or ""
            cleanup = config.get("cleanup_downloads", False)
            return {
                "model": model,
                "revision": rev or "",
                "device_flag": f"--device {device}" if device else "",
                "cleanup_flag": "--cleanup-downloads" if cleanup else "",
            }
    raise ValueError(f"No run config found for run_key={wildcards.run_key!r}")


rule primary:
    """Download model weights, extract per-head stats → HF Dataset. Stage 1 of 2."""
    output:
        touch("done/{run_key}.primary.done"),
    params:
        p=_primary_params,
        out_dir=config["output_dir"],
        cache_dir=config["cache_dir"],
        max_workers=config.get("max_workers", 4),
        mlflow_uri=config["mlflow_uri"],
        mlflow_experiment=config["mlflow_experiment"],
        weight_types_flag=lambda _: (
            "--weight-types " + " ".join(config["weight_types"])
            if config.get("weight_types") else ""
        ),
    shell:
        """
        python scripts/run_primary.py \
            --model {params.p[model]} \
            --revision "{params.p[revision]}" \
            --out-dir {params.out_dir} \
            --cache-dir {params.cache_dir} \
            --max-workers {params.max_workers} \
            --mlflow-uri {params.mlflow_uri} \
            --mlflow-experiment {params.mlflow_experiment} \
            {params.p[device_flag]} \
            {params.p[cleanup_flag]} \
            {params.weight_types_flag}
        """
