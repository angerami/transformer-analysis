# Transform rule: re-run derived metrics on an existing primary HF Dataset.
# Re-run alone with: snakemake done/<run_key>.transform.done --forcerun transform -j<N>


def _transform_params(wildcards):
    for run in config["runs"]:
        model = run["model"]
        rev = run.get("revision") or None
        key = f"{model}_{rev}" if rev else model
        if key == wildcards.run_key:
            return {"model": model, "revision": rev or ""}
    raise ValueError(f"No run config found for run_key={wildcards.run_key!r}")


rule transform:
    input:
        "done/{run_key}.primary.done",
    output:
        touch("done/{run_key}.transform.done"),
    params:
        p=_transform_params,
        out_dir=config["output_dir"],
        mlflow_uri=config["mlflow_uri"],
        mlflow_experiment=config["mlflow_experiment"],
        drop_flag=lambda _: (
            "--drop-columns " + " ".join(config.get("transform", {}).get("drop_columns", []))
            if config.get("transform", {}).get("drop_columns")
            else ""
        ),
    shell:
        """
        python scripts/run_transform.py \
            --model {params.p[model]} \
            --revision "{params.p[revision]}" \
            --dataset-dir {params.out_dir}/{wildcards.run_key} \
            --out-dir {params.out_dir}/{wildcards.run_key}_refined \
            --mlflow-uri {params.mlflow_uri} \
            --mlflow-experiment {params.mlflow_experiment} \
            {params.drop_flag}
        """
