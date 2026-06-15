# Ultrametricity rule: triple + clustering diagnostics on the correlation Q matrices.
# Pure post-processing on the correlations stage; not part of the default `all` target.
# Run all with:
#   snakemake ultrametricity_all
# or a single run with:
#   snakemake done/<run_key>.ultrametricity.done


def _ultra_params(wildcards):
    run_key = wildcards.run_key
    for run in config["runs"]:
        model = run["model"]
        rev = run.get("revision") or None
        key = f"{model}_{rev}" if rev else model
        if key == run_key:
            return {"model": model, "revision": rev or ""}
    raise ValueError(f"No run config found for run_key={run_key!r}")


rule ultrametricity:
    """Triple + clustering ultrametricity diagnostics on Q_hh → .npz. Optional stage."""
    input:
        config["experiment_dir"] + "/done/{run_key}.correlations.done",
    output:
        touch(config["experiment_dir"] + "/done/{run_key}.ultrametricity.done"),
    params:
        p=_ultra_params,
        metrics=" ".join(config.get("ultrametricity", {}).get("metrics",
                         ["frob_cosine", "hist_jensen_shannon"])),
        sample_cap=config.get("ultrametricity", {}).get("sample_cap", 2_000_000),
        quad_sample_cap=config.get("ultrametricity", {}).get("quad_sample_cap", 1_000_000),
        max_dense=config.get("ultrametricity", {}).get("max_dense", 3000),
        tol=config.get("ultrametricity", {}).get("tol", 0.05),
        seed=config.get("ultrametricity", {}).get("seed", 0),
        corr_dir=config["experiment_dir"] + "/correlations",
        ultra_dir=config["experiment_dir"] + "/ultrametricity",
        mlflow_uri=config["mlflow_uri"],
        mlflow_experiment=config["mlflow_experiment"],
    shell:
        """
        python scripts/run_ultrametricity.py \
            --model {params.p[model]} \
            --revision "{params.p[revision]}" \
            --corr-dir {params.corr_dir} \
            --out-dir {params.ultra_dir} \
            --metrics {params.metrics} \
            --sample-cap {params.sample_cap} \
            --quad-sample-cap {params.quad_sample_cap} \
            --max-dense {params.max_dense} \
            --tol {params.tol} \
            --seed {params.seed} \
            --mlflow-uri {params.mlflow_uri} \
            --mlflow-experiment {params.mlflow_experiment}
        """
