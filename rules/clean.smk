# Clean rules — explicit-only, never part of the default DAG.
#
# Full reset (files + MLflow DB):
#   snakemake clean
#
# Keep primary datasets, wipe transform and everything downstream:
#   snakemake clean_transform
#
# Per-family (sentinels + outputs for that subset):
#   snakemake clean_gpt2_family
#   snakemake clean_pythia_family
#   snakemake clean_llama_family
#
# Single run:
#   snakemake clean_run --config run_key=gpt2
#
# MLflow only (keep all output files, just reset experiment history):
#   snakemake clean_mlflow


rule clean:
    """Full reset: experiment outputs + sentinels + Snakemake cache. MLflow DB preserved (use clean_mlflow separately)."""
    shell:
        """
        rm -rf {config[experiment_dir]}/ .snakemake/
        echo "Full clean complete."
        """


rule clean_transform:
    """
    Wipe transform and all downstream stages; keep primary datasets.
    Re-run with: snakemake -j8
    """
    shell:
        """
        rm -f {config[experiment_dir]}/done/*.transform.done \\
              {config[experiment_dir]}/done/*.correlations.done \\
              {config[experiment_dir]}/done/*.pair_figures.done \\
              {config[experiment_dir]}/done/*.eval.done \\
              {config[experiment_dir]}/done/cross_model.merge.done \\
              {config[experiment_dir]}/done/*_checkpoints.merge.done
        rm -rf {config[experiment_dir]}/*_refined {config[experiment_dir]}/eval
        echo "Transform clean complete."
        """


rule clean_mlflow:
    """Wipe the MLflow DB only (all experiment history, files untouched)."""
    shell:
        "rm -f {config[output_dir]}/mlruns.db && echo 'MLflow DB wiped.'"


rule clean_run:
    """
    Remove one run's sentinels and output directories.
    Usage: snakemake clean_run --config run_key=gpt2
           snakemake clean_run --config run_key=pythia-70m-deduped_step143000
    """
    run:
        import glob, shutil
        run_key = config.get("run_key")
        if not run_key:
            raise ValueError("Specify run_key: snakemake clean_run --config run_key=<key>")
        ed = config["experiment_dir"]
        for f in glob.glob(f"{ed}/done/{run_key}.*.done"):
            os.remove(f)
            print(f"  rm {f}")
        for path in [f"{ed}/{run_key}", f"{ed}/{run_key}_refined", f"{ed}/eval/{run_key}.parquet"]:
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"  rm -rf {path}")
            elif os.path.isfile(path):
                os.remove(path)
                print(f"  rm {path}")
        print(f"Cleaned {run_key}.")


# Per-family clean rules — mirrors the target_<name> rules.
# Generated from the same config["targets"] so they stay in sync automatically.

def _clean_family(target_runs):
    import glob, shutil
    ed = config["experiment_dir"]
    run_keys = [_run_key(r) for r in target_runs]
    for rk in run_keys:
        for f in glob.glob(f"{ed}/done/{rk}.*.done"):
            os.remove(f)
            print(f"  rm {f}")
        for path in [f"{ed}/{rk}", f"{ed}/{rk}_refined", f"{ed}/eval/{rk}.parquet"]:
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"  rm -rf {path}")
            elif os.path.isfile(path):
                os.remove(path)
                print(f"  rm {path}")
    print(f"Cleaned {len(run_keys)} run(s).")


for _target_name, _target_runs in config.get("targets", {}).items():
    _runs_snapshot = list(_target_runs)
    rule:
        name: f"clean_{_target_name}"
        run:
            _clean_family(_runs_snapshot)
