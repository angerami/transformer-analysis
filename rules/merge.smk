# Merge rules.
#
# merge_cross_model  — incremental cross-model merge.  Runs whenever a new
#   run's transform sentinel appears (i.e. a new model was added to config).
#   The script appends only new entries; existing ones are untouched.
#   To force-replace a model after re-transforming it, add it to
#   config["merge"]["cross_model"]["refresh"] and run:
#       snakemake done/cross_model.merge.done --forcerun merge_cross_model
#
# merge_checkpoints  — per-model checkpoint collapse.  One rule instance per
#   entry in config["merge"]["checkpoints"].  Runs when any of that model's
#   transform sentinels are newer than the merge sentinel.


# ── cross-model ────────────────────────────────────────────────────────────────

def _run_key(run):
    model = run["model"]
    rev = run.get("revision") or None
    return f"{model}_{rev}" if rev else model


def _all_transform_sentinels():
    return [f"done/{_run_key(r)}.transform.done" for r in config["runs"]]


def _all_dataset_dirs():
    out = config["output_dir"]
    return " ".join(f"{out}/{_run_key(r)}" for r in config["runs"])


def _cross_model_params(_):
    refresh_list = config.get("merge", {}).get("cross_model", {}).get("refresh", [])
    out_name = config["merge"]["cross_model"]["out_name"]
    return {
        "dataset_dirs": _all_dataset_dirs(),
        "out_path": f"{config['output_dir']}/{out_name}",
        "refresh_flag": f"--refresh {' '.join(refresh_list)}" if refresh_list else "",
    }


if config.get("merge", {}).get("cross_model", {}).get("enabled", False):
    rule merge_cross_model:
        """Incrementally merge all runs into one cross-model HF Dataset; appends new, skips existing."""
        input:
            _all_transform_sentinels(),
        output:
            touch("done/cross_model.merge.done"),
        params:
            p=_cross_model_params,
        shell:
            """
            python scripts/run_merge.py cross-model \
                --dataset-dirs {params.p[dataset_dirs]} \
                --out-path {params.p[out_path]} \
                {params.p[refresh_flag]}
            """


# ── per-model checkpoint collapse ─────────────────────────────────────────────

def _checkpoint_sentinels(model_name):
    """All transform sentinels for every revision of model_name."""
    from transformer_analysis.model_registry import get_model_versions
    revisions = get_model_versions(model_name)
    return [f"done/{model_name}_{rev}.transform.done" for rev in revisions]


# template rule — one instance per entry in config["merge"]["checkpoints"]
for _cp_model in config.get("merge", {}).get("checkpoints", []):
    rule:
        name: f"merge_checkpoints_{_cp_model.replace('-', '_').replace('.', '_')}"
        input:
            _checkpoint_sentinels(_cp_model),
        output:
            touch(f"done/{_cp_model}_checkpoints.merge.done"),
        params:
            model=_cp_model,
            out_dir=config["output_dir"],
        shell:
            """
            python scripts/run_merge.py checkpoints \
                --model {params.model} \
                --out-dir {params.out_dir}
            """
