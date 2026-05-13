configfile: "config.yaml"

wildcard_constraints:
    output_dir = r"[^/]+",
    run_key = r"[^/]+",

include: "rules/primary.smk"
include: "rules/transform.smk"
include: "rules/correlations.smk"
include: "rules/pair_figures.smk"
include: "rules/merge.smk"
include: "rules/eval.smk"


def _run_key(run):
    """Convert a {model, revision} dict to the output directory name."""
    model = run["model"]
    rev = run.get("revision") or None
    return f"{model}_{rev}" if rev else model


def _all_targets():
    targets = [f"done/{_run_key(r)}.transform.done" for r in config["runs"]]
    if config.get("merge", {}).get("cross_model", {}).get("enabled", False):
        targets.append("done/cross_model.merge.done")
    for cp_model in config.get("merge", {}).get("checkpoints", []):
        targets.append(f"done/{cp_model}_checkpoints.merge.done")
    return targets


rule all:
    input:
        _all_targets(),


# Convenience targets for optional stages (not in default `all`).
rule correlations_all:
    input:
        expand("done/{run_key}.correlations.done", run_key=[_run_key(r) for r in config["runs"]]),

rule pair_figures_all:
    input:
        expand("done/{run_key}.pair_figures.done", run_key=[_run_key(r) for r in config["runs"]]),

# Eval appends to a shared parquet — use -j1 to avoid concurrent write conflicts.
rule eval_all:
    input:
        expand("done/{run_key}.eval.done", run_key=[_run_key(r) for r in config["runs"]]),
