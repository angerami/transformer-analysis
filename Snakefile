configfile: "config.yaml"

import os
import mlflow

config["output_dir"] = os.environ.get("OUTPUT_DIR", config["output_dir"])
# keep mlruns next to the outputs so they travel together
config["mlflow_uri"] = f"sqlite:///{config['output_dir']}/mlruns.db"

# Pre-initialize the MLflow DB and experiment here (single-threaded, before
# any parallel jobs start) so workers don't race to CREATE TABLE.
os.makedirs(config["output_dir"], exist_ok=True)
mlflow.set_tracking_uri(config["mlflow_uri"])
mlflow.set_experiment(config["mlflow_experiment"])

wildcard_constraints:
    output_dir = r"[^/]+",
    run_key = r"[^/]+",

include: "rules/primary.smk"
include: "rules/transform.smk"
include: "rules/correlations.smk"
include: "rules/merge.smk"


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
