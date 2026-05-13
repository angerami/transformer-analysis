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


# ── Named family targets ───────────────────────────────────────────────────────
# Each entry in config["targets"] becomes a snakemake target_<name> rule that
# drives only the runs in that subset:
#   snakemake target_gpt2_family -j4
#   snakemake target_pythia_family -j4
#   snakemake target_llama_family -j4
#   snakemake target_all -j8

for _target_name, _target_runs in config.get("targets", {}).items():
    rule:
        name: f"target_{_target_name}"
        input:
            expand("done/{run_key}.transform.done",
                   run_key=[_run_key(r) for r in _target_runs]),


# ── Pythia checkpoint sweep targets ───────────────────────────────────────────
# Each entry in config["pythia_steps"] generates target_pythia_{shortname}_steps
# covering all PYTHIA_REVISIONS for that model.
# Enable in config.yaml under pythia_steps:, then add the per-step runs to runs:.
#   snakemake target_pythia_70m_steps -j4

from transformer_analysis.model_registry import PYTHIA_REVISIONS

for _ps in config.get("pythia_steps", []):
    _ps_model = _ps["model"]
    _ps_name  = _ps["shortname"]
    rule:
        name: f"target_pythia_{_ps_name}_steps"
        input:
            expand("done/{run_key}.transform.done",
                   run_key=[f"{_ps_model}_{rev}" for rev in PYTHIA_REVISIONS]),
