configfile: "config.yaml"

import os
import mlflow

config["output_dir"] = os.environ.get("OUTPUT_DIR", config["output_dir"])
# mlruns.db lives at output_dir level — shared across experiments
config["mlflow_uri"] = f"sqlite:///{config['output_dir']}/mlruns.db"
# all stage outputs and sentinels live under experiment_dir
config["experiment_dir"] = os.path.join(config["output_dir"], config["mlflow_experiment"])
# eval.out is derived from experiment_dir so it doesn't need to be in config.yaml
config.setdefault("eval", {})["out"] = f"{config['experiment_dir']}/eval_metrics/eval_metrics.parquet"

# Pre-initialize the MLflow DB and experiment here (single-threaded, before
# any parallel jobs start) so workers don't race to CREATE TABLE.
os.makedirs(config["experiment_dir"], exist_ok=True)
mlflow.set_tracking_uri(config["mlflow_uri"])
mlflow.set_experiment(config["mlflow_experiment"])

wildcard_constraints:
    output_dir = r"[^/]+",
    run_key = r"[^/]+",

include: "rules/clean.smk"
include: "rules/primary.smk"
include: "rules/transform.smk"
include: "rules/correlations.smk"
include: "rules/ultrametricity.smk"
include: "rules/pair_figures.smk"
include: "rules/merge.smk"
include: "rules/eval.smk"


def _run_key(run):
    """Convert a {model, revision} dict to the output directory name."""
    model = run["model"]
    rev = run.get("revision") or None
    return f"{model}_{rev}" if rev else model


def _all_targets():
    ed = config["experiment_dir"]
    targets = [f"{ed}/done/{_run_key(r)}.transform.done" for r in config["runs"]]
    if config.get("merge", {}).get("cross_model", {}).get("enabled", False):
        targets.append(f"{ed}/done/cross_model.merge.done")
    for cp_model in config.get("merge", {}).get("checkpoints", []):
        targets.append(f"{ed}/done/{cp_model}_checkpoints.merge.done")
    return targets


rule all:
    """Default target: primary + transform (+ cross-model merge) for all configured runs."""
    input:
        _all_targets(),


# Convenience targets for optional stages (not in default `all`).
rule correlations_all:
    """Optional target: correlations for all configured runs."""
    input:
        expand(config["experiment_dir"] + "/done/{run_key}.correlations.done",
               run_key=[_run_key(r) for r in config["runs"]]),

rule ultrametricity_all:
    """Optional target: ultrametricity diagnostics for all configured runs."""
    input:
        expand(config["experiment_dir"] + "/done/{run_key}.ultrametricity.done",
               run_key=[_run_key(r) for r in config["runs"]]),

rule pair_figures_all:
    """Optional target: pair figures for all configured runs."""
    input:
        expand(config["experiment_dir"] + "/done/{run_key}.pair_figures.done",
               run_key=[_run_key(r) for r in config["runs"]]),

rule eval_all:
    """Optional target: eval + merge for all configured runs."""
    input:
        config.get("eval", {}).get("out", "outputs/eval_metrics/eval_metrics.parquet"),


# ── Named family targets ───────────────────────────────────────────────────────
# Each entry in config["targets"] becomes a snakemake target_<name> rule that
# drives only the runs in that subset:
#   snakemake target_gpt2_family -j4
#   snakemake target_pythia_family -j4
#   snakemake target_llama_family -j4
#   snakemake target_all -j8

# template rule — one instance per entry in config["targets"]
for _target_name, _target_runs in config.get("targets", {}).items():
    rule:
        name: f"target_{_target_name}"
        input:
            expand(config["experiment_dir"] + "/done/{run_key}.transform.done",
                   run_key=[_run_key(r) for r in _target_runs]),

rule target_all:
    """Run primary + transform for all model families (composes all family targets)."""
    input:
        rules.target_gpt2_family.input,
        rules.target_pythia_family.input,
        rules.target_llama_family.input,
        rules.target_olmo2_family.input,
        rules.target_llama32_family.input,
        rules.target_gemma2_family.input,
        rules.target_smollm2_family.input,


# ── Pythia checkpoint sweep targets ───────────────────────────────────────────
# Each entry in config["pythia_steps"] generates target_pythia_{shortname}_steps
# covering all PYTHIA_REVISIONS for that model.
# Enable in config.yaml under pythia_steps:, then add the per-step runs to runs:.
#   snakemake target_pythia_70m_steps -j4

from transformer_analysis.model_registry import PYTHIA_REVISIONS

# template rule — one instance per entry in config["pythia_steps"]
for _ps in config.get("pythia_steps", []):
    _ps_model = _ps["model"]
    _ps_name  = _ps["shortname"]
    rule:
        name: f"target_pythia_{_ps_name}_steps"
        input:
            expand(config["experiment_dir"] + "/done/{run_key}.transform.done",
                   run_key=[f"{_ps_model}_{rev}" for rev in PYTHIA_REVISIONS]),
