# Pair figures rule: generate correlation figures from saved .npz intermediates.
# This is the pair-analysis equivalent of transform: re-run independently to
# regenerate plots without re-running the expensive correlation computation.
#
# Run all figures with:
#   snakemake pair_figures_all
# or a single run with:
#   snakemake done/<run_key>.pair_figures.done
#
# Depends on the correlations sentinel, not on primary/transform — figures
# can be regenerated from the .npz alone.


def _pair_figures_params(wildcards):
    for run in config["runs"]:
        model = run["model"]
        rev = run.get("revision") or None
        key = f"{model}_{rev}" if rev else model
        if key == wildcards.run_key:
            return {"model": model, "revision": rev or "main"}
    raise ValueError(f"No run config found for run_key={wildcards.run_key!r}")


rule pair_figures:
    input:
        "done/{run_key}.correlations.done",
    output:
        touch("done/{run_key}.pair_figures.done"),
    params:
        p=_pair_figures_params,
        corr_dir="correlations",
        fig_dir="figures/pairs",
    shell:
        """
        python scripts/plot_pair_correlations.py \
            --data {params.corr_dir} \
            --model {params.p[model]} \
            --revision {params.p[revision]} \
            --out {params.fig_dir}/{wildcards.run_key}
        """
