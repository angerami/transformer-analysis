# Eval rule: compute perplexity for each (model, revision) and append to a
# shared parquet side table (eval_metrics.parquet).
#
# This is an optional stage; not part of the default `all` target.
# Run all evals with:
#   snakemake eval -j1
# or a single run with:
#   snakemake done/<run_key>.eval.done -j1
#
# NOTE: use -j1 here. compute_perplexity.py appends to a shared parquet via
# read-modify-write, so concurrent instances would clobber each other.


def _eval_params(wildcards):
    for run in config["runs"]:
        model = run["model"]
        rev = run.get("revision") or None
        key = f"{model}_{rev}" if rev else model
        if key == wildcards.run_key:
            device = config.get("device") or ""
            eval_cfg = config.get("eval", {})
            corpus = eval_cfg.get("corpus", "wikitext103")
            pile_cache = eval_cfg.get("pile_cache") or ""
            pile_tokens = eval_cfg.get("pile_tokens", 204800)
            return {
                "model": model,
                "revision": rev or "",
                "corpus": corpus,
                "pile_tokens": pile_tokens,
                "device_flag": f"--device {device}" if device else "",
                "pile_cache_flag": f"--pile-cache {pile_cache}" if pile_cache else "",
            }
    raise ValueError(f"No run config found for run_key={wildcards.run_key!r}")


rule eval:
    input:
        "done/{run_key}.transform.done",
    output:
        touch("done/{run_key}.eval.done"),
    params:
        p=_eval_params,
        out=lambda _: config.get("eval", {}).get("out", "outputs/eval_metrics/eval_metrics.parquet"),
        cache_dir=config["cache_dir"],
    shell:
        """
        python scripts/compute_perplexity.py \
            --model {params.p[model]} \
            --revision "{params.p[revision]}" \
            --corpus {params.p[corpus]} \
            --pile-tokens {params.p[pile_tokens]} \
            --out {params.out} \
            --cache {params.cache_dir} \
            {params.p[device_flag]} \
            {params.p[pile_cache_flag]}
        """
