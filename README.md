# Transformer Weight Analysis

[![Post](https://img.shields.io/badge/📝-Transformer--Spin_Post-lightgrey)](https://angerami.github.io/posts/transformer-spin/)
[![Dashboard](https://img.shields.io/badge/📊-Interactive_Dashboard-orange)](https://huggingface.co/spaces/angerami/transformer-weights)
[![Data](https://img.shields.io/badge/🗂️-Datasets_on_HuggingFace-yellow)](https://huggingface.co/collections/angerami/transformer-weight-evolution-study)

Statistical analysis of transformer weight matrices (W_Q, W_K, W_QK) across architectures and training. Motivated by correspondences between self-attention and spin glass mechanics, the package builds a metrics pipeline that characterizes weight distributions against random-matrix baselines, decomposes spectral structure via SVD, measures cross-head correlations, and tracks how all of these evolve across 154 training checkpoints for the full Pythia suite (70M–12B).

## Installation

**Requirements:**
- Python 3.8+
- No GPU required (analysis works on CPU - no training or inference)

**Setup:**
```bash
git clone https://github.com/angerami/transformer-analysis.git
cd transformer-analysis
pip install -e .
```

**Authentication (for HuggingFace datasets):**
```bash
export HF_TOKEN="your_token_here"
```

## Layout

The package organizes work as analysis pipelines. Each pipeline takes model weights as input (the eval stage also requires input token data) and produces structured data artifacts:

```
model weights ──► primary ──► transform ──► [merge] ──► HF Datasets
input tokens  ──► eval ─────────────────────────────► eval_metrics.parquet
model weights ──► correlations ──────────────────────► .npz correlation matrices
```

Three subsystems support the pipelines:

- **Data processing** (`src/transformer_analysis/`): per-head stat extraction, distributions, SVD, correlation metrics, and perplexity evaluation.
- **Pipeline execution** (`Snakefile`, `rules/`): Snakemake orchestrates download → extract → transform → merge across all configured models and checkpoints; each stage calls a dedicated script in `scripts/`. MLflow logs metrics and parameters for every run to a portable SQLite database at `{output_dir}/mlruns.db`.
- **Dashboards** (`dashboards/`): Streamlit app for interactive exploration of the data artifacts — weight distributions, singular value spectra, cross-model comparison, and training evolution.

### Running the pipeline

| Command | What it does |
|---|---|
| `snakemake -j<N>` | Default target: primary + transform for all configured runs, then cross-model merge (if enabled). |
| `snakemake eval_all -j<N>` | Compute perplexity on WikiText-103 (or Pile) for all models; merge into `eval_metrics.parquet`. |
| `snakemake correlations_all -j<N>` | Compute head-head correlation matrices (Frobenius, Pearson, Jensen-Shannon) for all runs. |
| `snakemake pair_figures_all` | Generate correlation heatmaps from pre-computed matrices (re-runnable, fast). |
| `snakemake target_gpt2_family -j<N>` | Process only GPT-2 family (four scales). Equivalents: `target_pythia_family`, `target_llama_family`. |
| `snakemake target_pythia_70m_steps -j<N>` | Checkpoint sweep: all 154 training steps for Pythia-70M. Equivalents for other sizes once `pythia_steps` is configured. |
| `snakemake clean` | Full reset: remove all outputs, sentinels, and MLflow DB. |
| `snakemake clean_transform` | Keep primary datasets; wipe transform and all downstream. Use to re-run metric extraction without re-downloading weights. |
| `snakemake clean_run --config run_key=<key>` | Remove outputs for a single run (e.g., `run_key=gpt2` or `run_key=pythia-70m-deduped_step143000`). |

### Experiment tracking

Pipeline runs are logged to a local MLflow database. To browse:

```bash
mlflow ui --backend-store-uri sqlite:///outputs/mlruns.db
```

Open `http://localhost:5000`. Each stage (primary, transform, eval, correlations) creates its own run entry under the experiment configured in `config.yaml`.

### Configuration

Pipelines are parameterized via `config.yaml`. The most common things to change:

| Key | What to edit |
|---|---|
| `output_dir` | Root directory for all outputs. Override at runtime: `OUTPUT_DIR=/path snakemake -j8`. |
| `cache_dir` | Where downloaded model weights are cached. Set to a path with sufficient disk space. |
| `runs` | List of `{model, revision}` pairs to process. Remove entries to restrict scope, or add new models. |
| `targets` | Named subsets of `runs` that drive family-level targets (e.g., `target_gpt2_family`). |
| `pythia_steps` | Uncomment and add `{model, shortname}` entries to enable checkpoint sweeps. |
| `max_workers` | Thread pool size for weight extraction in the primary stage. |
| `cleanup_downloads` | Set `true` to delete model weights after processing and save disk space. |
| `merge.cross_model.enabled` | Toggle incremental cross-model dataset merging after transform. |
| `merge.cross_model.refresh` | List model names to force-replace in an existing merged dataset. |
| `merge.checkpoints` | Models whose checkpoint revisions should be collapsed into a single dataset. |
| `mlflow_experiment` | Experiment name for grouping runs in MLflow. Change when starting a new analysis phase. |

## Quick Start

**Analyze a single model's weights:**
```python
from transformer_analysis.weight_analysis import process_model

# Single model
process_model(
    model_name="gpt2",
    out_dir="results/",
    cache_dir="downloads/",
    cleanup_downloads=False,
)
```

**View your results:**
```bash
streamlit run dashboards/streamlit_app.py
# Note: Edit the script to point at your output directory
```

**Explore published datasets:**
- **Dashboards:** [HuggingFace Spaces](https://huggingface.co/spaces/angerami/transformer-weights)
- **Cross-model comparison:** `angerami/weight_study_ana-003`
- **Checkpoint evolution:** `angerami/pythia-{size}-deduped_weight_evolution_001`
  - Available sizes: 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b

## Usage

**Run analysis:**
```python
from transformer_analysis.weight_analysis import process_model

process_model(
    model_name="pythia-70m-deduped",
    revision="step3000",  # optional, for checkpoints
    out_dir="results/",
    cache_dir="downloads/",
)
```

**Load results:**
```python
from datasets import load_from_disk

ds = load_from_disk("results/pythia-70m-deduped_step3000/")
df = ds.to_pandas()

# Each row = one attention head
print(df.info())
# Columns: weight_type, sum, mean, std, max, min, skew, kurtosis,
#          differential_entropy, entropy, fit_mu, fit_sigma,
#          kl_vs_empirical_normal, head, layer, model, job_id
#          P_w (histogram bins), SVD (singular values), P_sv (SV distribution)
#
# Checkpoint datasets add: revision, step
```

**Visualize:**
```bash
streamlit run dashboards/streamlit_app.py
# Edit DATASET_PATH to point at your results/
```
## Data

**Published datasets (HuggingFace Hub):**

**Cross-model comparison:**
- `angerami/weight_study_ana-003` - Static weight analysis across GPT-2, Pythia, LLaMA, Mistral

**Checkpoint evolution (Pythia suite):**
- `angerami/pythia-70m-deduped_weight_evolution_001`
- `angerami/pythia-160m-deduped_weight_evolution_001`
- `angerami/pythia-410m-deduped_weight_evolution_001`
- `angerami/pythia-1b-deduped_weight_evolution_001`
- `angerami/pythia-1.4b-deduped_weight_evolution_001`
- `angerami/pythia-2.8b-deduped_weight_evolution_001`
- `angerami/pythia-6.9b-deduped_weight_evolution_001`
- `angerami/pythia-12b-deduped_weight_evolution_001`

Each contains 154 checkpoints spanning full training (steps 0-143000).

**Access:**
```python
from datasets import load_dataset

ds = load_dataset("angerami/weight_study_ana-003")
df = ds['train'].to_pandas()
```

## Package

### `scripts/` — pipeline entry points

Each Snakemake stage calls one script. The scripts handle CLI argument parsing and MLflow logging; the heavy lifting is delegated to `src/transformer_analysis/`.

| Script | Stage | What it does |
|---|---|---|
| `run_primary.py` | primary | Download weights → extract per-head stats → save HF Dataset |
| `run_transform.py` | transform | Re-compute derived metrics on existing primary dataset → refined HF Dataset |
| `run_eval.py` | eval | Compute perplexity on WikiText-103 or Pile → per-run parquet |
| `run_correlations.py` | correlations | Extract W_QK per head → head-head correlation matrices (.npz) |
| `run_merge.py` | merge | Merge runs into cross-model dataset, or collapse checkpoint revisions |
| `plot_pair_correlations.py` | pair_figures | Load .npz intermediates → correlation heatmaps |
| `prepare_eval_corpus.py` | (one-time) | Download and cache a fixed Pile test-set sample for perplexity eval |

### `src/transformer_analysis/` — analysis library

Source files organized by the pipeline stage they primarily support:

**Primary & transform:**
- `head_pipeline.py` — top-level orchestration: `process_model()`, `reprocess_metrics()`, `merge_versions()`
- `head_analyzer.py` — per-head metric computation: weight distributions, SVD, normality measures
- `head_metrics.py` — metric definitions, bin configurations, histogram and singular-value strategies

**Correlations:**
- `pair_pipeline.py` — orchestrates multi-circuit correlation analysis
- `pair_analyzer.py` — `HeadStore` and pairwise correlation matrix computation
- `pair_metrics.py` — metric implementations (Frobenius cosine, Pearson, Jensen-Shannon, etc.)

**Eval:**
- `eval_metrics.py` — perplexity computation, corpus loaders (WikiText-103, Pile), parquet export

**Shared:**
- `model_registry.py` — architecture configs and weight extractors for all supported model families
- `device_utils.py` — compute device detection (CPU / CUDA / MPS)
- `perf_logger.py` — phase timing and memory monitoring, with optional MLflow export
## Models Supported

**Currently implemented:**
- **Pythia suite** (70M - 12B): Full checkpoint support (154 steps)
- **GPT-2** (124M, 355M, 774M, 1.5B)
- **LLaMA 3/3.1** (8B, 70B) - requires authentication
- **Mistral** (7B v0.1, v0.3)

**Adding new models:**

Edit `transformer_analysis/model_registry.py`:
```python
ModelConfig(
    name="your-model",
    hf_name="org/model-name",
    num_layers=32,
    num_heads=32,
    d_model=4096,
    weight_names=["q_proj", "k_proj", "v_proj"],  # model-specific
    requires_auth=False,
)
```

Architecture must use standard attention with separable Q, K, V projection matrices.

## Analysis Types

**Weight distributions:**
- Histograms (configurable bins)
- Statistical moments (mean, std, skewness, kurtosis)
- Deviation from normality (KL divergence, fitted μ/σ)
- Differential entropy

**Singular value decomposition:**
- Singular value spectra
- Effective rank analysis
- Distribution of singular values

**Granularity:**
- Per attention head (W_Q, W_K, W_V)
- Combined matrices (W_QK = W_Q @ W_K^T)
- Layer-wise comparisons
- Cross-model comparisons

**Checkpoint evolution (Pythia):**
- Track all metrics across 154 training checkpoints
- Correlate with training steps (optionally with loss via W&B integration)
## Citation

If you use this toolkit in your research, please cite:
```bibtex
@software{transformer_weight_analysis,
  author = {Angerami, Aaron},
  title = {Transformer Weight Analysis: Understanding Weight Distributions and Training Dynamics},
  year = {2025},
  url = {https://github.com/angerami/transformer-analysis}
}
```
**Related work:**
- Interactive dashboards: https://huggingface.co/spaces/angerami/transformer-weights

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

**Model datasets:**
- **Pythia:** Biderman et al. (2023) "Pythia: A Suite for Analyzing Large Language Models" [[paper]](https://arxiv.org/abs/2304.01373) [[models]](https://huggingface.co/EleutherAI)
- **GPT-2:** Radford et al. (2019) "Language Models are Unsupervised Multitask Learners" [[paper]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) [[models]](https://huggingface.co/openai-community)
- **LLaMA 3/3.1:** Dubey et al. (2024) "The Llama 3 Herd of Models" [[paper]](https://arxiv.org/abs/2407.21783) [[models]](https://huggingface.co/meta-llama)
- **Mistral:** Jiang et al. (2023) "Mistral 7B" [[paper]](https://arxiv.org/abs/2310.06825) [[models]](https://huggingface.co/mistralai)

**Inspired by:** Sebastian Raschka's "Build a Large Language Model (From Scratch)"