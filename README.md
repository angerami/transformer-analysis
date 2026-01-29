# Transformer Weight Analysis

Analysis software developed to inspect weights in transformers (the $W_{Q}$ and $W_{K}$ matrices and their product, $W_{QK}$) and analyze the matrix elements as a statistical ensemble. Measure ensemble properties, how they differ from normal distributions, vary across attention heads and layers, and evolve during training. Compare statistical properties across model architectures and sizes.

Supports systematic checkpoint analysis (Pythia 70M-12B with 154 checkpoints) and cross-model comparison (Pythia, GPT-2, extensible to LLaMA/Mistral).

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

## Project Structure
```
transformer-analysis/
├── transformer_analysis/
│   ├── weight_analysis.py        # Main analysis entry point
│   ├── model_registry.py         # Supported model configurations
│   ├── model_loader.py           # Model-agnostic weight extraction
│   ├── weight_stats.py           # Statistical computations
│   └── utils/                    # Logging, performance monitoring
├── dashboards/
│   └── streamlit_app.py          # Interactive visualization
├── results/                      # Local output directory
└── requirements.txt
```

**Key modules:**
- `model_registry.py`: ModelConfig definitions for Pythia, GPT-2, LLaMA, Mistral
- `model_loader.py`: Extracts W_Q, W_K, W_V matrices; handles authentication
- `weight_stats.py`: Computes distributions, entropy, KL divergence, SVD
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