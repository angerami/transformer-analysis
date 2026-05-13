"""
Step Evolution Dashboard
Visualizes how statistics evolve across training checkpoints (steps)
"""

import json
from pathlib import Path
import subprocess
import streamlit as st
from datasets import load_from_disk, load_dataset
import os

def is_HF_environment():
    return "SPACE_ID" in os.environ

def get_data_path():
    return os.environ.get("OUTPUT_DIR", "outputs")


def model_size_from_name(ds_name: str) -> float:
    """Extract model size for sorting (in millions of parameters)."""
    import re

    # Extract size like "70m", "1.4b", "12b"
    match = re.search(r"(\d+\.?\d*)([mb])", ds_name.lower())
    if not match:
        return 0

    size, unit = match.groups()
    size = float(size)

    # Convert to millions for consistent comparison
    if unit == "b":
        size *= 1000

    return size


def ensure_offline_available(path: Path):
    """Pin files for offline access via Google Drive."""
    real_path = path.resolve()

    try:
        subprocess.run(
            [
                "find",
                str(real_path),
                "-type",
                "f",
                "-exec",
                "xattr",
                "-w",
                "com.google.drivefs.pinned",
                "true",
                "{}",
                ";",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        st.warning(f"Could not pin files: {e}")
        return False


_SKIP_DIRS = {"eval", "logs", "all_models"}


def get_available_datasets(hf_version: str = None) -> list[str]:
    """Scan OUTPUT_DIR for available per-run datasets (run_keys).

    In HF Spaces mode, hf_version is used to filter hub datasets.
    Locally, scans OUTPUT_DIR directly — no campaign subdirectory.
    """
    if is_HF_environment():
        from huggingface_hub import HfApi
        api = HfApi()
        datasets = api.list_datasets(author="angerami", search=hf_version or "")
        return [ds.id.split('/')[-1] for ds in datasets]

    out = Path(get_data_path())
    if not out.exists():
        return []
    names = set()
    for item in out.iterdir():
        if not item.is_dir() or item.name.startswith("."):
            continue
        base = item.name.removesuffix("_refined")
        if base not in _SKIP_DIRS:
            names.add(base)
    return sorted(names, key=model_size_from_name)


@st.cache_data
def load_dataset_with_metadata(ds_name: str, hf_version: str = None, hf_repo_id: str = None):
    if is_HF_environment():
        repo_id = hf_repo_id if hf_repo_id else f"angerami/{ds_name}_{hf_version}"
        with st.spinner("Loading dataset..."):
            df = load_dataset(repo_id, split="train")
        from huggingface_hub import hf_hub_download
        metadata_path = hf_hub_download(repo_id=repo_id, filename="metadata.json", repo_type="dataset")
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        out = Path(get_data_path())
        refined = out / f"{ds_name}_refined"
        dataset_path = refined if refined.exists() else out / ds_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        with st.spinner("Ensuring files are available offline..."):
            ensure_offline_available(dataset_path)
        with st.spinner("Loading dataset..."):
            df = load_from_disk(str(dataset_path))
        metadata = {}
        metadata_path = dataset_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

    return df.to_pandas(), metadata


def get_unique_values(df, column):
    """Get sorted unique values from a column"""
    return sorted(df[column].unique())


def compute_sv_stat(svd_array, stat_type, d_head=None):
    """Compute a derived statistic from a singular value array.

    Centralizes the SV stat logic used across dashboard pages.
    d_head defaults to len(svd_array) when not provided.
    """
    import numpy as np

    sv = np.array(svd_array)
    if d_head is None:
        d_head = len(sv)

    if stat_type == "mean":
        return float(np.mean(sv))
    elif stat_type == "variance":
        return float(np.var(sv))
    elif stat_type == "skewness":
        mean = np.mean(sv)
        std = np.std(sv)
        return float(np.mean((sv - mean) ** 3) / std ** 3) if std > 0 else 0.0
    elif stat_type == "kurtosis":
        mean = np.mean(sv)
        std = np.std(sv)
        return float(np.mean((sv - mean) ** 4) / std ** 4 - 3) if std > 0 else 0.0
    elif stat_type == "sum":
        return float(np.sum(sv))
    elif stat_type == "sum_squares":
        return float(np.sum(sv ** 2))
    elif stat_type == "participation_ratio":
        sum_sv = np.sum(sv)
        sum_sv2 = np.sum(sv ** 2)
        return float((sum_sv ** 2) / sum_sv2) if sum_sv2 > 0 else 0.0
    elif stat_type == "normalized_participation_ratio":
        sum_sv = np.sum(sv)
        sum_sv2 = np.sum(sv ** 2)
        pr = (sum_sv ** 2) / sum_sv2 if sum_sv2 > 0 else 0.0
        return float(pr / d_head) if d_head > 0 else 0.0
    elif stat_type == "spectral_entropy":
        sv2 = sv ** 2
        sum_sv2 = np.sum(sv2)
        if sum_sv2 > 0:
            p = sv2 / sum_sv2
            p = p[p > 0]
            return float(-np.sum(p * np.log(p)))
        return 0.0
    elif stat_type == "condition_number":
        sv_nonzero = sv[sv > 1e-10]
        if len(sv_nonzero) > 0 and sv_nonzero[-1] > 0:
            return float(sv_nonzero[0] / sv_nonzero[-1])
        return 0.0
    elif stat_type == "stable_rank":
        sum_sv2 = np.sum(sv ** 2)
        max_sv2 = sv[0] ** 2
        return float(sum_sv2 / max_sv2) if max_sv2 > 0 else 0.0
    elif stat_type == "leading_sv":
        return float(sv[0])
    return 0.0


# Display name mappings (same as weights_dashboard_app)
stat_display = {
    "σ (Std Dev)": "std",
    "σ (fit)": "fit_sigma",
    "Entropy (hist)": "entropy",
    "Entropy (KDE)": "differential_entropy",
    "μ (Mean)": "mean",
    "μ (fit)": "fit_mu",
    "sum": "sum",
    "max": "max",
    "min": "min",
    "skew": "skew",
    "kurtosis": "kurtosis",
    "D_KL(P || N(μ,σ))": "kl_vs_empirical_normal",
}


def load_eval_metrics(out_dir: str = None) -> "pd.DataFrame":
    """Load eval_metrics.parquet if it exists; return empty DataFrame otherwise."""
    import pandas as pd
    if out_dir is None:
        out_dir = os.path.join(get_data_path(), "eval_metrics")
    path = os.path.join(out_dir, "eval_metrics.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["model", "revision", "step", "metric", "value", "source", "corpus"])


def kde_from_histogram(bin_centers, counts, n_points=300):
    """Reconstruct a smooth KDE curve from stored histogram bin centers and counts.

    Treats each bin center as a weighted pseudo-sample and fits a Gaussian KDE
    (Scott's bandwidth). Returns a (x_grid, kde_values) pair suitable for line plots.
    """
    from scipy.stats import gaussian_kde
    import numpy as np

    counts = np.asarray(counts, dtype=float)
    bin_centers = np.asarray(bin_centers, dtype=float)
    total = counts.sum()
    if total == 0:
        x_grid = np.linspace(bin_centers[0], bin_centers[-1], n_points)
        return x_grid, np.zeros(n_points)

    weights = counts / total
    kde = gaussian_kde(bin_centers, weights=weights)
    x_grid = np.linspace(bin_centers[0], bin_centers[-1], n_points)
    kde_values = kde(x_grid)
    return x_grid, kde_values
