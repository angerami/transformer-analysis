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
import plotly.graph_objects as go
from typing import Dict, Any, Optional

def is_HF_environment():
    return "SPACE_ID" in os.environ

def get_data_path():
    return os.environ.get("DATA_PATH", "Drive")


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


def get_available_datasets(campaign: str = "step-analysis_001") -> list[str]:

    from huggingface_hub import HfApi
    """Get datasets from local FS or HF Hub based on environment."""
    if is_HF_environment():
        # HF Spaces mode - scan hub
        api = HfApi()
        datasets = api.list_datasets(author="angerami", search=campaign)
        return [ds.id.split('/')[-1].removesuffix(f'_{campaign}') for ds in datasets]

    else:
        drive_path = Path(get_data_path()) / campaign
        if not drive_path.exists():
            return []
        
    datasets = []
    for item in drive_path.iterdir():
        if item.is_dir() and item.name.endswith("_all_checkpoints"):
            # Extract DS_NAME by removing suffix
            ds_name = item.name.replace("_all_checkpoints", "")
            datasets.append(ds_name)
    return sorted(datasets, key=model_size_from_name)



def get_available_campaigns(campaign_pattern: str = "ana-") -> list[str]:
    """Scan Drive for available datasets matching pattern."""
    drive_path = Path(get_data_path())
    if not drive_path.exists():
        return []

    datasets = []
    for item in drive_path.iterdir():
        if item.is_dir() and item.name.startswith(campaign_pattern):
            # Extract DS_NAME by removing suffix
            datasets.append(item.name)
    return sorted(datasets, key=model_size_from_name)


@st.cache_data
def load_dataset_with_metadata(ds_name: str, campaign: str, hf_version: str = None, hf_repo_id: str = None):
    if is_HF_environment():
        repo_id = hf_repo_id if hf_repo_id else f"angerami/{ds_name}_{hf_version}"
        with st.spinner("Loading dataset..."):
            df = load_dataset(repo_id, split="train")
        from huggingface_hub import hf_hub_download
        metadata_path = hf_hub_download(repo_id=repo_id, filename="metadata.json", repo_type="dataset")
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        if campaign.startswith('step-'):
            ds_name += '_all_checkpoints'
        dataset_path = Path(get_data_path()) / campaign / ds_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        # Ensure files are downloaded
        with st.spinner("Ensuring files are available offline..."):
            ensure_offline_available(dataset_path)

        # Load from disk
        with st.spinner("Loading dataset..."):
            df = load_from_disk(str(dataset_path))
        # Load metdata
        metadata_path = dataset_path / df.info.description
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

    return df.to_pandas(), metadata


def get_unique_values(df, column):
    """Get sorted unique values from a column"""
    return sorted(df[column].unique())


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
    "D_KL(P || N(μ,σ)": "kl_vs_empirical_normal",
}


def create_snapshot_button(
    fig: go.Figure,
    metadata: Dict[str, Any],
    section_name: str,
    button_label: str = "📸 Snapshot",
    key: Optional[str] = None,
) -> None:
    """
    Create a button to save a plot snapshot with metadata.

    Args:
        fig: Plotly figure to snapshot
        metadata: Dictionary containing plot metadata (selections, filters, etc.)
        section_name: Name of the section (used in filename prefix)
        button_label: Label for the snapshot button
        key: Unique key for the button widget
    """
    from snapshot_utils import save_snapshot

    # Use a unique key for each button
    button_key = key or f"snapshot_{section_name}"

    if st.button(button_label, key=button_key):
        try:
            paths = save_snapshot(
                fig=fig,
                metadata=metadata,
                name_prefix=section_name,
                include_hash=True,
            )

            st.success(
                f"✓ Snapshot saved!\n\n"
                f"Image: `{paths['image']}`\n\n"
                f"Metadata: `{paths['metadata']}`"
            )

        except ImportError as e:
            st.error(
                f"❌ {str(e)}\n\n"
                "Install kaleido with: `pip install kaleido`"
            )
        except Exception as e:
            st.error(f"❌ Error saving snapshot: {str(e)}")