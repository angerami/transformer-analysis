"""
Utilities for capturing and managing plot snapshots with metadata.

This module provides functions to save Plotly figures along with their associated
metadata (selections, filters, parameters) to enable reproducibility and
downstream analysis.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go
import numpy as np


def _convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy/pandas types to JSON-serializable Python types.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def compute_metadata_hash(metadata: Dict[str, Any], length: int = 8) -> str:
    """
    Compute a short hash of the metadata for use in filenames.

    Args:
        metadata: Dictionary containing plot metadata
        length: Length of hash to return (default 8 characters)

    Returns:
        Short hash string
    """
    # Convert to JSON-serializable format first
    serializable_metadata = _convert_to_json_serializable(metadata)
    # Create a stable string representation
    metadata_str = json.dumps(serializable_metadata, sort_keys=True)
    full_hash = hashlib.sha256(metadata_str.encode()).hexdigest()
    return full_hash[:length]


def save_snapshot(
    fig: go.Figure,
    metadata: Dict[str, Any],
    snapshot_dir: Path = Path("snapshots"),
    name_prefix: Optional[str] = None,
    include_hash: bool = True,
    image_format: str = "png",
    image_width: int = 1200,
    image_height: int = 800,
) -> Dict[str, Path]:
    """
    Save a Plotly figure and its metadata to disk.

    Args:
        fig: Plotly figure to save
        metadata: Dictionary containing plot metadata (selections, filters, etc.)
        snapshot_dir: Base directory for snapshots
        name_prefix: Optional prefix for the snapshot name (e.g., "section_1")
        include_hash: Whether to include metadata hash in filename
        image_format: Image format (png, svg, pdf, jpg)
        image_width: Width of saved image in pixels
        image_height: Height of saved image in pixels

    Returns:
        Dictionary with paths to saved files: {"image": Path, "metadata": Path}
    """
    # Create snapshot directory if it doesn't exist
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp-based name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build filename
    parts = []
    if name_prefix:
        parts.append(name_prefix)
    parts.append(timestamp)
    if include_hash:
        parts.append(compute_metadata_hash(metadata))

    base_name = "_".join(parts)

    # Define file paths
    image_path = snapshot_dir / f"{base_name}.{image_format}"
    metadata_path = snapshot_dir / f"{base_name}.json"

    # Add snapshot metadata
    full_metadata = {
        "timestamp": timestamp,
        "image_file": image_path.name,
        "image_format": image_format,
        "image_width": image_width,
        "image_height": image_height,
        **metadata,
    }

    # Convert to JSON-serializable format
    full_metadata = _convert_to_json_serializable(full_metadata)

    # Save the figure
    try:
        fig.write_image(
            str(image_path),
            format=image_format,
            width=image_width,
            height=image_height,
        )
    except Exception as e:
        # If kaleido is not installed, provide helpful error
        if "kaleido" in str(e).lower():
            raise ImportError(
                "Kaleido is required for image export. Install with: pip install kaleido"
            ) from e
        raise

    # Save metadata
    with open(metadata_path, "w") as f:
        json.dump(full_metadata, f, indent=2)

    return {
        "image": image_path,
        "metadata": metadata_path,
    }


def load_snapshot_metadata(metadata_path: Path) -> Dict[str, Any]:
    """
    Load snapshot metadata from a JSON file.

    Args:
        metadata_path: Path to metadata JSON file

    Returns:
        Dictionary containing snapshot metadata
    """
    with open(metadata_path, "r") as f:
        return json.load(f)


def list_snapshots(
    snapshot_dir: Path = Path("snapshots"),
    name_prefix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List all available snapshots with their metadata.

    Args:
        snapshot_dir: Directory containing snapshots
        name_prefix: Optional prefix to filter snapshots

    Returns:
        List of dictionaries containing snapshot information
    """
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists():
        return []

    snapshots = []
    pattern = f"{name_prefix}_*.json" if name_prefix else "*.json"

    for metadata_file in sorted(snapshot_dir.glob(pattern), reverse=True):
        try:
            metadata = load_snapshot_metadata(metadata_file)
            image_file = snapshot_dir / metadata["image_file"]

            snapshots.append({
                "metadata_path": metadata_file,
                "image_path": image_file,
                "timestamp": metadata.get("timestamp"),
                "metadata": metadata,
            })
        except Exception:
            # Skip invalid metadata files
            continue

    return snapshots


def filter_snapshots(
    snapshots: List[Dict[str, Any]],
    filters: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Filter snapshots based on metadata criteria.

    Args:
        snapshots: List of snapshot dictionaries from list_snapshots()
        filters: Dictionary of metadata keys and values to filter by

    Returns:
        Filtered list of snapshots
    """
    filtered = []
    for snapshot in snapshots:
        metadata = snapshot["metadata"]
        match = all(
            metadata.get(key) == value
            for key, value in filters.items()
        )
        if match:
            filtered.append(snapshot)

    return filtered


def create_snapshot_summary(snapshot_dir: Path = Path("snapshots")) -> str:
    """
    Create a markdown summary of all snapshots.

    Args:
        snapshot_dir: Directory containing snapshots

    Returns:
        Markdown-formatted string summarizing snapshots
    """
    snapshots = list_snapshots(snapshot_dir)

    if not snapshots:
        return "No snapshots found."

    lines = [
        "# Snapshot Summary",
        f"\nTotal snapshots: {len(snapshots)}\n",
        "## Recent Snapshots\n",
    ]

    for snapshot in snapshots[:10]:  # Show most recent 10
        meta = snapshot["metadata"]
        lines.append(f"### {snapshot['image_path'].name}")
        lines.append(f"- **Timestamp**: {meta.get('timestamp')}")
        lines.append(f"- **Image**: `{snapshot['image_path']}`")
        lines.append(f"- **Metadata**: `{snapshot['metadata_path']}`")

        # Add key metadata fields
        for key in ["campaign", "model", "weight_type", "layer", "head"]:
            if key in meta:
                lines.append(f"- **{key.title()}**: {meta[key]}")

        lines.append("")

    return "\n".join(lines)
