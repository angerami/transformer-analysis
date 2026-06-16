#!/usr/bin/env python3
"""Generate JSON index for experiment viewer.

Scans the figures directory, parses each filename into structured metadata
(model, component, plot_type, metric) following the convention in
docs/FIGURE_NAMING.md, and writes a JSON index that viewer.html consumes.

Usage:
    python generate_viewer_index.py --out corr_out
    python generate_viewer_index.py --out corr_out --serve
"""

import argparse
import json
import os
import re
from pathlib import Path
import http.server
import socketserver
import webbrowser
from threading import Timer


# ── Filename parsing ──────────────────────────────────────────────────

COMPONENTS = ["W_QK", "W_OV", "b_Q", "b_K", "b_V", "b_O"]
_COMP_RE = "|".join(re.escape(c) for c in COMPONENTS)

PLOT_TYPES = [
    "corr_vs_layer_distance", "dominant_eigenvectors", "MP_overlay",
    "Q_heatmap", "Q_eigenvalues", "block_means", "P_Q",
    "scalars_vs_layer", "scalars_heatmap", "scalar_correlations",
]

METRICS = [
    "hist_symmetric_kl", "hist_jensen_shannon",
    "symmetric_kl", "jensen_shannon",
    "frob_cosine", "pearson_corr", "two_point", "connected_corr",
]

# Display labels for the viewer
PLOT_TYPE_DISPLAY = {
    "Q_heatmap":                "Correlation heatmap",
    "block_means":              "Block means",
    "P_Q":                      "Overlap distribution P(Q)",
    "Q_eigenvalues":            "Eigenvalue spectrum",
    "corr_vs_layer_distance":   "Correlation vs distance",
    "MP_overlay":               "Marchenko-Pastur",
    "dominant_eigenvectors":    "Dominant eigenvectors",
    "cross_heatmap":            "Cross-correlation heatmap",
    "cross_diagonal":           "Cross-correlation diagonal",
    "scalars_vs_layer":         "Scalars vs layer",
    "scalars_heatmap":          "Scalar heatmaps",
    "scalar_correlations":      "Scalar correlations",
}

METRIC_DISPLAY = {
    "frob_cosine":          "Frobenius cosine",
    "pearson_corr":         "Pearson correlation",
    "two_point":            "Two-point function",
    "connected_corr":       "Connected correlation",
    "hist_symmetric_kl":    "Symmetric KL (hist)",
    "hist_jensen_shannon":  "Jensen-Shannon (hist)",
    "symmetric_kl":         "Symmetric KL (KDE)",
    "jensen_shannon":       "Jensen-Shannon (KDE)",
}

# Compiled regexes for cross-correlation filenames
_CROSS_DIAG_RE = re.compile(
    rf"^(.+?)_cross_diagonal_({_COMP_RE})_vs_({_COMP_RE})$"
)
_CROSS_RE = re.compile(
    rf"^(.+?)_cross_({_COMP_RE})_vs_({_COMP_RE})_(.+)$"
)
# Standard self-correlation: {model}_{component}_{plot_type}[_{metric}]
_SELF_RE = re.compile(
    rf"^(.+?)_({_COMP_RE})_(.+)$"
)


def parse_filename(filename):
    """Parse a figure filename into structured metadata.

    Returns dict with: model, component, plot_type, metric, cross_pair.
    """
    name = filename.removesuffix(".png")
    result = {
        "model": "", "component": "", "plot_type": "", "metric": "",
        "cross_pair": "",
    }

    # 1. Cross-diagonal: {model}_cross_diagonal_{compA}_vs_{compB}
    m = _CROSS_DIAG_RE.match(name)
    if m:
        result["model"] = m.group(1)
        result["component"] = f"{m.group(2)} vs {m.group(3)}"
        result["cross_pair"] = result["component"]
        result["plot_type"] = "cross_diagonal"
        return result

    # 2. Cross-heatmap: {model}_cross_{compA}_vs_{compB}_{metric}
    m = _CROSS_RE.match(name)
    if m:
        result["model"] = m.group(1)
        result["component"] = f"{m.group(2)} vs {m.group(3)}"
        result["cross_pair"] = result["component"]
        result["plot_type"] = "cross_heatmap"
        result["metric"] = m.group(4)
        return result

    # 3. Scalars: {model}_scalars_vs_layer / _scalars_heatmap / _scalar_correlations
    for pt in ("scalars_vs_layer", "scalars_heatmap", "scalar_correlations"):
        tag = f"_{pt}"
        if name.endswith(tag):
            result["model"] = name[:-len(tag)]
            result["component"] = "scalars"
            result["plot_type"] = pt
            return result

    # 4. Standard: {model}_{component}_{plot_type}[_{metric}]
    m = _SELF_RE.match(name)
    if m:
        result["model"] = m.group(1)
        result["component"] = m.group(2)
        rest = m.group(3)

        # Match known plot types (longest first is handled by ordering)
        for pt in PLOT_TYPES:
            if rest == pt:
                result["plot_type"] = pt
                return result
            if rest.startswith(pt + "_"):
                result["plot_type"] = pt
                result["metric"] = rest[len(pt) + 1:]
                return result

        # Fallback: entire rest is plot_type
        result["plot_type"] = rest
        return result

    # 5. Total fallback
    result["model"] = name
    return result


# ── Index generation ──────────────────────────────────────────────────

def find_images(figures_dir):
    figures_path = Path(figures_dir)
    if not figures_path.exists():
        print(f"Warning: Figures directory not found: {figures_dir}")
        return []

    images = []
    for img_path in sorted(figures_path.glob("*.png")):
        stat = img_path.stat()
        meta = parse_filename(img_path.name)
        images.append({
            "filename": img_path.name,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            **meta,
        })
    return images


def generate_index(output_dir, figures_subdir="figures", relative_paths=True):
    output_path = Path(output_dir)
    figures_dir = output_path / figures_subdir
    images = find_images(figures_dir)

    base_path = figures_subdir if relative_paths else str(figures_dir.absolute())

    # Collect unique values for each filter dimension
    models = sorted({img["model"] for img in images if img["model"]})
    components = sorted({img["component"] for img in images if img["component"]})
    plot_types = sorted({img["plot_type"] for img in images if img["plot_type"]})
    metrics = sorted({img["metric"] for img in images if img["metric"]})

    index_data = {
        "basePath": base_path,
        "imageCount": len(images),
        "images": images,
        "filters": {
            "models": models,
            "components": components,
            "plot_types": plot_types,
            "metrics": metrics,
        },
        "display": {
            "plot_types": PLOT_TYPE_DISPLAY,
            "metrics": METRIC_DISPLAY,
        },
    }

    index_path = output_path / "image_index.json"
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"Generated index: {len(images)} images")
    print(f"  Models:     {models}")
    print(f"  Components: {components}")
    print(f"  Plot types: {plot_types}")
    print(f"  Metrics:    {metrics}")
    print(f"  Saved to:   {index_path}")

    return index_path, len(images)


def copy_viewer(output_dir, script_dir, force=False):
    import shutil
    viewer_source = Path(script_dir) / "viewer.html"
    viewer_dest = Path(output_dir) / "viewer.html"
    if not viewer_source.exists():
        print(f"Warning: viewer.html not found at {viewer_source}")
        return None
    if not viewer_dest.exists() or force:
        shutil.copy(viewer_source, viewer_dest)
        print(f"Copied viewer.html to {viewer_dest}")
    else:
        print(f"viewer.html already exists (use --force-copy to update)")
    return viewer_dest


def serve_viewer(output_dir, port=8000):
    os.chdir(output_dir)
    Handler = http.server.SimpleHTTPRequestHandler

    def open_browser():
        webbrowser.open(f"http://localhost:{port}/viewer.html")

    Timer(1.5, open_browser).start()

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"\nServing viewer at: http://localhost:{port}/viewer.html")
        print(f"Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate index for experiment viewer")
    parser.add_argument("--out", type=str, default="corr_out",
                        help="Output directory containing experiment results")
    parser.add_argument("--figures-dir", type=str, default="figures",
                        help="Subdirectory containing figures (default: figures)")
    parser.add_argument("--serve", action="store_true",
                        help="Start HTTP server to view results")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--absolute-paths", action="store_true",
                        help="Use absolute paths in index (for file:// viewing)")
    parser.add_argument("--no-copy-viewer", action="store_true")
    parser.add_argument("--force-copy", action="store_true",
                        help="Force update viewer.html")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    print(f"Scanning: {args.out}/{args.figures_dir}")
    index_path, image_count = generate_index(
        args.out, args.figures_dir,
        relative_paths=not args.absolute_paths,
    )

    if image_count == 0:
        print(f"\nNo images found in {args.out}/{args.figures_dir}")
        return

    if not args.no_copy_viewer:
        copy_viewer(args.out, script_dir, force=args.force_copy)

    if args.serve:
        serve_viewer(args.out, args.port)
    else:
        print(f"\nTo view: run with --serve, or:")
        print(f"  cd {args.out} && python -m http.server {args.port}")
        print(f"  open http://localhost:{args.port}/viewer.html")


if __name__ == "__main__":
    main()
