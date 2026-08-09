"""
visualization.py
=================
Plotting helpers for EDA and model-output visualisation (Tasks B and F).
All functions save a figure to disk (so they can be called headlessly
from scripts) and also return the matplotlib Figure for notebook use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

# A fixed, readable qualitative palette for the 20 PASTIS classes
# (index 0 = background -> black, index 19 = void -> light grey).
_CLASS_COLORS_HEX = [
    "#000000", "#a6cee3", "#ff7f00", "#e31a1c", "#33a02c", "#66c2a5",
    "#d62728", "#fb9a99", "#9467bd", "#c2c2f0", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#bcbd22", "#dbdb8d", "#c7c7c7",
    "#17becf", "#d9d9d9",
]


def _class_cmap(num_classes: int = 20):
    colors = _CLASS_COLORS_HEX[:num_classes]
    cmap = ListedColormap(colors)
    bounds = np.arange(num_classes + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _save(fig, save_path_name: Optional[str]):
    if save_path_name:
        Path(save_path_name).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path_name, dpi=150, bbox_inches="tight")
        print(f"[visualization] saved {save_path_name}")
    return fig


# --------------------------------------------------------------------------- #
# RGB composite
# --------------------------------------------------------------------------- #

def make_rgb_composite(s2_patch: np.ndarray, time_index: Optional[int] = None,
                        red_idx: int = 2, green_idx: int = 1, blue_idx: int = 0,
                        clip_percentile=(2, 98)) -> np.ndarray:
    """
    Build a true-color RGB image from a Sentinel-2 patch using bands
    B4 (red, idx 2), B3 (green, idx 1), B2 (blue, idx 0) -- the standard
    natural-color combination.

    If time_index is None, the temporal MEDIAN composite is used (more
    robust to residual cloud/shadow contamination in any single
    acquisition than picking one arbitrary date).
    """
    if s2_patch.ndim == 4:
        if time_index is None:
            img = np.median(s2_patch, axis=0)  # (C, H, W)
        else:
            img = s2_patch[time_index]
    else:
        img = s2_patch  # already (C, H, W)

    rgb = np.stack([img[red_idx], img[green_idx], img[blue_idx]], axis=-1).astype(np.float32)

    lo, hi = np.percentile(rgb, clip_percentile[0]), np.percentile(rgb, clip_percentile[1])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    return rgb


def plot_rgb(s2_patch: np.ndarray, title: str = "Sentinel-2 RGB composite",
             save_path: Optional[str] = None, ax=None):
    rgb = make_rgb_composite(s2_patch)
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")

    save_path_name = None

    if created_fig:
        filename = title.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
        save_path_name = str(Path(save_path) / filename)
        _save(fig, save_path_name)

    return fig


# --------------------------------------------------------------------------- #
# Label / prediction maps
# --------------------------------------------------------------------------- #

def plot_label_map(label: np.ndarray, title: str = "Ground truth", num_classes: int = 20,
                    save_path: Optional[str] = None, ax=None):
    cmap, norm = _class_cmap(num_classes)
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    ax.imshow(label, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    save_path_name = None

    if save_path:
        filename = title.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
        save_path_name = str(Path(save_path) / filename)

    if created_fig:
        _save(fig, save_path_name)
    return fig


def plot_rgb_label_prediction(s2_patch: np.ndarray, label: np.ndarray, prediction: np.ndarray,
                               class_names: Dict[int, str], num_classes: int = 20,
                               patch_id: str = "", save_path: Optional[str] = None):
    """Task F: three-panel figure -- RGB | ground truth | prediction, shared legend."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_rgb(s2_patch, title=f"S2 RGB (patch {patch_id})", ax=axes[0])
    plot_label_map(label, title="Ground truth", num_classes=num_classes, ax=axes[1])
    plot_label_map(prediction, title="Prediction", num_classes=num_classes, ax=axes[2])

    cmap, norm = _class_cmap(num_classes)
    present = sorted(set(np.unique(label).tolist()) | set(np.unique(prediction).tolist()))
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cmap(norm(c))) for c in present
    ]
    labels = [class_names.get(c, str(c)) for c in present]
    fig.legend(handles, labels, loc="lower center", ncol=min(len(present), 7),
               bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# --------------------------------------------------------------------------- #
# Class distribution
# --------------------------------------------------------------------------- #

def plot_class_distribution(pixel_counts, class_names: Dict[int, str],
                             title: str = "Class distribution (pixel counts)",
                             log_scale: bool = True, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [class_names.get(i, str(i)) for i in pixel_counts.index]
    ax.bar(names, pixel_counts.values, color="#4c72b0")
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel("Pixel count" + (" (log scale)" if log_scale else ""))
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.tight_layout()

    save_path_name = None

    if save_path:
        filename = title.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
        save_path_name = str(Path(save_path) / filename)

    _save(fig, save_path_name)
    return fig


# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], normalize: bool = True,
                           title: str = "Confusion matrix", save_path: Optional[str] = None):
    cm = cm.astype(np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(class_names)), max(6, 0.5 * len(class_names))))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# --------------------------------------------------------------------------- #
# Approximate AOI extent (from metadata.geojson, if available)
# --------------------------------------------------------------------------- #

def plot_aoi_extent(metadata_gdf, save_path: Optional[str] = None, title: str = "AOI extent (patch footprints)"):
    """Requires a GeoDataFrame as returned by data_loading.load_metadata."""
    fig, ax = plt.subplots(figsize=(6, 6))
    metadata_gdf.plot(ax=ax, edgecolor="black", facecolor="#4c72b0", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("Longitude / X")
    ax.set_ylabel("Latitude / Y")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
