"""
data_loading.py
================
Utilities for discovering, loading, and splitting the PASTIS-derived
Sentinel-2 crop classification dataset.

Expected directory layout (paths configurable via configs/config.yaml):

    root_dir/
        DATA_S2/
            S2_10000.npy      # (T, C=10, H, W) float array
            S2_10001.npy
            ...
        ANNOTATIONS/
            TARGET_10000.npy  # (1, H, W) int array of class labels
            TARGET_10001.npy
            ...
        metadata.geojson      # per-patch acquisition dates / AOI geometry

Design notes
------------
- Each patch can have a *different* number of temporal observations (T),
  and Sentinel-2 revisit dates are irregular. We therefore never assume a
  fixed T; all downstream code either aggregates over the time axis
  (see preprocessing.py) or handles variable-length sequences explicitly.
- Loading is lazy / per-patch. For a dataset of this size (patch-level
  .npy files, not one giant array) there is no need to load everything
  into memory at once -- we load what we need, when we need it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import geopandas as gpd
except ImportError:  # geopandas is optional -- only needed for AOI/metadata EDA
    gpd = None


# Note: filenames look like, for example, "S2_20003.npy" / "TARGET_20003.npy". The
# leading "S2" prefix itself contains a digit, so we must take the LAST
# run of digits in the stem (immediately before the extension), not the
# first, or "S2_20003" would incorrectly yield "2" instead of "20003".
PATCH_ID_RE = re.compile(r"(\d+)(?!.*\d)")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """Load the YAML pipeline configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Patch discovery
# --------------------------------------------------------------------------- #

def _extract_patch_id(filename: str) -> Optional[str]:
    m = PATCH_ID_RE.search(filename)
    return m.group(1) if m else None


def list_patch_ids(root_dir: str | Path, s2_dir: str = "DATA_S2",
                    ann_dir: str = "ANNOTATIONS") -> List[str]:
    """
    Discover patch IDs that have BOTH a Sentinel-2 file and a matching
    annotation file present on disk. Returns a sorted list of patch-id
    strings (e.g. ["10000", "10001", ...]).
    """
    root_dir = Path(root_dir)
    s2_path = root_dir / s2_dir
    ann_path = root_dir / ann_dir

    if not s2_path.exists():
        raise FileNotFoundError(
            f"Sentinel-2 directory not found: {s2_path}. "
            f"Place the PASTIS data under this path (see README.md)."
        )

    s2_ids = {
        _extract_patch_id(p.name) for p in s2_path.glob("S2_*.npy")
    }
    ann_ids = {
        _extract_patch_id(p.name) for p in ann_path.glob("TARGET_*.npy")
    } if ann_path.exists() else set()

    s2_ids.discard(None)
    ann_ids.discard(None)

    matched = sorted(s2_ids & ann_ids, key=lambda x: int(x))

    missing_ann = s2_ids - ann_ids
    if missing_ann:
        print(f"[data_loading] Warning: {len(missing_ann)} patches have S2 data "
              f"but no annotation and will be skipped (e.g. {sorted(missing_ann)[:5]}).")

    return matched


# --------------------------------------------------------------------------- #
# Single-patch loading
# --------------------------------------------------------------------------- #

def load_s2_patch(patch_id: str, root_dir: str | Path, s2_dir: str = "DATA_S2") -> np.ndarray:
    """
    Load a single Sentinel-2 patch.

    Returns
    -------
    np.ndarray of shape (T, C=10, H, W)
    """
    path = Path(root_dir) / s2_dir / f"S2_{patch_id}.npy"
    arr = np.load(path)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array (T,C,H,W) for {path}, got shape {arr.shape}")
    return arr


def load_target_patch(patch_id: str, root_dir: str | Path, ann_dir: str = "ANNOTATIONS") -> np.ndarray:
    """
    Load a single annotation patch (0th layer only, as provided for this
    assignment).

    Returns
    -------
    np.ndarray of shape (H, W), integer class labels.
    """
    path = Path(root_dir) / ann_dir / f"TARGET_{patch_id}.npy"
    arr = np.load(path)
    # Provided as (1, H, W) -- squeeze the leading singleton layer axis.
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected target shape for {path}: {arr.shape}")
    return arr.astype(np.int64)


def load_patch(patch_id: str, root_dir: str | Path,
                s2_dir: str = "DATA_S2", ann_dir: str = "ANNOTATIONS") -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning (s2_array, target_array) for one patch."""
    s2 = load_s2_patch(patch_id, root_dir, s2_dir)
    target = load_target_patch(patch_id, root_dir, ann_dir)
    return s2, target


# --------------------------------------------------------------------------- #
# Metadata (temporal / geographic)
# --------------------------------------------------------------------------- #

def load_metadata(root_dir: str | Path, metadata_file: str = "metadata.geojson"):
    """
    Load the per-patch metadata.geojson (acquisition dates, AOI geometry,
    patch id, etc). Returns a GeoDataFrame if geopandas is available,
    otherwise a plain dict parsed from the raw JSON.
    """
    path = Path(root_dir) / metadata_file
    if not path.exists():
        print(f"[data_loading] metadata file not found at {path}; "
              f"AOI / date-based EDA will be skipped.")
        return None

    if gpd is not None:
        return gpd.read_file(path)

    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Dataset-level inspection helpers (used by EDA, but generic enough for reuse)
# --------------------------------------------------------------------------- #

@dataclass
class PatchInfo:
    patch_id: str
    n_obs: int
    n_bands: int
    height: int
    width: int
    n_valid_labeled_pixels: int
    unique_classes: List[int]


def inspect_patch(patch_id: str, root_dir: str | Path,
                   s2_dir: str = "DATA_S2", ann_dir: str = "ANNOTATIONS",
                   void_label: int = 19) -> PatchInfo:
    """Load one patch just to record its shape / class statistics (lightweight EDA)."""
    s2 = load_s2_patch(patch_id, root_dir, s2_dir)
    target = load_target_patch(patch_id, root_dir, ann_dir)
    T, C, H, W = s2.shape
    valid_mask = target != void_label
    return PatchInfo(
        patch_id=patch_id,
        n_obs=T,
        n_bands=C,
        height=H,
        width=W,
        n_valid_labeled_pixels=int(valid_mask.sum()),
        unique_classes=sorted(np.unique(target).tolist()),
    )


def build_dataset_summary(patch_ids: List[str], root_dir: str | Path,
                           s2_dir: str = "DATA_S2", ann_dir: str = "ANNOTATIONS",
                           void_label: int = 19) -> pd.DataFrame:
    """Build a per-patch summary DataFrame -- the backbone of the EDA notebook section."""
    rows = []
    for pid in patch_ids:
        info = inspect_patch(pid, root_dir, s2_dir, ann_dir, void_label)
        rows.append({
            "patch_id": info.patch_id,
            "n_observations": info.n_obs,
            "n_bands": info.n_bands,
            "height": info.height,
            "width": info.width,
            "n_valid_pixels": info.n_valid_labeled_pixels,
            "n_unique_classes": len(info.unique_classes),
        })
    return pd.DataFrame(rows)


def class_pixel_counts(patch_ids: List[str], root_dir: str | Path,
                        ann_dir: str = "ANNOTATIONS", num_classes: int = 20) -> pd.Series:
    """
    Count total pixels per class across a list of patches. This is the
    basis for the class-imbalance analysis in the EDA and for setting
    class weights during training.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for pid in patch_ids:
        target = load_target_patch(pid, root_dir, ann_dir)
        vals, cnts = np.unique(target, return_counts=True)
        for v, c in zip(vals, cnts):
            if 0 <= v < num_classes:
                counts[v] += c
    return pd.Series(counts, index=range(num_classes), name="pixel_count")


# --------------------------------------------------------------------------- #
# Train / val / test split
# --------------------------------------------------------------------------- #

def create_splits(patch_ids: List[str], train_ratio: float = 0.7,
                   val_ratio: float = 0.15, test_ratio: float = 0.15,
                   seed: int = 42) -> Dict[str, List[str]]:
    """
    Create a PATCH-LEVEL random split (not pixel-level). Splitting by
    patch, rather than by pixel, avoids spatial leakage: pixels from the
    same physical field / patch are highly spatially autocorrelated, so
    a pixel-level split would let the model "see" neighbouring pixels of
    a test field during training and inflate reported performance.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "split ratios must sum to 1.0"

    rng = np.random.default_rng(seed)
    ids = np.array(patch_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    train_ids = ids[:n_train].tolist()
    val_ids = ids[n_train:n_train + n_val].tolist()
    test_ids = ids[n_train + n_val:].tolist()

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def save_splits(splits: Dict[str, List[str]], out_dir: str | Path) -> None:
    """Persist each split as a plain-text file, one patch id per line."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        with open(out_dir / f"{name}.txt", "w") as f:
            f.write("\n".join(ids) + "\n")
    print(f"[data_loading] Saved splits to {out_dir}: "
          f"{ {k: len(v) for k, v in splits.items()} }")


def load_splits(splits_dir: str | Path) -> Dict[str, List[str]]:
    """Load previously-saved split text files back into a dict of lists."""
    splits_dir = Path(splits_dir)
    splits = {}
    for name in ["train", "val", "test"]:
        path = splits_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}. Run make_splits first.")
        with open(path) as f:
            splits[name] = [line.strip() for line in f if line.strip()]
    return splits


# --------------------------------------------------------------------------- #
# CLI entry point: `python -m src.data_loading --config configs/config.yaml`
# --------------------------------------------------------------------------- #

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Discover patches and create train/val/test splits.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root_dir = cfg["data"]["root_dir"]

    patch_ids = list_patch_ids(root_dir, cfg["data"]["s2_dir"], cfg["data"]["ann_dir"])
    print(f"[data_loading] Found {len(patch_ids)} usable patches.")

    max_patches = cfg.get("sampling", {}).get("max_patches")
    if max_patches:
        rng = np.random.default_rng(cfg["sampling"]["seed"])
        patch_ids = rng.choice(patch_ids, size=min(max_patches, len(patch_ids)), replace=False).tolist()
        print(f"[data_loading] Subsampled to {len(patch_ids)} patches (max_patches={max_patches}).")

    splits = create_splits(
        patch_ids,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
        seed=cfg["split"]["seed"],
    )
    save_splits(splits, cfg["split"]["splits_dir"])


if __name__ == "__main__":
    main()
