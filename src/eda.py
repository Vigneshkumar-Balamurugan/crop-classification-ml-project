"""
eda.py
======
Task B: exploratory data analysis. Produces the dataset summary table,
class distribution plot, a handful of RGB/label example figures, and
(if metadata.geojson is available) the AOI extent plot.

Run with: python -m src.eda --config configs/config.yaml
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_loading import (
    load_config, list_patch_ids, build_dataset_summary, class_pixel_counts,
    load_s2_patch, load_target_patch, load_metadata,
)
from src.visualization import (
    plot_rgb, plot_label_map, plot_class_distribution, plot_aoi_extent,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run exploratory data analysis.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n_examples", type=int, default=4, help="number of example patches to visualize")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    fig_dir = Path(cfg["output"]["figures_dir"])
    metrics_dir = Path(cfg["output"]["metrics_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    patch_ids = list_patch_ids(data_cfg["root_dir"], data_cfg["s2_dir"], data_cfg["ann_dir"])
    print(f"[eda] {len(patch_ids)} patches available.")

    max_patches = cfg.get("sampling", {}).get("max_patches")
    eda_ids = patch_ids
    if max_patches:
        rng = np.random.default_rng(cfg["sampling"]["seed"])
        eda_ids = rng.choice(patch_ids, size=min(max_patches, len(patch_ids)), replace=False).tolist()

    # 1. Per-patch summary table (n_observations, dims, valid pixels, n classes)
    summary = build_dataset_summary(eda_ids, data_cfg["root_dir"], data_cfg["s2_dir"], data_cfg["ann_dir"],
                                     void_label=data_cfg["void_label"])
    summary.to_csv(metrics_dir / "patch_summary.csv", index=False)
    print("[eda] Patch summary:")
    print(summary.describe(include="all"))

    # 2. Class distribution across the (sampled) dataset
    counts = class_pixel_counts(eda_ids, data_cfg["root_dir"], data_cfg["ann_dir"],
                                 num_classes=data_cfg["num_classes"])
    counts.to_csv(metrics_dir / "class_pixel_counts.csv")
    plot_class_distribution(
        counts, data_cfg["class_names"],
        save_path=str(fig_dir / "class_distribution.png"),
    )

    # 3. Example RGB / label visualisations
    rng = np.random.default_rng(cfg["sampling"]["seed"])
    example_ids = rng.choice(eda_ids, size=min(args.n_examples, len(eda_ids)), replace=False)
    for pid in example_ids:
        s2 = load_s2_patch(pid, data_cfg["root_dir"], data_cfg["s2_dir"])
        target = load_target_patch(pid, data_cfg["root_dir"], data_cfg["ann_dir"])
        plot_rgb(s2, title=f"Patch {pid} RGB (median composite)",
                 save_path=str(fig_dir / f"rgb_example_{pid}.png"))
        plot_label_map(target, title=f"Patch {pid} labels", num_classes=data_cfg["num_classes"],
                        save_path=str(fig_dir / f"label_example_{pid}.png"))

    # 4. AOI extent, if metadata is available
    metadata = load_metadata(data_cfg["root_dir"], data_cfg["metadata_file"])
    if metadata is not None and hasattr(metadata, "plot"):
        plot_aoi_extent(metadata, save_path=str(fig_dir / "aoi_extent.png"))

    print(f"[eda] Figures written to {fig_dir}, tables written to {metrics_dir}")


if __name__ == "__main__":
    main()
