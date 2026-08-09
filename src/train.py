"""
train.py
========
Trains the crop-type classification baseline (Task D).

Model choice: per-pixel classification (Random Forest or XGBoost) on temporal band statistics.
------------------------------------------------------------------
Rationale (see report.md for the full discussion):
  - Strong, well-understood baseline for multi-temporal optical crop
    classification; performs competitively with deep models on small-
    to-medium labeled datasets and needs no GPU.
  - Handles the class imbalance reasonably well via `class_weight` (in RandomForestClassifier) or `scale_pos_weight` (in XGBClassifier),
    is robust to modest amounts of noisy/mislabeled pixels, and gives
    interpretable feature importances (which spectral band / stat
    combination actually drives class separability).
  - Fast to train and iterate on CPU, well-reasoned baseline over an
    unnecessarily complex architecture.

The script:
  1. Loads the train split's patch IDs.
  2. For each patch: loads S2 + target, computes temporal features,
     fits/normalizes, and randomly samples a fraction of valid pixels.
  3. Concatenates sampled pixels across all training patches into one
     tabular (X, y) dataset.
  4. Fits a RandomForestClassifier or XGBClassifier.
  5. Saves the model (joblib), the fitted normalization stats (needed
     to transform val/test patches identically), and a feature-
     importance plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
try:
    from tqdm import tqdm
except ImportError:  # tqdm is a convenience progress bar, not a hard requirement
    def tqdm(iterable, **kwargs):
        return iterable

from src.data_loading import load_config, load_splits, load_s2_patch, load_target_patch
from src.preprocessing import compute_temporal_features, clip_and_normalize, sample_pixels


def build_training_set(patch_ids, cfg):
    data_cfg = cfg["data"]
    prep_cfg = cfg["preprocessing"]

    X_list, y_list = [], []
    norm_stats = None  # fit normalization on the fly using the first patch, then reuse

    for pid in tqdm(patch_ids, desc="Building training features"):
        s2 = load_s2_patch(pid, data_cfg["root_dir"], data_cfg["s2_dir"])
        target = load_target_patch(pid, data_cfg["root_dir"], data_cfg["ann_dir"])

        features = compute_temporal_features(s2, stats=prep_cfg["temporal_stats"])

        if prep_cfg["normalize"]:
            features, stats = clip_and_normalize(
                features, tuple(prep_cfg["clip_percentile"]), stats=norm_stats
            )
            if norm_stats is None:
                norm_stats = stats  # freeze stats fitted on the first (train) patch batch

        exclude = [data_cfg["void_label"]] if prep_cfg["exclude_void_label"] else []
        if prep_cfg.get("exclude_background_from_training"):
            exclude.append(data_cfg["background_label"])

        X, y = sample_pixels(
            features, target,
            fraction=prep_cfg["pixel_sample_fraction"],
            exclude_labels=exclude,
            seed=cfg["sampling"]["seed"],
        )
        if len(X) > 0:
            X_list.append(X)
            y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y, norm_stats

def create_model(model_cfg):

    model_type = model_cfg["type"].lower()

    if model_type == "random_forest":
        return RandomForestClassifier(
            **model_cfg["random_forest"]
        )

    elif model_type == "xgboost":
        return XGBClassifier(
            **model_cfg["xgboost"]
        )

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            "Choose 'random_forest' or 'xgboost'."
        )


def main():
    import argparse
    from sklearn.preprocessing import LabelEncoder
    parser = argparse.ArgumentParser(description="Train the crop classification baseline.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = load_splits(cfg["split"]["splits_dir"])
    train_ids = splits["train"]
    print(f"[train] {len(train_ids)} training patches.")

    X, y, norm_stats = build_training_set(train_ids, cfg)
    print(f"[train] Training matrix: X={X.shape}, y={y.shape}, "
          f"classes present={sorted(np.unique(y).tolist())}")

    clf = create_model(cfg["model"])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    clf.fit(X, y_encoded)
    train_acc = clf.score(X, y_encoded)
    print(f"[train] In-sample training accuracy: {train_acc:.4f} "
          f"(expect this to be optimistic vs. val/test -- see evaluate.py)")

    model_dir = Path(cfg["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, model_dir / "model.joblib")
    joblib.dump(norm_stats, model_dir / "normalization_stats.joblib")
    joblib.dump(label_encoder, model_dir / "label_encoder.joblib")

    with open(model_dir / "train_config_used.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # Feature importances -- useful diagnostic for report.md
    stats_names = cfg["preprocessing"]["temporal_stats"]
    band_names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    feature_names = [f"{b}_{s}" for s in stats_names for b in band_names]
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]

    metrics_dir = Path(cfg["output"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "feature_importances.json", "w") as f:
        json.dump(
            {feature_names[i]: float(importances[i]) for i in order},
            f, indent=2,
        )

    print(f"[train] Model + normalization stats saved to {model_dir}")
    print(f"[train] Top 10 features by importance:")
    for i in order[:10]:
        print(f"    {feature_names[i]:>10s}: {importances[i]:.4f}")


if __name__ == "__main__":
    main()
