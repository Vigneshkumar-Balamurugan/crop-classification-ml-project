"""
evaluate.py
===========
Task E: evaluate the trained model on validation and test patches.

Unlike training (which samples a fraction of pixels per patch for
speed), evaluation runs the classifier on EVERY valid pixel of every
patch in the split, giving a full-image prediction that can also be
used directly for the visualisations in Task F.

Metrics computed:
  - Overall pixel accuracy
  - Per-class precision / recall / F1 (sklearn classification_report)
  - Mean IoU (averaged over classes actually present in the evaluated
    split, since classes absent from val/test have an undefined IoU)
  - Confusion matrix (raw counts + row-normalized figure)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
try:
    from tqdm import tqdm
except ImportError:  # tqdm is a convenience progress bar, not a hard requirement
    def tqdm(iterable, **kwargs):
        return iterable

from src.data_loading import load_config, load_splits, load_s2_patch, load_target_patch
from src.preprocessing import compute_temporal_features, clip_and_normalize, flatten_all_pixels
from src.visualization import plot_confusion_matrix, plot_rgb_label_prediction


def compute_iou_per_class(cm: np.ndarray) -> np.ndarray:
    """
    IoU_c = TP_c / (TP_c + FP_c + FN_c), derived directly from a
    (num_classes, num_classes) confusion matrix where rows=true,
    cols=predicted.
    """
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.divide(tp, denom, out=np.full_like(tp, np.nan), where=denom != 0)
    return iou


def predict_patch(patch_id, clf, norm_stats, cfg):
    """Run the classifier on every valid pixel of one patch; returns (pred_map, label_map, valid_mask, s2)."""
    data_cfg = cfg["data"]
    prep_cfg = cfg["preprocessing"]

    s2 = load_s2_patch(patch_id, data_cfg["root_dir"], data_cfg["s2_dir"])
    target = load_target_patch(patch_id, data_cfg["root_dir"], data_cfg["ann_dir"])

    features = compute_temporal_features(s2, stats=prep_cfg["temporal_stats"])
    if prep_cfg["normalize"]:
        features, _ = clip_and_normalize(features, stats=norm_stats)

    exclude = [data_cfg["void_label"]] if prep_cfg["exclude_void_label"] else []
    X, y, valid_mask = flatten_all_pixels(features, target, exclude_labels=exclude)

    # Pixels the classifier never predicts on (e.g. void-labelled) are
    # filled with the void_label class itself rather than -1, so
    # qualitative plots render them with their own distinct colour
    # instead of colliding visually with class 0 (background).
    fill_value = data_cfg["void_label"]

    if len(X) == 0:
        pred_map = np.full(target.shape, fill_value, dtype=np.int64)
        return pred_map, target, valid_mask, s2

    preds = clf.predict(X)
    pred_map = np.full(target.shape, fill_value, dtype=np.int64)
    pred_map[valid_mask] = preds
    return pred_map, target, valid_mask, s2


def evaluate_split(split_name, patch_ids, clf, norm_stats, cfg, label_encoder, save_predictions=True,
                    save_qualitative_examples=3):
    data_cfg = cfg["data"]
    num_classes = data_cfg["num_classes"]
    class_names_map = data_cfg["class_names"]
    metrics_dir = Path(cfg["output"]["metrics_dir"])
    pred_dir = Path(cfg["output"]["predictions_dir"])
    fig_dir = Path(cfg["output"]["figures_dir"])
    model_type = cfg["model"]["type"].lower()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_true, all_pred = [], []
    qualitative_saved = 0

    for pid in tqdm(patch_ids, desc=f"Evaluating [{split_name}]"):
        pred_map, target, valid_mask, s2 = predict_patch(pid, clf, norm_stats, cfg)

        y_true = target[valid_mask]
        y_pred_encoded  = pred_map[valid_mask]

        y_pred = label_encoder.inverse_transform(y_pred_encoded.astype(int))  

        all_true.append(y_true)
        all_pred.append(y_pred)

        if save_predictions:
            np.save(pred_dir / f"pred_{split_name}_{pid}_{model_type}.npy", pred_map)

        if save_qualitative_examples and qualitative_saved < save_qualitative_examples:
            plot_rgb_label_prediction(
                s2, target, pred_map, class_names_map, num_classes=num_classes,
                patch_id=pid,
                save_path=str(fig_dir / f"qualitative_{split_name}_{pid}_{model_type}.png"),
            )
            qualitative_saved += 1

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    acc = accuracy_score(y_true, y_pred)
    labels_present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    target_names = [class_names_map.get(c, str(c)) for c in labels_present]

    report = classification_report(
        y_true, y_pred, labels=labels_present, target_names=target_names,
        output_dict=True, zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    iou = compute_iou_per_class(cm)
    miou = float(np.nanmean(iou))

    metrics = {
        "split": split_name,
        "overall_accuracy": float(acc),
        "mean_iou": miou,
        "per_class_iou": {class_names_map.get(c, str(c)): float(i) for c, i in zip(labels_present, iou)},
        "classification_report": report,
        "n_pixels_evaluated": int(len(y_true)),
        "n_patches": len(patch_ids),
    }

    with open(metrics_dir / f"metrics_{split_name}_{model_type}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(
        cm, target_names, normalize=True,
        title=f"Confusion matrix ({split_name}, row-normalized)",
        save_path=str(fig_dir / f"confusion_matrix_{split_name}_{model_type}.png"),
    )

    print(f"[evaluate] {split_name}: overall accuracy={acc:.4f}, mean IoU={miou:.4f} "
          f"over {len(patch_ids)} patches / {len(y_true):,} pixels")

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the trained model on val and test splits.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = load_splits(cfg["split"]["splits_dir"])

    model_dir = Path(cfg["output"]["model_dir"])
    clf = joblib.load(model_dir / "model.joblib")
    norm_stats = joblib.load(model_dir / "normalization_stats.joblib")
    label_encoder = joblib.load(model_dir / "label_encoder.joblib")

    for split_name in args.splits:
        evaluate_split(split_name, splits[split_name], clf, norm_stats, cfg, label_encoder)


if __name__ == "__main__":
    main()
