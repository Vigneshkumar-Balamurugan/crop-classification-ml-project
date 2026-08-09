"""
preprocessing.py
=================
Turns a raw Sentinel-2 patch cube (T, C, H, W) plus a target map (H, W)
into fixed-size features suitable for a per-pixel classifier, and
handles normalization / class-imbalance-aware pixel sampling.

Why per-pixel temporal statistics (mean/std/min/max) instead of feeding
raw time series?
---------------------------------------------------------------------
1. Each patch can have a different number of observations T (irregular
   Sentinel-2 revisits, cloud-affected acquisitions dropped upstream,
   etc.), so a fixed-length feature vector cannot simply be "the T
   observations flattened".
2. Temporal statistics are a well-established, cheap/efficient way to summarize a
   growing-season time series: the mean captures the average spectral
   signature, std captures phenological variability (crops that change
   a lot through the season vs. stable land cover like grassland),
   min/max capture the extremes of the growth cycle (e.g. NDVI peak).
3. This keeps the baseline model simple, fast to train on CPU.

"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

STAT_FUNCS = {
    "mean": lambda x, axis: np.mean(x, axis=axis),
    "std": lambda x, axis: np.std(x, axis=axis),
    "min": lambda x, axis: np.min(x, axis=axis),
    "max": lambda x, axis: np.max(x, axis=axis),
    "median": lambda x, axis: np.median(x, axis=axis),
}


def mask_invalid_observations(s2: np.ndarray, nodata_value: float = 0.0) -> np.ndarray:
    """
    Very simple cloud / no-data screening: an observation (time step) is
    treated as unusable for a pixel if ALL bands are exactly the nodata
    sentinel value at that pixel (a common convention for masked-out
    Sentinel-2 acquisitions). This is intentionally conservative -- see
    report.md limitations section regarding more robust cloud masking
    (e.g. using a supplied cloud probability layer, if available).

    Returns a boolean array of shape (T, H, W): True where the
    observation is considered valid.
    """
    # A pixel/time-step is "all zero" across every band -> treat as invalid.
    invalid = np.all(s2 == nodata_value, axis=1)  # (T, H, W)
    return ~invalid


def compute_temporal_features(s2: np.ndarray, stats: List[str] = ("mean", "std", "min", "max"),
                               mask_nodata: bool = True) -> np.ndarray:
    """
    Collapse the temporal axis of a (T, C, H, W) cube into per-pixel
    statistics, returning an array of shape (C * len(stats), H, W).

    If mask_nodata=True, invalid observations (see
    mask_invalid_observations) are excluded from the statistics via
    masked arrays, so a cloud-affected acquisition does not bias the
    mean/std for that pixel.
    """
    T, C, H, W = s2.shape
    feature_stack = []

    if mask_nodata:
        valid = mask_invalid_observations(s2)  # (T, H, W)
        # Broadcast to (T, C, H, W) and build a masked array.
        valid_b = np.broadcast_to(valid[:, None, :, :], s2.shape)
        s2_masked = np.ma.array(s2, mask=~valid_b)
    else:
        s2_masked = np.ma.array(s2, mask=False)

    for stat in stats:
        if stat == "mean":
            agg = s2_masked.mean(axis=0)
        elif stat == "std":
            agg = s2_masked.std(axis=0)
        elif stat == "min":
            agg = s2_masked.min(axis=0)
        elif stat == "max":
            agg = s2_masked.max(axis=0)
        elif stat == "median":
            agg = np.ma.median(s2_masked, axis=0)
        else:
            raise ValueError(f"Unsupported stat: {stat}")
        feature_stack.append(np.ma.filled(agg, 0.0))

    return np.concatenate(feature_stack, axis=0).astype(np.float32)  # (C*len(stats), H, W)


def clip_and_normalize(features: np.ndarray, clip_percentile: Tuple[float, float] = (1, 99),
                        stats: Dict[str, np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Robustly normalize a (F, H, W) feature stack per-channel: clip to
    the given percentile range (guards against outlier reflectance
    spikes / residual cloud contamination) then min-max scale to [0, 1].

    If `stats` (a dict with 'low'/'high' arrays of shape (F,)) is
    supplied, those bounds are reused instead of recomputed -- this is
    essential for applying the SAME normalization fitted on the
    training set to validation/test patches, avoiding data leakage.
    """
    F = features.shape[0]
    flat = features.reshape(F, -1)

    if stats is None:
        low = np.percentile(flat, clip_percentile[0], axis=1)
        high = np.percentile(flat, clip_percentile[1], axis=1)
        stats = {"low": low, "high": high}
    else:
        low, high = stats["low"], stats["high"]

    low = low.reshape(F, 1, 1)
    high = high.reshape(F, 1, 1)
    denom = np.clip(high - low, 1e-6, None)

    normalized = np.clip((features - low) / denom, 0.0, 1.0)
    return normalized.astype(np.float32), stats


def sample_pixels(features: np.ndarray, target: np.ndarray, fraction: float = 0.15,
                   exclude_labels: List[int] = (19,), seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample a fraction of valid pixels from one patch for
    per-pixel model training. Sampling (rather than using every pixel)
    keeps training tractable given that neighbouring pixels within a
    field are highly redundant, and lets us control memory usage when
    working across many patches.

    Returns
    -------
    X : (N, F) sampled feature vectors
    y : (N,) corresponding class labels
    """
    F, H, W = features.shape
    valid_mask = np.ones((H, W), dtype=bool)
    for lbl in exclude_labels:
        valid_mask &= (target != lbl)

    ys, xs = np.where(valid_mask)
    n_valid = len(ys)
    if n_valid == 0:
        return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=np.int64)

    n_sample = max(1, int(round(n_valid * fraction)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_valid, size=min(n_sample, n_valid), replace=False)

    sel_y, sel_x = ys[idx], xs[idx]
    X = features[:, sel_y, sel_x].T  # (N, F)
    y = target[sel_y, sel_x]
    return X.astype(np.float32), y.astype(np.int64)


def flatten_all_pixels(features: np.ndarray, target: np.ndarray,
                        exclude_labels: List[int] = (19,)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten every valid pixel of a patch (used at inference time, where
    we need a prediction for the whole image, not just a sample).

    Returns
    -------
    X : (N, F) feature vectors for valid pixels
    y : (N,) true labels for valid pixels
    valid_mask : (H, W) boolean mask marking which pixels were kept
    """
    F, H, W = features.shape
    valid_mask = np.ones((H, W), dtype=bool)
    for lbl in exclude_labels:
        valid_mask &= (target != lbl)

    X = features[:, valid_mask].T
    y = target[valid_mask]
    return X.astype(np.float32), y.astype(np.int64), valid_mask
