# Crop Type Classification from Multi-Temporal Sentinel-2 Imagery

A reproducible pipeline for pixel-level crop-type classification on a
PASTIS-derived subset of Sentinel-2 imagery.

Here is the model performance for one case:

![Model Performance](/outputs/figures/qualitative_test_30123_xgboost.png)

## 1. Project overview

Given per-patch Sentinel-2 time series (`(T, 10, H, W)`) and matching
crop-type annotation maps (`(H, W)`, 20 classes including background
and a void label), this project:

1. Loads and inspects the raw data (Task A)
2. Explores dataset structure, class balance, and imagery (Task B)
3. Creates a reproducible patch-level train/val/test split (Task C)
4. Trains a **Random Forest** or **XGBoost** baseline on per-pixel temporal band
   statistics (Task D)
5. Evaluates it with pixel accuracy, per-class precision/recall/F1,
   mean IoU, and a confusion matrix (Task E)
6. Visualises predictions against ground truth (Task F)
7. Discusses results, limitations, and next steps in
   [`report.md`](report.md) (Task G)

**Model choice.** The assignment explicitly favours "a simpler baseline
model with strong reasoning ... over an unnecessarily complex model."
A per-pixel Random Forest or XGBoost trained on temporal mean/std/min/max
statistics per band is a well-established, CPU-only, fast-to-train
baseline for multi-temporal optical crop classification, and yields
interpretable feature importances. See `report.md` for the full justification 
and a discussion of deep-learning alternatives (e.g. a temporal CNN / U-Net) as a next step.

## 2. Dataset structure

This repository does **not** include the dataset due to space constraints, but it can be provided upon request. 
Place the data as follows:

```
project-root/
└── data/
    └── PASTIS/
        ├── DATA_S2/
        │   ├── S2_10000.npy      # (T, 10, H, W) float array
        │   ├── S2_10001.npy
        │   └── ...
        ├── ANNOTATIONS/
        │   ├── TARGET_10000.npy  # (1, H, W) int array, 0th annotation layer
        │   ├── TARGET_10001.npy
        │   └── ...
        └── metadata.geojson       # per-patch acquisition dates / AOI geometry
```

`configs/config.yaml` -> `data.root_dir` points at `data/PASTIS` by
default; change it if you store the data elsewhere.

Sentinel-2 bands (channel order in `DATA_S2/S2_*.npy`):

| Index | Band | Name |
|---|---|---|
| 0 | B2 | Blue |
| 1 | B3 | Green |
| 2 | B4 | Red |
| 3 | B5 | Red Edge 1 |
| 4 | B6 | Red Edge 2 |
| 5 | B7 | Red Edge 3 |
| 6 | B8 | NIR |
| 7 | B8A | Narrow NIR |
| 8 | B11 | SWIR 1 |
| 9 | B12 | SWIR 2 |

Class scheme (0 = background, 1-18 = crop/land-cover classes, 19 =
void label) is defined in `configs/config.yaml` -> `data.class_names`
and matches the PASTIS label scheme given in the assignment brief.

## 3. Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`geopandas` is used only for the optional AOI-extent plot in the EDA
(from `metadata.geojson`); if it's not installed, everything else in
the pipeline still runs and that one plot is skipped with a warning.
`tqdm` is used only for progress bars; the pipeline runs fine without
it too.

## 4. How to run

All scripts read their configuration from `configs/config.yaml` (paths,
class list, split ratios, model hyperparameters, etc.) — edit that file
rather than script arguments for most changes.

```bash
# 1. Discover patches and create the reproducible train/val/test split
#    (patch IDs saved to .../outputs/splits/{train,val,test}.txt)
python -m src.data_loading --config configs/config.yaml

# 2. Exploratory data analysis: dataset summary, class distribution,
#    example RGB/label figures, AOI extent (if metadata available)
python -m src.eda --config configs/config.yaml

# 3. Train the Random Forest or XGBoost baseline (saves model + normalization
#    stats + feature importances to .../outputs/models & .../outputs/metrics)
python -m src.train --config configs/config.yaml

# 4. Evaluate on validation and test splits (saves metrics JSON,
#    confusion matrices, per-patch predictions, qualitative figures)
python -m src.evaluate --config configs/config.yaml --splits val test
```

Or run everything interactively, with inline plots, from
`...notebooks/exploration_and_training.ipynb`.

### Reproducing the split

`.../outputs/splits/{train,val,test}.txt` (one patch ID per line) are the
artifact of record for the split — re-running `src.data_loading` with
the same `configs/config.yaml` (`split.seed`) regenerates the identical
split from any copy of the dataset. Downstream scripts (`train.py`,
`evaluate.py`) always read these files rather than re-deriving splits,
so the split is fixed once created.

## 5. Approach summary

- **Preprocessing**: for each patch, the raw `(T, 10, H, W)` cube is
  collapsed along the time axis into per-pixel `mean/std/min/max`
  statistics per band (`40` features per pixel), masking out
  all-zero ("no-data") acquisitions before aggregating. Features are
  then robustly normalized per-channel (1st/99th percentile clip +
  min-max scale), with normalization statistics **fit on the training
  set only** and reused unchanged for validation/test to avoid leakage.
- **Sampling**: during training, a configurable fraction of valid
  pixels per patch (`preprocessing.pixel_sample_fraction`, default
  15%) is randomly sampled rather than using every pixel — neighbouring
  pixels within a field are highly redundant, so sampling keeps
  training tractable without meaningfully hurting the model.
- **Splitting**: patch-level (not pixel-level) random split, to avoid
  spatial leakage between train and val/test.
- **Evaluation**: run on *every* valid pixel of every val/test patch
  (not just a sample), reporting overall accuracy, per-class
  precision/recall/F1, per-class and mean IoU, and a row-normalized
  confusion matrix.

## 6. Key assumptions

- **Void label (19) excluded from training and evaluation** pixel
  sets, since it does not correspond to a real class.
- **No-data / cloud screening is simplistic**: an observation is
  treated as invalid at a pixel only if all 10 bands read exactly 0 at
  that pixel/time-step. A production system would use an actual cloud
  probability layer if the raw PASTIS distribution provides one.
- **Patch-level split** assumes patches are drawn from a sufficiently
  large, spatially diverse AOI that random splitting is a reasonable
  proxy for generalization; if patches are known to be spatially
  clustered/adjacent, a spatial block split would be more rigorous
  (see `report.md` limitations).
- **`sampling.max_patches`** in the config lets you cap the number of
  patches used end-to-end during development/debugging on a laptop;
  set to `null` to use the full dataset for the final run.

## 7. Limitations / known issues

See `report.md` for the full discussion (class confusion
patterns, effects of class imbalance, resolution, seasonal timing,
etc.). Headline items:

- The temporal-statistics feature representation discards the
  *ordering* of observations, so it cannot capture phenological
  timing/shape directly (only summary statistics of it). A
  sequence-aware model (1D-CNN/LSTM over time, or a temporal U-Net)
  would likely improve classes whose spectral signature is defined by
  *when* a change happens (e.g. distinguishing winter vs. spring
  varieties of the same crop).
- Rare classes with very few labeled pixels are expected to have low
  recall regardless of model choice — more labeled data or class
  merging would help.
- This repo include a saved model file (see next section) —
  results in `report.md` and `outputs/` can be reproduced by running the
  pipeline against the actual PASTIS data placed under `data/PASTIS/`,
  which is not distributed with this repository.

## 8. Saved model / reproducing results

`outputs/models/model.joblib` and
`outputs/models/normalization_stats.joblib` are produced by
`src/train.py` and are the two artifacts `src/evaluate.py` needs.

## 9. Repository structure

```
project-root/
└── configs/
    └── config.yaml
├── notebooks/
│   └── exploration_and_training.ipynb
├── outputs/
│   ├── figures/               # EDA + qualitative prediction figures
│   ├── metrics/                # metrics_{val,test}.json, patch summaries, feature importances
│   ├── predictions/            # per-patch prediction .npy arrays
│   ├── models/                  # trained model + normalization stats
│   └── splits/                  # train.txt / val.txt / test.txt
├── src/
│   ├── data_loading.py      # patch discovery, loading, splitting
│   ├── preprocessing.py     # temporal aggregation, normalization, sampling
│   ├── eda.py                # Task B: exploratory data analysis script
│   ├── train.py              # Task D: Random Forest baseline training
│   ├── evaluate.py           # Task E/F: evaluation + qualitative visualisation
│   └── visualization.py     # plotting helpers used across the pipeline
├── README.md
├── report.md
├── requirements.txt
```
