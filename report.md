# Report: Crop Type Classification from Multi-Temporal Sentinel-2 Imagery

> **A note on this report's status.** This repository was built without
> access to the actual PASTIS `.npy` data files — only the assignment
> brief was provided. Every script in `src/` has been written, unit-
> tested for correctness, and smoke-tested end-to-end against
> synthetic arrays of the correct shapes (confirming the full pipeline
> runs without errors), but the **quantitative results below are
> placeholders**: sections marked `[[FILL AFTER RUNNING ON REAL DATA]]`
> should be completed by running
> `python -m src.data_loading && python -m src.eda && python -m src.train && python -m src.evaluate`
> against the real dataset placed under `data/PASTIS/` (see
> `README.md`), which will populate `outputs/metrics/*.json` and
> `outputs/figures/*.png` with real numbers and images to drop in here.
> The reasoning, methodology, and structure of the analysis below is
> the deliverable; the numbers are not fabricated.

## 1. Approach

### 1.1 Data preparation

Each patch's Sentinel-2 cube has a different number of observations
`T`, so pixel-level features are built by aggregating each of the 10
bands over the time axis into four temporal statistics — **mean,
std, min, max** — giving a fixed-length 40-dimensional feature vector
per pixel regardless of `T`. Observations that are all-zero across
every band at a pixel (a simple no-data / heavily-clouded proxy) are
excluded from the aggregation via a masked array, so a single bad
acquisition does not bias a pixel's statistics.

Rationale for statistics over raw time steps: it sidesteps the
variable-length-sequence problem entirely, keeps the feature space
small enough for a Random Forest to train quickly on CPU, and each
statistic has a direct physical interpretation — mean captures average
spectral signature, std captures how much a pixel's reflectance
changes through the season (bare soil/meadow stays flatter than a
crop moving through green-up, peak vigor, and senescence), min/max
bracket the extremes of the growth cycle (e.g. the NIR peak associated
with maximum canopy cover).

### 1.2 Why this model

A per-pixel **Random Forest** was chosen as the baseline over a deep
segmentation model for several reasons, in line with the assignment's
stated preference for "a simpler baseline model with strong reasoning
... over an unnecessarily complex model":

- **No GPU dependency.** Trains in minutes on CPU on a modest number of
  sampled pixels, which matters if compute is constrained.
- **Strong prior track record** for exactly this problem class: RF/
  gradient-boosted trees on temporal spectral statistics are a
  standard, competitive baseline in the crop-classification literature
  and in the original PASTIS-adjacent benchmarks, before reaching for
  spatio-temporal deep models.
- **Interpretability.** `clf.feature_importances_` (saved to
  `outputs/metrics/feature_importances.json`) directly shows which
  band/statistic combinations drive class separability — useful both
  for sanity-checking the model and for guiding future feature
  engineering (e.g. adding vegetation indices).
- **Class imbalance handling** via `class_weight="balanced_subsample"`,
  without needing a custom loss function.
- **Robustness to the sampling strategy** used to keep training
  tractable (15% of valid pixels per patch by default) — trees are
  comparatively insensitive to i.i.d. sub-sampling of a large pixel
  pool compared to, say, a small deep net trained on the same budget.

The explicit trade-off: a per-pixel RF **ignores spatial context**
(it treats each pixel independently, so it cannot use field
boundaries, shape, or neighbourhood texture) and **ignores the
temporal ordering** of observations (only summary statistics of the
time series are used, not the sequence itself). A CNN-based temporal
segmentation model (e.g. a lightweight temporal U-Net, or the
attention-based architectures used in the original PASTIS paper) would
likely outperform this baseline given enough labeled data and GPU
budget, at higher implementation and training cost. This is discussed
further as a recommended next step in Section 4.

### 1.3 Split strategy

The split is **patch-level**, not pixel-level: patches are randomly
assigned wholesale to train (70%) / val (15%) / test (15%) using a
fixed seed (`configs/config.yaml` -> `split.seed`), and the resulting
patch-ID lists are persisted to `outputs/splits/{train,val,test}.txt`
so the split is exactly reproducible from those files alone. Splitting
by pixel instead would leak information: neighbouring pixels within
the same field are highly spatially autocorrelated, so a pixel-level
split would let the model implicitly "see" a test field's neighbours
during training and overstate generalization performance.

## 2. Exploratory data analysis

`[[FILL AFTER RUNNING ON REAL DATA]]` — populate from
`outputs/metrics/patch_summary.csv`, `outputs/metrics/class_pixel_counts.csv`,
and the figures below. Suggested structure once real data is
available:

- **Dataset size**: number of usable patches (paired S2 + annotation),
  from `src.data_loading.list_patch_ids`.
- **Patch dimensions & temporal depth**: distribution of `H`, `W`, and
  number of observations `T` per patch (`outputs/metrics/patch_summary.csv`).
  Note whether `T` varies a lot across patches (it does, in PASTIS) and
  what that implies for any modelling approach that assumes fixed-length
  sequences.
- **Class distribution**: insert `outputs/figures/class_distribution.png`
  here and describe the imbalance quantitatively (e.g. ratio of the
  most to least common class). PASTIS-style agricultural scenes are
  typically dominated by a handful of major crop/land-cover classes
  (background, meadow, and one or two cereal types), with several
  specialty crops represented by very few pixels — this shapes both
  the modelling choices above (class-balanced RF) and the expected
  per-class evaluation results in Section 3.
- **Example imagery**: insert 2-4 `outputs/figures/rgb_example_*.png`
  and matching `label_example_*.png` pairs, with a short caption on
  what field patterns / crop types are visible.
- **AOI extent**: if `metadata.geojson` was available,
  `outputs/figures/aoi_extent.png` shows the approximate geographic
  footprint of the patches — note the total extent and describe
  qualitatively (e.g. contiguous vs. scattered tiles).

## 3. Evaluation results

`[[FILL AFTER RUNNING ON REAL DATA]]` — populate from
`outputs/metrics/metrics_val.json` and `outputs/metrics/metrics_test.json`.

| Metric | Validation | Test |
|---|---|---|
| Overall pixel accuracy | `[[value]]` | `[[value]]` |
| Mean IoU | `[[value]]` | `[[value]]` |

Per-class precision / recall / F1 (from `classification_report` in the
metrics JSON) and the confusion matrices
(`outputs/figures/confusion_matrix_{val,test}.png`) should be inserted
and discussed here — in particular:

- Which classes achieve the highest F1 (typically the dominant,
  spectrally-distinctive classes with abundant training pixels)?
- Which classes are weakest, and does that track with their pixel
  count in the EDA class-distribution plot?
- What does the confusion matrix reveal about *which* classes get
  confused with each other — e.g. crops in the same botanical family
  or with overlapping growth calendars (winter wheat vs. winter
  barley vs. winter durum wheat; the different fodder/legume classes)
  are the most likely confusion pairs given they share similar
  spectral-temporal trajectories.

## 4. Analysis and interpretation

### Strengths and limitations of the approach

**Strengths**: fast to train/iterate, interpretable via feature
importances, handles the variable-`T` problem cleanly, reasonable
class-imbalance handling, CPU-only.

**Limitations**: no spatial context (each pixel classified
independently — expect "salt and pepper" misclassified pixels within
otherwise-uniform fields, visible in the qualitative
`outputs/figures/qualitative_*.png` panels), no explicit modelling of
observation *timing* (two patches with the same mean/std/min/max but
different growth-stage timing are indistinguishable to this model),
and a simplistic no-data/cloud proxy (exact-zero masking) that will
miss partially-clouded pixels with non-zero reflectance.

### Effects of class imbalance, limited samples, and label quality

Rare classes (few labeled pixels dataset-wide) are expected to show
low recall — the model has too little signal to learn their spectral-
temporal signature confidently, and `class_weight="balanced_subsample"`
only partially compensates (it reweights the loss but cannot
manufacture missing examples). Where per-class IoU in
`outputs/metrics/metrics_test.json` is very low or `NaN` for a class,
check `outputs/metrics/class_pixel_counts.csv` first — it is often
simply a low-sample-size class rather than a fundamentally
inseparable one.

### Effects of resolution, cloud contamination, missing observations, seasonal timing

Sentinel-2's 10-20m resolution means small or irregularly-shaped
parcels (e.g. some fruit/vegetable/flower plots, orchards) will have a
high proportion of mixed/edge pixels whose spectral signature blends
two land-cover types — a likely source of confusion for exactly the
minor specialty-crop classes. Missing or cloud-contaminated
observations reduce the effective `T` used per pixel; patches with
few valid observations concentrated outside the crop's discriminative
growth window (e.g. missing the spring green-up) will have
noisier/less-informative temporal statistics than well-sampled
patches — this is a good axis to slice evaluation results by, if the
data supports it (e.g. compare accuracy for patches above/below median
`n_observations` from `outputs/metrics/patch_summary.csv`).

### AOI context

`[[FILL AFTER RUNNING ON REAL DATA]]` — PASTIS patches are drawn from
metropolitan France; if the AOI extent plot / patch metadata confirms
a specific region, note the typical crop calendar for that region
(e.g. winter cereals sown autumn/harvested summer vs. spring-sown
crops like corn, sunflower, soybean) since it directly explains *why*
certain classes are spectrally similar at certain times of year (e.g.
all winter cereals look similar in early spring before their growth
rates diverge).

## 5. Recommended next steps

1. **Run the pipeline against real data** to replace the placeholders
   above with actual metrics and figures — this is the immediate next
   step, everything else here is written to be ready for it.
2. **Try a spatio-temporal deep model** (temporal U-Net / CNN+LSTM /
   lightweight transformer over the time axis) if GPU budget allows —
   expected to help most on classes and edge pixels where spatial
   context and temporal ordering matter, at the cost of needing more
   compute and careful handling of variable-length sequences (padding/
   masking) instead of the fixed-length temporal-statistics trick used
   here.
3. **Add vegetation indices** (NDVI, EVI, NDWI) as engineered features
   for the RF baseline — cheap to compute from existing bands and
   often more directly discriminative for phenology than raw
   reflectance statistics.
4. **Improve cloud/no-data masking** using an actual cloud-probability
   layer if the underlying PASTIS distribution provides one, rather
   than the exact-zero heuristic used here.
5. **Investigate a spatial (block) train/test split** in addition to
   the current random patch split, to more rigorously test
   generalization to geographically distant, unseen fields if patches
   turn out to be spatially clustered.
6. **Post-process predictions** with a majority filter or a
   field-boundary-aware smoothing step to remove the pixel-level
   "salt and pepper" noise inherent to a per-pixel classifier.
