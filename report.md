# Report: Crop Type Classification from Multi-Temporal Sentinel-2 Imagery

## 1. Approach

### 1.1 Data preparation

Each patch's Sentinel-2 cube can have a different number of observations
`T` (our case it is same), so pixel-level features are built by aggregating each of the 10
bands over the time axis into four temporal statistics — **mean,
std, min, max** — giving a fixed-length 40-dimensional feature vector
per pixel regardless of `T`. Observations that are all-zero across
every band at a pixel (a simple no-data / heavily-clouded proxy) are
excluded from the aggregation via a masked array, so a single bad
acquisition does not bias a pixel's statistics.

Rationale for statistics over raw time steps: it sidesteps the
variable-length-sequence problem entirely, keeps the feature space
small enough for a ML model to train quickly on CPU, and each
statistic has a direct physical interpretation — mean captures average
spectral signature, std captures how much a pixel's reflectance
changes through the season, min/max
bracket the extremes of the growth cycle (e.g. the NIR peak associated
with maximum canopy cover).

### 1.2 Why this model

A per-pixel **Random Forest** or **XGBoost** was chosen as the baseline over a deep
segmentation model for several reasons:

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
- **Robustness to the sampling strategy** used to keep training
  tractable (15% of valid pixels per patch by default) — trees are
  comparatively insensitive to i.i.d. sub-sampling of a large pixel
  pool compared to, say, a small deep net trained on the same budget.

The explicit trade-off: a per-pixel RF or XGB **ignores spatial context**
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

- **Dataset size**: 102 number of usable patches (paired S2 + annotation),
  from `src.data_loading.list_patch_ids`.
- **Patch dimensions & temporal depth**: Each patch has dimensions of 128 (`H`), 128 (`W`), with
  46 number of observations `T` (`outputs/metrics/patch_summary.csv`).
- **Class distribution**: `outputs/figures/class_distribution.png`
  shows a clear imbalance among the different land-cover classes in the PASTIS dataset. Background pixels constitute the largest proportion of the dataset (506141 pixels), followed by Meadow (296576), Soft Winter Wheat (199108), Corn (196704), and Void Label (144087). In contrast, several classes are highly underrepresented, particularly Orchard (75 pixels), and Potatoes (158). Notably, the Beet class has no pixels in the provided results, indicating that this class is absent from the analyzed data. 
- **Example imagery**: RGB composites are available in 
  `outputs/figures/Patch_*_RGB_median_composite*.png`, with the corresponding class labels shown in `Patch_*_labels.png`.
- **AOI extent**:  `outputs/figures/aoi_extent.png` shows the approximate geographic
  footprint of the patches — note the total extent and describe
  qualitatively (e.g. contiguous vs. scattered tiles).

## 3. Evaluation results

Both Random Forest and XGBoost models were evaluated on the same independent test dataset comprising 240341 pixels across 16 patches. XGBoost consistently outperformed Random Forest across the main evaluation metrics.

XGBoost achieved an overall accuracy of 70.22%, representing a 6.25% improvement over Random Forest. The mean IoU also increased from 18.88% to 22.69%, indicating improved spatial agreement between predicted and reference crop classes.

The largest improvements were observed for the major crop classes. XGBoost increased the IoU for Corn from 31.76% to 46.82%, Winter barley from 12.30% to 21.15%, Soft winter wheat from 41.30% to 49.43%, and Winter rapeseed from 64.77% to 77.94%. Performance for Soybeans also improved from 40.96% to 47.88%.

XGBoost achieved an F1-score of 0.88 for Winter rapeseed, compared with 0.79 for Random Forest, while improvements were also observed for Meadow, Soft winter wheat, Corn, Winter barley, and Soybeans.

Despite the overall improvement, both models struggled with minority classes. Winter triticale, Winter durum wheat, Fruits/vegetables/flowers, Potatoes, Leguminous fodder, Mixed cereal, and Sorghum remained unclassified (IoU = 0) by both models. Spring barley and Sunflower also showed very low IoUs in XGBoost.

Overall, XGBoost provided the better classification performance, particularly for the dominant crop classes, and is therefore the preferred model among the two evaluated approaches. However, the relatively low macro-level metrics indicate that class imbalance and limited representation of minority crop classes remain important challenges.

## 4. Analysis and interpretation

### Strengths and limitations of the approach

**Strengths**: fast to train/iterate, interpretable via feature
importances, handles the variable-`T` problem cleanly, reasonable
class-imbalance handling, CPU-only.

**Limitations**: no spatial context (each pixel classified
independently), no explicit modelling of
observation *timing* (two patches with the same mean/std/min/max but
different growth-stage timing are indistinguishable to this model),
and a simplistic no-data/cloud proxy (exact-zero masking) that will
miss partially-clouded pixels with non-zero reflectance.

### Effects of class imbalance, limited samples, and label quality

Rare classes (few labeled pixels dataset-wide) are expected to show
low recall — the model has too little signal to learn their spectral-
temporal signature confidently, and `class_weight="balanced_subsample"`
(in RF model) only partially compensates (it reweights the loss but cannot
manufacture missing examples). Where per-class IoU in
`outputs/metrics/metrics_test.json` is very low or `NaN` for a class,
check `outputs/metrics/class_pixel_counts.csv` first — it is often
simply a low-sample-size class rather than a fundamentally
inseparable one.

### Effects of cloud contamination, missing observations, seasonal timing

Missing or cloud-contaminated observations reduce the effective `T` used per pixel; patches with
few valid observations concentrated outside the crop's discriminative
growth window (e.g. missing the spring green-up) will have
noisier/less-informative temporal statistics than well-sampled
patches — this is a good axis to slice evaluation results by, if the
data supports it (e.g. compare accuracy for patches above/below median
`n_observations` from `.../outputs/metrics/patch_summary.csv`).

## 5. Recommended next steps

1. **Try a spatio-temporal deep model** (temporal U-Net / CNN+LSTM /
   lightweight transformer over the time axis) if GPU budget allows —
   expected to help most on classes and edge pixels where spatial
   context and temporal ordering matter, at the cost of needing more
   compute and careful handling of variable-length sequences (padding/
   masking) instead of the fixed-length temporal-statistics trick used
   here.
3. **Add vegetation indices** (NDVI, EVI, NDWI) as engineered features
   for the RF/XGB baseline — cheap to compute from existing bands and
   often more directly discriminative for phenology than raw
   reflectance statistics.
4. **Improve cloud/no-data masking** using an actual cloud-probability
   layer if the underlying PASTIS distribution provides one, rather
   than the exact-zero heuristic used here.
5. **Investigate a spatial (block) train/test split** in addition to
   the current random patch split, to more rigorously test
   generalization to geographically distant, unseen fields if patches
   turn out to be spatially clustered.
