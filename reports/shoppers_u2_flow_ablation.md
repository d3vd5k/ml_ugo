# Shoppers: U2 + Flow gate ablation metrics

## Protocol and provenance

Source: the completed native downweighting study, run [`20261012T120400-c73514825f95`](../results/runs/20261012T120400-c73514825f95/independent_report.all_24_metrics.csv). All values below were independently recomputed from **80 saved outer-test prediction files**, checking six predictive metrics per file and identical ordered row IDs/labels across all arms. No models were retrained for this table.

- Dataset: Online Shoppers Purchasing Intention. One shared outer holdout: **2,466 observations, including 382 minority positives**.
- Five matched sampler/classifier seeds, **0–4**. Values are **mean ± sample seed SD (ddof=1)**, not confidence intervals or independent-fold estimates.
- Positive/minority-class binary F1 is the primary presentation metric. Classifier decision threshold is **0.5** throughout. The historical component name `F1` means the **FlowGMM compatibility gate**, not F1-score optimization.
- These experiments selected canonical U2 learner/group settings and synthetic weights using **real-validation AP**, not F1. LR/RF/LightGBM inherit their recorded historical gate tau; native MLP selected its tau using inner unit-weight AP. This is a retrospective F1 evaluation, **not an F1-tuned Shoppers ablation**.
- The real-only control uses the matched locked learner settings. It is not the independently tuned real-only baseline from earlier studies. Generation/native scoring here uses current pure U2 without OOF; inherited LR/RF/LightGBM tau provenance is documented in the [source study](downweight.md#scope-and-verified-evidence).

## Available matched configurations

| Configuration | What is retained or removed |
|---|---|
| No oversampling | Entire augmentation branch removed; matched native learner trained on real observations only. |
| U2 (gate removed) | Real-only U2 grouping/allocation, native difficulty scoring and interpolation retained; native neighbor post-detection/relocation replaces the Flow gate. Synthetic weight 1. |
| U2 + Flow gate (weight 1) | U2 proposals filtered on original coordinates by the frozen compatibility gate; no native relocation and no rejected-quota refill. Synthetic weight 1. |
| U2 + Flow gate (AP-selected weight) | Same gated family with synthetic weight selected from 0.25/0.5 on real-validation AP; real rows retain weight 1. Native iterative trajectories are refitted at that weight. |

## Outer-test metrics

“Added rows” is the realized synthetic count, not the requested cap. Augmented arms share a **1,000-proposal/addition cap**, but gate rejection and native trajectories produce different realized counts. Counts are not artificially equalized.

| Classifier | Configuration | Minority F1 | Precision | Recall | AP | Balanced accuracy | MCC | Added rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LR | No oversampling | 0.4725 ± 0.0019 | 0.7063 ± 0.0029 | 0.3550 ± 0.0014 | 0.6044 ± 0.0020 | 0.6640 ± 0.0008 | 0.4428 ± 0.0023 | 0.0 ± 0.0 |
| LR | U2 (gate removed) | 0.5482 ± 0.0044 | 0.6245 ± 0.0019 | 0.4885 ± 0.0071 | 0.5854 ± 0.0023 | 0.7173 ± 0.0031 | 0.4819 ± 0.0040 | 1000.0 ± 0.0 |
| LR | U2 + Flow gate (weight 1) | 0.5272 ± 0.0037 | 0.6731 ± 0.0148 | 0.4335 ± 0.0084 | 0.5966 ± 0.0029 | 0.6974 ± 0.0029 | 0.4767 ± 0.0047 | 519.0 ± 65.5 |
| LR | U2 + Flow gate (AP-selected weight) | 0.4940 ± 0.0126 | 0.7098 ± 0.0014 | 0.3791 ± 0.0154 | 0.6016 ± 0.0020 | 0.6753 ± 0.0070 | 0.4605 ± 0.0096 | 517.8 ± 66.1 |
| RF | No oversampling | 0.6539 ± 0.0125 | 0.7121 ± 0.0105 | 0.6047 ± 0.0173 | 0.7434 ± 0.0024 | 0.7799 ± 0.0085 | 0.5993 ± 0.0135 | 0.0 ± 0.0 |
| RF | U2 (gate removed) | 0.6527 ± 0.0075 | 0.6367 ± 0.0081 | 0.6696 ± 0.0099 | 0.7178 ± 0.0052 | 0.7998 ± 0.0050 | 0.5874 ± 0.0088 | 1000.0 ± 0.0 |
| RF | U2 + Flow gate (weight 1) | 0.6558 ± 0.0042 | 0.6614 ± 0.0066 | 0.6503 ± 0.0071 | 0.7197 ± 0.0052 | 0.7946 ± 0.0031 | 0.5933 ± 0.0050 | 644.8 ± 113.4 |
| RF | U2 + Flow gate (AP-selected weight) | 0.6565 ± 0.0030 | 0.6681 ± 0.0077 | 0.6455 ± 0.0124 | 0.7291 ± 0.0067 | 0.7934 ± 0.0046 | 0.5950 ± 0.0022 | 644.2 ± 114.4 |
| LightGBM | No oversampling | 0.6473 ± 0.0033 | 0.7040 ± 0.0062 | 0.5990 ± 0.0012 | 0.7490 ± 0.0006 | 0.7764 ± 0.0012 | 0.5913 ± 0.0043 | 0.0 ± 0.0 |
| LightGBM | U2 (gate removed) | 0.6577 ± 0.0080 | 0.6279 ± 0.0096 | 0.6906 ± 0.0116 | 0.7281 ± 0.0054 | 0.8078 ± 0.0056 | 0.5924 ± 0.0096 | 1000.0 ± 0.0 |
| LightGBM | U2 + Flow gate (weight 1) | 0.6590 ± 0.0021 | 0.6461 ± 0.0115 | 0.6728 ± 0.0137 | 0.7306 ± 0.0021 | 0.8026 ± 0.0046 | 0.5952 ± 0.0022 | 690.4 ± 115.2 |
| LightGBM | U2 + Flow gate (AP-selected weight) | 0.6608 ± 0.0030 | 0.6735 ± 0.0063 | 0.6487 ± 0.0082 | 0.7415 ± 0.0019 | 0.7955 ± 0.0032 | 0.6002 ± 0.0033 | 690.0 ± 114.0 |
| MLP | No oversampling | 0.6113 ± 0.0038 | 0.6690 ± 0.0049 | 0.5628 ± 0.0081 | 0.6981 ± 0.0038 | 0.7559 ± 0.0033 | 0.5500 ± 0.0035 | 0.0 ± 0.0 |
| MLP | U2 (gate removed) | 0.6279 ± 0.0067 | 0.5688 ± 0.0259 | 0.7047 ± 0.0484 | 0.6792 ± 0.0064 | 0.8029 ± 0.0161 | 0.5567 ± 0.0077 | 1000.0 ± 0.0 |
| MLP | U2 + Flow gate (weight 1) | 0.6284 ± 0.0090 | 0.6340 ± 0.0074 | 0.6230 ± 0.0136 | 0.6882 ± 0.0123 | 0.7786 ± 0.0066 | 0.5611 ± 0.0102 | 550.4 ± 54.5 |
| MLP | U2 + Flow gate (AP-selected weight) | 0.6233 ± 0.0107 | 0.6480 ± 0.0053 | 0.6005 ± 0.0156 | 0.6921 ± 0.0097 | 0.7704 ± 0.0076 | 0.5579 ± 0.0116 | 550.6 ± 53.9 |

Selected gated-weight counts across five seeds: LR, 0.25 in four and 0.5 in one; RF, 0.25 in one and 0.5 in four; LightGBM, 0.25 in two and 0.5 in three; MLP, 0.25 in one and 0.5 in four. The selected-weight row is not a fixed-weight-0.25 or fixed-weight-0.5 estimate.

## Gate ablation: paired changes

Each delta is **U2 + Flow gate (weight 1) minus U2 (weight 1)**, paired by classifier and seed before computing mean/SD. A positive delta favors F1/precision/recall/AP; no independent-fold significance claim is made.

| Classifier | Paired ΔF1 | F1-positive seeds | Paired Δprecision | Paired Δrecall | Paired ΔAP |
|---|---:|---:|---:|---:|---:|
| LR | -0.0210 ± 0.0069 | 0/5 | +0.0486 ± 0.0147 | -0.0550 ± 0.0110 | +0.0112 ± 0.0028 |
| RF | +0.0030 ± 0.0071 | 3/5 | +0.0247 ± 0.0081 | -0.0194 ± 0.0129 | +0.0019 ± 0.0038 |
| LightGBM | +0.0013 ± 0.0074 | 4/5 | +0.0182 ± 0.0106 | -0.0178 ± 0.0141 | +0.0025 ± 0.0038 |
| MLP | +0.0005 ± 0.0107 | 2/5 | +0.0652 ± 0.0297 | -0.0817 ± 0.0424 | +0.0090 ± 0.0093 |

## Interpretation and limits

- The gate raises mean precision and AP but lowers mean recall in every learner. Mean F1 decreases for LR and increases only modestly for RF, LightGBM and MLP. It is not a universal F1 improvement.
- AP-selected synthetic downweighting modestly improves tree F1 relative to unit-weight gating, but reduces LR/MLP F1. It must not be described as F1-optimal weighting.
- Gate removal changes the post-handler, realized synthetic dose and native trajectory. These are matched-seed **recipe ablations**, not a dose-matched proof that density filtering improves point quality.
- This table does not invent Flow-only, no-grouping, no-allocation, F1-threshold-tuning, XGBoost or SVM measurements. Those contrasts are not supplied by this four-learner study. Earlier original-UGO and neighbor-rejection experiments remain in [the separate gate report](f1.md); their selected configurations differ and are not silently mixed into this table.
- All results reuse one development holdout. Neither the small positive F1 means nor seed SD establish population-level superiority.

## Source files

- [Per-seed all-24 metrics](../results/runs/20261012T120400-c73514825f95/independent_report.all_24_metrics.csv)
- [Audited metric summary](../results/runs/20261012T120400-c73514825f95/independent_report.all_metrics_summary.csv)
- [Recorded per-seed weight decisions](../results/runs/20261012T120400-c73514825f95/independent_report.per_seed_decisions.csv)
- [Study protocol, inherited tau provenance and full results](downweight.md)

Verification: six predictive metrics recomputed from every saved test probability file for all 16 classifier/configuration groups × five seeds, including restoration of native float32 MLP probability precision. Test identities and labels were identical across all 80 files; threshold 0.5, class counts, five-seed coverage and reported per-seed metric agreement were checked. No new training or test-based recipe selection was performed.
