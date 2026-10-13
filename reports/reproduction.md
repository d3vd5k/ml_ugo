# Stage 1 acquisition and reproduction audit

Audit date: 2026-10-03. This setup does **not** reproduce any published score. The 21 entries in `results/published_targets.csv` are reported targets, not local results. No full CLIMB, BAF training, APS training, or comparative experiment was performed by the protocol setup.

## Controlled execution contracts

`load_protocol(path)` validates schema, objectives, threshold rules, budgets, declared methods and split contracts. `load_dataset(protocol, data_home, local_data=None, download=False)` returns raw named pandas features, contiguous int64 labels, unique nonmissing source-row identities and JSON-serializable metadata. `make_splits(dataset, protocol, seed)` returns shared outer and fit/validation partitions. Inner transforms fit only on fit_indices; after selection freezes, a separate final transform/model refits on full outer training without test feedback.

| Dataset | Controlled protocol | Outer test | Inner validation | Primary objective |
|---|---|---|---|---|
| Shoppers | `shoppers_uci_controlled_v1` | Natural-prevalence stratified 20% holdout, fixed outer seed 2026 | Stratified 20% of outer training; per-run seed | Average precision; fixed 0.5 classification threshold |
| APS | `aps_uci_official_controlled_v1` | Untouched official 16,000 rows | Stratified 20% of official 60,000 training rows | Minimize `10 FP + 500 FN`; validation-only threshold |
| BAF Base | `baf_base_temporal_controlled_v1` | Months 6 and 7 | Latest training month 5; fit months 0–4 | Recall at FPR 0.05; validation-selected operating threshold |
| Satimage, optional | `satimage_uci_official_controlled_v1` | Official `sat.tst`, 2,000 rows | Stratified inner holdout from `sat.trn`, 4,435 rows | Macro F1; argmax multiclass predictions |

Shoppers categories are categorical, not continuous integer codes. APS `na` and BAF's documented `-1` missing sentinels normalize to NaN without fitting statistics. Resampling, imputation and scaling remain inside actual training partitions. No identifiable entities are released: source-row positions identify rows, not people or trucks. Satimage's spatial groups are unavailable; the official split is retained rather than inventing spatial grouping.

Shoppers' full-table raw fingerprint and CLIMB OpenML 45560 represent potentially the same evidence, not independent datasets. Row-level alignment to OpenML is not yet audited. Six-class Satimage must not be conflated with CLIMB's binary Satellite 40900; raw-source/spatial overlap is unresolved. APS and BAF are absent by source/name from the current 73-row catalog; that does not establish absence of unknown private-source overlap.

Controlled methods are baseline, class weighting, ROS, SMOTE, BorderlineSMOTE, KMeansSMOTE, original UGO and corrected UGO. The intrinsically balanced random forest classifier is restricted to the baseline method. Augmentation defaults to a declared matched total of 1,000, not million-row balancing. Smoke-only KMeansSMOTE parameters (`cluster_balance_threshold=0.05`, three clusters) are declared separately in the Shoppers config and are not a full-study tuning result. Exact equality of achieved augmentation counts remains a runner/reporting requirement, not an assumption.

## Acquisition and observed setup smoke

Public UCI Shoppers and Satimage ZIPs and the pinned SOS author arrays were downloaded and parsed locally. APS's generated ZIP stream was incomplete on two attempts; the official UCI individual CSV endpoints succeeded. Its source SHA256 pins are the original CSV bytes, **not** a fabricated archive checksum. The failed partial archive was removed. Credentials and Kaggle authentication files were not accessed.

Executed successfully (zero model fits):

```sh
python -m benchmarks.protocols --data-home .cache/benchmarks --output results/smoke/protocol_acquisition.json
```

The first acquisition smoke exposed string-vs-integer manifest count keys; the count comparison was corrected, and the command then succeeded. The saved acquisition artifact records source checksums, raw fingerprints, missingness and SHA256 hashes of all split-index arrays. Later schema/type and dataset-name guards do not change the declared split construction. Parent integration verification is separate.

| Dataset | Fit rows (negative / positive) | Validation | Outer test | Outer train |
|---|---|---|---|---|
| Shoppers | 7,891 (6,670 / 1,221) | 1,973 (1,668 / 305) | 2,466 (2,084 / 382) | 9,864 (8,338 / 1,526) |
| APS | 48,000 (47,200 / 800) | 12,000 (11,800 / 200) | 16,000 (15,625 / 375) | 60,000 (59,000 / 1,000) |

These are partition-contract checks, not performance estimates. For a fresh cache, add `--download`; public bytes are checksum-checked before use. The default cache location also recognizes acquired files under `.cache/benchmarks/audit`. The acquisition CLI intentionally fits no models, and reading official test metadata here does not select a classifier or threshold.

The raw APS CSV has **170 predictors plus `class`**, not 171 predictors. Missing predictor cells: 850,015 train and 228,680 test. The current UCI API inaccurately says no missing values and reports 171 features; parsed source schema/counts take precedence. Shoppers has **17 predictors plus Revenue**, not 18 predictors, and zero missing cells. Satimage has 36 predictors, no class 6, 6,435 total rows and zero missing cells.

The final integrated setup verification passed **75 regression tests**, including source checksums, official-test isolation, temporal validation, category-safe sampling, weighted loss, original descendant dynamics, validation-only thresholds, result identities and smoke exclusion. Shoppers CLI checks completed 25 applicable LR/RF/LightGBM/BRF cells (7 explicitly inapplicable ensemble combinations) plus 8 MLP/XGBoost CUDA-path cells. All smoke rows came from original outer training; the 2,466 original holdout rows remained unused. No performance improvement or published reproduction is claimed.

Persisted setup evidence:

- Final five-seed Shoppers no-fit preparation: results/runs/20261003T174417-f6aed7b7dbe5; 125 prepared and 35 explicitly inapplicable cells, zero model fits. Verified source snapshot contains all 21 published targets and 44 recoverable source/config/document hashes.
- Training-only control matrix: results/smoke/20261003T173337-9830e8ef546b.
- MLP/XGBoost supported CUDA paths: results/smoke/20261003T173508-50da8b0f4cc1.
- APS and Satimage no-fit preparation completed in temporary output roots; BAF missing-source preparation correctly refused and recorded its external prerequisite.
- Existing sklearn/SciPy iprint, LightGBM feature-name and XGBoost CUDA CPU-input prediction warnings were observed, not suppressed.

## Full Shoppers baseline (controlled development)

Completed run: `results/runs/20261003T180121-ee6dd58a68ac` under `shoppers_uci_controlled_v1` and `stage1_baseline_controls_v1`. This is a full-data baseline, not a smoke run or published-protocol reproduction.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -B -m benchmarks.run \
  --run --protocol configs/benchmarks/shoppers_controlled.yaml \
  --study configs/ablations/stage1_controls.yaml --methods baseline
```

All **12,330 rows** were accounted for: **9,864 outer-training rows** and **2,466 fixed holdout rows**, including 382 positive holdout examples. Each seed used 7,891 inner-fit and 1,973 inner-validation rows; preprocessing was training-fitted, candidate selection used inner-validation average precision, and selected models were refitted on all outer-training rows. The classification threshold remained 0.5.

All **20 cells completed**: four classifiers, seeds 0–4, ten candidate proposals per cell (200 inner fits and 20 final refits). LR had six distinct candidate dictionaries with repeated proposals; the other classifiers had ten. No external resampling or class weighting was applied, and actual generated counts were zero. Balanced RF intrinsically balances its training samples and is therefore an imbalance-aware reference, not an unbalanced no-resampling learner. Recorded run time was **172.89 seconds**.

Values below are means ± sample SD across five seeds; AP is the primary metric.

| Classifier | Average precision | Minority F1 | Minority recall | Balanced accuracy |
|---|---|---|---|---|
| Logistic regression | 0.6089 ± 0.0028 | 0.4567 ± 0.0157 | 0.3340 ± 0.0201 | 0.6553 ± 0.0085 |
| Random forest | 0.7395 ± 0.0057 | 0.6100 ± 0.0254 | 0.5136 ± 0.0569 | 0.7415 ± 0.0232 |
| LightGBM | 0.7428 ± 0.0039 | 0.6347 ± 0.0177 | 0.5670 ± 0.0363 | 0.7635 ± 0.0156 |
| Balanced random forest | 0.7396 ± 0.0045 | 0.6297 ± 0.0089 | 0.8325 ± 0.0041 | 0.8418 ± 0.0037 |

The seeds share one fixed outer holdout; their SD is descriptive training/selection variability, **not a confidence interval or independent test-sampling uncertainty**. LightGBM has the highest observed mean AP here; this does not establish a statistically significant advantage or published superiority. The holdout has now been evaluated and remains development evidence, not fresh confirmatory evidence. No UGO or oversampling improvement was evaluated by this baseline run.

Saved reference summaries are `baseline_summary.json`, `baseline_summary.csv` and `ap_summary.json` inside the run directory. Canonical cell evidence remains `benchmark_records.json`, the completed manifest/source snapshot, candidate/selection/preprocessing audits, row-level splits and predictions. Verification covered all source-row identities, shared holdout identities, validation-only candidate selection, zero generated counts, prediction labels, all reported metrics recomputed from saved predictions, and matching canonical/linked registry cells. Use `pd.read_csv(path, float_precision="round_trip")` when recomputing metrics: default approximate float parsing can alter near-tied probability ranks and therefore AP. Existing sklearn/SciPy and LightGBM warnings were observed rather than suppressed.

## Full Shoppers UGO baselines (controlled development)

Completed run: `results/runs/20261003T181637-fabbd21ddfb8`; paired no-resampling reference: `results/runs/20261003T180121-ee6dd58a68ac`. Both runs have identical protocol/study/evaluation-policy hashes, raw dataset fingerprints, real inner/outer splits, candidate proposals, tuning budgets and fitted preprocessing states. This uses the same full **12,330-row** dataset, **9,864 training / 2,466 holdout** partition, seeds **0–4**, **ten tuning proposals per cell**, validation-only AP selection and fixed **0.5** threshold.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -B -m benchmarks.run \
  --run --protocol configs/benchmarks/shoppers_controlled.yaml \
  --study configs/ablations/stage1_controls.yaml \
  --methods ugo_original ugo_corrected
```

All **30 supported cells completed**: original/corrected UGO × LR/RF/LightGBM × five seeds. Ten balanced-RF combinations are explicitly unsupported because that intrinsic ensemble remains baseline-only; no failed fit is hidden. Recorded run time: **1,446.87 seconds** (24.11 minutes). The operational iteration cap was 20, with the existing lambda_select=0.5, lambda_step=0.1, phi=2.5 and alpha=0.05 defaults.

The declared matched augmentation cap was **1,000 minority rows per sampler run**. Every one of the 300 inner candidate and 30 final outer sampler runs actually reached it: inner class counts became **6,670 / 2,221**, outer counts **8,338 / 2,526**. Final original UGO stopped at generation_cap; corrected UGO recorded target_counts_reached. All 15 corrected-versus-original seed/classifier pairs therefore qualify as actual-count-matched comparisons; the no-resampling reference is a non-generation control, not a matched generation mechanism. These are sampler-run counts, not total classifier fits: UGO retrains iteratively.

AP and paired AP delta are means ± sample SD across five seeds; other columns are means. Deltas are UGO minus the same-seed no-resampling classifier, on the 0–1 scale.

| Classifier | UGO variant | Average precision | Paired AP delta | Minority F1 | Minority recall | Balanced accuracy |
|---|---|---|---|---|---|---|
| Logistic regression | Original | 0.6077 ± 0.0016 | -0.0012 ± 0.0031 | 0.5591 | 0.4712 | 0.7160 |
| Logistic regression | Corrected | 0.6074 ± 0.0019 | -0.0015 ± 0.0031 | 0.5574 | 0.4670 | 0.7144 |
| Random forest | Original | 0.7244 ± 0.0052 | -0.0151 ± 0.0107 | 0.6456 | 0.6157 | 0.7812 |
| Random forest | Corrected | 0.7220 ± 0.0039 | -0.0175 ± 0.0087 | 0.6383 | 0.6042 | 0.7757 |
| LightGBM | Original | 0.7302 ± 0.0033 | -0.0126 ± 0.0060 | 0.6485 | 0.6346 | 0.7877 |
| LightGBM | Corrected | 0.7262 ± 0.0086 | -0.0166 ± 0.0088 | 0.6477 | 0.6330 | 0.7870 |

**Observed tradeoff:** both UGO variants increased mean minority F1/recall and balanced accuracy at threshold 0.5, but reduced mean AP for every classifier family. The primary objective therefore shows no mean improvement over the established no-resampling references. These shared-holdout seed summaries are descriptive, not confidence intervals or significance tests; no published-score reproduction or Stage 2 freeze is claimed.

Original UGO is the separately identified paper-derived UGOSMOTE reference: descendants enter scoring/support, paper ambiguities are explicitly resolved, and categorical partner projection plus matched caps are controlled-track deviations. It is not verified author-code reproduction. Corrected UGO retains real-only scoring/support and ancestry; it is not the unimplemented combined U5 variant.

Saved summaries: `ugo_summary.json` and `ugo_summary.csv`; per-seed paired AP differences: `paired_vs_baseline.json` / `.csv`; count-matched mechanism differences: `paired_corrected_vs_original.json`; all candidate generation counts: `inner_generation_counts.csv`; canonical AP aggregation: `ap_summary.json`. Original candidate/selection/preprocessing audits, predictions, split identities, generation rounds, ancestry and source snapshots remain in the run directory. Verification covered all 30 prediction metric sets using round-trip float parsing, 330 generation audits including parent/class/round bounds and corrected real-only ancestry, exact baseline split/preprocessing/candidate equality, validation-only selection, and all 40 registry/index statuses.

```sh
conda run -n ml_project python -m benchmarks.summarize results/benchmark_runs.csv \
  --run-id 20261003T180121-ee6dd58a68ac \
  --run-id 20261003T181637-fabbd21ddfb8 --reference baseline
```

## Phase 1 Shoppers controls — user review pending

The remaining full-data controls were executed on the established fixed holdout with LR/RF/LightGBM, seeds 0–4, ten tuning proposals, validation-only AP selection and threshold 0.5. All 75 unique requested cells are accounted for: 60/60 class-weight/ROS/SMOTE/BorderlineSMOTE cells completed; KMeansSMOTE completed six and failed nine. No parameter relaxation, fallback sampler or retention decision was applied.

Complete per-cell metrics and resources: `results/runs/20261004T062555-1727fffcde17/all_metrics.csv`. Every available scalar metric has mean/sample-SD summaries in `controls_summary.csv` and `controls_summary.json` in that directory, with contributing run IDs, failures, generated counts, underspend and recovery provenance. All 66 prediction metric sets were independently recomputed using round-trip probability parsing; all 75 cells are unique and protocol/split/tuning compatible with the established no-resampling reference.

The table contains completed-seed means. KMeans rows describe **only two successful seeds**, not a five-seed performance estimate; three failures per classifier remain part of the outcome.

| Classifier | Control | Completed / requested | AP | Minority F1 | Minority recall | Balanced accuracy |
|---|---|---|---|---|---|---|
| LR | Class weight | 5 / 5 | 0.6164 | 0.5868 | 0.7461 | 0.8000 |
| LR | ROS | 5 / 5 | 0.6116 | 0.5469 | 0.4508 | 0.7073 |
| LR | SMOTE | 5 / 5 | 0.5983 | 0.5406 | 0.4450 | 0.7041 |
| LR | BorderlineSMOTE | 5 / 5 | 0.6046 | 0.5667 | 0.4874 | 0.7224 |
| LR | KMeansSMOTE | 2 / 5 | 0.6186 | 0.4873 | 0.3665 | 0.6707 |
| RF | Class weight | 5 / 5 | 0.7365 | 0.6411 | 0.6681 | 0.7969 |
| RF | ROS | 5 / 5 | 0.7386 | 0.6493 | 0.6131 | 0.7814 |
| RF | SMOTE | 5 / 5 | 0.7274 | 0.6560 | 0.6565 | 0.7967 |
| RF | BorderlineSMOTE | 5 / 5 | 0.7286 | 0.6606 | 0.6681 | 0.8016 |
| RF | KMeansSMOTE | 2 / 5 | 0.7260 | 0.6151 | 0.5262 | 0.7462 |
| LightGBM | Class weight | 5 / 5 | 0.7375 | 0.6301 | 0.8403 | 0.8443 |
| LightGBM | ROS | 5 / 5 | 0.7428 | 0.6683 | 0.6880 | 0.8100 |
| LightGBM | SMOTE | 5 / 5 | 0.7403 | 0.6578 | 0.6576 | 0.7975 |
| LightGBM | BorderlineSMOTE | 5 / 5 | 0.7406 | 0.6728 | 0.7052 | 0.8168 |
| LightGBM | KMeansSMOTE | 2 / 5 | 0.7377 | 0.6423 | 0.5982 | 0.7749 |

KMeansSMOTE failed at seeds 0, 1 and 4 for every classifier with: `RuntimeError: No clusters found with sufficient samples of class 1. Try lowering the cluster_balance_threshold or increasing the number of clusters.` Seeds 2 and 3 succeeded. Metrics for failed cells remain unavailable; successful generating outer refits added 1,000 minority rows, while class weighting added zero.

Infrastructure caveat: the first control invocation was terminated by a 300-second command deadline. Its 30 completed final prediction sets were preserved and their metrics reconstructed, while original runtime/peak-memory measurements were lost and remain explicitly unavailable. The original interrupted manifest/source snapshot is retained, with interruption recorded separately in `controls_summary.json`; it must not be interpreted as a completed canonical invocation. Remaining final evaluations completed in separately linked runs. The unfinished RF/SMOTE seed-2 cell repeated its ten tuning fits during recovery, but no completed final evaluation was repeated or double-counted. This extra compute is a documented deviation, not hidden inside the nominal ten-proposal budget.

Current development uses Shoppers only. The user decides whether each U1–U5 change is retained; CLIMB and other named benchmark evaluation remain deferred until Phase 1 configuration approval.

## Phase 1 U1 OOF MMA — user-retained

Completed run: `results/runs/20261004T063713-7b42103e131e`. All **45 cells** completed: no-resampling baseline, corrected_current and U1_oof_mma × LR/RF/LightGBM × seeds 0–4. The same 12,330 Shoppers rows, fixed 9,864/2,466 split, ten tuning proposals, AP objective and 0.5 threshold were retained. Fresh references share the candidate run policy and reproduce the historical AP/F1/recall/macro-F1/weighted-F1/balanced-accuracy/MCC scores.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -B -m benchmarks.run \
  --run --protocol configs/benchmarks/shoppers_controlled.yaml \
  --study configs/ablations/stage1_components.yaml \
  --classifiers logistic_regression random_forest lightgbm \
  --methods baseline corrected_current U1_oof_mma
```

**Change under evaluation:** replace corrected-current in-sample seed difficulty with fixed, real-only three-fold out-of-fold MMA. Each scoring model uses the same candidate classifier parameters and fits preprocessing only on its real fold-training rows. Scoring probabilities are computed before generation and reused across rounds; neither scored points nor any synthetic descendants enter the scoring models. The downstream generator/classifier still retrains on augmentation, and final refitting rebuilds cross-fitting on the full outer-training partition. Support, proposal, acceptance, allocation caps and existing stopping code were unchanged. This is the first scoring candidate, not a bundled committee/calibration implementation or U5.

### Predictive metrics

AP is mean ± sample SD; other entries below are five-seed means. Complete mean/sample-SD values for every recorded numeric metric are in `metric_summary.csv`; every seed-level value is in `all_metrics.csv` in the run directory.

| Classifier | Variant | AP | Macro AP | Minority F1 | Minority recall | Macro F1 | Weighted F1 | Inverse-support F1 | Balanced accuracy | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| LR | Baseline | 0.6089 ± 0.0028 | 0.7913 | 0.4567 | 0.3340 | 0.6937 | 0.8573 | 0.5301 | 0.6553 | 0.4362 |
| LR | Corrected current | 0.6074 ± 0.0019 | 0.7903 | 0.5574 | 0.4670 | 0.7457 | 0.8757 | 0.6157 | 0.7144 | 0.5069 |
| LR | U1 OOF MMA | 0.6092 ± 0.0016 | 0.7913 | 0.5463 | 0.4534 | 0.7397 | 0.8732 | 0.6062 | 0.7078 | 0.4963 |
| RF | Baseline | 0.7395 ± 0.0057 | 0.8616 | 0.6100 | 0.5136 | 0.7759 | 0.8905 | 0.6614 | 0.7415 | 0.5705 |
| RF | Corrected current | 0.7220 ± 0.0039 | 0.8527 | 0.6383 | 0.6042 | 0.7881 | 0.8915 | 0.6847 | 0.7757 | 0.5780 |
| RF | U1 OOF MMA | 0.7206 ± 0.0055 | 0.8520 | 0.6436 | 0.6136 | 0.7909 | 0.8926 | 0.6892 | 0.7800 | 0.5833 |
| LightGBM | Baseline | 0.7428 ± 0.0039 | 0.8641 | 0.6347 | 0.5670 | 0.7881 | 0.8939 | 0.6822 | 0.7635 | 0.5834 |
| LightGBM | Corrected current | 0.7262 ± 0.0086 | 0.8554 | 0.6477 | 0.6330 | 0.7924 | 0.8923 | 0.6925 | 0.7870 | 0.5852 |
| LightGBM | U1 OOF MMA | 0.7319 ± 0.0023 | 0.8585 | 0.6543 | 0.6398 | 0.7963 | 0.8943 | 0.6983 | 0.7910 | 0.5929 |

Weighted F1 is conventional support-weighted F1. Inverse-support F1 is separately named and must not be presented as a published SOS reproduction.

### Confusion counts and operating points

Counts and rates are five-seed means. The deployed evaluation threshold remained 0.5. Test-ROC recall at FPR ≤0.05 is a separate attainable operating-point description, not a test-selected deployable threshold. Recorded generic cost is `10 FP + 500 FN`; it is **not the Shoppers selection objective or an APS benchmark result**.

| Classifier | Variant | TP | FN | FP | TN | Generic cost | Realized FPR | Test-ROC recall ≤5% FPR | Attained ROC FPR |
|---|---|---|---|---|---|---|---|---|---|
| LR | Baseline | 127.6 | 254.4 | 48.8 | 2035.2 | 127688 | 0.0234 | 0.5162 | 0.0496 |
| LR | Corrected current | 178.4 | 203.6 | 79.6 | 2004.4 | 102596 | 0.0382 | 0.5126 | 0.0497 |
| LR | U1 OOF MMA | 173.2 | 208.8 | 78.8 | 2005.2 | 105188 | 0.0378 | 0.5168 | 0.0498 |
| RF | Baseline | 196.2 | 185.8 | 63.6 | 2020.4 | 93536 | 0.0305 | 0.6173 | 0.0497 |
| RF | Corrected current | 230.8 | 151.2 | 110.2 | 1973.8 | 76702 | 0.0529 | 0.5974 | 0.0498 |
| RF | U1 OOF MMA | 234.4 | 147.6 | 111.8 | 1972.2 | 74918 | 0.0536 | 0.5927 | 0.0492 |
| LightGBM | Baseline | 216.6 | 165.4 | 83.4 | 2000.6 | 83534 | 0.0400 | 0.6152 | 0.0494 |
| LightGBM | Corrected current | 241.8 | 140.2 | 122.8 | 1961.2 | 71328 | 0.0589 | 0.5796 | 0.0496 |
| LightGBM | U1 OOF MMA | 244.4 | 137.6 | 120.6 | 1963.4 | 70006 | 0.0579 | 0.5890 | 0.0498 |

`realized_recall_test` duplicates the minority-recall column; `primary_metric_value` duplicates AP. ROC FPR limit is 0.05 and threshold is 0.5 for every cell. These fields are retained explicitly in the machine-readable exports.

### Paired change relative to corrected_current

All 15 U1/reference pairs have identical actual generation counts, tuning proposals, splits and policy. Deltas are U1 minus corrected_current, mean ± descriptive paired seed SD.

| Classifier | AP delta | Minority F1 delta | Recall delta | Balanced accuracy delta |
|---|---|---|---|---|
| LR | +0.001853 ± 0.002498 | -0.011140 ± 0.013634 | -0.013613 ± 0.017697 | -0.006614 ± 0.008343 |
| RF | -0.001363 ± 0.007392 | +0.005291 ± 0.007009 | +0.009424 ± 0.011795 | +0.004328 ± 0.005353 |
| LightGBM | +0.005687 ± 0.009561 | +0.006625 ± 0.011371 | +0.006806 ± 0.015988 | +0.003931 ± 0.007988 |

Against no resampling, mean AP deltas were +0.000343 (LR), -0.018848 (RF), -0.010918 (LightGBM). Effects are mixed and descriptive; no confidence interval, significance claim or automatic keep/discard conclusion is made. Every metric has both reference comparisons in `paired_metric_deltas.csv`.

### Counts, compute and verification

All 300 generating inner candidate and 30 generating final sampler runs added exactly 1,000 minority rows; baseline added zero. No underspend or padding occurred. The audit recorded 1,485 downstream classifier fits plus 495 real-only scoring fits. Total run time was **1,861.99 seconds**. Below, cell seconds are summed across five seeds, RSS is maximum process RSS (not incremental model memory), and CUDA columns cover Torch allocations/cache only, not total GPU usage. Shared cached preprocessing and allocator state affect these resource descriptions.

| Classifier | Variant | Total cell seconds | Peak RSS MiB | Peak Torch CUDA allocated MiB | Peak Torch CUDA reserved MiB |
|---|---|---|---|---|---|
| LR | Baseline | 1.49 | 1695.47 | 0.00 | 24.00 |
| LR | Corrected current | 70.25 | 1695.47 | 15.67 | 24.00 |
| LR | U1 OOF MMA | 76.01 | 1695.47 | 15.67 | 24.00 |
| RF | Baseline | 103.97 | 1695.47 | 0.00 | 24.00 |
| RF | Corrected current | 574.11 | 1731.79 | 15.67 | 24.00 |
| RF | U1 OOF MMA | 741.18 | 1733.73 | 15.67 | 24.00 |
| LightGBM | Baseline | 13.61 | 1695.47 | 0.00 | 24.00 |
| LightGBM | Corrected current | 126.99 | 1695.47 | 15.67 | 24.00 |
| LightGBM | U1 OOF MMA | 153.07 | 1695.47 | 15.67 | 24.00 |

Verification: seven Shoppers-only regression tests passed; a nine-cell training-only smoke passed (`results/smoke/20261004T063101-9a745595d620`). Full-run checks recomputed all 45 metric sets from saved round-trip predictions, matched historical reference scores, verified all source/split identities, validation-only selection, real-only parent bounds for 495 pipeline audits, exact fold identities/preprocessing/MMA coverage for 165 scoring runs (495 fold audits), and all registry/index links. Outer-test feature/label perturbation and heldout-fold label/category/missingness perturbation tests passed. Existing sklearn/SciPy and LightGBM warnings were not suppressed. No CLIMB/APS/BAF evaluation ran.

Artifacts: `phase1_u1_summary.json`, `all_metrics.csv`, `metric_summary.csv`, `paired_metric_deltas.csv`, `generation_audit_summary.csv`, `paired_vs_corrected.json` and `ap_summary.json` inside the run directory; original per-cell candidate/selection/outer/preprocessing/prediction/source evidence remains canonical. U1_oof_mma is implemented, evaluated and **kept by the user after metric review**. U2–U5 remain unselected; U2 is now runnable for its separately authorized comparison. Full U1 committee/calibration is not bundled, and the Phase 1 configuration is not frozen.

The explicit retention decision applies to the fixed real-only three-fold OOF MMA candidate, not to unimplemented committee/calibration alternatives. Relative to corrected_current, LR AP increased slightly (+0.001853), while LR minority F1/recall decreased; RF minority F1/recall increased (+0.005291/+0.009424), while RF AP decreased (-0.001363). These tradeoffs remain reported despite retention. Current study declarations record U1=keep and U1_oof_mma=kept_by_user; immutable experiment snapshots and derived run summaries preserve their original awaiting-decision state. No model, prediction or numerical metric changed, and no new benchmark was run to record this decision. U2–U5 and confirmatory evaluation require their own subsequent authorization/decisions.

At the time of the retention-only change, verification resolved U1_oof_mma as kept_by_user and allowed development, while refusing the then-unimplemented U2–U5 execution and confirmatory evaluation. All seven Shoppers leakage/user-gate regressions passed again (1.839 seconds). The pending-U1 gate test explicitly sets its own unresolved-decision fixture rather than depending on the current user decision.


## Phase 1 U2 group protection/allocation — user-retained

**After reviewing this complete comparison, the user kept canonical U2 KMeans/stability.** Alternative grouping/allocation results remain evidence, not automatically selected replacements. U1 stays kept; U3 was separately evaluated and discarded, the user also excluded both U4 policies. The user subsequently discarded U5 after testing that combination and reaffirmed canonical U2; confirmatory evaluation remains closed. The historical run below isolates U2 against corrected_current, not U1+U2, and did not evaluate U3/CLIMB/APS/BAF.

Complete invocation: `results/runs/20261004T084817-f93a994968f7`. All **180 cells completed**: 12 variants × LR/RF/LightGBM × seeds 0–4, with ten common classifier proposals per cell. Raw UCI Shoppers has 12,330 rows; the unchanged fixed stratified outer split has 9,864 training and 2,466 test rows (382 positive). Each generating pipeline requested 1,000 additional minority rows; threshold remains 0.5. This is the controlled track, not published SOS/UGO author reproduction; original U0 retains its documented cap/category deviations.

### What U2 changes, and what stays fixed

Real minority groups replace the pooled-class seed allocation: KMeans has eight clusters capped at distinct coordinates; the sparse graph uses mutual five-nearest-neighbor edges below the class empirical 90th-percentile kth-neighbor radius. Five independently fitted 80% real-row subsamples measure incident co-membership Jaccard, including both splits and mergers. Group geometry deduplicates coordinates, but support counts retain distinct authentic source observations. Unsupported/single-coordinate/zero-stability groups receive no quota. Cross-class neighbor conflict is a separate diagnostic, not evidence of label corruption.

Stability allocation uses r_g(h_g+0.001)/(n_g+0.001)^alpha, with mean within-class percentile MMA h_g. Size, uniform-group and difficulty-only weights are separate comparisons on the same support/caps. Saturated largest-remainder integer quotas redistribute only among eligible remaining capacity. Lifetime caps are floor(2*n_g) per group and four accepted descendants per real seed; top-half within-group difficulty selection excludes exhausted seeds. Real support/stability never grow with generated rows.

The two stability variants jointly search alpha=0,0.5,1 by cycling candidate IDs over the same ten classifier proposals (four alpha=0 pairs, three each for 0.5/1). Selection uses inner AP only; full-training refit replays the selected sampler/classifier pair. This is **ten joint proposals, not a 30-pair Cartesian search**, so the chosen alpha is not an independently identified optimum. No policy is chosen using the outer results. Current in-sample MMA, classifier families, class-wide real-neighbor proposal/relocation and original stopping remain unchanged. Generated group counts are charged to the seed's group; **group-local final geometry is not claimed** (U3 was not part of this run).

### All predictive metrics

Every entry is mean ± sample SD over five seeds on one shared fixed holdout; these are descriptive spreads, not confidence intervals or independent-dataset inference. F1/recall target the minority. Conventional support-weighted F1 and SOS inverse-support F1 are separate diagnostics; no published-author aggregation equivalence is claimed.

| Classifier | Variant | AP | Macro AP | F1 | Recall | Macro F1 | Weighted F1 | Inverse-support F1 | BAC | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| LR | Baseline | 0.6089 ± 0.0028 | 0.7913 ± 0.0015 | 0.4567 ± 0.0157 | 0.3340 ± 0.0201 | 0.6937 ± 0.0078 | 0.8573 ± 0.0024 | 0.5301 ± 0.0132 | 0.6553 ± 0.0085 | 0.4362 ± 0.0075 |
| LR | Original UGO (U0) | 0.6077 ± 0.0016 | 0.7903 ± 0.0009 | 0.5591 ± 0.0086 | 0.4712 ± 0.0124 | 0.7465 ± 0.0045 | 0.8758 ± 0.0017 | 0.6172 ± 0.0073 | 0.7160 ± 0.0056 | 0.5073 ± 0.0072 |
| LR | Corrected current | 0.6074 ± 0.0019 | 0.7903 ± 0.0010 | 0.5574 ± 0.0121 | 0.4670 ± 0.0152 | 0.7457 ± 0.0065 | 0.8757 ± 0.0027 | 0.6157 ± 0.0103 | 0.7144 ± 0.0073 | 0.5069 ± 0.0111 |
| LR | Kept U1 OOF MMA | 0.6092 ± 0.0016 | 0.7913 ± 0.0009 | 0.5463 ± 0.0137 | 0.4534 ± 0.0164 | 0.7397 ± 0.0075 | 0.8732 ± 0.0031 | 0.6062 ± 0.0118 | 0.7078 ± 0.0080 | 0.4963 ± 0.0131 |
| LR | KMeans stability (U2) | 0.5854 ± 0.0023 | 0.7787 ± 0.0012 | 0.5482 ± 0.0044 | 0.4885 ± 0.0071 | 0.7379 ± 0.0023 | 0.8689 ± 0.0009 | 0.6069 ± 0.0038 | 0.7173 ± 0.0031 | 0.4819 ± 0.0040 |
| LR | Graph stability | 0.5977 ± 0.0028 | 0.7853 ± 0.0014 | 0.5439 ± 0.0107 | 0.4524 ± 0.0209 | 0.7382 ± 0.0051 | 0.8724 ± 0.0013 | 0.6041 ± 0.0090 | 0.7069 ± 0.0083 | 0.4934 ± 0.0056 |
| LR | KMeans size | 0.6008 ± 0.0027 | 0.7873 ± 0.0014 | 0.5455 ± 0.0079 | 0.4545 ± 0.0149 | 0.7391 ± 0.0039 | 0.8727 ± 0.0013 | 0.6054 ± 0.0066 | 0.7078 ± 0.0059 | 0.4946 ± 0.0052 |
| LR | KMeans uniform | 0.5881 ± 0.0025 | 0.7803 ± 0.0013 | 0.5439 ± 0.0017 | 0.4738 ± 0.0026 | 0.7364 ± 0.0010 | 0.8692 ± 0.0006 | 0.6035 ± 0.0015 | 0.7123 ± 0.0010 | 0.4815 ± 0.0023 |
| LR | KMeans difficulty | 0.5814 ± 0.0031 | 0.7765 ± 0.0016 | 0.5447 ± 0.0039 | 0.4827 ± 0.0060 | 0.7362 ± 0.0022 | 0.8683 ± 0.0010 | 0.6040 ± 0.0034 | 0.7148 ± 0.0026 | 0.4790 ± 0.0042 |
| LR | Graph size | 0.5964 ± 0.0023 | 0.7843 ± 0.0012 | 0.5520 ± 0.0105 | 0.4670 ± 0.0201 | 0.7422 ± 0.0051 | 0.8735 ± 0.0015 | 0.6109 ± 0.0089 | 0.7129 ± 0.0081 | 0.4983 ± 0.0064 |
| LR | Graph uniform | 0.5968 ± 0.0010 | 0.7849 ± 0.0006 | 0.5489 ± 0.0081 | 0.4634 ± 0.0153 | 0.7405 ± 0.0040 | 0.8728 ± 0.0012 | 0.6083 ± 0.0068 | 0.7111 ± 0.0061 | 0.4951 ± 0.0051 |
| LR | Graph difficulty | 0.5972 ± 0.0017 | 0.7851 ± 0.0009 | 0.5453 ± 0.0106 | 0.4571 ± 0.0195 | 0.7387 ± 0.0052 | 0.8723 ± 0.0016 | 0.6052 ± 0.0090 | 0.7085 ± 0.0079 | 0.4928 ± 0.0067 |
| RF | Baseline | 0.7395 ± 0.0057 | 0.8616 ± 0.0030 | 0.6100 ± 0.0254 | 0.5136 ± 0.0569 | 0.7759 ± 0.0124 | 0.8905 ± 0.0034 | 0.6614 ± 0.0214 | 0.7415 ± 0.0232 | 0.5705 ± 0.0131 |
| RF | Original UGO (U0) | 0.7244 ± 0.0052 | 0.8540 ± 0.0031 | 0.6456 ± 0.0081 | 0.6157 ± 0.0224 | 0.7921 ± 0.0039 | 0.8932 ± 0.0011 | 0.6910 ± 0.0068 | 0.7812 ± 0.0091 | 0.5857 ± 0.0067 |
| RF | Corrected current | 0.7220 ± 0.0039 | 0.8527 ± 0.0022 | 0.6383 ± 0.0088 | 0.6042 ± 0.0239 | 0.7881 ± 0.0042 | 0.8915 ± 0.0011 | 0.6847 ± 0.0074 | 0.7757 ± 0.0097 | 0.5780 ± 0.0070 |
| RF | Kept U1 OOF MMA | 0.7206 ± 0.0055 | 0.8520 ± 0.0031 | 0.6436 ± 0.0125 | 0.6136 ± 0.0265 | 0.7909 ± 0.0065 | 0.8926 ± 0.0026 | 0.6892 ± 0.0106 | 0.7800 ± 0.0114 | 0.5833 ± 0.0121 |
| RF | KMeans stability (U2) | 0.7178 ± 0.0052 | 0.8512 ± 0.0029 | 0.6527 ± 0.0075 | 0.6696 ± 0.0099 | 0.7935 ± 0.0044 | 0.8907 ± 0.0023 | 0.6963 ± 0.0065 | 0.7998 ± 0.0050 | 0.5874 ± 0.0088 |
| RF | Graph stability | 0.7277 ± 0.0077 | 0.8562 ± 0.0044 | 0.6554 ± 0.0079 | 0.6660 ± 0.0214 | 0.7955 ± 0.0043 | 0.8922 ± 0.0022 | 0.6988 ± 0.0068 | 0.7994 ± 0.0085 | 0.5914 ± 0.0088 |
| RF | KMeans size | 0.7294 ± 0.0055 | 0.8571 ± 0.0030 | 0.6570 ± 0.0066 | 0.6707 ± 0.0212 | 0.7963 ± 0.0037 | 0.8924 ± 0.0022 | 0.7002 ± 0.0057 | 0.8014 ± 0.0079 | 0.5931 ± 0.0073 |
| RF | KMeans uniform | 0.7180 ± 0.0085 | 0.8510 ± 0.0047 | 0.6555 ± 0.0038 | 0.6581 ± 0.0211 | 0.7960 ± 0.0021 | 0.8930 ± 0.0018 | 0.6990 ± 0.0032 | 0.7970 ± 0.0072 | 0.5924 ± 0.0043 |
| RF | KMeans difficulty | 0.7179 ± 0.0058 | 0.8512 ± 0.0031 | 0.6500 ± 0.0087 | 0.6665 ± 0.0137 | 0.7919 ± 0.0051 | 0.8899 ± 0.0026 | 0.6939 ± 0.0076 | 0.7980 ± 0.0065 | 0.5842 ± 0.0103 |
| RF | Graph size | 0.7193 ± 0.0083 | 0.8518 ± 0.0046 | 0.6544 ± 0.0047 | 0.6634 ± 0.0160 | 0.7950 ± 0.0030 | 0.8921 ± 0.0021 | 0.6980 ± 0.0041 | 0.7983 ± 0.0055 | 0.5904 ± 0.0059 |
| RF | Graph uniform | 0.7274 ± 0.0081 | 0.8560 ± 0.0044 | 0.6551 ± 0.0078 | 0.6660 ± 0.0235 | 0.7953 ± 0.0043 | 0.8921 ± 0.0024 | 0.6985 ± 0.0067 | 0.7993 ± 0.0089 | 0.5911 ± 0.0088 |
| RF | Graph difficulty | 0.7274 ± 0.0081 | 0.8560 ± 0.0044 | 0.6551 ± 0.0078 | 0.6660 ± 0.0235 | 0.7953 ± 0.0043 | 0.8921 ± 0.0024 | 0.6985 ± 0.0067 | 0.7993 ± 0.0089 | 0.5911 ± 0.0088 |
| LGB | Baseline | 0.7428 ± 0.0039 | 0.8641 ± 0.0020 | 0.6347 ± 0.0177 | 0.5670 ± 0.0363 | 0.7881 ± 0.0091 | 0.8939 ± 0.0032 | 0.6822 ± 0.0150 | 0.7635 ± 0.0156 | 0.5834 ± 0.0144 |
| LGB | Original UGO (U0) | 0.7302 ± 0.0033 | 0.8576 ± 0.0017 | 0.6485 ± 0.0065 | 0.6346 ± 0.0094 | 0.7928 ± 0.0037 | 0.8925 ± 0.0018 | 0.6932 ± 0.0056 | 0.7877 ± 0.0046 | 0.5859 ± 0.0073 |
| LGB | Corrected current | 0.7262 ± 0.0086 | 0.8554 ± 0.0048 | 0.6477 ± 0.0074 | 0.6330 ± 0.0122 | 0.7924 ± 0.0041 | 0.8923 ± 0.0019 | 0.6925 ± 0.0064 | 0.7870 ± 0.0057 | 0.5852 ± 0.0081 |
| LGB | Kept U1 OOF MMA | 0.7319 ± 0.0023 | 0.8585 ± 0.0012 | 0.6543 ± 0.0069 | 0.6398 ± 0.0109 | 0.7963 ± 0.0038 | 0.8943 ± 0.0018 | 0.6983 ± 0.0059 | 0.7910 ± 0.0052 | 0.5929 ± 0.0076 |
| LGB | KMeans stability (U2) | 0.7281 ± 0.0054 | 0.8567 ± 0.0027 | 0.6577 ± 0.0080 | 0.6906 ± 0.0116 | 0.7956 ± 0.0048 | 0.8908 ± 0.0026 | 0.7004 ± 0.0070 | 0.8078 ± 0.0056 | 0.5924 ± 0.0096 |
| LGB | Graph stability | 0.7343 ± 0.0036 | 0.8598 ± 0.0020 | 0.6568 ± 0.0078 | 0.6754 ± 0.0149 | 0.7959 ± 0.0043 | 0.8919 ± 0.0019 | 0.6999 ± 0.0067 | 0.8028 ± 0.0068 | 0.5922 ± 0.0088 |
| LGB | KMeans size | 0.7406 ± 0.0021 | 0.8631 ± 0.0011 | 0.6628 ± 0.0066 | 0.6838 ± 0.0090 | 0.7993 ± 0.0040 | 0.8936 ± 0.0023 | 0.7051 ± 0.0058 | 0.8071 ± 0.0043 | 0.5992 ± 0.0080 |
| LGB | KMeans uniform | 0.7319 ± 0.0055 | 0.8585 ± 0.0029 | 0.6545 ± 0.0060 | 0.6686 ± 0.0163 | 0.7948 ± 0.0031 | 0.8916 ± 0.0013 | 0.6979 ± 0.0051 | 0.8000 ± 0.0066 | 0.5899 ± 0.0063 |
| LGB | KMeans difficulty | 0.7266 ± 0.0042 | 0.8558 ± 0.0022 | 0.6625 ± 0.0051 | 0.6874 ± 0.0126 | 0.7989 ± 0.0032 | 0.8931 ± 0.0021 | 0.7047 ± 0.0045 | 0.8082 ± 0.0046 | 0.5986 ± 0.0061 |
| LGB | Graph size | 0.7318 ± 0.0030 | 0.8583 ± 0.0016 | 0.6567 ± 0.0050 | 0.6817 ± 0.0143 | 0.7955 ± 0.0028 | 0.8912 ± 0.0016 | 0.6997 ± 0.0043 | 0.8047 ± 0.0055 | 0.5917 ± 0.0059 |
| LGB | Graph uniform | 0.7372 ± 0.0026 | 0.8614 ± 0.0014 | 0.6628 ± 0.0063 | 0.6906 ± 0.0104 | 0.7989 ± 0.0035 | 0.8929 ± 0.0017 | 0.7050 ± 0.0054 | 0.8093 ± 0.0050 | 0.5987 ± 0.0073 |
| LGB | Graph difficulty | 0.7372 ± 0.0026 | 0.8614 ± 0.0014 | 0.6628 ± 0.0063 | 0.6906 ± 0.0104 | 0.7989 ± 0.0035 | 0.8929 ± 0.0017 | 0.7050 ± 0.0054 | 0.8093 ± 0.0050 | 0.5987 ± 0.0073 |

### Paired changes versus corrected_current

All **120 U2/reference pairs** share exact split/source identities, classifier proposals, tuning budget and actual per-class generated counts. Table entries are mean paired difference ± sample SD of the five differences. Per-seed outcomes and every numeric delta against baseline, original U0, corrected_current and kept U1 are linked below.

| Classifier | U2 policy | Delta AP | Delta F1 | Delta recall | Delta BAC | Delta MCC |
|---|---|---|---|---|---|---|
| LR | KMeans stability (U2) | -0.0220 ± 0.0032 | -0.0092 ± 0.0117 | +0.0215 ± 0.0145 | +0.0029 ± 0.0070 | -0.0250 ± 0.0109 |
| LR | Graph stability | -0.0097 ± 0.0038 | -0.0135 ± 0.0127 | -0.0147 ± 0.0213 | -0.0075 ± 0.0089 | -0.0135 ± 0.0101 |
| LR | KMeans size | -0.0066 ± 0.0015 | -0.0119 ± 0.0075 | -0.0126 ± 0.0107 | -0.0066 ± 0.0046 | -0.0123 ± 0.0088 |
| LR | KMeans uniform | -0.0192 ± 0.0028 | -0.0135 ± 0.0117 | +0.0068 ± 0.0139 | -0.0021 ± 0.0068 | -0.0253 ± 0.0114 |
| LR | KMeans difficulty | -0.0260 ± 0.0031 | -0.0126 ± 0.0130 | +0.0157 ± 0.0158 | +0.0004 ± 0.0076 | -0.0278 ± 0.0127 |
| LR | Graph size | -0.0109 ± 0.0013 | -0.0054 ± 0.0062 | +0.0000 ± 0.0091 | -0.0015 ± 0.0037 | -0.0086 ± 0.0085 |
| LR | Graph uniform | -0.0106 ± 0.0019 | -0.0085 ± 0.0058 | -0.0037 ± 0.0044 | -0.0033 ± 0.0025 | -0.0118 ± 0.0084 |
| LR | Graph difficulty | -0.0102 ± 0.0025 | -0.0121 ± 0.0110 | -0.0099 ± 0.0165 | -0.0059 ± 0.0071 | -0.0140 ± 0.0106 |
| RF | KMeans stability (U2) | -0.0042 ± 0.0073 | +0.0144 ± 0.0120 | +0.0654 ± 0.0236 | +0.0241 ± 0.0103 | +0.0095 ± 0.0121 |
| RF | Graph stability | +0.0057 ± 0.0062 | +0.0172 ± 0.0110 | +0.0618 ± 0.0282 | +0.0238 ± 0.0112 | +0.0135 ± 0.0110 |
| RF | KMeans size | +0.0073 ± 0.0044 | +0.0188 ± 0.0088 | +0.0665 ± 0.0223 | +0.0257 ± 0.0086 | +0.0152 ± 0.0087 |
| RF | KMeans uniform | -0.0040 ± 0.0114 | +0.0172 ± 0.0097 | +0.0539 ± 0.0281 | +0.0213 ± 0.0109 | +0.0144 ± 0.0089 |
| RF | KMeans difficulty | -0.0041 ± 0.0093 | +0.0117 ± 0.0163 | +0.0623 ± 0.0362 | +0.0224 ± 0.0154 | +0.0062 ± 0.0159 |
| RF | Graph size | -0.0027 ± 0.0069 | +0.0162 ± 0.0122 | +0.0592 ± 0.0233 | +0.0227 ± 0.0101 | +0.0124 ± 0.0120 |
| RF | Graph uniform | +0.0054 ± 0.0066 | +0.0168 ± 0.0116 | +0.0618 ± 0.0277 | +0.0237 ± 0.0111 | +0.0131 ± 0.0119 |
| RF | Graph difficulty | +0.0054 ± 0.0066 | +0.0168 ± 0.0116 | +0.0618 ± 0.0277 | +0.0237 ± 0.0111 | +0.0131 ± 0.0119 |
| LGB | KMeans stability (U2) | +0.0019 ± 0.0127 | +0.0100 ± 0.0150 | +0.0576 ± 0.0232 | +0.0207 ± 0.0113 | +0.0072 ± 0.0171 |
| LGB | Graph stability | +0.0081 ± 0.0057 | +0.0091 ± 0.0120 | +0.0424 ± 0.0188 | +0.0157 ± 0.0092 | +0.0070 ± 0.0135 |
| LGB | KMeans size | +0.0145 ± 0.0086 | +0.0151 ± 0.0080 | +0.0508 ± 0.0040 | +0.0201 ± 0.0030 | +0.0140 ± 0.0099 |
| LGB | KMeans uniform | +0.0057 ± 0.0125 | +0.0068 ± 0.0126 | +0.0356 ± 0.0232 | +0.0130 ± 0.0106 | +0.0047 ± 0.0137 |
| LGB | KMeans difficulty | +0.0004 ± 0.0101 | +0.0148 ± 0.0035 | +0.0545 ± 0.0146 | +0.0211 ± 0.0053 | +0.0134 ± 0.0033 |
| LGB | Graph size | +0.0056 ± 0.0095 | +0.0090 ± 0.0109 | +0.0487 ± 0.0138 | +0.0177 ± 0.0071 | +0.0066 ± 0.0126 |
| LGB | Graph uniform | +0.0110 ± 0.0073 | +0.0151 ± 0.0126 | +0.0576 ± 0.0219 | +0.0222 ± 0.0103 | +0.0136 ± 0.0139 |
| LGB | Graph difficulty | +0.0110 ± 0.0073 | +0.0151 ± 0.0126 | +0.0576 ± 0.0219 | +0.0222 ± 0.0103 | +0.0136 ± 0.0139 |

The result is mixed, not an automatic acceptance:
- Default stability-adjusted KMeans loses AP for LR (-0.0220) and RF (-0.0042); its LGB AP gain is +0.0019. LR F1/MCC decrease even though recall increases. RF/LGB F1, recall and BAC increase, at higher deployed false-positive rates.
- Every U2 policy lowers LR AP and minority F1 relative to corrected_current. No U2 policy exceeds no-augmentation baseline AP for any of the three classifier families.
- KMeans-size has higher mean AP than stability-adjusted KMeans in all three families. Its paired AP gains for RF/LGB are +0.0073/+0.0145, but LR still loses -0.0066. This does not establish an advantage for the stability/density weighting formula, or separately identify protection versus cap effects.
- Graph uniform/difficulty produce byte-identical predictions for all ten RF/LGB seed pairs, with identical quotas and ancestry for all 110 paired inner/outer audits. They are policy-equivalent outcomes here, not additional independent evidence.

Selected alphas (seed order 0–4): LR KMeans [0,0,0,0,0], graph [0.5,0.5,0.5,0.5,0.5]; RF KMeans [1,1,1,0.5,0.5], graph [0,1,1,0.5,0.5]; LGB KMeans [0.5,1,0.5,0.5,0.5], graph [1,0.5,1,0.5,1]. These choices came from inner selection, not this comparison.

### Operating points, confusion counts and cost diagnostic

All entries are mean ± sample SD. The deployable decision uses fixed threshold 0.5; its realized FPR is not constrained to 5%. Attainable recall/FPR are descriptive points on the outer-test ROC with unsplit tied scores, **not test-selected deployment thresholds**. Cost=10 FP+500 FN is a generic diagnostic, not a Shoppers economic objective or an APS result; it did not select models.

| Classifier | Variant | TP | FP | TN | FN | FPR @0.5 | Attainable recall <=5% FPR | Attained FPR | 10 FP+500 FN |
|---|---|---|---|---|---|---|---|---|---|
| LR | Baseline | 127.6 ± 7.7 | 48.8 ± 6.6 | 2035.2 ± 6.6 | 254.4 ± 7.7 | 0.0234 ± 0.0032 | 0.5162 ± 0.0043 | 0.0496 ± 0.0003 | 127688.0 ± 3768.3 |
| LR | Original UGO (U0) | 180.0 ± 4.7 | 81.8 ± 2.9 | 2002.2 ± 2.9 | 202.0 ± 4.7 | 0.0393 ± 0.0014 | 0.5105 ± 0.0037 | 0.0495 ± 0.0002 | 101818.0 ± 2346.4 |
| LR | Corrected current | 178.4 ± 5.8 | 79.6 ± 1.8 | 2004.4 ± 1.8 | 203.6 ± 5.8 | 0.0382 ± 0.0009 | 0.5126 ± 0.0039 | 0.0497 ± 0.0004 | 102596.0 ± 2892.4 |
| LR | Kept U1 OOF MMA | 173.2 ± 6.3 | 78.8 ± 0.8 | 2005.2 ± 0.8 | 208.8 ± 6.3 | 0.0378 ± 0.0004 | 0.5168 ± 0.0035 | 0.0498 ± 0.0002 | 105188.0 ± 3123.6 |
| LR | KMeans stability (U2) | 186.6 ± 2.7 | 112.2 ± 1.9 | 1971.8 ± 1.9 | 195.4 ± 2.7 | 0.0538 ± 0.0009 | 0.4738 ± 0.0045 | 0.0490 ± 0.0008 | 98822.0 ± 1333.9 |
| LR | Graph stability | 172.8 ± 8.0 | 80.4 ± 8.8 | 2003.6 ± 8.8 | 209.2 ± 8.0 | 0.0386 ± 0.0042 | 0.5000 ± 0.0041 | 0.0496 ± 0.0004 | 105404.0 ± 3902.8 |
| LR | KMeans size | 173.6 ± 5.7 | 80.8 ± 7.0 | 2003.2 ± 7.0 | 208.4 ± 5.7 | 0.0388 ± 0.0034 | 0.5094 ± 0.0048 | 0.0495 ± 0.0004 | 105008.0 ± 2777.3 |
| LR | KMeans uniform | 181.0 ± 1.0 | 102.6 ± 2.3 | 1981.4 ± 2.3 | 201.0 ± 1.0 | 0.0492 ± 0.0011 | 0.4743 ± 0.0029 | 0.0492 ± 0.0004 | 101526.0 ± 485.3 |
| LR | KMeans difficulty | 184.4 ± 2.3 | 110.6 ± 3.5 | 1973.4 ± 3.5 | 197.6 ± 2.3 | 0.0531 ± 0.0017 | 0.4733 ± 0.0068 | 0.0491 ± 0.0009 | 99906.0 ± 1129.9 |
| LR | Graph size | 178.4 ± 7.7 | 85.8 ± 8.6 | 1998.2 ± 8.6 | 203.6 ± 7.7 | 0.0412 ± 0.0041 | 0.5005 ± 0.0060 | 0.0495 ± 0.0004 | 102658.0 ± 3751.8 |
| LR | Graph uniform | 177.0 ± 5.8 | 85.8 ± 6.7 | 1998.2 ± 6.7 | 205.0 ± 5.8 | 0.0412 ± 0.0032 | 0.4974 ± 0.0019 | 0.0493 ± 0.0006 | 103358.0 ± 2852.1 |
| LR | Graph difficulty | 174.6 ± 7.4 | 83.6 ± 8.0 | 2000.4 ± 8.0 | 207.4 ± 7.4 | 0.0401 ± 0.0038 | 0.4984 ± 0.0014 | 0.0494 ± 0.0006 | 104536.0 ± 3641.6 |
| RF | Baseline | 196.2 ± 21.7 | 63.6 ± 21.9 | 2020.4 ± 21.9 | 185.8 ± 21.7 | 0.0305 ± 0.0105 | 0.6173 ± 0.0079 | 0.0497 ± 0.0003 | 93536.0 ± 10651.6 |
| RF | Original UGO (U0) | 235.2 ± 8.6 | 111.2 ± 9.1 | 1972.8 ± 9.1 | 146.8 ± 8.6 | 0.0534 ± 0.0044 | 0.5958 ± 0.0080 | 0.0497 ± 0.0004 | 74512.0 ± 4188.2 |
| RF | Corrected current | 230.8 ± 9.1 | 110.2 ± 9.8 | 1973.8 ± 9.8 | 151.2 ± 9.1 | 0.0529 ± 0.0047 | 0.5974 ± 0.0062 | 0.0498 ± 0.0002 | 76702.0 ± 4477.6 |
| RF | Kept U1 OOF MMA | 234.4 ± 10.1 | 111.8 ± 9.1 | 1972.2 ± 9.1 | 147.6 ± 10.1 | 0.0536 ± 0.0044 | 0.5927 ± 0.0071 | 0.0492 ± 0.0007 | 74918.0 ± 4990.2 |
| RF | KMeans stability (U2) | 255.8 ± 3.8 | 146.0 ± 4.6 | 1938.0 ± 4.6 | 126.2 ± 3.8 | 0.0701 ± 0.0022 | 0.6026 ± 0.0129 | 0.0495 ± 0.0006 | 64560.0 ± 1883.4 |
| RF | Graph stability | 254.4 ± 8.2 | 139.8 ± 11.9 | 1944.2 ± 11.9 | 127.6 ± 8.2 | 0.0671 ± 0.0057 | 0.6162 ± 0.0156 | 0.0493 ± 0.0005 | 65198.0 ± 3987.6 |
| RF | KMeans size | 256.2 ± 8.1 | 141.6 ± 13.5 | 1942.4 ± 13.5 | 125.8 ± 8.1 | 0.0679 ± 0.0065 | 0.6089 ± 0.0129 | 0.0491 ± 0.0005 | 64316.0 ± 3933.8 |
| RF | KMeans uniform | 251.4 ± 8.0 | 133.6 ± 14.7 | 1950.4 ± 14.7 | 130.6 ± 8.0 | 0.0641 ± 0.0071 | 0.5942 ± 0.0147 | 0.0498 ± 0.0002 | 66636.0 ± 3882.6 |
| RF | KMeans difficulty | 254.6 ± 5.2 | 146.8 ± 6.1 | 1937.2 ± 6.1 | 127.4 ± 5.2 | 0.0704 ± 0.0029 | 0.5969 ± 0.0093 | 0.0495 ± 0.0006 | 65168.0 ± 2593.2 |
| RF | Graph size | 253.4 ± 6.1 | 139.0 ± 12.3 | 1945.0 ± 12.3 | 128.6 ± 6.1 | 0.0667 ± 0.0059 | 0.6126 ± 0.0098 | 0.0495 ± 0.0006 | 65690.0 ± 2943.7 |
| RF | Graph uniform | 254.4 ± 9.0 | 140.2 ± 14.3 | 1943.8 ± 14.3 | 127.6 ± 9.0 | 0.0673 ± 0.0069 | 0.6084 ± 0.0134 | 0.0492 ± 0.0008 | 65202.0 ± 4370.2 |
| RF | Graph difficulty | 254.4 ± 9.0 | 140.2 ± 14.3 | 1943.8 ± 14.3 | 127.6 ± 9.0 | 0.0673 ± 0.0069 | 0.6084 ± 0.0134 | 0.0492 ± 0.0008 | 65202.0 ± 4370.2 |
| LGB | Baseline | 216.6 ± 13.9 | 83.4 ± 11.1 | 2000.6 ± 11.1 | 165.4 ± 13.9 | 0.0400 ± 0.0053 | 0.6152 ± 0.0052 | 0.0494 ± 0.0006 | 83534.0 ± 6826.1 |
| LGB | Original UGO (U0) | 242.4 ± 3.6 | 123.2 ± 2.6 | 1960.8 ± 2.6 | 139.6 ± 3.6 | 0.0591 ± 0.0012 | 0.5963 ± 0.0104 | 0.0496 ± 0.0004 | 71032.0 ± 1785.1 |
| LGB | Corrected current | 241.8 ± 4.7 | 122.8 ± 3.1 | 1961.2 ± 3.1 | 140.2 ± 4.7 | 0.0589 ± 0.0015 | 0.5796 ± 0.0066 | 0.0496 ± 0.0004 | 71328.0 ± 2312.8 |
| LGB | Kept U1 OOF MMA | 244.4 ± 4.2 | 120.6 ± 2.9 | 1963.4 ± 2.9 | 137.6 ± 4.2 | 0.0579 ± 0.0014 | 0.5890 ± 0.0158 | 0.0498 ± 0.0002 | 70006.0 ± 2067.9 |
| LGB | KMeans stability (U2) | 263.8 ± 4.4 | 156.4 ± 6.7 | 1927.6 ± 6.7 | 118.2 ± 4.4 | 0.0750 ± 0.0032 | 0.6047 ± 0.0052 | 0.0496 ± 0.0006 | 60664.0 ± 2203.1 |
| LGB | Graph stability | 258.0 ± 5.7 | 145.6 ± 3.9 | 1938.4 ± 3.9 | 124.0 ± 5.7 | 0.0699 ± 0.0019 | 0.6089 ± 0.0078 | 0.0490 ± 0.0012 | 63456.0 ± 2822.9 |
| LGB | KMeans size | 261.2 ± 3.4 | 145.0 ± 6.4 | 1939.0 ± 6.4 | 120.8 ± 3.4 | 0.0696 ± 0.0031 | 0.6063 ± 0.0130 | 0.0492 ± 0.0003 | 61850.0 ± 1691.6 |
| LGB | KMeans uniform | 255.4 ± 6.2 | 143.0 ± 7.4 | 1941.0 ± 7.4 | 126.6 ± 6.2 | 0.0686 ± 0.0036 | 0.6136 ± 0.0119 | 0.0493 ± 0.0008 | 64730.0 ± 3048.0 |
| LGB | KMeans difficulty | 262.6 ± 4.8 | 148.2 ± 9.9 | 1935.8 ± 9.9 | 119.4 ± 4.8 | 0.0711 ± 0.0048 | 0.6084 ± 0.0095 | 0.0493 ± 0.0006 | 61182.0 ± 2333.7 |
| LGB | Graph size | 260.4 ± 5.5 | 150.6 ± 8.6 | 1933.4 ± 8.6 | 121.6 ± 5.5 | 0.0723 ± 0.0042 | 0.6010 ± 0.0174 | 0.0496 ± 0.0004 | 62306.0 ± 2657.2 |
| LGB | Graph uniform | 263.8 ± 4.0 | 150.2 ± 2.4 | 1933.8 ± 2.4 | 118.2 ± 4.0 | 0.0721 ± 0.0011 | 0.6115 ± 0.0122 | 0.0492 ± 0.0013 | 60602.0 ± 1973.0 |
| LGB | Graph difficulty | 263.8 ± 4.0 | 150.2 ± 2.4 | 1933.8 ± 2.4 | 118.2 ± 4.0 | 0.0721 ± 0.0011 | 0.6115 ± 0.0122 | 0.0492 ± 0.0013 | 60602.0 ± 1973.0 |

Raw fields also retain primary_metric=average_precision and primary_metric_value=AP, realized_recall_test=minority_recall, threshold=0.5, roc_fpr_limit=0.05 and the ROC-point definition. Tuning_budget and unique_candidate_count are both 10. Duplicate/constant fields are not silently dropped from the machine-readable summaries.

### Real-group support and generation behavior

KMeans has eight supported/used groups on every outer refit, preserving all 1,526 real minority observations and all 944 minority cross-class-conflict flags. Its stability range over the five real partitions is 0.488399–0.896700 (unweighted supported-group mean 0.680215). Graph has 264 components, averaging 49.8 supported groups and 1,311.6 supported real observations; it restores 785.6 flagged minority observations on average. Graph stability spans 0–1, with supported-group mean 0.500747; structurally unsupported groups are not labelled corrupt. An eligible group is not guaranteed a positive integer quota: RF graph-stability averages 49.6 used versus 49.8 supported groups.

Below are mean ± sample SD over five outer refits. Duplicates are exact feature-vector repeats relative to the cumulative training matrix. Cross-group relocation is a fraction of relocated points, while same-group partner fraction uses all proposals. Group usage is seed ancestry, not a spatial assignment of the final point.

| Classifier | Policy | Groups receiving quota | Distinct source seeds | Exact feature duplicates | Relocated | Same-group partners | Cross-group relocations |
|---|---|---|---|---|---|---|---|
| LR | KMeans stability (U2) | 8.0 ± 0.0 | 590.0 ± 10.0 | 70.4 ± 6.9 | 70.4 ± 6.9 | 0.754 ± 0.005 | 0.131 ± 0.027 |
| LR | Graph stability | 49.8 ± 0.4 | 661.8 ± 0.4 | 98.0 ± 9.4 | 98.0 ± 9.4 | 0.814 ± 0.007 | 0.113 ± 0.049 |
| LR | KMeans size | 8.0 ± 0.0 | 764.4 ± 0.9 | 81.6 ± 6.8 | 81.6 ± 6.8 | 0.801 ± 0.011 | 0.083 ± 0.043 |
| LR | KMeans uniform | 8.0 ± 0.0 | 587.2 ± 6.9 | 52.8 ± 9.6 | 52.8 ± 9.6 | 0.728 ± 0.012 | 0.183 ± 0.068 |
| LR | KMeans difficulty | 8.0 ± 0.0 | 535.8 ± 4.0 | 57.8 ± 7.2 | 57.8 ± 7.2 | 0.733 ± 0.012 | 0.189 ± 0.041 |
| LR | Graph size | 49.8 ± 0.4 | 661.8 ± 0.4 | 107.0 ± 11.6 | 107.0 ± 11.6 | 0.925 ± 0.003 | 0.034 ± 0.021 |
| LR | Graph uniform | 49.8 ± 0.4 | 661.8 ± 0.4 | 100.4 ± 10.5 | 100.4 ± 10.5 | 0.813 ± 0.006 | 0.123 ± 0.046 |
| LR | Graph difficulty | 49.8 ± 0.4 | 661.8 ± 0.4 | 101.2 ± 10.0 | 101.2 ± 10.0 | 0.811 ± 0.005 | 0.122 ± 0.043 |
| RF | KMeans stability (U2) | 8.0 ± 0.0 | 413.6 ± 35.9 | 52.0 ± 5.7 | 52.0 ± 5.7 | 0.732 ± 0.011 | 0.180 ± 0.039 |
| RF | Graph stability | 49.6 ± 0.5 | 660.8 ± 2.2 | 95.6 ± 7.7 | 95.6 ± 7.7 | 0.816 ± 0.009 | 0.113 ± 0.039 |
| RF | KMeans size | 8.0 ± 0.0 | 764.4 ± 0.9 | 87.4 ± 8.4 | 87.4 ± 8.4 | 0.823 ± 0.016 | 0.099 ± 0.018 |
| RF | KMeans uniform | 8.0 ± 0.0 | 587.2 ± 6.9 | 57.8 ± 5.8 | 57.8 ± 5.8 | 0.759 ± 0.010 | 0.176 ± 0.029 |
| RF | KMeans difficulty | 8.0 ± 0.0 | 508.2 ± 4.7 | 64.0 ± 5.2 | 64.0 ± 5.2 | 0.756 ± 0.006 | 0.162 ± 0.043 |
| RF | Graph size | 49.8 ± 0.4 | 662.0 ± 0.7 | 100.4 ± 8.5 | 100.4 ± 8.5 | 0.923 ± 0.006 | 0.028 ± 0.019 |
| RF | Graph uniform | 49.8 ± 0.4 | 661.8 ± 0.4 | 94.8 ± 8.5 | 94.8 ± 8.5 | 0.810 ± 0.010 | 0.127 ± 0.056 |
| RF | Graph difficulty | 49.8 ± 0.4 | 661.8 ± 0.4 | 94.8 ± 8.5 | 94.8 ± 8.5 | 0.810 ± 0.010 | 0.127 ± 0.056 |
| LGB | KMeans stability (U2) | 8.0 ± 0.0 | 436.0 ± 28.5 | 63.6 ± 8.6 | 63.6 ± 8.6 | 0.730 ± 0.020 | 0.175 ± 0.026 |
| LGB | Graph stability | 49.8 ± 0.4 | 661.8 ± 0.4 | 99.8 ± 7.4 | 99.8 ± 7.4 | 0.811 ± 0.008 | 0.108 ± 0.048 |
| LGB | KMeans size | 8.0 ± 0.0 | 764.4 ± 0.9 | 91.6 ± 11.5 | 91.6 ± 11.5 | 0.805 ± 0.006 | 0.098 ± 0.008 |
| LGB | KMeans uniform | 8.0 ± 0.0 | 587.2 ± 6.9 | 58.6 ± 8.8 | 58.6 ± 8.8 | 0.755 ± 0.016 | 0.138 ± 0.032 |
| LGB | KMeans difficulty | 8.0 ± 0.0 | 509.4 ± 3.7 | 67.0 ± 5.1 | 67.0 ± 5.1 | 0.740 ± 0.011 | 0.199 ± 0.075 |
| LGB | Graph size | 49.8 ± 0.4 | 661.8 ± 0.4 | 97.6 ± 3.6 | 97.6 ± 3.6 | 0.924 ± 0.002 | 0.037 ± 0.016 |
| LGB | Graph uniform | 49.8 ± 0.4 | 661.8 ± 0.4 | 99.4 ± 11.8 | 99.4 ± 11.8 | 0.811 ± 0.007 | 0.122 ± 0.052 |
| LGB | Graph difficulty | 49.8 ± 0.4 | 661.8 ± 0.4 | 99.4 ± 11.8 | 99.4 ± 11.8 | 0.811 ± 0.007 | 0.122 ± 0.052 |

### Actual counts and resources

**Zero failed/unsupported cells and zero full-run underspend.** All 165 generating outer cells and all 1,815 generating inner/outer pipelines accepted exactly 1,000 points; baseline generated zero. No synthetic padding was used. The genuine conflicting-islands regression requested nine but accepted only eight under group caps and exposed the one-point underspend. Every full-run U2 seed multiplicity was at most four and every group/class cap held.

The complete CLI took **8,753.26 seconds**; summed completed-cell time is **8,748.40 seconds**. Audits record **7,260 downstream fits plus 495 real-only OOF fits** for the kept U1 reference (7,755 total). Resource entries below are mean ± sample SD of per-cell wall time and sampled process peaks; overall sampled peak RSS was 1,771.04 MiB.

| Classifier | Variant | Cell seconds | RSS MiB | Torch allocated MiB | Torch reserved MiB | Generated | Underspend |
|---|---|---|---|---|---|---|---|
| LR | Baseline | 0.32 ± 0.03 | 1601.39 ± 289.51 | 0.00 ± 0.00 | 20.80 ± 11.63 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| LR | Original UGO (U0) | 14.06 ± 0.25 | 1700.84 ± 67.48 | 12.77 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LR | Corrected current | 14.80 ± 0.57 | 1712.24 ± 41.57 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LR | Kept U1 OOF MMA | 15.76 ± 0.37 | 1715.22 ± 34.97 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LR | KMeans stability (U2) | 15.89 ± 0.35 | 1715.35 ± 34.68 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LR | Graph stability | 16.55 ± 0.37 | 1715.88 ± 33.56 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LR | KMeans size | 15.58 ± 0.26 | 1715.88 ± 33.55 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LR | KMeans uniform | 15.61 ± 0.22 | 1715.88 ± 33.55 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LR | KMeans difficulty | 15.68 ± 0.21 | 1715.88 ± 33.55 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LR | Graph size | 15.72 ± 0.22 | 1715.89 ± 33.54 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LR | Graph uniform | 15.71 ± 0.24 | 1715.90 ± 33.53 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LR | Graph difficulty | 15.73 ± 0.19 | 1715.91 ± 33.50 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| RF | Baseline | 21.32 ± 0.46 | 1721.13 ± 22.01 | 0.00 ± 0.00 | 26.00 ± 0.00 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| RF | Original UGO (U0) | 89.05 ± 4.58 | 1739.31 ± 13.29 | 12.77 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| RF | Corrected current | 115.19 ± 8.86 | 1741.26 ± 15.44 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| RF | Kept U1 OOF MMA | 147.89 ± 3.65 | 1741.64 ± 15.95 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| RF | KMeans stability (U2) | 118.77 ± 7.60 | 1731.25 ± 4.57 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| RF | Graph stability | 113.27 ± 7.11 | 1739.76 ± 14.59 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| RF | KMeans size | 107.85 ± 2.27 | 1739.92 ± 14.49 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| RF | KMeans uniform | 114.90 ± 9.43 | 1739.13 ± 18.22 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| RF | KMeans difficulty | 117.02 ± 7.24 | 1731.48 ± 3.78 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| RF | Graph size | 113.63 ± 7.18 | 1740.01 ± 14.36 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| RF | Graph uniform | 110.82 ± 6.32 | 1739.90 ± 14.11 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| RF | Graph difficulty | 111.06 ± 6.10 | 1740.02 ± 14.38 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LGB | Baseline | 2.74 ± 0.26 | 1731.45 ± 4.83 | 0.00 ± 0.00 | 26.00 ± 0.00 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| LGB | Original UGO (U0) | 22.03 ± 0.20 | 1731.50 ± 4.77 | 12.77 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LGB | Corrected current | 25.68 ± 0.47 | 1731.50 ± 4.77 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LGB | Kept U1 OOF MMA | 31.04 ± 0.15 | 1731.50 ± 4.77 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LGB | KMeans stability (U2) | 26.38 ± 0.21 | 1731.50 ± 4.77 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LGB | Graph stability | 27.28 ± 1.46 | 1731.50 ± 4.77 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LGB | KMeans size | 26.82 ± 0.40 | 1731.50 ± 4.77 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LGB | KMeans uniform | 26.99 ± 0.44 | 1731.50 ± 4.77 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LGB | KMeans difficulty | 26.97 ± 0.50 | 1731.50 ± 4.77 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LGB | Graph size | 27.63 ± 1.45 | 1731.50 ± 4.77 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LGB | Graph uniform | 26.92 ± 0.26 | 1731.50 ± 4.77 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| LGB | Graph difficulty | 27.01 ± 0.38 | 1731.50 ± 4.77 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |

RSS is whole-process memory sampled every 10 ms, not an isolated incremental-method allocation or an exact heap peak. Torch CUDA counters cover its allocator only, not all GPU libraries. Reserved memory includes earlier cached allocations (hence nonzero baseline reservations despite zero baseline Torch allocation). Shared preprocessing/group partitions are reused across classifiers/policies; first-use variants pay construction time. Cell time includes ten-candidate selection, outer refit and audit writing, not dataset acquisition; selected hyperparameters/fit paths differ. These are observed resources, not a controlled microbenchmark or cross-run speed claim.

### Verification, interruption and complete artifacts

Sixty focused regressions passed: 25 grouping/quota, nine Shoppers leakage/manual-gate, seven existing UGO including actual CUDA behavior, and 19 result-contract tests. The training-only 36-cell CLI smoke at results/smoke/20261004T083250-b2c2948fb6b9 is not study evidence; all saved smoke metrics, 72 pipelines, 48 group audits and four exact real-only group refits passed verification.

Full-run proof recomputed all 180 metric sets from round-trip natural-test predictions; verified 1,980 pipeline/1,320 U2 group audits, 20 exact real-only group refits, all preprocessing states, alpha selection/replay, quota weights/caps, actual multiplicities and real same-class ancestry. All 180 cells matched reconstructed split roles/source IDs and classifier proposal lists; inner groups excluded inner validation/test, while outer groups legitimately included full real training only. Seven standard predictive reference metrics (AP, minority F1/recall, macro F1, weighted F1, BAC and MCC) reproduced for all 60 baseline/U0/current/U1 cells to absolute tolerance 1e-12. Canonical registry/index links and paired count flags passed. Outer-test feature/label perturbation tests left training/groups/selection unchanged. Existing sklearn/SciPy iprint and LightGBM feature-name warnings were not suppressed.

**Infrastructure interruption is separate from model failures:** the tool clamped the first invocation's requested 86,400-second timeout to 3,600. It was cancelled proactively and restarted unchanged with no deadline. results/runs/20261004T083820-03e4afb1e1e6 preserves 16 completed prediction sets (all 12 LR plus four RF references, seed 0), any completed inner proposals, its original unfinalized manifest and interruption.json. The clean invocation repeats that work for complete timing/memory capture. Repeated configurations are not extra independent seeds or additional policy evidence; **the partial attempt and its extra compute are excluded from every aggregate/resource table above**. No selection/search adaptation was made after the interruption.

Complete artifacts in the final run (immutable original candidate/selection/outer/preprocessing/source/prediction evidence remains canonical):
- [All 180 per-seed outcomes/resources](../results/runs/20261004T084817-f93a994968f7/all_metrics.csv), [all 1,044 numeric mean/sample-SD summaries](../results/runs/20261004T084817-f93a994968f7/metric_summary.csv), [all 3,828 numeric paired-delta summaries against four references](../results/runs/20261004T084817-f93a994968f7/paired_metric_deltas.csv).
- [All 1,980 pipeline generation audits](../results/runs/20261004T084817-f93a994968f7/generation_audit_summary.csv), [all outer real-group audits](../results/runs/20261004T084817-f93a994968f7/group_audit_summary.csv), [selected sampler/group configurations](../results/runs/20261004T084817-f93a994968f7/selection_group_summary.csv).
- [All 120 per-cell mechanism diagnostics](../results/runs/20261004T084817-f93a994968f7/mechanism_diagnostics.csv), [their full mean/sample-SD summaries](../results/runs/20261004T084817-f93a994968f7/mechanism_summary.csv), [canonical AP/count pairings](../results/runs/20261004T084817-f93a994968f7/paired_vs_corrected.json), [AP summaries](../results/runs/20261004T084817-f93a994968f7/ap_summary.json).
- [Complete verification, metrics, equivalences and interruption summary](../results/runs/20261004T084817-f93a994968f7/phase1_u2_summary.json).

**User decision: keep canonical U2 KMeans/stability.** Its published mixed outcomes remain unchanged; historical experiment snapshots still correctly record awaiting review at execution time. No best outer-test alternative, combination or Phase 1 freeze was selected automatically. U3 was separately evaluated and subsequently discarded.

## Phase 1 U3 real-reference constraints — manual review

**User decision: discard U3, including shortening, relocation and rejection. U1 and canonical U2 KMeans/stability remain kept.** Historical code/results and execution-time pending-review snapshots are preserved; no U3 policy is retained. This experiment isolates U3 against corrected_current: it does not combine U1 scoring or U2 support/allocation, and does not authorize U4/CLIMB/APS/BAF/Satimage or a freeze.

Complete invocation: results/runs/20261005T184048-c25b2afe26ae. **120/120 cells completed**, no failed/unsupported cells: eight variants × LR/RF/LightGBM × seeds 0–4, ten shared classifier proposals and one selected-procedure refit per cell. The fixed Shoppers outer split is unchanged: 9,864 real training rows (1,526 positive) and 2,466 natural test rows (382 positive); threshold is 0.5. Every generating pipeline requested 1,000 additional minority rows. Original U0 retains its controlled-track cap/category deviations; no published-author reproduction claim is made.

### Before/after: what U3 changes

| Mechanism | Original/current UGO reference | Isolated U3 |
|---|---|---|
| Partner scope | Same-class neighbors, potentially crossing groups/categories | Eligible distinct real neighbors in one connected numeric component and an identical entire categorical tuple |
| Numeric geometry | No explicit seed/opposing-distance radius or distance ratio | Both real-reference radius and same/opposing-class ratio constraints |
| Failed point | Class-global relocation, accepted without these bounds | Reject, halve at most four times, or compatible group-local relocation; repaired coordinates must pass the unchanged bounds |
| Categories | Parent-value projection in the controlled reference | Identical parent tuples preserve valid complete one-hot/nominal values without category blending |
| Scoring/allocation/stop | Current in-sample true-label MMA, pooled-class allocation and original stopping | Unchanged; no U2 flag restoration/stability/quota/caps, no U1 OOF scoring, no correctly-classified-seed requirement |
| References | Current is real-only; original U0 can use descendants | Geometry, support/pools and distance trees are frozen from real training rows only |
| Accounting | Generation/ancestry/duplicates | Also attempted seeds, missing compatible partners, initial/final failures, repairs, exact emitted coordinates, retry counts and underspend |

Numeric mutual-five-NN components use the empirical 90th-percentile kth-neighbor radius, after deduplicating numeric coordinates for graph geometry. Exact full categorical tuples further restrict partner pools. Categorical-only/single-coordinate pools cannot interpolate. All real same/opposing-class observations participate in distance queries, including observations excluded as seeds by baseline neighbor-conflict flags. Conflict flags remain diagnostics, not known corruption labels.

Actual emitted Float32 coordinates are evaluated with Float64 numeric norms:

- displacement from seed <= eta × nearest real opposing-class distance at the seed;
- nearest real same-class distance / (nearest real opposing-class distance + 1e-12) <= rho.

Checks are inclusive, with no extra slack or silent weakening. Interpolation/shortening stay on a compatible parent segment; relocation copies an eligible compatible real point and then rechecks both constraints relative to the original seed. Numerical features retain the protocol's existing continuous-valued standardized representation; no application-specific integer rounding or logical-domain rules are claimed. These empirical checks do not guarantee correct labels or coverage of curved support.

Three paired settings (eta,rho)=(0.25,0.5),(0.5,1.0),(1.0,1.5) cycle across the same ten classifier proposals, with 4/3/3 slots. Inner AP alone selects the joint sampler/classifier pair; outer training rebuilds real geometry and replays the selected pair. This is not a nine-setting Cartesian search or an independently identified eta/rho optimum. Geometry is shared across repair policies/families only within a real training role; the cache also includes target-class scope. No outer-score adaptation or automatic best-policy selection occurred.

### Every predictive metric

Mean ± sample SD across five seeds on one fixed holdout; descriptive spreads, not confidence intervals or independent-dataset evidence. F1/recall target Revenue=true. Conventional weighted F1 and inverse-support F1 are distinct diagnostics.

| Classifier | Variant | AP | Macro AP | F1 | Recall | Macro F1 | Weighted F1 | Inverse-support F1 | BAC | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| LR | Baseline | 0.6089 ± 0.0028 | 0.7913 ± 0.0015 | 0.4567 ± 0.0157 | 0.3340 ± 0.0201 | 0.6937 ± 0.0078 | 0.8573 ± 0.0024 | 0.5301 ± 0.0132 | 0.6553 ± 0.0085 | 0.4362 ± 0.0075 |
| LR | Original UGO (U0) | 0.6077 ± 0.0016 | 0.7903 ± 0.0009 | 0.5591 ± 0.0086 | 0.4712 ± 0.0124 | 0.7465 ± 0.0045 | 0.8758 ± 0.0017 | 0.6172 ± 0.0073 | 0.7160 ± 0.0056 | 0.5073 ± 0.0072 |
| LR | Corrected current | 0.6074 ± 0.0019 | 0.7903 ± 0.0010 | 0.5574 ± 0.0121 | 0.4670 ± 0.0152 | 0.7457 ± 0.0065 | 0.8757 ± 0.0027 | 0.6157 ± 0.0103 | 0.7144 ± 0.0073 | 0.5069 ± 0.0111 |
| LR | Kept U1 OOF MMA | 0.6092 ± 0.0016 | 0.7913 ± 0.0009 | 0.5463 ± 0.0137 | 0.4534 ± 0.0164 | 0.7397 ± 0.0075 | 0.8732 ± 0.0031 | 0.6062 ± 0.0118 | 0.7078 ± 0.0080 | 0.4963 ± 0.0131 |
| LR | Kept U2 KMeans stability | 0.5854 ± 0.0023 | 0.7787 ± 0.0012 | 0.5482 ± 0.0044 | 0.4885 ± 0.0071 | 0.7379 ± 0.0023 | 0.8689 ± 0.0009 | 0.6069 ± 0.0038 | 0.7173 ± 0.0031 | 0.4819 ± 0.0040 |
| LR | U3 shortening | 0.5906 ± 0.0008 | 0.7809 ± 0.0004 | 0.5392 ± 0.0044 | 0.4571 ± 0.0084 | 0.7348 ± 0.0021 | 0.8698 ± 0.0007 | 0.5998 ± 0.0037 | 0.7067 ± 0.0033 | 0.4825 ± 0.0030 |
| LR | U3 relocation | 0.5874 ± 0.0013 | 0.7791 ± 0.0006 | 0.5368 ± 0.0064 | 0.4576 ± 0.0119 | 0.7332 ± 0.0031 | 0.8687 ± 0.0009 | 0.5976 ± 0.0054 | 0.7061 ± 0.0048 | 0.4784 ± 0.0041 |
| LR | U3 rejection | 0.5994 ± 0.0032 | 0.7859 ± 0.0015 | 0.5059 ± 0.0102 | 0.3953 ± 0.0146 | 0.7189 ± 0.0052 | 0.8660 ± 0.0017 | 0.5719 ± 0.0087 | 0.6823 ± 0.0062 | 0.4679 ± 0.0064 |
| RF | Baseline | 0.7395 ± 0.0057 | 0.8616 ± 0.0030 | 0.6100 ± 0.0254 | 0.5136 ± 0.0569 | 0.7759 ± 0.0124 | 0.8905 ± 0.0034 | 0.6614 ± 0.0214 | 0.7415 ± 0.0232 | 0.5705 ± 0.0131 |
| RF | Original UGO (U0) | 0.7244 ± 0.0052 | 0.8540 ± 0.0031 | 0.6456 ± 0.0081 | 0.6157 ± 0.0224 | 0.7921 ± 0.0039 | 0.8932 ± 0.0011 | 0.6910 ± 0.0068 | 0.7812 ± 0.0091 | 0.5857 ± 0.0067 |
| RF | Corrected current | 0.7220 ± 0.0039 | 0.8527 ± 0.0022 | 0.6383 ± 0.0088 | 0.6042 ± 0.0239 | 0.7881 ± 0.0042 | 0.8915 ± 0.0011 | 0.6847 ± 0.0074 | 0.7757 ± 0.0097 | 0.5780 ± 0.0070 |
| RF | Kept U1 OOF MMA | 0.7206 ± 0.0055 | 0.8520 ± 0.0031 | 0.6436 ± 0.0125 | 0.6136 ± 0.0265 | 0.7909 ± 0.0065 | 0.8926 ± 0.0026 | 0.6892 ± 0.0106 | 0.7800 ± 0.0114 | 0.5833 ± 0.0121 |
| RF | Kept U2 KMeans stability | 0.7178 ± 0.0052 | 0.8512 ± 0.0029 | 0.6527 ± 0.0075 | 0.6696 ± 0.0099 | 0.7935 ± 0.0044 | 0.8907 ± 0.0023 | 0.6963 ± 0.0065 | 0.7998 ± 0.0050 | 0.5874 ± 0.0088 |
| RF | U3 shortening | 0.7215 ± 0.0084 | 0.8521 ± 0.0036 | 0.6222 ± 0.0037 | 0.5581 ± 0.0201 | 0.7806 ± 0.0011 | 0.8900 ± 0.0008 | 0.6713 ± 0.0029 | 0.7575 ± 0.0069 | 0.5677 ± 0.0017 |
| RF | U3 relocation | 0.7310 ± 0.0029 | 0.8572 ± 0.0015 | 0.6380 ± 0.0113 | 0.5880 ± 0.0345 | 0.7889 ± 0.0056 | 0.8930 ± 0.0024 | 0.6847 ± 0.0095 | 0.7707 ± 0.0130 | 0.5822 ± 0.0096 |
| RF | U3 rejection | 0.7314 ± 0.0022 | 0.8576 ± 0.0007 | 0.6347 ± 0.0135 | 0.5864 ± 0.0376 | 0.7869 ± 0.0064 | 0.8919 ± 0.0018 | 0.6818 ± 0.0113 | 0.7693 ± 0.0149 | 0.5779 ± 0.0097 |
| LGB | Baseline | 0.7428 ± 0.0039 | 0.8641 ± 0.0020 | 0.6347 ± 0.0177 | 0.5670 ± 0.0363 | 0.7881 ± 0.0091 | 0.8939 ± 0.0032 | 0.6822 ± 0.0150 | 0.7635 ± 0.0156 | 0.5834 ± 0.0144 |
| LGB | Original UGO (U0) | 0.7302 ± 0.0033 | 0.8576 ± 0.0017 | 0.6485 ± 0.0065 | 0.6346 ± 0.0094 | 0.7928 ± 0.0037 | 0.8925 ± 0.0018 | 0.6932 ± 0.0056 | 0.7877 ± 0.0046 | 0.5859 ± 0.0073 |
| LGB | Corrected current | 0.7262 ± 0.0086 | 0.8554 ± 0.0048 | 0.6477 ± 0.0074 | 0.6330 ± 0.0122 | 0.7924 ± 0.0041 | 0.8923 ± 0.0019 | 0.6925 ± 0.0064 | 0.7870 ± 0.0057 | 0.5852 ± 0.0081 |
| LGB | Kept U1 OOF MMA | 0.7319 ± 0.0023 | 0.8585 ± 0.0012 | 0.6543 ± 0.0069 | 0.6398 ± 0.0109 | 0.7963 ± 0.0038 | 0.8943 ± 0.0018 | 0.6983 ± 0.0059 | 0.7910 ± 0.0052 | 0.5929 ± 0.0076 |
| LGB | Kept U2 KMeans stability | 0.7281 ± 0.0054 | 0.8567 ± 0.0027 | 0.6577 ± 0.0080 | 0.6906 ± 0.0116 | 0.7956 ± 0.0048 | 0.8908 ± 0.0026 | 0.7004 ± 0.0070 | 0.8078 ± 0.0056 | 0.5924 ± 0.0096 |
| LGB | U3 shortening | 0.7231 ± 0.0014 | 0.8540 ± 0.0007 | 0.6341 ± 0.0123 | 0.6005 ± 0.0203 | 0.7856 ± 0.0067 | 0.8902 ± 0.0029 | 0.6810 ± 0.0106 | 0.7734 ± 0.0094 | 0.5728 ± 0.0126 |
| LGB | U3 relocation | 0.7233 ± 0.0015 | 0.8541 ± 0.0008 | 0.6385 ± 0.0070 | 0.6157 ± 0.0171 | 0.7875 ± 0.0037 | 0.8904 ± 0.0017 | 0.6847 ± 0.0060 | 0.7792 ± 0.0069 | 0.5759 ± 0.0071 |
| LGB | U3 rejection | 0.7294 ± 0.0026 | 0.8572 ± 0.0013 | 0.6416 ± 0.0075 | 0.6136 ± 0.0166 | 0.7896 ± 0.0039 | 0.8918 ± 0.0015 | 0.6874 ± 0.0064 | 0.7794 ± 0.0070 | 0.5804 ± 0.0071 |

### Paired changes against corrected_current

Same splits, seeds, classifier families, ten proposals and requested cap. **All 45 U3 outer cells underspent; none matches actual per-class counts.** These are descriptive full-procedure contrasts, not equal-count evidence isolating the causal effect of geometry. Baseline is also a non-generation control, not an equal-count mechanism comparison.

| Classifier | U3 policy | ΔAP | ΔMacro AP | ΔF1 | ΔRecall | ΔMacro F1 | ΔWeighted F1 | ΔInverse-support F1 | ΔBAC | ΔMCC | Equal counts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LR | U3 shortening | -0.0167 ± 0.0012 | -0.0094 ± 0.0007 | -0.0182 ± 0.0084 | -0.0099 ± 0.0073 | -0.0109 ± 0.0048 | -0.0059 ± 0.0024 | -0.0160 ± 0.0073 | -0.0077 ± 0.0042 | -0.0244 ± 0.0096 | 0/5 |
| LR | U3 relocation | -0.0199 ± 0.0021 | -0.0112 ± 0.0011 | -0.0206 ± 0.0062 | -0.0094 ± 0.0051 | -0.0125 ± 0.0037 | -0.0070 ± 0.0020 | -0.0181 ± 0.0054 | -0.0083 ± 0.0030 | -0.0285 ± 0.0077 | 0/5 |
| LR | U3 rejection | -0.0080 ± 0.0035 | -0.0044 ± 0.0018 | -0.0515 ± 0.0122 | -0.0717 ± 0.0163 | -0.0268 ± 0.0065 | -0.0097 ± 0.0025 | -0.0438 ± 0.0104 | -0.0321 ± 0.0074 | -0.0389 ± 0.0106 | 0/5 |
| RF | U3 shortening | -0.0005 ± 0.0114 | -0.0006 ± 0.0055 | -0.0161 ± 0.0074 | -0.0461 ± 0.0209 | -0.0074 ± 0.0038 | -0.0015 ± 0.0016 | -0.0134 ± 0.0063 | -0.0182 ± 0.0081 | -0.0102 ± 0.0077 | 0/5 |
| RF | U3 relocation | +0.0090 ± 0.0065 | +0.0044 ± 0.0033 | -0.0003 ± 0.0086 | -0.0162 ± 0.0153 | +0.0008 ± 0.0049 | +0.0015 ± 0.0026 | +0.0000 ± 0.0075 | -0.0050 ± 0.0063 | +0.0042 ± 0.0106 | 0/5 |
| RF | U3 rejection | +0.0094 ± 0.0048 | +0.0048 ± 0.0024 | -0.0036 ± 0.0062 | -0.0178 ± 0.0145 | -0.0012 ± 0.0033 | +0.0005 ± 0.0015 | -0.0029 ± 0.0053 | -0.0063 ± 0.0058 | -0.0001 ± 0.0060 | 0/5 |
| LGB | U3 shortening | -0.0030 ± 0.0088 | -0.0014 ± 0.0049 | -0.0136 ± 0.0092 | -0.0325 ± 0.0167 | -0.0068 ± 0.0049 | -0.0021 ± 0.0020 | -0.0115 ± 0.0079 | -0.0136 ± 0.0076 | -0.0124 ± 0.0092 | 0/5 |
| LGB | U3 relocation | -0.0029 ± 0.0099 | -0.0014 ± 0.0055 | -0.0092 ± 0.0089 | -0.0173 ± 0.0230 | -0.0049 ± 0.0045 | -0.0019 ± 0.0016 | -0.0078 ± 0.0075 | -0.0078 ± 0.0094 | -0.0092 ± 0.0082 | 0/5 |
| LGB | U3 rejection | +0.0032 ± 0.0065 | +0.0017 ± 0.0037 | -0.0061 ± 0.0103 | -0.0194 ± 0.0215 | -0.0028 ± 0.0054 | -0.0005 ± 0.0020 | -0.0051 ± 0.0088 | -0.0076 ± 0.0094 | -0.0048 ± 0.0100 | 0/5 |

The declared shortening candidate regresses in all nine predictive means for every family: AP changes are LR -0.0167, RF -0.0005, LGB -0.0030; F1 -0.0182/-0.0161/-0.0136; recall -0.0099/-0.0461/-0.0325. Relocation/rejection improve RF AP (+0.0090/+0.0094), and rejection improves LGB AP (+0.0032), but all tested U3 policies have lower mean F1/recall than corrected_current. RF relocation has small macro/weighted-F1/MCC gains, preserved above; it is not a uniformly better procedure. None beats no-resampling baseline AP in any family. These results do not evaluate a U1+U2+U3 combination or disprove all possible constraint designs.

### Operating points, confusion counts and diagnostic cost

FPR is realized at the fixed 0.5 threshold. Recall@FPR≤0.05 is the best attainable natural-test ROC point with ties intact and no interpolation, not a validation-selected operating threshold. Realized recall is already shown above. The cost 10×FP+500×FN is a generic recorded diagnostic, not a Shoppers economic objective or an APS experiment. Threshold/FPR-limit constants and duplicate primary/recall values are retained in the complete CSV.

| Classifier | Variant | FPR@0.5 | Recall@FPR≤.05 | Attained FPR | TP | FP | TN | FN | 10FP+500FN |
|---|---|---|---|---|---|---|---|---|---|
| LR | Baseline | 0.023 ± 0.003 | 0.516 ± 0.004 | 0.050 ± 0.000 | 127.600 ± 7.668 | 48.800 ± 6.573 | 2035.200 ± 6.573 | 254.400 ± 7.668 | 127688.000 ± 3768.331 |
| LR | Original UGO (U0) | 0.039 ± 0.001 | 0.510 ± 0.004 | 0.050 ± 0.000 | 180.000 ± 4.743 | 81.800 ± 2.864 | 2002.200 ± 2.864 | 202.000 ± 4.743 | 101818.000 ± 2346.448 |
| LR | Corrected current | 0.038 ± 0.001 | 0.513 ± 0.004 | 0.050 ± 0.000 | 178.400 ± 5.814 | 79.600 ± 1.817 | 2004.400 ± 1.817 | 203.600 ± 5.814 | 102596.000 ± 2892.374 |
| LR | Kept U1 OOF MMA | 0.038 ± 0.000 | 0.517 ± 0.004 | 0.050 ± 0.000 | 173.200 ± 6.261 | 78.800 ± 0.837 | 2005.200 ± 0.837 | 208.800 ± 6.261 | 105188.000 ± 3123.631 |
| LR | Kept U2 KMeans stability | 0.054 ± 0.001 | 0.474 ± 0.005 | 0.049 ± 0.001 | 186.600 ± 2.702 | 112.200 ± 1.924 | 1971.800 ± 1.924 | 195.400 ± 2.702 | 98822.000 ± 1333.930 |
| LR | U3 shortening | 0.044 ± 0.002 | 0.480 ± 0.003 | 0.050 ± 0.000 | 174.600 ± 3.209 | 91.000 ± 4.062 | 1993.000 ± 4.062 | 207.400 ± 3.209 | 104610.000 ± 1567.370 |
| LR | U3 relocation | 0.045 ± 0.002 | 0.475 ± 0.005 | 0.049 ± 0.001 | 174.800 ± 4.550 | 94.400 ± 4.930 | 1989.600 ± 4.930 | 207.200 ± 4.550 | 104544.000 ± 2226.866 |
| LR | U3 rejection | 0.031 ± 0.002 | 0.493 ± 0.007 | 0.049 ± 0.001 | 151.000 ± 5.568 | 63.800 ± 4.438 | 2020.200 ± 4.438 | 231.000 ± 5.568 | 116138.000 ± 2740.341 |
| RF | Baseline | 0.031 ± 0.011 | 0.617 ± 0.008 | 0.050 ± 0.000 | 196.200 ± 21.742 | 63.600 ± 21.950 | 2020.400 ± 21.950 | 185.800 ± 21.742 | 93536.000 ± 10651.628 |
| RF | Original UGO (U0) | 0.053 ± 0.004 | 0.596 ± 0.008 | 0.050 ± 0.000 | 235.200 ± 8.556 | 111.200 ± 9.094 | 1972.800 ± 9.094 | 146.800 ± 8.556 | 74512.000 ± 4188.230 |
| RF | Corrected current | 0.053 ± 0.005 | 0.597 ± 0.006 | 0.050 ± 0.000 | 230.800 ± 9.149 | 110.200 ± 9.783 | 1973.800 ± 9.783 | 151.200 ± 9.149 | 76702.000 ± 4477.619 |
| RF | Kept U1 OOF MMA | 0.054 ± 0.004 | 0.593 ± 0.007 | 0.049 ± 0.001 | 234.400 ± 10.139 | 111.800 ± 9.094 | 1972.200 ± 9.094 | 147.600 ± 10.139 | 74918.000 ± 4990.217 |
| RF | Kept U2 KMeans stability | 0.070 ± 0.002 | 0.603 ± 0.013 | 0.050 ± 0.001 | 255.800 ± 3.768 | 146.000 ± 4.637 | 1938.000 ± 4.637 | 126.200 ± 3.768 | 64560.000 ± 1883.388 |
| RF | U3 shortening | 0.043 ± 0.006 | 0.592 ± 0.011 | 0.050 ± 0.000 | 213.200 ± 7.662 | 90.000 ± 12.981 | 1994.000 ± 12.981 | 168.800 ± 7.662 | 85300.000 ± 3701.263 |
| RF | U3 relocation | 0.047 ± 0.009 | 0.597 ± 0.011 | 0.050 ± 0.000 | 224.600 ± 13.164 | 97.200 ± 19.228 | 1986.800 ± 19.228 | 157.400 ± 13.164 | 79672.000 ± 6402.810 |
| RF | U3 rejection | 0.048 ± 0.008 | 0.595 ± 0.010 | 0.049 ± 0.001 | 224.000 ± 14.353 | 99.400 ± 16.876 | 1984.600 ± 16.876 | 158.000 ± 14.353 | 79994.000 ± 7012.380 |
| LGB | Baseline | 0.040 ± 0.005 | 0.615 ± 0.005 | 0.049 ± 0.001 | 216.600 ± 13.867 | 83.400 ± 11.082 | 2000.600 ± 11.082 | 165.400 ± 13.867 | 83534.000 ± 6826.074 |
| LGB | Original UGO (U0) | 0.059 ± 0.001 | 0.596 ± 0.010 | 0.050 ± 0.000 | 242.400 ± 3.578 | 123.200 ± 2.588 | 1960.800 ± 2.588 | 139.600 ± 3.578 | 71032.000 ± 1785.125 |
| LGB | Corrected current | 0.059 ± 0.001 | 0.580 ± 0.007 | 0.050 ± 0.000 | 241.800 ± 4.658 | 122.800 ± 3.114 | 1961.200 ± 3.114 | 140.200 ± 4.658 | 71328.000 ± 2312.784 |
| LGB | Kept U1 OOF MMA | 0.058 ± 0.001 | 0.589 ± 0.016 | 0.050 ± 0.000 | 244.400 ± 4.159 | 120.600 ± 2.881 | 1963.400 ± 2.881 | 137.600 ± 4.159 | 70006.000 ± 2067.929 |
| LGB | Kept U2 KMeans stability | 0.075 ± 0.003 | 0.605 ± 0.005 | 0.050 ± 0.001 | 263.800 ± 4.438 | 156.400 ± 6.656 | 1927.600 ± 6.656 | 118.200 ± 4.438 | 60664.000 ± 2203.050 |
| LGB | U3 shortening | 0.054 ± 0.002 | 0.585 ± 0.015 | 0.049 ± 0.001 | 229.400 ± 7.765 | 112.000 ± 4.848 | 1972.000 ± 4.848 | 152.600 ± 7.765 | 77420.000 ± 3849.981 |
| LGB | U3 relocation | 0.057 ± 0.004 | 0.591 ± 0.010 | 0.049 ± 0.000 | 235.200 ± 6.535 | 119.400 ± 8.620 | 1964.600 ± 8.620 | 146.800 ± 6.535 | 74594.000 ± 3194.672 |
| LGB | U3 rejection | 0.055 ± 0.003 | 0.586 ± 0.011 | 0.050 ± 0.000 | 234.400 ± 6.348 | 114.200 ± 6.017 | 1969.800 ± 6.017 | 147.600 ± 6.348 | 74942.000 ± 3120.436 |

### Generation, repairs, coverage and selection

Every U3 inner and outer pipeline underspent: 495/495; original/current/U1/U2 reached their requested caps in all 660 generating audits. Across the 45 selected U3 refits, 433,991 seed attempts yielded 31,818 accepted rows: 339,644 attempts lacked a compatible distinct partner, and 62,529 constructed candidates failed the unchanged constraints after any permitted repair. No rejected row was padded or replaced to meet counts.

The outer numeric graph has 177 real minority components, 140 with one distinct numeric coordinate. All accepted ancestry is charged to **one component per cell**, not a group-coverage guarantee; the class-pooled baseline allocation was intentionally preserved. Used seed counts and maximum descendants show the concentration below; U2's per-seed cap of four is not part of isolated U3. Relocation contributes 6,104 exact real duplicates; shortening/rejection produce zero exact duplicates in these selected outer refits. Accepted group identity denotes compatible real-source ancestry, not a theorem about the segment interior.

Mean ± sample SD, selected outer refits:

| Classifier | Policy | Attempts | No compatible partner | Constraint rejected | Accepted shortened | Accepted relocated | Exact duplicates | Used seeds | Max descendants/seed |
|---|---|---|---|---|---|---|---|---|---|
| LR | Shortening | 3963.4 ± 32.8 | 2960.0 ± 36.3 | 9.0 ± 2.5 | 676.0 ± 9.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 78.2 ± 0.4 | 21.2 ± 1.3 |
| LR | Relocation | 7050.4 ± 149.2 | 5225.8 ± 109.3 | 884.2 ± 57.1 | 0.0 ± 0.0 | 495.6 ± 191.5 | 495.6 ± 191.5 | 77.8 ± 0.4 | 26.2 ± 3.1 |
| LR | Rejection | 16134.2 ± 90.1 | 11911.2 ± 71.0 | 3852.2 ± 65.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 73.8 ± 1.1 | 20.6 ± 1.5 |
| RF | Shortening | 7997.6 ± 2892.4 | 7034.4 ± 2916.9 | 145.8 ± 124.8 | 660.2 ± 23.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 103.8 ± 5.2 | 32.0 ± 5.0 |
| RF | Relocation | 9917.2 ± 1474.6 | 8931.6 ± 1277.2 | 498.8 ± 241.3 | 0.0 ± 0.0 | 304.8 ± 148.5 | 304.8 ± 148.5 | 108.4 ± 2.9 | 16.4 ± 9.9 |
| RF | Rejection | 16246.6 ± 1798.8 | 13146.0 ± 2120.3 | 2781.4 ± 551.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 96.2 ± 2.4 | 13.4 ± 3.9 |
| LGB | Shortening | 4323.2 ± 547.0 | 3238.8 ± 465.5 | 96.4 ± 95.5 | 819.0 ± 86.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 105.0 ± 3.2 | 23.6 ± 2.6 |
| LGB | Relocation | 7463.6 ± 495.1 | 5697.0 ± 517.8 | 869.6 ± 71.0 | 0.0 ± 0.0 | 420.4 ± 175.0 | 420.4 ± 175.0 | 107.0 ± 2.1 | 20.6 ± 3.4 |
| LGB | Rejection | 13702.0 ± 1346.3 | 9784.0 ± 793.4 | 3368.4 ± 656.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 96.8 ± 5.5 | 19.8 ± 3.7 |

The original max-20-iteration/stopping procedure remains fixed. Stops are 38 iteration caps, six no-valid-partner batches and one batch with no candidate satisfying constraints; no selected refit reached the 1,000-row cap. A small final requested batch may contain no compatible seed even though other seeds exist; original stopping is not silently replaced by extra retries. Step shortening still stops at its declared cap of four and never falls back to forced copies.

| Classifier | Policy | Selected (eta,rho), seeds 0–4 | Stop reasons |
|---|---|---|---|
| LR | Shortening | (1,1.5), (1,1.5), (1,1.5), (1,1.5), (1,1.5) | no_valid_partner: 4, iteration_cap: 1 |
| LR | Relocation | (0.25,0.5), (1,1.5), (1,1.5), (1,1.5), (0.5,1) | iteration_cap: 5 |
| LR | Rejection | (0.25,0.5), (0.25,0.5), (0.25,0.5), (0.25,0.5), (0.25,0.5) | iteration_cap: 5 |
| RF | Shortening | (0.25,0.5), (1,1.5), (1,1.5), (0.25,0.5), (0.25,0.5) | iteration_cap: 5 |
| RF | Relocation | (0.25,0.5), (1,1.5), (0.5,1), (0.25,0.5), (0.5,1) | iteration_cap: 3, no_candidate_satisfies_constraints: 1, no_valid_partner: 1 |
| RF | Rejection | (0.25,0.5), (0.5,1), (0.25,0.5), (0.25,0.5), (0.5,1) | iteration_cap: 5 |
| LGB | Shortening | (1,1.5), (0.25,0.5), (0.25,0.5), (0.5,1), (0.5,1) | iteration_cap: 4, no_valid_partner: 1 |
| LGB | Relocation | (1,1.5), (0.25,0.5), (1,1.5), (1,1.5), (1,1.5) | iteration_cap: 5 |
| LGB | Rejection | (0.5,1), (0.25,0.5), (0.5,1), (0.5,1), (0.5,1) | iteration_cap: 5 |

### All resources and actual budgets

Cell time includes ten-proposal selection, final refit and artifact writing. RSS is whole-process memory sampled at 10 ms, not incremental sampler memory or an exact heap peak. Torch CUDA counters cover only its allocator; reservation/heap caches carry over between cells. Shared preprocessing/U2 groups/U3 geometry are billed to first-use cells, and selected fit paths differ. These are same-invocation observations, not controlled microbenchmarks or cross-run speed claims.

| Classifier | Variant | Cell seconds | RSS MiB | Torch allocated MiB | Torch reserved MiB | Generated | Underspend |
|---|---|---|---|---|---|---|---|
| LR | Baseline | 0.31 ± 0.03 | 1639.73 ± 310.05 | 0.00 ± 0.00 | 20.80 ± 11.63 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| LR | Original UGO (U0) | 14.94 ± 0.20 | 1738.94 ± 88.29 | 12.77 ± 0.00 | 26.00 ± 0.00 | 1000.00 ± 0.00 | 0.00 ± 0.00 |
| LR | Corrected current | 14.15 ± 0.28 | 1750.56 ± 62.37 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.00 ± 0.00 | 0.00 ± 0.00 |
| LR | Kept U1 OOF MMA | 15.19 ± 0.44 | 1753.47 ± 55.87 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.00 ± 0.00 | 0.00 ± 0.00 |
| LR | Kept U2 KMeans stability | 15.52 ± 0.51 | 1753.42 ± 55.48 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.00 ± 0.00 | 0.00 ± 0.00 |
| LR | U3 shortening | 17.07 ± 0.89 | 1761.90 ± 36.17 | 11.10 ± 0.00 | 26.00 ± 0.00 | 994.40 ± 3.78 | 5.60 ± 3.78 |
| LR | U3 relocation | 17.52 ± 1.06 | 1762.92 ± 33.91 | 11.10 ± 0.00 | 26.00 ± 0.00 | 940.40 ± 11.87 | 59.60 ± 11.87 |
| LR | U3 rejection | 16.93 ± 1.09 | 1769.34 ± 27.00 | 11.10 ± 0.00 | 26.00 ± 0.00 | 370.80 ± 6.57 | 629.20 ± 6.57 |
| RF | Baseline | 20.78 ± 0.69 | 1770.25 ± 24.95 | 0.00 ± 0.00 | 26.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| RF | Original UGO (U0) | 90.01 ± 6.21 | 1787.38 ± 10.68 | 12.77 ± 0.00 | 26.00 ± 0.00 | 1000.00 ± 0.00 | 0.00 ± 0.00 |
| RF | Corrected current | 116.57 ± 9.50 | 1789.10 ± 12.75 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.00 ± 0.00 | 0.00 ± 0.00 |
| RF | Kept U1 OOF MMA | 149.31 ± 4.87 | 1789.41 ± 13.49 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.00 ± 0.00 | 0.00 ± 0.00 |
| RF | Kept U2 KMeans stability | 120.01 ± 5.94 | 1777.15 ± 8.28 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.00 ± 0.00 | 0.00 ± 0.00 |
| RF | U3 shortening | 529.56 ± 50.26 | 1795.11 ± 14.82 | 11.10 ± 0.00 | 26.00 ± 0.00 | 817.40 ± 149.52 | 182.60 ± 149.52 |
| RF | U3 relocation | 361.24 ± 41.58 | 1791.98 ± 14.70 | 11.10 ± 0.00 | 26.00 ± 0.00 | 486.80 ± 234.74 | 513.20 ± 234.74 |
| RF | U3 rejection | 463.15 ± 17.30 | 1788.86 ± 13.87 | 11.10 ± 0.00 | 26.00 ± 0.00 | 319.20 ± 148.91 | 680.80 ± 148.91 |
| LGB | Baseline | 2.73 ± 0.22 | 1789.49 ± 14.31 | 0.00 ± 0.00 | 26.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| LGB | Original UGO (U0) | 23.09 ± 0.17 | 1780.64 ± 11.77 | 12.77 ± 0.00 | 26.00 ± 0.00 | 1000.00 ± 0.00 | 0.00 ± 0.00 |
| LGB | Corrected current | 25.84 ± 0.37 | 1780.64 ± 11.77 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.00 ± 0.00 | 0.00 ± 0.00 |
| LGB | Kept U1 OOF MMA | 31.17 ± 0.37 | 1780.64 ± 11.77 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.00 ± 0.00 | 0.00 ± 0.00 |
| LGB | Kept U2 KMeans stability | 26.72 ± 0.37 | 1780.64 ± 11.77 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.00 ± 0.00 | 0.00 ± 0.00 |
| LGB | U3 shortening | 86.38 ± 5.27 | 1780.64 ± 11.78 | 11.10 ± 0.00 | 26.00 ± 0.00 | 988.00 ± 9.72 | 12.00 ± 9.72 |
| LGB | U3 relocation | 86.04 ± 6.48 | 1772.44 ± 10.42 | 11.10 ± 0.00 | 26.00 ± 0.00 | 897.00 ± 27.74 | 103.00 ± 27.74 |
| LGB | U3 rejection | 78.17 ± 1.49 | 1778.87 ± 4.96 | 11.10 ± 0.00 | 26.00 ± 0.00 | 549.60 ± 102.77 | 450.40 ± 102.77 |

Complete invocation wall time was 11,618.98 seconds (3 h 13 m 39 s), versus summed cell time 11,612.14 seconds. Peak sampled RSS was 1,810.85 MiB. Audit-observed classifier fits, including OOF scoring, total 13,238. Default RF shortening takes 529.56 versus current 116.57 mean seconds (~4.54×); LGB 86.38 versus 25.84 (~3.34×). Lower Torch allocation does not offset those observed training costs or establish lower total memory.

### Verification, interruptions and complete artifacts

All **133 repository tests passed**, including 12 independent numerical/graph/category geometry checks, eight repair/pipeline/native-budget regressions, three real-Shoppers selection/replay/leakage tests and existing CUDA behavior. The actual training-only 24-cell CLI smoke results/smoke/20261005T182550-240211b0ce26 completed; independently recomputed all predictions and checked 96 pipeline/36 constraint audits, 1,440 attempts and 114 accepted points. Its U3 outer counts 0–5 of 40 are smoke-only, not additional experimental seeds.

The preceding smoke results/smoke/20261005T182324-71e4e824cd70 was interrupted after ten completed prediction sets by the enclosing Eval cell's default deadline. Its original unfinalized manifest, source archive, completed/partial cells and interruption.json remain preserved. It is excluded from every aggregate/resource table; repeated smoke work is not independent evidence. The complete 120-cell study had no interruption and no model failures. Existing sklearn/SciPy iprint and LightGBM feature-name warnings remain unsuppressed.

Full-run proof independently recomputed every saved metric from round-trip predictions; verified all 1,320 pipeline/495 constraint audits, 4,861,611 attempts and 333,607 accepted emissions, real compatible ancestry/eligibility, exact connected groups, categories, selected-parameter replay and both initial/final real-reference bounds. Reconstructed all 120 split roles/source IDs and ten shared classifier proposal lists. There are ten actual geometry role fits (inner/outer × five seeds), but six unique real training partitions because outer training is fixed; all six unique states were independently rebuilt and all repeated snapshots checked. Geometry never incorporated validation/test rows or synthetic descendants. All numeric metrics/counts reproduced for the 75 rerun baseline/U0/current/U1/U2 references to absolute tolerance 1e-12, and all 120 canonical registry identities/summary CLI pairings were verified. Outer-test feature/category/label perturbation isolation is exercised by the permanent regression, not claimed solely from provenance.

Artifacts in results/runs/20261005T184048-c25b2afe26ae (original manifest/source/candidates/selection/outer/preprocessing/source IDs/predictions remain immutable):

- [All 120 per-seed metrics/resources](../results/runs/20261005T184048-c25b2afe26ae/all_metrics.csv), [all numeric mean/SD summaries](../results/runs/20261005T184048-c25b2afe26ae/metric_summary.csv), [all numeric per-seed deltas against baseline/U0/current/kept-U1/kept-U2](../results/runs/20261005T184048-c25b2afe26ae/paired_metric_deltas.csv).
- [All 1,320 generation/fit audits](../results/runs/20261005T184048-c25b2afe26ae/generation_audit_summary.csv), [inner/outer constraint diagnostics](../results/runs/20261005T184048-c25b2afe26ae/mechanism_diagnostics.csv), [all 45 selected outer mechanism rows](../results/runs/20261005T184048-c25b2afe26ae/outer_constraint_mechanisms.csv), [their mean/SD summaries](../results/runs/20261005T184048-c25b2afe26ae/mechanism_summary.csv), [selected settings/stops](../results/runs/20261005T184048-c25b2afe26ae/selection_constraint_summary.csv).
- [Canonical AP/count pairings](../results/runs/20261005T184048-c25b2afe26ae/paired_vs_corrected.json), [AP summaries](../results/runs/20261005T184048-c25b2afe26ae/ap_summary.json), [complete verification/resource/retention summary](../results/runs/20261005T184048-c25b2afe26ae/phase1_u3_summary.json).

**User decision: discard U3.** All three repair comparisons remain reproducible evidence, not retained components. U1/U2 stay kept; no combination, U4 progression or confirmatory evaluation is selected automatically.


## Phase 1 U4 stopping selection — manual review

**U1 and canonical U2 KMeans/stability remain kept; U3 and both U4 policies have been excluded by the user.** The historical comparison below is stopping-only against corrected_current, not U1+U2+U4. U5 testing was subsequently authorized with kept U1/U2 only, then the user discarded U5 and reaffirmed canonical U2. No U3/U4 geometry/stopping, other dataset or confirmatory progression is selected automatically.

Complete invocation: `results/runs/20261006T034029-0513bf17dc6c`. **105/105 cells completed**, no failed/unsupported cells: seven variants × LR/RF/LightGBM × seeds 0–4, with ten common learner proposals. The fixed real Shoppers split is 9,864 outer-training and 2,466 natural test rows (382 positive); preprocessing/support and selection remain training-only, and threshold is 0.5. These are development results on one fixed outer test, not five independent datasets or published-author reproduction. All 75 freshly rerun baseline/U0/current/U1/U2 reference cells reproduce the previous U3 study's numeric metrics and actual counts to absolute 1e-12.

### Change versus UGO and selection accounting

Original/corrected UGO retain their Welch difficulty-balance stopping. U4 bypasses Welch termination, retains its p-values diagnostically and compares zero/250/500/1,000 additional real-label-budgeted rows on **real inner AP**. U4_fixed_grid starts an independent generation procedure for each budget. Default U4 uses one trajectory clipped at the checkpoints; this changes batch/retraining cadence, not merely the final number of rows. Real conflict filtering, in-sample MMA, pooled-class scheduling, real interpolation and ordinary relocation are unchanged. No U1 OOF scoring, U2 restoration/allocation or discarded U3 restrictions are silently added.

Each of the ten identical learner proposals is evaluated at all four reached budgets: **40 scored settings per U4 cell versus ten reference settings**. Zero explicitly competes. Exact ties prefer lower actual augmentation, then lower requested count, then the earliest learner proposal. Checkpoint fractions mapping to duplicate integer counts are coalesced; unvisited checkpoints are not fabricated. Iteration/native/support caps can underspend without padding. Native mode retains the original phi cap rather than becoming balancing. For multiclass checkpoint replay, cumulative per-class caps use monotone incremental remaining-deficit apportionment; independent fixed budgets use original-deficit apportionment. The binary study generates only the minority class.

The selected fraction, cadence and epoch sequence are replayed on outer-training-only preprocessing/support. Outer refit performs **no checkpoint search or validation feedback**; zero fits the real-only learner once. Selected fractions apply to the outer real-label maximum, not a copied inner actual count. Saved checkpoint probabilities, audits and split roles support independent score/selection verification.

All generating reference search/refit procedures in this study generated 1,000 rows. Consequently, this dataset does not demonstrate rescue from an original/corrected Welch stop below the cap; it tests the full budget/checkpoint selection procedures. Differing selected counts and validation opportunities prevent a pure equal-count/equal-compute stopping-mechanism claim, even where an individual pair has matching counts. No outer metric chooses which U4 policy to retain.

### Average precision and primary tradeoff

All cells below are mean ± sample SD over five seeds on the same outer test; higher AP is better.

| Variant | LR AP | RF AP | LightGBM AP |
|---|---:|---:|---:|
| baseline | 0.6089 ± 0.0028 | 0.7395 ± 0.0057 | 0.7428 ± 0.0039 |
| U0 | 0.6077 ± 0.0016 | 0.7244 ± 0.0052 | 0.7302 ± 0.0033 |
| corrected_current | 0.6074 ± 0.0019 | 0.7220 ± 0.0039 | 0.7262 ± 0.0086 |
| U1_oof_mma | 0.6092 ± 0.0016 | 0.7206 ± 0.0055 | 0.7319 ± 0.0023 |
| U2 | 0.5854 ± 0.0023 | 0.7178 ± 0.0052 | 0.7281 ± 0.0054 |
| U4 | 0.6088 ± 0.0015 | 0.7395 ± 0.0057 | 0.7428 ± 0.0039 |
| U4_fixed_grid | 0.6077 ± 0.0018 | 0.7395 ± 0.0057 | 0.7428 ± 0.0039 |

- **Default checkpoint U4 versus corrected_current:** mean AP increases LR +0.0014 / RF +0.0175 / LGB +0.0166, while minority F1 changes -0.0302 / -0.0282 / -0.0130 and recall changes -0.0450 / -0.0906 / -0.0660. Balanced accuracy and MCC also decline in all three families; LGB weighted F1 improves slightly. This is a ranking/threshold tradeoff, not a uniform improvement.
- **Fixed-grid U4 versus corrected_current:** LR AP +0.0004, F1 -0.0109 and recall -0.0157. Tree outcomes are the same as default U4. Both U4 policies have lower mean minority F1/recall than corrected_current in every family.
- **All 20 U4 tree refits select zero augmentation.** Their learner proposal, every saved prediction/probability and all recorded metrics are exactly equal to their paired no-resampling baseline, independently verified. Tree AP gains versus corrected_current are therefore recovery of no-resampling performance, not improved synthetic examples. U4 and fixed-grid tree outcomes are equivalent procedures here, not independent positive confirmations. Neither U4 policy beats the no-resampling AP mean in any family.
- LR default U4 selects [500, 1,000, 1,000, 250, 250] rows for seeds 0–4 (mean 600); fixed-grid selects [1,000, 1,000, 1,000, 250, 1,000] (mean 850). All 30 selected outer budgets are achieved exactly. Choosing zero or a smaller budget is intentional selection, **not underspend of the selected budget**.

### Every predictive metric

| Variant | Classifier | average_precision | macro_average_precision | minority_f1 | minority_recall | macro_f1 | weighted_f1 | sos_inverse_support_f1 | balanced_accuracy | mcc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | logistic_regression | 0.6089 ± 0.0028 | 0.7913 ± 0.0015 | 0.4567 ± 0.0157 | 0.3340 ± 0.0201 | 0.6937 ± 0.0078 | 0.8573 ± 0.0024 | 0.5301 ± 0.0132 | 0.6553 ± 0.0085 | 0.4362 ± 0.0075 |
| baseline | random_forest | 0.7395 ± 0.0057 | 0.8616 ± 0.0030 | 0.6100 ± 0.0254 | 0.5136 ± 0.0569 | 0.7759 ± 0.0124 | 0.8905 ± 0.0034 | 0.6614 ± 0.0214 | 0.7415 ± 0.0232 | 0.5705 ± 0.0131 |
| baseline | lightgbm | 0.7428 ± 0.0039 | 0.8641 ± 0.0020 | 0.6347 ± 0.0177 | 0.5670 ± 0.0363 | 0.7881 ± 0.0091 | 0.8939 ± 0.0032 | 0.6822 ± 0.0150 | 0.7635 ± 0.0156 | 0.5834 ± 0.0144 |
| U0 | logistic_regression | 0.6077 ± 0.0016 | 0.7903 ± 0.0009 | 0.5591 ± 0.0086 | 0.4712 ± 0.0124 | 0.7465 ± 0.0045 | 0.8758 ± 0.0017 | 0.6172 ± 0.0073 | 0.7160 ± 0.0056 | 0.5073 ± 0.0072 |
| U0 | random_forest | 0.7244 ± 0.0052 | 0.8540 ± 0.0031 | 0.6456 ± 0.0081 | 0.6157 ± 0.0224 | 0.7921 ± 0.0039 | 0.8932 ± 0.0011 | 0.6910 ± 0.0068 | 0.7812 ± 0.0091 | 0.5857 ± 0.0067 |
| U0 | lightgbm | 0.7302 ± 0.0033 | 0.8576 ± 0.0017 | 0.6485 ± 0.0065 | 0.6346 ± 0.0094 | 0.7928 ± 0.0037 | 0.8925 ± 0.0018 | 0.6932 ± 0.0056 | 0.7877 ± 0.0046 | 0.5859 ± 0.0073 |
| corrected_current | logistic_regression | 0.6074 ± 0.0019 | 0.7903 ± 0.0010 | 0.5574 ± 0.0121 | 0.4670 ± 0.0152 | 0.7457 ± 0.0065 | 0.8757 ± 0.0027 | 0.6157 ± 0.0103 | 0.7144 ± 0.0073 | 0.5069 ± 0.0111 |
| corrected_current | random_forest | 0.7220 ± 0.0039 | 0.8527 ± 0.0022 | 0.6383 ± 0.0088 | 0.6042 ± 0.0239 | 0.7881 ± 0.0042 | 0.8915 ± 0.0011 | 0.6847 ± 0.0074 | 0.7757 ± 0.0097 | 0.5780 ± 0.0070 |
| corrected_current | lightgbm | 0.7262 ± 0.0086 | 0.8554 ± 0.0048 | 0.6477 ± 0.0074 | 0.6330 ± 0.0122 | 0.7924 ± 0.0041 | 0.8923 ± 0.0019 | 0.6925 ± 0.0064 | 0.7870 ± 0.0057 | 0.5852 ± 0.0081 |
| U1_oof_mma | logistic_regression | 0.6092 ± 0.0016 | 0.7913 ± 0.0009 | 0.5463 ± 0.0137 | 0.4534 ± 0.0164 | 0.7397 ± 0.0075 | 0.8732 ± 0.0031 | 0.6062 ± 0.0118 | 0.7078 ± 0.0080 | 0.4963 ± 0.0131 |
| U1_oof_mma | random_forest | 0.7206 ± 0.0055 | 0.8520 ± 0.0031 | 0.6436 ± 0.0125 | 0.6136 ± 0.0265 | 0.7909 ± 0.0065 | 0.8926 ± 0.0026 | 0.6892 ± 0.0106 | 0.7800 ± 0.0114 | 0.5833 ± 0.0121 |
| U1_oof_mma | lightgbm | 0.7319 ± 0.0023 | 0.8585 ± 0.0012 | 0.6543 ± 0.0069 | 0.6398 ± 0.0109 | 0.7963 ± 0.0038 | 0.8943 ± 0.0018 | 0.6983 ± 0.0059 | 0.7910 ± 0.0052 | 0.5929 ± 0.0076 |
| U2 | logistic_regression | 0.5854 ± 0.0023 | 0.7787 ± 0.0012 | 0.5482 ± 0.0044 | 0.4885 ± 0.0071 | 0.7379 ± 0.0023 | 0.8689 ± 0.0009 | 0.6069 ± 0.0038 | 0.7173 ± 0.0031 | 0.4819 ± 0.0040 |
| U2 | random_forest | 0.7178 ± 0.0052 | 0.8512 ± 0.0029 | 0.6527 ± 0.0075 | 0.6696 ± 0.0099 | 0.7935 ± 0.0044 | 0.8907 ± 0.0023 | 0.6963 ± 0.0065 | 0.7998 ± 0.0050 | 0.5874 ± 0.0088 |
| U2 | lightgbm | 0.7281 ± 0.0054 | 0.8567 ± 0.0027 | 0.6577 ± 0.0080 | 0.6906 ± 0.0116 | 0.7956 ± 0.0048 | 0.8908 ± 0.0026 | 0.7004 ± 0.0070 | 0.8078 ± 0.0056 | 0.5924 ± 0.0096 |
| U4 | logistic_regression | 0.6088 ± 0.0015 | 0.7911 ± 0.0007 | 0.5272 ± 0.0225 | 0.4220 ± 0.0343 | 0.7302 ± 0.0115 | 0.8704 ± 0.0039 | 0.5901 ± 0.0191 | 0.6948 ± 0.0148 | 0.4859 ± 0.0152 |
| U4 | random_forest | 0.7395 ± 0.0057 | 0.8616 ± 0.0030 | 0.6100 ± 0.0254 | 0.5136 ± 0.0569 | 0.7759 ± 0.0124 | 0.8905 ± 0.0034 | 0.6614 ± 0.0214 | 0.7415 ± 0.0232 | 0.5705 ± 0.0131 |
| U4 | lightgbm | 0.7428 ± 0.0039 | 0.8641 ± 0.0020 | 0.6347 ± 0.0177 | 0.5670 ± 0.0363 | 0.7881 ± 0.0091 | 0.8939 ± 0.0032 | 0.6822 ± 0.0150 | 0.7635 ± 0.0156 | 0.5834 ± 0.0144 |
| U4_fixed_grid | logistic_regression | 0.6077 ± 0.0018 | 0.7905 ± 0.0010 | 0.5465 ± 0.0238 | 0.4513 ± 0.0346 | 0.7401 ± 0.0123 | 0.8737 ± 0.0044 | 0.6065 ± 0.0202 | 0.7075 ± 0.0152 | 0.4990 ± 0.0174 |
| U4_fixed_grid | random_forest | 0.7395 ± 0.0057 | 0.8616 ± 0.0030 | 0.6100 ± 0.0254 | 0.5136 ± 0.0569 | 0.7759 ± 0.0124 | 0.8905 ± 0.0034 | 0.6614 ± 0.0214 | 0.7415 ± 0.0232 | 0.5705 ± 0.0131 |
| U4_fixed_grid | lightgbm | 0.7428 ± 0.0039 | 0.8641 ± 0.0020 | 0.6347 ± 0.0177 | 0.5670 ± 0.0363 | 0.7881 ± 0.0091 | 0.8939 ± 0.0032 | 0.6822 ± 0.0150 | 0.7635 ± 0.0156 | 0.5834 ± 0.0144 |

### Paired differences versus corrected_current

Each entry is the mean ± sample SD of the five within-seed differences; do not subtract rounded summary means to reconstruct these values. Additional paired deltas against baseline/U0/kept-U1/kept-U2 are linked below.

| Variant | Classifier | average_precision | macro_average_precision | minority_f1 | minority_recall | macro_f1 | weighted_f1 | sos_inverse_support_f1 | balanced_accuracy | mcc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U4 | logistic_regression | +0.0014 ± 0.0022 | +0.0009 ± 0.0011 | -0.0302 ± 0.0235 | -0.0450 ± 0.0336 | -0.0155 ± 0.0122 | -0.0053 ± 0.0044 | -0.0257 ± 0.0200 | -0.0196 ± 0.0148 | -0.0210 ± 0.0177 |
| U4 | random_forest | +0.0175 ± 0.0087 | +0.0088 ± 0.0048 | -0.0282 ± 0.0198 | -0.0906 ± 0.0444 | -0.0121 ± 0.0096 | -0.0010 ± 0.0026 | -0.0233 ± 0.0166 | -0.0341 ± 0.0180 | -0.0074 ± 0.0097 |
| U4 | lightgbm | +0.0166 ± 0.0088 | +0.0087 ± 0.0049 | -0.0130 ± 0.0222 | -0.0660 ± 0.0419 | -0.0044 ± 0.0117 | +0.0016 ± 0.0044 | -0.0103 ± 0.0189 | -0.0235 ± 0.0186 | -0.0017 ± 0.0195 |
| U4_fixed_grid | logistic_regression | +0.0004 ± 0.0008 | +0.0002 ± 0.0005 | -0.0109 ± 0.0244 | -0.0157 ± 0.0351 | -0.0056 ± 0.0126 | -0.0020 ± 0.0045 | -0.0093 ± 0.0208 | -0.0069 ± 0.0155 | -0.0079 ± 0.0177 |
| U4_fixed_grid | random_forest | +0.0175 ± 0.0087 | +0.0088 ± 0.0048 | -0.0282 ± 0.0198 | -0.0906 ± 0.0444 | -0.0121 ± 0.0096 | -0.0010 ± 0.0026 | -0.0233 ± 0.0166 | -0.0341 ± 0.0180 | -0.0074 ± 0.0097 |
| U4_fixed_grid | lightgbm | +0.0166 ± 0.0088 | +0.0087 ± 0.0049 | -0.0130 ± 0.0222 | -0.0660 ± 0.0419 | -0.0044 ± 0.0117 | +0.0016 ± 0.0044 | -0.0103 ± 0.0189 | -0.0235 ± 0.0186 | -0.0017 ± 0.0195 |

### Operating points, confusion counts and cost

realized_recall_test/realized_fpr_test use the fixed deployed threshold 0.5. recall_at_fpr_test/roc_attained_fpr_test describe attainable test ROC points at the protocol's 0.05 FPR limit; they never select a deployable threshold. cost is the existing diagnostic 10*FP + 500*FN, not a Shoppers selection objective or an APS experiment.

| Variant | Classifier | recall_at_fpr_test | roc_attained_fpr_test | realized_recall_test | realized_fpr_test | fp | fn | tp | tn | cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | logistic_regression | 0.5162 ± 0.0043 | 0.0496 ± 0.0003 | 0.3340 ± 0.0201 | 0.0234 ± 0.0032 | 48.8000 ± 6.5727 | 254.4000 ± 7.6681 | 127.6000 ± 7.6681 | 2035.2000 ± 6.5727 | 127688.0000 ± 3768.3312 |
| baseline | random_forest | 0.6173 ± 0.0079 | 0.0497 ± 0.0003 | 0.5136 ± 0.0569 | 0.0305 ± 0.0105 | 63.6000 ± 21.9499 | 185.8000 ± 21.7417 | 196.2000 ± 21.7417 | 2020.4000 ± 21.9499 | 93536.0000 ± 10651.6280 |
| baseline | lightgbm | 0.6152 ± 0.0052 | 0.0494 ± 0.0006 | 0.5670 ± 0.0363 | 0.0400 ± 0.0053 | 83.4000 ± 11.0815 | 165.4000 ± 13.8672 | 216.6000 ± 13.8672 | 2000.6000 ± 11.0815 | 83534.0000 ± 6826.0735 |
| U0 | logistic_regression | 0.5105 ± 0.0037 | 0.0495 ± 0.0002 | 0.4712 ± 0.0124 | 0.0393 ± 0.0014 | 81.8000 ± 2.8636 | 202.0000 ± 4.7434 | 180.0000 ± 4.7434 | 2002.2000 ± 2.8636 | 101818.0000 ± 2346.4484 |
| U0 | random_forest | 0.5958 ± 0.0080 | 0.0497 ± 0.0004 | 0.6157 ± 0.0224 | 0.0534 ± 0.0044 | 111.2000 ± 9.0940 | 146.8000 ± 8.5557 | 235.2000 ± 8.5557 | 1972.8000 ± 9.0940 | 74512.0000 ± 4188.2299 |
| U0 | lightgbm | 0.5963 ± 0.0104 | 0.0496 ± 0.0004 | 0.6346 ± 0.0094 | 0.0591 ± 0.0012 | 123.2000 ± 2.5884 | 139.6000 ± 3.5777 | 242.4000 ± 3.5777 | 1960.8000 ± 2.5884 | 71032.0000 ± 1785.1246 |
| corrected_current | logistic_regression | 0.5126 ± 0.0039 | 0.0497 ± 0.0004 | 0.4670 ± 0.0152 | 0.0382 ± 0.0009 | 79.6000 ± 1.8166 | 203.6000 ± 5.8138 | 178.4000 ± 5.8138 | 2004.4000 ± 1.8166 | 102596.0000 ± 2892.3745 |
| corrected_current | random_forest | 0.5974 ± 0.0062 | 0.0498 ± 0.0002 | 0.6042 ± 0.0239 | 0.0529 ± 0.0047 | 110.2000 ± 9.7826 | 151.2000 ± 9.1488 | 230.8000 ± 9.1488 | 1973.8000 ± 9.7826 | 76702.0000 ± 4477.6188 |
| corrected_current | lightgbm | 0.5796 ± 0.0066 | 0.0496 ± 0.0004 | 0.6330 ± 0.0122 | 0.0589 ± 0.0015 | 122.8000 ± 3.1145 | 140.2000 ± 4.6583 | 241.8000 ± 4.6583 | 1961.2000 ± 3.1145 | 71328.0000 ± 2312.7840 |
| U1_oof_mma | logistic_regression | 0.5168 ± 0.0035 | 0.0498 ± 0.0002 | 0.4534 ± 0.0164 | 0.0378 ± 0.0004 | 78.8000 ± 0.8367 | 208.8000 ± 6.2610 | 173.2000 ± 6.2610 | 2005.2000 ± 0.8367 | 105188.0000 ± 3123.6309 |
| U1_oof_mma | random_forest | 0.5927 ± 0.0071 | 0.0492 ± 0.0007 | 0.6136 ± 0.0265 | 0.0536 ± 0.0044 | 111.8000 ± 9.0940 | 147.6000 ± 10.1390 | 234.4000 ± 10.1390 | 1972.2000 ± 9.0940 | 74918.0000 ± 4990.2174 |
| U1_oof_mma | lightgbm | 0.5890 ± 0.0158 | 0.0498 ± 0.0002 | 0.6398 ± 0.0109 | 0.0579 ± 0.0014 | 120.6000 ± 2.8810 | 137.6000 ± 4.1593 | 244.4000 ± 4.1593 | 1963.4000 ± 2.8810 | 70006.0000 ± 2067.9289 |
| U2 | logistic_regression | 0.4738 ± 0.0045 | 0.0490 ± 0.0008 | 0.4885 ± 0.0071 | 0.0538 ± 0.0009 | 112.2000 ± 1.9235 | 195.4000 ± 2.7019 | 186.6000 ± 2.7019 | 1971.8000 ± 1.9235 | 98822.0000 ± 1333.9303 |
| U2 | random_forest | 0.6026 ± 0.0129 | 0.0495 ± 0.0006 | 0.6696 ± 0.0099 | 0.0701 ± 0.0022 | 146.0000 ± 4.6368 | 126.2000 ± 3.7683 | 255.8000 ± 3.7683 | 1938.0000 ± 4.6368 | 64560.0000 ± 1883.3879 |
| U2 | lightgbm | 0.6047 ± 0.0052 | 0.0496 ± 0.0006 | 0.6906 ± 0.0116 | 0.0750 ± 0.0032 | 156.4000 ± 6.6558 | 118.2000 ± 4.4385 | 263.8000 ± 4.4385 | 1927.6000 ± 6.6558 | 60664.0000 ± 2203.0502 |
| U4 | logistic_regression | 0.5147 ± 0.0060 | 0.0495 ± 0.0002 | 0.4220 ± 0.0343 | 0.0324 ± 0.0049 | 67.6000 ± 10.1143 | 220.8000 ± 13.0843 | 161.2000 ± 13.0843 | 2016.4000 ± 10.1143 | 111076.0000 ± 6443.1537 |
| U4 | random_forest | 0.6173 ± 0.0079 | 0.0497 ± 0.0003 | 0.5136 ± 0.0569 | 0.0305 ± 0.0105 | 63.6000 ± 21.9499 | 185.8000 ± 21.7417 | 196.2000 ± 21.7417 | 2020.4000 ± 21.9499 | 93536.0000 ± 10651.6280 |
| U4 | lightgbm | 0.6152 ± 0.0052 | 0.0494 ± 0.0006 | 0.5670 ± 0.0363 | 0.0400 ± 0.0053 | 83.4000 ± 11.0815 | 165.4000 ± 13.8672 | 216.6000 ± 13.8672 | 2000.6000 ± 11.0815 | 83534.0000 ± 6826.0735 |
| U4_fixed_grid | logistic_regression | 0.5157 ± 0.0052 | 0.0496 ± 0.0004 | 0.4513 ± 0.0346 | 0.0364 ± 0.0043 | 75.8000 ± 9.0111 | 209.6000 ± 13.2212 | 172.4000 ± 13.2212 | 2008.2000 ± 9.0111 | 105558.0000 ± 6523.9651 |
| U4_fixed_grid | random_forest | 0.6173 ± 0.0079 | 0.0497 ± 0.0003 | 0.5136 ± 0.0569 | 0.0305 ± 0.0105 | 63.6000 ± 21.9499 | 185.8000 ± 21.7417 | 196.2000 ± 21.7417 | 2020.4000 ± 21.9499 | 93536.0000 ± 10651.6280 |
| U4_fixed_grid | lightgbm | 0.6152 ± 0.0052 | 0.0494 ± 0.0006 | 0.5670 ± 0.0363 | 0.0400 ± 0.0053 | 83.4000 ± 11.0815 | 165.4000 ± 13.8672 | 216.6000 ± 13.8672 | 2000.6000 ± 11.0815 | 83534.0000 ± 6826.0735 |

### Selection by seed

Both policies select zero for every RF/LightGBM seed. The LR choices below are the actual inner-selected procedures, not outer-score winners.

| Seed | U4 learner proposal | U4 selected rows | Fixed-grid learner proposal | Fixed-grid selected rows |
|---|---:|---:|---:|---:|
| 0 | 4 | 500 | 4 | 1,000 |
| 1 | 1 | 1,000 | 1 | 1,000 |
| 2 | 4 | 1,000 | 1 | 1,000 |
| 3 | 1 | 250 | 1 | 250 |
| 4 | 1 | 250 | 1 | 1,000 |

### Resources and actual counts

Cell time includes all ten learner proposals, every U4 budget/checkpoint evaluation, generation/retraining, selected outer refit and artifact writing. Whole-invocation wall time is **6,003.33 seconds (1 h 40 m 03 s)**; summed cell time is 6,001.44 seconds. Sampled peak process RSS is 1,782.13 MiB. The experiment performed **4,984 classifier fits**, including U1's real-only three-fold scoring fits, and generated 1,079,750 rows across search trajectories and refits (not distinct final-training rows).

| Variant | Classifier | runtime_seconds | peak_rss_MiB | cuda_peak_torch_allocated_MiB | cuda_peak_torch_reserved_MiB | actual_count | requested_count | underspend |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | logistic_regression | 0.3677 ± 0.0313 | 1609.2398 ± 293.6677 | 0.0000 ± 0.0000 | 20.8000 ± 11.6276 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| baseline | random_forest | 21.9678 ± 0.7981 | 1728.7477 ± 26.9339 | 0.0000 ± 0.0000 | 26.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| baseline | lightgbm | 2.8931 ± 0.2846 | 1741.3211 ± 4.3590 | 0.0000 ± 0.0000 | 26.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| U0 | logistic_regression | 15.8451 ± 0.4723 | 1708.8180 ± 71.3725 | 12.7739 ± 0.0000 | 26.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| U0 | random_forest | 93.8873 ± 5.7648 | 1745.6656 ± 6.8662 | 12.7739 ± 0.0000 | 26.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| U0 | lightgbm | 23.7944 ± 0.2261 | 1741.5680 ± 4.0355 | 12.7739 ± 0.0000 | 26.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| corrected_current | logistic_regression | 16.2327 ± 0.7131 | 1720.3008 ± 45.7372 | 15.6689 ± 0.0000 | 26.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| corrected_current | random_forest | 121.1892 ± 9.4695 | 1747.3680 ± 8.6941 | 15.6689 ± 0.0000 | 26.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| corrected_current | lightgbm | 27.1870 ± 0.4008 | 1741.5680 ± 4.0355 | 15.6689 ± 0.0000 | 26.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| U1_oof_mma | logistic_regression | 17.1919 ± 0.4211 | 1723.3531 ± 38.9342 | 15.6689 ± 0.0000 | 26.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| U1_oof_mma | random_forest | 154.3238 ± 4.6407 | 1747.8109 ± 9.6438 | 15.6689 ± 0.0000 | 26.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| U1_oof_mma | lightgbm | 32.9634 ± 0.2492 | 1741.5680 ± 4.0355 | 15.6689 ± 0.0000 | 26.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| U2 | logistic_regression | 17.2250 ± 0.4366 | 1723.4820 ± 38.6470 | 15.6787 ± 0.0000 | 26.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| U2 | random_forest | 124.4347 ± 7.1806 | 1741.7219 ± 14.8973 | 15.6787 ± 0.0000 | 26.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| U2 | lightgbm | 28.2129 ± 0.1880 | 1741.5680 ± 4.0355 | 15.6787 ± 0.0000 | 26.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 1000.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| U4 | logistic_regression | 16.6644 ± 0.5819 | 1723.4828 ± 38.6473 | 15.6689 ± 0.0000 | 26.0000 ± 0.0000 | 600.0000 ± 379.1438 | 600.0000 ± 379.1438 | 0.0000 ± 0.0000 |
| U4 | random_forest | 104.9299 ± 1.3178 | 1746.2305 ± 9.7994 | 15.6689 ± 0.0000 | 26.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| U4 | lightgbm | 28.1719 ± 0.4635 | 1741.5680 ± 4.0355 | 15.6689 ± 0.0000 | 26.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| U4_fixed_grid | logistic_regression | 54.4014 ± 1.1755 | 1723.4891 ± 38.6334 | 15.6689 ± 0.0000 | 26.0000 ± 0.0000 | 850.0000 ± 335.4102 | 850.0000 ± 335.4102 | 0.0000 ± 0.0000 |
| U4_fixed_grid | random_forest | 223.5161 ± 2.7648 | 1779.4539 ± 3.8764 | 15.6689 ± 0.0000 | 26.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| U4_fixed_grid | lightgbm | 74.8890 ± 0.6335 | 1741.5680 ± 4.0355 | 15.6689 ± 0.0000 | 26.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

Totals over all five seeds, including search/refit work:

| Variant | Classifier | classifier_fit_count | total_generated_in_search_and_refits | pipeline_count |
|---|---|---:|---:|---:|
| baseline | logistic_regression | 55 | 0 | 55 |
| baseline | random_forest | 55 | 0 | 55 |
| baseline | lightgbm | 55 | 0 | 55 |
| U0 | logistic_regression | 165 | 55000 | 55 |
| U0 | random_forest | 165 | 55000 | 55 |
| U0 | lightgbm | 165 | 55000 | 55 |
| corrected_current | logistic_regression | 220 | 55000 | 55 |
| corrected_current | random_forest | 220 | 55000 | 55 |
| corrected_current | lightgbm | 220 | 55000 | 55 |
| U1_oof_mma | logistic_regression | 385 | 55000 | 55 |
| U1_oof_mma | random_forest | 385 | 55000 | 55 |
| U1_oof_mma | lightgbm | 385 | 55000 | 55 |
| U2 | logistic_regression | 220 | 55000 | 55 |
| U2 | random_forest | 220 | 55000 | 55 |
| U2 | lightgbm | 220 | 55000 | 55 |
| U4 | logistic_regression | 215 | 53000 | 55 |
| U4 | random_forest | 205 | 50000 | 55 |
| U4 | lightgbm | 205 | 50000 | 55 |
| U4_fixed_grid | logistic_regression | 414 | 91750 | 205 |
| U4_fixed_grid | random_forest | 405 | 87500 | 205 |
| U4_fixed_grid | lightgbm | 405 | 87500 | 205 |

Default checkpoint RF/LGB cell time is 104.93/28.17 seconds versus baseline 21.97/2.89: **4.78×/9.74× cost for identical final baseline predictions**. Fixed-grid costs 223.52/74.89 seconds, or 10.17×/25.89× baseline. Relative to corrected_current, default LR/RF/LGB time ratios are 1.03×/0.87×/1.04×; fixed-grid ratios are 3.35×/1.84×/2.75×. Four checkpoint scores can reuse generation fits, so more validation opportunities do not necessarily mean more classifier fits than the reference procedure. These are observed full-procedure times, not isolated cross-run speed claims.

RSS and Torch reserved memory can carry over from earlier cells; shared real partitions/group preparation are billed at first use. Torch allocation counters are not total GPU/library memory. Baseline rows can show reserved GPU memory while allocating none. No controlled microbenchmark or memory superiority claim is made.

### Verification and complete artifacts

- All **149 repository tests** passed; 16 new U4 behavior/boundary tests remain after removing one incidental default/forwarding test. They exercise zero ties, equal-difficulty Welch bypass, real ancestry/category donors, native/iteration caps, exact multiclass prefix replay through the Alabama paradox, real Shoppers held-out numeric/category/label perturbations, independently refitted predictions and capped outer-budget replay. Existing CPU/CUDA regressions also passed.
- The actual 21-cell training-only CLI smoke at `results/smoke/20261006T033538-db0233b6c428` completed. Independent checks verified 21 prediction sets, 111 pipeline audits, 72 checkpoint metric sets and 2,331 numeric values, plus source/split/proposal/index provenance. Its original test snapshot is preserved; the only later test-only cleanup removed the incidental forwarding/default test. Smoke AP is not experimental evidence.
- Full study verification reconstructed all five ordered real inner/outer splits, checked all 105 canonical index identities and ten shared proposal arrays, matched all seven changed implementation/test files to source.zip, verified **1,605 pipeline audits**, and recomputed every numeric outcome for **105 outer prediction sets and 1,200 checkpoint probability sets**: 33,705 numeric metric checks at absolute 1e-12. Prediction CSV parsing uses float_precision=round_trip to preserve ranking ties. All selected budgets, support bounds, provenance counts, source scopes, learner/checkpoint tie precedence and exact selected-procedure refits passed audit.
- All 75 reference cells reproduce 1,575 previously recorded numeric metric values and actual counts at absolute 1e-12. All 20 tree procedure/baseline prediction, metric and learner equivalences were verified exactly. No model failure or interrupted experiment occurred. Existing sklearn/SciPy iprint and LightGBM feature-name warnings were left unsuppressed.
- Execution-time manual decisions were U1/U2 keep, U3 discard, U4/U5 pending; U5/W5/R5 were unavailable. Subsequently the user excluded both U4 policies and separately authorized U5 testing with kept U1/U2 only, reserving the inclusion decision. After reviewing the completed U5 comparison, the user discarded U5 and reaffirmed canonical U2. U1 remains kept; the live study remains not_frozen and confirmatory evaluation remains refused. Historical study/source snapshots are not rewritten.
- [All 105 per-seed metrics/resources](../results/runs/20261006T034029-0513bf17dc6c/all_metrics.csv), [all 1,134 numeric mean/sample-SD summaries](../results/runs/20261006T034029-0513bf17dc6c/metric_summary.csv), [all 12,150 numeric per-seed deltas against five references](../results/runs/20261006T034029-0513bf17dc6c/paired_metric_deltas.csv), [paired mean/SD versus corrected_current](../results/runs/20261006T034029-0513bf17dc6c/paired_summary_vs_corrected.csv).
- [All 1,200 checkpoint metric sets](../results/runs/20261006T034029-0513bf17dc6c/checkpoint_metrics.csv), [actual learner/checkpoint selections](../results/runs/20261006T034029-0513bf17dc6c/stopping_selection.csv), [all 1,605 pipeline audits](../results/runs/20261006T034029-0513bf17dc6c/generation_audit_summary.csv), [fit/generation work totals](../results/runs/20261006T034029-0513bf17dc6c/work_summary.csv).
- [Exact baseline-equivalence and runtime diagnostics](../results/runs/20261006T034029-0513bf17dc6c/procedure_diagnostics.json), [prediction/checkpoint/budget verification](../results/runs/20261006T034029-0513bf17dc6c/verification_summary.json), [source/split/index/reference provenance](../results/runs/20261006T034029-0513bf17dc6c/provenance_verification.json), [publication verification](../results/runs/20261006T034029-0513bf17dc6c/publication_verification.json), [canonical AP summaries](../results/runs/20261006T034029-0513bf17dc6c/ap_summary.json), [canonical AP pairings](../results/runs/20261006T034029-0513bf17dc6c/paired_vs_corrected.json).

**User decision: exclude U4, including checkpoint selection and the independent fixed grid.** The user separately authorized testing U5 with kept U1/U2 only; the user subsequently discarded U5 and reaffirmed canonical U2. No retention, freeze or confirmatory evaluation is selected automatically.



## Phase 1 U5 kept-components combination — manual review

**The user discarded U5 after reviewing every metric and reaffirmed canonical U2 KMeans/stability as the preferred candidate. U1_oof_mma remains kept; U3/U4/U5 are excluded.** The historical U5 comparison and immutable execution-time snapshots (which recorded U5 pending) are preserved below. No W5/R5, other dataset, configuration freeze or confirmatory evaluation is enabled by this decision.

Run: `results/runs/20261006T060142-8e042c6ffcf0`. Completed **90/90 cells**, six variants × three classifier families × five seeds (0–4), with no failed cells or interrupted run. Every cell uses the same real 9,864-row outer training partition and 2,466-row test (382 positives), and seed-matched real inner partitions. The five seeds share that fixed test: results below are descriptive mean ± sample SD, not five independent datasets, confidence intervals or significance claims. All 75 freshly rerun reference cells reproduce the previous U4-study metrics and actual counts, at absolute metric tolerance 1e-12; old study/source snapshots remain unchanged.

### Composition, selection and fairness

- U5 (ugo_combined) uses the existing U1 scorer: three real-only, same-family OOF models per learner proposal/refit, each with raw preprocessing fitted on its own fold training rows. Every real row is scored exactly once while excluded from its scoring model's training. MMA scores are fixed before generation; synthetic rows never train the scorer or grouping.
- The existing U2 support/allocation uses real-only KMeans/subsampling stability, restores supported conflicting minority islands, weights class-wise score percentiles by stability and inverse group size, and caps descendants at floor(2*n_g) per real group and four per real seed. Generated rows are attributed to the source seed group, not claimed to lie inside that group's spatial geometry. Partners/ordinary relocation remain eligible same-class real neighbors, not group-restricted U3 geometry.
- Corrected interpolation, real one-hot category donors, ordinary relocation and the original Welch stopping remain in force. No U3 component/ratio/radius constraints, U4 checkpoint/fixed-grid selection or zero-budget policy enters U5. The unchanged Welch boundary is behaviorally tested; this study reached all requested budgets.
- The same ten learner proposals compete for every variant. U2/U5 jointly cycle alpha=0/0.5/1 over those ten slots (4/3/3 settings), select by real inner AP, and replay the selected learner/alpha/epoch sequence on outer training. Outer/test results never select the alpha. Threshold is fixed at 0.5.
- All 75 generation cells emit exactly the requested 1,000 minority rows, with zero underspend, versus zero for baseline. Per-class actual counts match among generation variants. The three additional OOF fits per proposal/refit are recorded: equal realized generation counts and learner-proposal opportunities are **not equal compute**. Prepared real grouping/OOF preprocessing is shared within training-role/seed scopes; OOF learners are fitted anew.

### All predictive metrics

All values are mean ± sample SD across the five seeds. Inverse-support F1 is the diagnostic SOS inverse-class-support weighted F1, not a published SOS aggregation claim.

| Variant | Family | AP | Macro AP | Minority F1 | Minority recall | Macro F1 | Weighted F1 | Inverse-support F1 | BAC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | LR | 0.6089 ± 0.0028 | 0.7913 ± 0.0015 | 0.4567 ± 0.0157 | 0.3340 ± 0.0201 | 0.6937 ± 0.0078 | 0.8573 ± 0.0024 | 0.5301 ± 0.0132 | 0.6553 ± 0.0085 | 0.4362 ± 0.0075 |
| U0 | LR | 0.6077 ± 0.0016 | 0.7903 ± 0.0009 | 0.5591 ± 0.0086 | 0.4712 ± 0.0124 | 0.7465 ± 0.0045 | 0.8758 ± 0.0017 | 0.6172 ± 0.0073 | 0.7160 ± 0.0056 | 0.5073 ± 0.0072 |
| corrected_current | LR | 0.6074 ± 0.0019 | 0.7903 ± 0.0010 | 0.5574 ± 0.0121 | 0.4670 ± 0.0152 | 0.7457 ± 0.0065 | 0.8757 ± 0.0027 | 0.6157 ± 0.0103 | 0.7144 ± 0.0073 | 0.5069 ± 0.0111 |
| U1_oof_mma | LR | 0.6092 ± 0.0016 | 0.7913 ± 0.0009 | 0.5463 ± 0.0137 | 0.4534 ± 0.0164 | 0.7397 ± 0.0075 | 0.8732 ± 0.0031 | 0.6062 ± 0.0118 | 0.7078 ± 0.0080 | 0.4963 ± 0.0131 |
| U2 | LR | 0.5854 ± 0.0023 | 0.7787 ± 0.0012 | 0.5482 ± 0.0044 | 0.4885 ± 0.0071 | 0.7379 ± 0.0023 | 0.8689 ± 0.0009 | 0.6069 ± 0.0038 | 0.7173 ± 0.0031 | 0.4819 ± 0.0040 |
| U5 | LR | 0.5854 ± 0.0015 | 0.7788 ± 0.0009 | 0.5448 ± 0.0022 | 0.4838 ± 0.0050 | 0.7361 ± 0.0011 | 0.8681 ± 0.0006 | 0.6041 ± 0.0018 | 0.7151 ± 0.0018 | 0.4787 ± 0.0021 |
| baseline | RF | 0.7395 ± 0.0057 | 0.8616 ± 0.0030 | 0.6100 ± 0.0254 | 0.5136 ± 0.0569 | 0.7759 ± 0.0124 | 0.8905 ± 0.0034 | 0.6614 ± 0.0214 | 0.7415 ± 0.0232 | 0.5705 ± 0.0131 |
| U0 | RF | 0.7244 ± 0.0052 | 0.8540 ± 0.0031 | 0.6456 ± 0.0081 | 0.6157 ± 0.0224 | 0.7921 ± 0.0039 | 0.8932 ± 0.0011 | 0.6910 ± 0.0068 | 0.7812 ± 0.0091 | 0.5857 ± 0.0067 |
| corrected_current | RF | 0.7220 ± 0.0039 | 0.8527 ± 0.0022 | 0.6383 ± 0.0088 | 0.6042 ± 0.0239 | 0.7881 ± 0.0042 | 0.8915 ± 0.0011 | 0.6847 ± 0.0074 | 0.7757 ± 0.0097 | 0.5780 ± 0.0070 |
| U1_oof_mma | RF | 0.7206 ± 0.0055 | 0.8520 ± 0.0031 | 0.6436 ± 0.0125 | 0.6136 ± 0.0265 | 0.7909 ± 0.0065 | 0.8926 ± 0.0026 | 0.6892 ± 0.0106 | 0.7800 ± 0.0114 | 0.5833 ± 0.0121 |
| U2 | RF | 0.7178 ± 0.0052 | 0.8512 ± 0.0029 | 0.6527 ± 0.0075 | 0.6696 ± 0.0099 | 0.7935 ± 0.0044 | 0.8907 ± 0.0023 | 0.6963 ± 0.0065 | 0.7998 ± 0.0050 | 0.5874 ± 0.0088 |
| U5 | RF | 0.7176 ± 0.0071 | 0.8509 ± 0.0041 | 0.6471 ± 0.0130 | 0.6623 ± 0.0311 | 0.7904 ± 0.0068 | 0.8892 ± 0.0027 | 0.6915 ± 0.0111 | 0.7960 ± 0.0131 | 0.5813 ± 0.0135 |
| baseline | LGB | 0.7428 ± 0.0039 | 0.8641 ± 0.0020 | 0.6347 ± 0.0177 | 0.5670 ± 0.0363 | 0.7881 ± 0.0091 | 0.8939 ± 0.0032 | 0.6822 ± 0.0150 | 0.7635 ± 0.0156 | 0.5834 ± 0.0144 |
| U0 | LGB | 0.7302 ± 0.0033 | 0.8576 ± 0.0017 | 0.6485 ± 0.0065 | 0.6346 ± 0.0094 | 0.7928 ± 0.0037 | 0.8925 ± 0.0018 | 0.6932 ± 0.0056 | 0.7877 ± 0.0046 | 0.5859 ± 0.0073 |
| corrected_current | LGB | 0.7262 ± 0.0086 | 0.8554 ± 0.0048 | 0.6477 ± 0.0074 | 0.6330 ± 0.0122 | 0.7924 ± 0.0041 | 0.8923 ± 0.0019 | 0.6925 ± 0.0064 | 0.7870 ± 0.0057 | 0.5852 ± 0.0081 |
| U1_oof_mma | LGB | 0.7319 ± 0.0023 | 0.8585 ± 0.0012 | 0.6543 ± 0.0069 | 0.6398 ± 0.0109 | 0.7963 ± 0.0038 | 0.8943 ± 0.0018 | 0.6983 ± 0.0059 | 0.7910 ± 0.0052 | 0.5929 ± 0.0076 |
| U2 | LGB | 0.7281 ± 0.0054 | 0.8567 ± 0.0027 | 0.6577 ± 0.0080 | 0.6906 ± 0.0116 | 0.7956 ± 0.0048 | 0.8908 ± 0.0026 | 0.7004 ± 0.0070 | 0.8078 ± 0.0056 | 0.5924 ± 0.0096 |
| U5 | LGB | 0.7296 ± 0.0043 | 0.8574 ± 0.0022 | 0.6573 ± 0.0045 | 0.6838 ± 0.0077 | 0.7957 ± 0.0027 | 0.8913 ± 0.0015 | 0.7002 ± 0.0039 | 0.8055 ± 0.0034 | 0.5922 ± 0.0054 |

### Confusion, diagnostic cost and operating points

Mean count entries average five predictions on the same 382-positive/2,084-negative test. The 10FP+500FN cost is a diagnostic, **not the Shoppers selection objective** or an established business cost. Recall at FPR≤0.05 is the maximum attainable test ROC recall with tied scores intact and no interpolation; it is not a deployable threshold selected on test. The attained FPR is shown separately. Deployed threshold-0.5 recall equals minority recall above; deployed FPR is the last column.

| Variant | Family | TP | FP | FN | TN | 10FP+500FN | Recall @ FPR≤.05 | Attained FPR | FPR @ .5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | LR | 127.6 ± 7.7 | 48.8 ± 6.6 | 254.4 ± 7.7 | 2035.2 ± 6.6 | 127688.0 ± 3768.3 | 0.5162 ± 0.0043 | 0.0496 ± 0.0003 | 0.0234 ± 0.0032 |
| U0 | LR | 180.0 ± 4.7 | 81.8 ± 2.9 | 202.0 ± 4.7 | 2002.2 ± 2.9 | 101818.0 ± 2346.4 | 0.5105 ± 0.0037 | 0.0495 ± 0.0002 | 0.0393 ± 0.0014 |
| corrected_current | LR | 178.4 ± 5.8 | 79.6 ± 1.8 | 203.6 ± 5.8 | 2004.4 ± 1.8 | 102596.0 ± 2892.4 | 0.5126 ± 0.0039 | 0.0497 ± 0.0004 | 0.0382 ± 0.0009 |
| U1_oof_mma | LR | 173.2 ± 6.3 | 78.8 ± 0.8 | 208.8 ± 6.3 | 2005.2 ± 0.8 | 105188.0 ± 3123.6 | 0.5168 ± 0.0035 | 0.0498 ± 0.0002 | 0.0378 ± 0.0004 |
| U2 | LR | 186.6 ± 2.7 | 112.2 ± 1.9 | 195.4 ± 2.7 | 1971.8 ± 1.9 | 98822.0 ± 1333.9 | 0.4738 ± 0.0045 | 0.0490 ± 0.0008 | 0.0538 ± 0.0009 |
| U5 | LR | 184.8 ± 1.9 | 111.6 ± 3.5 | 197.2 ± 1.9 | 1972.4 ± 3.5 | 99716.0 ± 931.3 | 0.4717 ± 0.0034 | 0.0492 ± 0.0009 | 0.0536 ± 0.0017 |
| baseline | RF | 196.2 ± 21.7 | 63.6 ± 21.9 | 185.8 ± 21.7 | 2020.4 ± 21.9 | 93536.0 ± 10651.6 | 0.6173 ± 0.0079 | 0.0497 ± 0.0003 | 0.0305 ± 0.0105 |
| U0 | RF | 235.2 ± 8.6 | 111.2 ± 9.1 | 146.8 ± 8.6 | 1972.8 ± 9.1 | 74512.0 ± 4188.2 | 0.5958 ± 0.0080 | 0.0497 ± 0.0004 | 0.0534 ± 0.0044 |
| corrected_current | RF | 230.8 ± 9.1 | 110.2 ± 9.8 | 151.2 ± 9.1 | 1973.8 ± 9.8 | 76702.0 ± 4477.6 | 0.5974 ± 0.0062 | 0.0498 ± 0.0002 | 0.0529 ± 0.0047 |
| U1_oof_mma | RF | 234.4 ± 10.1 | 111.8 ± 9.1 | 147.6 ± 10.1 | 1972.2 ± 9.1 | 74918.0 ± 4990.2 | 0.5927 ± 0.0071 | 0.0492 ± 0.0007 | 0.0536 ± 0.0044 |
| U2 | RF | 255.8 ± 3.8 | 146.0 ± 4.6 | 126.2 ± 3.8 | 1938.0 ± 4.6 | 64560.0 ± 1883.4 | 0.6026 ± 0.0129 | 0.0495 ± 0.0006 | 0.0701 ± 0.0022 |
| U5 | RF | 253.0 ± 11.9 | 146.6 ± 12.2 | 129.0 ± 11.9 | 1937.4 ± 12.2 | 65966.0 ± 5840.4 | 0.5979 ± 0.0109 | 0.0488 ± 0.0018 | 0.0703 ± 0.0058 |
| baseline | LGB | 216.6 ± 13.9 | 83.4 ± 11.1 | 165.4 ± 13.9 | 2000.6 ± 11.1 | 83534.0 ± 6826.1 | 0.6152 ± 0.0052 | 0.0494 ± 0.0006 | 0.0400 ± 0.0053 |
| U0 | LGB | 242.4 ± 3.6 | 123.2 ± 2.6 | 139.6 ± 3.6 | 1960.8 ± 2.6 | 71032.0 ± 1785.1 | 0.5963 ± 0.0104 | 0.0496 ± 0.0004 | 0.0591 ± 0.0012 |
| corrected_current | LGB | 241.8 ± 4.7 | 122.8 ± 3.1 | 140.2 ± 4.7 | 1961.2 ± 3.1 | 71328.0 ± 2312.8 | 0.5796 ± 0.0066 | 0.0496 ± 0.0004 | 0.0589 ± 0.0015 |
| U1_oof_mma | LGB | 244.4 ± 4.2 | 120.6 ± 2.9 | 137.6 ± 4.2 | 1963.4 ± 2.9 | 70006.0 ± 2067.9 | 0.5890 ± 0.0158 | 0.0498 ± 0.0002 | 0.0579 ± 0.0014 |
| U2 | LGB | 263.8 ± 4.4 | 156.4 ± 6.7 | 118.2 ± 4.4 | 1927.6 ± 6.7 | 60664.0 ± 2203.1 | 0.6047 ± 0.0052 | 0.0496 ± 0.0006 | 0.0750 ± 0.0032 |
| U5 | LGB | 261.2 ± 2.9 | 151.6 ± 5.1 | 120.8 ± 2.9 | 1932.4 ± 5.1 | 61916.0 ± 1448.0 | 0.6079 ± 0.0075 | 0.0489 ± 0.0009 | 0.0727 ± 0.0025 |

### Paired comparisons: U5 minus each reference

Means and sample SDs use split/family/seed-matched differences, not subtraction of SDs. All raw metric/resource deltas, including diagnostic cost and operating points, are linked below. The LR U5-minus-U2 AP mean is -0.0000178104 (the four-decimal table rounds to -0.0000).

| U5 minus | Family | AP | Macro AP | Minority F1 | Minority recall | Macro F1 | Weighted F1 | Inverse-support F1 | BAC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | LR | -0.0235 ± 0.0032 | -0.0125 ± 0.0018 | +0.0881 ± 0.0155 | +0.1497 ± 0.0190 | +0.0424 ± 0.0078 | +0.0109 ± 0.0026 | +0.0739 ± 0.0131 | +0.0598 ± 0.0081 | +0.0425 ± 0.0080 |
| baseline | RF | -0.0219 ± 0.0102 | -0.0107 ± 0.0058 | +0.0371 ± 0.0287 | +0.1487 ± 0.0620 | +0.0144 ± 0.0144 | -0.0012 ± 0.0047 | +0.0301 ± 0.0243 | +0.0544 ± 0.0258 | +0.0108 ± 0.0200 |
| baseline | LGB | -0.0132 ± 0.0064 | -0.0067 ± 0.0033 | +0.0226 ± 0.0163 | +0.1168 ± 0.0387 | +0.0076 ± 0.0081 | -0.0027 ± 0.0026 | +0.0180 ± 0.0137 | +0.0420 ± 0.0159 | +0.0088 ± 0.0129 |
| U0 | LR | -0.0223 ± 0.0011 | -0.0115 ± 0.0006 | -0.0143 ± 0.0084 | +0.0126 ± 0.0097 | -0.0104 ± 0.0046 | -0.0076 ± 0.0020 | -0.0131 ± 0.0072 | -0.0009 ± 0.0049 | -0.0286 ± 0.0081 |
| U0 | RF | -0.0068 ± 0.0073 | -0.0031 ± 0.0044 | +0.0015 ± 0.0185 | +0.0466 ± 0.0468 | -0.0018 ± 0.0093 | -0.0040 ± 0.0032 | +0.0005 ± 0.0156 | +0.0148 ± 0.0194 | -0.0043 ± 0.0179 |
| U0 | LGB | -0.0006 ± 0.0071 | -0.0002 ± 0.0037 | +0.0088 ± 0.0084 | +0.0492 ± 0.0091 | +0.0029 ± 0.0051 | -0.0012 ± 0.0029 | +0.0070 ± 0.0074 | +0.0178 ± 0.0048 | +0.0063 ± 0.0100 |
| corrected_current | LR | -0.0220 ± 0.0013 | -0.0115 ± 0.0007 | -0.0126 ± 0.0110 | +0.0168 ± 0.0115 | -0.0096 ± 0.0062 | -0.0075 ± 0.0028 | -0.0117 ± 0.0095 | +0.0007 ± 0.0060 | -0.0282 ± 0.0112 |
| corrected_current | RF | -0.0044 ± 0.0078 | -0.0019 ± 0.0047 | +0.0089 ± 0.0181 | +0.0581 ± 0.0470 | +0.0023 ± 0.0091 | -0.0023 ± 0.0032 | +0.0068 ± 0.0153 | +0.0203 ± 0.0192 | +0.0034 ± 0.0172 |
| corrected_current | LGB | +0.0034 ± 0.0056 | +0.0020 ± 0.0033 | +0.0096 ± 0.0098 | +0.0508 ± 0.0174 | +0.0033 ± 0.0055 | -0.0011 ± 0.0027 | +0.0076 ± 0.0085 | +0.0185 ± 0.0078 | +0.0071 ± 0.0110 |
| U1_oof_mma | LR | -0.0238 ± 0.0025 | -0.0125 ± 0.0014 | -0.0014 ± 0.0135 | +0.0304 ± 0.0159 | -0.0036 ± 0.0074 | -0.0050 ± 0.0033 | -0.0021 ± 0.0116 | +0.0073 ± 0.0078 | -0.0177 ± 0.0133 |
| U1_oof_mma | RF | -0.0030 ± 0.0105 | -0.0011 ± 0.0061 | +0.0036 ± 0.0173 | +0.0487 ± 0.0449 | -0.0006 ± 0.0087 | -0.0034 ± 0.0032 | +0.0023 ± 0.0147 | +0.0160 ± 0.0184 | -0.0020 ± 0.0164 |
| U1_oof_mma | LGB | -0.0023 ± 0.0044 | -0.0011 ± 0.0023 | +0.0030 ± 0.0074 | +0.0440 ± 0.0153 | -0.0006 ± 0.0040 | -0.0030 ± 0.0016 | +0.0019 ± 0.0063 | +0.0146 ± 0.0068 | -0.0007 ± 0.0080 |
| U2 | LR | -0.0000 ± 0.0029 | +0.0001 ± 0.0017 | -0.0034 ± 0.0050 | -0.0047 ± 0.0079 | -0.0018 ± 0.0027 | -0.0007 ± 0.0011 | -0.0029 ± 0.0042 | -0.0022 ± 0.0034 | -0.0032 ± 0.0049 |
| U2 | RF | -0.0002 ± 0.0103 | -0.0003 ± 0.0059 | -0.0056 ± 0.0126 | -0.0073 ± 0.0345 | -0.0032 ± 0.0062 | -0.0015 ± 0.0018 | -0.0048 ± 0.0106 | -0.0038 ± 0.0141 | -0.0061 ± 0.0125 |
| U2 | LGB | +0.0015 ± 0.0090 | +0.0007 ± 0.0046 | -0.0004 ± 0.0082 | -0.0068 ± 0.0094 | +0.0001 ± 0.0050 | +0.0005 ± 0.0027 | -0.0003 ± 0.0072 | -0.0023 ± 0.0052 | -0.0002 ± 0.0099 |

Observed tradeoffs, without an inclusion decision:

- Versus baseline, U5 AP is lower by 0.0235/0.0219/0.0132 (LR/RF/LGB) while threshold-0.5 minority F1 and recall increase. False positives also increase; weighted F1 falls for RF/LGB, and recall at FPR≤0.05 falls for all three. Lower diagnostic cost reflects the high FN penalty, not the AP objective.
- Versus kept U1, AP falls for all three families (-0.0238/-0.0030/-0.0023), while minority recall rises (+0.0304/+0.0487/+0.0440). Minority F1 falls for LR and rises slightly for RF/LGB; weighted F1 falls for all three.
- Versus kept U2, AP changes -0.000018/-0.000192/+0.001461; **minority F1 and recall are slightly lower for every family**. Diagnostic cost increases by 894/1,406/1,252. ROC recall falls for LR/RF and rises slightly for LGB. Macro/weighted F1 improve only slightly for LGB. These are mixed descriptive results, not evidence of a uniformly beneficial combination.
- Versus corrected_current, AP falls for LR/RF and rises for LGB. Minority recall rises for all families; minority F1 falls for LR and rises for RF/LGB. No retained component or alternative is automatically substituted on the basis of outer-test scores.

### Actual joint selections

| Variant | Family | Selected alpha for seeds 0–4 |
|---|---:|---:|
| U2 | LR | 0.0, 0.0, 0.0, 0.0, 0.0 |
| U5 | LR | 0.0, 0.0, 0.0, 0.0, 0.0 |
| U2 | RF | 1.0, 1.0, 1.0, 0.5, 0.5 |
| U5 | RF | 0.5, 1.0, 1.0, 0.5, 0.0 |
| U2 | LGB | 0.5, 1.0, 0.5, 0.5, 0.5 |
| U5 | LGB | 0.5, 1.0, 0.5, 0.5, 0.5 |

The complete selected learner parameters, inner AP, outer AP, requested/actual budgets and stop reasons are in [joint_selection.csv](../results/runs/20261006T060142-8e042c6ffcf0/joint_selection.csv). U5's scoring probabilities/fold metadata match U1 exactly for all 150 shared inner learner proposals; U5's real grouping/support state matches U2 for all 165 inner/outer pipelines. Actual quotas and descendants, not grouping or scorer definitions, change under the composition.

### Runtime, memory and generation

Whole-run measured wall time: **4,558.824 seconds (1 h 15 m 59 s)**. Max observed cell RSS: **1,768.42 MiB**; maximum Torch-allocated CUDA memory: **15.68 MiB**, reserved: **26.00 MiB**. RSS/reserved maxima include persistent process/allocator/cache state, and Torch allocation does not include every classifier's GPU allocation; these are not standalone method memory superiority claims. Real group preparation is charged to its first cache user, so cell runtimes describe this recorded procedure/order.

| Variant | Family | Cell s | RSS MiB | Torch allocated MiB | Torch reserved MiB | Generated | Underspend |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | LR | 0.36 ± 0.02 | 1602.53 ± 289.88 | 0.00 ± 0.00 | 20.80 ± 11.63 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| U0 | LR | 15.56 ± 0.19 | 1702.32 ± 67.72 | 12.77 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| corrected_current | LR | 16.05 ± 0.66 | 1713.54 ± 41.58 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| U1_oof_mma | LR | 17.16 ± 0.49 | 1716.53 ± 34.89 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| U2 | LR | 17.22 ± 0.52 | 1716.66 ± 34.60 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| U5 | LR | 17.59 ± 0.48 | 1716.85 ± 34.18 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| baseline | RF | 22.13 ± 0.58 | 1721.99 ± 22.73 | 0.00 ± 0.00 | 26.00 ± 0.00 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| U0 | RF | 93.78 ± 5.40 | 1739.78 ± 10.71 | 12.77 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| corrected_current | RF | 121.22 ± 9.88 | 1741.31 ± 13.13 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| U1_oof_mma | RF | 154.73 ± 3.85 | 1741.69 ± 13.73 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| U2 | RF | 124.73 ± 7.28 | 1730.60 ± 4.65 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| U5 | RF | 161.95 ± 11.50 | 1738.70 ± 16.97 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| baseline | LGB | 2.82 ± 0.27 | 1730.77 ± 3.60 | 0.00 ± 0.00 | 26.00 ± 0.00 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| U0 | LGB | 23.69 ± 0.14 | 1731.26 ± 3.36 | 12.77 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| corrected_current | LGB | 26.99 ± 0.30 | 1730.87 ± 3.58 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| U1_oof_mma | LGB | 32.65 ± 0.13 | 1730.87 ± 3.58 | 15.67 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| U2 | LGB | 28.10 ± 0.14 | 1730.87 ± 3.58 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |
| U5 | LGB | 34.63 ± 2.25 | 1730.87 ± 3.58 | 15.68 ± 0.00 | 26.00 ± 0.00 | 1000.0 ± 0.0 | 0.0 ± 0.0 |

U5 mean cell time is 17.59/161.95/34.63 seconds, versus U2 17.22/124.73/28.10: about 1.02×/1.30×/1.23×. Each family uses 385 U5 classifier fits versus 220 U2 fits across five seeds, a 1.75× fit-count ratio; 165 of the U5 fits are real-only OOF models. Whole study: **4,290 classifier fits (990 OOF), 990 pipeline audits and 825,000 search/refit emissions**. All generation variants use 55,000 search/refit emissions per family; outer counts alone do not describe total work.

| Variant | Family | Classifier fits | OOF fits | Search+refit emissions | Pipeline audits |
|---|---:|---:|---:|---:|---:|
| baseline | LR | 55 | 0 | 0 | 55 |
| U0 | LR | 165 | 0 | 55000 | 55 |
| corrected_current | LR | 220 | 0 | 55000 | 55 |
| U1_oof_mma | LR | 385 | 165 | 55000 | 55 |
| U2 | LR | 220 | 0 | 55000 | 55 |
| U5 | LR | 385 | 165 | 55000 | 55 |
| baseline | RF | 55 | 0 | 0 | 55 |
| U0 | RF | 165 | 0 | 55000 | 55 |
| corrected_current | RF | 220 | 0 | 55000 | 55 |
| U1_oof_mma | RF | 385 | 165 | 55000 | 55 |
| U2 | RF | 220 | 0 | 55000 | 55 |
| U5 | RF | 385 | 165 | 55000 | 55 |
| baseline | LGB | 55 | 0 | 0 | 55 |
| U0 | LGB | 165 | 0 | 55000 | 55 |
| corrected_current | LGB | 220 | 0 | 55000 | 55 |
| U1_oof_mma | LGB | 385 | 165 | 55000 | 55 |
| U2 | LGB | 220 | 0 | 55000 | 55 |
| U5 | LGB | 385 | 165 | 55000 | 55 |

### Verification and complete artifacts

- All **159 repository tests** passed, including ten new U5 behavioral/real Shoppers tests. Actual learners, independent fold preprocessors/OOF models, mathematical quota weights/caps, rescued islands, real ancestry/categories, original Welch transitions, held-out perturbations, shared-cache isolation, independently refitted predictions and user-only development/confirmatory gates were exercised. Existing sklearn/SciPy iprint and LightGBM feature-name warnings were not suppressed.
- Actual 18-cell training-only smoke: `results/smoke/20261006T055011-c27595b0db9f`; all 18 predictions/360 numeric metric values, 72 pipelines, 24 OOF and 24 group audits (12 combined), three joint selections and 252 fits were checked. Its rows exclude the official outer test; smoke scores are not study evidence.
- Full run: all 90 predictions and **1,800 numeric metric values** were recomputed from round-trip parsed saved probabilities. All 990 pipeline audits were checked; 330 OOF, 330 group and 165 combined audits passed training-domain, fold exclusion/coverage, probability-to-MMA, real-ancestry, group/seed cap and selected-joint-configuration checks. All 90 canonical identities/source/index entries and seven changed source/config/test archive files match. All 75 references reproduce (1,500 metric values and actual counts).
- Publication: every one of **972 numeric mean/sample-SD values, 10,125 per-seed paired deltas, 4,050 paired mean/SD values and 162 predictive table cells** was independently verified against canonical records; per-seed metric/resource values and fit/emission totals were also checked. Resources have their documented process/cache scope.
- [All 90 per-seed metrics/resources](../results/runs/20261006T060142-8e042c6ffcf0/all_metrics.csv), [all mean/sample-SD summaries](../results/runs/20261006T060142-8e042c6ffcf0/metric_summary.csv), [all per-seed metric/resource deltas against five references](../results/runs/20261006T060142-8e042c6ffcf0/paired_metric_deltas.csv), [all paired means/SDs](../results/runs/20261006T060142-8e042c6ffcf0/paired_metric_summary.csv), [paired means/SDs versus corrected_current](../results/runs/20261006T060142-8e042c6ffcf0/paired_summary_vs_corrected.csv).
- [All 990 generation/scoring/fit audit summaries](../results/runs/20261006T060142-8e042c6ffcf0/generation_audit_summary.csv), [actual joint learner/alpha choices](../results/runs/20261006T060142-8e042c6ffcf0/joint_selection.csv), [search/refit work totals](../results/runs/20261006T060142-8e042c6ffcf0/work_summary.csv).
- [Prediction/OOF/group/source/index/reference verification](../results/runs/20261006T060142-8e042c6ffcf0/verification_summary.json), [publication verification](../results/runs/20261006T060142-8e042c6ffcf0/publication_verification.json), [canonical AP summaries](../results/runs/20261006T060142-8e042c6ffcf0/ap_summary.json), [canonical AP pairings](../results/runs/20261006T060142-8e042c6ffcf0/paired_vs_corrected.json), [complete review tables](../results/runs/20261006T060142-8e042c6ffcf0/review_tables.json), [study summary](../results/runs/20261006T060142-8e042c6ffcf0/phase1_u5_summary.json), [execution-time protocol/study/source manifest](../results/runs/20261006T060142-8e042c6ffcf0/manifest.json).

**User decision: discard U5; canonical U2 KMeans/stability is reaffirmed as the preferred candidate.** U1 remains kept, U3/U4/U5 are excluded, W5/R5 remain unavailable, the configuration not_frozen and confirmatory evaluation closed. U5's execution-time pending-decision snapshots and all metrics/source/predictions remain unchanged; retaining U1 and U2 individually does not retain their rejected U5 combination.



## SOS published-track audit: unresolved, not interchangeable with controlled data

Pinned source: `JayoungKim408/SOS@f1b450cf199d5e0013b401e380008446ef8757f7`. Individual NPZ, JSON, configuration and evaluator hashes are in the manifest. The Samsung SDS Public License V1.0 restricts use to noncommercial research/evaluation and restricts redistribution of modified code. No author implementation was vendored.

### Shoppers

Released arrays preserve 12,330 rows but use train/test counts **9,864 / 2,466**, with class counts **8,307 / 1,557** and **2,115 / 351**, respectively. They are not the controlled stratified holdout. Their combined row multiset matches raw UCI after recovered encodings at eight-decimal tolerance. There are 125 duplicate raw rows, so an unambiguous original source-index identity cannot be recovered from feature values alone.

Month, VisitorType, Weekend and label use metadata category positions. OperatingSystems, Browser, Region and TrafficType retain original one-based codes despite zero-based metadata lists. Author `FeatureMaker` shuffles data, truncates at 50,000 rows, passes ordinal/binary categories through and train-fits one-hot encoding for categories with more than two values. Continuous columns use `log(max(x,1e-2))` when metadata minimum is nonnegative and maximum is at least 1,000; otherwise they use `(x-min)/(max-min)*5`.

The metadata continuous bounds match the **full raw table**. In particular Administrative maximum 27 is in author test, whereas author training maximum is 26. This establishes a held-out-metadata boundary in the author scaling recipe. It must not be relabeled as train-fit controlled preprocessing.

The released binary evaluator uses DT(max_depth=20), default AdaBoost, LR(lbfgs,max_iter=50) and MLP(hidden_layer_sizes=(50,),max_iter=50). Its misleadingly named `weighted_f1` is:

`F1(class0) * support(class1)/N + F1(class1) * support(class0)/N`.

This is **inverse-support** F1, not sklearn's support-weighted F1. The target registry uses `sos_inverse_support_f1`; ordinary controlled weighted F1 remains a separate metric.

Training fixes NumPy/torch RNG to 2021. Generator evaluation repeats generation five times, averages classifiers within each evaluator repetition, then reports mean and NumPy population standard deviation over generation repetitions. Although `compute_scores` defaults to five iterations, the Shoppers/Satimage generator configs set `test.n_iter=1`. The repository does not release the no-resampling/SMOTE baseline execution recipe or five explicit baseline seeds. Those exact RNG sequences, baseline preprocessing/SMOTE recipe and reported spread hierarchy remain missing prerequisites. Replacing them with five invented seeds would not reproduce Table 2.

### Satimage

Author arrays contain **6,430**, not UCI's 6,435 rows: train 5,144, test 1,286. The mapping is raw labels `[1,7,3,5,2,4]` to `[0,1,2,3,4,5]`. Features are already standardized, but removed-row identities and preprocessing provenance are absent. These are not official 4,435/2,000 partitions. The released multiclass evaluator has **two** classifiers: balanced DT(max_depth=30) and MLP((100,),max_iter=50), not the four-classifier claim in the initial handoff. Its inverse-support weight is `(1-support_c/N)/(K-1)`.

Until the table-producing baseline recipe, actual classifier/repetition hierarchy, removed rows and preprocessing provenance are supplied, published Shoppers and Satimage configs remain `unresolved` and refuse execution.

## Feedzai BAF baseline audit

Pinned source: `feedzai/bank-account-fraud@3377cfd8fc20edcda3dd9dd14c10ed3dd276c6de`. The datasheet explicitly verifies month values **0 through 7**, independent synthetic applications, fraud_bool label and the dataset license **CC BY-NC-ND 4.0**. Code LICENSE contains Apache 2.0 plus a commercial-contact preface; notebook/random-search file headers prohibit reproduction. The audit records facts and source hashes without copying those implementations.

The schema has 30 application features plus month, hence **31 predictors / 32 columns including fraud_bool**. Expected Base class counts (988,971 / 11,029) come from the independent study in PUBLISHED_RESULTS; they are not locally verified release counts. The loader enforces the named Base schema/counts and records the supplied file SHA256, but no upstream authoritative release checksum was available. Its runtime metadata explicitly preserves that provenance limitation.

The official notebook train-fits LabelEncoder on five categoricals (payment_type, employment_status, housing_status, source, device_os), passes them as categorical features to LightGBM, retains month as a predictor and leaves numeric -1 sentinels intact. The controlled preprocessing differs explicitly: categorical one-hot encoding and known missing-value normalization. Datasheet `device_distinct_emails` corresponds to released schema `device_distinct_emails_8w`. Protected groups are age **>=50**, not the initial handoff's >50.

Official search: 100 unique trial seeds sampled without replacement from range(1,000,000) after NumPy seed 42; each initializes Optuna RandomSampler. Audited distributions:

| Parameter | Distribution |
|---|---|
| n_estimators | Log integer, 20–10,000 |
| max_depth | Integer uniform, 3–30 |
| learning_rate | Log float, 0.02–0.1 |
| num_leaves | Log integer, 10–100 |
| boosting_type | gbdt or goss |
| min_data_in_leaf | Log integer, 5–200 |
| max_bin | Integer uniform, 100–500 |
| enable_bundle | True or False |

`baf_base_controlled.yaml` contains 100 literal LightGBM candidate dictionaries in these audited distribution families, generated with NumPy RandomState per recovered trial seed; quantized log integers use half-step interval boundaries. Categorical RNG implementation differs from Optuna: this is **not exact author candidate-stream replay**. No Optuna dependency is required to execute the controlled candidates. All controls share the same candidates. Source environment pins include NumPy 1.20.2, Optuna 2.10.0, LightGBM 3.1.1 and pandas 1.1.4; the requirements file itself misspells LightGBM with an accented leading character.

The official notebook has **no inner validation**. It computes an empirical operating point at the largest test ROC FPR strictly below 0.05, uses strict `prediction > threshold`, and highlights five models ranked by **test recall**. A safe controlled protocol cannot claim to reproduce that selection: fit months 0–4, validate month 5, test months 6–7; report test-ROC recall separately from realized test recall/FPR of the validation-selected threshold. The published notebook also shuffles train/test without explicit permutation seeds. No verified scalar Base target exists; Figure 1 is a distribution, not a number to digitize into a target.

Reachable local BAF prerequisite: supply an authentic official Kaggle **Base.csv**, **Base.parquet**, or release ZIP containing exactly one Base.csv via `local_data`. Exact published replay additionally requires authoritative release version/checksum, shuffle identities and an explicitly segregated author test-ranked analysis; it is intentionally not run by the controlled runner. No credentials or alternate fraud dataset are substituted.

## Target disagreements and remaining comparability gates

- DATASETS/PLAN initially labeled SOS inverse-support F1 as conventional weighted F1 and described five seeds; audited source instead exposes a fixed RNG and generator repetitions. Satimage's released evaluator has two classifiers, not four. PUBLISHED_RESULTS originally omitted SOS and described no comparable modern named-task number. All interpretations are recorded without changing numerical targets.
- SOS Shoppers targets: baseline 0.601 ± 0.039, SMOTE 0.648 ± 0.004, SOS-VE 0.675 ± 0.004. Satimage: SMOTE 0.846 ± 0.005 and SOS-VP 0.857 ± 0.006. These handoff Table-2 numbers were not re-estimated locally.
- APS challenge winner cost 9,920 (542 FP, 9 FN) is a historical official-test target. DATASETS adds the 2026 transformer mean cost 3,440 (mean FP=4,FN=6.8; five initializations), absent from the original PUBLISHED_RESULTS APS section. Exact winner recipe and transformer operation order/seeds/checkpoint/threshold details remain unaudited, so their published config is unresolved.
- PUBLISHED_RESULTS Shoppers CLIMB Table-5 AP values (percent, including base 38.1) carry source extraction/method-column caveats and use a different five-fold/tuned protocol. They are not SOS or controlled holdout targets, and repeated use of Shoppers is not independent evidence.
- PUBLISHED_RESULTS Satimage Statlog KNN 90.6% / LVQ 89.5% use official-test accuracy, not SOS's inverse-support F1 or the controlled macro F1.

All unresolved published configurations list exact prerequisites and fail closed. The controlled configurations remain executable when their real data/dependencies are available; no missing published detail is concealed by a fabricated fallback, no future-U1 method is enabled, and no published-comparability or performance improvement is claimed.
