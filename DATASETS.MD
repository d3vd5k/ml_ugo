# UGO dataset and benchmark handoff

## Purpose

Add explicit, published-result benchmarks alongside the existing UGO–CLIMB baseline. Follow [PLAN.MD](PLAN.MD) for method development. Order: **UGO pipeline → learned generators → images**.

Research checked: **2026-10-03**. Scores below are **paper-reported references, not locally reproduced results or verified current SOTA**. Author implementations, exact splits and preprocessing still require audit.

## Download sources

Use the **Download** button on UCI/Kaggle pages. These are source pages; archive downloads and checksums have not been tested locally.

| Dataset | Download source | Files / preparation |
|---|---|---|
| Online Shoppers | [UCI](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) | `online_shoppers_intention.csv`; compare with the [SOS data directory](https://github.com/JayoungKim408/SOS/tree/master/tabular_datasets) for paper reproduction |
| APS Failure | [UCI](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) | `aps_failure_training_set.csv`, `aps_failure_test_set.csv`, description; preserve official partitions |
| BAF Base | [Official Kaggle release](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022), linked by [Feedzai](https://github.com/feedzai/bank-account-fraud) | Select Base; CSV/parquet; Kaggle may require sign-in |
| ISOLET, raw | [UCI](https://archive.ics.uci.edu/dataset/54/isolet) | `isolet1+2+3+4.data.Z`, `isolet5.data.Z`; raw 26-class data, not the prepared MGVAE binary task |
| Musk Version 2, raw | [UCI](https://archive.ics.uci.edu/dataset/75/musk+version+2) | `clean2.data.Z`, names/info files; 6,598 conformations; validate MGVAE correspondence and splits |
| Water Quality, candidate source | [Kaggle Water quality](https://www.kaggle.com/datasets/mssmartypants/water-quality) | Source page found; exact correspondence to MGVAE remains unverified. Do not claim reproduction until schema/counts/preprocessing match |
| CIFAR-10 / CIFAR-100 | [Official Toronto download page](https://www.cs.toronto.edu/~kriz/cifar.html) | Choose Python versions. Construct long-tailed training subsets using the reference recipe; standard downloads are balanced |
| Satimage | [UCI](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite) | `sat.trn`, `sat.tst`, `sat.doc`; compare with SOS processed data |

For MGVAE, use the [author repository](https://github.com/Aiqz/MGVAE) and [tabular loaders](https://github.com/Aiqz/MGVAE/tree/master/utils/load_data) to recover task construction. Its README requires a local data path and does not provide a prepared-dataset download link. Keep binary ISOLET mapping and exact Water Quality identity unresolved until verified.

## Execution order

| Order | Dataset | Role | First action |
|---|---|---|---|
| 1 | Online Shoppers | Fast tabular development; published score-based generation comparison | Recover SOS data/splits and reproduce its simple baselines |
| 2 | APS Failure | Severe imbalance; application-cost result | Load official train/test; reproduce cost-sensitive controls |
| 3 | BAF Base | Scale and temporal generalization | Recover official temporal protocol; establish LightGBM baseline |
| 4 | Isolet, exact MGVAE binary task | Primary learned-generator comparison | Recover class mapping, data and balanced-test construction |
| 5 | Water Quality; Musk, exact MGVAE tasks | Supporting generator evidence | Add only after Isolet reproduction works |
| 6 | CIFAR-10-LT → CIFAR-100-LT | Image transfer; feature-generation comparison | Reproduce reference representation and classifier pipeline |
| Optional | Satimage, six classes | Tabular multiclass extension | Recover SOS processed version |

Keep CLIMB as the broad generalization suite. Identify overlap using source IDs, versions and row fingerprints; do not count the same dataset twice as independent evidence. Freeze method choices on declared development datasets before confirmatory CLIMB/benchmark evaluation.

## Two separate evaluation tracks

**A. Published-protocol reproduction:** match dataset version, labels, split, preprocessing, classifier, tuning, metric, aggregation and generation budget. Preserve author protocol deviations from the controlled track and document them.

**B. Controlled comparison:** identical real-data splits, preprocessing, downstream classifier, validation access and tuning budget across methods. Compare equal synthetic counts first; report native-budget results separately.

Never compare a new split/classifier score directly with a published number. If the reference protocol is unrecoverable, mark it `unresolved`; continue the controlled track under a new protocol ID.

## 1. Online Shoppers Purchasing Intention

- **Source:** [UCI dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).
- **Task:** predict `Revenue`; 12,330 sessions, 1,908 purchases, 10,422 non-purchases; minority 15.5%, imbalance ratio ≈5.46. Mixed numerical/categorical features.
- **Reference:** [Score-based Oversampling for Tabular Data, KDD 2022](https://arxiv.org/abs/2206.08555); [author code](https://github.com/JayoungKim408/SOS).
- **Reported Table 2:** no oversampling **0.601 ± 0.039**; SMOTE **0.648 ± 0.004**; SOS-VE **0.675 ± 0.004**.
- **Metric audit:** the SOS code calls its metric `weighted_f1` but uses inverse-support weights `(1 - n_c/N)/(K - 1)`, not sklearn support-weighted F1. Its binary evaluation averages decision tree, logistic regression, AdaBoost and MLP per repetition; the available code fixes training RNG to 2021 and repeats evaluation five times, rather than defining five independent seeds. Published-score comparability remains unresolved; see `reports/reproduction.md`.

**Do:**

1. Recover author-processed data, split indices/seeds, encodings and exact classifier/metric aggregation.
2. Reproduce no-oversampling and SMOTE rows before SOS. Verify author training-class balancing; retain imbalanced test distribution.
3. Run original UGO and improved UGO on the same protocol; add SOS in the generator stage.
4. For controlled evaluation, also report minority average precision, minority F1/recall and balanced accuracy.
5. Keep categories valid; do not interpolate integer category codes as continuous values.

**Unresolved:** author-processed data and code have been audited, but paper-to-code metric/seed correspondence and exact reported baseline reproduction remain unresolved. Controlled raw-UCI execution is a separate protocol, not a reproduction claim.

## 2. APS Failure at Scania Trucks

- **Source:** [UCI dataset and challenge description](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks).
- **Task:** APS-related failure; official training set **60,000**, including **1,000 positives / 59,000 negatives**; official test set **16,000**. Anonymized numerical measurements with `na` missing values. Verify predictor count from schema; exclude label.
- **Primary metric:** minimize challenge cost:

\[
C=10\,FP+500\,FN.
\]

| Reference | Reported cost | Interpretation |
|---|---:|---|
| IDA 2016 challenge winner | 9,920 | 542 FP; 9 FN |
| [Cost-sensitive transformer, 2026](https://link.springer.com/article/10.1007/s10586-026-06080-2) | 3,440 | Mean over five initializations; mean FP=4, FN=6.8 |

The recent pipeline combines Bayesian ridge imputation, SVM-SMOTE + RepeatedENN, a transformer and focal loss. [Author code](https://github.com/AliBeikmohammadi/Cost-Sensitive-Transformer-Model).

**Do:**

1. Preserve the official test set. Create stratified inner validation from training data.
2. Fit imputation, scaling, selection and resampling inside training partitions. Audit the reference implementation's operation order.
3. Establish class-weighted boosting and original UGO; choose calibration/decision threshold using inner validation cost.
4. Reproduce the published pipeline when feasible. Replace its resampling component with UGO while holding the remaining recipe fixed.
5. Report cost, FP, FN, minority average precision, threshold, seeds and runtime. Keep cost on the original test-set scale.

**Unresolved:** recent code has not been run or audited; 3,440 is a reproduction target, not an established local baseline.

## 3. Bank Account Fraud — Base

- **Source/data entry:** [official BAF repository](https://github.com/feedzai/bank-account-fraud); follow its current download instructions.
- **Reference:** [Turning the Tables, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/d9696563856bd350e4e7ac5e5812f23c-Paper-Datasets_and_Benchmarks.pdf).
- **Use:** `Base` only initially. Approximately **1 million rows**, **30 features**, **1.1% fraud**, eight months. Public data are synthetic, derived from real banking data.
- **Published protocol:** first six months train; remaining two test. LightGBM with **100 random-search configurations**. Main operating metric: recall at **5% FPR**.
- **Numerical target:** no single verified scalar is recorded here; the paper presents performance distributions. Reproduce the official baseline rather than inventing a target from a plot.

**Do:**

1. Verify month encoding, target and feature schema against author code.
2. Use time-respecting inner validation within the first six months; keep the final two months untouched.
3. Recover the official search space and selection rule. Match tuning opportunities for controlled comparisons.
4. Distinguish test-ROC recall at FPR=5% from a validation-selected threshold's realized test recall/FPR; report both with explicit labels when applicable.
5. Smoke-test on a training-only subset; use full official partitions for final claims. Record generation cap, wall time and peak memory.

Do not fully balance a million-row dataset by default. Select generation budget on validation data and compare methods at matched budgets.

## 4. MGVAE tabular tasks

**Reference:** [Generative Oversampling for Imbalanced Data via Majority-Guided VAE, AISTATS 2023](https://proceedings.mlr.press/v206/ai23a.html); [paper PDF](https://proceedings.mlr.press/v206/ai23a/ai23a.pdf); [author code](https://github.com/Aiqz/MGVAE).

The paper evaluates these binary tasks on balanced test distributions with a two-layer fully connected classifier and three random trials. It calls accuracy on the balanced test set **B-ACC**. Preserve the exact metric implementation and class balance.

| Task | SMOTE B-ACC | MGVAE B-ACC | Paper table | Priority |
|---|---:|---:|---|---|
| Isolet | 93.8 ± 0.7% | **95.1 ± 0.6%** | 7 | Primary |
| Water Quality | 56.8 ± 1.4% | **67.2 ± 1.1%** | 6 | Supporting |
| Musk | 83.6 ± 1.3% | **92.4 ± 1.2%** | 5 | Supporting |

**Do:**

1. Resolve exact files, class mapping and split construction before training. Standard ISOLET has 26 classes; do not invent its binary conversion. Do not substitute an unrelated water-potability dataset for “Water Quality.”
2. Record raw and split-specific class counts. Do not interpret the paper's sample-count table as raw totals without checking its scope.
3. Recover normalization, network architecture, optimizer, checkpoint selection, oversampling counts and seeds from author code/configuration.
4. Reproduce SMOTE and MGVAE, then insert original/improved UGO into the same classifier recipe.
5. In the controlled track, evaluate a shared neural classifier and a shared boosting classifier separately.
6. Inspect speaker identities for Isolet and molecule/conformer identities for Musk. If available, add group-disjoint evaluation as a separate protocol.

Balanced-test reproduction is an explicit exception to the natural-prevalence controlled protocol. Add a natural-prevalence evaluation where recoverable; never compare its raw accuracy directly with published B-ACC.

## 5. Image stage: CIFAR long-tailed variants

Start with **CIFAR-10-LT**, then **CIFAR-100-LT**. Primary imbalance factor **100**; sensitivity **50 and 10** when supported by the reproduced recipe.

\[
\mathrm{IF}=\frac{\max_c n_c}{\min_c n_c}.
\]

Recover exact long-tail sampling, class ordering, seeds and balanced test protocol from each reference implementation. “IF=100” alone does not identify a reproducible dataset split.

| Reference | Dataset / IF | Reported top-1 | Role |
|---|---|---:|---|
| [LDMLR, CVPR Workshops 2024](https://openaccess.thecvf.com/content/CVPR2024W/L3D-IVU/papers/Han_Latent-based_Diffusion_Model_for_Long-tailed_Recognition_CVPRW_2024_paper.pdf), WCDAS + LDMLR | CIFAR-10-LT / 100 | 86.29% | Feature diffusion reference |
| Same recipe | CIFAR-100-LT / 100 | 51.92% | Feature diffusion reference |
| [LOS, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) | CIFAR-100-LT / 100 | 54.9% | Strong classifier-retraining control |

[LOS code](https://github.com/Thinklab-SJTU/LOS). LDMLR code availability remains unresolved. These scores use different training recipes; they are not a direct method ranking.

**Do:**

1. Reproduce representation learning before comparing generators.
2. Freeze one encoder/checkpoint and use identical extracted features for no-generation, original UGO, improved UGO and learned feature generators.
3. Match classifier-head training, augmentations, synthetic counts and compute accounting; include a strong classifier-retraining baseline.
4. Report top-1, macro recall and per-class recall; define any many/medium/few class bins explicitly.
5. Put external-pretraining results, such as CLIP/DINO features, in a separate table. Do not compare them directly with training-from-scratch reference scores.

## Optional: Satimage multiclass

[UCI Statlog Landsat Satellite](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite); use the **six-class SOS task**, not a binary variant. SOS Table 2 reports its named weighted-F1 target **0.846 ± 0.005** for SMOTE and **0.857 ± 0.006** for SOS-VP. The audited code instead uses inverse-support F1 and only decision tree/MLP for multiclass evaluation; do not assume the binary four-classifier/five-seed recipe. Resolve the paper/code correspondence before claiming reproduction. This is a tabular descriptor benchmark, not the image stage.

## Shared evaluation rules

For the controlled track, split real data before fitting any preprocessing, clustering, calibration, pseudo-labeling or generation. Keep real and synthetic provenance; prevent descendants of held-out examples from entering training.

\[
\mathrm{BA}=\frac1K\sum_{c=1}^{K}\frac{TP_c}{TP_c+FN_c},
\qquad
F_{1,\mathrm{weighted}}=\sum_{c=1}^{K}\frac{n_c}{N}F_{1,c}.
\]

The formula above is conventional support-weighted F1. It is distinct from the audited SOS inverse-support metric; both have separate metric IDs in the result registry.

- Use average precision with its implementation recorded; do not silently substitute trapezoidal PR-AUC.
- Match splits/seeds across methods; save per-run predictions and report paired differences with uncertainty. State whether each ± is SD, SE or a confidence interval.
- Controls: no resampling, class weighting, ROS, SMOTE, original UGO; add the stage-specific published method. Reuse broader controls from the execution plan.
- Select configurations and thresholds on validation only. Report all prespecified datasets, including failures and regressions.
- Real benchmarks do not establish true overlap or latent minority groups. Retain synthetic diagnostic datasets with known boundaries/subclusters for mechanism claims.
- For unlabeled-data experiments, hide labels only within training data; exclude test data from the unlabeled pool. Record label budget, sampling seed and whether label removal is stratified.

## Required repository artifacts

| Artifact | Required contents |
|---|---|
| `benchmarks/manifest.yaml` | Source/version, license, checksums, target/class mapping, feature types, missingness, raw/split class counts, entity/time fields, CLIMB overlap |
| `configs/benchmarks/` | Separate published/controlled protocol IDs; splits, preprocessing, classifier, tuning, seeds, synthetic budget, metric/threshold definitions |
| `results/published_targets.csv` | Paper/table, method, dataset variant, protocol, metric/unit, reported mean/spread, spread type, source URL, reproduction status |
| `results/benchmark_runs.csv` | Commit/config hash, protocol/split/seed, method, metrics, runtime/memory, generated counts, status, prediction location |
| `reports/reproduction.md` | Reported vs reproduced values; deviations, unresolved details and run commands |

## First work package for Codex

1. Locate the UGO–CLIMB repository and preserve its current baseline/configuration.
2. Create the manifest and target registry from this document; mark unverified fields explicitly.
3. Acquire Online Shoppers author data/protocol and APS official files; record checksums and split counts.
4. Verify leakage boundaries, category handling, label encoding and metric implementations.
5. Reproduce Shoppers no-resampling/SMOTE and APS cost-sensitive controls; save commands and predictions.
6. Run original UGO on those exact splits. Establish baseline reproducibility before adding UGO improvements.
7. Add BAF after the pipeline is stable; start MGVAE comparisons after the tabular generation design is frozen; start images last.

**Completion gate:** every claimed published comparison has a reproducible protocol, matched baseline and explicit reported/reproduced status. Explain numerical discrepancies before claiming improvement. Missing reference details block that comparison, not unrelated benchmark work.
