# Published Baselines for Imbalanced Tabular / Long-Tailed Datasets

Compiled 2026-10-03. Every number below was read from the source listed beside it. Items marked **[inferred]** are my own reasoning, and items marked **[unverified]** could not be confirmed. Please check anything marked **[extraction caveat]** against the original PDF before citing.

## Contents

1. [Quick reference: where to find a comparable number](#1-quick-reference)
2. [MGVAE: Musk, Water Quality, ISOLET](#2-mgvae-musk-water-quality-isolet)
3. [CIFAR-10-LT / CIFAR-100-LT](#3-cifar-10-lt--cifar-100-lt)
4. [APS Failure (Scania)](#4-aps-failure-scania)
5. [BAF Base](#5-baf-base)
6. [Online Shoppers](#6-online-shoppers)
7. [Satimage (Statlog Landsat)](#7-satimage-statlog-landsat)
8. [CLIMB benchmark (recommended multi-dataset benchmark)](#8-climb-benchmark)
9. [Open items / not found](#9-open-items--not-found)
10. [Source list](#10-source-list)

---

## 1. Quick reference

| Dataset | Comparable published baseline? | Metric | Best number found | Protocol notes |
|---|---|---|---|---|
| Musk v2 | Yes (MGVAE only) | B-ACC / ASCA / GM (balanced test) | MGVAE 92.4 / 92.4 / 92.1 | Split not fully documented in paper |
| Water Quality | Yes (MGVAE only) | same | MGVAE 67.2 / 67.3 / 65.9 | Exact file match is an inference |
| ISOLET | Yes (MGVAE only) | same | MGVAE 95.1 / 95.0 / 94.9 | Binary conversion not stated in paper |
| CIFAR-10/100-LT | Yes (many papers) | Top-1 acc, balanced test | C100-LT r=100: 53.2 (2026 preprint), 52.5 (FeatRecon, ICLR'25) | ResNet-32, from scratch |
| APS Failure | Yes (official challenge) | 10·FP + 500·FN on official test | 9,920 (IDA 2016 winner) | Sub-9,920 claims online are not peer-reviewed |
| BAF Base | Partly | TPR @ 5% FPR (official) | Not tabulated in paper (figure only) | Temporal split: months 1-6 train, 7-8 test |
| Online Shoppers | Only via CLIMB or ad-hoc papers | AUPRC (CLIMB) | CLIMB: base 38.1, ensembles about 50 | Literature numbers vary widely |
| Satimage (Statlog) | Only the original Statlog result | Test accuracy | KNN 90.6% (Statlog era) | No modern peer-reviewed SOTA found |

---

## 2. MGVAE: Musk, Water Quality, ISOLET

**Paper:** Ai et al., *Generative Oversampling for Imbalanced Data via Majority-Guided VAE*, AISTATS 2023 (PMLR 206:3315-3330).
**Source:** https://arxiv.org/abs/2302.10910 · https://proceedings.mlr.press/v206/ai23a.html

### 2.1 Dataset splits as reported (paper Table 1)

| Dataset | Imbalance ratio | Majority (class 0) | Minority (class 1) |
|---|---|---|---|
| Musk-6.6 | 6.6 | 5,381 | 817 |
| Water Quality-11.1 | 11.1 | 6,784 | 612 |
| Isolet-17.5 | 17.5 | 6,997 | 400 |

### 2.2 Protocol (paper text)

- Tabular features scaled to [-1,1] or [0,1] per dimension, depending on the sign of the original data.
- Classifier: 2-layer fully connected network (Appendix E.2: 256 then 128 hidden units), 100 epochs, batch size 100, learning rate 1e-3.
- MGVAE: EWC weight 500 on tabular data. Generative model uses hidden dims 300/300 and latent dim 10 for tabular data.
- Metrics: Balanced Accuracy (B-ACC), Average Class-Specific Accuracy (ASCA), Geometric Mean (GM), all evaluated on a **balanced test distribution**. Results are averaged over **3 random trials**. Values are mean ± std.

### 2.3 Results (paper Tables 5, 6, 7)

**Musk (Table 5)**

| Method | B-ACC | ASCA | GM |
|---|---|---|---|
| ERM | 50.0 ± 0.0 | 50.0 ± 0.0 | 0.0 ± 0.0 |
| FOCAL | 87.1 ± 1.5 | 87.2 ± 1.6 | 86.1 ± 1.4 |
| RW | 85.6 ± 1.1 | 86.7 ± 1.0 | 84.3 ± 1.2 |
| CBRW | 84.2 ± 0.5 | 84.3 ± 0.5 | 82.7 ± 0.6 |
| LDAM | 89.7 ± 1.2 | 89.8 ± 1.2 | 89.0 ± 1.3 |
| RS | 84.9 ± 0.9 | 85.0 ± 1.0 | 83.6 ± 1.1 |
| SMOTE | 83.6 ± 1.3 | 83.5 ± 1.4 | 81.8 ± 1.7 |
| OCVAE | 90.5 ± 0.3 | 90.1 ± 0.2 | 89.8 ± 0.2 |
| **MGVAE** | **92.4 ± 1.2** | **92.4 ± 1.3** | **92.1 ± 1.4** |

**Water Quality (Table 6)**

| Method | B-ACC | ASCA | GM |
|---|---|---|---|
| ERM | 50.0 ± 0.0 | 50.0 ± 0.0 | 0.0 ± 0.0 |
| FOCAL | 65.7 ± 0.7 | 65.7 ± 0.7 | 63.3 ± 1.1 |
| RW | 57.7 ± 0.1 | 57.6 ± 0.3 | 48.7 ± 0.2 |
| CBRW | 56.4 ± 0.2 | 56.4 ± 0.3 | 49.5 ± 0.6 |
| LDAM | 65.8 ± 0.3 | 65.9 ± 0.2 | 62.0 ± 0.3 |
| RS | 58.3 ± 1.5 | 58.2 ± 1.4 | 51.3 ± 1.1 |
| SMOTE | 56.8 ± 1.4 | 56.0 ± 1.5 | 51.9 ± 1.6 |
| OCVAE | 61.9 ± 1.2 | 62.1 ± 1.9 | 58.3 ± 1.5 |
| **MGVAE** | **67.2 ± 1.1** | **67.3 ± 1.3** | **65.9 ± 1.2** |

**Isolet (Table 7)**

| Method | B-ACC | ASCA | GM |
|---|---|---|---|
| ERM | 50.0 ± 0.0 | 50.0 ± 0.0 | 0.0 ± 0.0 |
| FOCAL | 50.0 ± 0.0 | 50.0 ± 0.0 | 0.0 ± 0.0 |
| RW | 92.9 ± 0.4 | 92.9 ± 0.2 | 92.7 ± 0.2 |
| CBRW | 92.8 ± 0.4 | 92.8 ± 0.5 | 92.6 ± 0.6 |
| LDAM | 93.5 ± 1.0 | 93.5 ± 1.0 | 93.3 ± 1.1 |
| RS | 92.9 ± 0.3 | 92.8 ± 0.4 | 92.7 ± 0.4 |
| SMOTE | 93.8 ± 0.7 | 93.7 ± 0.8 | 93.6 ± 0.6 |
| OCVAE | 91.6 ± 0.4 | 91.7 ± 0.2 | 91.5 ± 0.2 |
| **MGVAE** | **95.1 ± 0.6** | **95.0 ± 0.7** | **94.9 ± 0.7** |

Baselines: ERM (cross-entropy), FOCAL (γ = 1.0), RW (re-weighting), CBRW (class-balanced re-weighting, β = 0.9999), LDAM, RS (random over-sampling, called ROS in the text), SMOTE, OCVAE (conditional-VAE oversampling).

The paper also reports MNIST, FashionMNIST and CelebA results, which are not reproduced here. It reports **no CIFAR results**.

### 2.4 Split reconstruction **[inferred]**

These points are my own arithmetic. I used remembered full-dataset sizes (Musk v2 = 6,598 rows with 1,017 musk; ISOLET = 7,797 rows), which were not checked in the sources above.

- Musk: 5,381 + 817 = 6,198, which is 400 fewer than 6,598. This fits a balanced held-out test of 200 per class (5,581 − 200 = 5,381; 1,017 − 200 = 817).
- ISOLET: 6,997 + 400 = 7,397, which is 400 fewer than 7,797. This fits a minority of two letters (about 600 samples) with 200 held out, and a majority of 24 letters (7,197 − 200 = 6,997). The paper does not say which letters.
- Water Quality: 6,784 + 612 = 7,396. If the Kaggle file has 7,996 rows with 912 `is_safe = 1` (my recollection, **[unverified]**), this fits a 300-per-class held-out test.
- The paper says implementation details are in its code release. Check the MGVAE repository to confirm.

### 2.5 Water Quality file identification

- The Kaggle candidate (https://www.kaggle.com/datasets/mssmartypants/water-quality) has 20 numeric contaminant features and a binary `is_safe` label (0 = not safe, 1 = safe), as listed in a downstream project README: https://github.com/praneeth-motapally/Water_Quality_Prediction
- A different Kaggle "water potability" dataset has 3,276 rows with 59.67% non-potable (https://dspace.unia.es/handle/10334/9193). It does not match MGVAE's counts.

---

## 3. CIFAR-10-LT / CIFAR-100-LT

**MGVAE does not evaluate on CIFAR.** The standard protocol is the long-tailed construction of Cao et al. (LDAM, NeurIPS 2019), with the original balanced test set.

**Comparison table source:** Chen et al., *Reframing Long-Tailed Learning via Loss Landscape Geometry*, arXiv 2603.21217 (March 2026). Peer-review status **[unverified]**. Numbers for other methods are quoted by that paper from the original publications, and I did not check each one. https://arxiv.org/pdf/2603.21217

**Protocol (same paper, supplement):** ResNet-32 trained from scratch, 200 epochs, batch size 256, SGD with momentum 0.9 and weight decay 5e-4, learning rate warmed up to 0.15 over 5 epochs and decayed by 0.1 at epochs 160 and 180. No external data or pre-trained models. Imbalance ratio r ∈ {100, 50, 10}. Metric: top-1 accuracy (%).

| Method (venue) | C100-LT r=100 | r=50 | r=10 | C10-LT r=100 | r=50 | r=10 |
|---|---|---|---|---|---|---|
| CE (baseline) | 38.3 | 43.9 | 55.7 | 70.4 | 74.8 | 86.4 |
| Focal Loss (ICCV'17) | 38.4 | 44.3 | 55.8 | 70.4 | 76.7 | 86.7 |
| LDAM-DRW (NeurIPS'19) | 42.0 | 46.6 | 58.7 | 77.0 | 81.0 | 88.2 |
| cRT (ICLR'20) | 42.3 | 46.8 | 58.1 | 75.7 | 80.4 | 88.3 |
| BBN (CVPR'20) | 42.6 | 47.0 | 59.1 | 79.8 | 82.2 | 88.3 |
| RIDE, 3 experts (ICLR'21) | 48.0 | - | - | - | - | - |
| CAM-BS (AAAI'21) | 41.7 | 46.0 | - | 75.4 | 81.4 | - |
| DiVE (ICCV'21) | 45.4 | 51.1 | 62.0 | - | - | - |
| SAM (NeurIPS'22) | 45.4 | - | - | 81.9 | - | - |
| BCL (CVPR'22) | 51.9 | 56.6 | 64.9 | 84.3 | 87.2 | 91.1 |
| CUDA (ICLR'23) | 47.6 | 51.1 | 58.4 | - | - | - |
| ADRW (NeurIPS'23) | 46.4 | - | 61.9 | 83.6 | - | 90.3 |
| H2T (AAAI'24) | 48.9 | 53.8 | - | - | - | - |
| GBG (AAAI'24) | 52.3 | 57.2 | - | 85.1 | 87.7 | - |
| DiffuLT (NeurIPS'24) | 51.5 | 56.3 | 63.8 | 84.7 | 86.9 | 90.7 |
| DiffuLT + BBN (NeurIPS'24) | 51.9 | 56.7 | 64.0 | 85.0 | 87.2 | 90.9 |
| SEL (ICCV'25) | 52.3 | 57.3 | 68.4 | 84.4 | 86.3 | 90.2 |
| Heuristic-CALA (AAAI'25) | 50.5 | - | 64.3 | 83.9 | - | 91.7 |
| Meta-CALA (AAAI'25) | 52.3 | - | 65.5 | 84.7 | - | 92.4 |
| FeatRecon (ICLR'25) | 52.5 | 57.0 | 65.3 | 85.2 | 87.8 | 91.6 |
| LLM-AutoDA, uses LLM (NeurIPS'24) | 51.0 | 54.8 | - | - | - | - |
| **Chen et al. (2026 preprint)** | **53.2** | **57.6** | **68.7** | **86.3** | **88.2** | **92.5** |

**CIFAR-100-LT, r=100, Many / Medium / Few accuracy (same source):**

| Method | Many | Med. | Few |
|---|---|---|---|
| CE | 65.2 | 37.1 | 9.1 |
| Focal | 65.3 | 38.4 | 8.1 |
| LDAM-DRW | 61.5 | 41.7 | 20.2 |
| cRT | 64.0 | 44.8 | 18.1 |
| RIDE (3 experts) | 68.1 | 49.2 | 23.9 |
| SAM | 64.4 | 46.2 | 20.8 |
| BCL | 67.2 | 53.1 | 32.9 |
| CUDA | 67.3 | 50.4 | 21.4 |
| DiffuLT | 69.0 | 51.6 | 29.7 |
| DiffuLT + BBN | 69.5 | 51.9 | 30.2 |
| LLM-AutoDA | 66.6 | 50.6 | 33.1 |
| Chen et al. (2026) | 67.3 | 54.9 | 34.9 |

**Caveats on comparability**

- CE baselines differ by training recipe. One paper reports CE on CIFAR-10-LT at imbalance factors 1000 / 500 / 200 / 100 / 50 / 10 of 53.5 / 59.5 / 66.4 / 71.2 / 77.4 / 86.8 (https://arxiv.org/pdf/2112.05958), compared with 70.4 at r=100 above.
- Some methods use an AutoAugment recipe or generated or external data. The DiffuLT paper notes that AutoAugment alone lifts CIFAR-100-LT (r=100) from 38.3 to 45.3 (https://arxiv.org/pdf/2403.05170).
- LLM-AutoDA uses a large language model, and DiffuLT uses diffusion-generated data.

---

## 4. APS Failure (Scania)

**Dataset:** UCI #421 (https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks). 60,000 training rows (59,000 negative, 1,000 positive), 16,000 test rows, 171 attributes (7 are histogram variables). Use the official partition.

**Challenge metric:** Total cost = 10 × (false positives) + 500 × (false negatives), where a false positive is an unnecessary workshop check and a false negative is a missed faulty truck. (UCI page; IDA 2016 challenge page: https://people.dsv.su.se/~isak-kar/IDA2016Challenge.txt)

### 4.1 IDA 2016 challenge results on the official test set

Source: Jose & Gopakumar, URSI AP-RASC 2019, Table 2, quoting the IDA 2016 website. https://2021.ursi.org/proceedings/procAP19/papers2019/AnImprovedRandomForestAlgorithmforclassificationinanimbalanceddataset.pdf

| Rank | FP | FN | Total cost |
|---|---|---|---|
| 1st | 542 | 9 | **9,920** |
| 2nd | 490 | 12 | 10,900 |
| 3rd | 398 | 15 | 11,480 |
| URSI 2019 improved RF (their own run) | 420 | 14 | 11,200 |

- The same paper credits Costa & Nascimento (IDA 2016, LNCS pp. 381-386) with the winning random-forest solution. I could not map the 2nd and 3rd scores to authors **[unverified]**.
- The paper also reports 10-fold cross-validation error rates on the training set, quoting the winner's paper for the first four rows:

| Classifier | FPR | FNR |
|---|---|---|
| SVM | 1.15% | 13.5% |
| LR | 2.36% | 9.5% |
| K-NN | 2.94% | 6.5% |
| RF | 3.74% | 3.7% |
| RF (URSI authors) | 2.8% | 2.7% |

### 4.2 Non-peer-reviewed results (treat with caution)

| Source | Method | FP | FN | Cost | Note |
|---|---|---|---|---|---|
| https://github.com/thiru51/truck-failure-cost | LightGBM, threshold chosen by 5-fold CV on train | 365 | 12 | 9,650 | GitHub repo |
| same | Random forest | 420 | 14 | 11,200 | |
| same | Logistic regression | 450 | 20 | 14,500 | |
| same | Never inspect / inspect all | - | - | 187,500 / 156,250 | trivial bounds |
| https://medium.com/analytics-vidhya/aps-component-failure-classification-in-scania-trucks-a971fcf71846 | RF with RF-imputed data, threshold tuned on CV | - | - | 9,510 | blog post |
| same | RF, median imputation | 792 | 4 | 9,920 | blog post |
| https://github.com/HoraEmbedded/pdm-aps-scania | Gradient boosting, threshold frozen before test | - | - | 11,370 | same repo reports 10,060 at the hindsight-optimal threshold |
| https://github.com/nipunmanral/Classification-APS-Failure-at-Scania-Trucks | RBF-SVM, full training set / down-sampled | - | - | 12,260 / 14,090 | class project |

Takeaway: threshold choice moves the cost by roughly 1,000+ (11,370 vs 10,060 in one repo). Any result below 9,920 should be checked for a threshold tuned on the test set.

---

## 5. BAF Base

**Paper:** Jesus et al., *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*, NeurIPS 2022 Datasets & Benchmarks. https://arxiv.org/abs/2211.13358
**Data:** https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 · Code: https://github.com/feedzai/bank-account-fraud

**Dataset facts (paper):** 1,000,000 applications per variant, 30 features, 8 months (the `month` column). Fraud prevalence by month is 0.85%-1.5%. Base variant group prevalence: 0.9% for the 77% majority age group and 1.8% for the 23% minority group (age > 50 is the sensitive split).

**Official protocol (paper):** temporal split, with the first 6 months for training and the last 2 for testing. Metric: **TPR at 5% FPR**. Fairness metric: ratio of group FPRs (predictive equality).

**Published baseline:** the paper trains 100 randomly searched LightGBM configurations per variant and plots recall against fairness in Figure 1. The recall axis spans about 0.2-0.7 across variants, and the paper reports **no tabulated headline number for Base** in its text. The associated notebook in the GitHub repo is the place to read exact values **[unverified]**. The paper does state that the top Base models sit around 0.3 on the FPR-ratio fairness axis.

**Related numbers from the appendix (original private data, not Base):**

- Train and test on the original (private) data: 75.4% TPR at 5% FPR.
- With Laplacian noise at the chosen scale (1e-4): about 70% TPR on the noisy test set.
- Best-model recall at 5% FPR in the feature-overlap study: Random Forest 69.1%, Logistic Regression 62.9%, Decision Tree 60.6%.
- Generative-model selection (best CTGAN): train and test on generated data 56.0%, train on generated and test on original 62.7%, train on original and test on generated 37.2%.

**Independent paper using Base (not comparable to the official protocol):**
Sun et al., *Objective over Architecture: Fraud Detection Under Extreme Imbalance in Bank Account Opening*, Computation 2025, 13(12):290. https://www.mdpi.com/2079-3197/13/12/290

- Random stratified 80/20 split (not temporal). Base fraud prevalence 1.10% (11,029 fraud, 988,971 non-fraud).
- Classical models (LR, RBF-SVM, RF, LightGBM) tuned on weighted F1 gave only 1-6% fraud recall at threshold 0.5. The GRU with class-weighted loss reached AUC 0.8800, 78% recall and about 5% precision.

---

## 6. Online Shoppers

**Dataset:** UCI #468 (https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). 12,330 sessions, 18 features, 84.5% negative (10,422) and 15.5% positive (1,908).
**Original paper:** Sakar et al., *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks*, Neural Computing and Applications, 2019. The abstract says the MLP gives significantly higher accuracy and F1 than RF and SVM, but I could not retrieve the exact figures **[unverified]**. https://csuepress.columbusstate.edu/bibliography_faculty/2748

### 6.1 CLIMB benchmark values (preferred reference; protocol in Section 8)

Dataset name in CLIMB: `online-shoppers-intention` (12,330 samples, IR 5.46, in the 5-10 imbalance group).

| Metric | No-balancing base (decision tree) | Best ensemble methods |
|---|---|---|
| AUPRC (×100) | 38.1 | about 50 (SPE 50.0, BC 50.4, SMOTEBagging 50.5, EE 48.1, UnderBagging 48.1, RUSBoost 47.7, BRF 46.2) |

**[extraction caveat]** These come from a garbled PDF table (CLIMB Table 5). Several cells were dropped, so the method-to-column mapping for the later columns is best-effort. Verify against the paper.

### 6.2 Other published numbers (protocols differ; not directly comparable)

| Source | Reported result | Protocol / caveat |
|---|---|---|
| ASEAN J. Sci. Tech. Report 2026, 29(1), e260159 (https://ph02.tci-thaijo.org/index.php/tsujournal/article/view/260159) | SVC: ROC-AUC 0.886, F1 0.633 | Independent held-out test set |
| GitLab project listing (https://gitlab.com/explore/projects/topics/juypter) | XGBoost: ROC-AUC 0.936, F1 0.71 at a 0.30 threshold | Repo description, not peer-reviewed |
| Springer, *Modeling online customer purchase intention...* (https://link.springer.com/article/10.1007/s44163-023-00086-0) | Best accuracy 92.39%, F-score 0.924 on a Z-score + Gain Ratio transformed subset; it cites earlier work with gradient boosting + RF at 90.34% and RF at 89.50% train / 87.50% test | Likely CV-based; protocol not checked |
| Agustyaningrum et al., JPPI (https://jurnal-ppi.kominfo.go.id/index.php/jppi/article/view/341) | Deep NN: accuracy 98.48%, F1 95.06%, precision 97.36%, recall 96.81%, AUC 96.81% | 10-fold CV. **Far above all other results; leakage such as resampling before CV is plausible [inferred, protocol not checked].** |
| Shamra Academia summary (https://shamra-academia.com/en/show/243a63daeae143) | CatBoost F1 0.93 and XGBoost F1 0.92; CatBoost ROC-AUC 0.985 | Source paper not identified; protocol unknown |

### 6.3 SOS reference and implementation audit

DATASETS.MD records SOS Table2 targets 0.601 ± 0.039 (no oversampling), 0.648 ± 0.004 (SMOTE), and 0.675 ± 0.004 (SOS-VE). The pinned SOS code uses inverse-support F1, not sklearn support-weighted F1; binary classifier scores are averaged per repetition, with training RNG fixed at2021 and five evaluation repetitions. Exact paper-to-code metric/seed correspondence remains unresolved. These targets are not directly comparable to controlled raw-UCI AP or conventional weighted-F1 results. See reports/reproduction.md and results/published_targets.csv.

---

## 7. Satimage (Statlog Landsat)

**Dataset:** UCI #146 (https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite). Official files: `sat.trn` (4,435 rows) and `sat.tst` (2,000 rows), 36 features, 6 classes (there is no class 6). The dataset documentation says **not to use cross-validation**: train once and test once on the given files (https://axon.cs.byu.edu/~martinez/classes/470/MLDB/statlog/satimage/sat.doc).
Full-data class sizes (6,435 total): red soil 1,533; cotton crop 703; grey soil 1,358; damp grey soil 626; vegetation stubble 707; very damp grey soil 1,508.

**Published baselines (Statlog `.names` file, success rate on test):**

| Algorithm | Train | Test |
|---|---|---|
| KNN | 91.1 | **90.6** |
| LVQ | ? | 89.5 |

Source: https://webdocs.cs.ualberta.ca/~rgreiner/RESEARCH/PAPERS/accurate/UCIData/satimage.names (Dipol92 is listed but its numbers are blank in the excerpt I retrieved).

No modern peer-reviewed number on this exact split was found **[unverified]**. No published binary-imbalanced protocol on `sat.trn`/`sat.tst` was found either. The CLIMB "Satellite" dataset (5,100 rows, IR 67) is a different variant and does not map onto these files.

SOS Table2 also reports Satimage SMOTE0.846 ± 0.005 and SOS-VP0.857 ± 0.006 under its named weighted-F1 metric. Audited multiclass evaluation uses inverse-support F1 and only decision tree/MLP. This is not the binary four-classifier recipe or a verified reproduction on the official UCI split; see reports/reproduction.md.

---

## 8. CLIMB benchmark

**Paper:** Liu et al., *CLIMB: Class-imbalanced Learning Benchmark on Tabular Data*, arXiv 2505.17451 (venue **[unverified]**). https://arxiv.org/abs/2505.17451 · Code: https://github.com/ZhiningLiu1998/imbalanced-ensemble

### 8.1 Protocol

- 73 naturally imbalanced real-world datasets from OpenML, with IR from 2.1 to 577.9. Selection criteria: IR > 2, no missing values, i.i.d., not deterministic targets, not trivially solvable (decision-tree AUPRC ≤ 0.95), and documented.
- 29 algorithms, grouped as follows.
  - **Undersampling:** RUS, CC, IHT, NM.
  - **Cleaning:** TL, ENN, RENN, AKNN, OSS, NCR.
  - **Oversampling:** ROS, SMOTE, BSMT, SVM-SMOTE, ADASYN.
  - **Undersample ensembles:** SPE, BC, BRF, EE, RUSBoost, UnderBagging.
  - **Oversample ensembles:** OverBoost, SMOTEBoost, OverBagging, SMOTEBagging.
  - **Cost-sensitive:** CS, AdaCost, AdaUBoost, AsymBoost.
- Preprocessing: numeric features standardized. Binary categoricals ordinal-encoded, other categoricals one-hot encoded.
- Split: 5-fold stratified, with the mean over folds reported.
- Base classifier: decision tree. Ensembles have 100 estimators. Hyperparameters tuned with Optuna (TPE, 100 trials, 5-fold stratified within each trial, AUPRC objective) for 23 of the 29 methods.
- Metrics: AUPRC (AP), macro-F1, balanced accuracy (BAC), in ×10⁻² units.
- Per-dataset results are in the paper appendix: Table 5 (AUPRC), Table 6 (macro-F1), Table 7 (BAC) and Table 8 (runtime).

### 8.2 Group-averaged scores (paper Table 2), selected methods

Dataset groups by IR: low (<5, 28 datasets), medium (5-10, 24), high (10-50, 15), extreme (>50, 6). Counts come from the paper's dataset table.

**AUPRC (×100)**

| IR group | Base | SPE | BC | BRF | EE | UnderBag (UBA) | SMOTEBag (SMBA) |
|---|---|---|---|---|---|---|---|
| <5 | 51.0 | 59.3 | 57.8 | 57.7 | 59.0 | 59.0 | 58.9 |
| 5-10 | 50.9 | 64.6 | 62.7 | 60.5 | 62.4 | 62.4 | 62.8 |
| 10-50 | 34.9 | 47.1 | 41.4 | 38.4 | 41.9 | 41.9 | 46.0 |
| >50 | 42.6 | 57.5 | 50.1 | 32.9 | 35.5 | 35.5 | 56.0 |

**Macro-F1 (×100)**

| IR group | Base | SPE | BC | BRF | EE | UBA | SMBA |
|---|---|---|---|---|---|---|---|
| <5 | 72.0 | 77.9 | 76.8 | 76.7 | 78.0 | 78.0 | 77.4 |
| 5-10 | 74.7 | 79.7 | 78.7 | 76.7 | 78.7 | 78.7 | 78.5 |
| 10-50 | 61.6 | 65.5 | 62.1 | 59.7 | 62.9 | 62.9 | 65.8 |
| >50 | 74.0 | 74.7 | 68.6 | 56.3 | 59.8 | 59.8 | 76.6 |

**Balanced accuracy (×100)**

| IR group | Base | SPE | BC | BRF | EE | UBA | SMBA |
|---|---|---|---|---|---|---|---|
| <5 | 72.1 | 78.5 | 77.3 | 79.7 | 79.7 | 79.7 | 77.0 |
| 5-10 | 74.7 | 82.1 | 80.8 | 83.1 | 82.9 | 82.9 | 77.7 |
| 10-50 | 61.8 | 70.7 | 67.8 | 72.2 | 72.1 | 72.1 | 65.1 |
| >50 | 74.6 | 85.9 | 83.0 | 88.0 | 87.3 | 87.3 | 74.4 |

**[extraction caveat]** Values were read from a PDF table with many columns. Rows were checked for having 30 values each. The **EE and UBA columns are identical in every row I parsed**, which is suspicious: they may be a duplicated column in the source table, or an extraction artifact. Please confirm against the paper before citing.

### 8.3 Dataset-level values relevant to this project (AUPRC ×100, Table 5)

| Dataset (IR) | Base | SPE | Note |
|---|---|---|---|
| online-shoppers-intention (5.46) | 38.1 | 50.0 | see Section 6.1 caveat |
| CreditCardFraudDetection (577.88) | 57.5 | 77.6 | best-effort extraction |

### 8.4 Paper's qualitative findings

- Naive rebalancing (under-/over-sampling, cost-sensitive reweighting) often hurts relative to no balancing, especially under extreme imbalance. Cleaning methods are more stable.
- Ensembles are the most consistently strong. Undersample ensembles give a good performance/runtime trade-off.
- Different metrics can lead to different conclusions. For IR > 50, undersampling and its ensembles improve BAC but degrade AUPRC and F1.
- Adding 10% label noise or 30% missing features hurts the base model about as much as raising the imbalance ratio by 500%.

---

## 9. Open items / not found

- **MGVAE exact splits and binarization** (Musk, ISOLET, Water Quality): not stated in the paper. Check the authors' code.
- **BAF Base numeric baselines under the official protocol:** shown only in a figure. Read the repo notebook.
- **Sakar et al. 2019 exact accuracy / F1:** not retrieved.
- **Modern Satimage results on `sat.trn`/`sat.tst`:** none found.
- **TILBench** (arXiv 2605.14915), a recent tabular imbalance benchmark: I saw only its reference list, so I did not extract anything from it.
- **CIFAR comparison table:** from a 2026 arXiv preprint whose review status I could not confirm. Original-paper numbers were not individually re-checked.

---

## 10. Source list

| # | Source | URL |
|---|---|---|
| 1 | Ai et al., MGVAE, AISTATS 2023 (arXiv) | https://arxiv.org/abs/2302.10910 |
| 2 | MGVAE (PMLR 206) | https://proceedings.mlr.press/v206/ai23a.html |
| 3 | Chen et al., Loss-landscape long-tail paper (2026 preprint) | https://arxiv.org/pdf/2603.21217 |
| 4 | You Only Need End-to-End Training for Long-Tailed Recognition | https://arxiv.org/pdf/2112.05958 |
| 5 | DiffuLT (NeurIPS 2024) | https://arxiv.org/pdf/2403.05170 |
| 6 | UCI APS Failure at Scania Trucks | https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks |
| 7 | IDA 2016 Challenge description | https://people.dsv.su.se/~isak-kar/IDA2016Challenge.txt |
| 8 | Jose & Gopakumar, URSI AP-RASC 2019 | https://2021.ursi.org/proceedings/procAP19/papers2019/AnImprovedRandomForestAlgorithmforclassificationinanimbalanceddataset.pdf |
| 9 | thiru51 truck-failure-cost repo | https://github.com/thiru51/truck-failure-cost |
| 10 | HoraEmbedded pdm-aps-scania repo | https://github.com/HoraEmbedded/pdm-aps-scania |
| 11 | Medium: APS component failure classification | https://medium.com/analytics-vidhya/aps-component-failure-classification-in-scania-trucks-a971fcf71846 |
| 12 | nipunmanral APS repo | https://github.com/nipunmanral/Classification-APS-Failure-at-Scania-Trucks |
| 13 | Jesus et al., BAF (NeurIPS 2022) | https://arxiv.org/abs/2211.13358 |
| 14 | BAF Kaggle | https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 |
| 15 | BAF GitHub | https://github.com/feedzai/bank-account-fraud |
| 16 | Sun et al., BAF Base GRU study (Computation 2025) | https://www.mdpi.com/2079-3197/13/12/290 |
| 17 | UCI Online Shoppers | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| 18 | Sakar et al. 2019 abstract | https://csuepress.columbusstate.edu/bibliography_faculty/2748 |
| 19 | ASEAN J. Sci. Tech. Report (SVC result) | https://ph02.tci-thaijo.org/index.php/tsujournal/article/view/260159 |
| 20 | Springer: Modeling online customer purchase intention | https://link.springer.com/article/10.1007/s44163-023-00086-0 |
| 21 | Agustyaningrum et al., JPPI | https://jurnal-ppi.kominfo.go.id/index.php/jppi/article/view/341 |
| 22 | GitLab topic listing (XGBoost project) | https://gitlab.com/explore/projects/topics/juypter |
| 23 | Shamra Academia summary | https://shamra-academia.com/en/show/243a63daeae143 |
| 24 | UCI Statlog (Landsat Satellite) | https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite |
| 25 | Statlog satimage `.names` (KNN 90.6, LVQ 89.5) | https://webdocs.cs.ualberta.ca/~rgreiner/RESEARCH/PAPERS/accurate/UCIData/satimage.names |
| 26 | Statlog `sat.doc` (no-CV note) | https://axon.cs.byu.edu/~martinez/classes/470/MLDB/statlog/satimage/sat.doc |
| 27 | Liu et al., CLIMB benchmark | https://arxiv.org/abs/2505.17451 |
| 28 | CLIMB code (imbalanced-ensemble) | https://github.com/ZhiningLiu1998/imbalanced-ensemble |
| 29 | Water Quality Kaggle candidate | https://www.kaggle.com/datasets/mssmartypants/water-quality |
| 30 | Water Quality project README (feature list) | https://github.com/praneeth-motapally/Water_Quality_Prediction |
| 31 | Water potability dataset (non-matching) | https://dspace.unia.es/handle/10334/9193 |