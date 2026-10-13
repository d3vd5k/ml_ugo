# U2 learning-benefit candidate selection — pool-utility discarded by user

Run: `results/runs/20261009T120039-97c3231e6f6b`. Five paired seeds, actual downstream sklearn logistic regression, Shoppers only. **U5+F1 is retained; Phase 1 stays open and pool-utility is discarded by the user.**

## Findings

- **Utility does not meet the AP-based retention criterion against pool-random.** On the identical initial pools, mean validation AP decreases and both ordinary/balanced validation cross-entropy increase. The full iterative utility trajectory also loses mean validation/test AP and minority F1/MCC against random, although test recall, balanced test loss and diagnostic cost improve. No single-metric dominance is claimed.
- The local estimator is accurate: all measured one-step signs agree and rank correlations are near one. Utility also slightly improves mean guidance loss after full retraining, but that training-information gain does not transfer into better independent-validation performance.
- Matched parent duplication has better mean initial validation AP than new coordinates for both selectors, especially utility. This provides no evidence that utility-selected coordinates help beyond changing parent weights.
- Utility selects more tightly spaced coordinates on these matched pools. Exact synthetic duplicates are removed, but geometric redundancy remains plausible; nearest-neighbor spacing is a diagnostic, not a causal identification of the failure.
- Operational iterative wall times are comparable between selectors here. Compute is not the primary observed failure; selection/retraining transfer and validation performance are. Both pool controls cost more than untouched U2 because K=5 and one-per-anchor feasibility require more rounds.
- These are descriptive repeated-development comparisons on a shared outer holdout, not independent folds or a confirmatory generalization claim.

## Preserved baseline and data access

Original U2 remains unchanged without the optional selector hook. Its KMeans/stability group construction, real-only anchor difficulty, anchor selection, group/seed caps, original interpolation with partner categorical projection, nearest-real neighbor detection/relocation, retraining procedure and stopping are reused. No U1 OOF scoring, FlowGMM/F1/F2, HDBSCAN/H1 or surrogate classifier is added. The retained composition/configuration is preserved separately in `configs/ablations/stage1_u5_f1.yaml`.
Canonical outer split uses seed 2026. Validation is the original stratified 20% of outer training. The remaining inner fitting partition is split 80/20 into FIT/G using seed+104729. FIT has 6,312 rows, G 1,579, validation 1,973 and test 2,466. G is labeled training information for selection, not an unbiased evaluation set. Preprocessing, real classifier fitting, grouping, anchors, partners and relocation references use FIT only. No G observations or descendants enter classifier fitting. Validation never joins FIT; there is no full-outer refit.
Ten original joint U2 learner/group-alpha proposals are selected on validation AP, then the same learner/alpha is locked across all controls for each seed. Each control sees the same role identities and fitting-only transform. Historical U2/U5/F1 results using a different fitting access policy are not pooled into this comparison.
An initial integration smoke incorrectly queried 120 official-test rows. Review caught and corrected the smoke splitter before the five-seed study: the final smoke draws all four roles from outer training, and a regression plus independent replay proves zero official-test access. The earlier smoke is not used as scientific evidence. Full-study test predictions never select settings or trajectories.

## Estimator, pool selection and count interpretation

At each live checkpoint, full logistic coefficient-and-bias gradients of class-balanced guidance CE are computed once. Every post-U2 candidate, including relocated candidates, is scored by the raw dot product with its individual unregularized CE gradient in batches. All live estimator attributes are hashed before/after; scoring never mutates them. sklearn has no persistent optimizer or normalization state; external preprocessing is fixed. The downstream solver remains original regularized L-BFGS. The estimator describes an additional plain-SGD CE step, not full retraining.
Generate K=5 independent original-generator proposals per distinct U2-planned anchor, without changing the planned anchor set or group quotas. Exact float32 final-coordinate duplicates have deterministic first-proposal ownership across the whole pool; prior accepted synthetic coordinates are also excluded. Equality with a real FIT coordinate remains valid because baseline relocation can legitimately return that point. All candidates are scored before deduplication so relocation decisions are auditable.
Utility chooses the best owned proposal per anchor, then the highest-scoring anchors within each original quota; random chooses random proposals/anchors under the same rules. Negative scores are not rejected. One acceptance per anchor per round is enforced. Same-checkpoint alternatives always have identical attainable counts; no global quota override or regeneration occurs.
Untouched U2 repeats anchors, so enforcing the requested one-per-anchor rule makes some original quotas infeasible. Initial U2 accepts 631; pool controls accept 373/400/428/391/404 in seeds 0–4. Their initial comparison is count-matched to each other, not U2. The nominal request, allocated U2 quota budget and dedup/unique-anchor quota shortfall are distinct. This report derives shortfall as allocated group quotas minus accepted; raw `shortfall` fields also include any pre-existing nominal-request underspend.
All final iterative methods reach 1,000 accepted synthetics for every seed. U2 uses two accepting rounds; both pool controls use four. The per-round allocator/stopping/cadence are unchanged, not optimized for fewer iterations. Later live checkpoints, anchor plans and pools can diverge after different batches are retrained. Equal final counts do not imply identical later pools.

## All predictive and operating-point metrics

Mean ± sample SD across five seeds (ddof=1). All 22 distinct benchmark numeric fields plus ordinary/balanced CE are reported. AP and primary_metric_value are the same verified quantity, not duplicated. Confusion counts are averaged. Lower is favorable for cross-entropy, FP, FN, FPR and diagnostic cost; thresholds/counts are descriptive. Diagnostic cost=10FP+500FN, not an established business loss. ROC recall at 5% FPR is a test/validation description, not a deployed threshold selected on test.

### Initial — validation

| Metric | U2 | Pool-random | Pool-utility |
|---|---:|---:|---:|
| AP | 0.631136 ± 0.010422 | 0.641776 ± 0.009999 | 0.637015 ± 0.010470 |
| Macro AP | 0.803640 ± 0.005114 | 0.809312 ± 0.005074 | 0.806321 ± 0.005358 |
| Minority F1 | 0.578261 ± 0.009789 | 0.565295 ± 0.009650 | 0.553176 ± 0.008546 |
| Minority precision | 0.664874 ± 0.020824 | 0.708503 ± 0.025274 | 0.695691 ± 0.020942 |
| Minority recall | 0.512131 ± 0.016621 | 0.470820 ± 0.016328 | 0.459672 ± 0.016621 |
| Macro F1 | 0.755685 ± 0.005540 | 0.750534 ± 0.005330 | 0.743672 ± 0.004418 |
| Weighted F1 | 0.878255 ± 0.002898 | 0.878502 ± 0.002669 | 0.875271 ± 0.001933 |
| SOS F1 | 0.633116 ± 0.008458 | 0.622566 ± 0.008296 | 0.612073 ± 0.007254 |
| Balanced accuracy | 0.732385 ± 0.006840 | 0.717604 ± 0.006523 | 0.711371 ± 0.006425 |
| MCC | 0.518754 ± 0.011470 | 0.518177 ± 0.011420 | 0.504599 ± 0.008185 |
| Accuracy | 0.884541 ± 0.003351 | 0.888089 ± 0.003112 | 0.885251 ± 0.002339 |
| Ordinary cross-entropy | 0.310410 ± 0.006682 | 0.300004 ± 0.006424 | 0.302599 ± 0.005492 |
| Balanced cross-entropy | 0.517831 ± 0.008041 | 0.540896 ± 0.012884 | 0.543589 ± 0.010238 |
| Realized recall | 0.512131 ± 0.016621 | 0.470820 ± 0.016328 | 0.459672 ± 0.016621 |
| Realized FPR | 0.047362 ± 0.005278 | 0.035612 ± 0.005220 | 0.036930 ± 0.004789 |
| ROC recall @5% FPR | 0.537049 ± 0.031737 | 0.558689 ± 0.031070 | 0.542951 ± 0.021945 |
| ROC attained FPR | 0.049520 ± 0.000536 | 0.049281 ± 0.000502 | 0.048681 ± 0.001072 |
| ROC FPR limit | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 |
| Classifier threshold | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| TP | 156.20 ± 5.07 | 143.60 ± 4.98 | 140.20 ± 5.07 |
| FN | 148.80 ± 5.07 | 161.40 ± 4.98 | 164.80 ± 5.07 |
| FP | 79.00 ± 8.80 | 59.40 ± 8.71 | 61.60 ± 7.99 |
| TN | 1,589.00 ± 8.80 | 1,608.60 ± 8.71 | 1,606.40 ± 7.99 |
| Diagnostic cost | 75,190.00 ± 2,476.94 | 81,294.00 ± 2,427.57 | 83,016.00 ± 2,467.87 |

### Initial — test

| Metric | U2 | Pool-random | Pool-utility |
|---|---:|---:|---:|
| AP | 0.586449 ± 0.006829 | 0.598178 ± 0.005359 | 0.595698 ± 0.005234 |
| Macro AP | 0.778845 ± 0.003932 | 0.785456 ± 0.002847 | 0.783394 ± 0.003024 |
| Minority F1 | 0.548213 ± 0.007217 | 0.533447 ± 0.014945 | 0.524445 ± 0.010225 |
| Minority precision | 0.633672 ± 0.009607 | 0.689529 ± 0.011957 | 0.673786 ± 0.011783 |
| Minority recall | 0.483246 ± 0.011795 | 0.435079 ± 0.016073 | 0.429319 ± 0.009618 |
| Macro F1 | 0.738390 ± 0.003854 | 0.733004 ± 0.008234 | 0.727694 ± 0.005763 |
| Weighted F1 | 0.869648 ± 0.001678 | 0.870735 ± 0.003617 | 0.867974 ± 0.002697 |
| SOS F1 | 0.607132 ± 0.006168 | 0.595272 ± 0.012865 | 0.587414 ± 0.008842 |
| Balanced accuracy | 0.715999 ± 0.005026 | 0.699593 ± 0.008159 | 0.695610 ± 0.005223 |
| MCC | 0.484246 ± 0.007068 | 0.486322 ± 0.015170 | 0.474573 ± 0.011412 |
| Accuracy | 0.876642 ± 0.001657 | 0.882157 ± 0.002876 | 0.879400 ± 0.002375 |
| Ordinary cross-entropy | 0.334289 ± 0.001720 | 0.321504 ± 0.000365 | 0.322969 ± 0.000682 |
| Balanced cross-entropy | 0.556756 ± 0.007485 | 0.579632 ± 0.005737 | 0.580285 ± 0.005509 |
| Realized recall | 0.483246 ± 0.011795 | 0.435079 ± 0.016073 | 0.429319 ± 0.009618 |
| Realized FPR | 0.051248 ± 0.002744 | 0.035893 ± 0.001332 | 0.038100 ± 0.001463 |
| ROC recall @5% FPR | 0.477487 ± 0.008603 | 0.501047 ± 0.009899 | 0.496335 ± 0.009899 |
| ROC attained FPR | 0.049328 ± 0.000626 | 0.049520 ± 0.000401 | 0.049520 ± 0.000401 |
| ROC FPR limit | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 |
| Classifier threshold | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| TP | 184.60 ± 4.51 | 166.20 ± 6.14 | 164.00 ± 3.67 |
| FN | 197.40 ± 4.51 | 215.80 ± 6.14 | 218.00 ± 3.67 |
| FP | 106.80 ± 5.72 | 74.80 ± 2.77 | 79.40 ± 3.05 |
| TN | 1,977.20 ± 5.72 | 2,009.20 ± 2.77 | 2,004.60 ± 3.05 |
| Diagnostic cost | 99,768.00 ± 2,212.86 | 108,648.00 ± 3,074.13 | 109,794.00 ± 1,852.95 |

### Iterative — validation

| Metric | U2 | Pool-random | Pool-utility |
|---|---:|---:|---:|
| AP | 0.633375 ± 0.011051 | 0.638651 ± 0.008477 | 0.631087 ± 0.010893 |
| Macro AP | 0.804938 ± 0.005408 | 0.807797 ± 0.003992 | 0.803302 ± 0.005637 |
| Minority F1 | 0.613643 ± 0.009363 | 0.616077 ± 0.011539 | 0.600895 ± 0.011420 |
| Minority precision | 0.652610 ± 0.024907 | 0.660920 ± 0.029082 | 0.635812 ± 0.027331 |
| Minority recall | 0.580328 ± 0.025502 | 0.578361 ± 0.026291 | 0.570492 ± 0.018107 |
| Macro F1 | 0.773752 ± 0.005223 | 0.775453 ± 0.006664 | 0.766106 ± 0.007071 |
| Weighted F1 | 0.884359 ± 0.003041 | 0.885555 ± 0.003885 | 0.880239 ± 0.004372 |
| SOS F1 | 0.663144 ± 0.008041 | 0.665352 ± 0.009993 | 0.651974 ± 0.010052 |
| Balanced accuracy | 0.761747 ± 0.009454 | 0.761842 ± 0.009930 | 0.755210 ± 0.007128 |
| MCC | 0.549564 ± 0.010794 | 0.553492 ± 0.013691 | 0.533868 ± 0.015040 |
| Accuracy | 0.887076 ± 0.004108 | 0.888596 ± 0.004958 | 0.882818 ± 0.005381 |
| Ordinary cross-entropy | 0.328343 ± 0.008096 | 0.329130 ± 0.007569 | 0.336198 ± 0.006938 |
| Balanced cross-entropy | 0.482623 ± 0.006469 | 0.479663 ± 0.009011 | 0.481720 ± 0.004653 |
| Realized recall | 0.580328 ± 0.025502 | 0.578361 ± 0.026291 | 0.570492 ± 0.018107 |
| Realized FPR | 0.056835 ± 0.008235 | 0.054676 ± 0.008957 | 0.060072 ± 0.008003 |
| ROC recall @5% FPR | 0.546230 ± 0.031585 | 0.552787 ± 0.035585 | 0.531148 ± 0.027136 |
| ROC attained FPR | 0.049520 ± 0.000328 | 0.049161 ± 0.001341 | 0.049281 ± 0.000657 |
| ROC FPR limit | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 |
| Classifier threshold | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| TP | 177.00 ± 7.78 | 176.40 ± 8.02 | 174.00 ± 5.52 |
| FN | 128.00 ± 7.78 | 128.60 ± 8.02 | 131.00 ± 5.52 |
| FP | 94.80 ± 13.74 | 91.20 ± 14.94 | 100.20 ± 13.35 |
| TN | 1,573.20 ± 13.74 | 1,576.80 ± 14.94 | 1,567.80 ± 13.35 |
| Diagnostic cost | 64,948.00 ± 3,771.79 | 65,212.00 ± 3,890.80 | 66,502.00 ± 2,676.34 |

### Iterative — test

| Metric | U2 | Pool-random | Pool-utility |
|---|---:|---:|---:|
| AP | 0.588738 ± 0.006577 | 0.594553 ± 0.007115 | 0.590918 ± 0.006961 |
| Macro AP | 0.780096 ± 0.003798 | 0.783512 ± 0.003912 | 0.780793 ± 0.004101 |
| Minority F1 | 0.560968 ± 0.006905 | 0.569722 ± 0.006073 | 0.567139 ± 0.008682 |
| Minority precision | 0.598784 ± 0.012907 | 0.612772 ± 0.008219 | 0.602078 ± 0.010056 |
| Minority recall | 0.527749 ± 0.005735 | 0.532461 ± 0.010240 | 0.536126 ± 0.010857 |
| Macro F1 | 0.743029 ± 0.004326 | 0.748445 ± 0.003357 | 0.746441 ± 0.004998 |
| Weighted F1 | 0.868685 ± 0.002602 | 0.871797 ± 0.001631 | 0.870194 ± 0.002530 |
| SOS F1 | 0.617373 ± 0.006102 | 0.625093 ± 0.005223 | 0.622689 ± 0.007536 |
| Balanced accuracy | 0.731437 ± 0.003251 | 0.735376 ± 0.004388 | 0.735577 ± 0.005390 |
| MCC | 0.487777 ± 0.009047 | 0.499042 ± 0.006491 | 0.494351 ± 0.009937 |
| Accuracy | 0.872019 ± 0.003016 | 0.875426 ± 0.001754 | 0.873236 ± 0.002558 |
| Ordinary cross-entropy | 0.353066 ± 0.002112 | 0.352159 ± 0.001258 | 0.356666 ± 0.001894 |
| Balanced cross-entropy | 0.521432 ± 0.006138 | 0.516387 ± 0.005180 | 0.514431 ± 0.006625 |
| Realized recall | 0.527749 ± 0.005735 | 0.532461 ± 0.010240 | 0.536126 ± 0.010857 |
| Realized FPR | 0.064875 ± 0.003483 | 0.061708 ± 0.002685 | 0.064971 ± 0.002597 |
| ROC recall @5% FPR | 0.482199 ± 0.010536 | 0.485864 ± 0.013803 | 0.480105 ± 0.015443 |
| ROC attained FPR | 0.049520 ± 0.000626 | 0.048656 ± 0.000995 | 0.048944 ± 0.000679 |
| ROC FPR limit | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 |
| Classifier threshold | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| TP | 201.60 ± 2.19 | 203.40 ± 3.91 | 204.80 ± 4.15 |
| FN | 180.40 ± 2.19 | 178.60 ± 3.91 | 177.20 ± 4.15 |
| FP | 135.20 ± 7.26 | 128.60 ± 5.59 | 135.40 ± 5.41 |
| TN | 1,948.80 ± 7.26 | 1,955.40 ± 5.59 | 1,948.60 ± 5.41 |
| Diagnostic cost | 91,552.00 ± 1,092.83 | 90,586.00 ± 1,920.58 | 89,954.00 ± 2,066.26 |

## Matched later-checkpoint ranking comparisons

Utility-minus-random differences from the SAME frozen checkpoint, prefix and pool, with matched counts. Only rounds 2–4 are included; the initial matched pool is reported above. Average the three later checkpoints within each seed, then report five-seed mean ± sample SD. Counterfactual fits never become live.

| Metric | Random live-prefix checkpoints | Utility live-prefix checkpoints |
|---|---:|---:|
| accuracy | -0.000338 ± 0.001048 | -0.001115 ± 0.000850 |
| average_precision | -0.000635 ± 0.001706 | -0.001836 ± 0.001385 |
| balanced_accuracy | -0.001093 ± 0.002561 | -0.000570 ± 0.002059 |
| balanced_cross_entropy | -0.000583 ± 0.001860 | 0.000941 ± 0.001785 |
| cost | 333.333333 ± 732.416093 | -10.666667 ± 587.108735 |
| fn | 0.666667 ± 1.452966 | -0.066667 ± 1.164283 |
| fp | -0.000000 ± 0.666667 | 2.266667 ± 0.547723 |
| macro_average_precision | -0.000408 ± 0.000951 | -0.001156 ± 0.000848 |
| macro_f1 | -0.000889 ± 0.002443 | -0.001474 ± 0.001979 |
| mcc | -0.001697 ± 0.004802 | -0.003285 ± 0.003948 |
| minority_f1 | -0.001593 ± 0.004294 | -0.002251 ± 0.003478 |
| minority_recall | -0.002186 ± 0.004764 | 0.000219 ± 0.003817 |
| ordinary_cross_entropy | 0.001270 ± 0.000841 | 0.001819 ± 0.000904 |
| precision | -0.000770 ± 0.003658 | -0.005673 ± 0.003170 |
| realized_fpr_test | -0.000000 ± 0.000400 | 0.001359 ± 0.000328 |
| realized_recall_test | -0.002186 ± 0.004764 | 0.000219 ± 0.003817 |
| recall_at_fpr_test | -0.002623 ± 0.005710 | -0.005464 ± 0.003186 |
| roc_attained_fpr_test | -0.000040 ± 0.001013 | 0.000360 ± 0.000876 |
| roc_fpr_limit | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| sos_inverse_support_f1 | -0.001376 ± 0.003722 | -0.002011 ± 0.003014 |
| threshold | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| tn | 0.000000 ± 0.666667 | -2.266667 ± 0.547723 |
| tp | -0.666667 ± 1.452966 | 0.066667 ± 1.164283 |
| weighted_f1 | -0.000403 ± 0.001165 | -0.000937 ± 0.000943 |

## Paired utility-minus-random differences

Each cell is mean paired difference ± sample SD. Raw per-seed values are exported; neither the best seed nor pooled checkpoints replace seed-level comparisons.

| Metric | Initial validation | Initial test | Iterative validation | Iterative test |
|---|---:|---:|---:|---:|
| AP | -0.004761 ± 0.004025 | -0.002480 ± 0.002485 | -0.007564 ± 0.005851 | -0.003635 ± 0.001977 |
| Macro AP | -0.002991 ± 0.002324 | -0.002062 ± 0.001530 | -0.004495 ± 0.003463 | -0.002718 ± 0.001091 |
| Minority F1 | -0.012119 ± 0.005944 | -0.009002 ± 0.007323 | -0.015182 ± 0.007510 | -0.002583 ± 0.007552 |
| Minority precision | -0.012812 ± 0.008745 | -0.015743 ± 0.003568 | -0.025108 ± 0.009275 | -0.010694 ± 0.005716 |
| Minority recall | -0.011148 ± 0.006391 | -0.005759 ± 0.008954 | -0.007869 ± 0.010264 | 0.003665 ± 0.010071 |
| Macro F1 | -0.006862 ± 0.003328 | -0.005310 ± 0.003937 | -0.009347 ± 0.004339 | -0.002004 ± 0.004214 |
| Weighted F1 | -0.003230 ± 0.001552 | -0.002761 ± 0.001610 | -0.005316 ± 0.002198 | -0.001604 ± 0.001938 |
| SOS F1 | -0.010494 ± 0.005134 | -0.007858 ± 0.006273 | -0.013378 ± 0.006526 | -0.002404 ± 0.006516 |
| Balanced accuracy | -0.006233 ± 0.003176 | -0.003983 ± 0.004266 | -0.006632 ± 0.004994 | 0.000201 ± 0.004945 |
| MCC | -0.013578 ± 0.006684 | -0.011749 ± 0.006618 | -0.019624 ± 0.008325 | -0.004690 ± 0.008092 |
| Accuracy | -0.002838 ± 0.001416 | -0.002758 ± 0.001125 | -0.005778 ± 0.002198 | -0.002190 ± 0.001711 |
| Ordinary cross-entropy | 0.002595 ± 0.001732 | 0.001465 ± 0.000590 | 0.007068 ± 0.003781 | 0.004507 ± 0.000957 |
| Balanced cross-entropy | 0.002693 ± 0.004803 | 0.000653 ± 0.003146 | 0.002057 ± 0.007154 | -0.001956 ± 0.003065 |
| Realized recall | -0.011148 ± 0.006391 | -0.005759 ± 0.008954 | -0.007869 ± 0.010264 | 0.003665 ± 0.010071 |
| Realized FPR | 0.001319 ± 0.001367 | 0.002207 ± 0.000728 | 0.005396 ± 0.002283 | 0.003263 ± 0.001288 |
| ROC recall @5% FPR | -0.015738 ± 0.016621 | -0.004712 ± 0.007264 | -0.021639 ± 0.018196 | -0.005759 ± 0.009865 |
| ROC attained FPR | -0.000600 ± 0.001406 | 0.000000 ± 0.000480 | 0.000120 ± 0.001072 | 0.000288 ± 0.001296 |
| ROC FPR limit | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| Classifier threshold | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| TP | -3.40 ± 1.95 | -2.20 ± 3.42 | -2.40 ± 3.13 | 1.40 ± 3.85 |
| FN | 3.40 ± 1.95 | 2.20 ± 3.42 | 2.40 ± 3.13 | -1.40 ± 3.85 |
| FP | 2.20 ± 2.28 | 4.60 ± 1.52 | 9.00 ± 3.81 | 6.80 ± 2.68 |
| TN | -2.20 ± 2.28 | -4.60 ± 1.52 | -9.00 ± 3.81 | -6.80 ± 2.68 |
| Diagnostic cost | 1,722.00 ± 971.86 | 1,146.00 ± 1,701.10 | 1,290.00 ± 1,556.90 | -632.00 ± 1,918.26 |

## Counts, retraining and measured runtime

| Variant | Accepted | Accepting rounds | Own operational fits | Observed trajectory seconds | Operational seconds | Guidance + candidate scoring seconds | Duplicates in augmented training |
|---|---:|---:|---:|---:|---:|---:|---:|
| initial_U2 | 631.00 ± 0.00 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.637777 ± 0.074398 | 1.637777 ± 0.074398 | 0.000000 ± 0.000000 | 36.200000 ± 6.300794 |
| initial_pool_random | 399.20 ± 20.04 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.829147 ± 0.104023 | 1.822216 ± 0.103777 | 0.000744 ± 0.000015 | 18.200000 ± 3.346640 |
| initial_pool_utility | 399.20 ± 20.04 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.863276 ± 0.125439 | 1.856061 ± 0.125105 | 0.000739 ± 0.000032 | 24.800000 ± 4.024922 |
| iterative_U2 | 1,000.00 ± 0.00 | 2.000000 ± 0.000000 | 4.000000 ± 0.000000 | 1.794283 ± 0.090720 | 1.794283 ± 0.090720 | 0.000000 ± 0.000000 | 60.400000 ± 3.781534 |
| iterative_pool_random | 1,000.00 ± 0.00 | 4.000000 ± 0.000000 | 6.000000 ± 0.000000 | 2.506729 ± 0.098988 | 2.478825 ± 0.099689 | 0.002966 ± 0.000162 | 41.400000 ± 3.974921 |
| iterative_pool_utility | 1,000.00 ± 0.00 | 4.000000 ± 0.000000 | 6.000000 ± 0.000000 | 2.481436 ± 0.118662 | 2.456786 ± 0.117227 | 0.003118 ± 0.000221 | 47.200000 ± 5.540758 |

Observed trajectory time includes scoring-state checks, one-step verification and alternative-mask diagnostics. Operational time subtracts documented diagnostics; random’s utility-gradient scoring is diagnostic-only and is also subtracted. Parent-duplication/shared-checkpoint retrains run after the live trajectory and are separately charged, never fed back into it. Initial cells reuse one shared initial real model; their own fits exclude that shared fit. Common preprocessing/group fitting, canonical tuning and artifact I/O are not hidden inside per-trajectory comparisons.

| Seed | Canonical tuning fits | Shared/initial fits | Operational fits | Paired-batch diagnostic fits | Parent diagnostic fits | Total physical fits | Measured seed evaluation seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40 | 1 | 19 | 18 | 18 | 96 | 31.720 | 946.3 |
| 1 | 40 | 1 | 19 | 18 | 18 | 96 | 33.559 | 951.0 |
| 2 | 40 | 1 | 19 | 18 | 18 | 96 | 32.198 | 949.8 |
| 3 | 40 | 1 | 19 | 18 | 18 | 96 | 33.151 | 952.0 |
| 4 | 40 | 1 | 19 | 18 | 18 | 96 | 29.599 | 955.4 |

The full CLI completed in 163.80 seconds. Seed evaluation excludes earlier dataset acquisition, split/preprocessing work and source snapshot creation; it includes canonical tuning, all trajectories, saved same-checkpoint/parent controls and artifact writes. RSS is sampled per seed at 10ms and attributed to all of that seed’s records, not an isolated model-memory delta. Both unadjusted and diagnostic-adjusted times are retained. The regression suite ran concurrently during part of the study; these are observed development wall times, not isolated performance estimates or evidence that utility is faster.

## One-step accuracy and live-state invariance

Independently replayed 45 saved checkpoints and 2789 candidate plain-SGD steps at eta=1e-4. Spearman range 0.999817–1.000000; sign agreement is 100% throughout. No full-study checkpoint needed the fixed eta=1e-5 fallback. Maximum absolute first-order error is 2.45e-07. A separate small real-Shoppers probe also evaluated eta=1e-5, and a regression exercises the failed-approximation fallback without validation/test tuning.
Full-parameter guidance/candidate derivatives were replayed with independent torch autograd; all final-coordinate dot products, original real-only checkpoint fits and state hashes pass. All live model attributes are invariant; original U2 output coordinates/labels/ancestry/quotas/stopping/predictions reproduce its pinned pre-hook source exactly.

## Matched parent duplication and batch redundancy

Each candidate’s original anchor and partner receive weight 0.5 each in the duplication control. The prefix is the identical FIT plus previously accepted live synthetics, each weight one. Total added weight is exactly the selected batch count, using the same LR settings and fresh-retraining procedure. Both alternative selectors are replayed on every saved checkpoint/pool; parent controls never become live checkpoints.

| Source | Selector | Validation AP: actual − parents | Ordinary validation CE: actual − parents | Balanced validation CE: actual − parents |
|---|---|---:|---:|---:|
| initial_shared_pool | random | -0.001981 ± 0.001911 | 0.000105 ± 0.000639 | 0.002268 ± 0.001507 |
| initial_shared_pool | utility | -0.005960 ± 0.003145 | 0.001743 ± 0.001168 | 0.005496 ± 0.003485 |
| iterative_pool_random/pools | random | -0.001439 ± 0.001137 | 0.000015 ± 0.000372 | 0.002018 ± 0.001122 |
| iterative_pool_random/pools | utility | -0.002962 ± 0.001603 | 0.000876 ± 0.000623 | 0.002896 ± 0.001918 |
| iterative_pool_utility/pools | random | -0.001660 ± 0.000834 | -0.000026 ± 0.000275 | 0.002027 ± 0.001062 |
| iterative_pool_utility/pools | utility | -0.003557 ± 0.001657 | 0.001056 ± 0.000799 | 0.003377 ± 0.001993 |

Later-row statistics average each seed’s four checkpoints before computing the five-seed mean/SD. They are not 20 independent comparisons. Full actual/parent metrics and every matched weight/prefix are in the CSVs and per-checkpoint artifacts.

| Source | Selector | Mean selected score | Mean within-batch nearest-neighbor distance | Selected real matches | Selected negative scores | Allocated quota shortfall |
|---|---|---:|---:|---:|---:|---:|
| initial_shared_pool | random | 0.805158 ± 0.064525 | 1.524761 ± 0.068908 | 18.200000 ± 3.346640 | 0.200000 ± 0.447214 | 231.80 ± 20.04 |
| initial_shared_pool | utility | 0.929261 ± 0.069376 | 1.263683 ± 0.051615 | 24.800000 ± 4.024922 | 0.000000 ± 0.000000 | 231.80 ± 20.04 |
| iterative_pool_random/pools | random | 0.508629 ± 0.050012 | 2.252746 ± 0.141225 | 10.350000 ± 0.993730 | 0.250000 ± 0.353553 | 111.25 ± 13.39 |
| iterative_pool_random/pools | utility | 0.606799 ± 0.061085 | 2.054449 ± 0.141881 | 14.600000 ± 1.781853 | 0.050000 ± 0.111803 | 111.25 ± 13.39 |
| iterative_pool_utility/pools | random | 0.475145 ± 0.046874 | 2.220329 ± 0.160929 | 10.400000 ± 1.654539 | 0.100000 ± 0.223607 | 110.30 ± 13.13 |
| iterative_pool_utility/pools | utility | 0.560112 ± 0.051378 | 2.040306 ± 0.166651 | 11.800000 ± 1.385190 | 0.000000 ± 0.000000 | 110.30 ± 13.13 |

Distance uses final FIT-preprocessed coordinates, not raw units or a claim of semantic diversity. Coordinate dedup eliminates exact repeated synthetics, but legitimate relocated real matches remain and contribute duplicates against FIT. Utility chooses more tightly spaced batches on average here. This is consistent with redundancy, but cannot distinguish redundancy from the objective/solver mismatch without another controlled ablation. No extra complexity is added to explain the failure.

### Guidance-loss transfer (training diagnostic only)

| Source | Selector | Actual full-retrain guidance loss reduction | Parent full-retrain guidance loss reduction |
|---|---|---:|---:|
| initial_shared_pool | random | 0.084059 ± 0.008509 | 0.088480 ± 0.008129 |
| initial_shared_pool | utility | 0.084351 ± 0.012163 | 0.090216 ± 0.009606 |
| iterative_pool_random/pools | random | 0.036008 ± 0.003372 | 0.039104 ± 0.002778 |
| iterative_pool_random/pools | utility | 0.037240 ± 0.004720 | 0.040431 ± 0.003185 |
| iterative_pool_utility/pools | random | 0.035857 ± 0.002669 | 0.039019 ± 0.002342 |
| iterative_pool_utility/pools | utility | 0.036666 ± 0.003455 | 0.039944 ± 0.002484 |

Guidance labels influence ranking. These losses diagnose transfer to the full-training update; they are not evaluation evidence. The independent validation/test results, not guidance improvement or approximation accuracy, govern retention.

## Evidence and reproduction

All **241 tests passed**. The corrected real training-only smoke `results/smoke/20261009T115900-73388c6a3fe0` completed six records; the full study completed 30/30. Independent full replay passed **9,689 checks**, covering all 1,440 final validation/test numeric metrics, 45 saved pools/checkpoints, 90 same-pool matched selector/parent comparisons, frozen state, fitting-only preprocessing/ancestry and unchanged U2 quotas/anchors. AP aliases are checked separately.
```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.learning_benefit --run
```
Use `--smoke --seeds 0` for the training-only integration surface. Experiment declaration: `configs/ablations/stage1_learning_benefit.yaml`. Existing sklearn/SciPy iprint OptimizeWarnings were observed and not suppressed.

Machine-readable outputs in the run directory: `all_metrics.csv`, `metric_summary.csv`, `paired_metric_deltas.csv`, `paired_differences.json`, `matched_checkpoint_deltas.csv`, `matched_checkpoint_summary.csv`, `selection_diagnostics.csv`, `one_step_diagnostics.csv`, `parent_comparisons.csv`, `parent_metric_summary.csv`, `guidance_retraining.csv`, `resource_diagnostics.csv`, `group_generation.csv`, and `independent_verification.json`.

Per-seed source manifests, raw-role identities, FIT-only preprocessor, chosen LR/group settings, final models/predictions/training arrays, original/final K=5 pools/IDs/masks, checkpoint coefficients/vG and equal-weight parent arrays are retained.

**User decision: pool-utility discarded.** U5+F1 is retained. Historical utility results remain archived; pool-random+F1 is the next experiment, with retention pending. Phase 1 is not concluded/frozen; broader or Stage 2 evaluation remains closed.
