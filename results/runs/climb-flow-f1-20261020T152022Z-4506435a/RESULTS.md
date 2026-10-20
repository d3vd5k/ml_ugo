# Native CLIMB UGO versus U2 + FlowGMM

Mode: **run**. Independent artifact verification: **passed** (120 cells).
Datasets: ozone-level-8hr, wilt, mammography, dis, Satellite. Classifiers: mlp, logistic_regression, xgboost, random_forest. Seeds: 42, 43, 44.

Minority F1 is the primary outcome. Thresholds maximize inner-validation F1 over whole tied-probability groups; exact ties choose closest to 0.5, then higher. Outer-test probabilities never select thresholds or settings. Precision, recall, AP and runtime are secondary descriptions, not substitutes for F1.

## Equal-dataset macro averages

Each dataset has equal weight; values first average datasets within each seed. ± is sample SD across repeated stratified splits, not an independent-replicate standard error. A subset macro covers only its declared datasets, not the full five-dataset suite.

| Classifier | Method | Minority F1 ± SD | Precision | Recall | AP | Runtime s | Accepted / requested | Native fits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | original_ugo | 0.5840 ± 0.0118 | 0.5837 | 0.6186 | 0.6210 | 10.00 | 708.3 / 1000.0 | 43.1 |
| mlp | u2_flow_gate | 0.6019 ± 0.0315 | 0.6300 | 0.6098 | 0.6417 | 11.84 | 235.3 / 1000.0 | 46.0 |
| logistic_regression | original_ugo | 0.5020 ± 0.0278 | 0.5437 | 0.5123 | 0.5313 | 4.64 | 733.3 / 1000.0 | 45.3 |
| logistic_regression | u2_flow_gate | 0.5294 ± 0.0587 | 0.5644 | 0.5239 | 0.5418 | 6.08 | 236.3 / 1000.0 | 46.0 |
| xgboost | original_ugo | 0.6520 ± 0.0119 | 0.7919 | 0.6017 | 0.7321 | 14.20 | 629.3 / 1000.0 | 43.4 |
| xgboost | u2_flow_gate | 0.6171 ± 0.0362 | 0.7259 | 0.5929 | 0.7144 | 14.90 | 232.7 / 1000.0 | 46.0 |
| random_forest | original_ugo | 0.6173 ± 0.0813 | 0.7087 | 0.5716 | 0.6907 | 59.66 | 614.3 / 1000.0 | 40.9 |
| random_forest | u2_flow_gate | 0.6304 ± 0.0604 | 0.7578 | 0.5859 | 0.7143 | 57.51 | 234.3 / 1000.0 | 46.0 |

## Per-dataset native results

| Dataset | Classifier | Method | Minority F1 ± SD | Precision | Recall | AP | Runtime s | Accepted / requested | Native fits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ozone-level-8hr | mlp | original_ugo | 0.5037 ± 0.0429 | 0.4013 | 0.6771 | 0.5387 | 11.72 | 936.0 / 1000.0 | 71.0 |
| ozone-level-8hr | mlp | u2_flow_gate | 0.4930 ± 0.0654 | 0.4458 | 0.5625 | 0.5187 | 9.68 | 235.3 / 1000.0 | 55.0 |
| ozone-level-8hr | logistic_regression | original_ugo | 0.4005 ± 0.0224 | 0.3558 | 0.5000 | 0.4570 | 3.48 | 1000.0 / 1000.0 | 84.3 |
| ozone-level-8hr | logistic_regression | u2_flow_gate | 0.4629 ± 0.0139 | 0.4315 | 0.5417 | 0.4414 | 2.74 | 235.3 / 1000.0 | 55.0 |
| ozone-level-8hr | xgboost | original_ugo | 0.5268 ± 0.0501 | 0.4778 | 0.6146 | 0.5334 | 23.02 | 602.7 / 1000.0 | 65.0 |
| ozone-level-8hr | xgboost | u2_flow_gate | 0.4587 ± 0.1118 | 0.3983 | 0.5625 | 0.5003 | 18.66 | 236.7 / 1000.0 | 55.0 |
| ozone-level-8hr | random_forest | original_ugo | 0.4413 ± 0.2050 | 0.4362 | 0.4479 | 0.5009 | 156.35 | 602.7 / 1000.0 | 61.3 |
| ozone-level-8hr | random_forest | u2_flow_gate | 0.4292 ± 0.0392 | 0.4496 | 0.4271 | 0.4810 | 125.97 | 235.3 / 1000.0 | 55.0 |
| wilt | mlp | original_ugo | 0.8702 ± 0.0052 | 0.8248 | 0.9231 | 0.9528 | 10.78 | 1000.0 / 1000.0 | 54.0 |
| wilt | mlp | u2_flow_gate | 0.8814 ± 0.0211 | 0.8634 | 0.9038 | 0.9585 | 10.79 | 403.3 / 1000.0 | 55.0 |
| wilt | logistic_regression | original_ugo | 0.7094 ± 0.0401 | 0.7423 | 0.6795 | 0.6997 | 3.97 | 1000.0 / 1000.0 | 54.0 |
| wilt | logistic_regression | u2_flow_gate | 0.7034 ± 0.0477 | 0.7149 | 0.6923 | 0.7151 | 3.97 | 408.3 / 1000.0 | 55.0 |
| wilt | xgboost | original_ugo | 0.8763 ± 0.0272 | 0.8694 | 0.8846 | 0.9383 | 13.36 | 1000.0 / 1000.0 | 52.7 |
| wilt | xgboost | u2_flow_gate | 0.8537 ± 0.0147 | 0.8324 | 0.8846 | 0.9399 | 13.18 | 395.3 / 1000.0 | 55.0 |
| wilt | random_forest | original_ugo | 0.8686 ± 0.0314 | 0.8307 | 0.9103 | 0.9120 | 45.23 | 924.7 / 1000.0 | 52.7 |
| wilt | random_forest | u2_flow_gate | 0.8676 ± 0.0411 | 0.8728 | 0.8654 | 0.9361 | 45.36 | 403.7 / 1000.0 | 55.0 |
| mammography | mlp | original_ugo | 0.6279 ± 0.0794 | 0.7008 | 0.5705 | 0.6701 | 15.20 | 1000.0 / 1000.0 | 33.0 |
| mammography | mlp | u2_flow_gate | 0.6701 ± 0.0770 | 0.6916 | 0.6603 | 0.7078 | 21.64 | 357.7 / 1000.0 | 40.0 |
| mammography | logistic_regression | original_ugo | 0.6206 ± 0.0687 | 0.6305 | 0.6154 | 0.6452 | 10.66 | 1000.0 / 1000.0 | 33.0 |
| mammography | logistic_regression | u2_flow_gate | 0.6152 ± 0.0493 | 0.5944 | 0.6410 | 0.6395 | 16.41 | 352.7 / 1000.0 | 40.0 |
| mammography | xgboost | original_ugo | 0.6451 ± 0.1146 | 0.7696 | 0.5705 | 0.7206 | 18.31 | 1000.0 / 1000.0 | 33.0 |
| mammography | xgboost | u2_flow_gate | 0.6374 ± 0.1064 | 0.7115 | 0.5897 | 0.7095 | 24.05 | 346.3 / 1000.0 | 40.0 |
| mammography | random_forest | original_ugo | 0.6399 ± 0.0912 | 0.7389 | 0.5833 | 0.7050 | 47.30 | 1000.0 / 1000.0 | 33.0 |
| mammography | random_forest | u2_flow_gate | 0.6767 ± 0.0735 | 0.7163 | 0.6538 | 0.7433 | 55.72 | 343.3 / 1000.0 | 40.0 |
| dis | mlp | original_ugo | 0.3063 ± 0.0607 | 0.2659 | 0.3889 | 0.2477 | 4.59 | 0.0 / 1000.0 | 22.0 |
| dis | mlp | u2_flow_gate | 0.3131 ± 0.1225 | 0.2704 | 0.3889 | 0.2401 | 7.92 | 70.0 / 1000.0 | 40.0 |
| dis | logistic_regression | original_ugo | 0.0596 ± 0.0516 | 0.0375 | 0.1667 | 0.0510 | 1.87 | 0.0 / 1000.0 | 22.0 |
| dis | logistic_regression | u2_flow_gate | 0.1573 ± 0.1032 | 0.1556 | 0.1667 | 0.0754 | 3.00 | 73.7 / 1000.0 | 40.0 |
| dis | xgboost | original_ugo | 0.5124 ± 0.1382 | 0.9333 | 0.3611 | 0.6604 | 5.24 | 0.0 / 1000.0 | 22.0 |
| dis | xgboost | u2_flow_gate | 0.4308 ± 0.1795 | 0.8690 | 0.3056 | 0.6069 | 8.78 | 73.3 / 1000.0 | 40.0 |
| dis | random_forest | original_ugo | 0.4550 ± 0.2790 | 0.6468 | 0.3611 | 0.5359 | 8.01 | 0.0 / 1000.0 | 22.0 |
| dis | random_forest | u2_flow_gate | 0.4840 ± 0.2865 | 0.9524 | 0.3611 | 0.5902 | 16.17 | 78.0 / 1000.0 | 40.0 |
| Satellite | mlp | original_ugo | 0.6117 ± 0.0999 | 0.7257 | 0.5333 | 0.6959 | 7.71 | 605.3 / 1000.0 | 35.3 |
| Satellite | mlp | u2_flow_gate | 0.6520 ± 0.1039 | 0.8788 | 0.5333 | 0.7836 | 9.18 | 110.3 / 1000.0 | 40.0 |
| Satellite | logistic_regression | original_ugo | 0.7199 ± 0.0979 | 0.9524 | 0.6000 | 0.8036 | 3.24 | 666.7 / 1000.0 | 33.0 |
| Satellite | logistic_regression | u2_flow_gate | 0.7084 ± 0.1319 | 0.9259 | 0.5778 | 0.8378 | 4.29 | 111.3 / 1000.0 | 40.0 |
| Satellite | xgboost | original_ugo | 0.6993 ± 0.0667 | 0.9091 | 0.5778 | 0.8080 | 11.06 | 544.0 / 1000.0 | 44.3 |
| Satellite | xgboost | u2_flow_gate | 0.7050 ± 0.1258 | 0.8184 | 0.6222 | 0.8156 | 9.80 | 111.7 / 1000.0 | 40.0 |
| Satellite | random_forest | original_ugo | 0.6815 ± 0.0813 | 0.8910 | 0.5556 | 0.7998 | 41.39 | 544.0 / 1000.0 | 35.3 |
| Satellite | random_forest | u2_flow_gate | 0.6944 ± 0.1026 | 0.7977 | 0.6222 | 0.8208 | 44.32 | 111.0 / 1000.0 | 40.0 |

## Paired descriptive deltas

Deltas are U2 + FlowGMM minus original UGO, paired on dataset, classifier, seed and the exact split. Positive F1 is favorable; runtime and dose differences are descriptive.

| Classifier | Δ minority F1 ± SD | Δ precision | Δ recall | Δ AP | Δ runtime s | Δ accepted |
| --- | --- | --- | --- | --- | --- | --- |
| mlp | 0.0179 ± 0.0198 | 0.0463 | -0.0088 | 0.0207 | 1.84 | -472.9 |
| logistic_regression | 0.0274 ± 0.0330 | 0.0207 | 0.0116 | 0.0105 | 1.44 | -497.1 |
| xgboost | -0.0349 ± 0.0242 | -0.0659 | -0.0088 | -0.0177 | 0.70 | -396.7 |
| random_forest | 0.0131 ± 0.0283 | 0.0490 | 0.0143 | 0.0235 | -2.15 | -380.0 |

## Scope and interpretation

Both methods receive 10 common learner proposals and an augmentation cap of 1000, with at most 20 rounds. U2 cycles canonical alpha, locks the winning learner/alpha, then selects tau on validation F1. A common cap is **not** a matched accepted dose: no count-matched quality claim is supported.
The original UGO is the uncorrected baseline; its synthetic parents are permitted only with validated ancestry rooted in its real training role. Gated U2 uses real parents only. Unit synthetic weights and the fixed native learner protocol are recorded by the run; no historical U1, corrected UGO, downweighting or ensemble is compared.
Flow is fitted only to its declared inner real-fit core. Calibration, validation and test rows are excluded; the frozen core is shared for outer replay. Shared Flow fit seconds are exposed separately and are not summed repeatedly as independent fits.
Native test and selected pre-refit validation CSVs independently reproduce the metrics and selected threshold. Compact hashed fit/provenance audits are checked for role identity, accounting and recorded validation assertions; they are **not** a replay of generation or an independent reconstruction of omitted raw ancestry. Candidate ranking is checked against recorded candidate scores; only the selected candidate's validation probabilities are retained and independently rescored.
Seeds are descriptive repeated-split evidence, not independent significance tests. No superiority or compatibility verdict is inferred solely from mean AP. Smoke results are training-pipeline checks, not scientific evidence.

Artifacts: `verification.json`, `summary.csv`, `paired_deltas.csv`; source predictions, selections, splits and the effective configuration remain in this run folder.
