# Validation-guided U2, original data access — pool-utility discarded by user

Run: `results/runs/20261009T153348-af844bdecf5a`. 5 matched seeds, actual LR/RF/LightGBM, Shoppers only. Completed 240/240 records: four original references plus inner/outer initial and iterative controls per classifier/seed.

## Scope and interpretation

- Historical data access is restored: inner fitting 7,891, guidance/original validation 1,973, final outer fitting 9,864 and test 2,466. Inner/outer preprocessing and real generation references are fitted independently on those original partitions. Guidance is excluded from inner fitting/generation but deliberately included in outer fitting/generation.
- Guidance labels influence selection and canonical settings. Guidance metrics are training/selection feedback, **not independent validation**. Outer test never selects settings, trajectories or scorers. This is repeated exploratory development on one shared holdout, not independent-fold confirmation.
- Original UGO/U2/U5/U5+F1 references use original ten-proposal selection conventions; the published correction bundle imports 225 unchanged cells from the initial full study and freshly reruns the 15 retained U5+F1 cells on the historical auto/CUDA device; U5+F1 locks canonical U5 then selects its four compatibility taus. All three pool controls lock the same same-seed native U2 learner/group-alpha selection. The U2 replay is an exact original-source prediction check, not another independent replicate.
- LR scores full coefficient-and-bias class-balanced guidance CE dot products using its actual checkpoint. **RF/LightGBM use an explicitly user-authorized LR surrogate**, separately fitted to each native checkpoint’s exact current real-plus-synthetic training prefix. The LR settings come from same-seed canonical LR-U2 selection. Native tree predictions still determine U2 difficulty/anchors and native classifiers perform every downstream/parent refit.
- SGD verification concerns only the LR scorer, never native RF/LightGBM gradients or retraining. Accurate local surrogate estimates do not establish native tree-classifier learning benefit.
- K=5 proposals per distinct planned anchor; unchanged U2 groups/quotas/caps/interpolation/relocation/stopping/cadence. Score final post-relocation coordinates. Deduplicate final float32 coordinates consistently, allowing real-coordinate relocation matches. One acceptance per anchor, negative top-fill and explicit quota shortfalls; no regeneration or global quota override.
- Initial random/utility checkpoints and saved pools are identical with matched attainable counts. Full live pools can diverge. U2 repeats anchors; its initial accepted count need not equal pool controls. U5+F1 rejection may underspend the original proposal budget, so final reference dose must also be disclosed.
- Diagnostic cost is 10×FP+500×FN, not an established Shoppers business loss. ROC recall at 5% FPR is descriptive, not a deployed test-selected threshold. AP/primary-metric value are aliases.

## All final outer-test metrics

Mean ± sample SD across matched seeds. All 22 benchmark numeric fields plus ordinary/balanced CE are included.

### logistic_regression

| Metric | Original UGO | U2 | Original U5 | Retained U5+F1 | Pool-random | Pool-utility |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.884915 ± 0.001050 | 0.875264 ± 0.000544 | 0.874777 ± 0.000831 | 0.880373 ± 0.001647 | 0.878508 ± 0.001057 | 0.876723 ± 0.000907 |
| average_precision | 0.607707 ± 0.001577 | 0.585396 ± 0.002322 | 0.585378 ± 0.001492 | 0.596144 ± 0.001100 | 0.592631 ± 0.001426 | 0.588032 ± 0.001425 |
| balanced_accuracy | 0.715976 ± 0.005611 | 0.717321 ± 0.003135 | 0.715109 ± 0.001828 | 0.699393 ± 0.002136 | 0.716034 ± 0.002110 | 0.717758 ± 0.002408 |
| balanced_cross_entropy | 0.584263 ± 0.003460 | 0.552186 ± 0.001523 | 0.551751 ± 0.001445 | 0.595965 ± 0.007695 | 0.544489 ± 0.000714 | 0.546229 ± 0.001131 |
| cost | 101,818.00 ± 2,346.45 | 98,822.00 ± 1,333.93 | 99,716.00 ± 931.25 | 108,300.00 ± 913.59 | 100,212.00 ± 952.72 | 98,982.00 ± 955.70 |
| fn | 202.00 ± 4.74 | 195.40 ± 2.70 | 197.20 ± 1.92 | 215.00 ± 1.87 | 198.40 ± 1.95 | 195.80 ± 1.92 |
| fp | 81.80 ± 2.86 | 112.20 ± 1.92 | 111.60 ± 3.51 | 80.00 ± 4.64 | 101.20 ± 3.42 | 108.20 ± 1.92 |
| macro_average_precision | 0.790296 ± 0.000903 | 0.778694 ± 0.001237 | 0.778757 ± 0.000871 | 0.784284 ± 0.000468 | 0.782995 ± 0.000690 | 0.779639 ± 0.000759 |
| macro_f1 | 0.746462 ± 0.004519 | 0.737900 ± 0.002320 | 0.736103 ± 0.001110 | 0.731224 ± 0.002489 | 0.740221 ± 0.001802 | 0.739559 ± 0.002166 |
| mcc | 0.507335 ± 0.007212 | 0.481914 ± 0.004027 | 0.478696 ± 0.002134 | 0.480670 ± 0.005924 | 0.489141 ± 0.003676 | 0.485989 ± 0.004154 |
| minority_f1 | 0.559105 ± 0.008591 | 0.548157 ± 0.004441 | 0.544804 ± 0.002153 | 0.531006 ± 0.004203 | 0.550685 ± 0.003223 | 0.550554 ± 0.003903 |
| minority_recall | 0.471204 ± 0.012417 | 0.488482 ± 0.007073 | 0.483770 ± 0.005035 | 0.437173 ± 0.004897 | 0.480628 ± 0.005103 | 0.487435 ± 0.005035 |
| ordinary_cross_entropy | 0.325952 ± 0.003698 | 0.334213 ± 0.001825 | 0.334047 ± 0.001020 | 0.319272 ± 0.001676 | 0.333209 ± 0.000800 | 0.336334 ± 0.001008 |
| precision | 0.687585 ± 0.003663 | 0.624504 ± 0.001883 | 0.623552 ± 0.005365 | 0.676308 ± 0.011689 | 0.644737 ± 0.006496 | 0.632485 ± 0.004088 |
| realized_fpr_test | 0.039251 ± 0.001374 | 0.053839 ± 0.000923 | 0.053551 ± 0.001683 | 0.038388 ± 0.002225 | 0.048560 ± 0.001641 | 0.051919 ± 0.000923 |
| realized_recall_test | 0.471204 ± 0.012417 | 0.488482 ± 0.007073 | 0.483770 ± 0.005035 | 0.437173 ± 0.004897 | 0.480628 ± 0.005103 | 0.487435 ± 0.005035 |
| recall_at_fpr_test | 0.510471 ± 0.003702 | 0.473822 ± 0.004534 | 0.471728 ± 0.003413 | 0.506806 ± 0.005103 | 0.489005 ± 0.005675 | 0.476963 ± 0.009865 |
| roc_attained_fpr_test | 0.049520 ± 0.000215 | 0.049040 ± 0.000788 | 0.049232 ± 0.000935 | 0.049712 ± 0.000263 | 0.049808 ± 0.000215 | 0.049232 ± 0.000728 |
| roc_fpr_limit | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 |
| sos_inverse_support_f1 | 0.617151 ± 0.007329 | 0.606942 ± 0.003783 | 0.604071 ± 0.001821 | 0.593036 ± 0.003665 | 0.609405 ± 0.002777 | 0.609111 ± 0.003363 |
| threshold | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| tn | 2,002.20 ± 2.86 | 1,971.80 ± 1.92 | 1,972.40 ± 3.51 | 2,004.00 ± 4.64 | 1,982.80 ± 3.42 | 1,975.80 ± 1.92 |
| tp | 180.00 ± 4.74 | 186.60 ± 2.70 | 184.80 ± 1.92 | 167.00 ± 1.87 | 183.60 ± 1.95 | 186.20 ± 1.92 |
| weighted_f1 | 0.875773 ± 0.001722 | 0.868859 ± 0.000869 | 0.868135 ± 0.000575 | 0.869412 ± 0.001399 | 0.871036 ± 0.000920 | 0.870008 ± 0.000996 |

### random_forest

| Metric | Original UGO | U2 | Original U5 | Retained U5+F1 | Pool-random | Pool-utility |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.895377 ± 0.000641 | 0.889619 ± 0.002389 | 0.888240 ± 0.002358 | 0.893917 ± 0.001915 | 0.889457 ± 0.002493 | 0.888727 ± 0.002668 |
| average_precision | 0.724427 ± 0.005188 | 0.717795 ± 0.005226 | 0.717602 ± 0.007071 | 0.720737 ± 0.005911 | 0.719535 ± 0.007747 | 0.725924 ± 0.004997 |
| balanced_accuracy | 0.781174 ± 0.009055 | 0.799788 ± 0.005025 | 0.795979 ± 0.013060 | 0.789717 ± 0.012235 | 0.800975 ± 0.003972 | 0.800543 ± 0.006678 |
| balanced_cross_entropy | 0.491336 ± 0.038552 | 0.452869 ± 0.024049 | 0.459299 ± 0.031952 | 0.496757 ± 0.018818 | 0.455654 ± 0.031107 | 0.448011 ± 0.024944 |
| cost | 74,512.00 ± 4,188.23 | 64,560.00 ± 1,883.39 | 65,966.00 ± 5,840.36 | 70,236.00 ± 5,918.81 | 63,976.00 ± 1,965.96 | 63,994.00 ± 2,925.28 |
| fn | 146.80 ± 8.56 | 126.20 ± 3.77 | 129.00 ± 11.90 | 138.00 ± 12.14 | 125.00 ± 4.06 | 125.00 ± 5.96 |
| fp | 111.20 ± 9.09 | 146.00 ± 4.64 | 146.60 ± 12.18 | 123.60 ± 15.76 | 147.60 ± 8.65 | 149.40 ± 8.53 |
| macro_average_precision | 0.853983 ± 0.003146 | 0.851227 ± 0.002868 | 0.850882 ± 0.004141 | 0.852336 ± 0.003331 | 0.852040 ± 0.004149 | 0.855301 ± 0.002682 |
| macro_f1 | 0.792135 ± 0.003933 | 0.793546 ± 0.004401 | 0.790365 ± 0.006758 | 0.794106 ± 0.004274 | 0.793846 ± 0.003045 | 0.792849 ± 0.004587 |
| mcc | 0.585657 ± 0.006665 | 0.587428 ± 0.008833 | 0.581324 ± 0.013512 | 0.588974 ± 0.007389 | 0.588185 ± 0.005888 | 0.586272 ± 0.009199 |
| minority_f1 | 0.645647 ± 0.008137 | 0.652711 ± 0.007468 | 0.647136 ± 0.012999 | 0.650766 ± 0.009296 | 0.653455 ± 0.004841 | 0.651923 ± 0.007970 |
| minority_recall | 0.615707 ± 0.022397 | 0.669634 ± 0.009865 | 0.662304 ± 0.031140 | 0.638743 ± 0.031793 | 0.672775 ± 0.010634 | 0.672775 ± 0.015597 |
| ordinary_cross_entropy | 0.248658 ± 0.012051 | 0.248019 ± 0.006908 | 0.249605 ± 0.008282 | 0.250325 ± 0.007580 | 0.250818 ± 0.008512 | 0.246995 ± 0.005913 |
| precision | 0.679391 ± 0.010214 | 0.636679 ± 0.008118 | 0.633535 ± 0.011033 | 0.664886 ± 0.018999 | 0.635432 ± 0.010874 | 0.632579 ± 0.010569 |
| realized_fpr_test | 0.053359 ± 0.004364 | 0.070058 ± 0.002225 | 0.070345 ± 0.005843 | 0.059309 ± 0.007561 | 0.070825 ± 0.004150 | 0.071689 ± 0.004094 |
| realized_recall_test | 0.615707 ± 0.022397 | 0.669634 ± 0.009865 | 0.662304 ± 0.031140 | 0.638743 ± 0.031793 | 0.672775 ± 0.010634 | 0.672775 ± 0.015597 |
| recall_at_fpr_test | 0.595812 ± 0.007983 | 0.602618 ± 0.012878 | 0.597906 ± 0.010888 | 0.600000 ± 0.009329 | 0.583770 ± 0.007632 | 0.596335 ± 0.013010 |
| roc_attained_fpr_test | 0.049712 ± 0.000429 | 0.049520 ± 0.000626 | 0.048752 ± 0.001815 | 0.049232 ± 0.000728 | 0.049232 ± 0.000728 | 0.049424 ± 0.000831 |
| roc_fpr_limit | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 |
| sos_inverse_support_f1 | 0.691031 ± 0.006833 | 0.696343 ± 0.006515 | 0.691510 ± 0.011056 | 0.695175 ± 0.007729 | 0.696950 ± 0.004271 | 0.695584 ± 0.006911 |
| threshold | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| tn | 1,972.80 ± 9.09 | 1,938.00 ± 4.64 | 1,937.40 ± 12.18 | 1,960.40 ± 15.76 | 1,936.40 ± 8.65 | 1,934.60 ± 8.53 |
| tp | 235.20 ± 8.56 | 255.80 ± 3.77 | 253.00 ± 11.90 | 244.00 ± 12.14 | 257.00 ± 4.06 | 257.00 ± 5.96 |
| weighted_f1 | 0.893239 ± 0.001086 | 0.890748 ± 0.002322 | 0.889220 ± 0.002667 | 0.893037 ± 0.001258 | 0.890742 ± 0.001980 | 0.890114 ± 0.002427 |

### lightgbm

| Metric | Original UGO | U2 | Original U5 | Retained U5+F1 | Pool-random | Pool-utility |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.893431 ± 0.001657 | 0.888646 ± 0.002833 | 0.889538 ± 0.001758 | 0.892863 ± 0.002301 | 0.891484 ± 0.001057 | 0.883536 ± 0.002786 |
| average_precision | 0.730186 ± 0.003323 | 0.728091 ± 0.005350 | 0.729552 ± 0.004270 | 0.733878 ± 0.001105 | 0.732201 ± 0.003138 | 0.727222 ± 0.002609 |
| balanced_accuracy | 0.787719 ± 0.004630 | 0.807764 ± 0.005615 | 0.805512 ± 0.003366 | 0.799997 ± 0.002947 | 0.810085 ± 0.002914 | 0.809444 ± 0.002404 |
| balanced_cross_entropy | 0.469407 ± 0.008495 | 0.432006 ± 0.003080 | 0.435084 ± 0.009482 | 0.461716 ± 0.006168 | 0.421839 ± 0.005929 | 0.426825 ± 0.004236 |
| cost | 71,032.00 ± 1,785.12 | 60,664.00 ± 2,203.05 | 61,916.00 ± 1,447.97 | 65,264.00 ± 1,767.49 | 60,300.00 ± 1,402.52 | 58,634.00 ± 570.25 |
| fn | 139.60 ± 3.58 | 118.20 ± 4.44 | 120.80 ± 2.95 | 127.80 ± 3.70 | 117.60 ± 2.88 | 113.80 ± 1.10 |
| fp | 123.20 ± 2.59 | 156.40 ± 6.66 | 151.60 ± 5.13 | 136.40 ± 8.96 | 150.00 ± 4.53 | 173.40 ± 6.47 |
| macro_average_precision | 0.857640 ± 0.001749 | 0.856694 ± 0.002742 | 0.857428 ± 0.002174 | 0.859613 ± 0.000542 | 0.858889 ± 0.001663 | 0.856176 ± 0.001414 |
| macro_f1 | 0.792825 ± 0.003700 | 0.795597 ± 0.004788 | 0.795715 ± 0.002707 | 0.797267 ± 0.002051 | 0.799637 ± 0.001487 | 0.790706 ± 0.003836 |
| mcc | 0.585898 ± 0.007324 | 0.592374 ± 0.009560 | 0.592208 ± 0.005407 | 0.594693 ± 0.003954 | 0.600138 ± 0.003050 | 0.584070 ± 0.007226 |
| minority_f1 | 0.648455 ± 0.006509 | 0.657688 ± 0.008022 | 0.657273 ± 0.004501 | 0.658055 ± 0.002841 | 0.663982 ± 0.002623 | 0.651319 ± 0.005918 |
| minority_recall | 0.634555 ± 0.009366 | 0.690576 ± 0.011619 | 0.683770 ± 0.007721 | 0.665445 ± 0.009689 | 0.692147 ± 0.007542 | 0.702094 ± 0.002868 |
| ordinary_cross_entropy | 0.241562 ± 0.001136 | 0.244166 ± 0.001577 | 0.243416 ± 0.001451 | 0.238946 ± 0.001855 | 0.242974 ± 0.000722 | 0.246767 ± 0.001164 |
| precision | 0.663023 ± 0.005295 | 0.627895 ± 0.009599 | 0.632825 ± 0.006889 | 0.651091 ± 0.011860 | 0.638093 ± 0.005105 | 0.607438 ± 0.009277 |
| realized_fpr_test | 0.059117 ± 0.001242 | 0.075048 ± 0.003194 | 0.072745 ± 0.002461 | 0.065451 ± 0.004300 | 0.071977 ± 0.002173 | 0.083205 ± 0.003102 |
| realized_recall_test | 0.634555 ± 0.009366 | 0.690576 ± 0.011619 | 0.683770 ± 0.007721 | 0.665445 ± 0.009689 | 0.692147 ± 0.007542 | 0.702094 ± 0.002868 |
| recall_at_fpr_test | 0.596335 ± 0.010373 | 0.604712 ± 0.005236 | 0.607853 ± 0.007496 | 0.609424 ± 0.007721 | 0.614660 ± 0.003413 | 0.587435 ± 0.020159 |
| roc_attained_fpr_test | 0.049616 ± 0.000429 | 0.049616 ± 0.000644 | 0.048944 ± 0.000898 | 0.049424 ± 0.000480 | 0.049040 ± 0.000923 | 0.048848 ± 0.000858 |
| roc_fpr_limit | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 |
| sos_inverse_support_f1 | 0.693183 ± 0.005637 | 0.700414 ± 0.007015 | 0.700164 ± 0.003940 | 0.701185 ± 0.002580 | 0.706010 ± 0.002262 | 0.694503 ± 0.005273 |
| threshold | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| tn | 1,960.80 ± 2.59 | 1,927.60 ± 6.66 | 1,932.40 ± 5.13 | 1,947.60 ± 8.96 | 1,934.00 ± 4.53 | 1,910.60 ± 6.47 |
| tp | 242.40 ± 3.58 | 263.80 ± 4.44 | 261.20 ± 2.95 | 254.20 ± 3.70 | 264.40 ± 2.88 | 268.20 ± 1.10 |
| weighted_f1 | 0.892468 ± 0.001781 | 0.890780 ± 0.002631 | 0.891266 ± 0.001548 | 0.893348 ± 0.001653 | 0.893265 ± 0.000845 | 0.886909 ± 0.002407 |

## Paired final utility-minus-reference differences

Per-seed deltas, then mean ± sample SD. Same real data/holdout; reference learners follow their original selections and accepted doses can differ.

### logistic_regression

| Metric | U2 | Pool-random | Original UGO | Original U5 | Retained U5+F1 |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.001460 ± 0.000615 | -0.001784 ± 0.000679 | -0.008191 ± 0.001754 | 0.001946 ± 0.001125 | -0.003650 ± 0.002382 |
| average_precision | 0.002636 ± 0.003215 | -0.004598 ± 0.002388 | -0.019675 ± 0.001670 | 0.002654 ± 0.001695 | -0.008112 ± 0.002048 |
| balanced_accuracy | 0.000436 ± 0.002032 | 0.001724 ± 0.000634 | 0.001781 ± 0.006086 | 0.002648 ± 0.003008 | 0.018365 ± 0.004280 |
| balanced_cross_entropy | -0.005956 ± 0.002573 | 0.001740 ± 0.001304 | -0.038034 ± 0.003130 | -0.005522 ± 0.001027 | -0.049736 ± 0.007640 |
| cost | 160.00 ± 829.28 | -1,230.00 ± 265.24 | -2,836.00 ± 2,375.65 | -734.00 ± 1,196.28 | -9,318.00 ± 1,628.16 |
| fn | 0.40 ± 1.67 | -2.60 ± 0.55 | -6.20 ± 4.76 | -1.40 ± 2.41 | -19.20 ± 3.27 |
| fp | -4.00 ± 1.41 | 7.00 ± 1.87 | 26.40 ± 1.52 | -3.40 ± 2.41 | 28.20 ± 5.40 |
| macro_average_precision | 0.000945 ± 0.001739 | -0.003356 ± 0.001129 | -0.010657 ± 0.000942 | 0.000882 ± 0.000964 | -0.004645 ± 0.001017 |
| macro_f1 | 0.001659 ± 0.001700 | -0.000661 ± 0.000874 | -0.006902 ± 0.005525 | 0.003456 ± 0.002696 | 0.008335 ± 0.004576 |
| mcc | 0.004075 ± 0.003185 | -0.003152 ± 0.002137 | -0.021345 ± 0.009586 | 0.007293 ± 0.005135 | 0.005319 ± 0.009718 |
| minority_f1 | 0.002397 ± 0.003128 | -0.000130 ± 0.001377 | -0.008551 ± 0.010137 | 0.005750 ± 0.004866 | 0.019548 ± 0.007956 |
| minority_recall | -0.001047 ± 0.004380 | 0.006806 ± 0.001434 | 0.016230 ± 0.012472 | 0.003665 ± 0.006304 | 0.050262 ± 0.008563 |
| ordinary_cross_entropy | 0.002121 ± 0.001580 | 0.003125 ± 0.000355 | 0.010382 ± 0.003135 | 0.002288 ± 0.000920 | 0.017062 ± 0.001878 |
| precision | 0.007981 ± 0.002676 | -0.012252 ± 0.004100 | -0.055101 ± 0.004807 | 0.008933 ± 0.004873 | -0.043824 ± 0.014180 |
| realized_fpr_test | -0.001919 ± 0.000679 | 0.003359 ± 0.000898 | 0.012668 ± 0.000728 | -0.001631 ± 0.001156 | 0.013532 ± 0.002593 |
| realized_recall_test | -0.001047 ± 0.004380 | 0.006806 ± 0.001434 | 0.016230 ± 0.012472 | 0.003665 ± 0.006304 | 0.050262 ± 0.008563 |
| recall_at_fpr_test | 0.003141 ± 0.010206 | -0.012042 ± 0.008401 | -0.033508 ± 0.009689 | 0.005236 ± 0.009255 | -0.029843 ± 0.014528 |
| roc_attained_fpr_test | 0.000192 ± 0.001382 | -0.000576 ± 0.000858 | -0.000288 ± 0.000872 | 0.000000 ± 0.000339 | -0.000480 ± 0.000960 |
| roc_fpr_limit | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| sos_inverse_support_f1 | 0.002169 ± 0.002684 | -0.000295 ± 0.001219 | -0.008040 ± 0.008708 | 0.005039 ± 0.004192 | 0.016074 ± 0.006904 |
| threshold | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| tn | 4.00 ± 1.41 | -7.00 ± 1.87 | -26.40 ± 1.52 | 3.40 ± 2.41 | -28.20 ± 5.40 |
| tp | -0.40 ± 1.67 | 2.60 ± 0.55 | 6.20 ± 4.76 | 1.40 ± 2.41 | 19.20 ± 3.27 |
| weighted_f1 | 0.001149 ± 0.000733 | -0.001028 ± 0.000549 | -0.005765 ± 0.002348 | 0.001873 ± 0.001236 | 0.000596 ± 0.002319 |

### random_forest

| Metric | U2 | Pool-random | Original UGO | Original U5 | Retained U5+F1 |
|---|---:|---:|---:|---:|---:|
| accuracy | -0.000892 ± 0.003159 | -0.000730 ± 0.003798 | -0.006650 ± 0.002729 | 0.000487 ± 0.003349 | -0.005191 ± 0.004089 |
| average_precision | 0.008129 ± 0.003416 | 0.006389 ± 0.003780 | 0.001497 ± 0.009345 | 0.008322 ± 0.011599 | 0.005187 ± 0.009248 |
| balanced_accuracy | 0.000755 ± 0.010253 | -0.000432 ± 0.005022 | 0.019369 ± 0.013769 | 0.004564 ± 0.016717 | 0.010826 ± 0.013572 |
| balanced_cross_entropy | -0.004858 ± 0.029347 | -0.007644 ± 0.017786 | -0.043325 ± 0.054430 | -0.011289 ± 0.051415 | -0.048747 ± 0.026116 |
| cost | -566.00 ± 4,089.55 | 18.00 ± 2,131.03 | -10,518.00 ± 6,011.86 | -1,972.00 ± 7,657.53 | -6,242.00 ± 7,041.38 |
| fn | -1.20 ± 8.23 | 0.00 ± 4.36 | -21.80 ± 12.21 | -4.00 ± 15.64 | -13.00 ± 14.54 |
| fp | 3.40 ± 5.81 | 1.80 ± 10.62 | 38.20 ± 11.41 | 2.80 ± 18.10 | 25.80 ± 23.67 |
| macro_average_precision | 0.004074 ± 0.001923 | 0.003261 ± 0.001945 | 0.001318 ± 0.005305 | 0.004419 ± 0.006585 | 0.002964 ± 0.005133 |
| macro_f1 | -0.000697 ± 0.007266 | -0.000997 ± 0.005304 | 0.000714 ± 0.007848 | 0.002484 ± 0.008050 | -0.001257 ± 0.003618 |
| mcc | -0.001157 ± 0.014738 | -0.001913 ± 0.010382 | 0.000615 ± 0.014499 | 0.004948 ± 0.016465 | -0.002703 ± 0.006669 |
| minority_f1 | -0.000788 ± 0.012906 | -0.001532 ± 0.008455 | 0.006276 ± 0.014764 | 0.004787 ± 0.015616 | 0.001157 ± 0.008054 |
| minority_recall | 0.003141 ± 0.021539 | 0.000000 ± 0.011411 | 0.057068 ± 0.031976 | 0.010471 ± 0.040933 | 0.034031 ± 0.038071 |
| ordinary_cross_entropy | -0.001024 ± 0.009266 | -0.003823 ± 0.006324 | -0.001663 ± 0.014720 | -0.002611 ± 0.012384 | -0.003330 ± 0.008412 |
| precision | -0.004100 ± 0.008738 | -0.002852 ± 0.014983 | -0.046811 ± 0.011323 | -0.000956 ± 0.017240 | -0.032307 ± 0.028981 |
| realized_fpr_test | 0.001631 ± 0.002790 | 0.000864 ± 0.005094 | 0.018330 ± 0.005475 | 0.001344 ± 0.008686 | 0.012380 ± 0.011357 |
| realized_recall_test | 0.003141 ± 0.021539 | 0.000000 ± 0.011411 | 0.057068 ± 0.031976 | 0.010471 ± 0.040933 | 0.034031 ± 0.038071 |
| recall_at_fpr_test | -0.006283 ± 0.020991 | 0.012565 ± 0.016073 | 0.000524 ± 0.011320 | -0.001571 ± 0.007542 | -0.003665 ± 0.011939 |
| roc_attained_fpr_test | -0.000096 ± 0.001242 | 0.000192 ± 0.000263 | -0.000288 ± 0.001051 | 0.000672 ± 0.002316 | 0.000192 ± 0.000263 |
| roc_fpr_limit | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| sos_inverse_support_f1 | -0.000760 ± 0.011156 | -0.001366 ± 0.007470 | 0.004553 ± 0.012614 | 0.004073 ± 0.013253 | 0.000409 ± 0.006615 |
| threshold | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| tn | -3.40 ± 5.81 | -1.80 ± 10.62 | -38.20 ± 11.41 | -2.80 ± 18.10 | -25.80 ± 23.67 |
| tp | 1.20 ± 8.23 | 0.00 ± 4.36 | 21.80 ± 12.21 | 4.00 ± 15.64 | 13.00 ± 14.54 |
| weighted_f1 | -0.000634 ± 0.003430 | -0.000628 ± 0.003243 | -0.003125 ± 0.003241 | 0.000894 ± 0.003274 | -0.002923 ± 0.002241 |

### lightgbm

| Metric | U2 | Pool-random | Original UGO | Original U5 | Retained U5+F1 |
|---|---:|---:|---:|---:|---:|
| accuracy | -0.005109 ± 0.004051 | -0.007948 ± 0.003507 | -0.009895 ± 0.002774 | -0.006002 ± 0.003013 | -0.009327 ± 0.005000 |
| average_precision | -0.000869 ± 0.005513 | -0.004979 ± 0.005189 | -0.002964 ± 0.002249 | -0.002329 ± 0.005305 | -0.006656 ± 0.003278 |
| balanced_accuracy | 0.001680 ± 0.004256 | -0.000640 ± 0.004473 | 0.021725 ± 0.003606 | 0.003932 ± 0.001593 | 0.009447 ± 0.003088 |
| balanced_cross_entropy | -0.005181 ± 0.002705 | 0.004987 ± 0.005280 | -0.042581 ± 0.006335 | -0.008258 ± 0.007065 | -0.034891 ± 0.004758 |
| cost | -2,030.00 ± 1,950.40 | -1,666.00 ± 1,815.99 | -12,398.00 ± 1,626.74 | -3,282.00 ± 1,092.60 | -6,630.00 ± 1,847.53 |
| fn | -4.40 ± 4.04 | -3.80 ± 3.70 | -25.80 ± 3.35 | -7.00 ± 2.35 | -14.00 ± 3.94 |
| fp | 17.00 ± 11.85 | 23.40 ± 9.40 | 50.20 ± 8.23 | 21.80 ± 9.36 | 37.00 ± 15.28 |
| macro_average_precision | -0.000517 ± 0.002814 | -0.002713 ± 0.002763 | -0.001463 ± 0.001247 | -0.001251 ± 0.002729 | -0.003437 ± 0.001720 |
| macro_f1 | -0.004891 ± 0.005101 | -0.008932 ± 0.004831 | -0.002120 ± 0.003677 | -0.005009 ± 0.003271 | -0.006561 ± 0.005618 |
| mcc | -0.008304 ± 0.009593 | -0.016068 ± 0.009232 | -0.001828 ± 0.007149 | -0.008139 ± 0.005932 | -0.010624 ± 0.010638 |
| minority_f1 | -0.006369 ± 0.007818 | -0.012663 ± 0.007603 | 0.002864 ± 0.005861 | -0.005953 ± 0.004632 | -0.006736 ± 0.008107 |
| minority_recall | 0.011518 ± 0.010569 | 0.009948 ± 0.009689 | 0.067539 ± 0.008761 | 0.018325 ± 0.006139 | 0.036649 ± 0.010306 |
| ordinary_cross_entropy | 0.002601 ± 0.001169 | 0.003794 ± 0.001480 | 0.005205 ± 0.000824 | 0.003351 ± 0.002438 | 0.007821 ± 0.001706 |
| precision | -0.020457 ± 0.015140 | -0.030655 ± 0.012607 | -0.055586 ± 0.010820 | -0.025388 ± 0.011835 | -0.043653 ± 0.020840 |
| realized_fpr_test | 0.008157 ± 0.005688 | 0.011228 ± 0.004509 | 0.024088 ± 0.003948 | 0.010461 ± 0.004494 | 0.017754 ± 0.007332 |
| realized_recall_test | 0.011518 ± 0.010569 | 0.009948 ± 0.009689 | 0.067539 ± 0.008761 | 0.018325 ± 0.006139 | 0.036649 ± 0.010306 |
| recall_at_fpr_test | -0.017277 ± 0.015772 | -0.027225 ± 0.021949 | -0.008901 ± 0.027493 | -0.020419 ± 0.021219 | -0.021990 ± 0.021154 |
| roc_attained_fpr_test | -0.000768 ± 0.001251 | -0.000192 ± 0.001540 | -0.000768 ± 0.000644 | -0.000096 ± 0.000923 | -0.000576 ± 0.001288 |
| roc_fpr_limit | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| sos_inverse_support_f1 | -0.005911 ± 0.006967 | -0.011507 ± 0.006737 | 0.001320 ± 0.005174 | -0.005661 ± 0.004207 | -0.006682 ± 0.007329 |
| threshold | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| tn | -17.00 ± 11.85 | -23.40 ± 9.40 | -50.20 ± 8.23 | -21.80 ± 9.36 | -37.00 ± 15.28 |
| tp | 4.40 ± 4.04 | 3.80 ± 3.70 | 25.80 ± 3.35 | 7.00 ± 2.35 | 14.00 ± 3.94 |
| weighted_f1 | -0.003872 ± 0.003337 | -0.006356 ± 0.002998 | -0.005559 ± 0.002300 | -0.004358 ± 0.002368 | -0.006440 ± 0.003965 |

## Initial matched-pool selector comparison

Utility-minus-random after actual native batch retraining. Guidance losses/AP are reused selection feedback; outer-test changes are descriptive.

### logistic_regression

| Metric | Inner guidance | Outer guidance (in-sample) | Outer test |
|---|---:|---:|---:|
| accuracy | -0.002737 ± 0.001504 | 0.000608 ± 0.002132 | -0.002758 ± 0.001845 |
| average_precision | -0.004373 ± 0.002890 | -0.004410 ± 0.002542 | -0.003186 ± 0.001998 |
| balanced_accuracy | -0.005905 ± 0.005579 | 0.001967 ± 0.006660 | -0.004197 ± 0.003902 |
| balanced_cross_entropy | 0.001816 ± 0.002975 | 0.002948 ± 0.003615 | 0.001820 ± 0.001960 |
| cost | 1,622.00 ± 1,772.64 | -600.00 ± 2,094.23 | 1,244.00 ± 1,365.13 |
| fn | 3.20 ± 3.56 | -1.20 ± 4.21 | 2.40 ± 2.70 |
| fp | 2.20 ± 1.64 | 0.00 ± 2.83 | 4.40 ± 2.41 |
| macro_average_precision | -0.002690 ± 0.001622 | -0.002787 ± 0.001499 | -0.002326 ± 0.001072 |
| macro_f1 | -0.006570 ± 0.005027 | 0.001846 ± 0.006162 | -0.005506 ± 0.004387 |
| mcc | -0.012918 ± 0.008826 | 0.003345 ± 0.011255 | -0.012021 ± 0.008762 |
| minority_f1 | -0.011588 ± 0.009313 | 0.003365 ± 0.011275 | -0.009403 ± 0.007755 |
| minority_recall | -0.010492 ± 0.011684 | 0.003934 ± 0.013794 | -0.006283 ± 0.007073 |
| ordinary_cross_entropy | 0.002160 ± 0.001846 | 0.002643 ± 0.001192 | 0.001428 ± 0.000509 |
| precision | -0.012136 ± 0.005507 | 0.001819 ± 0.010091 | -0.015678 ± 0.009322 |
| realized_fpr_test | 0.001319 ± 0.000985 | -0.000000 ± 0.001696 | 0.002111 ± 0.001156 |
| realized_recall_test | -0.010492 ± 0.011684 | 0.003934 ± 0.013794 | -0.006283 ± 0.007073 |
| recall_at_fpr_test | -0.003279 ± 0.012047 | -0.013115 ± 0.017349 | -0.010995 ± 0.013527 |
| roc_attained_fpr_test | -0.000600 ± 0.001406 | 0.000480 ± 0.000502 | 0.000384 ± 0.001146 |
| roc_fpr_limit | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| sos_inverse_support_f1 | -0.010037 ± 0.007987 | 0.002895 ± 0.009692 | -0.008196 ± 0.006711 |
| threshold | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| tn | -2.20 ± 1.64 | 0.00 ± 2.83 | -4.40 ± 2.41 |
| tp | -3.20 ± 3.56 | 1.20 ± 4.21 | -2.40 ± 2.70 |
| weighted_f1 | -0.003103 ± 0.002079 | 0.000796 ± 0.002663 | -0.002817 ± 0.002072 |

### random_forest

| Metric | Inner guidance | Outer guidance (in-sample) | Outer test |
|---|---:|---:|---:|
| accuracy | -0.001521 ± 0.002889 | 0.000710 ± 0.004158 | -0.000081 ± 0.003312 |
| average_precision | -0.000524 ± 0.002826 | 0.002361 ± 0.003610 | 0.002766 ± 0.005021 |
| balanced_accuracy | 0.000172 ± 0.002044 | 0.000688 ± 0.009638 | 0.000166 ± 0.004931 |
| balanced_cross_entropy | -0.003036 ± 0.003097 | -0.000737 ± 0.003866 | -0.000683 ± 0.028310 |
| cost | -362.00 ± 413.91 | -112.00 ± 2,727.28 | -96.00 ± 1,585.87 |
| fn | -0.80 ± 0.84 | -0.20 ± 5.40 | -0.20 ± 3.11 |
| fp | 3.80 ± 5.76 | -1.20 ± 3.27 | 0.40 ± 6.31 |
| macro_average_precision | -0.000271 ± 0.001364 | 0.001205 ± 0.001868 | 0.001382 ± 0.002826 |
| macro_f1 | -0.001682 ± 0.004103 | 0.001388 ± 0.008498 | -0.000009 ± 0.005683 |
| mcc | -0.003637 ± 0.008444 | 0.002836 ± 0.016810 | -0.000040 ± 0.011451 |
| minority_f1 | -0.002386 ± 0.006420 | 0.002358 ± 0.014575 | 0.000040 ± 0.009407 |
| minority_recall | 0.002623 ± 0.002743 | 0.000656 ± 0.017717 | 0.000524 ± 0.008153 |
| ordinary_cross_entropy | 0.000552 ± 0.001239 | -0.000184 ± 0.001346 | -0.000592 ± 0.009333 |
| precision | -0.008162 ± 0.013501 | 0.004236 ± 0.012038 | -0.000505 ± 0.013016 |
| realized_fpr_test | 0.002278 ± 0.003454 | -0.000719 ± 0.001961 | 0.000192 ± 0.003027 |
| realized_recall_test | 0.002623 ± 0.002743 | 0.000656 ± 0.017717 | 0.000524 ± 0.008153 |
| recall_at_fpr_test | 0.002623 ± 0.008487 | 0.007869 ± 0.012825 | -0.004712 ± 0.011910 |
| roc_attained_fpr_test | 0.000839 ± 0.001380 | -0.003597 ± 0.007440 | -0.000192 ± 0.000728 |
| roc_fpr_limit | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| sos_inverse_support_f1 | -0.002168 ± 0.005703 | 0.002058 ± 0.012696 | 0.000025 ± 0.008252 |
| threshold | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| tn | -3.80 ± 5.76 | 1.20 ± 3.27 | -0.40 ± 6.31 |
| tp | 0.80 ± 0.84 | 0.20 ± 5.40 | 0.20 ± 3.11 |
| weighted_f1 | -0.001195 ± 0.002510 | 0.000718 ± 0.004301 | -0.000042 ± 0.003131 |

### lightgbm

| Metric | Inner guidance | Outer guidance (in-sample) | Outer test |
|---|---:|---:|---:|
| accuracy | -0.001723 ± 0.001272 | 0.003142 ± 0.002192 | -0.001379 ± 0.000790 |
| average_precision | 0.002852 ± 0.004745 | 0.002383 ± 0.002272 | 0.000147 ± 0.003717 |
| balanced_accuracy | 0.003267 ± 0.002295 | 0.013647 ± 0.005144 | -0.001030 ± 0.004071 |
| balanced_cross_entropy | -0.003255 ± 0.002797 | -0.001797 ± 0.004827 | 0.003039 ± 0.003913 |
| cost | -1,534.00 ± 1,040.93 | -4,374.00 ± 1,520.44 | 132.00 ± 1,933.80 |
| fn | -3.20 ± 2.17 | -8.80 ± 3.03 | 0.20 ± 3.96 |
| fp | 6.60 ± 4.51 | 2.60 ± 2.70 | 3.20 ± 5.07 |
| macro_average_precision | 0.001456 ± 0.002373 | 0.001221 ± 0.001154 | -0.000020 ± 0.001923 |
| macro_f1 | -0.000784 ± 0.000986 | 0.008590 ± 0.004391 | -0.002021 ± 0.001630 |
| mcc | -0.001594 ± 0.001936 | 0.016872 ± 0.008618 | -0.004027 ± 0.003220 |
| minority_f1 | -0.000375 ± 0.001348 | 0.015493 ± 0.007535 | -0.003183 ± 0.003280 |
| minority_recall | 0.010492 ± 0.007108 | 0.028852 ± 0.009945 | -0.000524 ± 0.010373 |
| ordinary_cross_entropy | 0.000140 ± 0.000888 | 0.000048 ± 0.001110 | 0.000880 ± 0.001426 |
| precision | -0.011318 ± 0.007734 | 0.001284 ± 0.007299 | -0.005803 ± 0.005692 |
| realized_fpr_test | 0.003957 ± 0.002701 | 0.001559 ± 0.001620 | 0.001536 ± 0.002433 |
| realized_recall_test | 0.010492 ± 0.007108 | 0.028852 ± 0.009945 | -0.000524 ± 0.010373 |
| recall_at_fpr_test | 0.007213 ± 0.002743 | 0.023607 ± 0.012136 | -0.010995 ± 0.013653 |
| roc_attained_fpr_test | 0.000240 ± 0.000328 | 0.000000 ± 0.000848 | -0.000480 ± 0.000588 |
| roc_fpr_limit | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| sos_inverse_support_f1 | -0.000501 ± 0.001221 | 0.013359 ± 0.006562 | -0.002823 ± 0.002763 |
| threshold | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| tn | -6.60 ± 4.51 | -2.60 ± 2.70 | -3.20 ± 5.07 |
| tp | 3.20 ± 2.17 | 8.80 ± 3.03 | -0.20 ± 3.96 |
| weighted_f1 | -0.001066 ± 0.000867 | 0.003821 ± 0.002234 | -0.001219 ± 0.000648 |

## Accepted counts and observed trajectory work

Reference cell wall times include their tuning/refit paths. Pool trajectory times exclude shared canonical U2 selection and diagnostic batch/parent fits outside the live trajectory. Observed trajectory time includes checkpoint/pool audits and diagnostic scoring/SGD checks; the derived operational value removes logged scoring diagnostics, not all artifact IO. These quantities are not equivalent end-to-end timings. Random’s LR scorer work exists only for paired audit; shared initial LR fits may be cached before utility runs. No isolated selector speed claim is made.

| Classifier | Variant | Accepted | Recorded wall seconds |
|---|---:|---:|---:|
| logistic_regression | Original UGO | 1,000.00 ± 0.00 | 20.489185 ± 0.390955 |
| logistic_regression | U2 | 1,000.00 ± 0.00 | 32.235012 ± 0.590798 |
| logistic_regression | Original U5 | 1,000.00 ± 0.00 | 32.486929 ± 0.502994 |
| logistic_regression | Retained U5+F1 | 522.40 ± 63.94 | 19.169080 ± 0.541113 |
| logistic_regression | Pool-random | 1,000.00 ± 0.00 | 5.364141 ± 0.111902 |
| logistic_regression | Pool-utility | 1,000.00 ± 0.00 | 5.394667 ± 0.153465 |
| random_forest | Original UGO | 1,000.00 ± 0.00 | 88.711577 ± 5.291605 |
| random_forest | U2 | 1,000.00 ± 0.00 | 130.421603 ± 5.978865 |
| random_forest | Original U5 | 1,000.00 ± 0.00 | 166.393768 ± 11.335656 |
| random_forest | Retained U5+F1 | 622.60 ± 112.25 | 75.005319 ± 36.440143 |
| random_forest | Pool-random | 1,000.00 ± 0.00 | 31.441150 ± 9.754928 |
| random_forest | Pool-utility | 1,000.00 ± 0.00 | 31.359410 ± 9.849324 |
| lightgbm | Original UGO | 1,000.00 ± 0.00 | 28.343309 ± 0.395808 |
| lightgbm | U2 | 1,000.00 ± 0.00 | 42.408440 ± 0.604729 |
| lightgbm | Original U5 | 1,000.00 ± 0.00 | 48.097245 ± 2.387728 |
| lightgbm | Retained U5+F1 | 689.20 ± 113.99 | 12.841892 ± 7.400927 |
| lightgbm | Pool-random | 1,000.00 ± 0.00 | 6.226286 ± 0.121408 |
| lightgbm | Pool-utility | 1,000.00 ± 0.00 | 6.244480 ± 0.181637 |

Full timing/fit counts for both inner and outer control roles are in `resource_diagnostics.csv`; reference candidate/outer audits retain their actual fit counts including OOF. Per-algorithm totals in raw records include all four references, all controls and all parent/checkpoint diagnostics. Flow is physically shared across reference families per seed, not part of the utility selector.

## Local estimate, state and matched-checkpoint evidence

- logistic_regression: 5559 saved candidate-step diagnostics (initial branch copies included); minimum Spearman 0.999908, minimum sign agreement 1.000000, maximum absolute first-order error 2.63056717e-07. Eta values: [np.float64(0.0001)]. This verifies the LR scorer only.
- random_forest: 6405 saved candidate-step diagnostics (initial branch copies included); minimum Spearman 0.999954, minimum sign agreement 0.984375, maximum absolute first-order error 2.06968222e-07. Eta values: [np.float64(0.0001)]. This verifies the LR scorer only.
- lightgbm: 6400 saved candidate-step diagnostics (initial branch copies included); minimum Spearman 0.999863, minimum sign agreement 1.000000, maximum absolute first-order error 2.55120654e-07. Eta values: [np.float64(0.0001)]. This verifies the LR scorer only.

Live native and LR-scorer before/after hashes are recorded. Independently reconstructed native tree dumps/arrays and LR coefficients are the cross-serialization proof; tree pickle hashes need not reproduce after joblib loading. Native/scorer checkpoints and exact prefixes include terminal scoring checkpoints.

`matched_checkpoint_metric_deltas.csv` compares both selectors after native retraining from each identical prefix/checkpoint/pool; `parent_metric_deltas.csv` compares new coordinates against half-weight anchor plus half-weight partner with equal added mass. Neither counterfactual fit becomes live. Parent duplication uses two half-weight rows per synthetic, not the same physical row count; native tree row-count/min-leaf effects remain part of this comparison. It is not a pure isolation of coordinate geometry for RF/LightGBM. Within-trajectory later pools differ; do not mix checkpoint observations as independent seeds.

## Verification and reproduction

Independent verification: **passed (1,400,828 checks)**; named counts and checks in `results/runs/20261009T153348-af844bdecf5a/independent_verification.json`. All **251 tests passed**, including ten new real-Shoppers/analytical regressions covering role overlap, test poison exclusion, native anchor scoring, exact refreshed surrogate prefixes, frozen state, negative top-fill, quotas/caps, matched pools and equal parent mass.

Training-only all-classifier smoke: `results/smoke/20261009T131743-67015812e27e`, 48 completed records. The initial full study was launched after the smoke and full suite finished; neither ran concurrently with it. The final targeted ten-test check passed again before the corrected references began fitting. Independent verifier runtimes are additional to benchmark runtime. Existing sklearn iprint OptimizeWarnings and LightGBM feature-name warnings were observed, not suppressed.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.validation_guided --run
```

Experiment declaration: `configs/ablations/stage1_validation_guided.yaml`. `all_metrics.csv`, `metric_summary.csv`, `paired_metric_deltas.csv`, `paired_metric_summary.csv`, `selection_diagnostics.csv`, `group_generation.csv`, `one_step_diagnostics.csv`, `resource_diagnostics.csv`, `matched_checkpoint_metric_deltas.csv` and `parent_metric_deltas.csv` contain the complete per-seed/all-field data.

**User decision: pool-utility discarded.** retained U5+F1 is unchanged. Standalone U5 remains discarded and is evaluated only as the requested reference. Pool-random+F1 is the next experiment; its retention remains pending. Phase 1 remains open/unfrozen and broader evaluation remains closed.

## Findings and compute interpretation

- Logistic regression: utility loses AP to pool-random in all five seeds and worsens ordinary/balanced test CE in all five. The actual-model local dot product is accurate, but its additional-SGD prediction does not transfer to full native retraining.
- Random Forest: explicit LR-surrogate utility improves AP over pool-random in all five seeds; F1 and MCC are slightly lower on average and there is no mean recall gain. This is an AP/loss signal, not operating-point dominance or proof of native tree-gradient utility.
- LightGBM: utility loses AP in four of five seeds and F1/MCC/ordinary/balanced CE in all five. Higher recall and lower diagnostic cost are threshold tradeoffs, not a general quality improvement. Initial reused-guidance AP gains do not reliably transfer to the outer test.
- The full target is 1,000 accepted synthetics for UGO/U2/U5/random/utility in every seed. Actual initial budgets shortfall under the one-per-anchor cap; U2 reaches target in two accepting rounds, LR pools in three, RF/LightGBM pools in four. Retrain-once-per-round cadence and nominal budgets are unchanged; actual accepted batch sizes are not identical to U2.
- Utility batches have smaller mean nearest-other-selected-point distances than random in the shared initial pool for all three classifiers. This supports a redundancy hypothesis, but does not causally establish it. Parent controls and actual native refit outcomes, not local approximation accuracy alone, must decide retention.
Recorded source benchmark runtime (excluding interpreter startup): 6,621.94 seconds (110.37 minutes), including reference tuning, both roles, live trajectories and research diagnostics. Recorded corrected-reference benchmark runtime: 540.54 seconds. Independent verifier wall time is additional. These are sequential invocations, not an isolated production utility runtime.

| Classifier | Metric | Final utility minus random | Utility wins | Seeds |
|---|---:|---:|---:|---:|
| logistic_regression | average_precision | -0.004598 ± 0.002388 | 0 | 5 |
| logistic_regression | minority_f1 | -0.000130 ± 0.001377 | 1 | 5 |
| logistic_regression | minority_recall | 0.006806 ± 0.001434 | 5 | 5 |
| logistic_regression | mcc | -0.003152 ± 0.002137 | 1 | 5 |
| logistic_regression | ordinary_cross_entropy | 0.003125 ± 0.000355 | 0 | 5 |
| logistic_regression | balanced_cross_entropy | 0.001740 ± 0.001304 | 0 | 5 |
| logistic_regression | cost | -1,230.00 ± 265.24 | 5 | 5 |
| random_forest | average_precision | 0.006389 ± 0.003780 | 5 | 5 |
| random_forest | minority_f1 | -0.001532 ± 0.008455 | 2 | 5 |
| random_forest | minority_recall | 0.000000 ± 0.011411 | 2 | 5 |
| random_forest | mcc | -0.001913 ± 0.010382 | 2 | 5 |
| random_forest | ordinary_cross_entropy | -0.003823 ± 0.006324 | 4 | 5 |
| random_forest | balanced_cross_entropy | -0.007644 ± 0.017786 | 4 | 5 |
| random_forest | cost | 18.00 ± 2,131.03 | 2 | 5 |
| lightgbm | average_precision | -0.004979 ± 0.005189 | 1 | 5 |
| lightgbm | minority_f1 | -0.012663 ± 0.007603 | 0 | 5 |
| lightgbm | minority_recall | 0.009948 ± 0.009689 | 4 | 5 |
| lightgbm | mcc | -0.016068 ± 0.009232 | 0 | 5 |
| lightgbm | ordinary_cross_entropy | 0.003794 ± 0.001480 | 0 | 5 |
| lightgbm | balanced_cross_entropy | 0.004987 ± 0.005280 | 0 | 5 |
| lightgbm | cost | -1,666.00 ± 1,815.99 | 4 | 5 |

### Equal-added-weight parent control

Shared initial checkpoint/pool; actual synthetic batch minus matched half-anchor/half-partner parent duplication. Mean ± sample SD. Positive AP/F1/MCC favors new coordinates; negative CE favors new coordinates. Guidance and every other field, at every checkpoint, are in the linked CSV.

| Classifier | Selector | Delta AP | Delta F1 | Delta MCC | Delta ordinary CE | Delta balanced CE |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | random | -0.005614 ± 0.000949 | -0.002545 ± 0.002962 | -0.004083 ± 0.003176 | 0.001924 ± 0.000335 | 0.006683 ± 0.000619 |
| logistic_regression | utility | -0.008559 ± 0.000821 | -0.012372 ± 0.004856 | -0.014649 ± 0.005539 | 0.002983 ± 0.000541 | 0.009547 ± 0.001474 |
| random_forest | random | -0.003022 ± 0.003140 | 0.005295 ± 0.009746 | 0.006484 ± 0.011577 | 0.004283 ± 0.007417 | 0.020362 ± 0.020825 |
| random_forest | utility | 0.001880 ± 0.002239 | 0.009401 ± 0.005597 | 0.011304 ± 0.006061 | 0.000835 ± 0.009970 | 0.010530 ± 0.028971 |
| lightgbm | random | -0.004052 ± 0.001920 | -0.003275 ± 0.007578 | -0.004635 ± 0.008613 | 0.000073 ± 0.000920 | 0.008336 ± 0.002146 |
| lightgbm | utility | -0.003257 ± 0.002882 | -0.008259 ± 0.007052 | -0.010947 ± 0.007893 | 0.000254 ± 0.001131 | 0.009346 ± 0.003565 |

**User decision: pool-utility discarded.** The RF signal is classifier-specific; LR/LightGBM do not satisfy general improvement over matched pool-random. U5+F1 remains retained and Phase 1 is not concluded.


### Quantitative pool coverage and physical fit costs

Shared initial outer checkpoint: pool-random and pool-utility have identical accepted counts and quota shortfalls. Mean Euclidean distance to the nearest other selected point is measured in the existing preprocessed feature space, excluding self-distance; it is only a descriptive redundancy diagnostic.

| Classifier | Selector | Accepted | Quota shortfall | Mean nearest-other distance | Mean selected utility | Real-coordinate matches |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | random | 590.00 ± 10.05 | 396.00 ± 10.05 | 1.445595 ± 0.032495 | 0.819544 ± 0.035720 | 32.60 ± 2.88 |
| logistic_regression | utility | 590.00 ± 10.05 | 396.00 ± 10.05 | 1.211167 ± 0.055669 | 0.935144 ± 0.042942 | 47.00 ± 5.10 |
| random_forest | random | 413.00 ± 36.48 | 573.00 ± 36.48 | 1.780202 ± 0.072039 | 0.811298 ± 0.070505 | 19.40 ± 3.05 |
| random_forest | utility | 413.00 ± 36.48 | 573.00 ± 36.48 | 1.516129 ± 0.054713 | 0.936395 ± 0.075436 | 27.20 ± 4.09 |
| lightgbm | random | 435.40 ± 29.26 | 550.60 ± 29.26 | 1.724907 ± 0.062343 | 0.800367 ± 0.049065 | 21.20 ± 3.03 |
| lightgbm | utility | 435.40 ± 29.26 | 550.60 ± 29.26 | 1.501024 ± 0.041502 | 0.926002 ± 0.056018 | 29.80 ± 3.90 |

| Classifier | Physical native fits | Separate physical LR scorer fits | LR scorer fit seconds |
|---|---:|---:|---:|
| logistic_regression | 500 | 0 | 0.000000 |
| random_forest | 565 | 111 | 3.577549 |
| lightgbm | 560 | 110 | 3.574041 |

Counts above cover both control roles, all seeds and native paired/parent diagnostics; original reference/tuning fits are additional. LR uses its downstream model rather than fitting a second scorer. The initial scorer is shared where checkpoints coincide. The pool versus original-U2 live runtime overhead includes additional rounds caused by cap-related quota shortfalls, not only the utility dot products.

Reference cell timings are not isolated production timings. In particular, the corrected U5+F1 row records its own tau/Flow/refit rerun while canonical U5 selection is reused from the source study; that original tuning is not rerun or charged again to this cell.

Observed CLI wall time: initial scientific study 6,624.28 s; corrected-reference invocation 542.57 s; final independent audit 680.06 s. The first independent audit that identified the CPU-reference/parser issues took 695.01 s. Thus scientific execution totaled 7,166.85 s (119.45 min), and those two full artifact audits added 1,375.07 s (22.92 min). Tests, two all-classifier CLI smokes, the independent smoke audit and report generation are additional, not included in those totals.

### Independent evidence and archive dependencies

The final audit reproduced all 60 historical reference prediction tables (120 probability-column comparisons), independently refitted 589 distinct native/scorer prefixes, recomputed 7,920 metric values and 5,400 per-seed paired metric fields, and checked 18,364 candidate dots against Torch autograd. Counts are checks, not independent statistical trials. Saved native batch/parent diagnostic models were actually fitted by the benchmark; their probabilities, metric values and equal added training mass were independently checked, but their redundant native refits were not rerun by the independent verifier. Wall-clock timings can be reconciled but cannot be independently recreated.

The published bundle uses source-backed links into `/home/d3vd5k/Projects/ml_project_2/results/runs/20261009T131938-e676f682743e`. Keep the source run when archiving it; the initial CPU F1 cells are not used in the published comparison. The source manifest/records hashes and correction recipe are saved in the final bundle. The all-classifier public CLI was smoke-run again after the device-policy fix: `results/smoke/20261009T154302-b0336c0443e2`, 48 completed records in 32.70 s.

