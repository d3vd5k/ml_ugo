# Shoppers: TabDiff conditional-inpainting experiment

## Protocol and evidence

Completed bundle: results/runs/20261011T154542-1b40c4b555c4;165 records, five matched seeds, native LR/RF/LightGBM and one augmentation round. Existing iterative implementations/configurations and historical results remain unchanged.

Current U5 means canonical U2 without U1 OOF scoring. Historical U5/U5_F1 used fixed three-fold OOF scoring; native parameters differ in 6/15 U2-versus-historical-U5 cells, group alpha in 2/15 and outer predictions in 15/15. The new control locks genuine U2 native/group settings separately from the retained frozen F1 checkpoint and per-cell thresholds; history is not silently renamed.

A uses original U2 interpolation; B/C use TabDiff conditional inpainting near 25/50% of original fields. Real anchors, quotas, original seed caps, common 1,000-proposal cap and native settings match. Per-group minimum feasible counts are uniformly thinned, with no refill, filter relaxation or other-generator replacement. The user approved common ties-to-even count rounding before F1 in every arm; raw negative/nonfinite values are rejected before rounding. Consequently A is a controlled one-round/common-decoder experiment, not an exact historical continuous-count/end-to-end replay.

Inner TabDiff fitting uses 7,891 real rows and excludes 1,973 validation/2,466 outer-test rows. Native inner-validation AP chooses B/C. A final generator then refits on 9,864 real outer-training rows after former validation becomes training; test remains excluded. Separate train-only quantile/category preprocessing models both classes with a joint categorical label. Known coordinates are correctly time-noised at Euler/Heun evaluations and exactly restored in final records. Preserving the intended assigned label does not guarantee its true semantic correctness.

Full official recipe: 8,000 epochs, batch 4,096, hidden width 1,024, 50 reverse steps, AdamW lr .001/weight decay 0, EMA .997 once per epoch, final EMA checkpoint without heldout selection. One frozen checkpoint per seed/split role serves all three native classifiers and both masks. No utility, surrogate classifier, TAP policy, HDBSCAN or unvalidated Flow change is added.

## Interpretation

LR shows a repeatable conditional AP gain, but minority F1/recall and MCC decrease under the validation-selected method. RF and LightGBM selected AP means improve, but their shared-row/matched-seed intervals include zero; selected ordinary CE worsens for both trees. Generator fitting dominates total runtime. These results support an LR-specific research signal, not a general three-classifier iterative replacement; no automatic rollout or retention follows.

## Paired AP differences versus A

2000 percentile-bootstrap draws, precommitted seed 20261011. The same stratified test-row resample is shared across every seed/arm/classifier; joint intervals also share matched-seed resampling. These are conditional on shared realized training/holdout and frozen inner choices, not independent folds, training-population uncertainty or proof of superiority.

| Classifier | Comparison | Mean delta AP | Positive seeds | Shared-row 95% CI | Joint 95% CI |
|---|---|---|---|---|---|
| LR | B vs A | +0.007267 | 5/5 | [+0.003544, +0.011171] | [+0.003167, +0.011663] |
| LR | C vs A | +0.006611 | 5/5 | [+0.002682, +0.010504] | [+0.001529, +0.011850] |
| LR | inner_selected vs A | +0.008204 | 5/5 | [+0.004533, +0.012035] | [+0.003724, +0.012573] |
| RF | B vs A | +0.003867 | 5/5 | [-0.007530, +0.015524] | [-0.008439, +0.016447] |
| RF | C vs A | +0.004297 | 4/5 | [-0.006610, +0.015589] | [-0.007542, +0.017605] |
| RF | inner_selected vs A | +0.004786 | 5/5 | [-0.006174, +0.016131] | [-0.006877, +0.017993] |
| LightGBM | B vs A | +0.001086 | 3/5 | [-0.008019, +0.010124] | [-0.009241, +0.012537] |
| LightGBM | C vs A | +0.002040 | 3/5 | [-0.006472, +0.010583] | [-0.007499, +0.012802] |
| LightGBM | inner_selected vs A | +0.003159 | 4/5 | [-0.005647, +0.012103] | [-0.007661, +0.014738] |

## Every metric: mean ± sample seed SD

A/B25/C50 and Inner-selected are matched-count fixed-round arms. Old references retain original end-to-end counts/tuning/scoring. Inner-selected aliases the validation-chosen outer arm, with no extra fit and no outer-test selection.

### LR

| Metric | A | B25 | C50 | Inner-selected | UGO old | U2 old | U2+F1 old | OOF+U2+F1 old |
|---|---|---|---|---|---|---|---|---|
| average_precision | 0.598369 ± 0.002769 | 0.605635 ± 0.003459 | 0.604980 ± 0.003444 | 0.606573 ± 0.003331 | 0.607707 ± 0.001577 | 0.585396 ± 0.002322 | 0.596579 ± 0.002911 | 0.596144 ± 0.001100 |
| macro_average_precision | 0.785562 ± 0.001468 | 0.789364 ± 0.001750 | 0.789129 ± 0.001947 | 0.789886 ± 0.001850 | 0.790296 ± 0.000903 | 0.778694 ± 0.001237 | 0.784493 ± 0.001578 | 0.784284 ± 0.000468 |
| minority_f1 | 0.522566 ± 0.012109 | 0.515901 ± 0.010897 | 0.515672 ± 0.008464 | 0.515400 ± 0.011558 | 0.559105 ± 0.008591 | 0.548157 ± 0.004441 | 0.527201 ± 0.003677 | 0.531006 ± 0.004203 |
| minority_recall | 0.415183 ± 0.017226 | 0.407853 ± 0.016390 | 0.406283 ± 0.013653 | 0.406806 ± 0.017716 | 0.471204 ± 0.012417 | 0.488482 ± 0.007073 | 0.433508 ± 0.008401 | 0.437173 ± 0.004897 |
| macro_f1 | 0.727805 ± 0.006208 | 0.724195 ± 0.005489 | 0.724189 ± 0.004179 | 0.723975 ± 0.005781 | 0.746462 ± 0.004519 | 0.737900 ± 0.002320 | 0.729095 ± 0.001929 | 0.731224 ± 0.002489 |
| weighted_f1 | 0.869458 ± 0.002155 | 0.867957 ± 0.001763 | 0.868104 ± 0.001259 | 0.867932 ± 0.001798 | 0.875773 ± 0.001722 | 0.868859 ± 0.000869 | 0.868440 ± 0.001095 | 0.869412 ± 0.001399 |
| sos_inverse_support_f1 | 0.586152 ± 0.010280 | 0.580433 ± 0.009221 | 0.580273 ± 0.007135 | 0.580019 ± 0.009768 | 0.617151 ± 0.007329 | 0.606942 ± 0.003783 | 0.589751 ± 0.003116 | 0.593036 ± 0.003665 |
| balanced_accuracy | 0.691709 ± 0.007490 | 0.688092 ± 0.006952 | 0.687642 ± 0.005569 | 0.687712 ± 0.007439 | 0.715976 ± 0.005611 | 0.717321 ± 0.003135 | 0.697416 ± 0.002896 | 0.699393 ± 0.002136 |
| mcc | 0.481978 ± 0.008316 | 0.475772 ± 0.006643 | 0.476616 ± 0.004554 | 0.475801 ± 0.006603 | 0.507335 ± 0.007212 | 0.481914 ± 0.004027 | 0.476658 ± 0.004693 | 0.480670 ± 0.005924 |
| threshold | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| fp | 66.200000 ± 5.019960 | 66.000000 ± 5.244044 | 64.600000 ± 5.504544 | 65.400000 ± 5.941380 | 81.800000 ± 2.863564 | 112.200000 ± 1.923538 | 80.600000 ± 6.655825 | 80.000000 ± 4.636809 |
| fn | 223.400000 ± 6.580274 | 226.200000 ± 6.260990 | 226.800000 ± 5.215362 | 226.600000 ± 6.767570 | 202.000000 ± 4.743416 | 195.400000 ± 2.701851 | 216.400000 ± 3.209361 | 215.000000 ± 1.870829 |
| tp | 158.600000 ± 6.580274 | 155.800000 ± 6.260990 | 155.200000 ± 5.215362 | 155.400000 ± 6.767570 | 180.000000 ± 4.743416 | 186.600000 ± 2.701851 | 165.600000 ± 3.209361 | 167.000000 ± 1.870829 |
| tn | 2017.800000 ± 5.019960 | 2018.000000 ± 5.244044 | 2019.400000 ± 5.504544 | 2018.600000 ± 5.941380 | 2002.200000 ± 2.863564 | 1971.800000 ± 1.923538 | 2003.400000 ± 6.655825 | 2004.000000 ± 4.636809 |
| cost | 112362.000000 ± 3242.918439 | 113760.000000 ± 3078.595459 | 114046.000000 ± 2554.805276 | 113954.000000 ± 3324.609752 | 101818.000000 ± 2346.448380 | 98822.000000 ± 1333.930283 | 109006.000000 ± 1546.424909 | 108300.000000 ± 913.591813 |
| realized_recall_test | 0.415183 ± 0.017226 | 0.407853 ± 0.016390 | 0.406283 ± 0.013653 | 0.406806 ± 0.017716 | 0.471204 ± 0.012417 | 0.488482 ± 0.007073 | 0.433508 ± 0.008401 | 0.437173 ± 0.004897 |
| realized_fpr_test | 0.031766 ± 0.002409 | 0.031670 ± 0.002516 | 0.030998 ± 0.002641 | 0.031382 ± 0.002851 | 0.039251 ± 0.001374 | 0.053839 ± 0.000923 | 0.038676 ± 0.003194 | 0.038388 ± 0.002225 |
| recall_at_fpr_test | 0.513613 ± 0.010206 | 0.509424 ± 0.013425 | 0.502094 ± 0.018913 | 0.512042 ± 0.015988 | 0.510471 ± 0.003702 | 0.473822 ± 0.004534 | 0.507330 ± 0.011910 | 0.506806 ± 0.005103 |
| roc_attained_fpr_test | 0.049520 ± 0.000401 | 0.049616 ± 0.000429 | 0.049424 ± 0.000588 | 0.049616 ± 0.000429 | 0.049520 ± 0.000215 | 0.049040 ± 0.000788 | 0.049712 ± 0.000263 | 0.049712 ± 0.000263 |
| roc_fpr_limit | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 |
| accuracy | 0.882563 ± 0.001018 | 0.881509 ± 0.000529 | 0.881833 ± 0.000615 | 0.881590 ± 0.000406 | 0.884915 ± 0.001050 | 0.875264 ± 0.000544 | 0.879562 ± 0.001672 | 0.880373 ± 0.001647 |
| precision | 0.705801 ± 0.008157 | 0.702764 ± 0.008148 | 0.706497 ± 0.010810 | 0.704250 ± 0.009757 | 0.687585 ± 0.003663 | 0.624504 ± 0.001883 | 0.673066 ± 0.014796 | 0.676308 ± 0.011689 |
| ordinary_cross_entropy | 0.315004 ± 0.001774 | 0.314114 ± 0.001950 | 0.314026 ± 0.001378 | 0.313975 ± 0.002015 | 0.325952 ± 0.003698 | 0.334213 ± 0.001825 | 0.319370 ± 0.002299 | 0.319272 ± 0.001676 |
| balanced_cross_entropy | 0.617071 ± 0.006654 | 0.615438 ± 0.010101 | 0.616431 ± 0.008494 | 0.615073 ± 0.009437 | 0.584263 ± 0.003460 | 0.552186 ± 0.001523 | 0.596981 ± 0.006691 | 0.595965 ± 0.007695 |

### RF

| Metric | A | B25 | C50 | Inner-selected | UGO old | U2 old | U2+F1 old | OOF+U2+F1 old |
|---|---|---|---|---|---|---|---|---|
| average_precision | 0.727097 ± 0.006765 | 0.730964 ± 0.005119 | 0.731393 ± 0.003642 | 0.731883 ± 0.004071 | 0.724427 ± 0.005188 | 0.717795 ± 0.005226 | 0.721025 ± 0.004854 | 0.720737 ± 0.005911 |
| macro_average_precision | 0.855668 ± 0.003574 | 0.857559 ± 0.002718 | 0.857888 ± 0.002019 | 0.858015 ± 0.002198 | 0.853983 ± 0.003146 | 0.851227 ± 0.002868 | 0.852756 ± 0.002534 | 0.852336 ± 0.003331 |
| minority_f1 | 0.653393 ± 0.004010 | 0.652170 ± 0.003091 | 0.647646 ± 0.003682 | 0.650245 ± 0.004356 | 0.645647 ± 0.008137 | 0.652711 ± 0.007468 | 0.653307 ± 0.002521 | 0.650766 ± 0.009296 |
| minority_recall | 0.637696 ± 0.013167 | 0.643455 ± 0.016390 | 0.641885 ± 0.014266 | 0.640838 ± 0.014266 | 0.615707 ± 0.022397 | 0.669634 ± 0.009865 | 0.648691 ± 0.005675 | 0.638743 ± 0.031793 |
| macro_f1 | 0.795834 ± 0.001927 | 0.794704 ± 0.002157 | 0.791867 ± 0.002276 | 0.793604 ± 0.003161 | 0.792135 ± 0.003933 | 0.793546 ± 0.004401 | 0.795144 ± 0.001481 | 0.794106 ± 0.004274 |
| weighted_f1 | 0.894145 ± 0.000684 | 0.893079 ± 0.001959 | 0.891406 ± 0.001711 | 0.892548 ± 0.002544 | 0.893239 ± 0.001086 | 0.890748 ± 0.002322 | 0.893038 ± 0.000855 | 0.893037 ± 0.001258 |
| sos_inverse_support_f1 | 0.697523 ± 0.003358 | 0.696329 ± 0.002750 | 0.692328 ± 0.003212 | 0.694660 ± 0.003963 | 0.691031 ± 0.006833 | 0.696343 ± 0.006515 | 0.697250 ± 0.002192 | 0.695175 ± 0.007729 |
| balanced_accuracy | 0.790057 ± 0.005125 | 0.791497 ± 0.005219 | 0.789752 ± 0.004823 | 0.790141 ± 0.004233 | 0.781174 ± 0.009055 | 0.799788 ± 0.005025 | 0.793443 ± 0.002231 | 0.789717 ± 0.012235 |
| mcc | 0.592058 ± 0.003615 | 0.589764 ± 0.004581 | 0.583952 ± 0.004730 | 0.587565 ± 0.006632 | 0.585657 ± 0.006665 | 0.587428 ± 0.008833 | 0.590335 ± 0.002971 | 0.588974 ± 0.007389 |
| threshold | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| fp | 120.000000 ± 6.442049 | 126.000000 ± 13.190906 | 130.000000 ± 10.839742 | 126.200000 ± 13.645512 | 111.200000 ± 9.093954 | 146.000000 ± 4.636809 | 128.800000 ± 3.768289 | 123.600000 ± 15.757538 |
| fn | 138.400000 ± 5.029911 | 136.200000 ± 6.260990 | 136.800000 ± 5.449771 | 137.200000 ± 5.449771 | 146.800000 ± 8.555700 | 126.200000 ± 3.768289 | 134.200000 ± 2.167948 | 138.000000 ± 12.144958 |
| tp | 243.600000 ± 5.029911 | 245.800000 ± 6.260990 | 245.200000 ± 5.449771 | 244.800000 ± 5.449771 | 235.200000 ± 8.555700 | 255.800000 ± 3.768289 | 247.800000 ± 2.167948 | 244.000000 ± 12.144958 |
| tn | 1964.000000 ± 6.442049 | 1958.000000 ± 13.190906 | 1954.000000 ± 10.839742 | 1957.800000 ± 13.645512 | 1972.800000 ± 9.093954 | 1938.000000 ± 4.636809 | 1955.200000 ± 3.768289 | 1960.400000 ± 15.757538 |
| cost | 70400.000000 ± 2453.395606 | 69360.000000 ± 3003.731013 | 69700.000000 ± 2624.738082 | 69862.000000 ± 2597.714380 | 74512.000000 ± 4188.229936 | 64560.000000 ± 1883.387905 | 68388.000000 ± 1055.660930 | 70236.000000 ± 5918.811536 |
| realized_recall_test | 0.637696 ± 0.013167 | 0.643455 ± 0.016390 | 0.641885 ± 0.014266 | 0.640838 ± 0.014266 | 0.615707 ± 0.022397 | 0.669634 ± 0.009865 | 0.648691 ± 0.005675 | 0.638743 ± 0.031793 |
| realized_fpr_test | 0.057582 ± 0.003091 | 0.060461 ± 0.006330 | 0.062380 ± 0.005201 | 0.060557 ± 0.006548 | 0.053359 ± 0.004364 | 0.070058 ± 0.002225 | 0.061804 ± 0.001808 | 0.059309 ± 0.007561 |
| recall_at_fpr_test | 0.616230 ± 0.012082 | 0.603665 ± 0.006826 | 0.606806 ± 0.008153 | 0.604712 ± 0.007632 | 0.595812 ± 0.007983 | 0.602618 ± 0.012878 | 0.598953 ± 0.011910 | 0.600000 ± 0.009329 |
| roc_attained_fpr_test | 0.049424 ± 0.000339 | 0.049616 ± 0.000429 | 0.049520 ± 0.000401 | 0.049424 ± 0.000480 | 0.049712 ± 0.000429 | 0.049520 ± 0.000626 | 0.049136 ± 0.000995 | 0.049232 ± 0.000728 |
| roc_fpr_limit | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 |
| accuracy | 0.895215 ± 0.000888 | 0.893674 ± 0.002985 | 0.891809 ± 0.002490 | 0.893187 ± 0.003554 | 0.895377 ± 0.000641 | 0.889619 ± 0.002389 | 0.893350 ± 0.001034 | 0.893917 ± 0.001915 |
| precision | 0.670147 ± 0.007532 | 0.661850 ± 0.018168 | 0.653996 ± 0.014569 | 0.660623 ± 0.019731 | 0.679391 ± 0.010214 | 0.636679 ± 0.008118 | 0.658048 ± 0.005286 | 0.664886 ± 0.018999 |
| ordinary_cross_entropy | 0.249473 ± 0.007836 | 0.251144 ± 0.003821 | 0.245132 ± 0.009054 | 0.251200 ± 0.003876 | 0.248658 ± 0.012051 | 0.248019 ± 0.006908 | 0.246719 ± 0.007097 | 0.250325 ± 0.007580 |
| balanced_cross_entropy | 0.507983 ± 0.017916 | 0.510905 ± 0.006553 | 0.484317 ± 0.017494 | 0.508988 ± 0.008521 | 0.491336 ± 0.038552 | 0.452869 ± 0.024049 | 0.486614 ± 0.020726 | 0.496757 ± 0.018818 |

### LightGBM

| Metric | A | B25 | C50 | Inner-selected | UGO old | U2 old | U2+F1 old | OOF+U2+F1 old |
|---|---|---|---|---|---|---|---|---|
| average_precision | 0.735806 ± 0.004109 | 0.736892 ± 0.003620 | 0.737846 ± 0.004981 | 0.738965 ± 0.005541 | 0.730186 ± 0.003323 | 0.728091 ± 0.005350 | 0.731753 ± 0.003107 | 0.733878 ± 0.001105 |
| macro_average_precision | 0.860697 ± 0.002089 | 0.861187 ± 0.001850 | 0.861671 ± 0.002622 | 0.862275 ± 0.002879 | 0.857640 ± 0.001749 | 0.856694 ± 0.002742 | 0.858531 ± 0.001609 | 0.859613 ± 0.000542 |
| minority_f1 | 0.659773 ± 0.005592 | 0.662275 ± 0.009689 | 0.660147 ± 0.009162 | 0.661239 ± 0.009010 | 0.648455 ± 0.006509 | 0.657688 ± 0.008022 | 0.657342 ± 0.003281 | 0.658055 ± 0.002841 |
| minority_recall | 0.658115 ± 0.017522 | 0.674869 ± 0.023517 | 0.666492 ± 0.021714 | 0.673298 ± 0.023517 | 0.634555 ± 0.009366 | 0.690576 ± 0.011619 | 0.665445 ± 0.009689 | 0.665445 ± 0.009689 |
| macro_f1 | 0.798806 ± 0.002768 | 0.799501 ± 0.005280 | 0.798589 ± 0.004821 | 0.798915 ± 0.004820 | 0.792825 ± 0.003700 | 0.795597 ± 0.004788 | 0.796809 ± 0.001763 | 0.797267 ± 0.002051 |
| weighted_f1 | 0.894766 ± 0.001040 | 0.894212 ± 0.002500 | 0.894140 ± 0.001932 | 0.893937 ± 0.002212 | 0.892468 ± 0.001781 | 0.890780 ± 0.002631 | 0.893067 ± 0.000895 | 0.893348 ± 0.001653 |
| sos_inverse_support_f1 | 0.702847 ± 0.004709 | 0.704789 ± 0.008309 | 0.703038 ± 0.007812 | 0.703893 ± 0.007698 | 0.693183 ± 0.005637 | 0.700414 ± 0.007015 | 0.700550 ± 0.002801 | 0.701185 ± 0.002580 |
| balanced_accuracy | 0.798203 ± 0.006939 | 0.804181 ± 0.009766 | 0.800952 ± 0.009264 | 0.803396 ± 0.009630 | 0.787719 ± 0.004630 | 0.807764 ± 0.005615 | 0.799805 ± 0.003780 | 0.799997 ± 0.002947 |
| mcc | 0.597764 ± 0.005545 | 0.599402 ± 0.010656 | 0.597381 ± 0.009743 | 0.598223 ± 0.009790 | 0.585898 ± 0.007324 | 0.592374 ± 0.009560 | 0.593739 ± 0.003599 | 0.594693 ± 0.003954 |
| threshold | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| fp | 128.600000 ± 8.203658 | 138.600000 ± 11.013628 | 134.600000 ± 7.765307 | 138.600000 ± 11.013628 | 123.200000 ± 2.588436 | 156.400000 ± 6.655825 | 137.200000 ± 5.310367 | 136.400000 ± 8.961027 |
| fn | 130.600000 ± 6.693280 | 124.200000 ± 8.983318 | 127.400000 ± 8.294577 | 124.800000 ± 8.983318 | 139.600000 ± 3.577709 | 118.200000 ± 4.438468 | 127.800000 ± 3.701351 | 127.800000 ± 3.701351 |
| tp | 251.400000 ± 6.693280 | 257.800000 ± 8.983318 | 254.600000 ± 8.294577 | 257.200000 ± 8.983318 | 242.400000 ± 3.577709 | 263.800000 ± 4.438468 | 254.200000 ± 3.701351 | 254.200000 ± 3.701351 |
| tn | 1955.400000 ± 8.203658 | 1945.400000 ± 11.013628 | 1949.400000 ± 7.765307 | 1945.400000 ± 11.013628 | 1960.800000 ± 2.588436 | 1927.600000 ± 6.655825 | 1946.800000 ± 5.310367 | 1947.600000 ± 8.961027 |
| cost | 66586.000000 ± 3269.515255 | 63486.000000 ± 4403.763163 | 65046.000000 ± 4079.280574 | 63786.000000 ± 4398.991930 | 71032.000000 ± 1785.124646 | 60664.000000 ± 2203.050158 | 65272.000000 ± 1804.389093 | 65264.000000 ± 1767.492574 |
| realized_recall_test | 0.658115 ± 0.017522 | 0.674869 ± 0.023517 | 0.666492 ± 0.021714 | 0.673298 ± 0.023517 | 0.634555 ± 0.009366 | 0.690576 ± 0.011619 | 0.665445 ± 0.009689 | 0.665445 ± 0.009689 |
| realized_fpr_test | 0.061708 ± 0.003936 | 0.066507 ± 0.005285 | 0.064587 ± 0.003726 | 0.066507 ± 0.005285 | 0.059117 ± 0.001242 | 0.075048 ± 0.003194 | 0.065835 ± 0.002548 | 0.065451 ± 0.004300 |
| recall_at_fpr_test | 0.615707 ± 0.007264 | 0.607330 ± 0.010951 | 0.609948 ± 0.009618 | 0.608377 ± 0.012636 | 0.596335 ± 0.010373 | 0.604712 ± 0.005236 | 0.612042 ± 0.004301 | 0.609424 ± 0.007721 |
| roc_attained_fpr_test | 0.049424 ± 0.000480 | 0.049712 ± 0.000429 | 0.048656 ± 0.001540 | 0.049712 ± 0.000429 | 0.049616 ± 0.000429 | 0.049616 ± 0.000644 | 0.049136 ± 0.001051 | 0.049424 ± 0.000480 |
| roc_fpr_limit | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 | 0.050000 ± 0.000000 |
| accuracy | 0.894891 ± 0.001196 | 0.893431 ± 0.002665 | 0.893755 ± 0.001622 | 0.893187 ± 0.002392 | 0.893431 ± 0.001657 | 0.888646 ± 0.002833 | 0.892539 ± 0.001111 | 0.892863 ± 0.002301 |
| precision | 0.661837 ± 0.009053 | 0.650743 ± 0.012703 | 0.654344 ± 0.007716 | 0.650220 ± 0.012235 | 0.663023 ± 0.005295 | 0.627895 ± 0.009599 | 0.649562 ± 0.006228 | 0.651091 ± 0.011860 |
| ordinary_cross_entropy | 0.235921 ± 0.002602 | 0.237326 ± 0.001552 | 0.236394 ± 0.002919 | 0.236568 ± 0.002499 | 0.241562 ± 0.001136 | 0.244166 ± 0.001577 | 0.239252 ± 0.001353 | 0.238946 ± 0.001855 |
| balanced_cross_entropy | 0.462575 ± 0.006423 | 0.458864 ± 0.009704 | 0.456785 ± 0.007342 | 0.456655 ± 0.009486 | 0.469407 ± 0.008495 | 0.432006 ± 0.003080 | 0.461493 ± 0.005550 | 0.461716 ± 0.006168 |

## Per-seed decisions and paired results

| Classifier | Seed | Inner choice | Inner B AP | Inner C AP | Outer B-A AP | Outer C-A AP | Matched accepted | Round shortfall |
|---|---|---|---|---|---|---|---|---|
| LR | 0 | B | 0.647168 | 0.646532 | +0.008751 | +0.006112 | 356 | 630 |
| RF | 0 | C | 0.739748 | 0.742854 | +0.003063 | +0.001964 | 541 | 445 |
| LightGBM | 0 | B | 0.753294 | 0.753285 | +0.004783 | +0.000239 | 566 | 420 |
| LR | 1 | B | 0.653618 | 0.653381 | +0.009003 | +0.009120 | 349 | 637 |
| RF | 1 | C | 0.730496 | 0.741819 | +0.007572 | +0.013268 | 433 | 553 |
| LightGBM | 1 | C | 0.734148 | 0.746047 | -0.000369 | +0.009997 | 379 | 607 |
| LR | 2 | B | 0.655327 | 0.654666 | +0.006188 | +0.001549 | 273 | 713 |
| RF | 2 | B | 0.718005 | 0.714804 | +0.004274 | +0.006074 | 322 | 664 |
| LightGBM | 2 | B | 0.731625 | 0.726384 | +0.006269 | -0.000946 | 430 | 556 |
| LR | 3 | C | 0.648960 | 0.652369 | +0.006403 | +0.011092 | 249 | 737 |
| RF | 3 | B | 0.724912 | 0.723739 | +0.004090 | +0.002680 | 250 | 736 |
| LightGBM | 3 | B | 0.747833 | 0.746715 | -0.008907 | -0.004430 | 258 | 728 |
| LR | 4 | B | 0.616062 | 0.615996 | +0.005988 | +0.005181 | 384 | 602 |
| RF | 4 | B | 0.717930 | 0.707722 | +0.000336 | -0.002503 | 451 | 535 |
| LightGBM | 4 | B | 0.730587 | 0.721158 | +0.003655 | +0.005343 | 542 | 444 |

## Validity, counts and duplicates

Checks enforce finite/nonnegative numbers, integer-valued counts after the common decoder, rates/SpecialDay in [0,1], observed role-training categories and zero-count/zero-duration consistency. No empirical maximum or BounceRates≤ExitRates rule is imposed. Exact/near duplicates are diagnostic-only. Near means classifier-standardized Euclidean distance ≤ .001. diagnostics.csv retains accepted duplicate/anchor/group coverage, overlapping constraint failures and timing; group_quotas.csv retains every original quota and shortfall.

| Classifier | Arm | Proposals | Schema valid | F1 eligible | Accepted | Accepted real duplicates | Accepted pool duplicates | Accepted near |
|---|---|---|---|---|---|---|---|---|
| LightGBM | A | 4930 | 4476 | 3067 | 2175 | 0 | 0 | 1 |
| LightGBM | B | 4930 | 4929 | 2392 | 2175 | 220 | 125 | 221 |
| LightGBM | C | 4930 | 4914 | 2293 | 2175 | 0 | 2 | 0 |
| LR | A | 4930 | 4445 | 2218 | 1611 | 0 | 0 | 0 |
| LR | B | 4930 | 4928 | 1787 | 1611 | 128 | 59 | 129 |
| LR | C | 4930 | 4912 | 1788 | 1611 | 1 | 2 | 1 |
| RF | A | 4930 | 4524 | 2818 | 1997 | 0 | 0 | 0 |
| RF | B | 4930 | 4930 | 2250 | 1997 | 175 | 114 | 177 |
| RF | C | 4930 | 4916 | 2130 | 1997 | 2 | 3 | 2 |

## Checkpoints and exact configurations

| Seed | Role | Checkpoint SHA256 | Epochs | Gradient updates | Training seconds |
|---|---|---|---|---|---|
| 0 | inner | 571fd7949ea1f07077fe04333cf72e7d8a928b952c9f524f78c6964257bea094 | 8000 | 16000 | 1967.561 |
| 0 | outer | e3042394dffd6bad638474a809b52b9a73752d5b239b5ae33a9882317fa1d212 | 8000 | 24000 | 2521.480 |
| 1 | inner | 9b553ddb34cfb26213010c06a724a8fbe1b90325a5a53f76aec71320612ead0e | 8000 | 16000 | 1967.485 |
| 1 | outer | 350e63cc1116edfc5a1cebefb7a722608971ea94cd3f6d4980d04720e3247b8b | 8000 | 24000 | 2595.870 |
| 2 | inner | 70e17e3a8ab8375915c424d035b0ecdec2271b384f4232f9741385204dfaa762 | 8000 | 16000 | 2078.026 |
| 2 | outer | ac7c51ad676bcaa0650922b71ca051d0af1a3fdd3937516d5d21be4e8a89e7e6 | 8000 | 24000 | 2655.631 |
| 3 | inner | 8656bafa4ab3bcade25ce9df195f0d55931de592a02741c4c5d4dd2d1a47879b | 8000 | 16000 | 2093.173 |
| 3 | outer | 8648fa046a51981de58e10987d6b4d54c9a64c1d5ed71bc752ff123cdd2828d9 | 8000 | 24000 | 2660.378 |
| 4 | inner | 788fe05fa3d5d5e20d5abe6c4d173ead8b7b3f7e29bfab4e444da3f8592efd8c | 8000 | 16000 | 2098.451 |
| 4 | outer | 2cab4d5fb185efd68482d1c89ccab963e155050c71a9d769ade629c2cc93da92 | 8000 | 24000 | 2676.274 |

| Seed | B original columns | C original columns |
|---|---|---|
| 0 | 4/17: Administrative, Administrative_Duration, Browser, Region | 8/17: Administrative, Administrative_Duration, SpecialDay, Month, OperatingSystems, Browser, Region, VisitorType |
| 1 | 4/17: PageValues, SpecialDay, Browser, Weekend | 8/17: Administrative, Administrative_Duration, ProductRelated, ProductRelated_Duration, PageValues, SpecialDay, Browser, Weekend |
| 2 | 4/17: OperatingSystems, Region, TrafficType, Weekend | 9/17: ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, OperatingSystems, Region, TrafficType, VisitorType, Weekend |
| 3 | 4/17: PageValues, Month, Browser, Region | 9/17: Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, PageValues, Month, Browser, Region, TrafficType |
| 4 | 4/17: SpecialDay, OperatingSystems, Browser, VisitorType | 9/17: Administrative, Administrative_Duration, SpecialDay, Month, OperatingSystems, Browser, Region, TrafficType, VisitorType |

| Classifier | Seed | Native parameters | Group alpha | F1 tau | Frozen F1 checkpoint SHA256 |
|---|---|---|---|---|---|
| LR | 0 | {"C": 0.1} | 0 | 0.8 | b0cf2dac828dffc258bcb2ecda16d3083599df5a8881453c7e0f6b335fc47883 |
| RF | 0 | {"max_depth": null, "max_features": 0.8, "min_samples_leaf": 3, "n_estimators": 200} | 1 | 0.6 | b0cf2dac828dffc258bcb2ecda16d3083599df5a8881453c7e0f6b335fc47883 |
| LightGBM | 0 | {"learning_rate": 0.05, "min_child_samples": 50, "n_estimators": 100, "num_leaves": 15} | 0.5 | 0.5 | b0cf2dac828dffc258bcb2ecda16d3083599df5a8881453c7e0f6b335fc47883 |
| LR | 1 | {"C": 0.1} | 0 | 0.8 | 32ee43fa3543dbaa3bc65bac9b155c2a3aa8299e03c2ec538a05cd62c139c54c |
| RF | 1 | {"max_depth": null, "max_features": 0.8, "min_samples_leaf": 3, "n_estimators": 200} | 1 | 0.6 | 32ee43fa3543dbaa3bc65bac9b155c2a3aa8299e03c2ec538a05cd62c139c54c |
| LightGBM | 1 | {"learning_rate": 0.05, "min_child_samples": 20, "n_estimators": 100, "num_leaves": 15} | 1 | 0.6 | 32ee43fa3543dbaa3bc65bac9b155c2a3aa8299e03c2ec538a05cd62c139c54c |
| LR | 2 | {"C": 0.1} | 0 | 0.8 | 8478fa7dad3d638a2834d142e46efff4567dd362bfd815ea79e4bd2aae980b91 |
| RF | 2 | {"max_depth": null, "max_features": 0.8, "min_samples_leaf": 3, "n_estimators": 200} | 1 | 0.8 | 8478fa7dad3d638a2834d142e46efff4567dd362bfd815ea79e4bd2aae980b91 |
| LightGBM | 2 | {"learning_rate": 0.05, "min_child_samples": 50, "n_estimators": 100, "num_leaves": 15} | 0.5 | 0.6 | 8478fa7dad3d638a2834d142e46efff4567dd362bfd815ea79e4bd2aae980b91 |
| LR | 3 | {"C": 100} | 0 | 0.8 | f6cfab619a65163c5caba7e441531bfa2091e51f3ad09f27e607d063ecb63307 |
| RF | 3 | {"max_depth": null, "max_features": 0.8, "min_samples_leaf": 8, "n_estimators": 100} | 0.5 | 0.8 | f6cfab619a65163c5caba7e441531bfa2091e51f3ad09f27e607d063ecb63307 |
| LightGBM | 3 | {"learning_rate": 0.05, "min_child_samples": 50, "n_estimators": 100, "num_leaves": 15} | 0.5 | 0.8 | f6cfab619a65163c5caba7e441531bfa2091e51f3ad09f27e607d063ecb63307 |
| LR | 4 | {"C": 100} | 0 | 0.8 | 9ace7f8369bbdbe6676101f8bf05af0507f3648de86da77bba5e6507400b45d8 |
| RF | 4 | {"max_depth": null, "max_features": 0.8, "min_samples_leaf": 8, "n_estimators": 100} | 0.5 | 0.7 | 9ace7f8369bbdbe6676101f8bf05af0507f3648de86da77bba5e6507400b45d8 |
| LightGBM | 4 | {"learning_rate": 0.05, "min_child_samples": 50, "n_estimators": 100, "num_leaves": 15} | 0.5 | 0.6 | 9ace7f8369bbdbe6676101f8bf05af0507f3648de86da77bba5e6507400b45d8 |

Official MIT TabDiff commit 5ecdb3356261aea72716cc9a779f31d7ad083bf4; TAP reference f7de90f86a63f2a446d0eaf2b81c67ec59f7f7d0. Checkpoints live at shoppers/fold-0/seed-S/{inner,outer}/generator/generator.pt in the bundle. Source/config snapshots, fit IDs, categorical/quantile statistics and optimizer/EMA/scheduler/RNG states accompany them.

## Runtime accounting

Generator training is shared once per seed/role (10 fits), not charged repeatedly per classifier/mask. A sampling includes actual original proposal generation; shared scheduling excludes it. Initial native fits, matching and duplicate/near diagnostics are logged separately. Derived selected aliases and historical imports add no fits; historical time is not a fresh matched comparison.

| Component | Observed seconds |
|---|---|
| generator_training_seconds_total | 23314.328 |
| generator_total_seconds_total | 23319.905 |
| initial_native_fit_seconds_total | 41.670 |
| schedule_seconds_total | 2.531 |
| matching_seconds_total | 0.008 |
| native_arm_fit_seconds_total | 125.313 |
| wall_seconds | 23627.074 |

| Arm | Sampling seconds | Filtering seconds | Diagnostic seconds | Native fit seconds |
|---|---|---|---|---|
| A | 0.138 | 1.923 | 5.488 | 42.572 |
| B | 42.081 | 1.571 | 6.035 | 41.635 |
| C | 37.908 | 1.557 | 6.019 | 41.106 |

## Verification and decision boundary

Actual three-classifier training-only smoke preceded scientific training; 283 tests and 307 subtests passed. Independent evidence refits native models and real-only preprocessing/groups, replays anchors/partners/float32 interpolation/quotas/thinning, validates F1/schema/duplicates/counts and uses live frozen-weight Euler/Heun conditioning probes with independent noise witnesses. Checkpoint/model/buffer state remains unchanged during sampling/filtering. Generator fitting contains neither active validation/test nor synthetic observations; former validation enters only the explicit final training role.

TabDiff retention is not automatic. Shoppers remains a development benchmark; five means alone establish no superiority, and conditional intervals omit training-population and inner-selection uncertainty. No full iterative rollout, uniform-anchor test or unseen-dataset confirmation has started. Freeze the method before previously unexamined benchmarks. The retained U2+F1 configuration remains unchanged pending the user decision.

## Complete artifacts

Bundle: results/runs/20261011T154542-1b40c4b555c4.

- independent_report.metrics.csv: every seed/role/arm/reference and 24 independently recomputed metrics.
- all_metrics_summary.csv and paired_metrics_summary.csv: metric means/seedSD and all paired differences.
- independent_report.json and independent_verification.json: uncertainty and actual-source/fitting/native-state/conditioning evidence.
- diagnostics.csv and group_quotas.csv: accepted coverage/duplicates, constraint failures, matched quotas and time.
- control_audit.json, source.zip, configuration snapshots and recipes/: exact controls and executable durable recipes.

Sources: https://github.com/MinkaiXu/TabDiff ; https://github.com/oooranz/TAP ; https://arxiv.org/html/2605.10315v1 .
