# Pool-random + retained F1 — retention pending

Run: `/home/d3vd5k/Projects/ml_project_2/results/runs/20261009T170013-d5be50bc2107`
Independent verification: **passed**, 770287 checks.
Exploratory five matched seeds; three native classifiers.

## Design and scope
Shoppers only. One fixed shared outer holdout; matched seeds are not independent folds. Phase 1 remains open; retained U5+F1 remains retained, the new combination remains pending, and pool-utility is discarded.
Original UGO/U2/U5/U5+F1 references retain source provenance and original fitting/preprocessing/generation access. Plain random is the original-access native pool-random comparator. References are **not count-matched**.
Native logistic regression, random forest and LightGBM use locked canonical U2-selected parameters/group alpha. No LR surrogate, utility scores or guidance gradients.
K=5 physical proposals per distinct planned anchor. One random eligible representative per anchor precedes quota selection. F1 evaluates only attempted candidates on original coordinates, replaces neighbor relocation, and rejects after random selection. No screening all K, rescue, regeneration, refill or quota override.
Independent audit confirms identical **initial raw proposal pools, anchors and partners** across inner plain random/four tau trials and outer random/selected F1 at the same initial native prefix. Final coordinates/masks intentionally differ because F1 replaces relocation; later native-retrained trajectories and pools may diverge.
Nominal selected-attempt cap 1000; maximum 20 rounds. F1 rejection can reduce accepted dose. Native models are refitted on exact growing prefixes. The original frozen auto/CUDA F1 checkpoint is imported inference-only.
Four taus (0.5, 0.6, 0.7, 0.8) are selected only by inner validation AP, then locked for outer evaluation. Outer validation predictions deliberately overlap real outer fitting; outer test alone is final held-out evidence.
All fields use mean ± sample SD (ddof=1). AP and its primary alias are counted only once; ordinary and balanced cross-entropy are included. Confusion counts, thresholds, costs and ROC diagnostics are numeric fields, not all optimization objectives.

## Interpretation for the retention decision

- **Logistic regression:** AP improves over pool-random in all five seeds and over retained U5+F1 in all five. Against pool-random, minority F1/recall, balanced accuracy and mean MCC decline while ordinary CE improves and balanced CE worsens. Original UGO still has higher mean AP/F1/MCC. This is a ranking-versus-thresholded-recall tradeoff, not dominance.
- **Random forest:** AP improves over pool-random in four of five seeds, but mean minority F1/MCC decline and recall falls in every seed. Its AP advantage over retained U5+F1 is present in only three seeds. Ordinary CE improves on average; balanced CE and cost worsen against pool-random.
- **LightGBM:** AP falls against pool-random in four of five seeds and against retained U5+F1 in every seed. Minority F1/MCC improve against retained U5+F1 in every seed, but recall still declines against pool-random. Ordinary CE improves against pool-random; balanced CE and cost worsen.
- **Shared pattern:** F1 reduces false positives but increases false negatives against pool-random. Its mean accepted synthetic dose is smaller, so this experiment cannot isolate coordinate quality from synthetic training weight. All final new trajectories consume the complete selected-attempt budget and generate the same physical K-pool totals; none is refilled. LR has three accepting rounds and RF/LightGBM four, with matching new-random/new-F1 native fit counts. Lower observed F1 trajectory time is not an iteration-cadence reduction and excludes the threshold search/historical Flow training.

**No retention decision is applied.** Pool-utility is discarded; U5+F1 stays retained; pool-random+F1 remains pending your decision. Phase 1 is open/unfrozen and broader evaluation remains closed.

## All 24 held-out metrics

### logistic_regression
| Metric | Original UGO | Original U2 | Original U5 | Retained U5+F1 | Pool-random | Pool-random+F1 |
| --- | --- | --- | --- | --- | --- | --- |
| average_precision | 0.607707 ± 0.00157737 | 0.585396 ± 0.0023219 | 0.585378 ± 0.00149162 | 0.596144 ± 0.00109957 | 0.592631 ± 0.00142644 | 0.600917 ± 0.0016315 |
| macro_average_precision | 0.790296 ± 0.000902782 | 0.778694 ± 0.00123664 | 0.778757 ± 0.000871474 | 0.784284 ± 0.000467831 | 0.782995 ± 0.000689517 | 0.786984 ± 0.000756557 |
| minority_f1 | 0.559105 ± 0.00859084 | 0.548157 ± 0.00444078 | 0.544804 ± 0.00215273 | 0.531006 ± 0.00420265 | 0.550685 ± 0.00322349 | 0.532443 ± 0.0049132 |
| minority_recall | 0.471204 ± 0.0124173 | 0.488482 ± 0.00707291 | 0.48377 ± 0.00503544 | 0.437173 ± 0.00489746 | 0.480628 ± 0.00510303 | 0.433508 ± 0.00840147 |
| macro_f1 | 0.746462 ± 0.00451944 | 0.7379 ± 0.00231993 | 0.736103 ± 0.00111023 | 0.731224 ± 0.0024885 | 0.740221 ± 0.00180209 | 0.732486 ± 0.00251086 |
| weighted_f1 | 0.875773 ± 0.00172192 | 0.868859 ± 0.000868646 | 0.868135 ± 0.00057486 | 0.869412 ± 0.00139888 | 0.871036 ± 0.000920094 | 0.870553 ± 0.00098018 |
| sos_inverse_support_f1 | 0.617151 ± 0.00732893 | 0.606942 ± 0.00378319 | 0.604071 ± 0.00182082 | 0.593036 ± 0.00366527 | 0.609405 ± 0.00277744 | 0.594419 ± 0.00416378 |
| balanced_accuracy | 0.715976 ± 0.00561094 | 0.717321 ± 0.00313534 | 0.715109 ± 0.00182793 | 0.699393 ± 0.00213649 | 0.716034 ± 0.00210989 | 0.698904 ± 0.00337768 |
| mcc | 0.507335 ± 0.00721208 | 0.481914 ± 0.00402708 | 0.478696 ± 0.00213378 | 0.48067 ± 0.00592389 | 0.489141 ± 0.00367611 | 0.485656 ± 0.00411164 |
| threshold | 0.5 ± 0 | 0.5 ± 0 | 0.5 ± 0 | 0.5 ± 0 | 0.5 ± 0 | 0.5 ± 0 |
| fp | 81.8 ± 2.86356 | 112.2 ± 1.92354 | 111.6 ± 3.50714 | 80 ± 4.63681 | 101.2 ± 3.42053 | 74.4 ± 4.39318 |
| fn | 202 ± 4.74342 | 195.4 ± 2.70185 | 197.2 ± 1.92354 | 215 ± 1.87083 | 198.4 ± 1.94936 | 216.4 ± 3.20936 |
| tp | 180 ± 4.74342 | 186.6 ± 2.70185 | 184.8 ± 1.92354 | 167 ± 1.87083 | 183.6 ± 1.94936 | 165.6 ± 3.20936 |
| tn | 2002.2 ± 2.86356 | 1971.8 ± 1.92354 | 1972.4 ± 3.50714 | 2004 ± 4.63681 | 1982.8 ± 3.42053 | 2009.6 ± 4.39318 |
| cost | 101818 ± 2346.45 | 98822 ± 1333.93 | 99716 ± 931.252 | 108300 ± 913.592 | 100212 ± 952.717 | 108944 ± 1568.42 |
| realized_recall_test | 0.471204 ± 0.0124173 | 0.488482 ± 0.00707291 | 0.48377 ± 0.00503544 | 0.437173 ± 0.00489746 | 0.480628 ± 0.00510303 | 0.433508 ± 0.00840147 |
| realized_fpr_test | 0.0392514 ± 0.00137407 | 0.0538388 ± 0.000923003 | 0.0535509 ± 0.00168289 | 0.0383877 ± 0.00222496 | 0.0485605 ± 0.00164133 | 0.0357006 ± 0.00210805 |
| recall_at_fpr_test | 0.510471 ± 0.00370213 | 0.473822 ± 0.00453416 | 0.471728 ± 0.00341319 | 0.506806 ± 0.00510303 | 0.489005 ± 0.00567526 | 0.515183 ± 0.0059695 |
| roc_attained_fpr_test | 0.0495202 ± 0.000214594 | 0.0490403 ± 0.000788468 | 0.0492322 ± 0.000935393 | 0.0497121 ± 0.000262823 | 0.0498081 ± 0.000214594 | 0.0491363 ± 0.000935393 |
| roc_fpr_limit | 0.05 ± 0 | 0.05 ± 0 | 0.05 ± 0 | 0.05 ± 0 | 0.05 ± 0 | 0.05 ± 0 |
| accuracy | 0.884915 ± 0.00104965 | 0.875264 ± 0.000544055 | 0.874777 ± 0.000831058 | 0.880373 ± 0.00164721 | 0.878508 ± 0.00105745 | 0.882076 ± 0.00100972 |
| precision | 0.687585 ± 0.0036631 | 0.624504 ± 0.00188341 | 0.623552 ± 0.00536541 | 0.676308 ± 0.0116894 | 0.644737 ± 0.00649614 | 0.690205 ± 0.00958402 |
| ordinary_cross_entropy | 0.325952 ± 0.00369801 | 0.334213 ± 0.00182458 | 0.334047 ± 0.00102024 | 0.319272 ± 0.00167559 | 0.333209 ± 0.000799925 | 0.317721 ± 0.00170993 |
| balanced_cross_entropy | 0.584263 ± 0.00346049 | 0.552186 ± 0.00152274 | 0.551751 ± 0.0014448 | 0.595965 ± 0.00769547 | 0.544489 ± 0.000714031 | 0.595708 ± 0.00637202 |

### random_forest
| Metric | Original UGO | Original U2 | Original U5 | Retained U5+F1 | Pool-random | Pool-random+F1 |
| --- | --- | --- | --- | --- | --- | --- |
| average_precision | 0.724427 ± 0.00518838 | 0.717795 ± 0.00522554 | 0.717602 ± 0.00707065 | 0.720737 ± 0.00591063 | 0.719535 ± 0.00774684 | 0.722518 ± 0.00503147 |
| macro_average_precision | 0.853983 ± 0.00314597 | 0.851227 ± 0.0028684 | 0.850882 ± 0.00414086 | 0.852336 ± 0.00333072 | 0.85204 ± 0.00414859 | 0.853514 ± 0.00257837 |
| minority_f1 | 0.645647 ± 0.00813696 | 0.652711 ± 0.00746803 | 0.647136 ± 0.0129987 | 0.650766 ± 0.00929594 | 0.653455 ± 0.00484127 | 0.650469 ± 0.00683955 |
| minority_recall | 0.615707 ± 0.0223971 | 0.669634 ± 0.00986463 | 0.662304 ± 0.0311397 | 0.638743 ± 0.0317931 | 0.672775 ± 0.0106336 | 0.646597 ± 0.0125545 |
| macro_f1 | 0.792135 ± 0.00393277 | 0.793546 ± 0.00440128 | 0.790365 ± 0.00675783 | 0.794106 ± 0.00427367 | 0.793846 ± 0.00304479 | 0.793432 ± 0.00373899 |
| weighted_f1 | 0.893239 ± 0.00108631 | 0.890748 ± 0.00232212 | 0.88922 ± 0.00266679 | 0.893037 ± 0.00125762 | 0.890742 ± 0.00197994 | 0.892103 ± 0.00162889 |
| sos_inverse_support_f1 | 0.691031 ± 0.00683288 | 0.696343 ± 0.00651548 | 0.69151 ± 0.011056 | 0.695175 ± 0.00772896 | 0.69695 ± 0.0042705 | 0.694761 ± 0.00587745 |
| balanced_accuracy | 0.781174 ± 0.00905512 | 0.799788 ± 0.00502507 | 0.795979 ± 0.01306 | 0.789717 ± 0.0122352 | 0.800975 ± 0.00397152 | 0.792012 ± 0.00574108 |
| mcc | 0.585657 ± 0.00666505 | 0.587428 ± 0.00883332 | 0.581324 ± 0.0135116 | 0.588974 ± 0.00738865 | 0.588185 ± 0.0058881 | 0.58692 ± 0.00740525 |
| threshold | 0.5 ± 0 | 0.5 ± 0 | 0.5 ± 0 | 0.5 ± 0 | 0.5 ± 0 | 0.5 ± 0 |
| fp | 111.2 ± 9.09395 | 146 ± 4.63681 | 146.6 ± 12.1778 | 123.6 ± 15.7575 | 147.6 ± 8.6487 | 130.4 ± 3.28634 |
| fn | 146.8 ± 8.5557 | 126.2 ± 3.76829 | 129 ± 11.8954 | 138 ± 12.145 | 125 ± 4.06202 | 135 ± 4.79583 |
| tp | 235.2 ± 8.5557 | 255.8 ± 3.76829 | 253 ± 11.8954 | 244 ± 12.145 | 257 ± 4.06202 | 247 ± 4.79583 |
| tn | 1972.8 ± 9.09395 | 1938 ± 4.63681 | 1937.4 ± 12.1778 | 1960.4 ± 15.7575 | 1936.4 ± 8.6487 | 1953.6 ± 3.28634 |
| cost | 74512 ± 4188.23 | 64560 ± 1883.39 | 65966 ± 5840.36 | 70236 ± 5918.81 | 63976 ± 1965.96 | 68804 ± 2374.57 |
| realized_recall_test | 0.615707 ± 0.0223971 | 0.669634 ± 0.00986463 | 0.662304 ± 0.0311397 | 0.638743 ± 0.0317931 | 0.672775 ± 0.0106336 | 0.646597 ± 0.0125545 |
| realized_fpr_test | 0.0533589 ± 0.0043637 | 0.0700576 ± 0.00222496 | 0.0703455 ± 0.0058435 | 0.059309 ± 0.0075612 | 0.0708253 ± 0.00415005 | 0.062572 ± 0.00157694 |
| recall_at_fpr_test | 0.595812 ± 0.00798322 | 0.602618 ± 0.0128779 | 0.597906 ± 0.0108883 | 0.6 ± 0.00932907 | 0.58377 ± 0.00763214 | 0.594241 ± 0.0080686 |
| roc_attained_fpr_test | 0.0497121 ± 0.000429188 | 0.0495202 ± 0.000625643 | 0.0487524 ± 0.00181456 | 0.0492322 ± 0.000727723 | 0.0492322 ± 0.000727723 | 0.0496161 ± 0.000429188 |
| roc_fpr_limit | 0.05 ± 0 | 0.05 ± 0 | 0.05 ± 0 | 0.05 ± 0 | 0.05 ± 0 | 0.05 ± 0 |
| accuracy | 0.895377 ± 0.000641176 | 0.889619 ± 0.00238876 | 0.88824 ± 0.00235757 | 0.893917 ± 0.00191496 | 0.889457 ± 0.00249318 | 0.892376 ± 0.00136316 |
| precision | 0.679391 ± 0.0102144 | 0.636679 ± 0.00811824 | 0.633535 ± 0.011033 | 0.664886 ± 0.0189991 | 0.635432 ± 0.0108745 | 0.654492 ± 0.00397859 |
| ordinary_cross_entropy | 0.248658 ± 0.0120508 | 0.248019 ± 0.00690779 | 0.249605 ± 0.00828237 | 0.250325 ± 0.00757987 | 0.250818 ± 0.00851151 | 0.246849 ± 0.0102963 |
| balanced_cross_entropy | 0.491336 ± 0.0385517 | 0.452869 ± 0.0240491 | 0.459299 ± 0.0319521 | 0.496757 ± 0.0188185 | 0.455654 ± 0.0311065 | 0.482459 ± 0.0369899 |

### lightgbm
| Metric | Original UGO | Original U2 | Original U5 | Retained U5+F1 | Pool-random | Pool-random+F1 |
| --- | --- | --- | --- | --- | --- | --- |
| average_precision | 0.730186 ± 0.00332257 | 0.728091 ± 0.00535036 | 0.729552 ± 0.0042699 | 0.733878 ± 0.00110531 | 0.732201 ± 0.00313829 | 0.731082 ± 0.00137184 |
| macro_average_precision | 0.85764 ± 0.00174889 | 0.856694 ± 0.00274171 | 0.857428 ± 0.00217354 | 0.859613 ± 0.000542224 | 0.858889 ± 0.00166312 | 0.858219 ± 0.000714463 |
| minority_f1 | 0.648455 ± 0.00650872 | 0.657688 ± 0.00802195 | 0.657273 ± 0.00450093 | 0.658055 ± 0.00284149 | 0.663982 ± 0.00262258 | 0.663237 ± 0.00180674 |
| minority_recall | 0.634555 ± 0.00936573 | 0.690576 ± 0.011619 | 0.68377 ± 0.0077214 | 0.665445 ± 0.0096894 | 0.692147 ± 0.00754181 | 0.674346 ± 0.00542839 |
| macro_f1 | 0.792825 ± 0.00369958 | 0.795597 ± 0.00478829 | 0.795715 ± 0.0027072 | 0.797267 ± 0.00205053 | 0.799637 ± 0.00148687 | 0.800139 ± 0.00119159 |
| weighted_f1 | 0.892468 ± 0.00178098 | 0.89078 ± 0.0026308 | 0.891266 ± 0.00154835 | 0.893348 ± 0.00165306 | 0.893265 ± 0.000845185 | 0.894628 ± 0.000865923 |
| sos_inverse_support_f1 | 0.693183 ± 0.00563724 | 0.700414 ± 0.00701508 | 0.700164 ± 0.00393965 | 0.701185 ± 0.00257984 | 0.70601 ± 0.00226182 | 0.705651 ± 0.00160697 |
| balanced_accuracy | 0.787719 ± 0.00462986 | 0.807764 ± 0.0056149 | 0.805512 ± 0.00336624 | 0.799997 ± 0.00294721 | 0.810085 ± 0.00291393 | 0.804255 ± 0.00182044 |
| mcc | 0.585898 ± 0.00732407 | 0.592374 ± 0.00956022 | 0.592208 ± 0.00540718 | 0.594693 ± 0.00395401 | 0.600138 ± 0.00305044 | 0.60044 ± 0.00232093 |
| threshold | 0.5 ± 0 | 0.5 ± 0 | 0.5 ± 0 | 0.5 ± 0 | 0.5 ± 0 | 0.5 ± 0 |
| fp | 123.2 ± 2.58844 | 156.4 ± 6.65582 | 151.6 ± 5.12835 | 136.4 ± 8.96103 | 150 ± 4.52769 | 137.2 ± 4.54973 |
| fn | 139.6 ± 3.57771 | 118.2 ± 4.43847 | 120.8 ± 2.94958 | 127.8 ± 3.70135 | 117.6 ± 2.88097 | 124.4 ± 2.07364 |
| tp | 242.4 ± 3.57771 | 263.8 ± 4.43847 | 261.2 ± 2.94958 | 254.2 ± 3.70135 | 264.4 ± 2.88097 | 257.6 ± 2.07364 |
| tn | 1960.8 ± 2.58844 | 1927.6 ± 6.65582 | 1932.4 ± 5.12835 | 1947.6 ± 8.96103 | 1934 ± 4.52769 | 1946.8 ± 4.54973 |
| cost | 71032 ± 1785.12 | 60664 ± 2203.05 | 61916 ± 1447.97 | 65264 ± 1767.49 | 60300 ± 1402.52 | 63572 ± 996.78 |
| realized_recall_test | 0.634555 ± 0.00936573 | 0.690576 ± 0.011619 | 0.68377 ± 0.0077214 | 0.665445 ± 0.0096894 | 0.692147 ± 0.00754181 | 0.674346 ± 0.00542839 |
| realized_fpr_test | 0.0591171 ± 0.00124205 | 0.075048 ± 0.00319377 | 0.0727447 ± 0.00246082 | 0.0654511 ± 0.00429992 | 0.071977 ± 0.0021726 | 0.0658349 ± 0.00218317 |
| recall_at_fpr_test | 0.596335 ± 0.0103726 | 0.604712 ± 0.0052356 | 0.607853 ± 0.00749624 | 0.609424 ± 0.0077214 | 0.61466 ± 0.00341319 | 0.610995 ± 0.0105689 |
| roc_attained_fpr_test | 0.0496161 ± 0.000429188 | 0.0496161 ± 0.000643782 | 0.0489443 ± 0.000897711 | 0.0494242 ± 0.000479846 | 0.0490403 ± 0.000923003 | 0.0497121 ± 0.000262823 |
| roc_fpr_limit | 0.05 ± 0 | 0.05 ± 0 | 0.05 ± 0 | 0.05 ± 0 | 0.05 ± 0 | 0.05 ± 0 |
| accuracy | 0.893431 ± 0.00165716 | 0.888646 ± 0.00283281 | 0.889538 ± 0.00175827 | 0.892863 ± 0.0023011 | 0.891484 ± 0.00105745 | 0.893917 ± 0.00116828 |
| precision | 0.663023 ± 0.00529534 | 0.627895 ± 0.00959909 | 0.632825 ± 0.00688886 | 0.651091 ± 0.01186 | 0.638093 ± 0.00510526 | 0.652558 ± 0.00602608 |
| ordinary_cross_entropy | 0.241562 ± 0.00113565 | 0.244166 ± 0.00157702 | 0.243416 ± 0.00145135 | 0.238946 ± 0.00185465 | 0.242974 ± 0.000722182 | 0.239134 ± 0.00109588 |
| balanced_cross_entropy | 0.469407 ± 0.00849512 | 0.432006 ± 0.00308015 | 0.435084 ± 0.009482 | 0.461716 ± 0.00616819 | 0.421839 ± 0.00592921 | 0.455356 ± 0.00435039 |

## All-field matched-seed pool-random+F1-minus-reference summaries
Deltas are raw pool-random+F1 minus reference. Positive is favorable for AP/F1/accuracy/recall/precision/MCC, negative for CE/cost/errors; threshold/count/dose directions are descriptive. No inferential significance or independent-fold claim.

### logistic_regression
| Metric | Original UGO | Original U2 | Original U5 | Retained U5+F1 | Pool-random |
| --- | --- | --- | --- | --- | --- |
| average_precision | -0.00678983 ± 0.00247882 | 0.0155211 ± 0.00123552 | 0.0155389 ± 0.00265254 | 0.00477334 ± 0.000638227 | 0.00828668 ± 0.00153282 |
| macro_average_precision | -0.00331234 ± 0.00132927 | 0.00828957 ± 0.000876172 | 0.00822666 ± 0.00147077 | 0.00270036 ± 0.000378166 | 0.00398903 ± 0.000862163 |
| minority_f1 | -0.0266616 ± 0.00657017 | -0.0157135 ± 0.00302848 | -0.0123611 ± 0.00524315 | 0.00143692 ± 0.0062776 | -0.0182414 ± 0.00563869 |
| minority_recall | -0.0376963 ± 0.013552 | -0.0549738 ± 0.00489746 | -0.0502618 ± 0.0110134 | -0.00366492 ± 0.00682639 | -0.0471204 ± 0.0080686 |
| macro_f1 | -0.013976 ± 0.00313849 | -0.00541457 ± 0.00173102 | -0.0036174 ± 0.00265859 | 0.00126154 ± 0.00344765 | -0.00773482 ± 0.00321492 |
| weighted_f1 | -0.00522056 ± 0.00089967 | 0.00169365 ± 0.000978556 | 0.00241735 ± 0.00122511 | 0.00114049 ± 0.00149863 | -0.000483305 ± 0.0017255 |
| sos_inverse_support_f1 | -0.0227314 ± 0.00550338 | -0.0125228 ± 0.00261744 | -0.00965216 ± 0.00442721 | 0.00138258 ± 0.00540064 | -0.0149863 ± 0.00487652 |
| balanced_accuracy | -0.0170727 ± 0.0052786 | -0.0184178 ± 0.00187935 | -0.0162057 ± 0.0040448 | -0.000488891 ± 0.00344737 | -0.0171303 ± 0.00328577 |
| mcc | -0.0216782 ± 0.0037564 | 0.00374251 ± 0.00422035 | 0.00696015 ± 0.0046391 | 0.0049864 ± 0.00610922 | -0.0034849 ± 0.00698908 |
| threshold | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| fp | -7.4 ± 6.65582 | -37.8 ± 4.14729 | -37.2 ± 7.36206 | -5.6 ± 0.894427 | -26.8 ± 6.41872 |
| fn | 14.4 ± 5.17687 | 21 ± 1.87083 | 19.2 ± 4.20714 | 1.4 ± 2.60768 | 18 ± 3.08221 |
| tp | -14.4 ± 5.17687 | -21 ± 1.87083 | -19.2 ± 4.20714 | -1.4 ± 2.60768 | -18 ± 3.08221 |
| tn | 7.4 ± 6.65582 | 37.8 ± 4.14729 | 37.2 ± 7.36206 | 5.6 ± 0.894427 | 26.8 ± 6.41872 |
| cost | 7126 ± 2525.16 | 10122 ± 906.488 | 9228 ± 2039.22 | 644 ± 1305.02 | 8732 ± 1501.37 |
| realized_recall_test | -0.0376963 ± 0.013552 | -0.0549738 ± 0.00489746 | -0.0502618 ± 0.0110134 | -0.00366492 ± 0.00682639 | -0.0471204 ± 0.0080686 |
| realized_fpr_test | -0.00355086 ± 0.00319377 | -0.0181382 ± 0.00199006 | -0.0178503 ± 0.00353266 | -0.00268714 ± 0.000429188 | -0.0128599 ± 0.00308 |
| recall_at_fpr_test | 0.00471204 ± 0.00468286 | 0.0413613 ± 0.0102061 | 0.0434555 ± 0.00602663 | 0.00837696 ± 0.0059695 | 0.026178 ± 0.00716914 |
| roc_attained_fpr_test | -0.000383877 ± 0.00114562 | 9.59693e-05 ± 0.00149447 | -9.59693e-05 ± 0.000858375 | -0.000575816 ± 0.00104028 | -0.000671785 ± 0.000995031 |
| roc_fpr_limit | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| accuracy | -0.00283861 ± 0.000951017 | 0.00681265 ± 0.00126297 | 0.00729927 ± 0.00169639 | 0.00170316 ± 0.00116122 | 0.00356853 ± 0.00205577 |
| precision | 0.00262 ± 0.0127475 | 0.0657011 ± 0.0103683 | 0.0666531 ± 0.0141407 | 0.013897 ± 0.00393854 | 0.0454687 ± 0.0147375 |
| ordinary_cross_entropy | -0.00823145 ± 0.00451362 | -0.0164918 ± 0.00226478 | -0.0163255 ± 0.00229178 | -0.0015508 ± 0.000365973 | -0.0154884 ± 0.00182326 |
| balanced_cross_entropy | 0.0114445 ± 0.00592919 | 0.0435224 ± 0.00670487 | 0.0439565 ± 0.00544623 | -0.000257265 ± 0.00274912 | 0.0512189 ± 0.00574283 |

### random_forest
| Metric | Original UGO | Original U2 | Original U5 | Retained U5+F1 | Pool-random |
| --- | --- | --- | --- | --- | --- |
| average_precision | -0.00190852 ± 0.0092866 | 0.00472358 ± 0.00350895 | 0.00491595 ± 0.0114517 | 0.00178132 ± 0.00941273 | 0.00298356 ± 0.00404686 |
| macro_average_precision | -0.000469117 ± 0.00523028 | 0.00228739 ± 0.00174556 | 0.00263219 ± 0.00634826 | 0.00117778 ± 0.00496703 | 0.00147419 ± 0.00220686 |
| minority_f1 | 0.00482217 ± 0.0134656 | -0.00224152 ± 0.00925217 | 0.00333326 ± 0.0158751 | -0.000297227 ± 0.0102509 | -0.00298604 ± 0.00983135 |
| minority_recall | 0.0308901 ± 0.0313262 | -0.0230366 ± 0.0164942 | -0.0157068 ± 0.0380258 | 0.0078534 ± 0.0362261 | -0.026178 ± 0.0124173 |
| macro_f1 | 0.00129719 ± 0.00686253 | -0.00011364 ± 0.00511808 | 0.00306673 ± 0.0081994 | -0.00067383 ± 0.00461776 | -0.000414134 ± 0.00590009 |
| weighted_f1 | -0.00113569 ± 0.00232747 | 0.001355 ± 0.0023069 | 0.00288278 ± 0.0030982 | -0.000933757 ± 0.00122112 | 0.00136096 ± 0.00326134 |
| sos_inverse_support_f1 | 0.00373008 ± 0.0114191 | -0.00158227 ± 0.00796916 | 0.00325069 ± 0.0134891 | -0.000413904 ± 0.00849532 | -0.00218923 ± 0.00860827 |
| balanced_accuracy | 0.0108385 ± 0.0133181 | -0.00777552 ± 0.00765577 | -0.00396665 ± 0.0160153 | 0.00229522 ± 0.0138794 | -0.00896233 ± 0.00619244 |
| mcc | 0.0012636 ± 0.0124256 | -0.000507986 ± 0.0102062 | 0.0055967 ± 0.0162839 | -0.00205379 ± 0.00786116 | -0.00126429 ± 0.0115033 |
| threshold | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| fp | 19.2 ± 9.95992 | -15.6 ± 4.27785 | -16.2 ± 14.0428 | 6.8 ± 18.0472 | -17.2 ± 8.04363 |
| fn | -11.8 ± 11.9666 | 8.8 ± 6.30079 | 6 ± 14.5258 | -3 ± 13.8384 | 10 ± 4.74342 |
| tp | 11.8 ± 11.9666 | -8.8 ± 6.30079 | -6 ± 14.5258 | 3 ± 13.8384 | -10 ± 4.74342 |
| tn | -19.2 ± 9.95992 | 15.6 ± 4.27785 | 16.2 ± 14.0428 | -6.8 ± 18.0472 | 17.2 ± 8.04363 |
| cost | -5708 ± 5885.31 | 4244 ± 3124.15 | 2838 ± 7135.63 | -1432 ± 6741.85 | 4828 ± 2359.87 |
| realized_recall_test | 0.0308901 ± 0.0313262 | -0.0230366 ± 0.0164942 | -0.0157068 ± 0.0380258 | 0.0078534 ± 0.0362261 | -0.026178 ± 0.0124173 |
| realized_fpr_test | 0.00921305 ± 0.00477923 | -0.0074856 ± 0.00205271 | -0.00777351 ± 0.00673838 | 0.00326296 ± 0.00865987 | -0.00825336 ± 0.00385971 |
| recall_at_fpr_test | -0.00157068 ± 0.013425 | -0.00837696 ± 0.0179849 | -0.00366492 ± 0.0111986 | -0.00575916 ± 0.0150836 | 0.0104712 ± 0.013089 |
| roc_attained_fpr_test | -9.59693e-05 ± 0.000525645 | 9.59693e-05 ± 0.000401468 | 0.000863724 ± 0.00193134 | 0.000383877 ± 0.00104028 | 0.000383877 ± 0.00104028 |
| roc_fpr_limit | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| accuracy | -0.00300081 ± 0.00113254 | 0.0027575 ± 0.00201537 | 0.00413625 ± 0.00248988 | -0.00154096 ± 0.00207567 | 0.00291971 ± 0.00350483 |
| precision | -0.0248982 ± 0.00850312 | 0.0178134 ± 0.00545682 | 0.0209577 ± 0.011878 | -0.010394 ± 0.0213171 | 0.0190607 ± 0.0124241 |
| ordinary_cross_entropy | -0.00180889 ± 0.0181241 | -0.00116975 ± 0.00376718 | -0.00275604 ± 0.0140759 | -0.00347534 ± 0.00755684 | -0.00396846 ± 0.0139418 |
| balanced_cross_entropy | -0.00887729 ± 0.0647554 | 0.0295898 ± 0.0155345 | 0.0231592 ± 0.0512075 | -0.0142986 ± 0.0313699 | 0.0268044 ± 0.0475619 |

### lightgbm
| Metric | Original UGO | Original U2 | Original U5 | Retained U5+F1 | Pool-random |
| --- | --- | --- | --- | --- | --- |
| average_precision | 0.000895624 ± 0.0043603 | 0.00299084 ± 0.00533403 | 0.00153029 ± 0.0039385 | -0.00279654 ± 0.00105461 | -0.00111928 ± 0.00345382 |
| macro_average_precision | 0.00057964 ± 0.00224697 | 0.00152548 ± 0.00273041 | 0.000791413 ± 0.0020268 | -0.00139416 ± 0.00051021 | -0.000670139 ± 0.00187725 |
| minority_f1 | 0.0147817 ± 0.00713113 | 0.00554889 ± 0.00828724 | 0.00596433 ± 0.00284627 | 0.0051814 ± 0.00333226 | -0.000744983 ± 0.00350954 |
| minority_recall | 0.0397906 ± 0.00677601 | -0.0162304 ± 0.00932907 | -0.00942408 ± 0.0095469 | 0.00890052 ± 0.00682639 | -0.017801 ± 0.00794018 |
| macro_f1 | 0.00731404 ± 0.00431295 | 0.00454224 ± 0.0051318 | 0.0044244 ± 0.00175187 | 0.00287279 ± 0.00242393 | 0.000502027 ± 0.00195012 |
| weighted_f1 | 0.00215994 ± 0.00239376 | 0.00384746 ± 0.00301079 | 0.00336156 ± 0.00121488 | 0.00127942 ± 0.00183997 | 0.0013627 ± 0.000949247 |
| sos_inverse_support_f1 | 0.0124681 ± 0.00625621 | 0.00523701 ± 0.00730539 | 0.00548724 ± 0.0024895 | 0.00446616 ± 0.00304615 | -0.000358643 ± 0.00302225 |
| balanced_accuracy | 0.0165364 ± 0.00381212 | -0.00350866 ± 0.00478259 | -0.00125715 ± 0.00338614 | 0.00425832 ± 0.00173466 | -0.00582951 ± 0.00337213 |
| mcc | 0.0145421 ± 0.00850391 | 0.00806613 ± 0.0101445 | 0.008232 ± 0.003438 | 0.0057469 ± 0.0046523 | 0.00030271 ± 0.0040109 |
| threshold | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| fp | 14 ± 5.33854 | -19.2 ± 8.07465 | -14.4 ± 6.94982 | 0.8 ± 8.31865 | -12.8 ± 3.63318 |
| fn | -15.2 ± 2.58844 | 6.2 ± 3.56371 | 3.6 ± 3.64692 | -3.4 ± 2.60768 | 6.8 ± 3.03315 |
| tp | 15.2 ± 2.58844 | -6.2 ± 3.56371 | -3.6 ± 3.64692 | 3.4 ± 2.60768 | -6.8 ± 3.03315 |
| tn | -14 ± 5.33854 | 19.2 ± 8.07465 | 14.4 ± 6.94982 | -0.8 ± 8.31865 | 12.8 ± 3.63318 |
| cost | -7460 ± 1303.97 | 2908 ± 1771.87 | 1656 ± 1762.19 | -1692 ± 1227.16 | 3272 ± 1489.74 |
| realized_recall_test | 0.0397906 ± 0.00677601 | -0.0162304 ± 0.00932907 | -0.00942408 ± 0.0095469 | 0.00890052 ± 0.00682639 | -0.017801 ± 0.00794018 |
| realized_fpr_test | 0.00671785 ± 0.00256168 | -0.00921305 ± 0.00387459 | -0.00690979 ± 0.00333485 | 0.000383877 ± 0.00399168 | -0.00614203 ± 0.00174337 |
| recall_at_fpr_test | 0.0146597 ± 0.0185661 | 0.00628272 ± 0.0120817 | 0.00314136 ± 0.00503544 | 0.00157068 ± 0.0126361 | -0.00366492 ± 0.0117946 |
| roc_attained_fpr_test | 9.59693e-05 ± 0.000401468 | 9.59693e-05 ± 0.000788468 | 0.000767754 ± 0.000995031 | 0.000287908 ± 0.000547109 | 0.000671785 ± 0.00105129 |
| roc_fpr_limit | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| accuracy | 0.000486618 ± 0.00255507 | 0.0052717 ± 0.00338064 | 0.00437956 ± 0.00165716 | 0.00105434 ± 0.00242632 | 0.00243309 ± 0.000993305 |
| precision | -0.0104653 ± 0.0098079 | 0.0246628 ± 0.0123534 | 0.0197327 ± 0.00836165 | 0.00146734 ± 0.0114877 | 0.0144654 ± 0.00421108 |
| ordinary_cross_entropy | -0.00242873 ± 0.00187854 | -0.00503264 ± 0.00179398 | -0.00428234 ± 0.00159035 | 0.000187602 ± 0.00100598 | -0.00384017 ± 0.00145513 |
| balanced_cross_entropy | -0.014051 ± 0.00540796 | 0.0233497 ± 0.00149171 | 0.0202719 ± 0.00916787 | -0.00636031 ± 0.00526875 | 0.033517 ± 0.003644 |

## Accepted dose
| Classifier | Method | Accepted mean ± SD |
| --- | --- | --- |
| logistic_regression | Original UGO | 1000 ± 0 |
| logistic_regression | Original U2 | 1000 ± 0 |
| logistic_regression | Original U5 | 1000 ± 0 |
| logistic_regression | Retained U5+F1 | 522.4 ± 63.9398 |
| logistic_regression | Pool-random | 1000 ± 0 |
| logistic_regression | Pool-random+F1 | 491.6 ± 52.0317 |
| random_forest | Original UGO | 1000 ± 0 |
| random_forest | Original U2 | 1000 ± 0 |
| random_forest | Original U5 | 1000 ± 0 |
| random_forest | Retained U5+F1 | 622.6 ± 112.249 |
| random_forest | Pool-random | 1000 ± 0 |
| random_forest | Pool-random+F1 | 646 ± 86.1974 |
| lightgbm | Original UGO | 1000 ± 0 |
| lightgbm | Original U2 | 1000 ± 0 |
| lightgbm | Original U5 | 1000 ± 0 |
| lightgbm | Retained U5+F1 | 689.2 ± 113.992 |
| lightgbm | Pool-random | 1000 ± 0 |
| lightgbm | Pool-random+F1 | 682.2 ± 58.3069 |

## Comparable new-trajectory runtime
Only NEW pool-random vs NEW pool-random+F1 timings are operationally comparable: pipeline plus final native fit, including checkpoint capture; grouping, preprocessing, metrics and final serialization are excluded. Historical original-reference runtimes include canonical tuning and are provenance metadata, not operational baselines. No runtime deltas against original references are computed.
CLI total includes all four inner tau searches, frozen Flow loading and artifact work; historical frozen Flow training is not repeated in full runs. Early-stop attempt shortfalls are allowed, never rescued/refilled; physical K-pool rows are counted separately from attempts and accepts.
| Classifier | Method | Seconds mean ± SD |
| --- | --- | --- |
| logistic_regression | Pool-random | 5.38758 ± 0.0986552 |
| logistic_regression | Pool-random+F1 | 4.37182 ± 0.0994459 |
| random_forest | Pool-random | 32.0821 ± 10.3819 |
| random_forest | Pool-random+F1 | 30.1226 ± 9.89645 |
| lightgbm | Pool-random | 6.12001 ± 0.048353 |
| lightgbm | Pool-random+F1 | 5.05347 ± 0.0470592 |

### Per-seed dose/runtime diagnostics
| algorithm | seed | variant_id | accepted | requested | runtime_seconds | runtime_scope | attempted | rejected | physical_proposals | rounds | native_fit_count | native_fit_seconds | stop_reason | nominal_attempt_cap | attempt_underspend | flow_inference_rows | flow_inference_seconds | selected_tau |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0 | U0 | 1000 | 1000 | 20.34672784399936 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 0 | U2 | 1000 | 1000 | 31.86432363899985 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 0 | U5 | 1000 | 1000 | 32.085974282999814 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 0 | U5_F1 | 580 | 1000 | 19.26673584799937 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 0 | outer_iterative_pool_random | 1000 | 1000 | 5.328332126002351 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 3.0 | 5.0 | 0.14261343499674695 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| logistic_regression | 0 | outer_iterative_pool_random_f1 | 528 | 1000 | 4.307012992001546 | new_pipeline_plus_final_fit | 1000.0 | 472.0 | 5000.0 | 3.0 | 5.0 | 0.12471036499846377 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.09565166999891517 | 0.8 |
| random_forest | 0 | U0 | 1000 | 1000 | 85.15059242799998 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 0 | U2 | 1000 | 1000 | 136.16003472099965 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 0 | U5 | 1000 | 1000 | 158.08941801999936 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 0 | U5_F1 | 738 | 1000 | 55.30777363299967 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 0 | outer_iterative_pool_random | 1000 | 1000 | 39.56055350099996 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 4.0 | 6.0 | 33.25855866900383 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| random_forest | 0 | outer_iterative_pool_random_f1 | 703 | 1000 | 37.529446019001625 | new_pipeline_plus_final_fit | 1000.0 | 297.0 | 5000.0 | 4.0 | 6.0 | 32.34887726300076 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.10391672500554705 | 0.7 |
| lightgbm | 0 | U0 | 1000 | 1000 | 27.737478686998656 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 0 | U2 | 1000 | 1000 | 41.413857682000526 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 0 | U5 | 1000 | 1000 | 45.43153784199967 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 0 | U5_F1 | 781 | 1000 | 9.42457566199846 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 0 | outer_iterative_pool_random | 1000 | 1000 | 6.129952063001838 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 4.0 | 6.0 | 0.6338080449968402 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| lightgbm | 0 | outer_iterative_pool_random_f1 | 747 | 1000 | 5.073564318998251 | new_pipeline_plus_final_fit | 1000.0 | 253.0 | 5000.0 | 4.0 | 6.0 | 0.6046188040018023 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.1064408789970912 | 0.5 |
| logistic_regression | 1 | U0 | 1000 | 1000 | 20.007226947000163 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 1 | U2 | 1000 | 1000 | 31.515088444000867 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 1 | U5 | 1000 | 1000 | 31.83452282899998 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 1 | U5_F1 | 541 | 1000 | 18.62875135200011 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 1 | outer_iterative_pool_random | 1000 | 1000 | 5.32220884400158 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 3.0 | 5.0 | 0.14583499299988034 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| logistic_regression | 1 | outer_iterative_pool_random_f1 | 523 | 1000 | 4.330273891999241 | new_pipeline_plus_final_fit | 1000.0 | 477.0 | 5000.0 | 3.0 | 5.0 | 0.12031922600363032 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.0975009260000661 | 0.8 |
| random_forest | 1 | U0 | 1000 | 1000 | 88.41394149400003 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 1 | U2 | 1000 | 1000 | 134.01128820800113 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 1 | U5 | 1000 | 1000 | 179.353103114001 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 1 | U5_F1 | 725 | 1000 | 115.84658539699922 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 1 | outer_iterative_pool_random | 1000 | 1000 | 39.83280182100134 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 4.0 | 6.0 | 33.624570647007204 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| random_forest | 1 | outer_iterative_pool_random_f1 | 596 | 1000 | 37.202481748001446 | new_pipeline_plus_final_fit | 1000.0 | 404.0 | 5000.0 | 4.0 | 6.0 | 32.06150097700083 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.1049084589976701 | 0.8 |
| lightgbm | 1 | U0 | 1000 | 1000 | 28.66214394400049 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 1 | U2 | 1000 | 1000 | 42.44615603300008 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 1 | U5 | 1000 | 1000 | 51.95166264399995 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 1 | U5_F1 | 723 | 1000 | 26.080548664000162 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 1 | outer_iterative_pool_random | 1000 | 1000 | 6.200293062000128 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 4.0 | 6.0 | 0.6243759879980644 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| lightgbm | 1 | outer_iterative_pool_random_f1 | 632 | 1000 | 4.98778990700157 | new_pipeline_plus_final_fit | 1000.0 | 368.0 | 5000.0 | 4.0 | 6.0 | 0.595329928008141 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.1053051030030474 | 0.7 |
| logistic_regression | 2 | U0 | 1000 | 1000 | 20.36641501000122 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 2 | U2 | 1000 | 1000 | 33.054121089000546 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 2 | U5 | 1000 | 1000 | 32.70195313400109 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 2 | U5_F1 | 470 | 1000 | 18.83945692999987 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 2 | outer_iterative_pool_random | 1000 | 1000 | 5.302332875002321 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 3.0 | 5.0 | 0.14280832100121188 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| logistic_regression | 2 | outer_iterative_pool_random_f1 | 430 | 1000 | 4.296709701000509 | new_pipeline_plus_final_fit | 1000.0 | 570.0 | 5000.0 | 3.0 | 5.0 | 0.12898690900328802 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.09671784499732894 | 0.8 |
| random_forest | 2 | U0 | 1000 | 1000 | 86.15782632199989 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 2 | U2 | 1000 | 1000 | 134.04604030999872 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 2 | U5 | 1000 | 1000 | 177.6163996890009 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 2 | U5_F1 | 518 | 1000 | 111.73225400800038 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 2 | outer_iterative_pool_random | 1000 | 1000 | 39.5969371650026 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 4.0 | 6.0 | 33.426677856001334 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| random_forest | 2 | outer_iterative_pool_random_f1 | 529 | 1000 | 37.31581471499885 | new_pipeline_plus_final_fit | 1000.0 | 471.0 | 5000.0 | 4.0 | 6.0 | 32.104036901000654 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.10458692399333813 | 0.8 |
| lightgbm | 2 | U0 | 1000 | 1000 | 28.439590570998917 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 2 | U2 | 1000 | 1000 | 42.424783444999775 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 2 | U5 | 1000 | 1000 | 47.30689628999971 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 2 | U5_F1 | 679 | 1000 | 9.542505102001087 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 2 | outer_iterative_pool_random | 1000 | 1000 | 6.096694783002022 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 4.0 | 6.0 | 0.6215661069982161 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| lightgbm | 2 | outer_iterative_pool_random_f1 | 709 | 1000 | 5.114188657000341 | new_pipeline_plus_final_fit | 1000.0 | 291.0 | 5000.0 | 4.0 | 6.0 | 0.5957053580023057 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.10653164100222057 | 0.5 |
| logistic_regression | 3 | U0 | 1000 | 1000 | 21.042484779000006 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 3 | U2 | 1000 | 1000 | 32.498578783000994 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 3 | U5 | 1000 | 1000 | 33.02436153799863 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 3 | U5_F1 | 441 | 1000 | 20.036295265001172 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 3 | outer_iterative_pool_random | 1000 | 1000 | 5.522972511997068 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 3.0 | 5.0 | 0.2632012589965598 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| logistic_regression | 3 | outer_iterative_pool_random_f1 | 440 | 1000 | 4.538465944999189 | new_pipeline_plus_final_fit | 1000.0 | 560.0 | 5000.0 | 3.0 | 5.0 | 0.24559403100647614 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.09576928399837925 | 0.8 |
| random_forest | 3 | U0 | 1000 | 1000 | 85.9113397910005 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 3 | U2 | 1000 | 1000 | 124.2200787310012 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 3 | U5 | 1000 | 1000 | 161.99650113299867 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 3 | U5_F1 | 498 | 1000 | 56.84402129500086 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 3 | outer_iterative_pool_random | 1000 | 1000 | 20.632120054000552 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 4.0 | 6.0 | 14.876420215001417 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| random_forest | 3 | outer_iterative_pool_random_f1 | 655 | 1000 | 19.19173702400076 | new_pipeline_plus_final_fit | 1000.0 | 345.0 | 5000.0 | 4.0 | 6.0 | 14.553763783998875 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.1066736389984726 | 0.5 |
| lightgbm | 3 | U0 | 1000 | 1000 | 28.694118766999964 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 3 | U2 | 1000 | 1000 | 43.00038831200072 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 3 | U5 | 1000 | 1000 | 48.16116712799885 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 3 | U5_F1 | 498 | 1000 | 9.602206262999971 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 3 | outer_iterative_pool_random | 1000 | 1000 | 6.090086344000156 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 4.0 | 6.0 | 0.6268461819963704 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| lightgbm | 3 | outer_iterative_pool_random_f1 | 610 | 1000 | 5.058943799998815 | new_pipeline_plus_final_fit | 1000.0 | 390.0 | 5000.0 | 4.0 | 6.0 | 0.6022549029985385 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.10599264600023162 | 0.6 |
| logistic_regression | 4 | U0 | 1000 | 1000 | 20.683069262999197 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 4 | U2 | 1000 | 1000 | 32.24294637099956 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 4 | U5 | 1000 | 1000 | 32.787832784000784 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 4 | U5_F1 | 580 | 1000 | 19.074162728999 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| logistic_regression | 4 | outer_iterative_pool_random | 1000 | 1000 | 5.462075114999607 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 3.0 | 5.0 | 0.243322497994086 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| logistic_regression | 4 | outer_iterative_pool_random_f1 | 537 | 1000 | 4.386636552000709 | new_pipeline_plus_final_fit | 1000.0 | 463.0 | 5000.0 | 3.0 | 5.0 | 0.23414856199815404 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.0983055049982795 | 0.8 |
| random_forest | 4 | U0 | 1000 | 1000 | 97.92418663900025 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 4 | U2 | 1000 | 1000 | 123.67057250900143 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 4 | U5 | 1000 | 1000 | 154.91341940799975 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 4 | U5_F1 | 634 | 1000 | 35.29595975900156 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| random_forest | 4 | outer_iterative_pool_random | 1000 | 1000 | 20.78798917599852 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 4.0 | 6.0 | 15.038729391999368 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| random_forest | 4 | outer_iterative_pool_random_f1 | 747 | 1000 | 19.373335579999548 | new_pipeline_plus_final_fit | 1000.0 | 253.0 | 5000.0 | 4.0 | 6.0 | 14.636654813002679 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.10678933399685775 | 0.5 |
| lightgbm | 4 | U0 | 1000 | 1000 | 28.18321289800042 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 4 | U2 | 1000 | 1000 | 42.75701280300018 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 4 | U5 | 1000 | 1000 | 47.634958742999515 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 4 | U5_F1 | 765 | 1000 | 9.559623506000207 | historical_canonical_tuning_not_comparable | — | — | — | — | — | — | — | — | — | — | — | — |
| lightgbm | 4 | outer_iterative_pool_random | 1000 | 1000 | 6.083012620001682 | new_pipeline_plus_final_fit | 1000.0 | 0.0 | 5000.0 | 4.0 | 6.0 | 0.6179436019992863 | target_counts_reached | 1000.0 | 0.0 | 0.0 | 0.0 | — |
| lightgbm | 4 | outer_iterative_pool_random_f1 | 713 | 1000 | 5.032881353999983 | new_pipeline_plus_final_fit | 1000.0 | 287.0 | 5000.0 | 4.0 | 6.0 | 0.6067324490031751 | proposal_cap_reached | 1000.0 | 0.0 | 1000.0 | 0.10914769799637725 | 0.6 |

## Coverage and rejection diagnostics
Complete per-round/per-group planned-anchor, physical-proposal, eligible-anchor, attempted, accepted, rejection-reason and real-match diagnostics are in `group_count_coverage_diagnostics.csv`. Shortfalls and accepted doses are not repaired after F1 rejection.
| Classifier | Branch | Attempted | Accepted | Rejected | Physical proposals |
| --- | --- | --- | --- | --- | --- |
| lightgbm | outer_iterative_pool_random | 5000 | 5000 | 0 | 25000 |
| lightgbm | outer_iterative_pool_random_f1 | 5000 | 3411 | 1589 | 25000 |
| logistic_regression | outer_iterative_pool_random | 5000 | 5000 | 0 | 25000 |
| logistic_regression | outer_iterative_pool_random_f1 | 5000 | 2458 | 2542 | 25000 |
| random_forest | outer_iterative_pool_random | 5000 | 5000 | 0 | 25000 |
| random_forest | outer_iterative_pool_random_f1 | 5000 | 3230 | 1770 | 25000 |

## Supplemental full-field evidence
- `full_field_metric_summary.csv`: all 24 metrics × six final methods × classifiers.
- `full_field_paired_seed_deltas.csv`: every seed/metric paired F1-minus-each-reference and accepted-dose delta; runtime deltas only against new pool-random.
- `full_field_paired_summary.csv`: mean, sample SD, pair count and sign counts for every paired metric.
- `count_runtime_diagnostics.csv`: per-seed doses, attempts, rejections, tau, native fits and observed runtime.
- `group_count_coverage_diagnostics.csv`: full per-round/per-group coverage and rejection diagnostics.
- `new_trajectory_resource_paired.csv` / `new_trajectory_resource_summary.csv`: comparable new-random versus new-F1 observed resources, doses and physical proposals.

## Verification limitations
- Runtimes are checked for nonnegative accounting, not independently reproduced.
- Frozen Flow training is established from exact original checkpoint and audit provenance, not retrained.
- Tree learned-state arrays/dumps and predictions, not unstable post-load pickle hashes, establish prefix fit equivalence.
- Five matched seeds share one fixed outer holdout; these are exploratory seeds, not independent folds.

## Evidence and reproduction

Scientific bundle: `results/runs/20261009T170013-d5be50bc2107` — 120 completed records (60 explicit imported original references, 60 fresh inner/outer selected trajectories), plus all four inner F1 threshold trajectories. The imported source `results/runs/20261009T153348-af844bdecf5a` and its dependencies are required; they are intentionally preserved. All 15 freshly run outer pool-random prediction tables exactly reproduce the previously audited controls, without utility gradients or tree LR-surrogate fits.

The independent audit passed 770,287 checks with zero failures, including all 60 original reference predictions, raw-data preprocessing and ancestry, initial raw-pool identity, exact quota/anchor plans, pool dedup/random masks, attempted-only Flow decisions, native prefix refits, all 24 metric fields and 1,800 paired seed/metric comparisons. It binds consumed artifacts by hashes. The actual three-classifier development-only smoke completed in 21.59 seconds and passed 45,993 independent checks. The full regression suite passed 262 tests/307 subtests (79.47 seconds); the 11 new behavioral regressions passed again after removing an incidental exception-wording assertion. Existing sklearn/LightGBM warnings were not suppressed.

Observed scientific invocation: approximately 1,207.18 seconds (20.12 minutes); the independent verification/report invocation added approximately 947.22 seconds (15.79 minutes). These are invocation wall durations; validation-threshold tuning and artifact work are included in the former, audit refits in the latter. Frozen Flow training and historical canonical reference tuning were not repeated. Per-trajectory timings above have the separately stated, narrower scope.

`configs/ablations/stage1_pool_random_f1.yaml` declares the pending experiment; `benchmarks/pool_random_f1.py` is its runnable entrypoint. Reproduction:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.pool_random_f1 --run --seeds 0 1 2 3 4
conda run --no-capture-output -n ml_project python results/runs/20261009T170013-d5be50bc2107/verification_recipe.py results/runs/20261009T170013-d5be50bc2107
conda run --no-capture-output -n ml_project python results/runs/20261009T170013-d5be50bc2107/report_recipe.py results/runs/20261009T170013-d5be50bc2107
```

The two artifact recipes preserve the actual producing scripts. The reporting recipe regenerates the numeric tables and CSVs in its report draft; this document adds interpretation/provenance, not new metric calculations. `all_metrics.csv`, `metric_summary.csv`, `paired_metric_deltas.csv` and `paired_metric_summary.csv` additionally retain all inner/outer row-level metrics and paired comparisons.
