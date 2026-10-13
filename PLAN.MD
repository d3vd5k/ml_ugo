# UGO execution handoff

## Objective and order

1. Improve the existing UGO–CLIMB generation pipeline.
2. Replace interpolation with learned generators; then incorporate unlabeled data.
3. Transfer the validated pipeline to image features; optionally generate images.

Execute in order. Retain only justified components. Do not start contrastive-loss development, text experiments, or backbone fine-tuning during Stages 1–2.

## Start here

- Locate the existing repository, UGO implementation, CLIMB configuration, results and repository instructions. Ask for the repository path only if unavailable.
- Preserve the baseline. Record commit/configuration, dependencies, data versions, splits and seeds.
- Resolve UGO's augmented-data retraining and generation-cap ambiguities against the implementation and paper. Document decisions.
- Implement Stage 1 first. Complete each experiment before adding the next component.
- Maintain `experiment_registry.csv`: variant, dataset, split, seed, configuration, metrics, runtime, status.
- Use [DATASETS.MD](DATASETS.MD) for benchmark sources and paper-reported targets. Acquire/audit each dataset when its stage below requires it; do not begin learned-generator or image experiments early. Published numbers are reproduction targets, not locally verified results or current-SOTA claims.
- Preserve a separately identified original-UGO reference for U0. The current corrected real-only implementation is not automatically a reproduction of the original algorithm; do not relabel historical, globally preprocessed CLIMB results as leakage-free evidence.

## Experimental contract

- Let \(D_L=\{(x_i,y_i)\}_{i=1}^{n}\), \(D_U\) be unlabeled training data, and \(S_t\) synthetic data at round \(t\).
- Split real data first. Fit preprocessing, clustering, calibration and generation within training partitions only.
- Use inner validation for tuning/stopping/thresholds; reserve outer folds for evaluation.
- Use identical outer splits across methods. Preserve natural test prevalence. Respect known entity/time boundaries.
- Out-of-fold scoring must exclude the scored point and every synthetic descendant derived from it.
- Maintain real, pseudo-labeled and synthetic provenance separately. Initially generate only from real seeds.
- Judge support using real reference observations. Synthetic neighbors cannot establish their own reliability.
- Choose development datasets by size, imbalance, feature types and group structure—not favorable results. Freeze design before confirmatory evaluation.
- Match generated counts for mechanism comparisons; additionally report each method's native budget and total compute.
- Treat all formulas below, except original MMA, as candidate designs requiring ablation.
- Maintain two protocol IDs/tracks: **published reproduction**, matching source/version, class mapping, splits, preprocessing, classifier, tuning, seeds, generation budget, metric and aggregation; and **controlled comparison**, following the training-isolation contract above with shared splits and budgets. Document author-protocol deviations rather than copying leakage into the controlled track.
- If an author protocol is unrecoverable, mark that published comparison `unresolved`; continue the controlled track under a distinct protocol ID. Never compare a changed split/classifier score directly with a paper number. Identify CLIMB overlap by source IDs, versions and row fingerprints; overlapping datasets are not independent evidence.

## Stage 1 — UGO pipeline

### Benchmark progression — pipeline development and validation

Introduce these benchmarks in order while implementing Steps 1–6; keep learned generators out of this stage. Before running each, add its source/version/license/checksum, target mapping, feature types/missingness, class counts, entity/time fields and CLIMB-overlap status to `benchmarks/manifest.yaml`, and define separate reproduction/controlled configurations in `configs/benchmarks/`. Record paper targets in `results/published_targets.csv` with metric/unit, paper/table/URL, reported mean/spread, spread interpretation and reproduction status; mark unverified fields explicitly.

1. **Online Shoppers Purchasing Intention — first development benchmark.** Recover SOS author-processed data, encodings, splits and exact aggregation. Raw UCI: Revenue, 12,330 sessions; 1,908 purchases / 10,422 non-purchases. Preserve imbalanced test prevalence and valid categories; report minority AP, F1/recall and balanced accuracy. SOS Table 2 targets: no oversampling 0.601 ± 0.039; SMOTE 0.648 ± 0.004. The audited author code calls inverse-support F1 weighted_f1 and repeats evaluation five times with training RNG fixed to 2021; do not substitute sklearn support-weighted F1 or assume five independent seeds. Published metric/seed correspondence remains unresolved. Audit CLIMB OpenML45560 overlap, never double-count evidence. SOS generation enters Step9 only.
2. **APS Failure at Scania Trucks — severe-imbalance/cost benchmark.** Acquire the official 60,000-row training set (1,000 positive / 59,000 negative) and untouched 16,000-row test set. Verify the schema/predictor count, label and `na` handling. Split validation only within official training; fit imputation/scaling/selection/resampling within that split. Establish cost-sensitive/class-weighted boosting and original/corrected UGO; choose calibration, thresholds and checkpoints on inner validation cost `C = 10 FP + 500 FN`. Report cost, FP, FN, AP, threshold, seeds and runtime on the original test-set scale. Paper targets: challenge winner cost 9,920 (542 FP, 9 FN); 2026 cost-sensitive transformer mean cost 3,440 across five initializations (mean FP 4, FN 6.8). Audit/reproduce that recipe when feasible and replace only its resampling component with UGO, holding Bayesian-ridge imputation, transformer/focal-loss and other settings fixed. Neither number is an established local baseline.
3. **Bank Account Fraud (BAF) Base — add after the pipeline is stable.** Use the official Base variant only initially (approximately one million rows, 30 features, 1.1% fraud; synthetic public banking data). Recover month/target/schema and the official LightGBM baseline with 100 random-search configurations. Preserve months 1–6 for training and the last two for testing; use time-respecting inner validation within training, not the generic random-fold CLIMB runner. Match tuning opportunities in controlled comparisons. Report recall at 5% FPR, distinguishing test-ROC recall at that operating point from the realized test recall/FPR of an inner-selected threshold. No verified scalar target is available. Smoke-test on training-only subsets, then use full official partitions for claims; select bounded generation budgets on validation rather than fully balancing by default. Record generated counts, wall time and peak memory.
4. **Optional Satimage — after the binary benchmarks are stable.** Use the six-class UCI Landsat task with official partitions, not CLIMB binary Satellite or an image-generation benchmark. SOS targets SMOTE 0.846 ± 0.005 and SOS-VP 0.857 ± 0.006 require audited inverse-support F1 and resolved paper/code correspondence. Available multiclass code uses only decision tree/MLP, not the four-classifier binary recipe. SOS-VP enters Step9 only.

**Broad evaluation:** current user-directed Phase 1 uses Shoppers only, with training-isolated checks and known-boundary diagnostics. Do not run CLIMB until U1–U5 decisions and the final configuration are explicitly approved by the user. Declare development/confirmatory roles and freeze choices before subsequent CLIMB or held-out named-benchmark evaluation. Report failures and regressions. Published-track classifiers/metrics do not replace the controlled-track comparisons.

**Stage 1 benchmark gate:** reproduce the named simple baselines and resolve their leakage/category/metric boundaries before claiming published-score comparability. Missing reference details block that comparison only. Complete the ablations and freeze gate in Step 6 before starting the Stage 2 datasets.

**Implemented setup (2026-10-03):** checksum-pinned Shoppers/APS/Satimage acquisition and BAF Base local loader; four controlled and four explicitly unresolved published configurations; shared classifiers and baseline controls; paper-derived original UGOSMOTE with documented ambiguities; protocol-bound ablation declarations and linked result indexes. Shoppers is the quick-check default. Use benchmarks.run --prepare for no-fit setup or --smoke for training-only checks; --run is required for a study. Baseline setup does not complete Steps2–6, reproduce published scores, or open the freeze/Stage2 gate.

**Storage contract:** benchmark protocols own execution classifiers/seeds/budgets; study declarations own variant availability and pairing/freeze policy. Record CLI overrides and immutable snapshots, dataset/split fingerprints, selected configurations, actual counts, predictions, failures and resources. Separate setup, development, confirmatory and reproduction records; unavailable U1–U5/W5/R5 cannot execute. See README.md and reports/reproduction.md.

**Manual component review:** the user decides keep/discard for each U1–U5 change after receiving every metric. No automatic retention, combination or progression. corrected_current is the fixed controlled ablation base; original UGO/U0 remains distinct. The first runnable U1 candidate is U1_oof_mma: three-fold real-only scoring with fold-fitted preprocessing and fixed pre-generation MMA scores. This excludes all synthetic descendants from scoring-model training. Committee disagreement and calibration are separate later scoring experiments, not silently bundled into this candidate. New component-study references are rerun under the same policy.

**U1 retention decision:** after reviewing run `20261004T063713-7b42103e131e`, the user explicitly kept U1_oof_mma. LR AP improved slightly; RF minority F1/recall improved, with the reported RF AP and LR threshold-metric regressions preserved. Retention does not select committee/calibration alternatives, combine components or open confirmatory evaluation.

**U2 retention decision:** after reviewing run `20261004T084817-f93a994968f7`, the user explicitly kept the declared U2 KMeans/stability candidate. Its mixed predictive results and size/uniform/difficulty/graph comparisons remain reported; retention does not automatically substitute the best outer-test alternative. Real-only groups, subsampling stability and capped quotas remain separate from U1 scoring. Alpha=0,0.5,1 is cycled across ten joint proposals, selected inside training and replayed for outer refit. The user subsequently discarded U3/U4/U5 and reaffirmed canonical U2 as the preferred candidate. U1 remains kept; no combined U5 pipeline is retained and confirmatory evaluation remains closed.

**U3 retention decision — discard:** after reviewing run `20261005T184048-c25b2afe26ae`, the user explicitly discarded U3, including shortening, relocation and rejection. All 133 tests and 120 experiment cells passed; default shortening worsened every predictive mean across all three families, and all 45 U3 outer budgets underspent. Every metric/resource/repair/selection and count caveat remains in `reports/reproduction.md`. U3 implementation and immutable experiment evidence are preserved for explicit reproduction, not retained in the selected component set. U1/U2 stay kept; U4 was separately authorized afterward and is documented below.

**U4 implementation/review:** the user authorized stopping-only implementation and evaluation after discarding U3. U4 compares Welch-stopped corrected_current/original UGO with independent fixed budgets and one checkpoint trajectory at fractions 0/.25/.5/1 of the real-label maximum budget. Zero augmentation competes on real inner AP; checkpoint batches enforce cadence and monotonically cumulative multiclass class caps. The same ten learner proposals each receive four checkpoint evaluations, so selection cost and actual counts are reported rather than claiming equal compute/counts. Ties prefer lower actual/requested counts then earlier proposals. Selected fractions/cadence/epochs are replayed on outer training without test feedback. All 149 tests and the 21-cell Shoppers CLI smoke passed. Run `20261006T034029-0513bf17dc6c` completed 105 cells, with 1,200 checkpoint metrics and every prediction audited; all 75 reference cells reproduce. Both U4 policies select zero in all 20 tree refits and exactly recover baseline predictions/metrics at added search cost. LR default/grid use mean 600/850 rows; AP gains versus corrected_current are small and minority F1/recall decrease for every family. Every metric/resource/selection/count caveat is published in `reports/reproduction.md`. The user subsequently discarded both U4 policies. U1/U2 remain kept and U3/U4 excluded; U5 was separately tested and subsequently discarded below; canonical U2 is reaffirmed, with confirmatory progression closed.

**U5 testing authorization, not inclusion:** the user requested testing the combination after excluding U4 and reserved the final inclusion decision. U5 composes kept U1_oof_mma (fixed three-fold real-only OOF scoring) and canonical U2 KMeans/stability/capped quotas only, preserving corrected interpolation/relocation/Welch stopping and excluding U3/U4. Ten shared learner proposals cycle alpha=0/0.5/1 and select/replay the joint configuration on real inner validation. All 159 tests and an independently verified 18-cell training-only Shoppers smoke passed (360 metric values, 72 pipeline audits, 252 classifier fits). The comparison against baseline/U0/corrected_current/U1/U2 remains a development study; The user subsequently discarded U5 after reviewing the complete comparison and reaffirmed canonical U2; U1 remains kept and W5/R5/freeze/confirmatory gates remain closed.

**U5 retention decision — discard:** run `20261006T060142-8e042c6ffcf0` completed 90/90 Shoppers cells (six variants/three families/five seeds); all 75 references reproduce. All 90 predictions, 1,800 numeric metrics and 990 pipelines were independently checked, including 330 OOF/330 group/165 combined audits and exact U1 scoring/U2 group reuse. AP means are 0.5854/0.7176/0.7296 (LR/RF/LGB): little AP change versus U2, but slightly lower minority F1/recall in all families, higher diagnostic cost and 385 versus 220 classifier fits per family. All generation budgets reach 1,000 without padding; equal counts do not mean equal compute. All metrics/SDs/pairings/selections/resources are published in `reports/reproduction.md`. After reviewing all metrics, the user explicitly discarded U5 and reaffirmed canonical U2 as the preferred candidate. U1 remains kept; U3/U4/U5 are excluded. Historical U5 code/results are preserved for explicit reproduction, not retained or combined automatically; configuration freeze/confirmatory remain closed.

**H1 retention decision — discard; F1 testing authorized:** after reviewing the Shoppers H1 run `20261007T153639-17e4169cca5e` and saved-pool controls `20261007T151450-ab30c8827f99`, the user explicitly discarded H1. Its implementation and immutable evidence remain for historical reproduction only. The user next authorized F1 compatibility-only FlowGMM rejection on canonical U2 and Shoppers, with all metrics and original/corrected UGO/U2/U5 comparisons. F1 replaces post-generation neighbor detection and relocation; separate fit/calibration/validation data and frozen real-only flow fits are required. F2/H2 are not included. Final F1 inclusion remains the user's decision; this does not freeze Stage 1 or open CLIMB/confirmatory evaluation.

**F1 implementation/evidence — awaiting user retention decision:** real-only pinned shared RealNVP/diagonal-Gaussian FlowGMM compatibility rejection and the N-reject control are implemented, with original-coordinate scoring, frozen fit/calibration/validation roles, independent domain checks, no quota refill and explicit all-rejected termination. The Shoppers-only saved-pool run `20261007T172857-728d008031d6` completed 60 cells before iterative run `20261007T180310-edddfeea01ac` completed 90 cells. All 196 tests passed; independent checks recomputed 3,150 fresh numeric metric values and 315 historical U5 values and replayed 165,000 candidate decisions. All 60 shared reference predictions reproduce, and post-retraining proposal divergence is measured explicitly. Every metric, paired comparison with original/corrected UGO/U2/U5/N-reject, group support/recall and resource scope is in `reports/f1.md`. Results are mixed across families and operating points; no F1 keep/discard decision, F2/H2 implementation, Stage 1 freeze or broad evaluation is applied.

**F2 testing authorization, not inclusion:** the user authorized F2 likelihood-plus-compatibility testing after the F1 comparison and will decide about both F1 and F2 afterward. Reuse the exact frozen real-only F1 flow and private preprocessing, lock F1-selected tau, derive class-specific Jacobian-inclusive likelihood quantiles from the reserved real calibration rows, and select q on real validation. Shoppers-only saved-pool comparisons include explicit support-off and five repeated per-original-group count-matched random thinnings of F1 outputs before paired iterative evaluation. H1 remains discarded; neither F1 nor F2 is retained automatically, and no Stage 1 freeze or broader evaluation is authorized.

**F2 implementation/evidence — both retention decisions pending:** the frozen F1 detector now supports Jacobian-inclusive class-specific calibration quantiles and compatibility-plus-likelihood rejection. Explicit off and q=0 are distinct; all calibration rows remain counted, with fit-vocabulary unknowns audited as zero-support negative infinity and numerical failures rejected explicitly. Shoppers-only immutable-pool run `20261008T115909-d5b24e6dcbf9` completed 150 cells before iterative run `20261008T120348-bd801301d7cb` completed 75. All 219 tests passed; actual smoke CLIs completed 45 cells. Independent full checks recomputed 4,725 numeric metrics, replayed 570,000 candidate decisions, checked 225 checkpoint hashes and 1,200 group-matched thinning counts. Support-off reproduces F1 exactly; no new flow fits, relocation or quota refill occurs. All metrics/SDs/paired deltas, five-repeat per-original-group F1 thinning controls, calibration counts, smallest-group recall and compute/acceptance diagnostics are in `reports/f2.md`. F2 shows small mean AP/F1/MCC gains over F1 but lower recall/higher diagnostic cost in every family and no consistent advantage over matched thinning. Neither F1 nor F2 is automatically retained; the user decides both, H1 stays discarded, and Stage 1 remains unfrozen with broader evaluation closed.

**F1/F2 retention decision — keep F1 only, discard F2:** after reviewing the complete Shoppers F2 pool/iterative comparisons and group-count-matched thinning controls in `reports/f2.md`, the user explicitly kept compatibility-only F1 and discarded the F2 likelihood cutoff. This supersedes pending-review language in the experiment chronology above. F1 retains its canonical U2 base, frozen real-only flow, original-coordinate rejection and disabled F2 support filter. F2 code and immutable evidence remain for historical reproduction only. Earlier U1/U2 keep decisions and U3/U4/U5/H1 discard decisions are unchanged; this does not authorize a new composition, freeze Stage 1 or open broader evaluation.

Retention-state verification: both current study policies load with F1 kept/F2 discarded and confirmatory evaluation blocked. Retained-F1 Shoppers smoke `20261008T124811-b2899545ccb8` completed its cell; historical pool metadata smoke `20261008T124857-33bc3c084d9f` completed ten cells and persisted the same decisions without freezing Stage 1. All four affected Shoppers F2 source-loading regressions passed. No historical run manifests, predictions or metrics were rewritten.

**U5+F1 testing authorization, not retention:** the user next requested testing the existing U5 composition (fixed real-only three-fold U1 OOF MMA plus canonical U2 KMeans/stability allocation) with compatibility-only F1 replacing post-generation neighbor detection and relocation. Shoppers-only LR/RF/LightGBM comparisons use five paired seeds and report every metric against freshly matched U5, F1, U2, no augmentation and original/corrected UGO. Select the U5 learner/group alpha first, then select F1 tau on real inner validation with the same frozen real-only detector and reserved roles; reject original proposals without refill. U5+F1 retention remains pending. Standalone U5 and F2 remain discarded, F1 remains kept, and no Stage 1 freeze or broader evaluation is authorized.

**U5+F1 implementation/evidence — retention pending:** explicit `ugo_combined_f1` composes actual fixed real-only OOF MMA and canonical U2 allocation with the unchanged frozen F1 original-proposal gate. All 223 tests passed, including real Shoppers OOF/ancestry/cap/zero-acceptance/outer-test poisoning regressions. The real 21-cell training-only smoke preceded full run `20261008T130740-f85501108e91`, which completed 105/105 cells. Independent replay recomputed all 2,205 numeric metric values, reproduced 90 historical prediction/learner references exactly and passed 40,386 checks, including 180 flow/tau replays. Shared detector weights/roles, OOF scopes/margins and original groups/caps are verified. Selected U5/composition originals match through later rounds; accepted counts and compute differ. All metrics/SDs/pairings/group/acceptance/resource diagnostics are in `reports/u5_f1.md`: relative to F1, LR trades slightly lower AP for better threshold metrics/cost, RF regresses on the headline metrics, and LightGBM improves AP/F1/MCC with equal recall and nearly equal cost. Retention remains the user's decision; no discarded component is reinstated or Stage 1 gate advanced automatically.

**U5+F1 retention and next experiment:** the user explicitly retained U5+F1 after reviewing U5/F1/original-UGO comparisons. This supersedes preceding pending-retention chronology; standalone U5 and F2/H1 remain discarded. Before concluding Phase 1, the user authorized a separate Shoppers U2 learning-benefit candidate-selection experiment with actual downstream logistic regression, fitting-only preprocessing/generation, stratified training guidance, separate validation/test, K=5 saved pools, group-quota-preserving random/gradient-dot-product selectors, matched-count controls, cloned-model SGD checks and equal-weight parent duplication. No FlowGMM/HDBSCAN changes are combined into this experiment; Phase 1 remains open and unfrozen.

**U2 learning-benefit implementation/evidence — pool-utility discarded:** optional candidate selection reuses canonical U2 and actual downstream logistic regression, fitting-only preprocessing/ancestry and stratified labeled training guidance. Frozen full-weight-and-bias balanced-guidance CE gradients rank K=5 post-relocation proposals within unchanged quotas/caps, with one acceptance per anchor, deterministic deduplication, negative-score top-fill and explicit shortfalls. Validation/test stay separate; no outer refit or FlowGMM/HDBSCAN/U1 addition occurs.

The corrected training-only smoke completed six records; full Shoppers run `20261009T120039-97c3231e6f6b` completed 30/30. All 241 tests passed. Independent replay passed 9,689 checks, including all 1,440 final validation/test metric values, 45 saved frozen pools, matched selector/parent replays, true plain-SGD changes, state invariance and original U2 allocation/ancestry. Pinned pre-hook U2 output and predictions reproduce exactly. The complete all-metric/paired/group/resource report is `reports/learning_benefit.md`.

Utility improves its guidance objective and the local estimate is accurate, but independent validation/test AP falls versus pool-random; equal-weight parent duplication is competitive or better. Utility batches are more tightly spaced, but redundancy is not causally identified. All final trajectories reach the same target; original U2 accepts larger first batches because it repeats anchors. The user discarded pool-utility. U5+F1 stays retained, pool-random+F1 is pending, and Phase 1 remains open/unfrozen.

**Validation-guided original-access study authorization:** following the leakage-isolated U2 experiment, the user requested original validation as guidance and historical U2 fitting/preprocessing/generation access, then comparisons with original UGO, original U5 and retained U5+F1 for LR/RF/LightGBM. Native LR gradients are used for LR; the user explicitly selected a separately fitted LR surrogate for RF/LightGBM. Native tree models retain anchor difficulty and downstream fitting; the scorer is refreshed on the exact current training prefix. This is not native tree-gradient utility.

Guidance is excluded from inner fitting/generation but deliberately included in full outer-training refitting/generation. Its metrics are selection/training feedback, not independent validation. Original U2-selected settings lock the three controls; references keep their canonical tuning conventions. K=5 shared initial pools, quotas/caps, one-per-anchor negative top-fill, final-coordinate deduplication, unchanged cadence/stopping, parent-weight controls and checkpoint/state audits are preserved. Five matched seeds and all metrics/compute are required. No retention or Phase 1 freeze is authorized by this experiment.

**Validation-guided study completed:** [all metrics and evidence](reports/validation_guided.md), published bundle `results/runs/20261009T153348-af844bdecf5a`: 240 completed records, five matched seeds, three actual native classifiers, all 24 numeric metric fields and paired differences. The independent audit passed 1,400,828 checks, including historical prediction reproduction, raw-data roles/ancestry, reconstructed pool masks and exact native/scorer prefix refits. The full 251-test suite passed; the final corrected public CLI smoke completed 48 records and the ten new targeted regressions passed again.

The source study's F1 reference inadvertently used CPU rather than retained `auto`/CUDA. Only those 15 F1 cells were rerun; 225 unchanged source cells are explicitly imported. The final references reproduce historical predictions and settings. Source scientific runtime was 6,621.94 seconds; the corrected-reference invocation added 540.54 seconds, with independent auditing additional. Original UGO/U2/U5/random/utility all accept 1,000 final synthetics per seed; U5+F1 keeps its original rejection and smaller dose. Caps cause pool quota shortfalls and more accepting rounds (LR 3, RF/LightGBM 4 versus U2 2); nominal budgets and once-per-round retraining are unchanged, not actual batch counts.

Utility minus pool-random final AP is -0.004598 for LR (loses 5/5), +0.006389 for RF (wins 5/5), and -0.004979 for LightGBM (loses 4/5). LR/LightGBM test losses worsen; RF F1/MCC do not improve on average despite AP/loss gains. Local LR-SGD approximation accuracy does not guarantee actual native refitting benefit. Utility batches are more compact; redundancy remains a hypothesis. Equal-added-weight parent controls are reported, with physical-row/min-leaf caveats for native trees. The user retains the decision: U5+F1 remains kept, pool-utility is discarded by the user; pool-random+F1 is the next experiment, with retention pending, Phase 1 remains open/unfrozen, and broader evaluation stays closed.

**Pool-random+F1 implementation, retention pending:** the user discarded utility selection and requested this combination. Native LR/RF/LightGBM determine unchanged U2 anchor difficulty; K=5 distinct-anchor proposals use existing deterministic deduplication and random per-group selection with one selected candidate per anchor. Retained F1 then checks only chosen original coordinates, replacing relocation; rejection consumes the selected-attempt cap without rescue/refill. Canonical same-seed U2 learner/group settings are locked, four F1 taus use original inner validation, and outer preprocessing/classifier fitting/generation access is unchanged. Frozen retained FlowGMM fit/calibration roles and auto device are preserved; no utility gradients, surrogate, H1, F2 or HDBSCAN changes occur. The actual all-three-classifier smoke passed in 21.59 seconds and the full 262-test suite passed; the five matched Shoppers seeds completed in `results/runs/20261009T170013-d5be50bc2107` (120 records) and passed 770,287 independent checks. [All 24 metrics, paired differences, dose/coverage and runtime](reports/pool_random_f1.md) are reported; imported references preserve explicit source provenance and all 15 native random outer predictions match historical controls exactly. Against pool-random, AP improves for LR/RF but not LightGBM; mean minority recall falls and balanced CE worsens for all three. Actual accepted counts are smaller and not claimed matched. U5+F1 remains kept; Phase 1 is open/unfrozen.

**Flow generation geometry — retention pending:** the user requested U2+random-pool latent interpolation of same-class, same-group real parents. Existing U2 partners are not group-restricted, so a matched within-group input-space control isolates geometry. Initial parent/lambda streams match, while subsequent native-retrained pools may diverge.

Use the original frozen Flow checkpoint and content-keyed dequantized real-parent cache, shared once per seed/role across both geometries and native LR/RF/LightGBM. Decode numeric inverse coordinates into original classifier preprocessing; preserve partner categorical values. This projected mixed-feature adaptation has no density/label gate or new Flow training. Intrinsic invalidity precedes unchanged relocation; no clipping or fallback occurs.

**Geometry verification:** the all-three-classifier Shoppers development-only smoke completed in 27.75 seconds; all 268 tests/307 subtests passed. The full five matched seeds completed in `results/runs/20261009T184911-73fbcd66cdde` (150 records, 1039.26 seconds observed shell time) and passed 1,019,888 independent checks, including 490 native prefix refits. [All 24 metrics for eight final methods, paired differences, counts, validity, relocation and runtime](reports/flow_geometry.md) are reported. Against matched within-group input, Flow improves RF/LightGBM AP and ordinary CE but reduces threshold-0.5 minority recall and worsens balanced CE/error cost; LR loses AP/F1/MCC. Outer Flow invalidity is 62.2%; accepted counts have small shortfalls and are not claimed equal-dose. U5+F1 is unchanged and retained; earlier pool-random+F1 and new Flow geometry remain pending the user’s decision. Phase 1 remains open/unfrozen.

**Current semantics and TabDiff authorization:** the latest user explicitly defines current U5 as U2 alone, not the historical U1-OOF+U2 composition. Preserve historical bundles with their actual semantics; audit genuine U2 native settings separately from the retained frozen F1 checkpoint/thresholds. The user authorized a Shoppers-only fixed-round TabDiff conditional-inpainting experiment before any iterative modification. This narrow learned-generator trial supersedes the earlier sequencing restriction for this experiment only; it does not open Stage2 or broader evaluation.

`configs/ablations/stage1_tabdiff.yaml` and `benchmarks.tabdiff_experiment` fix A original interpolation, B/C nested domain-block masks nearest25/50%, common1000-proposal cap, identical real U2 anchors/quotas and native LR/RF/LightGBM. The user approved common ties-to-even count rounding in all arms before unchanged F1; reject raw invalidity before decoding, enforce known bounds/zero-count-duration/category constraints, and report matched per-group minimum counts and shortfalls without refill. Train actual joint-label mixed-type TabDiff only on real role-training data, share its frozen checkpoint across native families, use correctly time-noised known coordinates and restore unmasked fields exactly. Select B/C on inner-validation AP, then refit the final generator on real outer training after role conversion; never use outer test for selection. Five matched seeds, full metrics, checkpoint/config provenance, duplicate diagnostics and shared-split uncertainty are required. Iterative advancement and retention remain conditional on repeatable evidence and the user decision; Phase1 remains open/unfrozen.


**TabDiff controlled evidence — user review pending:** full Shoppers bundle `results/runs/20261011T154542-1b40c4b555c4` completed all five matched seeds and 165 records after the native LR/RF/LightGBM smoke. All 283 tests/307 subtests and 236,159 independent checks passed, including 120 native refits and 990 actual frozen-weight denoiser evaluations. [All 24 metrics, paired uncertainty, counts, constraints, checkpoint/configuration identities and runtime](reports/tabdiff.md) are published. Inner-selected AP means improve for all three families, but only LR shows a positive conditional interval, alongside lower minority F1/recall/MCC. Tree intervals include zero and generator training dominates cost. The retained U2+F1 configuration is unchanged; no general iterative rollout, unseen-data confirmation, automatic retention or Phase1 freeze follows.


**U2+TabDiff without F1 — user-authorized controlled trial:** remove only the F1 density gate from the preceding fixed-round Shoppers experiment. Reuse its verified training-only frozen generators and complete pre-F1 pools, real U2 schedule/quotas, original-field masks, schema/count decoder and native classifier settings; do not add relocation, utility or another learned filter. Match per-group schema-valid counts without refill, refit actual LR/RF/LightGBM and select mask strength again on no-F1 inner-validation AP. Incremental cost is separate from prior generator fitting/sampling. Historical end-to-end and prior F1 results are context with explicit count/protocol differences. The retained configuration and Phase1 freeze remain the user decision. See `configs/ablations/stage1_tabdiff_u2.yaml` and `benchmarks.tabdiff_u2`.

**No-F1 trial completed — user review pending:** `results/runs/20261012T050806-2ad08eceaf20` contains all five seeds/225 current and reference records. The actual native smoke and full replay passed independent verification, including 18/90 cold native refits, exact schema-only selections and preserved unmasked fields; all 283 tests/307 subtests passed. Mean validation-selected AP is 0.593939/0.722277/0.731298 for LR/RF/LightGBM. Comparisons are development-only, and full iterative retention remains pending the user; retain current U2+F1 unchanged.

The [complete no-F1 report](reports/tabdiff_u2.md) publishes all24 metrics, paired differences, conditional uncertainty, exact counts/shortfalls, diagnostics, frozen checkpoint provenance and separately accounted runtime. Only LR excludes zero in the conditional AP interval versus the matched one-round control; this is not evidence of broad superiority or permission to advance the iterative loop.

**Synthetic downweighting authorization:** include native MLP with LR/RF/LightGBM now and in future controlled Shoppers studies; test U2+F1+Downweighting, U2+Downweighting and U2+random-pool+Downweighting without adding F1 to the random arm. Bagging is explicitly deferred. Keep U2/F1 configuration, real-only preprocessing/parents/groups, quotas, generator/post-handler, retraining cadence, stopping and1000-budget/20-round limits fixed. Weight every accumulated synthetic0.25 or0.5, real rows1; choose the coefficient only by inner-validation AP and compare against fresh unit-weight and real-only controls. Lock existing native settings and active retained F1 checkpoint/threshold provenance; new MLP settings/tau are selected on inner validation and then frozen. Log initial-stream identity, later trajectory/count divergence, all24 metrics, paired shared-split uncertainty, effective added weight, matched-anchor duplication and runtime. See `configs/ablations/stage1_downweight.yaml` and `benchmarks.downweight`; no automatic retention, freeze or broader evaluation follows.

**Downweighting evidence completed — user review pending:** `results/runs/20261012T120400-c73514825f95` completed all five seeds/400 records for native LR/RF/LightGBM/MLP. The real smoke and full study independently passed; full verification covers 1,352,609 checks, 1,581 prefix and 470 final/canonical native refits. All 294 tests/307 subtests passed. [All 24 metrics and paired shared-row/matched-seed intervals](reports/downweight.md), counts, effective weights, divergence, MLP states and runtime are published. All twelve method/classifier mean AP contrasts favor downweighting over unit augmentation, while every recall contrast declines and balanced CE worsens. Real-only mean AP remains higher throughout. Bagging remains deferred; retained U2+F1 and Phase1 status are unchanged pending user review.


### 1. Instrument and modularize

Expose independent components:

`support → seed_score → group_budget → propose → accept → retrain → stop`

Log per round: excluded seeds/groups, proposed/accepted/relocated counts, unique seeds, descendants per seed, duplicates, metrics and retrains.

Initial classifiers: logistic regression, random forest, LightGBM and native MLP. Include MLP in future Shoppers trials; historical three-family cohorts remain unchanged. Bagging is deferred until separately authorized.

Controls: no resampling, class weighting, ROS, SMOTE, BorderlineSMOTE or ADASYN, KMeansSMOTE, original UGO, one imbalance-aware ensemble.

Give all applicable methods comparable tuning and threshold-selection opportunities.

### 2. Improve scoring

Original label-aware margin:

\[
u_i=\max_{c\ne y_i}p(c\mid x_i)-p(y_i\mid x_i).
\]

Binary case: \(u_i=1-2p(y_i\mid x_i)\). Call this difficulty, not a verified epistemic estimate. **Never maximize MMA over candidate coordinates:** it pushes probability away from the assigned label.

Compare separately:

- Original in-sample MMA.
- Out-of-fold MMA.
- Out-of-fold committee disagreement, using the same classifier family:

\[
\bar p_i=\frac1M\sum_{m=1}^{M}p_{mi},\qquad
d_i=H(\bar p_i)-\frac1M\sum_{m=1}^{M}H(p_{mi}),
\quad H(p)=-\sum_c p_c\log p_c.
\]

Start with \(M=3\); increase only if justified. Interpret \(d_i\) as disagreement, not ground-truth uncertainty decomposition.

Calibrate only on real inner-held-out observations. Report calibration separately from ranking improvement.

### 3. Protect groups and allocate budget

- Build minority groups from real training observations. Compare simple clustering and a neighborhood graph; optionally HDBSCAN.
- Do not equate clustering noise with label corruption.
- Estimate group stability \(r_g\in[0,1]\) from repeated real-data subsampling/co-membership. Log cross-class conflict separately.
- Define \(n_g\) as distinct real support; \(h_g\in[0,1]\) as mean within-class percentile-ranked difficulty.
- Compare size-proportional, uniform-group and difficulty-only budgets against:

\[
a_g=\frac{r_g(h_g+\epsilon)}{(n_g+\epsilon)^\alpha},\qquad
q_g=B\frac{a_g}{\sum_h a_h}.
\]

Tune \(\alpha\) on development data; include \(\alpha=0\). Round quotas to integers, apply group/seed caps, redistribute only to eligible groups. If no group is eligible, generate nothing.

- Define singleton behavior explicitly; no interpolation without a valid partner.
- Do not force quotas for unsupported groups. Do not let synthetic counts increase \(n_g\) or reliability.

### 4. Constrain generation

Start with within-group interpolation between compatible real seeds:

\[
x'=x_i+\lambda(x_j-x_i),\qquad \lambda\in[0,1],\quad y_i=y_j.
\]

For continuous features, test these empirical constraints in a training-fitted distance space:

\[
\|x'-x_i\|\le\eta d_-(x_i),\qquad
\frac{d_+(x')}{d_-(x')+\epsilon}\le\rho,
\]

where \(d_+\) and \(d_-\) are distances to nearest real same-class and opposing-class observations. Tune \(\eta,\rho\) inside training data. These checks do not guarantee label correctness.

- Avoid interpolation across disconnected groups.
- Use type-aware handling for categorical features and domain constraints.
- Compare original relocation, rejection and step shortening \(\lambda\leftarrow\lambda/2\), with a fixed retry cap.
- Do not require all accepted seeds to already be correctly classified.
- Record rejected candidates and unspent budgets; never silently weaken constraints to fill quotas.

### 5. Select stopping

Compare original stopping with a small fixed budget grid and inner-validation checkpoints, including zero augmentation:

\[
t^*=\arg\max_{t\in\mathcal T\cup\{0\}}\widehat{\mathrm{AP}}_{\mathrm{inner}}(f_t).
\]

Use a fixed generation/retraining cap. Refit the selected procedure on the outer training partition before outer evaluation. Do not use outer scores to choose checkpoints.

Use the prespecified dataset objective when selecting checkpoints: AP for the general controlled binary track, cost minimization for APS, and validation operating-point performance for BAF. Keep each objective and threshold rule fixed in its protocol; outer/test scores never select checkpoints.

### 6. Run ablations and freeze

| ID | Modification |
|---|---|
| U0 | Original UGO |
| U1 | Scoring only |
| U2 | Group protection/allocation only |
| U3 | Candidate constraints only |
| U4 | Stopping only |
| U5 | Successful components combined |
| W5 | U5 seed allocation implemented as sample weights |
| R5 | U5 seed allocation implemented as exact duplication |

For seed multiplicity \(k_i\), compare weights \(w_i=1+k_i\) against duplication, controlling loss normalization and regularization where applicable.

Simulations must vary independently: imbalance, minority-group size, true overlap, label corruption and irrelevant dimensions. Include separated islands and disconnected/curved minority support.

**Gate:** explain each retained component; demonstrate where generation adds value beyond W5/R5. Otherwise retain weighting as a competitive outcome and baseline. Freeze the Stage 1 configuration.

## Stage 2 — Learned generators

### Benchmark progression — learned generators only after the Stage 1 gate

Keep Stage 1 configurations frozen. Extend the benchmark manifest, target registry and protocol configurations only as these tasks enter this stage.

- **Isolet, exact MGVAE binary task — primary benchmark for Steps 7–8.** Recover the author files, binary class mapping and split construction before training; standard ISOLET has 26 classes and must not be converted ad hoc. Recover normalization, two-layer fully connected classifier, optimizer/checkpoint selection, oversampling counts and three-trial seeds. Reproduce SMOTE and MGVAE before comparing statistical generation/CVAE and original/frozen Stage 1 UGO with matched output counts. Paper Table 7 targets: SMOTE 93.8 ± 0.7%, MGVAE 95.1 ± 0.6%, on the author's **balanced-test accuracy called B-ACC**; verify the exact metric rather than assuming it is a different balanced-accuracy implementation.
- **Water Quality and Musk, exact MGVAE tasks — add only after Isolet reproduction works.** Resolve the precise data/files, labels and raw versus split-specific counts; do not substitute a water-potability dataset or assume a Musk variant. Table 6 Water Quality targets: SMOTE 56.8 ± 1.4%, MGVAE 67.2 ± 1.1%; Table 5 Musk targets: SMOTE 83.6 ± 1.3%, MGVAE 92.4 ± 1.2%, under the same balanced-test/two-layer-classifier/three-trial protocol. Inspect Isolet speaker and Musk molecule/conformer identities; add group-disjoint evaluation as a distinct protocol where recoverable.
- **Controlled companion evaluation:** share one neural classifier and one boosting classifier across methods, report AP and the common metrics at natural prevalence where recoverable, and keep these results separate from author balanced-test B-ACC. Subsample only minority training data at predetermined levels in Step 8; validation/test remain fixed.
- **Carry forward Shoppers, APS and BAF** for controlled learned-generator generalization once a generator is viable; preserve their audited category, official-test, cost and temporal contracts. Do not use confirmatory outcomes to revise the frozen Stage 1 method. Shoppers' SOS reproduction belongs specifically to Step 9 below.

### 7. Define conditioning; implement CVAE

Keep Stage 1 selection, budgets, acceptance and evaluation fixed initially.

Implement a simple statistical generator and one shared CVAE. Avoid independent generators for tiny groups.

For \(c=(y,g)\), minimize:

\[
\mathcal L_{\mathrm{CVAE}}
=\mathbb E_{q_\phi(z\mid x,c)}[-\log p_\theta(x\mid z,c)]
+\beta D_{\mathrm{KL}}(q_\phi(z\mid x,c)\|p(z\mid c)).
\]

Make selection operational: sample near selected seeds through their latent posteriors, or sample selected groups according to \(q_g\). Label these as different mechanisms; group conditioning does not preserve individual-seed selection automatically.

Compare uniform class sampling, group-budget sampling and UGO-guided sampling using the identical generator and output count.

### 8. Measure data requirements

- Subsample minority training observations at predetermined levels; keep validation/test fixed.
- Plot utility, group coverage, duplication and runtime against real minority count.
- Compare interpolation, statistical generator and CVAE; include the reproduced MGVAE reference on its audited tasks.
- Retain interpolation on regimes where learned generation does not help.

### 9. Add diffusion

Start on sufficiently large datasets. First test conditional diffusion with the existing acceptance filter; then test internal guidance.

Add the **SOS published-generator comparison on Online Shoppers** after its Stage 1 no-resampling/SMOTE protocol is reproduced: SOS-VE weighted-F1 target 0.675 ± 0.004 (Table 2, four classifiers/five seeds). If Satimage was retained, add its six-class SOS-VP target 0.857 ± 0.006 under the same aggregation. Audit the author score-based process and recipe; do not claim a generic conditional-diffusion run reproduces SOS. Compare frozen UGO-guided and class-only generation in both matched-count controlled and native-budget tracks. Use APS/BAF for scale only when feasible and keep validation/test boundaries unchanged.

Continuous diffusion baseline:

\[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\mathcal L_{\mathrm{diff}}=\mathbb E\|\epsilon-\epsilon_\theta(x_t,t,c)\|^2.
\]

Use a categorical-compatible process for discrete features. Compare class-only conditioning, class-plus-group conditioning, post-generation filtering and internal guidance separately.

For optional boundary guidance, let \(m(x,y)=p(y\mid x)-\max_{c\ne y}p(c\mid x)\). Test targeting a small positive margin \(\delta\), subject to support constraints, rather than maximizing MMA:

\[
J(x,y)=-[m(x,y)-\delta]^2-\gamma R_{\mathrm{support}}(x).
\]

Start with \(R_{\mathrm{support}}(x)=[d_+(x)/(d_-(x)+\epsilon)-\rho]_+\), where \([a]_+=\max(a,0)\). Retain hard feature-validity checks.

Apply clean-input guidance to a predicted clean sample, or train a time-conditioned noisy-input guide. Do not apply an unadapted clean classifier directly to noisy states. Measure diversity loss and guidance cost.

### 10. Add unlabeled data

Construct \(D_U\) by hiding labels within training data only. Vary labeled fraction and unlabeled class prior separately.

Use the already-audited Stage 2 tasks and carried-forward tabular benchmarks, not new test-derived unlabeled pools. Record labeled budgets, label-removal seeds and whether label removal is stratified; retain APS' official test and BAF's future months completely outside the unlabeled pool.

Compare:

1. Labeled-only pipeline.
2. Unlabeled neighborhood structure without hard pseudo-labels.
3. Self-training without synthetic generation.
4. Pipeline plus accepted pseudo-labeled observations.
5. Generation around accepted pseudo-labeled anchors.
6. Fully labeled training-pool reference.

Candidate acceptance rule:

\[
\hat y=\arg\max_c\bar p(c\mid x),\qquad
\bar p(\hat y\mid x)\ge\tau_{\hat y},\quad d(x)\le\kappa,
\]

plus real-anchor support. Select class-specific thresholds using inner labeled data; cap pseudo-label weight and descendants. Use hidden labels only for evaluation.

**Gate:** establish when learned generation helps and whether guidance adds value. For unlabeled experiments, establish added value over self-training alone. Freeze the successful configuration.

## Stage 3 — Images

### 11. Transfer interpolation first

- Start with **CIFAR-10-LT**, primary imbalance factor IF=100; add IF=50 and IF=10 sensitivities only when supported by the audited reference recipe. Recover long-tail sampling, class order, seeds, exact counts and the balanced test protocol; IF alone does not identify the split.
- Reproduce the chosen representation/classifier recipe before generator comparisons, then freeze one encoder/checkpoint (`z=E(x)`, dimension `d`). Keep external-pretraining feature experiments (for example CLIP/DINO) in a separate table from training-from-scratch reference reproductions.
- Hold extracted features, classifier head, augmentations, generation counts and training budget fixed across methods. Report top-1, macro/per-class recall and explicitly defined many/medium/few bins in addition to applicable shared metrics.
- Compare no augmentation, weighting, duplication, feature interpolation, original UGO and Stage 1 UGO.
- Treat generated vectors as feature augmentation; do not claim generated images.

### 12. Transfer learned generation

- Train the successful Stage 2 generator on embeddings.
- Compare class-conditioned and UGO-guided generation with matched counts.
- Evaluate per-class and within-class group recall; record support counts.
- Expand to CIFAR-100-LT only after the first benchmark is stable.
- Treat backbone fine-tuning as a separate experiment requiring renewed controls.

**Stage 3 published comparisons:** audit LDMLR's WCDAS + feature-diffusion recipe and code availability before claiming reproduction: reported IF=100 top-1 is 86.29% on CIFAR-10-LT and 51.92% on CIFAR-100-LT. LDMLR code availability is unresolved. After CIFAR-10-LT is stable, add **CIFAR-100-LT** and the LOS classifier-retraining control (reported IF=100 top-1 54.9%). The recipes differ; these targets are not a direct method ranking or targets for an unrelated frozen pretrained encoder. Include a strong classifier-retraining baseline and matched-compute/count controlled comparisons before attributing gains to generation.

### 13. Optional raw-image generation

Use a pretrained conditional image generator only after feature experiments. Evaluate label fidelity, diversity, near-duplicates and downstream utility. Keep image synthesis results separate from feature synthesis.

**Gate:** show transfer with fixed representations before attributing gains to representation learning or image generation.

**Stage 3 benchmark gate:** retain exact LT split/checkpoint identities and separate reported/reproduced values and external-pretraining protocols. Complete feature-transfer evidence before optional raw-image synthesis. Do not acquire/start image experiments during Stages 1–2.

## Metrics and reporting

- Primary controlled binary metric: average precision (AP); specify multiclass one-vs-rest macro aggregation when applicable. Prespecified protocol objectives: APS challenge cost, BAF recall at 5% FPR, exact audited SOS inverse-support F1 (not sklearn weighted F1), MGVAE author B-ACC and CIFAR-LT top-1. Report companions, but never rename or substitute published objectives.
- Secondary: macro-F1, balanced accuracy, MCC, minority recall at an inner-selected fixed precision/FPR target.
- Groups: \(R_g=TP_g/(TP_g+FN_g)\), \(R_{\min}=\min_gR_g\); include support counts and confidence intervals. No subgroup F1 without a defined negative population.
- Calibration: Brier score, log loss, class-wise reliability; ECE supplementary.
- Generation: validity failures, duplicate rate, acceptance rate, seed/group coverage, descendant concentration.
- Safety: known-distribution contamination in simulations; neighborhood conflict only as a proxy on real data.
- Cost: full training/generation/scoring time, retrains, memory and tuning overhead.
- Statistics: paired dataset-level differences and intervals; corrected planned comparisons. Do not treat folds as independent datasets.
- Plots: learning curves, generation-round curves, per-group recall and performance–runtime tradeoffs. UMAP/t-SNE are illustrative only.

## Required deliverables per stage

- Frozen configuration and executable experiment entry point.
- Predictions, metrics, provenance logs and runtime records.
- Ablation table and failure cases.
- Short decision record: retain / remove / defer, with evidence.
- Updated experiment registry and exact next unfinished step.
- Stage-relevant benchmark manifest entries and separate published/controlled configurations; no unresolved field silently replaced by an assumption.
- `results/published_targets.csv` and `reports/reproduction.md`: reported versus reproduced values, exact metric/unit/aggregation, meaning of each spread (SD/SE/CI or unresolved), deviations, commands and unresolved comparison details.
- `results/benchmark_runs.csv`: protocol/split/seed, commit/config hash, method, metrics, runtime/peak memory, generated counts, status and prediction locations; link to the existing experiment registry/per-run artifacts using the same run and protocol IDs rather than treating duplicate exports as independent runs.

## Core references

- [UGO](https://doi.org/10.1016/j.knosys.2026.115886)
- [CLIMB](https://arxiv.org/abs/2505.17451)
- [KMeansSMOTE](https://arxiv.org/abs/1711.00837)
- [MGVAE](https://proceedings.mlr.press/v206/ai23a.html)
- [TabDDPM](https://proceedings.mlr.press/v202/kotelnikov23a.html)
- [CReST](https://arxiv.org/abs/2102.09559)
- [Benchmark datasets and protocol details](DATASETS.MD)
- [SOS](https://arxiv.org/abs/2206.08555)
- [APS challenge](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) and [cost-sensitive transformer](https://link.springer.com/article/10.1007/s10586-026-06080-2)
- [BAF](https://github.com/feedzai/bank-account-fraud)
- [LDMLR](https://openaccess.thecvf.com/content/CVPR2024W/L3D-IVU/papers/Han_Latent-based_Diffusion_Model_for_Long-tailed_Recognition_CVPRW_2024_paper.pdf)
- [LOS](https://github.com/Thinklab-SJTU/LOS)
