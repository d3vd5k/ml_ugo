# Uncertainty Guided Oversampling

Stage 1 now exposes independent `support → seed_score → group_budget → propose → accept → retrain → stop` components in `samplers/ugo.py`. `UGOPipeline(UGOConfig(...)).run(X, y, model_fn)` returns augmented tensors, round records, real-parent provenance, support counts and a stop reason. Each component is replaceable through a constructor callback. Labels must be contiguous integers starting at zero.

The current allocation baseline treats each class as a group, scores real seeds with in-sample true-label MMA difficulty, interpolates between eligible real same-class observations, and relocates conflicts to real eligible target-class references. This is the corrected real-only pipeline, **not a claim of complete original-paper reproduction or completion of Stage 1 experiments**. OOF scoring, learned grouping/budgets, alternative acceptance/stopping mechanisms and comparative ablations in `PLAN.MD` remain separate work.

For a concise, mechanism-only explanation, see [Technical changes to UGO](reports/technical_changes.md).

For the evidence-backed closing proposal, see [Phase 1 final pipeline draft](reports/phase1_final_pipeline.md). It targets minority-class F1 with validation-selected decision thresholds, optional U2/FlowGMM augmentation and synthetic-weight selection; it does not change implemented defaults, retention decisions or freeze status.

## Standalone 30-dataset notebook

Copy/download **[UGO_30_Datasets.ipynb](UGO_30_Datasets.ipynb)** and run its cells in order in a Python 3.11+ Jupyter/Colab kernel. The first cell installs dependencies; all pipeline source is embedded, so no repository checkout or sidecar `.py` files are needed.

The catalogue now has **32 stable IDs**, with `DATASET_RANGE = "1-32"` and `EXCLUDED_DATASETS = [10, 26]` selecting **30 active datasets**. IDs 1–30 are unchanged; the two large datasets remain numbered 10 and 26 but are skipped by default. Appended replacements are **31: haberman (306 rows, 3 predictors)** and **32: blood-transfusion-service-center (748 rows, 4 predictors)**. To run only the additions, set `DATASET_RANGE = "31-32"`; other inclusive ranges such as `"1-5"` still work without renumbering. Set `EXCLUDED_DATASETS = []` only to deliberately permit the large datasets again. These are binary CLIMB/imbalanced-ensemble sources, not imbalanced-learn's separate 27-dataset collection. The complete numbered catalogue and raw OpenML IDs are inside the notebook.

Each selected dataset compares **no oversampling, original UGO, U2 + FlowGMM compatibility gate, SMOTE, KMeansSMOTE and RWO (Random Walk Oversampling)**, with **MLP, logistic regression, XGBoost, random forest and RBF SVM**. Selection uses real-validation minority F1, including a frozen classifier decision threshold. MLP/FlowGMM/dense distances and XGBoost use CUDA when available; sklearn LR/RF/SVM and the three added sampler implementations run on CPU. `DEVICE = "cpu"` supports machines without NVIDIA GPUs.

The first cell installs imbalanced-learn as well as the existing dependencies. `SAMPLER_PARAMETERS` exposes fixed SMOTE/KMeansSMOTE settings; the default KMeansSMOTE balance threshold is explicitly 0.0 to admit rare-positive clusters, never a fallback after failure. Standard SMOTE/KMeansSMOTE operate on numeric/full-one-hot coordinates with whole-category argmax projection, not silently substituted SMOTENC. RWO uses real-class standard-deviation/sqrt(n) Gaussian steps for numeric features and nominal-frequency sampling for categorical blocks. Training-only sampler pools are reused across native learner candidates/classifiers; infeasible sampler settings remain recorded failures.

The notebook displays all six F1 means without hiding columns. `comparison.csv` adds per-method F1 SD/seed coverage, `summary.csv` contains all metrics, and `paired_deltas.csv` compares all 15 method pairs on identical data/splits. Partial coverage is flagged explicitly rather than ranked as a complete comparison.

Verified for the expanded catalogue: raw binary source sizes/targets for IDs 31/32, **60 CUDA smoke cells** outside the repository covering both additions × all five classifiers × all six methods, six independently recomputed metrics per cell, 60 validation-selected thresholds and all 150 matched method pairs. Five stable-range/exclusion regression tests passed. Earlier sampler-core evidence includes 60 CPU and 60 CUDA smoke cells, numerical sampler regressions and native/gated/all-rejected U2 parity; original catalogue entries are unchanged. The full **2,700-cell** default study was not run; other operating systems and clean-machine dependency installation were not exercised.

Maintainers can regenerate the notebook using `python notebooks/build_portable_notebook.py`; the builder embeds the tested root sampler cores plus `notebooks/portable_ugo/` and the fixed catalogue. Notebook users need only the `.ipynb`.

## Publication-quality synthetic comparison figures

From the repository root, using the project environment:

```bash
conda run --no-capture-output -n ml_project python -B -m reports.synthetic_six_methods --device auto
```

The generator loads the exact sampler implementations embedded in `UGO_30_Datasets.ipynb`, without running its installer or 30-dataset experiment. It creates `make_moons`, two-feature linear `make_classification`, and `make_circles` examples with the same **400 real majority / 100 real minority** observations for every method. The default seed is 42. Original UGO and U2 use the notebook-default MLP for native difficulty scoring (160 epochs); the frozen real-only FlowGMM uses 60 epochs, group alpha 0.5 and gate tau 0.6. Twenty percent of the real rows are reserved from FlowGMM fitting and are not used for calibration. Parameters are prespecified, not selected for the prettiest plot or claimed F1 performance.

Outputs are saved in `reports/figures/synthetic_six_methods/`:

- `moons_comparison`, `linear_comparison`, `circles_comparison`: six-panel publication figures, each 7.2 × 5.8 inches.
- `overview`: all 18 dataset/method panels; `overview_preview.png` is a smaller viewing copy.
- Each publication figure is saved as **600-dpi PNG plus vector PDF/SVG**; PDFs embed TrueType fonts.
- `points.csv`, `counts.csv`, and `manifest.json`: exact plotted coordinates, actual/requested augmentation counts, stopping reasons, implementation checksum, parameters, versions and sampler/FlowGMM audits.

All six methods use identical real observations and the same requested budget of 300 additions (baseline requests zero). Original UGO stopping, U2 quotas and gate rejection can leave fewer synthetic points; annotations show the actual count rather than silently refilling. Axes use original sklearn feature coordinates with equal aspect and shared limits across methods for each dataset; no outliers are clipped, and overlapping points are not jittered. Colors and marker shapes distinguish real majority, real minority and synthetic minority points.

These are **single-seed qualitative training illustrations, not held-out F1 evidence**. Use the separate notebook benchmark for predictive comparisons. `--device cpu`, `--seed`, `--dpi` (minimum 300), `--output`, and `--notebook` are available for reproduction. Generated figures remain local under the repository's source/text-only upload policy.

## Shoppers: controlled U2 + Flow gate ablation, weight 1

The dedicated [configuration](configs/ablations/shoppers_u2_flow_weight1.yaml) and `benchmarks.shoppers_ablation` runner implement the requested Shoppers-only study for **MLP, random forest and XGBoost**: five duplicate-predictor-group-disjoint outer folds, three matched seeds, eight factorial/knockout/reference variants, and ten random masks for each of two fixed-proposal replay controls. All real and synthetic weights are **1**. Unlike the [historical AP-selected table](reports/shoppers_u2_flow_ablation.md), learner candidates and decision thresholds are selected on **real-validation minority F1**; sampler alpha/tau are fixed at 0.5/0.6.

The study trains MLP, XGBoost and the shared real-only FlowGMM models on CUDA, and uses CUDA for native sampler neighbor math. Random forest trains on CPU with four jobs; probability accumulation uses one job to make MMA ranks, threshold ties and exact replay checks deterministic. Preprocessing/KMeans remain CPU operations. XGBoost may report its host-input DMatrix prediction path; its training booster device is checked to remain CUDA, with no silent CPU-training fallback. This runner requires CUDA for the registered training protocol; the standalone benchmark notebook remains CPU-capable.

Reproduce in a **new** run directory (preparation refuses to overwrite existing splits):

```bash
RUN_DIR=results/runs/shoppers-u2-flow-weight1-reproduction
conda run --no-capture-output -n ml_project python -B -m benchmarks.shoppers_ablation --prepare --run-dir "$RUN_DIR"
conda run --no-capture-output -n ml_project python -B -m benchmarks.shoppers_ablation --run --classifier mlp --run-dir "$RUN_DIR"
conda run --no-capture-output -n ml_project python -B -m benchmarks.shoppers_ablation --run --classifier random_forest --run-dir "$RUN_DIR"
conda run --no-capture-output -n ml_project python -B -m benchmarks.shoppers_ablation --run --classifier xgboost --run-dir "$RUN_DIR"
conda run --no-capture-output -n ml_project python -B -m benchmarks.shoppers_ablation --collect --run-dir "$RUN_DIR"
```

Requested run directory: `results/runs/shoppers-u2-flow-weight1-20261023T161535Z`. Completed workers can be resumed without duplicating their unit records. Preparation freezes source hashes, raw-source provenance, nested partitions, real-only preprocessing/groups and one detector checkpoint per fold/seed, shared by all classifiers. Classifier workers write selections, pre-outer threshold locks, inner/outer predictions, final training arrays, ancestry/audits and A4 original-proposal traces. Replay masks match counts by round or round+subgroup; they are conditional final-fit interventions on A4's stream, not independent random generation trajectories. Classifier model checkpoints are not retained.

Collection requires exactly **360 main + 945 replay outer evaluations**, independently verifies saved artifacts, and writes `RESULTS.md`, `metrics.csv`, `oof_metrics.csv`, `summary.csv`, `paired_deltas.csv`, `paired_summary.csv`, `replay_summary.csv` and `verification.json`. Scores pool the five held-out folds within each training seed; random-mask metrics are averaged within that seed before reporting seed means/sample SD. All metrics are also reported at threshold 0.5. Shared detector fit times are references, not costs to sum per classifier; outer cell timings exclude learner search and are not independent serial-speed benchmarks when workers run concurrently. These remain development results on the previously examined Shoppers dataset, not independent-fold significance or confirmatory claims.

Integration evidence: all 120 main and 75 replay cells passed on an isolated 800-row training-only smoke subset, including leakage/prefix/weight/cap/knockout checks and exact replay predictions. Native A4/A7 sampler geometry, ancestry and stopping also matched the notebook's canonical implementation exactly in a CUDA MLP smoke context. Smoke settings are explicitly labelled and are never substituted for full-study settings.

The requested full run **completed all 360 main and 945 replay evaluations**; independent saved-prediction, threshold, split, weight, ancestry, knockout and count-matching verification passed. See the [full results](results/runs/shoppers-u2-flow-weight1-20261023T161535Z/RESULTS.md), [main summary CSV](results/runs/shoppers-u2-flow-weight1-20261023T161535Z/summary.csv), or [results/source bundle](results/runs/shoppers-u2-flow-weight1-20261023T161535Z/results_bundle.zip). The bundle contains summaries, verification records and an offline-checked source/data archive; detailed prediction/training/trace artifacts remain in the full local run directory. All 24 main F1 means and sample SDs also independently matched direct pooling of per-fold confusion counts. The gate improves mean F1 over U2 pass-through for RF/XGBoost, not MLP; count-matched replay likewise does not show a universal gate benefit. No automatic retention/default change is applied.

## Git upload contents

`.gitignore` allows source, configuration, documentation, text metadata and the standalone `UGO_30_Datasets.ipynb`. Results are limited to text summaries directly in `results/`, the run registry, and immediate `results/runs/<run-id>/` files. All nested run artifacts, fold/seed directories, smoke outputs, caches, trained models, array files, archives, PDFs, and other generated notebooks stay local. Retained summaries use regular Git, not Git LFS.

Summary-only uploads do not contain the checkpoints or detailed per-fold/per-seed artifacts needed for artifact-level verification; regenerate them locally when needed. Ignore rules affect new additions, not files already committed or stored in Git history.

## Run

Use the existing Conda environment:

```bash
conda run -n ml_project python execute_ugp.py --dataset nonlinear --no-plots --random-state 42
conda run -n ml_project python execute_ugp.py --all-datasets --no-plots
```

The six geometries are `linear`, `nonlinear`, `moons`, `circles`, `blobs`, and `classification`. Synthetic controls are independent:

- `--minority-weight`: exact **clean class-1 prevalence** (rounded), including the nonlinear generator. Selection is without replacement.
- `--noise-ratio`: fraction of inner-fit labels flipped, not an overlap measure. Validation and test labels remain clean. Corruption can change observed training prevalence.
- `--overlap`: extra Gaussian feature-noise standard deviation, with clean labels unchanged.
- `--irrelevant-features`: independent standard-normal dimensions.

Geometry, selection, overlap, corruption and irrelevant dimensions use separate random streams. Train/test observations come from disjoint pools; `classification` shares its sampled latent geometry between pools to avoid accidental distribution shift. `--n-samples` is their combined count and `--test-fraction` defaults to 0.2. These controlled variants are not identical to the former paper-demonstration sampling protocol.

`--device auto` selects CUDA when available; `cpu` and `cuda` are explicit overrides. `--epochs` and `--ugo-max-iterations` control training and generation limits.

## Named benchmarks: setup first

Use Shoppers for Phase 1 quick checks, not the full CLIMB suite. `requirements.txt` records the installed dependency versions used for setup. The named runner requires an explicit mode: `--prepare` (no model fits), `--smoke` (small training-only checks), or `--run` (a study, only when deliberately requested).

```bash
# Acquire/validate Shoppers and save protocol, split and candidate snapshots; no model fitting.
conda run --no-capture-output -n ml_project python -m benchmarks.run --prepare --download

# Quick integration check; every smoke row comes from original outer TRAINING.
conda run --no-capture-output -n ml_project python -m benchmarks.run --smoke --download --seed 42
```

Smoke defaults: 400 outer-training rows for fit/validation, 120 separate training-pool rows for evaluation, one tuning candidate, one generation round, three MLP epochs and a 40-row augmentation cap. The Shoppers smoke-only KMeansSMOTE configuration declares three clusters and cluster-balance threshold0.05. These are setup settings, not frozen experimental choices; original outer/official test rows remain untouched. Smoke outputs are isolated under `results/smoke/` and excluded from study summaries.

`benchmarks/manifest.yaml` pins raw UCI bytes, schemas, labels, licenses, missingness, partition counts and overlap notes. `configs/benchmarks/` contains separate controlled/published IDs:

- **Shoppers:** raw UCI mixed-type features, fixed natural-prevalence outer holdout, real stratified inner validation, primary AP.
- **APS:** official60,000/16,000 partition; inner validation only within training; threshold selected by validation cost `10 FP + 500 FN`.
- **BAF Base:** official local `Base.csv`, `Base.parquet` or release ZIP supplied through `--local-data`; no credentials are accessed. Months0–5 train, month5 validates, months6–7 test. Controlled LightGBM has100 materialized candidates from audited distribution families, not a claim of exact Optuna random-stream replay. Missing-value sentinels are normalized according to the manifest.
- **Satimage:** optional six-class task with official UCI partitions; distinct from binary CLIMB Satellite and SOS prepared row sets.

Select another protocol with `--protocol configs/benchmarks/aps_controlled.yaml` (or `baf_base_controlled.yaml`, `satimage_controlled.yaml`). Published configurations list unresolved prerequisites and are refused rather than silently replacing the recipe. `reports/reproduction.md` records source pins, targets and deviations; no baseline scores have been claimed reproduced by setup checks.

### Classifiers and controls

The shared factory supports logistic regression, random forest, XGBoost, LightGBM, balanced random forest, KNN, RBF SVM, MLP, decision tree and AdaBoost. Named controlled presets use LR/RF/LightGBM plus the intrinsic balanced-random-forest baseline. All declared sampling methods receive the same inner validation, candidate configurations and tuning opportunities within each classifier. Repeated finite-space candidates and unique candidate counts are explicit. Class weighting uses balanced real-training sample weights; KNN weighting is an explicit unsupported cell. MLP uses normalized weighted cross-entropy and unchanged real validation.

Controls: no resampling, class weighting, ROS, SMOTE, BorderlineSMOTE, KMeansSMOTE, paper-derived original UGOSMOTE and corrected real-only UGO. ADASYN is available through the sampling API. SMOTE uses true SMOTENC on mixed features; one-hot blocks are collapsed to whole nominal variables and restored. Borderline/ADASYN/KMeans use their actual numeric-plus-one-hot algorithms followed by declared whole-category projection, not an invented native SMOTENC variant. Impossible support raises a recorded failure; no ROS fallback, neighbor weakening or synthetic padding occurs.

`samplers/ugo_original.py` is a distinct paper-derived reference whose current augmented observations, including descendants, participate in scoring/support. The paper contradicts itself about retraining and caps; the implementation documents its prose-based choices. Iteration limits, matched-count caps and controlled categorical projection remain explicit deviations. It is **not verified author-code reproduction**.

### Ablation and result methodology

- `configs/ablations/stage1_controls.yaml`: baseline/control study, separate from component claims.
- `configs/ablations/stage1_components.yaml`: U0, corrected_current, baseline, U1_oof_mma, eight U2 comparisons, the historical U5 kept-component combination, three historical U3 repairs and two historical U4 stopping comparisons are runnable for explicit experiments/reproduction. The user kept U1_oof_mma and canonical U2 KMeans/stability, and discarded U3/U4/U5. The user reaffirmed canonical U2 as the preferred candidate; U5 is preserved only for explicit historical reproduction, not retained; full U1 committee/calibration and W5/R5 remain unavailable. corrected_current is never mislabeled U5.
- Use `--study`, `--variant`, `--study-id`, `--classifiers`, `--methods`, repeated `--seed`, and explicit tuning/count overrides to declare a study. Configuration snapshots bind protocol/study/settings; unavailable variants and unfrozen confirmatory runs are refused. Setup does not open the Stage2 gate.
- Matched budgets have prespecified per-class maxima. Record actual counts and underspend; equal requested caps do not establish equal realized counts. Native budgets remain a separate track. Weighting and no-generation baselines are not matched-count generation mechanisms.
- Every named run reuses `RunArtifacts` and `experiment_registry.csv`. `results/benchmark_runs.csv` is a linked cell index for canonical `results/runs/`, not a second independent run registry. Other roots keep their index locally. Records identify run, immutable policy/selected-config hashes, dataset/split fingerprints, seed, classifier, method, variant, candidates, generated counts, status, predictions, runtime and memory.
- Per-cell artifacts retain all inner candidate metrics/thresholds, selected/replayed epochs, preprocessing state, predictions and generation logs. UGO/ROS ancestry is retained; imbalanced-learn does not expose SMOTE parent indices, which is recorded rather than fabricated. Original UGO ancestry may reference descendants and has explicit local-index/real-row mappings.
- Peak RSS is sampled at10ms over each cell; CUDA counters cover Torch allocations only, not total XGBoost/other GPU memory. Shared preprocessing is performed once per split, outside cell timing.
- Summaries keep protocols, dataset versions, study policies and budget tracks separate; aggregate folds within seeds before descriptive seed means/SD. Pairing requires matching real split/seed, tuning candidates and policy; actual class counts determine whether a generation comparison is count-matched. Duplicate run cells are refused rather than double-counted. No independent-fold significance claims are produced.
- Current Phase 1 runs Shoppers only. U1–U5 keep/discard decisions belong to the user; later components require prior decisions and freeze requires all five explicit decisions. CLIMB runs only after the user finalizes Phase 1.
- U1_oof_mma uses three stratified scoring folds inside each real training partition. Imputation, scaling and encoding fit separately inside each fold; the same-family scoring models train on real fold-training rows only. Their true-label margin scores are computed once per candidate before generation and remain fixed. Synthetic rows never enter those models; downstream UGO still retrains on augmentation. Full outer-training refit rebuilds cross-fitting without test feedback. Fold identities, fitted states, OOF probabilities and scores are saved in candidate/selection/outer seed_scoring audits.
- Component experiments rerun no-resampling and corrected_current under the same immutable component-study policy, so new candidate comparisons do not silently mix study hashes.

```bash
conda run -n ml_project python -m benchmarks.summarize results/benchmark_runs.csv --run-id RUN_ID --reference baseline
# Inspect smoke records only when explicitly requested; not experimental evidence.
conda run -n ml_project python -m benchmarks.summarize results/smoke/benchmark_runs.csv --include-smoke
```

**Audit caveats:** SOS code uses inverse-support F1, not sklearn weighted F1; Shoppers scaling metadata includes held-out feature bounds; its repetitions are not five explicit independent seeds. SOS Satimage uses different row partitions and only two evaluation classifiers. The BAF reference selects top configurations on test scores; controlled selection uses month5 instead. Published targets remain reported references, never inferred local results.

### Established full-data Shoppers baseline

Run `20261003T180121-ee6dd58a68ac` completed all 20 controlled baseline cells: LR, RF, LightGBM and intrinsic balanced RF, five seeds, ten tuning proposals each. All 12,330 rows were used under the declared split (9,864 training / 2,466 fixed holdout), with validation-only selection and no generated rows. AP means ± seed SD were **0.6089 ± 0.0028**, **0.7395 ± 0.0057**, **0.7428 ± 0.0039** and **0.7396 ± 0.0045**, respectively. These are development results on one shared holdout, not independent-replicate confidence intervals or published reproductions.

See [the baseline audit](reports/reproduction.md#full-shoppers-baseline-controlled-development) for the command, secondary metrics and verification. Machine-readable references: `results/runs/20261003T180121-ee6dd58a68ac/baseline_summary.json` and `baseline_summary.csv`; full predictions, candidate audits and source snapshots remain in that run directory.

### Established full-data Shoppers UGO baselines

Run `20261003T181637-fabbd21ddfb8` completed **30 original/corrected UGO cells** with LR/RF/LightGBM, using exactly the same full-data splits, five seeds, ten tuning proposals, preprocessing and fixed threshold as the no-resampling reference. Balanced RF remains baseline-only (ten explicitly unsupported combinations). All inner and final sampler runs generated exactly **1,000 minority rows** under the matched cap.

| Classifier | No-resampling AP | Original UGO AP | Corrected UGO AP |
|---|---|---|---|
| Logistic regression | 0.6089 ± 0.0028 | 0.6077 ± 0.0016 | 0.6074 ± 0.0019 |
| Random forest | 0.7395 ± 0.0057 | 0.7244 ± 0.0052 | 0.7220 ± 0.0039 |
| LightGBM | 0.7428 ± 0.0039 | 0.7302 ± 0.0033 | 0.7262 ± 0.0086 |

Mean ± descriptive seed SD on one shared holdout. Both UGO variants raised mean minority F1/recall at threshold 0.5 but lowered mean AP in every family; no AP improvement or published reproduction is claimed. [The UGO audit](reports/reproduction.md#full-shoppers-ugo-baselines-controlled-development) records paired deltas, secondary metrics, count/ancestry checks and reproducible commands. Machine-readable references: `results/runs/20261003T181637-fabbd21ddfb8/ugo_summary.json` and `ugo_summary.csv`.

### Phase 1 first scoring change — user-retained

`U1_oof_mma` has been implemented and evaluated on Shoppers only. Run `20261004T063713-7b42103e131e` completed all 45 baseline/corrected_current/U1 cells with five seeds and ten tuning proposals. U1 AP means were **0.6092 ± 0.0016** (LR), **0.7206 ± 0.0055** (RF), **0.7319 ± 0.0023** (LightGBM); paired AP deltas versus corrected_current were **+0.001853**, **-0.001363**, **+0.005687**. These mixed shared-holdout results do not choose retention automatically.

Seven Shoppers leakage/manual-gate regressions and the nine-cell training-only smoke passed; full predictions, real-only scoring-fold/preprocessing coverage and ancestry were verified. [Complete U1 metrics and compute](reports/reproduction.md#phase-1-u1-oof-mma--user-retained) are reported alongside [control results and failures](reports/reproduction.md#phase-1-shoppers-controls--user-review-pending). Every seed-level metric is in `results/runs/20261004T063713-7b42103e131e/all_metrics.csv`; all means/SDs are in `metric_summary.csv`, and paired differences are in `paired_metric_deltas.csv`. The user explicitly kept U1_oof_mma; U2–U5 decisions remain pending and no CLIMB evaluation has run. Immutable run snapshots retain their original awaiting-decision state; no metrics were changed or experiments rerun for this decision.

```sh
conda run --no-capture-output -n ml_project python -m benchmarks.run --run \
  --study configs/ablations/stage1_components.yaml \
  --classifiers logistic_regression random_forest lightgbm \
  --methods baseline corrected_current U1_oof_mma
```


### U2 group protection/allocation — user-retained

U2 is isolated against corrected_current; it does not silently combine retained U1 scoring. Real minority groups are built from inner-fit data, then rebuilt on full outer training for final refit. KMeans uses eight clusters capped at distinct coordinates; the graph uses mutual five-nearest-neighbor edges below the real-class 90th-percentile kth-neighbor radius. Five independent 80% real-row subsamples within each class measure incident co-membership Jaccard stability, penalizing splits and mergers. Unsupported/single-coordinate/zero-stability groups cannot receive quota; cross-class neighbor conflict is logged separately, not interpreted as corruption.

Compare stability-adjusted, size-proportional, uniform-group and difficulty-only allocation for both groupings. Stability-adjusted weight is r_g(h_g+0.001)/(n_g+0.001)^group_alpha; h_g is mean within-class percentile difficulty. U2/U2_graph jointly tune alpha=0,0.5,1 across the same ten classifier proposals (cyclic assignment, one sampler/classifier pair per proposal), selecting only on inner validation and replaying the selected alpha for full-training refit. There is no additional Cartesian search or outer-score selection.

Integer quotas use saturated largest-remainder apportionment with eligible-only redistribution. Group caps are floor(2*n_g); each real seed can generate at most four accepted descendants over all rounds. Within each group, UGO retains its top-half difficulty selection, with exhausted seeds excluded. Support/stability never grow with synthetic rows. Proposal and relocation remain eligible same-class real-neighbor operations, **not group-restricted geometry** (that is U3). Generated counts are charged to the source seed group, not a claimed spatial group of the final relocated point. Scoring remains current in-sample MMA; classifier training, original stopping, threshold and split are unchanged. Every group, conflict, quota, cap, actual multiplicity and underspend is auditable.

Runtime proof: 25 group/quota regressions, nine Shoppers leakage/manual-gate regressions, seven existing UGO regressions (including CUDA) and 19 result-contract regressions passed: 60 total. All 36 training-only smoke cells completed in `results/smoke/20261004T083250-b2c2948fb6b9`. Their saved metrics were recomputed; 72 pipeline/48 group audits and four exact real-only group refits passed. Smoke scores are not experimental evidence. The user subsequently kept canonical U2 KMeans/stability, separately evaluated U3 and discarded it. The user subsequently discarded U4 and U5 too, and reaffirmed canonical U2; U1 remains kept and confirmatory evaluation stays closed.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -B -m benchmarks.run --run \
  --protocol configs/benchmarks/shoppers_controlled.yaml \
  --study configs/ablations/stage1_components.yaml \
  --classifiers logistic_regression random_forest lightgbm \
  --methods baseline U0 corrected_current U1_oof_mma U2 U2_graph \
    U2_kmeans_size U2_kmeans_uniform U2_kmeans_difficulty \
    U2_graph_size U2_graph_uniform U2_graph_difficulty
```

Full Shoppers comparison: **180/180 completed**, five seeds, ten common joint proposals, all 1,000-row generating budgets reached and no failed cells. [Every metric, paired delta, group diagnostic, resource and interruption disclosure](reports/reproduction.md#phase-1-u2-group-protectionallocation--user-retained); complete per-seed artifacts are in `results/runs/20261004T084817-f93a994968f7`. AP is mean ± sample SD:

| Variant | LR AP | RF AP | LightGBM AP |
|---|---|---|---|
| Baseline | 0.6089 ± 0.0028 | 0.7395 ± 0.0057 | 0.7428 ± 0.0039 |
| Original UGO (U0) | 0.6077 ± 0.0016 | 0.7244 ± 0.0052 | 0.7302 ± 0.0033 |
| Corrected current | 0.6074 ± 0.0019 | 0.7220 ± 0.0039 | 0.7262 ± 0.0086 |
| Kept U1 OOF MMA | 0.6092 ± 0.0016 | 0.7206 ± 0.0055 | 0.7319 ± 0.0023 |
| KMeans stability (U2) | 0.5854 ± 0.0023 | 0.7178 ± 0.0052 | 0.7281 ± 0.0054 |
| Graph stability | 0.5977 ± 0.0028 | 0.7277 ± 0.0077 | 0.7343 ± 0.0036 |
| KMeans size | 0.6008 ± 0.0027 | 0.7294 ± 0.0055 | 0.7406 ± 0.0021 |
| KMeans uniform | 0.5881 ± 0.0025 | 0.7180 ± 0.0085 | 0.7319 ± 0.0055 |
| KMeans difficulty | 0.5814 ± 0.0031 | 0.7179 ± 0.0058 | 0.7266 ± 0.0042 |
| Graph size | 0.5964 ± 0.0023 | 0.7193 ± 0.0083 | 0.7318 ± 0.0030 |
| Graph uniform | 0.5968 ± 0.0010 | 0.7274 ± 0.0081 | 0.7372 ± 0.0026 |
| Graph difficulty | 0.5972 ± 0.0017 | 0.7274 ± 0.0081 | 0.7372 ± 0.0026 |

Default U2 has mixed results: relative to corrected_current, AP changes are LR -0.0220, RF -0.0042, LGB +0.0019; RF/LGB F1 and recall improve, but LR F1/MCC decline. KMeans-size improves tree AP (+0.0073 RF/+0.0145 LGB) but still loses LR AP/F1. No U2 policy beats baseline AP in any family. Graph uniform/difficulty are prediction/quota/ancestry-equivalent for RF/LGB, not independent confirmations. **The user kept canonical U2 KMeans/stability; no alternative was selected automatically. U1 stays kept.**

### U3 real-reference candidate constraints — user-discarded

U3 is an isolated geometry change against corrected_current, not a combination with kept U1/U2. Connected groups use a real-only mutual-five-NN graph in training-preprocessed numeric coordinates with a 90th-percentile class radius. Partners are eligible, distinct real points in the same component and with an identical entire categorical tuple. Numeric interpolation therefore preserves every valid categorical block; categorical-only or singleton tuple pools cannot interpolate. No conflict flags are restored, and baseline in-sample MMA, pooled-class allocation and stopping remain unchanged.

Every emitted Float32 candidate is checked with Float64 numeric norms against **all real** same/opposing-class training observations: displacement <= eta * nearest-opponent distance at its seed, and nearest-same distance / (nearest-opponent distance + 1e-12) <= rho. These empirical checks are not guarantees of label correctness. U3 halves a failing step at most four times; U3_rejection discards it; U3_relocation tries the nearest eligible compatible real point in the same component, then rechecks both unchanged bounds. The unconstrained class-global relocation remains the corrected_current reference. Nothing is padded, and unequal actual counts invalidate equal-count mechanism claims.

Three joint pairs (eta,rho)=(0.25,0.5),(0.5,1.0),(1.0,1.5) cycle across the same ten classifier proposals (4/3/3 slots), are selected on inner AP only and replayed in the full-training refit; this is not a full Cartesian search. Actual source IDs, components, attempts, failures, repairs, shortening steps, emitted coordinates, duplicates and underspend are saved.

All 133 repository tests pass, including geometric boundary/oracle checks, repair caps, category ordering, native-budget/cache boundaries, group ancestry and Shoppers held-out perturbation. The complete 24-cell training-only CLI smoke at results/smoke/20261005T182550-240211b0ce26 passed independent prediction/96-pipeline/36-constraint checks: 1,440 attempts and 114 accepted points verified. All nine U3 outer smoke cells underspent (0–5 of 40); smoke scores are not study evidence. Its preceding interrupted attempt is preserved with an interruption sidecar and excluded.

Reproduce the historical controlled comparison (Shoppers only; U3 was discarded and is not retained):

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -B -m benchmarks.run --run \
  --protocol configs/benchmarks/shoppers_controlled.yaml \
  --study configs/ablations/stage1_components.yaml \
  --classifiers logistic_regression random_forest lightgbm \
  --methods baseline U0 corrected_current U1_oof_mma U2 U3 U3_relocation U3_rejection
```

Full comparison: **120/120 completed**, five seeds and ten common joint proposals, no failed cells. All 45 U3 outer refits underspent the 1,000-row cap; none is an equal-actual-count mechanism comparison. [Every metric, paired delta, repair, selection, resource and verification](reports/reproduction.md#phase-1-u3-real-reference-constraints--manual-review); artifacts: results/runs/20261005T184048-c25b2afe26ae. AP is mean ± sample SD:

| Variant | LR AP | RF AP | LGB AP |
|---|---|---|---|
| Baseline | 0.6089 ± 0.0028 | 0.7395 ± 0.0057 | 0.7428 ± 0.0039 |
| Original UGO (U0) | 0.6077 ± 0.0016 | 0.7244 ± 0.0052 | 0.7302 ± 0.0033 |
| Corrected current | 0.6074 ± 0.0019 | 0.7220 ± 0.0039 | 0.7262 ± 0.0086 |
| Kept U1 OOF MMA | 0.6092 ± 0.0016 | 0.7206 ± 0.0055 | 0.7319 ± 0.0023 |
| Kept U2 KMeans stability | 0.5854 ± 0.0023 | 0.7178 ± 0.0052 | 0.7281 ± 0.0054 |
| U3 shortening | 0.5906 ± 0.0008 | 0.7215 ± 0.0084 | 0.7231 ± 0.0014 |
| U3 relocation | 0.5874 ± 0.0013 | 0.7310 ± 0.0029 | 0.7233 ± 0.0015 |
| U3 rejection | 0.5994 ± 0.0032 | 0.7314 ± 0.0022 | 0.7294 ± 0.0026 |

Default shortening regresses versus corrected_current in all nine predictive means for each family: AP -0.0167/-0.0005/-0.0030 (LR/RF/LGB), F1 -0.0182/-0.0161/-0.0136 and recall -0.0099/-0.0461/-0.0325. RF relocation/rejection and LGB rejection improve AP, but all U3 F1/recall means are lower; no U3 policy beats baseline AP. Mean generated rows are shortening 994.4/817.4/988.0, relocation 940.4/486.8/897.0 and rejection 370.8/319.2/549.6. Default RF/LGB cell time rises to 529.56/86.38 seconds versus current 116.57/25.84. The 3 h 13 m 39 s run and 1,320 audits verified every prediction, 4.86 million attempts and 333,607 accepted emissions; all 75 reference cells reproduce. **The user discarded U3, including all three repair policies. U1/U2 stay kept; historical code/results are preserved, without combination, U4 or confirmatory progression.**


### U4 stopping/checkpoint selection — user-discarded

U4 changes only stopping against corrected_current, not a combination with kept U1/U2; discarded U3 geometry is excluded. Welch p-values remain diagnostic but no longer terminate generation. The prespecified budget fractions [0, 0.25, 0.5, 1] mean 0/250/500/1,000 additional rows in this Shoppers study. U4 evaluates one trajectory at those checkpoints; U4_fixed_grid runs each budget independently. Real support filtering, in-sample MMA, pooled class scheduling, real interpolation and ordinary relocation remain unchanged. Checkpoint batch clipping changes retraining cadence. Multiclass checkpoint caps grow monotonically by incremental remaining-deficit apportionment; independent fixed budgets use original-deficit apportionment. Native mode retains the original phi cap instead of becoming class balancing.

Each of the same ten learner proposals competes at every reached checkpoint (up to 40 scored settings, versus ten reference settings); zero augmentation is explicit. Small integer-budget duplicates are coalesced. The real inner validation objective chooses the checkpoint and learner; Shoppers uses AP and a fixed 0.5 threshold. Exact ties prefer fewer actual synthetic rows, then a smaller requested budget, then the earliest learner proposal. Invalid-support/iteration/native caps can underspend; there is no padding or fabricated unvisited checkpoint. The chosen budget fraction, checkpoint cadence and selected epoch sequence are replayed on outer-training-only preprocessing/support; no outer/test score selects them. Zero refit trains the real-only baseline once. Saved checkpoint probabilities permit independent validation-metric recomputation.

Verification before the full comparison: all 149 repository tests pass, including real Shoppers held-out perturbations, independently refitted predictions, zero ties, capped refit, native/multiclass budgets and exact prefix replay through an Alabama-paradox grid. The actual 21-cell training-only smoke at results/smoke/20261006T033538-db0233b6c428 completed: all 21 prediction sets, 111 pipeline audits and 72 checkpoint metric sets were independently checked (2,331 numeric metric values; 292 classifier fits). Smoke scores are not study evidence. These are historical U4 verification results. The user subsequently excluded both U4 policies and authorized U5 testing with kept U1/U2 only; the user subsequently discarded U5 and reaffirmed canonical U2. W5/R5/confirmatory remain unavailable.

Reproduce the historical controlled U4 comparison (Shoppers only; U4 is excluded from the selected component set):

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -B -m benchmarks.run --run \
  --protocol configs/benchmarks/shoppers_controlled.yaml \
  --study configs/ablations/stage1_components.yaml \
  --classifiers logistic_regression random_forest lightgbm \
  --methods baseline U0 corrected_current U1_oof_mma U2 U4 U4_fixed_grid
```


Full run `results/runs/20261006T034029-0513bf17dc6c` completed **105/105 cells** (seven variants × three families × five seeds); all 75 reference cells reproduce. Every outer prediction and all 1,200 checkpoint probability sets were verified, along with 1,605 pipeline audits, 105 canonical identities and exact source/split/proposal provenance. Whole-run wall time: 1 h 40 m 03 s; 4,984 classifier fits. Mean ± sample SD AP:

| Variant | LR AP | RF AP | LightGBM AP |
|---|---:|---:|---:|
| baseline | 0.6089 ± 0.0028 | 0.7395 ± 0.0057 | 0.7428 ± 0.0039 |
| U0 | 0.6077 ± 0.0016 | 0.7244 ± 0.0052 | 0.7302 ± 0.0033 |
| corrected_current | 0.6074 ± 0.0019 | 0.7220 ± 0.0039 | 0.7262 ± 0.0086 |
| U1_oof_mma | 0.6092 ± 0.0016 | 0.7206 ± 0.0055 | 0.7319 ± 0.0023 |
| U2 | 0.5854 ± 0.0023 | 0.7178 ± 0.0052 | 0.7281 ± 0.0054 |
| U4 | 0.6088 ± 0.0015 | 0.7395 ± 0.0057 | 0.7428 ± 0.0039 |
| U4_fixed_grid | 0.6077 ± 0.0018 | 0.7395 ± 0.0057 | 0.7428 ± 0.0039 |

Default U4 AP changes versus corrected_current are +0.0014/+0.0175/+0.0166 (LR/RF/LGB), but minority F1 changes -0.0302/-0.0282/-0.0130 and recall -0.0450/-0.0906/-0.0660. Fixed-grid LR has AP +0.0004, F1 -0.0109 and recall -0.0157. **All 20 U4 tree refits select zero and exactly reproduce baseline learner/predictions/metrics**: their AP recovery is avoidance of augmentation, not better generated points. LR uses mean 600 rows for checkpoint selection and 850 for fixed-grid; all selected budgets are achieved, without padding. Default RF/LGB time is 4.78×/9.74× baseline for identical final predictions; fixed-grid is 10.17×/25.89×. This is a variable-budget, unequal-selection-effort procedure comparison, not an equal-count/equal-compute mechanism claim. [Every metric, paired delta, operating point, selection and resource](reports/reproduction.md#phase-1-u4-stopping-selection--manual-review). **The user excluded both U4 policies. U1/U2 remain kept, U3/U4 discarded; U5 was subsequently tested and discarded too; canonical U2 is reaffirmed and confirmatory evaluation remains closed.**


### U5 kept-components combination — user-discarded

U5 (ugo_combined) composes kept U1_oof_mma and canonical U2 KMeans/stability; it excludes discarded U3 and both U4 stopping policies. Three real-only OOF folds fit their own raw-row preprocessing and the proposed learner, producing fixed MMA scores before generation. Real-only KMeans/subsampling stability controls support and bounded group/seed allocation using those scores. Corrected UGO interpolation, category donors, ordinary relocation and Welch stopping remain unchanged; no constraint envelope or checkpoint/fixed-grid stopping enters U5. Inner and outer OOF/group states are training-only and separately rebuilt.

The same ten learner proposals jointly cycle group_alpha=0/0.5/1 (4/3/3 settings), as canonical U2; inner AP chooses the learner/alpha and replays them for outer refit at threshold 0.5. Three extra OOF fits per proposal/refit are counted, not hidden as equal compute. Requested/actual generation counts and any underspend remain explicit. The user discarded U5 after full metric review and reaffirmed canonical U2; U1 remains kept and U5 evidence/code is preserved for historical reproduction only; W5/R5, configuration freeze and confirmatory evaluation remain closed.

Verification: all 159 repository tests passed, including independent real Shoppers OOF classifiers/preprocessing, quota mathematics, held-out perturbations, shared-cache isolation, independently refitted outer predictions, real ancestry/caps, restored conflicting islands and unchanged Welch stopping. The actual 18-cell training-only smoke in results/smoke/20261006T055011-c27595b0db9f completed: all 18 predictions/360 numeric metric values, 72 pipeline audits, 24 OOF audits, 24 group audits (12 combined), three joint selections and 252 classifier fits were checked. No official outer-test rows entered the smoke; smoke scores are not study evidence.

Reproduce the historical controlled U5 comparison (Shoppers only; U5 is excluded from the selected component set):

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -B -m benchmarks.run --run \
  --protocol configs/benchmarks/shoppers_controlled.yaml \
  --study configs/ablations/stage1_components.yaml \
  --classifiers logistic_regression random_forest lightgbm \
  --methods baseline U0 corrected_current U1_oof_mma U2 U5
```

Full run `results/runs/20261006T060142-8e042c6ffcf0` completed **90/90 cells**; all 75 fresh reference cells reproduce. All 90 predictions/1,800 numeric metrics, 990 pipeline audits, 330 OOF and 330 group audits (165 combined), actual joint choices and source/split/index provenance were verified. U5 exactly reuses U1's 150 inner OOF score sets and U2's 165 real group states. Whole-run wall time: 1 h 15 m 59 s; 4,290 classifier fits and 825,000 search/refit emissions. Mean ± sample SD AP:

| Variant | LR AP | RF AP | LGB AP |
|---|---:|---:|---:|
| baseline | 0.6089 ± 0.0028 | 0.7395 ± 0.0057 | 0.7428 ± 0.0039 |
| U0 | 0.6077 ± 0.0016 | 0.7244 ± 0.0052 | 0.7302 ± 0.0033 |
| corrected_current | 0.6074 ± 0.0019 | 0.7220 ± 0.0039 | 0.7262 ± 0.0086 |
| U1_oof_mma | 0.6092 ± 0.0016 | 0.7206 ± 0.0055 | 0.7319 ± 0.0023 |
| U2 | 0.5854 ± 0.0023 | 0.7178 ± 0.0052 | 0.7281 ± 0.0054 |
| U5 | 0.5854 ± 0.0015 | 0.7176 ± 0.0071 | 0.7296 ± 0.0043 |

U5 largely tracks kept U2: AP changes -0.000018/-0.000192/+0.001461 (LR/RF/LGB), while minority F1 and recall are slightly lower in every family and diagnostic cost increases 894/1,406/1,252. Versus kept U1, AP falls in all families while recall rises; no resampling still has the highest mean AP in every family. All generation variants emit 1,000 outer rows without underspend. U5 mean cell times are 17.59/161.95/34.63 seconds (about 1.02×/1.30×/1.23× U2), with 385 versus 220 classifier fits per family over five seeds. This is equal actual generation count, not equal compute. [Every metric, paired delta, operating point, joint selection and resource](reports/reproduction.md#phase-1-u5-kept-components-combination--manual-review) is reported and independently checked. **The user discarded U5 and reaffirmed canonical U2. U1 remains kept; U3/U4/U5 are excluded. No final configuration freeze or confirmatory progression is selected automatically.**


### H1 HDBSCAN support-filter experiment — user-discarded

H1 adds a frozen Python `hdbscan==0.8.44` non-noise gate **after canonical U2 neighbor handling and relocation**. It does not change U2 scoring, real-only KMeans/stability groups, interpolation, pre-generation noise detection, classifier family or Welch stopping, and does not include U1/U3/U4/U5, H2 or FlowGMM. Real minority rows labeled clustering noise remain in classifier training. Membership strength is a model diagnostic, not a label-correctness probability; no usable clusters explicitly reject all candidates.

The training-fitted representation retains standardized continuous features and scales complete categorical one-hot coordinates by `1/sqrt(2)`: each categorical mismatch contributes one unit to squared Euclidean distance. The detector is fitted on real minority training rows only with `prediction_data=True`; candidates use `approximate_predict` on final relocated coordinates. It never refits on synthetic rows.

The independent study selects the same ten canonical U2 learner/group-alpha proposals first, then locks the selected learner and alpha while validating all 12 pairs `min_cluster_size=[5,10,20,40]` × `min_samples=[1,5,10]`. The detector search is extra tuning, not an equal-compute comparison. H1 consumes each planned proposal quota once; rejection never triggers quota refill. Accepted source-group counts are logged separately. An all-rejected round terminates explicitly and skips unchanged final retraining. The fixed 1,000-proposal and 20-round caps can therefore produce fewer than 1,000 accepted rows.

`benchmarks.h1_pool` first saves original/post-relocation U2 proposals, tunes H1 on an immutable inner pool, and compares H1 with five random-thinning repeats matched per original U2 source group. The iterative study separately tunes H1 on iterative inner trajectories; its proposals can diverge after classifier retraining. Pool and iterative selection policies/results must not be conflated. Group recall uses frozen nearest-real-minority assignments and reports support counts; these are model-based groups, not guaranteed true subpopulations.

The user explicitly discarded H1 after reviewing these comparisons and authorized F1 testing separately. `configs/ablations/stage1_h1.yaml` and its implementation remain available for explicit historical reproduction only. Full comparisons used LR/RF/LightGBM, seeds 0–4, unchanged outer/inner split policy and fixed 0.5 classifier threshold. Historical U5 is a comparison reference, not a retained component.

```sh
# Saved-pool mechanism comparison first; then full iterative comparison.
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.h1_pool --run \
  --classifiers logistic_regression random_forest lightgbm

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.run --run \
  --protocol configs/benchmarks/shoppers_controlled.yaml \
  --study configs/ablations/stage1_h1.yaml \
  --classifiers logistic_regression random_forest lightgbm \
  --methods baseline U0 corrected_current U2 H1
```

Use `--smoke` instead of `--run` for training-only integration checks. Historical H1 candidate logs preserve proposal IDs, real parents, original U2 groups, original/final coordinates, cluster assignments, membership strengths and rejection reasons. Saved predictions support recalculating every metric; use pandas `float_precision="round_trip"` so tied scores remain exact. See [the complete historical H1 report](reports/h1.md). H1 was discarded by the user; it is not part of F1 or a retained Stage 1 configuration. Iterative run `20261007T153639-17e4169cca5e` completed 75 cells and saved-pool run `20261007T151450-ab30c8827f99` completed 120 cells.

### F1 FlowGMM compatibility-only experiment — user-retained

F1 replaces **both** canonical U2's post-generation neighbor detector and relocation with a frozen FlowGMM gate on **original** proposals. U2 seed scoring, protected KMeans/stability allocation, interpolation, pre-generation noise detection, downstream learner and Welch stopping remain unchanged. N-reject is the specified diagnostic control: the original neighbor detector rejects conflicting original proposals without relocation. Neither handler refills rejected quotas; an all-rejected round terminates explicitly and avoids unchanged final retraining. H1 is discarded; F2/H2 and U1/U3/U4/U5 are not composed into F1.

`flowgmm_tabular_v1` is an independently written, source-hashed PyTorch shared RealNVP implementation with two class-associated learned diagonal Gaussian latent components, referencing the [FlowGMM paper](https://proceedings.mlr.press/v119/izmailov20a.html). It is not vendored upstream code or a plain GMM substitution. Its fixed six alternating affine-coupling layers, width-64 two-hidden-layer networks, 60 epochs, batch size 256 and Adam learning rate .001 fit supervised class-balanced likelihood on **both real classes only**. Private numeric imputation/standardization and full categorical one-hot encoding fit only detector-fit rows. Full one-hot uniform dequantization avoids treating arbitrary category codes as continuous densities; evaluation jitter is content-keyed and shared between classes. Numeric counts remain continuous, matching unchanged U2 interpolation. Densities are continuous proxies in this representation, not exact discrete probabilities or correctness guarantees.

A stratified 20% of real inner-fit rows is reserved as calibration (seed+7919); the other 80% fits the detector. Original inner validation and outer test never fit its preprocessing or weights. Calibration is explicitly unused by compatibility-only F1. These same detector roles and frozen core are retained for outer replay; only the inverse classifier-coordinate bridge changes. Every downstream classifier still receives all the same real inner-fit/outer-training rows as U2. Unknown detector categories, invalid classifier-domain features and nonfinite scores receive distinct rejection reasons.

Equal priors give `delta_y = log p(x|y) - log p(x|opposing)` and diagnostic `sigmoid(delta_y)`. Acceptance is computed stably as `delta_y >= log(tau/(1-tau))`, cancelling the shared flow Jacobian analytically. There is **no F2 likelihood/support threshold**. The same ten joint U2 learner/group-alpha proposals are selected first; their learner and alpha are locked while tau=0.5/0.6/0.7/0.8 competes on real inner validation AP. N-reject uses that same locked configuration without detector tuning. Fixed architecture and extra tau selection are disclosed, not claimed equal compute.

`benchmarks.f1_pool` first compares immutable U2 source proposals with U2 post-relocation replay, N-reject original-coordinate replay and F1 original-coordinate replay. It saves real parents, original source groups, both coordinate versions, decisions/likelihoods, checkpoints and hashes. Iterative evaluation is separate: proposals may diverge after downstream retraining. Both surfaces report every predictive/operating-point/confusion metric, actual acceptance, underspend, original-group support/recall, duplicates and detector/resource timing. Historical U5 is a reference only.

`configs/ablations/stage1_f1.yaml` restricts this development study to Shoppers. The user explicitly kept F1 and discarded F2; Stage 1 is not frozen and Stage 2/CLIMB/confirmatory evaluation remain closed. See [the complete F1 comparison](reports/f1.md).

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.f1_pool --run \
  --classifiers logistic_regression random_forest lightgbm

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.run --run \
  --protocol configs/benchmarks/shoppers_controlled.yaml \
  --study configs/ablations/stage1_f1.yaml \
  --classifiers logistic_regression random_forest lightgbm \
  --methods baseline U0 corrected_current U2 N_reject F1
```

Use `--smoke` for training-only checks. All 196 tests passed. The full Shoppers comparison completed 90 iterative and 60 immutable-pool cells; independent replay checked 3,150 stored numeric metric values and 165,000 candidate decisions, plus 315 historical U5 metric values. All 60 fresh baseline/original-UGO/corrected-UGO/U2 prediction files reproduce their historical references exactly. The report includes all 22 predictive/operating-point/confusion metric fields, paired deltas, group support/recall, duplicates, quotas and scoped compute costs. F1 was explicitly retained by the user after reviewing F2.

### F2 likelihood-plus-compatibility experiment — discarded by user

F2 reuses the **exact completed F1 checkpoint, private preprocessing, real fit/calibration/validation roles, learner, group alpha and selected tau**. It adds an inclusive intended-class likelihood cutoff including the flow Jacobian, estimated from the reserved real calibration rows. Quantile candidates are explicit off (`None`/`null`), .01/.05/.10; off is not q=0. Only real validation AP selects q, with exact ties preferring off then smaller q. Neither detector fitting nor F1 tau selection is repeated. Rejected original proposals never relocate or refill quotas; all-rejected rounds terminate without unchanged retraining.

The quantile convention is NumPy linear interpolation with an explicit extended-real endpoint rule. Categories absent from the fitted vocabulary have zero closed-vocabulary support: their calibration likelihood is `-inf`, and **all** original real calibration rows remain in the class counts/quantile denominator. Their IDs, classes and score kinds are audited; NaN/numerical/model failures still raise. Unknown candidates remain rejected independently. Finite minority calibration counts and the rare-group acceptance/recall audit are reported; this is a model-based continuous-density proxy, not guaranteed label correctness or true support.

The immutable-pool study first reuses the saved F1 proposal pool/model/tau, evaluates support-off and every fixed q, and compares selected F2 with five uniform thinnings of **F1-accepted original proposals**. Every thinning matches selected-F2 counts separately per **original U2 group**, for inner and outer pools. Repetition means are taken within classifier/sampler seeds before five-seed summaries; thinning repeats are not independent test folds. Iterative evaluation is separate, with fixed q diagnostics and support-off exact F1 reproduction; post-retraining proposals may diverge.


```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.f2_pool --run \
  --source-run results/runs/20261007T172857-728d008031d6

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.run --run \
  --study configs/ablations/stage1_f2.yaml \
  --classifiers logistic_regression random_forest lightgbm \
  --methods F2 F2_support_off F2_q001 F2_q005 F2_q010
```

Both commands accept `--smoke` for the existing training-only source. The iterative runner accepts `--f1-source-run`; source protocol, dataset/split identities, learner proposals, model roles and generation settings must match.

All 219 tests passed; the real CLIs completed 45 smoke cells, independently checked against 945 metric values and 4,008 candidate decisions. The full five-seed pool run `20261008T115909-d5b24e6dcbf9` completed 150 cells before iterative run `20261008T120348-bd801301d7cb` completed 75. Independent full replay recomputed all 4,725 numeric metric values, 570,000 candidate decisions and 1,200 thinning-group count comparisons. All 22 metric fields, SDs, pairings, group/tail/acceptance and resource comparisons with F1, U2, original/corrected UGO and historical U5 are in [reports/f2.md](reports/f2.md).

The user kept F1 only and discarded F2. F2 code, commands and immutable results remain for historical reproduction; F2 is excluded from the retained configuration. H1 stays discarded, Stage 1 is not frozen and broader evaluation stays closed.

### U5+F1 composition — retained by user

The user authorized a separate Shoppers-only test of **existing U5 (fixed real-only three-fold U1 OOF MMA plus canonical U2 KMeans/stability allocation) + compatibility-only F1**. `U5_F1` uses the explicit `ugo_combined_f1` method. Select the canonical U5 learner/group alpha over the same ten proposals first, then select F1 tau from .5/.6/.7/.8 on real inner-validation AP. Standalone F1 keeps its canonical U2 selection. The same pinned, fit-only frozen flow/roles are shared across both F1 variants, inner/outer replay and classifier families; reserved calibration remains unused and F2 filtering is disabled.

Both the fixed OOF scores and real groups control proposals; the F1 gate scores their original coordinates, replaces post-generation neighbor detection and relocation, never refills rejected quotas, and skips unchanged retraining after zero acceptance. Every OOF model trains on real fold-training rows with fold-fitted preprocessing. Each tau trial and outer trajectory fits its scoring models independently, so their actual classifier-fit cost is charged rather than claimed equal to F1.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.run --run \
  --study configs/ablations/stage1_u5_f1.yaml \
  --classifiers logistic_regression random_forest lightgbm \
  --methods baseline U0 corrected_current U2 U5 F1 U5_F1
```

Use `--smoke` for training-only integration checks. All 223 tests passed; the real CLI completed 21 smoke cells and full Shoppers run `20261008T130740-f85501108e91` completed 105/105 cells. Independent replay recomputed 2,205 numeric metric values, reproduced all 90 historical reference predictions/learner selections exactly and passed 40,386 checks. [Complete U5+F1 comparison](reports/u5_f1.md) reports all 22 metric fields/SDs/paired differences, original-group support/recall, proposal ancestry, acceptance/underspend, duplicates and measured compute.

The user retained U5+F1 after the four-way comparison. Standalone U5/F2/H1 remain discarded and F1 remains kept. A separate U2 learning-benefit experiment is authorized; Stage 1 is not concluded or frozen, and broader-evaluation gates remain closed.

### U2 learning-benefit candidate selection — utility discarded

A separate Shoppers experiment uses the actual downstream logistic regression and unchanged U2 KMeans/stability quotas, anchors, interpolation, neighbor relocation, caps and stopping. Reserve stratified training guidance from inner fitting; preprocessing and all generated ancestry remain FIT-only, with separate original validation/test and no outer refit. No FlowGMM/HDBSCAN or U1 scoring is combined into this experiment.

Generate K=5 proposals per distinct planned anchor, score full-parameter unregularized CE gradient dot products at frozen checkpoints, deduplicate final coordinates identically and select random/best-per-anchor candidates within original group quotas. Negative scores are allowed. One-per-anchor feasibility can underspend original repeated-anchor quotas; no regeneration or quota override is used. Paired initial and later-checkpoint comparisons match counts; live iterative pools may diverge.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.learning_benefit --run
```

Use `--smoke --seeds 0` for training-only integration. Study: `configs/ablations/stage1_learning_benefit.yaml`. The five-seed study completed 30/30 records; all 241 tests passed. Independent replay passed 9,689 checks, including all 22 benchmark fields plus ordinary/balanced CE, full-gradient SGD verification, frozen state, preserved U2 quotas/ancestry and equal-weight parent controls. [Complete report](reports/learning_benefit.md) includes all metrics/SDs, paired same-pool and live-trajectory differences, group/count/duplicate audits and measured compute.

The SGD estimate is accurate, but utility loses mean validation/test AP against random; parent duplication is competitive or better, and utility batches are more tightly spaced. This does not support utility retention on the current AP-based comparison. U5+F1 remains retained; the user has discarded pool-utility. Pool-random+F1 is the next experiment; Phase 1 conclusion remains the user’s decision. Phase 1 is not frozen and broader evaluation remains closed.

### U2 validation-guided original-access — utility discarded

This separate, user-authorized Shoppers experiment restores historical U2 data access: original inner fitting and validation selection, then full outer-training preprocessing, generation and classifier refitting. Original validation is reused as guidance. It is excluded from inner fitting/generation but deliberately included in outer fitting/generation, so guidance metrics are training/selection feedback, not independent validation. Outer test never supplies tuning or generation information.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.validation_guided --run
```

The runner evaluates actual LR, Random Forest and LightGBM across five matched seeds. It freshly runs original UGO, U2, original standalone U5 and retained U5+F1 references using their canonical tuning/refit conventions (the retained F1 detector keeps its historical `auto` device, resolving to CUDA on this host), then evaluates untouched U2, K=5 pool-random and K=5 pool-utility under locked U2 settings. The initial pools/checkpoints/counts match; later native-retrained trajectories can diverge. Group quotas, caps, generator, relocation, stopping and fit cadence stay unchanged; one-per-anchor shortfalls are logged without regeneration.

LR utility uses the actual full-weight-and-bias CE gradient. For RF/LightGBM, the user explicitly authorized a separately fitted LR surrogate, refreshed on each native checkpoint’s exact current training prefix. Native tree models still determine U2 anchor difficulty and perform downstream retraining. Gradient/SGD verification describes only the LR scorer; it is not a native tree-benefit guarantee. LR scorer hyperparameters come from same-seed canonical LR-U2 selection.

Use `--smoke --seeds 0 --tuning-budget 1` for training-only integration. `configs/ablations/stage1_validation_guided.yaml` records deliberate guidance overlap and scorer authorization. Artifacts retain native/scorer checkpoints, training prefixes, original/final pools, masks, equal-weight parent controls and all benchmark metrics plus ordinary/balanced CE. U5+F1 stays retained; standalone U5 remains a comparison reference, not reinstated. Pool-utility is discarded by the user; pool-random+F1 retention is pending and Phase 1 remains open/unfrozen. See [the full validation-guided report](reports/validation_guided.md) for all 24 metric fields, paired seed differences, parent controls and compute. The audited published bundle is `results/runs/20261009T153348-af844bdecf5a`; it preserves unchanged source t…

### U2 pool-random + F1 — retention pending

The user discarded pool-utility. This Shoppers experiment keeps native LR, Random Forest and LightGBM anchor difficulty and refitting; it does not fit gradient scorers or LR surrogates. Expand each distinct planned U2 anchor to five proposals, deduplicate using the existing final-coordinate convention, choose randomly under unchanged group quotas and one-per-anchor caps, then apply retained F1 **only to the chosen candidates**. F1 replaces neighbor relocation; rejected choices consume attempts and are not rescued by another proposal or refilled. Nominal budget, fitting/preprocessing/generation access and once-per-round retraining remain unchanged; actual accepted counts and quota shortfalls are reported.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.pool_random_f1 --run
```

Use `--smoke --seeds 0` for the actual three-classifier, training-only CLI check. The full experiment locks same-seed canonical U2 learner/group settings, selects only the four retained F1 thresholds on original inner validation, then performs the original outer refit. Previously audited UGO/U2/U5/U5+F1 reference predictions are explicitly imported with source provenance; plain pool-random is freshly run and must reproduce its historical predictions exactly. The frozen original F1 checkpoint retains `auto`/CUDA on this host. See `configs/ablations/stage1_pool_random_f1.yaml`. U5+F1 stays retained, the new combination is pending your decision, and Phase 1 is not frozen. [The full report](reports/pool_random_f1.md) includes all 24 metric fields for six final methods, every paired seed difference, dose/coverage/rejection counts and comparable new-trajectory runtime. The five-seed bundle `results/runs/20261009T170013-d5be50bc2107` passed 770,287 independent checks; all 15 native random controls reproduce historical predictions exactly.

### Flow-latent generation with U2 random pools — retention pending

This experiment changes generation geometry rather than adding F1 rejection. The primary pair uses **identical real same-class, same-U2-group parent rules**: straight-line input interpolation versus interpolation of frozen Flow latents followed by the true inverse. Existing U2 partners are not group-restricted, so historical pool-random is an additional context reference, not the geometry-isolating control. Initial anchors/partners/interpolation coefficients match; later native-retrained trajectories can diverge.

Parent representations use the existing deterministic dequantization and are cached once per seed/inner-or-outer role, shared by both geometries and all three native classifiers. Numeric inverse coordinates return through the detector scaler and original classifier preprocessing. Categorical values retain U2’s partner-copy rule: this is **projected mixed-feature latent interpolation**, not a continuously invertible path through discrete categories. Unknown/nonfinite parents and invalid decoded numeric domains are ineligible before the unchanged neighbor relocation; there is no density/class-label rejection, clipping, fallback or forced regeneration.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.flow_geometry --run --seeds 0 1 2 3 4
```

Use `--smoke --seeds 0` for actual all-three-classifier development-only integration. `configs/ablations/stage1_flow_geometry.yaml` keeps canonical U2 native settings, original data access, K=5 quotas/caps, final targets and retraining/stopping fixed. The six previously tested outer controls retain explicit reference provenance. U5+F1 remains retained, pool-random+F1 and Flow geometry remain pending, and Phase 1 is open/unfrozen. [Full five-seed results](reports/flow_geometry.md) include all 24 metrics for eight final methods, paired differences, raw-domain failures, relocation survival, actual counts and comparable fresh runtime. The 150-record bundle `results/runs/20261009T184911-73fbcd66cdde` passed 1,019,888 independent checks; RF/LightGBM improve AP but lose threshold recall and worsen balanced CE/error cost, while LR loses AP. Accepted counts are near, not exactly equal to, the target.

### TabDiff conditional inpainting — controlled experiment

The latest user specification defines **current U5 as canonical U2 only, without U1 OOF scoring**. Historical `U5` and `U5_F1` bundles actually used fixed three-fold OOF scoring; they remain explicitly labelled historical, not silently renamed. `configs/ablations/tabdiff_controls.json` locks audited genuine U2 native learner/group settings and the existing retained F1 checkpoint and per-cell thresholds. Existing iterative implementations, configurations and result bundles are unchanged.

`benchmarks.tabdiff_experiment` compares one round of original U2 interpolation (A), TabDiff feature-block inpainting near25% (B) and near50% (C), using identical real anchors, original U2 quotas, a common1000-proposal cap and native LR/RF/LightGBM. Domain blocks keep each count/duration pair, the two rate fields and each original categorical field intact; C nests B. Nearest feasible integer mask-size ties are uniform before choosing a compatible block subset. All arms use the user-approved ties-to-even count rounding before unchanged F1 filtering; raw negative/nonfinite values, unknown categories, rate violations and nonzero-duration/zero-count records are rejected, never clipped or rescued. Per-group minimum feasible accepted counts match all arms without refill; exact/near duplicates are diagnostic only, preserving the existing acceptance policy. This common count decoder distinguishes A from historical continuous-count references.

The actual official TabDiff backbone trains on both real classes with a joint categorical label, training-only numeric quantiles and categorical encodings. A frozen checkpoint is shared across all three native classifiers and both masks within each split role. Inpainting uses noisy known coordinates at the actual Euler/Heun diffusion times, an anchor-independent missing-coordinate prior, and exact final restoration of unmasked fields; no TAP policy, utility ranking or surrogate classifier is used. Inner generators exclude validation/test. Each classifier selects B/C using inner-validation AP only; then a separate final generator refits on real outer training, including the former inner-validation rows whose role is now training. Outer test never participates in fitting, generation or selection.

The official source is an explicit pinned prerequisite:

```sh
git clone https://github.com/MinkaiXu/TabDiff .cache/references/TabDiff
git -C .cache/references/TabDiff checkout 5ecdb3356261aea72716cc9a779f31d7ad083bf4
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.tabdiff_experiment --smoke --seeds 0
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.tabdiff_experiment --run --seeds 0 1 2 3 4
```

`configs/ablations/stage1_tabdiff.yaml` fixes the full official8000-epoch recipe, final EMA checkpoint and50 reverse steps; the development-only smoke uses32 epochs and never reads the official test. Artifacts retain raw/decoded proposals, masks, parent IDs, quotas, RNG streams, frozen-checkpoint hashes, matched selections, native models/predictions, all24 metrics and separately accounted training/sampling/filtering/native-fitting time. Shared-row plus matched-seed bootstrap intervals describe the realized development holdout, not independent splits or unseen-data superiority. TabDiff remains experimental: no automatic retention, iterative rollout, Stage1 freeze or broader-benchmark progression.


Five matched scientific seeds completed in `results/runs/20261011T154542-1b40c4b555c4` (165 records), following the actual three-classifier smoke. All 283 tests/307 subtests passed; independent verification passed 236,159 checks including 120 native refits and 990 live denoiser-conditioning evaluations. [Complete TabDiff comparison](reports/tabdiff.md) reports all 24 metrics/seed SDs, per-seed paired differences, shared-row/matched-seed uncertainty, checkpoints/configurations, accepted counts/shortfalls/duplicates and separately accounted runtime. Validation-selected TabDiff improves mean AP in every family, but only LR has a positive conditional uncertainty interval; LR minority F1/recall/MCC decline, and tree-model intervals include zero. This is a development signal, not automatic retention or authorization for a general iterative rollout.


### U2 + TabDiff without F1 — controlled replay

`benchmarks.tabdiff_u2` removes only F1 from the verified fixed-round trial. It replays complete pre-F1 proposal pools and the same frozen real-only TabDiff checkpoints, retaining original real U2 anchors/quotas, masks, classifier preprocessing/native parameters, common count rounding and schema validity. Relocation remains disabled as in the preceding controlled trial; A is therefore a schema-only original-generator control, not an untouched iterative U2 replay. No density/class-compatibility scores, utility rankings or additional learned filters participate in acceptance.

New per-group counts match across A/B/C using schema-valid candidates alone. Original quota shortfalls are logged, never refilled. The no-F1 batches receive fresh native LR/RF/LightGBM fits; B/C is selected again on no-F1 inner-validation AP before outer evaluation. Generator training is independent of mask strength/acceptance; reused outer checkpoints were fitted only on real outer-training data after its former-validation role conversion. No heldout feedback changes a checkpoint or chooses the new mask.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.tabdiff_u2 --smoke --seeds 0
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.tabdiff_u2 --run --seeds 0 1 2 3 4
```

`configs/ablations/stage1_tabdiff_u2.yaml` records this experimental policy. Actual incremental replay/refitting time is separate from the already-paid generator training and original sampling time. Historical UGO/U2/U2+F1 and prior TabDiff+F1 predictions are reference contexts with different accepted counts, not matched-count gate ablations. Retained U2+F1 and the unfrozen Phase1 gate are unchanged.

The no-F1 five-seed replay completed in `results/runs/20261012T050806-2ad08eceaf20` (225 current/reference records). The actual smoke and scientific bundles independently passed six/30 cells and 18/90 fresh native refits respectively; every accepted row, per-group count, preserved field, native prediction and all24 metrics were checked. All 283 tests/307 subtests passed. This remains a one-round development experiment, not an iterative-pipeline change or a decision to retain TabDiff.

[Complete no-F1 comparison](reports/tabdiff_u2.md) includes all24 metrics, per-seed decisions, accepted counts/shortfalls, diagnostics, checkpoints, runtime and paired shared-row/matched-seed uncertainty. Only LR has a positive conditional AP interval against A; RF/LightGBM intervals include zero. Prior F1 comparisons are descriptive because counts and selected masks differ.

### Synthetic downweighting — native four-classifier trials

`benchmarks.downweight` runs Shoppers U2+F1, plain U2 and U2+random-pool without F1, with native LR/RF/LightGBM/MLP. Future controlled Shoppers defaults include MLP; balanced RF remains explicitly available but is not a default or part of this trial, because bagging is deferred. Historical experiment cohorts and source bundles are not rewritten.

All real rows have weight1; every accumulated accepted synthetic has weight0.25 or0.5. Each method has a freshly fitted unit-weight control, and each classifier has genuine real-only fits plus selected-batch anchor-duplication controls with equal added row count and weight. Lock active native U2/group settings and retained F1 weights/thresholds for the original three families; choose new MLP settings and F1 threshold using only real inner validation before locking them across weights. Select synthetic weight on inner-validation AP; ties prefer0.5.

Keep real-only preprocessing/groups/parents, generator/post-handler, quotas,1000 attempt/augmentation target,20-round cap, native retraining and Welch stopping rules. Different weighted native prefixes can change later anchors, accepted counts and stop times; log that divergence rather than force count equality. Initial full K5 pools or one-per-anchor proposal streams must match their unit controls. Random pools use no F1, utility, rescue or refill.

MLP uses the existing two-hidden-layer ReLU/dropout network, full-batch RMSprop and normalized weighted CE, fixed160 epochs per fresh fit (20 for smoke), on CUDA when available. No validation rows or synthetics fit preprocessing/F1; validation is used only for setting selection, not fitting neural weights. Outer fitting deliberately includes former inner-validation rows after settings lock; outer-test predictions alone are held out.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.downweight --smoke --seeds 0
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n ml_project python -m benchmarks.downweight --run --seeds 0 1 2 3 4
```

`configs/ablations/stage1_downweight.yaml` fixes this user-authorized development-only experiment. Acceptance, ancestry, original groups, prefix weights/predictions, actual native fits, effective synthetic mass and runtime are saved. No downweight retention, bagging run, Phase1 freeze or broader-dataset progression is automatic.

Five matched seeds completed in `results/runs/20261012T120400-c73514825f95` (400 records; four native classifiers). The actual smoke and full invocation passed independent verification; the full proof includes 1,352,609 checks, 1,581 prefix and 470 final/canonical native refits, including fixed-epoch native MLP states. All 294 tests/307 subtests passed. [All 24 metrics, paired confidence intervals, counts, effective weights, trajectories and runtime](reports/downweight.md) are published. Mean AP and ordinary CE improve over corresponding unit-weight augmentation in every family/classifier, but recall falls and balanced CE worsens; real-only AP remains higher in all four families. These are conditional shared-holdout development results, not an automatic retention decision.

The executed [all-experiments notebook](reports/experiment_results.ipynb) includes downweighting and earlier methods, with AP, minority F1, precision, recall, MCC, ROC AUC, balanced accuracy and cross-entropy wherever recorded (unavailable historical metrics are marked —). Each experimental approach has original-versus-added-row figures for **linear, moons and circles**, plus held-out synthetic metrics: 61 figures covering 183 dataset–approach runs. These clean, single-seed illustrations are explicitly separate from Shoppers results; non-generative controls and quota shortfalls are shown honestly. In 2D, TabDiff's nominal 25%/50% whole-column masks both become one feature (50% actual), so each shared mask is plotted once per F1 setting.

Re-execute with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 reports/experiment_results.ipynb`; synthetic generation uses `reports/synthetic_approaches.py` and `reports/synthetic_extended.py`.


## CLIMB benchmark

The runner uses the 73-entry IMBENS catalog but loads **raw OpenML tables**, never globally preprocessed IMBENS NPZ data. Versions, checksums/fingerprints, label mappings and original row IDs are recorded. It compares baseline/corrected UGO for logistic regression, random forest, KNN, RBF SVM, MLP and XGBoost using the shared classifier factory. Shoppers is the default Phase1 quick-check surface; reserve CLIMB for deliberate broad evaluation.

```bash
conda run -n ml_project python -m benchmarks.climb_benchmark --dataset wilt --folds 3
conda run -n ml_project python -m benchmarks.climb_benchmark --dataset page-blocks --target-type multiclass --folds 3
```

Use repeated `--dataset` or `--algorithm` arguments to select entries. `--max-samples` is an explicitly stratified development subsample; it is not full-dataset evaluation. Tensor neighbors/interpolation/scoring and MLP use CUDA when available. XGBoost uses CUDA histogram training; sklearn logistic regression/KNN/SVM and preprocessing remain CPU operations. XGBoost currently receives NumPy inputs and may report its CPU-input/CUDA-booster DMatrix prediction fallback; a zero-copy CuPy input path is not installed. For an entirely CPU benchmark use `--device cpu --mlp-device cpu --xgb-device cpu`.

## Split and validation contract

1. Share outer real-data folds across all algorithms/strategies.
2. Split real outer-training rows into inner fit/validation **before** fitting preprocessing or generating descendants. Both sides retain at least one observation/class; the requested validation fraction is adjusted at those class-support bounds. Fewer than two outer-training observations/class is an explicit error, not a reason to discard rare classes.
3. Fit median imputation, scaling, categorical imputation and one-hot encoding on inner fit only. Unseen validation/test categories do not expand its feature schema.
4. Keep the same real validation observations outside every inner retraining, seed selection, interpolation and support reference. `train_classifier(..., validation_data=(X_val, y_val))` selects epochs; with `validation_data=None`, it trains all supplied rows for exactly the supplied epoch count, without an implicit split.
5. Freeze inner-selected MLP epoch counts, fit a fresh preprocessor on full outer training, and replay/refit there without validation or outer-test feedback. Former inner-validation rows may rejoin only this final fit. In simulation refits, their labels remain clean; already-corrupted fit labels are reused unchanged.

Categorical output blocks are copied from a real interpolation partner rather than fractionally interpolated. Singleton/all-identical unsupported groups generate nothing. The generic benchmark uses stratified observation-level folds; datasets requiring entity/time splitting need a corresponding split policy before confirmatory use.

## Stopping decisions

- Retrain on accumulated real plus accepted synthetic training rows; score, establish support, and choose parents from **real rows only**. Synthetic observations never certify support or become parents.
- Freeze the total augmented-data cap at `floor(phi * largest_real_class_count * class_count)`. Clip each batch to its remaining synthetic allowance; do not recompute a moving cap from generated class counts. Retain the independent 20-round default cap.
- Welch stopping occurs when class difficulties do not differ significantly. Constant groups stop only when their values agree; different constant values, mixed zero-variance groups, or insufficient support provide no evidence of equality and continue under the fixed cap.
- Log rejected/unspent budgets, real support/difficulty, proposed/accepted/relocated counts, duplicates, seed coverage, descendant counts, retrains and timing.

These are explicit implementation decisions required by the training-isolation contract, not evidence that the altered mechanism improves downstream utility.

## Metrics and reproducibility

Both evaluation paths define `pr_auc` as **sklearn average precision (AP)**, never trapezoidal PR area. Binary evaluation fixes the positive label from clean real outer-training prevalence; multiclass uses one-vs-rest macro AP. Accuracy, precision, recall, F1 and MCC are also saved. `eval/evaluate.py` additionally supports IAM and PR plotting.

Model runners create unique `results/runs/<run-id>/` directories (override with `--output-dir`); named `--smoke` checks instead use `results/smoke/`. Preparation records have no fitted-model metrics.

- `manifest.json`: arguments/command, Python and dependency versions, source/data fingerprints, CUDA hardware/runtime, status and total runtime.
- `source.zip`: recoverable source/docs/config/catalog snapshot, including when no Git commit is available.
- `metrics.csv`; CLIMB also writes `summary.csv` (dataset-fold means).
- Per dataset/fold: exact real split IDs, clean/training labels and fitted preprocessing statistics.
- Per algorithm: estimator/sampler configuration, selected/replayed epochs, prediction probabilities, inner/outer round records, stop/support summaries, validation metrics and synthetic ancestry mapped to real row IDs.

The output root contains an append-only `experiment_registry.csv`, including failed-run status. Historical `results/climb_benchmark*` files are preserved and reflect the **old**, leaky preprocessing protocol; do not merge them with new results as equivalent runs.

Seeded CPU/CUDA MLP training uses deterministic PyTorch algorithms. Replay guarantees apply to the recorded software/hardware configuration, not bitwise equality across different CUDA devices or dependency versions.

## Verification

```bash
conda run -n ml_project python -m unittest discover -s tests -p "test_*.py"
```

Regression coverage includes train-fitted preprocessing, rare-class holdout coverage, validation/descendant isolation, fixed-epoch refit, generation caps, constant-group stopping, categorical validity, independent simulation controls, AP semantics and deterministic CPU/CUDA execution.

Setup verification: **75 regression tests passed**; Shoppers training-only smoke exercised 25 applicable default classifier/control cells plus 8 MLP/XGBoost CUDA-path cells. Seven intrinsic-ensemble combinations were explicitly unsupported, not fake successes. APS/Satimage preparation validated official partitions without training; BAF requires a local official Base release. No full CLIMB or comparative study was run.

## F1-first five-dataset U2 + FlowGMM comparison

`configs/ablations/climb_flow_f1.json` declares the requested comparison of **original UGO only** against canonical **U2 + compatibility-only FlowGMM**, without corrected UGO, U1 composition, downweighting or an ensemble. The five previously used datasets are `ozone-level-8hr`, `wilt`, `mammography`, `dis` and `Satellite` from the **CLIMB/imbalanced-ensemble catalogue**, not imbalanced-learn's separate dataset collection. Raw cached OpenML tables are loaded so preprocessing fits within each training role.

```bash
# Training-only reduced check across all four native classifier families.
conda run --no-capture-output -n ml_project python -B -m benchmarks.climb_flow_f1 \
  --smoke --datasets ozone-level-8hr dis --seeds 42

# All five datasets, MLP/LR/XGBoost/RF, seeds 42/43/44, both methods: 120 cells.
conda run --no-capture-output -n ml_project python -B -u -m benchmarks.climb_flow_f1 --run

# Independently recompute saved prediction metrics and selected validation thresholds.
conda run --no-capture-output -n ml_project python -B -m benchmarks.climb_flow_f1_report RUN_DIR
```

Both methods receive the same ten native learner proposals per split. U2 cycles group alpha across those proposals; its winning learner/alpha is locked before selecting FlowGMM tau from 0.5/0.6/0.7/0.8. Selection maximizes **real-validation minority F1**, including a classifier threshold search over whole tied-score groups. Test scores never select thresholds. The positive class is identified from real outer training, including when its encoded label is 0. Report fixed-0.5 F1 separately; historical AP-selected/calibrated study scores are not interchangeable with this fresh protocol.

Three seeds define repeated stratified 80/20 outer splits with 20% inner validation. Native MLP fits use fixed 160 epochs; the frozen FlowGMM uses 60 epochs and fits only its reserved 80% real inner-fit core, excluding calibration, validation and test rows even during outer replay. Models and generation use CUDA where supported; LR/RF remain CPU estimators. Both methods retain their native stopping under a 1,000-proposal allowance and 20 rounds. Rejection/cap shortfalls are not refilled, and a common cap does **not** imply matched accepted dose.

This dedicated runner saves compact `manifest.json`, `records.json`, `metrics.csv`, `splits.json`, `selections.json`, and native test/selected-validation prediction CSVs; it deliberately avoids recursive source archives, checkpoints, augmented arrays and raw per-proposal logs. It does not use the older run registry. Source hashes and dependency versions remain in the manifest. `summary.csv`, `paired_deltas.csv`, `verification.json` and `RESULTS.md` are generated after checking complete coverage, paired identities/labels, selected validation thresholds, metrics and compact count/role evidence. Compact ancestry evidence is **not** independent raw-generation replay. Smoke metrics are integration evidence only.

Numeric-only FlowGMM coordinate bridges now derive their width from all transformer output blocks, rather than assuming a categorical block exists. The regression checks raw-coordinate reconstruction, density ratios and gate acceptance with a differently fitted classifier scaler.

Completed study: [all metrics and per-dataset comparisons](results/runs/climb-flow-f1-20261020T152022Z-4506435a/RESULTS.md). All **120 cells** completed; saved native predictions and selected validation thresholds passed independent artifact checks. A separate [raw-data audit](results/runs/climb-flow-f1-20261020T152022Z-4506435a/raw_data_verification.json) bound all 240 test/validation CSVs to raw OpenML labels and independently reconstructed split identities. Equal-dataset mean F1 improves for MLP/LR/RF but declines for XGBoost; outcomes are mixed across datasets. No universal superiority or component-retention decision follows. Scientific runtime was 2,756.45 seconds, including 69.27 seconds of shared Flow fitting; 5,349 native classifier fits were recorded. Six targeted gate/threshold regressions and the 16-cell training-only smoke passed.


