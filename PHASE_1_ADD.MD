# PHASE 1 extensions: HDBSCAN and FlowGMM

## Scope

U1–U5 were implemented; **retain the user's current U2 only**. Locate its implementation/configuration and preserve it as the baseline. Add the experiments below independently; do not combine HDBSCAN and FlowGMM initially or restore discarded U1/U3/U4 components.

Keep seed scoring, U2 group allocation, generation, pre-generation noise detection, downstream classifier and stopping unchanged except for explicit post-generation handling below. Record the baseline commit/configuration. Start with binary tasks.

## HDBSCAN: additional minority-support filter

- Fit on **real minority training observations only**, in a training-fitted distance representation. Freeze during generation; never refit with synthetic observations.
- Use the Python `hdbscan` package with `prediction_data=True`; assign new candidates through `hdbscan.approximate_predict`.
- Preserve U2's existing grouping/allocation. HDBSCAN groups serve the additional filter only.
- Apply H1 **after existing U2 post-generation handling**, to the final candidate coordinates. This tests an additional filter, not replacement of the neighbor rule.

Primary rule:

\[
A_H(x')=\mathbf1[\hat g(x')\ne-1].
\]

Optional follow-up, only after H1: source-cluster consistency plus membership strength,

\[
A_{H2}(x'\mid x_i)=
\mathbf1[\hat g(x_i)\ne-1]
\mathbf1[\hat g(x')=\hat g(x_i)]
\mathbf1[s(x')\ge\tau_H].
\]

Use the same frozen prediction convention for source and candidate assignments. Preserve the original source ID through relocation. Membership strength is not label-correctness probability; label `-1` and GLOSH outlier scores are distinct outputs.

Tune `min_cluster_size` and `min_samples` independently on validation; record the search space before experiments. Standardize continuous features and define categorical distances explicitly. If clustering produces no usable clusters, log the degenerate result; do not silently bypass the filter. Keep real observations labeled `-1` in classifier training.

## FlowGMM: replace post-generation detection and relocation

- Fit **both classes using real data only**; freeze throughout generation. Use a shared invertible flow with class-associated Gaussian latent components; pin the implementation/version.
- F1/F2 replace the original post-generation neighbor detector **and relocation**. Failed candidates are discarded. Do not run the original post-detector afterward.
- Keep feature/domain validity checks independent and unchanged.
- Within each training partition, reserve separate real fit, calibration and validation subsets. Fit flow/preprocessing on fit data; calculate likelihood quantiles on calibration data; select settings on validation. Preserve identical downstream training access across variants. Outer test data is never used here.

For intended class \(y\), opposing class \(\bar y\), and \(z=f_\theta(x')\):

\[
\ell_y(x')=
\log\mathcal N(z;\mu_y,\Sigma_y)
+\log|\det J_{f_\theta}(x')|,
\qquad
\Delta_y(x')=\ell_y(x')-\ell_{\bar y}(x').
\]

Use **equal class priors**, so \(s_y(x')=\sigma(\Delta_y(x'))\). Evaluate acceptance in log space.

**F1 — compatibility only:**

\[
A_1(x',y)=\mathbf1\left[\Delta_y(x')\ge\log\frac{\tau}{1-\tau}\right].
\]

**F2 — likelihood plus compatibility:**

\[
T_y(q)=Q_q\{\ell_y(x_i):x_i\in D_y^{\mathrm{cal}}\},
\qquad
A_2(x',y)=A_1(x',y)\mathbf1[\ell_y(x')\ge T_y(q)].
\]

Include the Jacobian in F2's likelihood check; it cancels only in the class ratio. Recompute calibration scores/thresholds whenever the flow or preprocessing is refitted. Document handling of discrete features; do not assume arbitrary category codes form a valid continuous-density model.

Initial grid: \(\tau\in\{0.5,0.6,0.7,0.8\}\), \(q\in\{0.01,0.05,0.10\}\). Select \(\tau\) using F1, then freeze it while selecting F2's \(q\). Reuse the same fitted flow for F1/F2. Include explicit support-filter-off; \(q=0\) does not disable filtering. Record calibration minority count and quantile interpolation convention.

## Experiments and controls

| ID | Post-generation behavior |
|---|---|
| U2 | Existing neighbor detection + relocation |
| H1 | U2 handling, then HDBSCAN non-noise gate |
| N-reject | Existing neighbor detection, replacing relocation with rejection |
| F1 | FlowGMM compatibility; reject failures |
| F2 | FlowGMM likelihood + compatibility; reject failures |

1. Begin with one saved U2 proposal pool, preserving parent/group IDs. Route copies through each handler. H1 scores the post-relocation coordinates; F1/F2 score original proposals. Save both coordinate versions.
2. Compare H1 against random thinning of U2 outputs, and F2 against random thinning of F1 outputs, matched to accepted counts per **original U2 group**. Repeat random thinning across seeds.
3. Run full iterative experiments afterward with matched seeds and fixed proposal/retraining caps. Pools may diverge after classifier retraining; report this explicitly.
4. No automatic retries to fill rejected quotas. If none are accepted, skip unchanged retraining and terminate that generation run with an explicit status.

## Verification, reporting and decision

- Verify frozen real-only fits, absence of validation/test leakage, log-space numerical stability and \(\{A_2=1\}\subseteq\{A_1=1\}\) at fixed model/\(\tau\).
- Log proposal ID, real parents, U2 group, round, original/final coordinates, scores, thresholds, acceptance and rejection reason. Track feature-invalid/non-finite candidates separately.
- Report paired predictive differences against U2, original-group acceptance/recall with support counts, duplicates, unspent quotas, detector training/inference time and total runtime.
- Retain F2 only if its selection adds value beyond F1 and matched thinning. Use N-reject to separate detector changes from relocation removal.
- Describe HDBSCAN membership and FlowGMM likelihood as **model-based checks**, not guarantees of correct labels or true support. Inspect whether filters suppress the smallest minority groups.
- Deliver reproducible configs/run commands, per-run predictions, metrics, candidate logs and a short retain/remove decision. Do not expand to new generators in this task.

Sources: [HDBSCAN prediction](https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html), [HDBSCAN parameters](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html), [FlowGMM paper](https://proceedings.mlr.press/v119/izmailov20a.html).
