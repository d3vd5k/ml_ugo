# Technical changes to UGO

Standalone changes only; assumes UGO familiarity. No results, retention decisions or combination explanations.

## Corrected real-only UGO — reference implementation

Generated rows never become scoring seeds, parents or support references; the classifier still retrains on real plus synthetic rows. Categories copy whole values from real partners. Fix the augmented-data cap from original class counts and handle undefined Welch cases explicitly.

## U1 — out-of-fold difficulty

Replace in-sample modified-margin difficulty with three-fold real-only predictions. Each fold fits its own preprocessing and classifier; score each row with a model that excluded it:

$$
d_i=\max_{c\ne y_i}p(c\mid x_i)-p(y_i\mid x_i).
$$

Higher means harder. Fix these scores before generation while downstream retraining continues. This estimates difficulty, not epistemic uncertainty.

## U2 — subgroup protection and allocation

Partition real minority rows using KMeans or mutual-kNN components. Estimate stability from co-membership agreement across real-row subsamples. Replace pooled class allocation with subgroup quotas.

The stability allocation uses

$$
w_g=\frac{r_g(h_g+\epsilon)}{(n_g+\epsilon)^\alpha},
$$

Here $r_g$ is stability, $h_g$ mean within-class difficulty percentile, $n_g$ real group size, $\epsilon=0.001$, and $\alpha$ controls the size penalty. Alternatives use size, uniform or difficulty-only weights. Capped largest-remainder quotas allow at most $2n_g$ accepted descendants per group and four per anchor. Unsupported/zero-stability groups receive none. **Allocation groups do not restrict interpolation partners.**

## U3 — constrained generation

Restrict partners to the same real mutual-kNN component and identical categorical tuple. For candidate $x'$ from anchor $x$, require

$$
\|x'-x\|\le\eta D_{\mathrm{opp}}(x),\qquad
\frac{D_{\mathrm{same}}(x')}{D_{\mathrm{opp}}(x')+\epsilon}\le\rho.
$$

$D_{\mathrm{same}}$ and $D_{\mathrm{opp}}$ are nearest real same/opposing-class distances in numeric coordinates. On failure: halve the step up to four times, reject, or relocate within the compatible component. Recheck repaired points. These bounds do not guarantee correct labels.

## U4 — validation-selected stopping budget

Replace Welch termination with 0%, 25%, 50% or 100% augmentation budgets. Compare checkpoints along one trajectory or independently generated budgets. Validation AP selects budget and learner; ties prefer fewer generated rows.

## H1 — HDBSCAN membership gate

Fit frozen HDBSCAN on real minority rows. After ordinary relocation, reject candidates assigned to noise by approximate prediction. Keep real noise-labelled rows in classifier training; membership is not label correctness.

## Neighbor rejection — `N_reject`

Reject original proposals that fail UGO's neighbor-label check instead of relocating them. This isolates removal of relocation.

## F1 — FlowGMM class compatibility

Fit a shared invertible RealNVP transform with one learned diagonal-Gaussian latent component per class, using real observations from both classes. Mixed categorical inputs use one-hot dequantization.

Replace neighbor detection **and relocation** with a gate on original proposals:

$$
\log p(x'\mid y)-\log p(x'\mid\bar y)\ge\log\frac{\tau}{1-\tau}.
$$

Here $y$ is the intended label and $\bar y$ the opposite label. Equal priors are assumed; the shared Jacobian cancels in the ratio. The frozen gate checks relative class compatibility, not absolute density. Validation selects $\tau$.

## F2 — absolute likelihood threshold

Keep the frozen F1 detector and compatibility threshold. Additionally require intended-class log likelihood to exceed a real calibration quantile: 0.01, 0.05 or 0.10, with an explicit off control. Unlike F1's ratio, this likelihood includes the flow Jacobian. Calibration rows are excluded from detector fitting; validation selects the quantile.

## Random K5 candidate pools

Generate five proposals per distinct planned real anchor. Deduplicate final coordinates; randomly select at most one candidate per anchor under existing group quotas. Do not regenerate shortfalls. Native difficulty scoring and retraining are unchanged.

## Guidance-gradient utility selection

Use the same K5 pool, but rank candidates by

$$
u(x',y)=\nabla_\theta L_G^\top\nabla_\theta\ell(x',y).
$$

$L_G$ is class-balanced guidance cross-entropy; $\ell$ is candidate cross-entropy. Larger scores estimate greater guidance-loss reduction after a small SGD step, not full retraining. Include weight and bias gradients without modifying the live model. Compare first-round and iterative selection.

Two data-access variants were tried: a separate held-out guidance partition, and reuse of validation as guidance. The latter is selection feedback, not independent evaluation. LR uses native gradients; RF/LightGBM use a separately fitted LR surrogate while native trees still score anchors and retrain.

## Within-group input versus Flow geometry

Restrict both arms to identical same-class, same-group real-parent rules. Compare ordinary input interpolation with

$$
x'=f^{-1}\big((1-t)f(x_a)+t f(x_b)\big),
$$

where $f$ is the frozen Flow transform. Decode numeric coordinates through the detector and classifier preprocessing; categorical values copy the real partner. Reject invalid decoded values before ordinary relocation. This changes geometry, not density-based acceptance.

## TabDiff — conditional feature-block inpainting

Replace interpolation with conditional diffusion: regenerate selected feature blocks while fixing the anchor's other fields and minority label. Train on real data only; during sampling, known coordinates follow the diffusion noise schedule, then unmasked fields are restored exactly.

Try approximately 25%/50% masks in **one augmentation round**. Preserve domain-linked blocks, reject schema-invalid outputs, and match accepted counts per source group across arms. In 2D, both masks round to one feature.

## Synthetic downweighting

Keep every real row at weight 1; give accumulated synthetic rows weight 0.25 or 0.5 instead of 1. Use native weighted fitting; MLP uses cross-entropy normalized by total sample weight. Validation selects the weight with other method settings locked. Weighted retraining can change later anchor choices and stopping, even though the generator itself is unchanged.

## Comparison controls

| Control | Technical difference |
| --- | --- |
| Real-only | No augmentation. |
| Class weighting | Increase minority training influence without adding rows. |
| Random oversampling | Duplicate minority observations. |
| SMOTE / SMOTENC | Neighbor interpolation; SMOTENC handles whole categorical variables. |
| BorderlineSMOTE | Concentrate interpolation on minority observations near class boundaries. |
| KMeansSMOTE | Allocate interpolation across eligible clusters. |
| Balanced random forest | Class-balanced resampling within tree training, not a UGO generator. |
| Matched-anchor duplication | Replace a generated batch by copies of its real source anchors at equal count and weight. |
| Group-matched random thinning | Randomly retain reference candidates to match a filter's accepted count per source group. |

For mixed Shoppers features, BorderlineSMOTE/KMeansSMOTE operate on numeric-plus-one-hot coordinates, followed by whole-category projection. Filter rejection and unsupported quotas are not refilled or padded.
