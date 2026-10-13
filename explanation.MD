1. What the architecture is after Phase 1                                               
                                                                                        
The preferred architecture is canonical U2: a group-aware oversampler wrapped around a  
standard classifier. It is not a new neural-network architecture.                       
                                                                                        
Your current decisions are:                                                             
                                                                                        
┌─────────────────────────────────────────┬───────────────┬────────────────────────────┐
│ Component                               │ Status        │ Role in the preferred U2   │
│                                         │               │ pipeline                   │
├─────────────────────────────────────────┼───────────────┼────────────────────────────┤
│ U1: out-of-fold MMA scoring             │ Kept          │ Not included in U2         │
│                                         │ separately    │                            │
├─────────────────────────────────────────┼───────────────┼────────────────────────────┤
│ U2: KMeans groups + stability-based     │ Kept;         │ Included                   │
│ allocation                              │ preferred     │                            │
├─────────────────────────────────────────┼───────────────┼────────────────────────────┤
│ U3: geometric generation constraints    │ Discarded     │ Excluded                   │
├─────────────────────────────────────────┼───────────────┼────────────────────────────┤
│ U4: validation-selected generation      │ Discarded     │ Excluded                   │
│ stopping                                │               │                            │
├─────────────────────────────────────────┼───────────────┼────────────────────────────┤
│ U5: U1 + U2 combination                 │ Discarded     │ Excluded                   │
└─────────────────────────────────────────┴───────────────┴────────────────────────────┘
                                                                                        
Keeping U1 and U2 individually does not mean combining them. That combination was U5,   
which you discarded.                                                                    
                                                                                        
The configuration is still not frozen, and Stage 2 remains gated. Below is the          
architecture of the current preferred U2 candidate.                                     
                                                                                        
────────────────────────────────────────────────────────────────────────────────        
                                                                                        
2. End-to-end structure                                                                 
                                                                                        
┌──────────────────────────┐        ┌───────────────────────┐                           
│                          │        │                       │                           
│    Real training data    │        │  Unseen observations  │                           
│                          │        │                       │                           
└─────────────┬────────────┘        └───────────┬───────────┘                           
              │                                 │                                       
              │                                 │                                       
              │                                 │                                       
              │                                 │                                       
              ▼                                 ▼                                       
┌──────────────────────────┐        ┌───────────────────────┐                           
│                          │        │                       │                           
│                          │        │                       │                           
│   Fit preprocessing on   │        │      Apply fitted     │                           
│    real training rows    │        │     preprocessing     │                           
│                          │        │                       │                           
└─────────────┬────────────┘        └───────────┬───────────┘                           
              │                                 │                                       
              │                                 │                                       
              │                     ┌───────────┘                                       
              │                     │                                                   
              ▼                     │                                                   
┌──────────────────────────┐        │                                                   
│                          │        │                                                   
│                          │        │                                                   
│  Immutable real feature  │        │                                                   
│          matrix          │        │                                                   
│                          │        │                                                   
└─────────────┬────────────┘     ┌──┘                                                   
              │                  │                                                      
              │                  │                                                      
              ├──────────────────┼──────────────┐                                       
              │                  │              │                                       
              ▼                  │              ▼                                       
┌──────────────────────────┐     │  ┌───────────────────────┐                           
│                          │     │  │                       │                           
│                          │     │  │                       │                           
│ Per-target-class KMeans  │     │  │                       │                           
│          groups          │     │  │ Real-only support and │                           
│     and subsampling      │     │  │     neighbor pools    │                           
│        stability         │     │  │                       │                           
│                          │     │  │                       │                           
└─────────────┬────────────┘     │  └───────────┬───────────┘                           
              │                  │              │                                       
              │                  │              │                                       
              │                  │              │                                       
              │                  │              │                                       
              ▼                  │              │                                       
┌──────────────────────────┐     │              │                                       
│                          │     │              │                                       
│   Choose target class;   │     │              │                                       
│  allocate capped group   │  No─┼──────────────┘                                       
│     and seed quotas      │     │                                                      
│                          │     │                                                      
└─────────────┬────────────┘     │                                                      
              │                  │                                                      
              │                  │                                                      
              │                  │                                                      
              │                  │                                                      
              ▼                  │                                                      
┌──────────────────────────┐     │                                                      
│                          │     │                                                      
│                          │     │                                                      
│ Interpolate between real │     │                                                      
│     same-class rows;     │     │                                                      
│  copy donor categorical  │     │                                                      
│       coordinates        │     │                                                      
│                          │     │                                                      
└─────────────┬────────────┘     │                                                      
              │                  │                                                      
              │                  │                                                      
              │                  │                                                      
              │                  │                                                      
              ▼                  │                                                      
┌──────────────────────────┐     │                                                      
│                          │     │                                                      
│                          │     │                                                      
│    Check against real    │     │                                                      
│        neighbors;        │     │                                                      
│   relocate conflicting   │     │                                                      
│        candidates        │     │                                                      
│                          │     │                                                      
└─────────────┬────────────┘     │                                                      
              │                  │                                                      
              │                  │                                                      
              │                  │                                                      
              │                  │                                                      
              ▼                  │                                                      
┌──────────────────────────┐     │                                                      
│                          │     │                                                      
│                          │     │                                                      
│  Commit accepted quotas  │     │                                                      
│      and provenance      │     │                                                      
│                          │     │                                                      
└─────────────┬────────────┘     │                                                      
              │                  │                                                      
              │                  │                                                      
              ├──────────────────┤                                                      
              │                  │                                                      
              ▼                  │                                                      
┌──────────────────────────┐     │                                                      
│                          │     │                                                      
│                          │     │                                                      
│  Training pool: real +   │     │                                                      
│ accepted synthetic rows  │     │                                                      
│                          │     │                                                      
└─────────────┬────────────┘     │                                                      
              │                  │                                                      
              │                  │                                                      
              │                  │                                                      
              │                  │                                                      
              ▼                  │                                                      
┌──────────────────────────┐     │                                                      
│                          │     │                                                      
│  Fit a fresh classifier  │     │                                                      
│                          │     │                                                      
└─────────────┬────────────┘     │                                                      
              │                  │                                                      
              │                  │                                                      
              │                  │                                                      
              │                  │                                                      
              ▼                  │                                                      
┌──────────────────────────┐     │                                                      
│                          │     │                                                      
│                          │     │                                                      
│  Compute MMA difficulty  │     │                                                      
│    on real rows only     │     │                                                      
│                          │     │                                                      
└─────────────┬────────────┘     │                                                      
              │                  │                                                      
              │                  │                                                      
              │                  │                                                      
              │                  │                                                      
              ▼                  │                                                      
◇──────────────────────────◇     │                                                      
│                          │     │                                                      
│          Stop?           │ ├───┤                                                      
│                          │     │                                                      
◇─────────────┬────────────◇     │                                                      
              │                  │                                                      
             Yes                 │                                                      
              │                  │                                                      
              │                  │                                                      
              ▼                  │                                                      
┌──────────────────────────┐     │                                                      
│                          │     │                                                      
│                          │     │                                                      
│ Final classifier fit on  │     │                                                      
│ resulting training pool  │     │                                                      
│                          │     │                                                      
└─────────────┬────────────┘     │                                                      
              │                  │                                                      
              │                  │                                                      
              ├──────────────────┘                                                      
              │                                                                         
              ▼                                                                         
┌──────────────────────────┐                                                            
│                          │                                                            
│    Final classifier:     │                                                            
│    probabilities and     │                                                            
│       predictions        │                                                            
│                          │                                                            
└──────────────────────────┘                                                            
                                                                                        
The key separation is:                                                                  
                                                                                        
- Geometry and ancestry: fixed, real-only.                                              
- Classifier training: real plus accepted synthetic rows.                               
- Difficulty scores: recomputed on real rows using that classifier.                     
                                                                                        
Therefore, synthetic observations can influence the learned decision boundary, but      
cannot become interpolation parents or inflate subgroup support.                        
                                                                                        
────────────────────────────────────────────────────────────────────────────────        
                                                                                        
3. Detailed training architecture                                                       
                                                                                        
### A. Data boundaries and preprocessing                                                
                                                                                        
The controlled Shoppers protocol uses:                                                  
                                                                                        
- A fixed, stratified 20% outer test split.                                             
- An inner validation split comprising 20% of outer training.                           
- Five experiment seeds for inner splitting and stochastic variation.                   
                                                                                        
Preprocessing is fitted separately for each training role:                              
                                                                                        
- Numeric: median imputation followed by standardization.                               
- Categorical: most-frequent imputation followed by one-hot encoding.                   
- Unknown categories at inference become all-zero categorical blocks.                   
                                                                                        
During candidate selection, preprocessing sees only inner-fit rows. After selection,    
preprocessing is rebuilt using the full outer-training partition.                       
                                                                                        
Neither validation nor test observations enter the sampler’s training geometry.         
Validation selects candidates; the outer test evaluates the selected refit.             
                                                                                        
The Phase 1 learner families were logistic regression, random forest and LightGBM. U2   
operates through a classifier-training callback rather than prescribing one classifier  
family.                                                                                 
                                                                                        
### B. Two distinct data stores                                                         
                                                                                        
U2 effectively maintains two datasets:                                                  
                                                                                        
Immutable real reference data                                                           
- Original training features and labels.                                                
- KMeans memberships and subgroup stability.                                            
- Eligibility masks and real neighbor pools.                                            
- Real seed identities and subgroup sizes.                                              
                                                                                        
Growing classifier-training data                                                        
- All original real observations.                                                       
- Accepted synthetic observations appended during generation.                           
                                                                                        
Synthetic additions never change the first store. Also, excluding a row from            
interpolation support does not delete it from classifier training.                      
                                                                                        
This separation removes the original UGO’s direct synthetic-parent feedback loop.       
                                                                                        
### C. Real subgroup discovery                                                          
                                                                                        
U2 groups each class that has an augmentation deficit separately.                       
                                                                                        
Canonical grouping uses:                                                                
                                                                                        
- KMeans.                                                                               
- Up to 8 clusters per target class.                                                    
- n_init=10.                                                                            
- Distinct coordinates for geometric fitting, followed by restoration of original row   
  memberships.                                                                          
                                                                                        
The cluster count cannot exceed the number of distinct coordinates.                     
                                                                                        
A subgroup can supply generation seeds only when it has:                                
                                                                                        
1. At least two real observations.                                                      
2. At least two distinct feature coordinates.                                           
3. Positive estimated stability.                                                        
                                                                                        
Consequently, repeated copies of one coordinate do not establish an interpolatable      
subgroup.                                                                               
                                                                                        
### D. Subgroup stability                                                               
                                                                                        
For each target class, U2 performs five independent subsampling repeats, each drawing   
approximately 80% of its real observations without replacement.                         
                                                                                        
Each repeat refits KMeans. U2 compares original and refitted pairwise co-membership     
using a subgroup-wise Jaccard measure.                                                  
                                                                                        
This detects both:                                                                      
                                                                                        
- Splitting: an original subgroup separates under resampling.                           
- Merging: its members become grouped with outsiders.                                   
                                                                                        
Cluster label numbering does not matter.                                                
                                                                                        
Only repeats with enough sampled subgroup members contribute to its average. A subgroup 
with no evaluable repeats receives zero stability.                                      
                                                                                        
Stability measures geometric reproducibility—not label correctness, calibrated          
confidence, or a probability of being noise.                                            
                                                                                        
### E. Minority-support protection                                                      
                                                                                        
UGO’s preliminary neighbor check flags a real observation when the majority label among 
its nearest neighbors disagrees with its label.                                         
                                                                                        
That is a local label-conflict signal. A legitimate minority island surrounded by       
majority observations can trigger it.                                                   
                                                                                        
U2 therefore restores generation eligibility for real observations belonging to         
supported target-class subgroups, even when they were flagged by this check.            
                                                                                        
The benefit is that neighbor disagreement alone cannot eliminate a supported minority   
subgroup. The limitation is that stable mislabeled structure could also be restored;    
stability is not a label-cleaning oracle.                                               
                                                                                        
### F. Difficulty scoring                                                               
                                                                                        
At each generation round, a fresh classifier is fitted on the current augmented training
pool.                                                                                   
                                                                                        
It scores the original real observations using MMA:                                     
                                                                                        
uᵢ= max p(c∣ xᵢ)-p(yᵢ∣ xᵢ).                                                             
   c≠ yᵢ                                                                                
                                                                                        
Interpretation:                                                                         
                                                                                        
- Negative: the true class has the strongest probability.                               
- Near zero: the observation is near a probability-margin boundary.                     
- Positive: a competing class outranks the true class.                                  
- Larger values mean greater difficulty.                                                
                                                                                        
This is a true-label probability margin, not Bayesian or epistemic uncertainty.         
                                                                                        
Crucially, canonical U2 uses in-sample scoring of real rows. Its scoring model has been 
trained on those rows and the accepted synthetic additions. It does not use U1’s        
out-of-fold scoring.                                                                    
                                                                                        
### G. Class scheduling and subgroup allocation                                         
                                                                                        
Classes with remaining augmentation deficits are ranked by their eligible real          
observations’ mean MMA difficulty. U2 attempts allocation for the hardest available     
target class.                                                                           
                                                                                        
The normal round batch is bounded by:                                                   
                                                                                        
max ⎛1,round⎛ 0.1N     ⎞⎞,                                                              
    ⎝       ⎝     real ⎠⎠                                                               
                                                                                        
remaining generation budget and available class/group/seed capacity.                    
                                                                                        
Within the selected class, subgroup weights are:                                        
                                                                                        
w =                                                                                     
 g                                                                                      
 s ⎛ h +ϵ ⎞                                                                             
  g⎝  g   ⎠                                                                             
────────────,                                                                           
         α                                                                              
 ⎛ n +ϵ ⎞ g                                                                             
 ⎝  g   ⎠                                                                               
     ϵ=0.001,                                                                           
                                                                                        
where:                                                                                  
                                                                                        
- s_g: subgroup stability.                                                              
- h_g: mean within-class MMA percentile of its real members.                            
- n_g: real subgroup size.                                                              
- α_g: size-penalty strength.                                                           
                                                                                        
The Phase 1 search considered α_g∈{0,0.5,1}.                                            
                                                                                        
Increasing α_g reduces large subgroups’ dominance, but does not force equal subgroup    
allocations.                                                                            
                                                                                        
Integer quotas use capped proportional largest-remainder allocation, redistributing     
capacity when groups saturate.                                                          
                                                                                        
### H. Seed selection and generation limits                                             
                                                                                        
Within each subgroup, U2 selects the highest-MMA available seeds, up to:                
                                                                                        
⎡ 0.5n ⎤.                                                                               
⎢     g⎥                                                                                
                                                                                        
Quota is distributed across those selected seeds subject to remaining capacity—not      
directly sampled in proportion to each seed’s MMA.                                      
                                                                                        
The default limits are:                                                                 
                                                                                        
- 4 accepted descendants per real seed.                                                 
- 2n_g accepted descendants per subgroup.                                               
- Explicit class targets and a global generation cap.                                   
                                                                                        
These counters track accepted observations. Synthetic additions cannot increase the     
capacities.                                                                             
                                                                                        
If the admissible support cannot spend the requested budget, U2 reports underspending   
rather than padding the dataset with an unrelated fallback.                             
                                                                                        
### I. Interpolation and relocation                                                     
                                                                                        
A selected real seed chooses a donor from its eligible same-class real nearest          
neighbors, normally using up to five neighbors.                                         
                                                                                        
Numeric coordinates are interpolated:                                                   
                                                                                        
z=xᵢ+t(xⱼ-xᵢ),     t∼ U(0,1).                                                           
                                                                                        
Categorical coordinates are copied together from the donor, preserving valid categorical
encodings.                                                                              
                                                                                        
The donor is not required to belong to the seed’s KMeans subgroup. Groups govern        
allocation, not strict geometric containment.                                           
                                                                                        
After generation, U2 checks the candidate against real neighbors; the default post-check
uses one nearest neighbor. If its label conflicts, the candidate is relocated by copying
the nearest eligible real observation of the target class exactly.                      
                                                                                        
Thus:                                                                                   
                                                                                        
- Relocation may create duplicates.                                                     
- The synthetic point need not remain inside its source subgroup.                       
- Its audited subgroup denotes its source seed’s group, not its final location.         
                                                                                        
Because U3 was discarded, there is no additional manifold, disconnected-component or    
subgroup-envelope constraint.                                                           
                                                                                        
### J. Stopping and final fitting                                                       
                                                                                        
U2 retains corrected UGO’s stopping backbone:                                           
                                                                                        
- Welch ANOVA compares MMA distributions across eligible real classes.                  
- A p-value above 0.05 stops generation.                                                
- Generation also terminates through exhausted targets/budget, unavailable support or   
  partners, or the 20-round default limit.                                              
                                                                                        
The backbone retains the original-style global guard:                                   
                                                                                        
B=max ⎛0,⌊2.5 nₘₐₓK⌋-N    ⎞.                                                            
      ⎝               real⎠                                                             
                                                                                        
In the controlled Shoppers study, explicit targets limited augmentation to 1,000        
additional minority observations. That is a study budget, not a universal architectural 
constant or a promise of complete class balancing.                                      
                                                                                        
Welch stopping means failure to detect a class-difficulty difference—not proof that the 
classes are equally difficult or that downstream performance is optimal.                
                                                                                        
Finally, the classifier is fitted again on the resulting training pool.                 
                                                                                        
────────────────────────────────────────────────────────────────────────────────        
                                                                                        
4. Selection, implementation and inference                                              
                                                                                        
### Candidate selection                                                                 
                                                                                        
U2 used ten joint learner/sampler proposals per cell, cycling the three group-alpha     
values across those ten proposals.                                                      
                                                                                        
This is not ten learner configurations multiplied by three alpha values.                
                                                                                        
Real-validation average precision selects the proposal. The selected learner parameters 
and group alpha are replayed during the outer refit, while groups are rebuilt from      
outer-training data.                                                                    
                                                                                        
The controlled binary threshold remains 0.5. U4’s checkpoint-based generation selection 
is absent.                                                                              
                                                                                        
### Implementation responsibilities                                                     
                                                                                        
┌────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Module                 │ Responsibility                                              │
├────────────────────────┼─────────────────────────────────────────────────────────────┤
│ benchmarks/run.py      │ Training boundaries, candidate selection, refitting and     │
│                        │ evaluation                                                  │
├────────────────────────┼─────────────────────────────────────────────────────────────┤
│ classifiers/factory.py │ Constructing and freshly fitting the selected learner       │
├────────────────────────┼─────────────────────────────────────────────────────────────┤
│ samplers/protected.py  │ Connecting U2 support protection and allocation to the      │
│                        │ pipeline                                                    │
├────────────────────────┼─────────────────────────────────────────────────────────────┤
│ samplers/groups.py     │ Real-only grouping and stability                            │
├────────────────────────┼─────────────────────────────────────────────────────────────┤
│ samplers/allocation.py │ Group weights, integer quotas and accepted-descendant       │
│                        │ limits                                                      │
├────────────────────────┼─────────────────────────────────────────────────────────────┤
│ samplers/ugo.py        │ MMA, interpolation, relocation, retraining and stopping     │
└────────────────────────┴─────────────────────────────────────────────────────────────┘
                                                                                        
Audits retain class counts, subgroup support/stability, seed multiplicities, real       
ancestry, relocation, duplicates and stopping reasons.                                  
                                                                                        
### Inference                                                                           
                                                                                        
Deployment needs the fitted preprocessor and final classifier.                          
                                                                                        
There is no inference-time clustering, oversampling, MMA scoring or synthetic           
generation. U2 adds training work, not an extra prediction-stage model.                 
                                                                                        
────────────────────────────────────────────────────────────────────────────────        
                                                                                        
5. Architectural comparison with UGO                                                    
                                                                                        
Here, original UGO/U0 means the repository’s paper-derived UGOSMOTE implementation. It  
is not a verified reproduction of executable author code; documented paper ambiguities  
required implementation choices.                                                        
                                                                                        
┌────────────────────┬──────────────────┬────────────────────────┬─────────────────────┐
│ Dimension          │ Original UGO /   │ Corrected UGO          │ Preferred U2        │
│                    │ U0               │                        │                     │
├────────────────────┼──────────────────┼────────────────────────┼─────────────────────┤
│ Classifier         │ Real + synthetic │ Real + synthetic       │ Real + synthetic    │
│ training           │                  │                        │                     │
├────────────────────┼──────────────────┼────────────────────────┼─────────────────────┤
│ Rows receiving MMA │ Current          │ Original real rows     │ Original real rows  │
│ scores             │ augmented rows   │                        │                     │
├────────────────────┼──────────────────┼────────────────────────┼─────────────────────┤
│ Interpolation      │ May include      │ Real-only              │ Real-only           │
│ parents            │ synthetic        │                        │                     │
│                    │ descendants      │                        │                     │
├────────────────────┼──────────────────┼────────────────────────┼─────────────────────┤
│ Allocation         │ Whole-class      │ Whole-class difficulty │ Stable real         │
│ structure          │ difficulty       │ ranking                │ subgroups           │
│                    │ ranking          │                        │                     │
├────────────────────┼──────────────────┼────────────────────────┼─────────────────────┤
│ Local label        │ Excludes flagged │ Excludes flagged real  │ Restores supported  │
│ conflict           │ initial seeds    │ support                │ target subgroups    │
├────────────────────┼──────────────────┼────────────────────────┼─────────────────────┤
│ Subgroup           │ None             │ None                   │ Subsampling         │
│ reliability        │                  │                        │ stability           │
├────────────────────┼──────────────────┼────────────────────────┼─────────────────────┤
│ Allocation         │ No U2 subgroup   │ No U2 subgroup limits  │ Seed and subgroup   │
│ concentration      │ limits           │                        │ caps                │
├────────────────────┼──────────────────┼────────────────────────┼─────────────────────┤
│ Donor geometry     │ Current          │ Eligible real          │ Eligible real       │
│                    │ same-class pool  │ same-class pool        │ same-class pool;    │
│                    │                  │                        │ cross-group allowed │
├────────────────────┼──────────────────┼────────────────────────┼─────────────────────┤
│ Relocation         │ Can include      │ Real-only              │ Real-only           │
│ reference          │ descendants      │                        │                     │
├────────────────────┼──────────────────┼────────────────────────┼─────────────────────┤
│ Undefined Welch    │ Explicit early   │ Defined handling with  │ Same corrected      │
│ handling           │ halt             │ bounded continuation   │ handling            │
│                    │                  │ where appropriate      │                     │
├────────────────────┼──────────────────┼────────────────────────┼─────────────────────┤
│ Prediction-time    │ Ordinary         │ Ordinary classifier    │ Ordinary classifier │
│ model              │ classifier       │                        │                     │
└────────────────────┴──────────────────┴────────────────────────┴─────────────────────┘
                                                                                        
U2’s main architectural improvement is controlled representation of real subgroups,     
rather than allowing whole-class rankings and synthetic ancestry to determine where     
generation concentrates.                                                                
                                                                                        
It does not replace the learner, introduce calibrated uncertainty, or guarantee         
geometrically novel synthetic observations.                                             
                                                                                        
### Where U1 fits                                                                       
                                                                                        
The separately retained U1 alternative uses three-fold, real-only out-of-fold MMA scores
computed before generation and held fixed across rounds.                                
                                                                                        
That reduces training-score optimism, but it is a different scoring path. Adding it to  
U2 would recreate the discarded U5.                                                     
                                                                                        
────────────────────────────────────────────────────────────────────────────────        
                                                                                        
6. What Phase 1 actually demonstrates                                                   
                                                                                        
Descriptive five-seed means from the completed Shoppers comparison:                     
                                                                                        
┌─────────────────────┬───────────────────┬──────────────┬───────────────┬────────┐     
│ Learner             │ Metric            │ Original UGO │ Corrected UGO │ U2     │     
├─────────────────────┼───────────────────┼──────────────┼───────────────┼────────┤     
│ Logistic regression │ Average precision │ 0.6077       │ 0.6074        │ 0.5854 │     
├─────────────────────┼───────────────────┼──────────────┼───────────────┼────────┤     
│ Logistic regression │ Minority F1       │ 0.5591       │ 0.5574        │ 0.5482 │     
├─────────────────────┼───────────────────┼──────────────┼───────────────┼────────┤     
│ Logistic regression │ Minority recall   │ 0.4712       │ 0.4670        │ 0.4885 │     
├─────────────────────┼───────────────────┼──────────────┼───────────────┼────────┤     
│ Random forest       │ Average precision │ 0.7244       │ 0.7220        │ 0.7178 │     
├─────────────────────┼───────────────────┼──────────────┼───────────────┼────────┤     
│ Random forest       │ Minority F1       │ 0.6456       │ 0.6383        │ 0.6527 │     
├─────────────────────┼───────────────────┼──────────────┼───────────────┼────────┤     
│ Random forest       │ Minority recall   │ 0.6157       │ 0.6042        │ 0.6696 │     
├─────────────────────┼───────────────────┼──────────────┼───────────────┼────────┤     
│ LightGBM            │ Average precision │ 0.7302       │ 0.7262        │ 0.7281 │     
├─────────────────────┼───────────────────┼──────────────┼───────────────┼────────┤     
│ LightGBM            │ Minority F1       │ 0.6485       │ 0.6477        │ 0.6577 │     
├─────────────────────┼───────────────────┼──────────────┼───────────────┼────────┤     
│ LightGBM            │ Minority recall   │ 0.6346       │ 0.6330        │ 0.6906 │     
└─────────────────────┴───────────────────┴──────────────┴───────────────┴────────┘     
                                                                                        
The evidence supports a recall-oriented tradeoff, with stronger minority F1 for the tree
learners—not universal dominance. Logistic regression loses F1 and AP, and original UGO 
retains higher AP across these three families.                                          
                                                                                        
Remaining limitations include:                                                          
                                                                                        
- In-sample MMA can still be optimistic.                                                
- KMeans stability depends on the feature representation and distance metric.           
- Positive stability does not ensure every group receives quota.                        
- Cross-group interpolation and exact-copy relocation remain possible.                  
- Generation benefits have not been isolated from implicit reweighting using matched    
  mechanism controls.                                                                   
                                                                                        
Runtime check: I exercised the actual U2 path on real Shoppers rows with logistic       
regression: 1,440 fit rows, 40 accepted additions, real-only ancestry, respected        
allocation caps and valid probabilities for 120 internal holdout observations. U1/U3/U4 
were absent. No configuration changed.                                                  
                                                                                        
Bottom line: after Phase 1, the preferred architecture is a real-subgroup-aware,        
stability-weighted, capped UGO training pipeline, followed by ordinary classifier       
inference—not the U1+U2 combination.