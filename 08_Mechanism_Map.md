# Mechanism Map — Paper 4

Purpose: bridge the literature review to experimental design. Claim Matrix answers "what has the
community shown?" Evidence Register answers "how solidly?" This file answers: **if Paper 4
succeeds, what mechanism would explain the phenomenon?** No experiments are proposed here —
that's Step 6.

Scope discipline (per project decision, session 3): this pass searched only for mechanisms that
could explain Paper 3's core observation — information stays linearly probe-decodable
(0.72–0.81 AUROC) while distance-based estimators (Mahalanobis and 7 others) stay flat/below
chance under disentanglement. No new foundation-model-specific or dermatology-specific search
was run; this is general representation-learning/geometry theory, checked for whether it applies
to both the CNN setting (Paper 3) and, hypothetically, an FM setting (Paper 4).

---

## Candidate mechanisms found this session

### 1. Mahalanobis performance is driven by two specific geometric properties, not by information content
**Source:** "Dissecting Mahalanobis: How Feature Geometry and Normalization Shape OOD Detection" (arXiv:2510.15202) — read at Level A (Abstract, Introduction/Contributions, Related Work, Background, Conclusion; Methods/Results sections between these read in part).
**Mechanism:** Across a large study of self-supervised/foundation-model backbones, Mahalanobis-based OOD detection performance is "strongly predicted" by two in-distribution geometry statistics: **local intrinsic dimensionality (LID)** and **within-class spectral decay (eigenvalue/scatter structure)**. Quoted from the Conclusion: "the product of Local Intrinsic Dimensionality and within-class scatter slope achieves strong correlations with detection effectiveness." Mahalanobis distance assumes class-conditional Gaussianity with a specific covariance structure; when the local manifold is complex (high LID) or the spectral decay is shallow (variance spread across many directions rather than concentrated), the quadratic-form distance stops tracking meaningful separation, independent of whether the class/attribute information is still linearly present.
**Evidence strength:** Level A (read directly), large-scale empirical study across many backbones, explicit regression linking geometry statistics to detector performance (R²=0.25–0.47 for predicting optimal normalization).
**Supports or contradicts Paper 3:** **Supports.** This gives a concrete, falsifiable candidate explanation for the CNN-era dissociation: if domain-adversarial disentanglement training changes LID and/or spectral decay of z_l (e.g., makes the domain-relevant subspace higher-LID or shallower-decaying) without removing the domain direction from the linear span, Mahalanobis would degrade exactly as observed while a linear probe would not. This is directly testable by computing LID and spectral-decay slope on Paper 3's existing checkpoints (retrospective) or on new FM checkpoints (prospective).

### 2. Training can concentrate variance in the task-discriminative subspace and suppress orthogonal directions (neural collapse)
**Source:** General neural-collapse literature, specifically "Beyond the Class Subspace: Teacher-Guided Training for Reliable Out-of-Distribution Detection in Single-Domain Models" (arXiv:2603.11269) — found via search-engine summary (Level B/C, not yet read at primary-source level).
**Mechanism:** Supervised training drives representations toward a low-rank, class-aligned structure — "variance is concentrated in a class-discriminative subspace, while orthogonal directions are suppressed for both in-distribution and out-of-distribution samples" (as summarized). If the training objective (task loss + adversarial domain loss) similarly concentrates variance into the task-relevant subspace, the domain-relevant subspace could be *suppressed in variance* — meaning it contributes little to the overall covariance structure Mahalanobis relies on — while still being present as a low-variance but linearly separable direction that a probe, which optimizes specifically for that direction, can still find.
**Evidence strength:** Level B/C (search-summary only) — needs a primary-source read before being cited as fact, but the underlying "neural collapse" phenomenon itself is well-established general theory.
**Supports or contradicts Paper 3:** **Supports**, with the same falsifiable structure as Mechanism 1 — suppressed-but-present variance in a specific direction is a distinct, complementary hypothesis to LID/spectral-decay (a direction can be linearly present while contributing negligibly to a covariance-based distance metric).
**Related:** "MahaVar: OOD Detection via Class-wise Mahalanobis Distance Variance under Neural Collapse" (arXiv:2605.14413) formalizes a connection between neural-collapse geometry and Mahalanobis-distance variance behavior directly — Tier 1 candidate for a follow-up read, not yet done.

### 3. Local intrinsic dimensionality rises under adversarial-style perturbation of the representation
**Source:** Adversarial-example/LID literature, e.g. "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality" (arXiv:1801.02613) — found via search-engine summary (Level B/C).
**Mechanism:** In the adversarial-robustness literature (a different but structurally related setting — adversarial *perturbation of inputs*, not adversarial *training of representations*), local intrinsic dimensionality around a point rises when the local geometry becomes more complex/"brittle." If domain-adversarial (GRL) training has an analogous effect on the *representation* geometry — raising local complexity specifically around the axis it's trying to erase — this would independently raise LID in exactly the region Mechanism 1 identifies as harmful to Mahalanobis.
**Evidence strength:** Level B/C, and note this is evidence from a different setting (adversarial input perturbation, not adversarial representation training) being used as an analogy, not a direct precedent. Flagged as the weakest-evidenced mechanism in this set — useful for hypothesis generation, not for citation as established fact.
**Supports or contradicts Paper 3:** **Supports, weakly / by analogy.**

### 4. Distance concentration: in high dimensions, all pairwise distances become similar, degrading distance-based methods specifically (not linear methods)
**Source:** "Interpreting the Curse of Dimensionality from Distance Concentration and Manifold Effect" (arXiv:2401.00422) — found via search-engine summary (Level B/C).
**Mechanism:** As dimensionality increases, Euclidean-type distances between points concentrate (the ratio of farthest to nearest distance approaches 1), making "far from centroid" progressively less meaningful as a signal — a generic property of high-dimensional Euclidean geometry, independent of any specific training intervention. Linear classifiers (probes) are not subject to the same failure mode because they operate via inner products/projections onto a learned direction, not via absolute distances in the full space.
**Evidence strength:** Level B/C (search-summary only), but this is a well-established general result in the high-dimensional statistics literature, not a novel or contested claim.
**Supports or contradicts Paper 3:** **Supports, as background/baseline mechanism** — this would predict that *any* sufficiently high-dimensional representation should show some Mahalanobis/probe gap, which is a confound Paper 4 needs to control for (e.g., by holding dimensionality fixed across intervention-strength conditions, as Paper 3 already does with its 16-d representation).

### 5. Self-supervised/foundation-model representations have specific linear-separability properties that may differ from supervised CNN representations
**Source:** "On Linear Separation Capacity of Self-Supervised Representation Learning" (arXiv:2310.19041) — found via search-engine summary (Level B/C, not yet read).
**Mechanism:** Provides theoretical conditions under which SSL-learned representations are linearly separable for downstream tasks. Relevant to the FM side of Paper 4's comparison: if FM/SSL representations are *more* linearly separable by construction than supervised CNN features (a property of the pretraining objective, not of any downstream intervention), a probe could recover domain information more easily or more robustly in FMs than in CNNs even before any disentanglement intervention is applied — a potential confound or boundary condition, not yet distinguished from the disentanglement-intervention effect this project wants to isolate.
**Evidence strength:** Level B/C, theoretical paper, not yet read at primary-source level.
**Supports or contradicts Paper 3:** **Neutral / boundary-condition relevant** — doesn't directly support or contradict the dissociation mechanism, but flags a needed control (measure probe-decodability of the *baseline*, non-disentangled FM before attributing any change to the intervention).

---

## Mechanism table

| Mechanism | Established in CNNs? | Established in FMs? | Evidence level | Status |
|---|---|---|---|---|
| Mahalanobis performance driven by LID × within-class spectral decay | Not directly tested by Paper 3 (Paper 3 measured condition number, not LID/spectral-decay specifically) | ✔ (2510.15202, Level A, large-scale FM study) | Strong (Level A) | **Primary candidate mechanism** |
| Neural collapse — variance concentrates in task subspace, suppresses orthogonal directions | ✔ (general neural-collapse literature, well established for supervised CNNs) | Unclear — not yet checked whether neural-collapse literature has been extended to FMs/SSL | Moderate (Level B/C, unread primary source) | Candidate |
| Local intrinsic dimensionality rises under adversarial-style training | ✔ for adversarial *input* perturbation; NOT established for adversarial *representation* training (GRL) specifically | Unknown | Weak (Level B/C, analogy only) | Candidate, weakest evidence |
| Distance concentration in high dimensions | ✔ (general high-dim statistics result, dimension-dependent not architecture-dependent) | ✔ (same, architecture-independent) | Strong as general theory (Level B/C for this specific citation, but the underlying math is standard) | Background/confound to control for |
| SSL/FM representations may be more linearly separable by construction | N/A (this is specifically an FM/SSL property) | ✔ (2310.19041, Level B/C, theoretical) | Moderate (Level B/C, unread) | Boundary condition to control for, not a mechanism for the dissociation itself |

---

## Synthesis: possible mechanistic explanations shared across CNNs and foundation models

Two mechanisms look like real, falsifiable candidates for *why* Paper 3's dissociation would (or
would not) replicate in a foundation model, and they are not mutually exclusive:

1. **Geometric-assumption violation (LID/spectral decay):** disentanglement training reshapes
   the local manifold structure and/or eigenvalue spectrum of the representation in ways that
   violate the (near-)Gaussian, low-LID assumptions Mahalanobis-family estimators depend on,
   without removing the linear separability a probe exploits. This is the best-evidenced
   candidate (Level A) and, notably, was demonstrated on foundation-model backbones already —
   just not under a disentanglement/debiasing intervention, which is exactly Paper 4's gap.
2. **Variance suppression without erasure (neural collapse):** the training objective can drive
   the domain-relevant direction toward low variance (making it a small contributor to the
   covariance Mahalanobis uses) while leaving it linearly present at a magnitude a supervised
   probe can still detect (since probes are not weighted by variance the way a Gaussian-model
   distance is).

Both mechanisms make the same qualitative prediction Paper 3 observed (geometry changes,
distance-based estimator degrades, probe still works) via different quantitative routes — (1) via
manifold/spectral shape, (2) via variance allocation. They are jointly measurable: Paper 4's
experimental design should compute LID, spectral-decay slope, and per-direction variance
alongside condition number (Paper 3's original geometry metric) so the intervention-strength
ladder can distinguish which mechanism, if either, is doing the work. Distance concentration
(mechanism 4) is a background confound rather than an explanation of the *intervention effect*
specifically, since it would apply equally at every disentanglement strength if dimensionality is
held fixed — useful as a sanity check, not as the headline mechanism.

No experiments are proposed here, per scope. This synthesis is meant to inform Step 6 design
choices (which geometry metrics to log) once the Research Gate (see `03_Novelty_Threats.md`)
opens.
