# Paper 4 — One-Page Summary

**Observation.** Under a domain-adversarial disentanglement dose-response ladder on a supervised
CNN (EfficientNet-B3 dual-encoder, 16-d representation, λ_orth = 0, 1, 5), representation geometry
changes substantially (condition number shifts two orders of magnitude, Kendall's τ=0.84). This
geometric change is not accompanied by any improvement in distance-based reliability estimation —
Mahalanobis-AUROC stays flat and below chance (≈0.40) at every level, and 6 of 7 additional
scorers spanning distinct principles converge on the same null result. Yet a supervised linear
probe recovers domain membership at 0.72–0.81 AUROC from the same embeddings. Information can
remain linearly decodable while becoming inaccessible to non-probing reliability estimators.

**Mechanism.** Two non-exclusive candidates. (A) Mahalanobis-family estimators assume
near-Gaussian, low-local-complexity structure; across a large study of foundation-model backbones,
Mahalanobis OOD performance is predicted by local intrinsic dimensionality × within-class
spectral-decay slope (arXiv:2510.15202, Level A) — a training intervention that shifts these
properties along a suppressed direction, without removing that direction from the linear span,
would break distance estimation while leaving a targeted probe unaffected. (B) Neural-collapse-
style theory: training can concentrate variance into a task-discriminative subspace while
suppressing orthogonal directions, leaving them linearly present but low-variance — detectable to
a probe, invisible to a covariance-weighted distance metric (Level B/C, unread at primary source).

**Hypothesis (relabeled 2026-08-04, session 6): Geometry-mediated accessibility after
representation intervention.** The probe/distance dissociation is not an artifact of CNN
architecture, the specific GRL+orthogonality objective, or the dermatology domain used to first
observe it. It is a general property of how a representation-reshaping intervention — of which
domain-adversarial disentanglement is the primary tested instantiation, not the definition —
reshapes representation geometry in ways that violate distance-estimator assumptions while leaving
information linearly accessible. Representation intervention → geometry → accessibility →
reliability, with intervention type and architecture both manipulable factors, not the object of
study. Gallo et al. (CVPR 2026W, now Level A) supplies a second, independently observed instance
of the first half of this chain (fine-tuning intensity → qualitative geometry compression → near-OOD
AUROC drop) without ever testing geometry quantitatively or probe-decodability — exactly the gap
Paper 4 fills, and now a required control arm rather than a competing paper.

**Predictions.**
1. Distance-estimator degradation co-varies with LID/spectral-decay shifts specifically, not with
   the intervention-strength label or with probe accuracy. *(Confidence: Medium.)*
2. Holding geometry constant across different intervention strengths (via post-hoc normalization
   manipulation) holds degradation constant too — the causal, not merely correlational, version of
   (1). *(Confidence: Low-Medium — a clean dissociating manipulation may be hard to construct.)*
3. Once geometry is known, architecture (CNN vs. foundation model) adds no further predictive
   power for distance-estimator failure. *(Confidence: Low — requires wide geometric diversity
   across multiple architectures that a resource-constrained study likely cannot sample; see
   `21_Prediction3_Removed_Reevaluation.md`.)*

**Alternative hypotheses** (none preferred; see `13_Competing_Hypotheses.md`): (1)
distributional-family mismatch (Gaussianity violated, not geometry per se) — in tension with
Paper 3's own data, since KDE, a non-parametric scorer, also failed; (2) entangled/distributed
encoding — domain information spread across many dimensions, invisible to unsupervised/marginal
methods but recoverable by any supervised linear map; testable almost for free via probe-weight
concentration; (3) finite-sample estimation artifact — covariance estimation, not true geometry,
degrades with a shrinking-variance direction; testable via a sample-size recovery sweep.

**Falsification criteria.** The hypothesis is contradicted if: distance-estimator failure persists
even when geometry is held constant (implicates something CNN/GRL-specific, not general geometry);
or foundation-model representations show a qualitatively different geometry-accessibility
relationship than CNNs do (implicates the pretraining paradigm as the true mediator); or
probe-decodability itself turns out to be geometry-dependent (the "decodable but not accessible"
framing needs revision, not just its explanation). All three outcomes remain scientifically
informative and publishable — this is a deliberate design property, not an accident.

**Open logical concern (per `19_Reviewer3_Logic_Attack.md`), not yet resolved:** for the
Mahalanobis distance specifically, geometry (Σ) and information content (whether class means
differ) are parameters of the same formula, (z−μ)ᵀΣ⁻¹(z−μ) — not obviously independent factors.
"Geometry mediates accessibility" may describe a genuine causal relationship between separable
quantities, or it may restate a definitional entanglement inside the estimator being studied. This
distinction is not yet settled and bears on how strongly the eventual paper can claim generality
beyond Mahalanobis-family methods specifically.

*If it doesn't fit above this line, it isn't essential yet.*
