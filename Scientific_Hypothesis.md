# Scientific Hypothesis — Paper 4

No experiments. No methods. No datasets. This is the pivot document between literature review
and experimental design — what Paper 4 would actually be explaining, stated as a falsifiable
claim, before any protocol is written.

---

## Observation (established — Paper 3)

Under a domain-adversarial disentanglement dose-response ladder (λ_orth = 0, 1, 5; same
architecture, same 16-d representation), representation geometry changes substantially — condition
number shifts two orders of magnitude (Kendall's τ = 0.84, p = 2.8×10⁻⁵). This geometric change is
**not** accompanied by any improvement in distance-based reliability estimation: Mahalanobis-AUROC
stays flat and below chance (≈0.40) at every disentanglement level, and 6 of 7 additional scorers
spanning distinct principles (cosine-to-centroid, kNN, energy, ViM, KDE) converge on the same null
result. Yet a supervised linear probe, with no access to the training objective, recovers domain
membership at 0.72–0.81 AUROC from the *same* embeddings. **Information can remain linearly
decodable while becoming inaccessible to non-probing reliability estimators.**

This is a fact about one architecture family (EfficientNet-B3 dual-encoder), one intervention
(GRL + orthogonality), one domain (dermoscopy vs. smartphone photography). It is not yet a
mechanism, and it is not yet known whether it is architecture-specific or general.

## Mechanism (candidate explanations — Cluster B/E, this session)

Two non-exclusive candidates, both making the same qualitative prediction via different
quantitative routes:

**Mechanism A — geometric-assumption violation.** Mahalanobis-family estimators assume
(near-)Gaussian, low-local-complexity class-conditional structure. Across a large study of
self-supervised/foundation-model backbones, Mahalanobis OOD performance is predicted by the
product of local intrinsic dimensionality (LID) and within-class spectral-decay slope — properties
of the representation's local manifold shape and eigenvalue structure, not of how much information
it contains (arXiv:2510.15202, read at primary-source level). A training intervention that reshapes
LID or spectral decay along the suppressed direction, without removing that direction from the
representation's linear span, would break a distance-based estimator while leaving a targeted
linear probe unaffected.

**Mechanism B — variance suppression without erasure.** General neural-collapse-style theory holds
that training can concentrate variance into a task-discriminative subspace while suppressing
orthogonal directions. If the domain-relevant direction is suppressed to low variance but not
erased, it contributes little to the covariance structure a distance-based estimator relies on,
while remaining detectable to a probe that directly optimizes for it — probes are not weighted by
variance the way a covariance-based distance is (Level B/C — not yet read at primary-source level,
flagged as less certain than Mechanism A).

## Hypothesis (revised 2026-08-04, session 6 — relabeled after Gallo et al. verification)

**Locked framing: Geometry-mediated accessibility after representation intervention.**

```
Representation intervention → Geometry changes → Accessibility changes → Reliability changes
```

**The dissociation between probe-decodability and distance-based reliability-estimator performance
is not an artifact of the specific architecture (CNN), the specific intervention (GRL +
orthogonality), or the specific domain (dermoscopy/smartphone) used in Paper 3. It is a general
property of how a representation-reshaping intervention — of which domain-adversarial
disentanglement is one instantiation, not the definition — changes local dimensionality, spectral
structure, and variance allocation in ways that violate the assumptions specific distance-based
estimators depend on, while leaving the suppressed direction accessible to any method that
directly optimizes for it. Under this hypothesis, the mediating variable is representation
geometry, not the intervention family, and not the architecture that produced the representation.**

**Why the label changed:** the hypothesis was previously named around its first tested
instantiation ("domain-adversarial disentanglement"), which Reviewer #3's logic attack
(`19_Reviewer3_Logic_Attack.md`, Leap 3) already flagged as a naming choice that smuggled in more
specificity than the argument earns. The Gallo et al. full-text read (2026-08-04) makes this
concrete rather than abstract: Gallo's fine-tuning-intensity ladder is *itself* a representation
intervention (ordinary fine-tuning, not adversarial) that produces a geometry-flavored effect they
describe qualitatively ("the margin between in-distribution and semantically adjacent conditions
is compressed") and a measured reliability-estimator consequence (near-OOD AUROC drops from ≈96%
to ≈83–85% as fine-tuning intensity increases) — without ever testing geometry quantitatively or
probe-decodability at all. That is a second, independently observed case consistent with the same
general chain, using a different intervention type. Naming the hypothesis after disentanglement
specifically would have made Gallo's finding look like a different phenomenon; naming it after the
chain makes Gallo's finding a second convergent data point the chain should also explain — and
exposes the specific gap Gallo leaves (no geometry metric, no probe, no decodability test) as
exactly what Paper 4 supplies.

**Domain-adversarial disentanglement (GRL + orthogonality) remains the primary instantiation Paper
4 tests directly**, because it is the one Paper 3 already characterizes in detail and the one
Papers 1–3 give us the tooling to reproduce. Ordinary fine-tuning intensity (Gallo-style) is now a
required **control arm**, not a separate paper's finding to work around — see
`22_Experimental_Design.md`. This also preempts an obvious reviewer question ("what if you'd used
adversarial debiasing instead of GRL, or plain domain adaptation?") by stating up front that the
claim is about the chain, with GRL+orthogonality as the tested instance, not a claim that only GRL
produces the effect.

This is the reframe: Paper 4 is not "do CNN findings generalize to foundation models." It is
**representation intervention → geometry → accessibility → reliability**, with intervention type
(disentanglement vs. ordinary fine-tuning vs., in principle, other suppression-style objectives)
and architecture (CNN vs. FM) both treated as manipulable factors that might or might not change
the geometry that actually does the mediating work — not as the object of study itself.

## Predictions (falsifiable)

**Prediction 1.** The magnitude of distance-based-estimator degradation under a
suppression-style intervention should co-vary with measurable shifts in LID and/or within-class
spectral-decay slope of the intervened direction — not simply with a scalar "intervention
strength" label, and not with probe accuracy (which should stay comparatively decoupled from
these geometry shifts).

**Prediction 2.** If a geometry metric (LID × spectral-decay slope, or per-direction variance of
the suppressed dimension) is held approximately constant across interventions nominally
different in strength, distance-based-estimator degradation should also stay approximately
constant. Geometry, not the intervention's nominal strength, is what should predict the
degradation. Conversely, an intervention engineered to suppress a direction's *salience* without
shifting LID/spectral-decay should not reproduce the Mahalanobis-style collapse.

**Prediction 3.** Probe-decodability and distance-estimator accessibility should be
conditionally independent given the geometry metrics — once LID, spectral-decay slope, and
variance-allocation are known, whether the representation came from a supervised CNN or a
foundation model should add no further predictive power for whether a distance-based estimator
fails.

## Falsifiable outcomes

**If the hypothesis holds:**
- Geometry metrics, not architecture family, predict distance-estimator failure across both CNN
  and FM representations subjected to comparable suppression-style interventions.
- The LID × spectral-decay relationship reported for foundation-model backbones in general OOD
  settings (arXiv:2510.15202) should replicate, at least directionally, when computed
  retrospectively on Paper 3's own existing CNN checkpoints — a test that requires no new
  training, no new architecture, and no foundation model at all, and could be attempted before
  any Step 6 protocol is designed.

**If the hypothesis is contradicted:**
- Distance-estimator failure occurs even when LID/spectral-decay are held constant across
  intervention strengths — implicating something specific to Paper 3's GRL/orthogonality setup
  rather than a general geometric mechanism.
- Or: foundation-model representations under an equivalent suppression intervention show a
  qualitatively different relationship between geometry metrics and distance-estimator
  performance than CNNs do — implicating the pretraining paradigm itself (e.g., contrastive
  objectives shaping ambient geometry) as a mediator that architecture-agnostic "geometry" alone
  does not capture.
- Or: probe-decodability turns out to be geometry-dependent too, in a way Paper 3's CNN data
  didn't reveal — in which case the "decodable but not accessible" framing itself needs revision,
  not just its explanation.

Any of these three outcomes would still be a publishable, scientifically meaningful result — the
hypothesis is falsifiable in a way that produces information regardless of which way it breaks.

---

This document does not select datasets, models, metrics implementations, or protocols in detail —
that is fully specified in `22_Experimental_Design.md`. Per the Research Gate in
`03_Novelty_Threats.md`, Gallo et al. is now Level A verified and the Gate is open
(2026-08-04, session 6, per Vinh's explicit decision) — Step 6 has begun.
