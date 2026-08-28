# Hidden Assumptions Audit — Paper 4

Every assumption currently load-bearing in `Scientific_Hypothesis.md` and `08_Mechanism_Map.md`
that isn't stated outright. For each: why it exists, whether literature supports it, whether it's
falsifiable, and whether the hypothesis actually needs it.

---

### Assumption 1 — LID and spectral-decay slope are a *sufficient* description of the geometry that matters

**Why it exists:** These are the two summary statistics the one Level-A mechanism source
(arXiv:2510.15202) happened to find predictive, in a different setting (ordinary pretraining/
fine-tuning variation, no adversarial objective). They were adopted for precedent, not derived
from first principles for this project's specific intervention.

**Literature support:** Partial — supports predictive power of these two statistics in the
setting that paper studied. Says nothing about completeness (that no other geometric quantity
would explain more).

**Falsifiable:** Yes — if, after controlling for LID×spectral-decay, a different quantity
(e.g., between-domain mean separation, flagged as a distinct alternative in
`10_Prediction_Audit.md`'s P2 audit) explains residual variance in Mahalanobis degradation, this
assumption is wrong as stated.

**Do we need it?** Only in its strong form. It can be weakened to "*some* geometric summary
mediates accessibility" without picking LID×spectral-decay specifically — a more defensible,
still-falsifiable claim that doesn't commit to one paper's particular operationalization. This is
the single weakest link in the current formulation.

---

### Assumption 2 — Domain-adversarial (GRL) training is mechanistically analogous to the general class of "suppression-style interventions" neural-collapse theory describes

**Why it exists:** Needed to import Mechanism B at all. Neural collapse is documented for
ordinary supervised classification training, not for adversarial representation-shaping
objectives specifically.

**Literature support:** Level B/C — the source itself (arXiv:2603.11269) is unread at
primary-source level per `07_Evidence_Register.md`'s standard.

**Falsifiable:** Yes, in principle — a synthetic/toy demonstration (suggested in
`12_Reviewer_Premortem.md`) that a GRL-style objective does or doesn't produce collapse-like
variance suppression would settle this directly.

**Do we need it?** Only for Mechanism B. Mechanism A (LID/spectral-decay) doesn't require this
assumption — it only requires that *some* geometric shift occurs under the intervention,
regardless of which specific training-theoretic story explains why.

---

### Assumption 3 — A post-hoc, adapter-level intervention on a *pretrained* foundation model is comparable to a *from-scratch* joint-objective intervention on a CNN

**Why it exists:** Required for any CNN-vs-FM comparison to mean what the hypothesis wants it to
mean. Paper 3's CNN was trained with the disentanglement objective from initialization; the
proposed FM experiment (per `16_Minimal_Sufficient_Experiment.md`) necessarily uses a lightweight
adapter on frozen, already-pretrained weights, for resource reasons.

**Literature support:** None found this session. This is arguably the single most load-bearing,
least-scrutinized assumption in the entire project — and it echoes exactly what Gallo et al.'s own
abstract (per Level B/C search summaries) already flags as mattering: adaptation intensity/depth
changes OOD-detection behavior on its own, independent of any disentanglement-specific effect.

**Falsifiable:** Yes — pretrained-then-adapted and from-scratch-trained interventions could be
shown to behave qualitatively differently even absent any geometry or architecture story, purely
because of adaptation depth.

**Do we need it?** Yes, unavoidably, for the FM arm to exist within the resource constraint — but
it must be stated explicitly as a scope limitation in any write-up, not silently assumed. A
full-fine-tuning FM condition, even a small one, would be the natural way to test whether
adaptation depth itself is a confound here, though it's cut from the minimal design on cost
grounds (see `16_Minimal_Sufficient_Experiment.md`).

---

### Assumption 4 — Linear-probe accuracy is a stable, intervention-invariant proxy for "how much domain information is present," not an artifact of probe training itself

**Why it exists:** The entire "decodable but not accessible" framing depends on probe accuracy
meaning what it claims to mean, not drifting for reasons unrelated to representation content
(e.g., regularization interacting differently with different geometry).

**Literature support:** Reasonably strong — Papers 1 and 3 already use multi-seed probing and a
shuffle-label sanity check, which is exactly the standard defense against this concern.

**Falsifiable:** Yes, and already partially tested.

**Do we need it?** Yes — this is inherited from Paper 3, not new to Paper 4, and is the
best-supported assumption on this list.

---

### Assumption 5 — The specific 8 distance/density-based scorers tested are representative of "distance-based reliability estimators" as a general category

**Why it exists:** Needed to license the broad phrase "distance-based estimators fail" rather
than the narrower, more defensible "these 8 specific implementations fail."

**Literature support:** Moderate — 6 of 7 additional scorers converging in Paper 3 is decent
internal evidence for a shared underlying vulnerability, but it remains a fixed, finite,
researcher-chosen set.

**Falsifiable:** Yes, trivially — a 9th, sufficiently different scorer succeeding where the other
8 failed would falsify the general-category claim while leaving the specific-8 claim intact.

**Do we need it?** No — this can and should be scoped down to "the tested class of estimators"
rather than a universal claim about all possible distance-based methods, at no cost to the
hypothesis's substance.

---

### Assumption 6 — "Architecture" (CNN vs. foundation model) is a coherent, unitary causal category, such that one CNN and one-to-four FMs are representative of their categories

**Why it exists:** Implicit in the entire "architecture-agnostic" framing (Prediction 3) — without
this assumption, a single CNN-vs-single-FM comparison is a comparison of *two specific models*,
not of *two architectural categories*.

**Literature support:** Weak. Architectures within each category vary enormously (ResNet vs.
EfficientNet vs. ViT-tiny vs. ViT-giant; contrastive vs. masked-modeling pretraining), and
nothing read this session establishes within-category homogeneity for this specific
geometry-accessibility phenomenon.

**Falsifiable:** Yes, trivially — testing more than one CNN and more than one FM and finding
within-category variance comparable to between-category variance would falsify it.

**Do we need it?** This is the assumption most worth explicitly dropping. It is exactly the
assumption Prediction 3 depends on and that `10_Prediction_Audit.md` (item 3) and
`12_Reviewer_Premortem.md` both independently flag as the weakest part of the project. Dropping
it — i.e., not claiming "architecture" as a category, only reporting what was observed in the
specific instances tested — directly motivates the case made in
`21_Prediction3_Removed_Reevaluation.md`.

---

## Summary: which assumptions survive scrutiny, which don't

| Assumption | Needed? | Verdict |
|---|---|---|
| 1. LID×spectral-decay is sufficient | Only in strong form | Weaken to "some geometric summary" |
| 2. GRL ≈ general suppression-style intervention | Only for Mechanism B | Keep flagged as unverified; Mechanism A doesn't need it |
| 3. Adapter-on-pretrained ≈ from-scratch intervention | Yes, unavoidably | Keep, but state explicitly as a scope limitation |
| 4. Probe accuracy is a stable information proxy | Yes | Keep — well-supported, inherited from Paper 3 |
| 5. The 8 scorers represent "distance-based estimators" generally | No | Drop the universal phrasing; scope to the tested set |
| 6. Architecture is a unitary category | No | Drop — this is the assumption Prediction 3 needs and shouldn't get |

Two of six assumptions (5 and 6) can be removed outright at no cost to what the hypothesis
actually claims, and removing 6 in particular is the same move `21_Prediction3_Removed_Reevaluation.md`
argues for on independent (robustness, not just assumption-hygiene) grounds — both audits converge
on the same recommendation from different directions, which is itself a useful cross-check.
