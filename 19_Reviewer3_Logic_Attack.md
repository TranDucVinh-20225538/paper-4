# Reviewer #3 — Logical Consistency Attack

A skeptical theoretician's review. Ignores novelty and experimental design entirely (that was
Reviewer #2, `12_Reviewer_Premortem.md`). Attacks only whether the argument, as stated in
`Scientific_Hypothesis.md` and `08_Mechanism_Map.md`, actually follows from its own premises.

---

## Logical leaps

**Leap 1 — Cross-regime induction without argument.** The hypothesis imports "LID×spectral-decay
predicts Mahalanobis performance" from a study of *ordinary pretraining/fine-tuning variation* in
general-vision OOD detection (arXiv:2510.15202) and applies it to *adversarial suppression
training* (domain-disentanglement) in a completely different domain. Nothing in the argument
establishes, or even attempts to establish, why a statistical relationship found under one kind of
training variation should transfer to a mechanistically different kind of training intervention.
The source paper contains no adversarial objective at all. This is an inductive leap across
regimes with no bridging argument — only the assertion that it's "a candidate."

**Leap 2 — Correlation-to-causation, stated as if already addressed.** Paper 3's actual finding is
a correlation over an ordinal variable (intervention strength): geometry and accessibility both
move together across λ_orth = 0, 1, 5. The hypothesis document states this as "geometry mediates
accessibility" — causal language — before any test has distinguished mediation from common-cause
correlation. `10_Prediction_Audit.md` and `15_Causal_Graph.md` both later acknowledge this gap and
propose ways to close it, which is good practice, but the *hypothesis statement itself* asserts
the causal conclusion up front rather than provisionally. The document's structure (state
hypothesis, then later admit it needs testing) inverts the correct order for a genuinely
skeptical presentation.

**Leap 3 — The name presupposes the result.** Calling the hypothesis "architecture-agnostic
geometry mediation" before any foundation-model evidence exists smuggles in the conclusion of
Prediction 3 (architecture doesn't matter) into the *label* of the hypothesis being tested. A
reader encountering the name before the predictions could reasonably believe the architecture-
independence question is already settled rhetorically, even though `13_Competing_Hypotheses.md`
and `14_Falsification_Matrix.md` both later treat it as the least-confident, least-tested part of
the whole project. Naming a hypothesis after its strongest possible confirmation is a mild but
real form of begging the question.

## Unsupported inferences

**Inference 1 — "Geometry reshapes" does not, by itself, imply "probes stay robust while distance
methods specifically fail."** The argument establishes (via the primary mechanism source) that
geometric shifts *can* degrade Mahalanobis-style methods. It does not establish, from any first
principle, why the *same* geometric shift should leave a linear probe's performance comparatively
unaffected. The hypothesis document treats probe robustness as needing no explanation ("probes are
not weighted by variance the way a Gaussian-model distance is" — Mechanism B) while treating
distance-method failure as needing a detailed geometric account. This is an asymmetric standard of
rigor: one side of the dissociation is explained, the other is asserted. A fully symmetric
argument would need to show *both* that geometry breaks distance methods *and* that the same
geometric regime provably preserves linear separability — the second half is currently missing.

**Inference 2 — Postdiction presented with the confidence of prediction.** Mechanism B was
selected, in part, *because* it would explain the already-observed pattern (Paper 3). Its
plausibility is then cited as support for the pattern being explicable. This is abductive
reasoning — reasonable as a starting point for hypothesis generation, but the Mechanism Map labels
it "Supports" with the same verb used for the better-evidenced Mechanism A, which had actual
predictive power demonstrated *before* being applied to this problem. The document does not
clearly distinguish "we found a pre-existing account with independent predictive success" (Mechanism
A) from "we constructed/selected an account that happens to fit what we already saw" (Mechanism B)
— these deserve different epistemic weight and are currently presented at the same confidence
level in places.

## Circular arguments

**Circularity 1 — The retrospective test as evidence.** `Scientific_Hypothesis.md`'s
falsifiable-outcomes section lists "the LID×spectral-decay relationship replicates on Paper 3's
own existing CNN checkpoints" as something that would support the hypothesis. But Paper 3's data
is the *same data* that motivated the hypothesis's construction in the first place (the
dissociation was observed there, then Mechanism A was sought out to explain it). Confirming a
mechanism on the data used to discover the need for a mechanism is, without independent data, not
confirmatory — it can at best show internal consistency. `17_Dependency_Graph.md` correctly labels
this "R0" as a *gate*, not a *conclusion*, which is the right move — but the original hypothesis
document's own falsifiable-outcomes section does not make this distinction as sharply as it
should; a careless reading could treat R0 succeeding as real support rather than a sanity check.

## Unstated assumptions (logical, not empirical, framing)

**Unstated 1 — "Geometry" and "information content" are treated as separable factors that can
independently vary and jointly determine accessibility.** But for the Mahalanobis distance
specifically, geometry (the covariance structure Σ) and information content (whether class means
differ) are not independent inputs to two different processes — they are both literally
parameters of the *same* formula, (z−μ)ᵀΣ⁻¹(z−μ). Asking "does geometry or information content
explain the failure" may be a category error if the estimator's own mathematical definition
doesn't permit cleanly separating the two. The hypothesis document does not address whether
"geometry mediates accessibility" is a well-posed causal claim or a restatement, in causal
language, of a definitional entanglement inside the estimator being studied. This is arguably the
deepest issue on this list: it is not obviously false, but it is not obviously a *causal* claim
either, and the document doesn't examine which it is.

**Unstated 2 — That a falsifiable *prediction* structure guarantees a well-posed *hypothesis*.**
The document's closing line ("the hypothesis is falsifiable in a way that produces information
regardless of which way it breaks") is true of the predictions as instrumentally stated, but
falsifiability of predictions doesn't by itself establish that the underlying causal claim
("geometry mediates accessibility") is coherently stated (see Unstated 1). A hypothesis can
generate falsifiable, well-defined statistical predictions while still resting on an ill-posed
causal foundation — Popperian falsifiability is necessary, not sufficient, for a hypothesis being
sound.

---

## Net logical assessment

The predictions themselves (P1, P2, P3) are each individually well-formed and testable — this
document's objections are about the *hypothesis statement and its justificatory apparatus*, not
about whether the predictions can be checked. The most serious issue is Unstated 1: before
building more infrastructure around "geometry mediates accessibility," it is worth explicitly
determining whether that phrase describes a testable causal relationship between independent
factors, or is a causal-sounding restatement of the fact that Mahalanobis distance is
mathematically a function of both Σ and (z−μ) at once. If it's the latter, the finding is still
real and interesting, but the correct framing is closer to "Mahalanobis-specific parametric
sensitivity" than "an architecture-agnostic law about geometry and accessibility in general."
