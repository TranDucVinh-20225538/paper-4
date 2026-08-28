# Prediction 3 Removed — Re-Evaluation

Assume Prediction 3 (architecture-independence, once geometry is known) is dropped entirely from
the hypothesis. Does the scientific contribution weaken or sharpen?

## Verdict: sharpens.

## Why

**P3 was carrying more logical weight than it could support.** Per `18_Hidden_Assumptions.md`
(Assumption 6), P3 requires treating "architecture" as a coherent, unitary causal category — that
one CNN and one-or-two foundation models are representative of *all* CNNs and *all* foundation
models with respect to this phenomenon. Nothing read this session establishes within-category
homogeneity, and the assumption is falsifiable trivially (test more instances per category and
find within-category variance comparable to between-category variance). No other prediction in the
hypothesis depends on this assumption — P1 and P2 are both stated and testable within a single
architecture.

**P3 was the lowest-confidence prediction across every independent audit this session.**
`10_Prediction_Audit.md` flagged its central vulnerability (range-restriction — a null result is
indistinguishable from an underpowered one). `12_Reviewer_Premortem.md` identified it as the
review's central objection (a null-result prediction requiring a power analysis the project
doesn't have). `14_Falsification_Matrix.md` scored it Low confidence, the lowest of the three.
`15_Causal_Graph.md` identified a genuine statistical hazard specific to P3 — Representation
geometry is a collider between Architecture and Intervention strength, so conditioning on geometry
(exactly what testing P3 requires) risks inducing spurious associations that a naive regression
would not detect as spurious. `17_Dependency_Graph.md` shows Conclusion D (the claim P3 feeds) has
four independent points of failure, the most fragile node in the entire dependency structure. Four
separate analytical passes, using different methods (statistical, rhetorical, causal, graph-
theoretic), converged on the same weak point without being asked to agree with each other.

**P3 was also the most expensive prediction to test.** `16_Minimal_Sufficient_Experiment.md`
explicitly could not afford to test it under the stated resource constraint (1 GPU, 6 weeks) —
testing P3 properly requires wide geometric-range sampling across multiple architectures, which a
single-FM-depth design cannot provide. Every other prediction fits the minimal budget; only P3
required cutting.

**What's left without P3 is still a real, falsifiable, publishable claim.** Per
`17_Dependency_Graph.md`: dropping the requirement for Conclusion D (architecture-agnostic
mediation) leaves Conclusions B and C standing on their own — "geometry mediates accessibility
within a CNN setting" and, if the FM experiment succeeds, "geometry mediates accessibility within
a foundation-model setting, independently." Two independent confirmations of the same mechanism in
two different representation-generating processes is a legitimate, interesting contribution. It is
just a smaller claim than "architecture never matters" — and per `12_Reviewer_Premortem.md`'s
proposed redesign, a *smaller, better-defended* claim was already the recommended fix for the
paper's most serious anticipated objection.

## What changes concretely

- The paper's central RQ shifts from *"Does a geometric account of Mahalanobis failure hold
  regardless of architecture?"* to *"Does a geometric account of Mahalanobis failure, established
  under ordinary pretraining variation in general vision, also explain a probe/distance
  dissociation under a domain-disentanglement intervention — and does this hold independently in
  both a supervised CNN and a foundation-model representation?"* CNN and FM become two separate,
  parallel confirmatory tests of the same mechanism, not two halves of a single unified claim.
- `16_Minimal_Sufficient_Experiment.md`'s design does not need to change — it was already sized
  for P1/P2, not P3.
- `20_One_Page_Summary.md`'s Prediction 3 line would be removed or explicitly demoted to "out of
  scope for this paper, a natural extension" rather than a core claim.
- The Reviewer Pre-Mortem's central objection (null-result prediction without a power analysis)
  is preempted rather than defended against, which is a stronger position than any rebuttal could
  achieve.

## What is lost

The "architecture doesn't matter" claim was the most striking, most citable version of the
hypothesis — the one that would make Paper 4 feel like it settles something large. Dropping it
means Paper 4 becomes a more modest, more defensible paper: a mechanism replicated twice, not a
law established once. That is a real loss of rhetorical ambition. It is not a loss of scientific
content — everything falsifiable and testable in P1 and P2 survives untouched, and per
`19_Reviewer3_Logic_Attack.md`, the "architecture-agnostic" framing was already identified as
smuggling its own conclusion into the hypothesis's name (Leap 3) — removing P3 also removes that
specific logical liability.

**Net assessment: fewer, stronger predictions beat many weak ones here.** This re-evaluation
agrees with the principle stated in the prompt that motivated it.
