# Stage 4 — Intervention Design Choice

**Exactly two decisions, both left open by `AR-01_Resolution.md` and
`Stage4_Architecture_Instantiation_Contract.md` on purpose, resolved here.** No code. No
hyperparameters (adapter width/rank, projection strength values, optimizer settings) — those come
after this, in a future execution plan, once these two decisions are reviewed. This document exists
so that decision has its own record, same as every prior turning point in Stage 4's design.

## Decision 1 — Nuisance-direction mechanism

**Choice: closed-form / data-derived projection, not a learned-adversarial head.**

| | Learned-adversarial | Closed-form / data-derived |
|---|---|---|
| Needs | A trained nuisance head + adversarial/orthogonal objective | A target direction estimated from labeled data, then projected against |
| Data requirement | None beyond what's already labeled | Domain labels — **already exists**: `ReferenceTrainEmbedding`'s `domain` column (`isic`/`pad_ufes`), the same population Group B already fits on |
| Gate 1 causal story | Confounded: a geometry change could come from the adapter's own capacity *or* from adversarial training dynamics (convergence quality, seed, minimax instability) — genuinely hard to attribute | Direct: a specific direction is removed; any resulting LID/spectral-decay shift traces mechanically to *that* removal, not to a second model's training behavior |
| Precedent in `AR-01_Architecture_Study.md` | FairNVT, DiCoOp | Closed-Form Concept Erasure, Uni-AdaVD, Identity Sanitization Projection, O-TPT — **this family accounts for most of Tier 5's non-"No" rows**, i.e. most of the survey's geometry-analyzing papers already work this way |
| Dose/intensity ladder | Entangled with training convergence — "strength" partly reflects how well the adversarial game converged, not just a chosen setting | Deterministic given the data and a chosen projection strength — reproducible across seeds by construction |
| Moving parts | Two trainable components (adapter + nuisance head) in a minimax relationship | One trainable component (adapter) + one fixed, precomputed projection target |

**Reasoning.** The deciding factor is Gate 1's exact criterion
(`Stage4_Architecture_Instantiation_Contract.md` §3: *"did LID and/or spectral-decay slope actually
change"*) and this project's own controls discipline (`22_Experimental_Design.md` §5 — isolate one
variable at a time). A learned-adversarial head introduces a second trainable system whose own
training dynamics become a plausible alternative explanation for any observed geometry shift —
exactly the kind of confound Stage 0 through Stage 3 have been built, step by step, to avoid. A
closed-form projection has no second training process to entangle with: the target direction is
fixed before the adapter trains, so a measured geometry shift has one candidate cause, not two. This
also reuses infrastructure already built and already trusted (`ReferenceTrainEmbedding`'s domain
labels, the same closed-form/fit-from-labeled-data pattern Group A's probe and Group B's Mahalanobis
family already use) rather than adding a new one.

**Acknowledged, not hidden**: whether an output-stage projection alone moves LID/spectral-decay
slope *enough* to pass Gate 1 is an empirical question this document cannot resolve — that is
precisely what Gate 1 and the multi-strength ladder (§3, `Stage4_Execution_Plan.md`) exist to check,
not a risk this decision eliminates.

## Decision 2 — Injection location

**Choice: output-stage only — applied to the same 1024-d representation `forward_features()`
already produces, not to any internal transformer block.**

**Reasoning**, converging from three independent directions, not one:

1. **`AR-01_Resolution.md`'s own family conclusion** — "single-backbone, output-stage adapter or
   projection" — already points here; per-block insertion (LoRA/adapter in every attention layer)
   was the specific pattern flagged as *weak on interpretability*, since its effect diffuses across
   many layers with no single identifiable intervened direction.
2. **`Stage4_Architecture_Instantiation_Contract.md` §2's requirement** — output must land in the
   exact 1024-d space `ReferenceEmbedding`/`ReferenceTrainEmbedding` already occupy. An
   output-stage-only intervention satisfies this by construction: everything up through the
   second-to-last step of the frozen forward pass is untouched, byte-identical to the Reference Arm.
   Per-block insertion would require separately verifying the final space hasn't subtly drifted
   despite changes propagating through many intermediate layers — more surface area for exactly the
   kind of feature-space drift that would make a Stage 5 reviewer's objection (*"you changed the
   whole feature space, of course the scorer changed"*) valid instead of refutable.
3. **Reuses the Reference Arm's own extraction pipeline directly** — `ReferenceEmbedding`/
   `ReferenceTrainEmbedding` were both extracted via one call to `forward_features()`. An
   output-stage adapter means Stage 4's embeddings can be extracted with the same script pattern,
   plus one additional transformation step after it — not a new extraction pipeline. Less new code,
   less new surface area for bugs, direct methodological continuity with what's already validated.

Combined with Decision 1, the resulting shape is concrete at the family level without being an
implementation spec: **a trainable adapter/bottleneck applied to PanDerm's frozen 1024-d output,
regularized against a closed-form projection of the acquisition-domain direction estimated from
`ReferenceTrainEmbedding`'s labeled data** — nothing upstream of `forward_features()` is touched.

## What is still not decided

Adapter width/rank. Exact projection strength value(s) for the dose ladder (mapping onto the 2–3
Intervention-arm strengths, `22_Experimental_Design.md` §3). How the projection target direction is
estimated exactly (e.g. difference-of-class-means, LDA-style, PCA-on-domain-residual — a specific
estimator, not yet chosen). Optimizer, learning rate, training length for the adapter. All are
execution-level parameters, to be resolved and logged in a future execution-plan document, only
after this design choice is reviewed.

## Status

```
AR-01 Step 1:                 COMPLETE
AR-01 Step 2:                  COMPLETE
Instantiation Contract:        COMPLETE
Intervention Design Choice:
  Nuisance mechanism:          DECIDED — closed-form / data-derived projection
  Injection location:          DECIDED — output-stage only (post forward_features())
  Exact estimator + hyperparameters: OPEN

Training:                      NOT AUTHORIZED
```
