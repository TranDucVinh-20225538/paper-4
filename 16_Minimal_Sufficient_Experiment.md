# Minimal Sufficient Experiment — Paper 4

Still no experiment is being launched — Research Gate stays closed. This is a design exercise:
given a hard constraint (1 GPU, 6 weeks, 4 candidate foundation models, 1 dermatology
dataset-resource), what is the smallest experiment that could actually falsify the hypothesis?
"One dermatology dataset" is read as the existing ISIC↔PAD-UFES-20 testbed already used by
Papers 1–3 — the only option that requires zero new data collection, which the time budget
cannot absorb anyway.

## Principle

Cut everything that isn't logically required to test Predictions 1 and 2 (Medium and
Low-Medium confidence, per `14_Falsification_Matrix.md`). Prediction 3 (Low confidence, requires
wide geometric diversity across multiple architectures to even be testable) is explicitly **not**
in the minimal core — see `21_Prediction3_Removed_Reevaluation.md` for why dropping it sharpens
rather than weakens the project. A minimal experiment sized for P1/P2 only fits the budget; one
sized for P3 does not.

## What's cut, and why

- **3 of the 4 candidate foundation models.** Breadth across architectures only matters for
  Prediction 3. Testing one well-chosen FM either replicates the CNN-era pattern or it doesn't;
  that's the falsifying result. Spending a 1-GPU, 6-week budget spread across 4 backbones instead
  of depth on 1 would produce four shallow, underpowered results instead of one that could
  actually move the hypothesis.
- **Full fine-tuning of the FM backbone.** Not indispensable, and not affordable: full FT of a
  large pretrained backbone on 1 GPU in a 6-week window competing with everything else below is
  unrealistic. A frozen backbone + lightweight adapter/adversarial head trained with the
  disentanglement objective is the standard, cheap way to intervene on an FM representation, and
  it's also the more externally valid choice (this is how FMs are actually used in practice).
- **New dataset collection.** Not needed — the existing ISIC/PAD-UFES-20 pair is the acquisition
  shift the whole project line is built on; introducing a new dataset would also reopen every
  dataset-specific confound already catalogued in `11_Confound_Map.md` for no logical gain.
- **The three alternative-hypothesis distinguishing tests** (skewness/multimodality for Alt 1,
  probe-weight-diffuseness for Alt 2, sample-size sweep for Alt 3, from `13_Competing_Hypotheses.md`)
  are cut from the *indispensable* core, with one exception noted below — they help interpret an
  ambiguous result but aren't required to falsify the primary hypothesis.

## What's indispensable, and why

**1. Retrospective mechanism check on Paper 3's existing CNN checkpoints (Week 0, before touching
the GPU).** Compute LID and spectral-decay slope on data that already exists. Cost: ~zero GPU,
days not weeks. **Indispensable because it's a free, decisive gate** — per `Scientific_Hypothesis.md`'s
own falsifiable-outcomes section, if the mechanism doesn't hold on the data that motivated it, that
is direct information before any FM budget is spent (Risk Register item #6's fallback trigger).

**2. FM baseline probe, no intervention (Week 1).** Extract frozen FM embeddings on ISIC/PAD-UFES-20,
train the domain-leakage probe (reusing Paper 1's exact multi-seed + shuffle-label protocol).
Cost: forward passes only, no training. **Indispensable because of Risk Register item #3** — if
the shortcut isn't even present in the untouched FM baseline, there is nothing for a disentanglement
intervention to act on, and the whole downstream design is moot. This has to be checked before
investing in an intervention ladder, not after.

**3. Lightweight disentanglement/debiasing adapter on the frozen FM, at 2–3 intervention
strengths (Weeks 2–4, the main compute sink).** Mirrors Paper 3's λ_orth = 0, 1, 5 design as
closely as feasible with an adapter instead of full fine-tuning. **Indispensable — this is the
intervention ladder Predictions 1 and 2 are directly about**; without it there is no
intervention-strength variation to test geometry-tracking against.

**4. Geometry metrics (LID, spectral-decay slope, condition number, per-direction variance) and
the 8 reliability scorers (reused from Paper 3's implementation) computed on every checkpoint from
steps 2–3 (Week 5).** Cost: CPU-bound post-hoc computation, cheap. **Indispensable — this is the
measurement layer both predictions are stated in terms of**; without it there's no way to check
whether degradation tracks geometry or tracks the intervention label.

**5. The β-scaled ℓ2-normalization knob applied post hoc, at a fixed intervention strength, on
both the CNN retrospective data and the FM checkpoints (also Week 5, cheap — no retraining).**
**Indispensable specifically for Prediction 1's distinguishing control** (`10_Prediction_Audit.md`,
control #1) — without a way to move geometry independently of intervention strength, Prediction 1
cannot be distinguished from the "common cause" alternative explanation. This is the cheapest
possible version of a causal (not just correlational) test, and it's already a documented,
reusable technique from the primary mechanism source rather than something to invent.

**6. Analysis and write-up of the result either way (Week 6).** Not optional — a minimal
experiment that can falsify the hypothesis has to actually report the falsifying case, not just
the confirming one.

## What's cheap enough to keep even though not strictly indispensable

**Probe-weight-diffuseness check (Alt 2's distinguishing test).** This is nearly free — it's a
property of a probe that's already being trained in step 2/3, requiring no new computation beyond
inspecting the weight vector. Keeping it costs approximately nothing and directly helps interpret
an ambiguous P1/P2 result (distinguishing "geometry mediates" from "entangled encoding," Alt 2).
Everything else from the alternative-hypothesis tests stays cut from the minimal design.

## Six-week shape

| Week | Task | GPU cost |
|---|---|---|
| 0 | Retrospective LID/spectral-decay on existing Paper 3 CNN checkpoints — decisive gate | None |
| 1 | FM baseline domain-leakage probe (frozen embeddings) — second gate | Low (inference only) |
| 2–4 | Adapter-based disentanglement intervention, 2–3 strengths, on 1 FM | Main budget |
| 5 | Geometry metrics + 8 scorers + normalization-knob manipulation on all checkpoints, both architectures | Low (CPU-bound) |
| 6 | Analysis against Predictions 1 and 2; write up falsifying or confirming result | None |

This is deliberately a single-architecture-depth design, not a multi-architecture-breadth one. It
can falsify Predictions 1 and 2. It cannot, on its own, meaningfully test Prediction 3 — which is
exactly the argument for not carrying Prediction 3 as part of the core claim under this resource
constraint (see `21_Prediction3_Removed_Reevaluation.md`).
