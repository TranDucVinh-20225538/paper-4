# Research Constitution — Paper 4

**Phase A (Concept & Protocol) closed 2026-08-04, session 9.** One page, by design. Everything
below is a compression of decisions already made and cited elsewhere in `Paper4/` — this document
adds no new content, only holds the line on what was already decided.

---

## What this project is NOT

- **NOT a benchmark paper.** The "feature-space beats logit-space on FM backbones" claim is
  confirmed, closed prior art (Gallo et al., Level A — `03_Novelty_Threats.md`). Re-showing it is
  not the contribution.
- **NOT a foundation-model comparison.** One FM backbone, one fallback. Prediction 3
  (architecture-agnosticism) is dropped (`21_Prediction3_Removed_Reevaluation.md`).
- **NOT an intervention paper.** Reference, Intervention, Conventional, and Adaptation are
  instruments for inducing geometry variation — not the object of study
  (`22_Experimental_Design.md` §3).
- **NOT an OOD detector paper.** Mahalanobis, kNN, energy, etc. are measurement instruments for
  accessibility, not the thing being proposed or optimized.

## What this project IS

- A **mechanistic** study — explaining a dissociation already observed (Paper 3), not
  re-discovering that it exists.
- A **causal hypothesis** — geometry mediates accessibility, tested with an actual causal
  manipulation (Gate 1 + normalization knob), not just correlation.
- A **representation-geometry** study — LID, spectral-decay slope, condition number, variance
  allocation are the primary quantities of interest.
- An **accessibility** study — the gap between what a representation *contains* (probe-decodable)
  and what a distance-based estimator can *use* (Mahalanobis-accessible).

## Rules that can never be broken

1. Never introduce an experiment without a claim it maps to in `02_Claim_Matrix.md`
   (`23_Experiment_Contract.md` §1, §7).
2. Never interpret a failed manipulation (Gate 1) as evidence against the hypothesis — it means
   the chain wasn't tested, not that it was falsified (`22_Experimental_Design.md` §3.5).
3. Never report mediation (C4) without every checkpoint involved having passed Gate 1.
4. Never add a post-hoc primary analysis. Anything not in `24_PreRegistration.md` §4–5 is
   exploratory, reported separately, and logged in §13's Deviations Log — never merged into the
   confirmatory result.
5. Every conclusion in the eventual paper must map back to a row in `02_Claim_Matrix.md`. If it
   doesn't, it doesn't belong in the paper.
6. Never lock a claim, threat, or conclusion on Level B/C evidence. Level A (primary source, or a
   pre-registered result that has passed its gates) only — the rule this entire project has run on
   since `07_Evidence_Register.md`.

## The protocol changes only for

- A logic or implementation error.
- New scientific evidence (Level A) that forces revision.
- A bug in the pipeline.

**Never because a result was disappointing.** A clean negative result, obtained honestly through a
gated protocol, is data — not a reason to reopen the hypothesis. Per `24_PreRegistration.md` §7,
"we understand why accessibility changes" is the success criterion, not any specific number moving
in a specific direction.

## The 24-Hour Rule (Phase B, added 2026-08-04)

If a result is surprising: no hypothesis edit, no protocol edit, no claim change for 24 hours.
Allowed in that window: check for bugs, check the implementation, log the observation in
`26_Lab_Notebook.md`. Only after 24 hours does "is this a deviation?" get asked — and it's answered
by `24_PreRegistration.md` §13's trigger table, not by impulse.

## Success criterion

Not: *Mahalanobis-AUROC improves by X%.*
Is: ***We understand why accessibility changes.***

Every outcome in `22_Experimental_Design.md` §7 — support, contradiction, or partial mediation —
satisfies this criterion if it is reached honestly through the gated protocol. None of them
requires the hypothesis to be right.

---

## Phase A checklist (closed)

Literature ✅ · Novelty ✅ · Hypothesis ✅ · Mechanism ✅ · Experimental Design ✅ · Experiment
Contract ✅ · Pre-Registration ✅

From the first line of code onward: day-to-day observations go in `26_Lab_Notebook.md`
(append-only); anything that becomes an actual deviation, after the 24-Hour Rule's waiting period,
goes in `24_PreRegistration.md` §13's Deviations Log — not edits to this constitution, the
hypothesis, or the Claim Matrix.
