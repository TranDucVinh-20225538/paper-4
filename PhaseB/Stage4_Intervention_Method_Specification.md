# Stage 4 — Intervention Method Specification

**Locks the four items `Stage4_Intervention_Design_Choice.md` left open. Not a code plan — no
implementation, no training authorized.** Once this is reviewed, `Stage4_Execution_Plan_v2.md` (not
yet written) formally folds these decisions into the step sequence already established in
`Stage4_Execution_Plan.md` §1.

## 0. Resolving the "is this still an adapter?" question first

**Confirmed: yes, a genuine trainable adapter is required — a pure parameter-free projection is not
sufficient**, for two independent reasons, both already locked elsewhere:

1. `22_Experimental_Design.md` §3 names the Intervention arm "**Adapter**-based domain-adversarial
   training" — the word carries weight beyond "some function of z."
2. The **Conventional arm** is explicitly defined as "matched-budget ordinary fine-tuning" —
   "matched-budget... to the Intervention arm" (§3), required specifically to isolate intervention
   *type* from intervention *strength/budget* (Gallo verification, C4). A zero-parameter, untrained
   projection has no parameter count and no training-step count for Conventional to be matched
   against — this would make C4's entire control comparison undefined, not just weaker.

So: the Intervention arm keeps a real, trainable adapter with its own parameters and training steps.
What Decision 1 (`Stage4_Intervention_Design_Choice.md`) actually resolved is narrower than "no
adapter" — it resolved *how the adapter's disentanglement target is obtained*: closed-form/
data-derived, not from a second trained adversarial network. The adapter itself is trained;
only the *nuisance direction it's regularized against* is closed-form and fixed before training
starts. This is consistent with, not a contradiction of, Decision 1.

## 1. Nuisance-direction estimator

**Population check, done before choosing anything (verified this session, not assumed):** read
`CSG-SKin/src/datasets/splits.py` directly. `build_csg_train_dataframe` — CSG-Skin's own
domain-adversarial training data builder — uses ISIC-train **plus the full PAD-UFES population**
(`pad_df = df[df["domain"] == "pad_ufes"].copy()`, no split applied) as domain-adversarial training
input. `build_id_ood_test_dataloaders` uses that **same, full** PAD-UFES population as the OOD test
set. Papers 1–3's own precedent already reuses PAD-UFES for both the domain-training signal and OOD
evaluation — this is not a Paper-4-specific risk, it's the established methodology this project has
been instructed to mirror. Consequence: estimating a nuisance direction from `ReferenceTrainEmbedding`
(ISIC) + `ReferenceEmbedding`'s PAD-UFES portion needs **no new artifact and no new Execution
Amendment** — it reuses exactly the population split Papers 1–3 already used for this specific
purpose (domain-training signal), which is methodologically distinct from Group B's Mahalanobis-family
fitting (a different purpose — scoring against a reference population — which is why *that* needed a
strictly separate ISIC-only fitting population, EA-01). Both separations are correct for what they
each serve; neither generalizes to the other.

**Estimator choice: difference of domain-conditional means** (μ_isic − μ_pad_ufes, both computed
over frozen 1024-d embeddings, normalized to a unit direction). Not a trained classifier weight
vector (Option A), not a PCA/subspace (Option B). Reasoning:

- **Zero additional trained artifact** — the most direct extension of Decision 1's own logic
  (avoid entangling the nuisance target with any training process). A logistic-regression weight
  vector (Option A) is well-behaved but still an extra fitted object with its own regularization
  and convergence choices; a class-mean difference has none.
- **Already-precedented in this project**: Mahalanobis and cosine-to-centroid (`GroupBExecution/`)
  already fit per-class means from labeled data as their operative quantity — this reuses the same
  pattern, just over domain labels instead of lesion-class labels.
- **Precedented in the literature survey**: `AR-01_Architecture_Study.md`'s "Ablating Protected
  Attribute Reps." (arXiv:2410.13976) uses exactly this mean-difference-direction approach,
  training-free.
- **Textually consistent with the protocol**: `22_Experimental_Design.md` §4 measurement 3 says
  "per-direction variance of **the** intervened direction" — singular. A subspace (Option B) would
  require reading "direction" as "subspace," a stretch the text doesn't support. A single
  mean-difference direction matches the wording as written.

**Not chosen, but not discarded**: the domain-leakage probe's own weight vector (Option A) already
exists for free (`ReferenceArmBaseline/BaselineReport/`) and can serve as a robustness cross-check —
does projecting out the mean-difference direction also suppress what the probe's own decision
boundary was using? A useful sanity comparison, not a required step.

## 2. Projection / regularization formulation

Adapter `A(z) = Δz` (architecture — width, placement details — from
`Stage4_Intervention_Design_Choice.md` Decision 2: output-stage, applied to the frozen 1024-d
`forward_features()` output). Training loss:

```
L = L_task  +  λ_proj · L_orth(z + Δz, w)
```

where `w` is the fixed, unit-norm nuisance direction from §1 (computed once, before adapter
training, never updated during it), and `L_orth` penalizes the adapted embedding's projection onto
`w` — e.g. `L_orth(z') = (z' · w)²`, the same squared-cosine-style penalty already precedented in
`AR-01_Architecture_Study.md` (FairNVT's cosine-similarity loss; Fairness-Aware LoRA's Frobenius-norm
penalty is the multi-direction analogue, not needed here given a single direction). `λ_proj` is a
**fixed training hyperparameter** — one value, not itself the dose axis (see §3). Its exact numeric
value is an execution-level choice, not fixed here.

## 3. Dose / intensity ladder

**Not a separately-trained checkpoint per dose level.** One adapter is trained (§2, fixed `λ_proj`),
producing a single correction function `Δz = A(z)`. The dose ladder is generated by **post-hoc
interpolation** at inference time:

```
z'(α) = z + α · Δz,   α ∈ {0, 0.25, 0.5, 0.75, 1}
```

`α = 0` recovers the Reference-arm-equivalent embedding exactly (untouched `z`); `α = 1` is the
full-strength adapted embedding. Gate 1, geometry metrics, and the reliability-scorer suite are
computed **separately at each α**, from one training run.

**Why interpolation over separately-trained checkpoints per strength**: `22_Experimental_Design.md`
§3 asks for "2–3 strengths"; this gives five from a single adapter, satisfying that requirement with
room to spare, at a fraction of the training cost — directly responsive to §8's own resource
constraint (6-week/1-GPU budget) and consistent with how the Adaptation arm was already flagged as
"the first candidate to shrink" if GPU time binds. It also isolates the dose variable cleanly: every
α-level shares the *exact same* trained adapter, so any geometry difference across the ladder is
attributable to α alone, not to seed-to-seed training variance across separately-trained checkpoints
(a cleaner causal story than Paper 3's own λ_orth ladder, which *did* require separate training runs
per λ value — not a deviation from the mechanism, an improvement in isolating the dose variable
specifically).

**Not decided here**: `λ_proj`'s fixed value, and whether the five α points above are final or need
adjustment once Gate 1 results are seen. Execution-level.

## 4. Gate 1 measurement timing

Extends (does not edit — `Stage4_Execution_Plan.md` stays FINAL) the step sequence with one new
one-time shared step and clarifies per-α gating:

```
[Shared prerequisite, once — extends Stage4_Execution_Plan.md §0]
Load ReferenceTrainEmbedding (ISIC) + ReferenceEmbedding's PAD-UFES portion
        ↓
Fit nuisance direction w = μ_isic − μ_pad_ufes (unit-normalized) — closed-form, no training
        ↓
[Per arm × seed — extends §1 Step 2]
Train adapter A(z), loss = L_task + λ_proj · L_orth(z+Δz, w)   (w fixed, not updated during training)
        ↓
[Per α ∈ {0, 0.25, 0.5, 0.75, 1} — new inner loop within §1 Steps 3–8]
Compute z'(α) = z + α·Δz for the held-out eval population
        ↓
Extract embeddings at this α (mirrors ReferenceEmbedding's own extraction pattern)
        ↓
Compute domain-leakage probe + ID task performance on z'(α)          [feeds Gate 0]
        ↓
GATE 0 (this α, aggregated across seeds)
        ↓
GATE 1 (this α): did LID and/or spectral-decay slope actually move relative to α=0's own values?
    FAIL → this α not usable for C2–C5; frozen anyway, flagged not-usable, not evidence against the hypothesis
    PASS → continue
        ↓
[Only for α values that passed both gates]
Full measurement suite: geometry (incl. per-direction variance of w), reliability scorers
        ↓
Execution Contract → Execution → Review → Sanity → Freeze (per α, per Stage4_Execution_Plan.md's own chain)
```

The scorer suite runs **after**, never before, Gate 1 — an α value that fails Gate 1 is not scored,
matching the Reference Arm's own discipline of never computing a downstream measurement on something
that hasn't passed its applicable gate.

## Status

```
Stage 4 method, four previously-open items:
  Adapter-vs-projection question:    RESOLVED — adapter required, closed-form target only
  Nuisance-direction estimator:      LOCKED — difference of domain-conditional means
  Projection / regularization eq.:   LOCKED — L = L_task + λ_proj · L_orth(z+Δz, w)
  Dose ladder:                       LOCKED — single adapter, α-interpolation, α ∈ {0,.25,.5,.75,1}
  Gate 1 timing:                     LOCKED — per-α, before the scorer suite, after Gate 0

Still open (execution-level, not this document's scope):
  λ_proj's exact value
  Adapter width / exact placement details
  Optimizer, learning rate, training length
  Whether 5 α-points is final

Training: NOT AUTHORIZED
```
