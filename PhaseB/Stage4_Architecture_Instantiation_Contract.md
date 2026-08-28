# Stage 4 — Architecture Instantiation Contract

**Bridge document between AR-01's family-level resolution and Stage 4's exact implementation. Not
another Architecture Resolution** (`ArchitectureResolutions/AR-01.md` is closed to further
additions on this point) **— locks what is already determinable from existing, locked protocol and
frozen artifacts, and explicitly marks what remains open for Stage 4 execution.** No code. No exact
architecture spec. No dimensionality, injection point, or hyperparameter chosen. No training
authorized by this document.

## 1. Intervention target

**Locked — already established, not a new decision.** The intervention target is the
acquisition-domain signal: exactly what the domain-leakage probe has measured throughout Stage 3.2
Group A (Gate 0 criterion 1) and what the `domain` column already records in both frozen embedding
artifacts (`isic` vs. `pad_ufes`, `ReferenceEmbedding/metadata.csv`,
`ReferenceTrainEmbedding/metadata.csv`). Not "context" — no context-crop pipeline exists anywhere in
this project (Gap 1, `Stage4_Intervention_Architecture_Gap.md`). Not a generic "shortcut proxy" —
too vague to gate against. The target is the specific, already-measured, already-frozen acquisition
label this project has used since Stage 0.

## 2. Where intervention happens

**Partially locked.** The output representation must land in the same fixed-dimensional space
`ReferenceEmbedding`/`ReferenceTrainEmbedding` already occupy — 1024-d, whatever
PanDerm's `forward_features()` already produces, or a space that maps onto it 1:1 — so that Gate 0,
Gate 1, the geometry metrics, and the reliability scorers remain directly comparable to the frozen
Reference Arm baseline (this is `AR-01_Resolution.md`'s "geometry is the mediator, not a byproduct"
constraint, restated at the placement level: the adapter must act on the space actually being
measured, not some internal signal invisible to it).

**Not locked**: exactly where within the transformer the adapter's trainable parameters are
inserted — a single late-stage module vs. one attached to every block. `AR-01_Resolution.md`'s
"single-backbone, output-stage" family constrains the *shape* (favors an identifiable,
low-dimensional output-adjacent representation over parameters diffused across every block, per its
own interpretability reasoning) but does not fix the exact injection point. Stage 4 execution
decision.

## 3. What geometry should change

**Fully locked — already specified in the preregistered protocol, not open at all.**
`22_Experimental_Design.md` §3.5, Gate 1, quoted exactly: *"Did LID and/or spectral-decay slope
actually change (not just condition number/scale)?"* — this is the primary criterion, and it
explicitly excludes condition-number/scale movement alone as sufficient evidence of a real
manipulation. Additionally measured per §4 measurement 3: condition number (kept for continuity
with Paper 3) and per-direction variance of the intervened direction (Mechanism 2, variance
suppression). This last quantity is where §1 (intervention target) and §2 (where intervention
happens) meet — per-direction variance can only be computed once both a target direction and an
output space are defined, which is exactly what §§1–2 above establish.

## 4. What remains fixed

**Fully locked — already the case; this section restates, it does not decide.** Per the Artifact
Immutability Rule (`00_execution_checklist.md`, adopted at Reference Arm close):

- **Backbone**: frozen PanDerm ViT-L/16 checkpoint, unmodified (`assets/panderm_vitl16_checkpoint_metadata.json`).
- **Preprocessing**: frozen (`assets/reference_preprocessing.json`).
- **Dataset/split**: frozen (`random_state=42`, `isic_test_fraction=0.2`, `val_fraction=0.2`, validated Stage 1.3a, reused unchanged through both embedding artifacts).
- **Evaluation artifacts**: `ReferenceEmbedding/`, `ReferenceTrainEmbedding/`, and every other Reference Arm artifact — frozen, no further edits under any circumstance. Stage 4 *adds* new artifacts; it never touches an existing one.

## 5. Why this is not a new hypothesis

The hypothesis, unchanged since `Scientific_Hypothesis.md`: representation intervention → geometry
changes → accessibility changes → reliability changes. `22_Experimental_Design.md` §3's own framing
note, quoted exactly: *"these four arms are instruments for producing controlled variation in
geometry — they are not, individually or collectively, the object of the paper."* Choosing a family
(AR-01 Step 2) and now narrowing what it targets, where it acts, and what geometry it must move
(§§1–3 above) instantiates this one, already-locked claim — it does not propose a new one. The test
this project runs is not "does adapter X beat adapter Y"; that would be a different, unauthorized
experiment. The test is whether *an* intervention that moves LID/spectral-decay slope changes what a
distance-based reliability estimator can access — same chain, same claim, regardless of which
evidenced family instantiates it.

## What remains explicitly open

- **Nuisance-direction mechanism** — learned-adversarial (a head reading the adapted embedding
  through a GRL, à la FairNVT/DiCoOp) vs. closed-form/data-derived (a projection fit from labeled
  domain data, à la Identity Sanitization Projection) — deliberately left open, per Vinh's explicit
  instruction: both are evidenced and valid per `AR-01_Resolution.md` §3, and this contract does not
  choose between them.
- **Exact adapter placement and dimensionality, λ/scale parameterization, optimizer and training
  hyperparameters** — Stage 4 execution decisions, not resolved here.

## Status

```
AR-01 Step 1:        COMPLETE
AR-01 Step 2:         COMPLETE
Reference Arm:        CLOSED

Stage 4:
  Architecture family: resolved (AR-01)
  Target / geometry / fixed-scope: locked (this document)
  Exact instantiation: NOT YET DEFINED
  Training: NOT AUTHORIZED
```
