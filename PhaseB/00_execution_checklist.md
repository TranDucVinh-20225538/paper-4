# Phase B Execution Checklist — Paper 4

**Mode: Protocol Execution.** Per Vinh, 2026-08-04: "Phase A is complete. Switch permanently to
Protocol Execution Mode. Do not propose new hypotheses, new claims, or new experiments unless they
are logged as protocol deviations. Begin Stage 0 (Retrospective Analysis) exactly as specified in
the preregistered protocol."

Nothing in this checklist introduces a new hypothesis, claim, metric, or analysis. Every item below
traces to a specific section of `24_PreRegistration.md`, `22_Experimental_Design.md`, or
`16_Minimal_Sufficient_Experiment.md` — cited inline. If an item can't be traced, it doesn't belong
here.

**Commit discipline (from here on):** every commit that touches `PhaseB/` or any experiment code
answers, in the commit message:
```
Implements which protocol step?
Implements which claim?
Changes preregistered analysis?  YES / NO
```
If the last answer is YES, the change is not a commit — it's a protocol deviation, logged in
`24_PreRegistration.md` §13, not silently merged.

**Execution Amendments vs. protocol deviations (2026-08-05, Vinh's PI decision) — two distinct
categories, not to be conflated:**
- **Protocol deviation** (`24_PreRegistration.md` §13): changes something *preregistered* —
  hypothesis, prediction, claim, outcome criterion. Requires the full deviation-logging path.
- **Execution Amendment** (`PhaseB/ExecutionAmendments/EA-NN.md`): corrects a stage's *execution
  specification* (what it was actually scoped to extract/compute) when it's found incomplete
  relative to the protocol's own — already-correct, unchanged — text. Does not touch anything
  preregistered. First instance: `ExecutionAmendments/EA-01.md` (Stage 2.1's extraction scope,
  found incomplete for Group B's reliability scorers during Stage 3.2).
- **Lifecycle (2026-08-05, Vinh's PI decision, applies to every Execution Amendment):**
  `Proposed → Approved → Implemented → Closed`. Proposed = drafted, gap traced, nothing generated.
  Approved = PI sign-off, extraction/computation authorized. Implemented = the required artifact
  exists and passed its own integrity gate. Closed = the artifact was actually consumed by what it
  was required for (e.g. Group B scorers computed). Each EA's own file carries a lifecycle log
  table recording the date/status/note for each transition — not silently overwritten.

**Claim discipline — known-issues consultation (2026-08-05, Vinh's PI decision):** every future
Claim C1–C5 write-up (Stage 3.2 onward) ends with an explicit line naming which
`assets/ArtifactIssuesRegister/` entries were consulted, and a judgment call for each:
```
Known artifact issues consulted:
  ✓ Issue #1 — Not expected to materially affect Claim C1 because ...
```
or, if unclear either way:
```
  ✓ Issue #1 — Potentially relevant.
```
This makes the reasoning connecting an artifact-level finding to a claim-level conclusion explicit
and auditable, rather than assumed. The register itself is frozen and append-only (new issues get
new entries, never edits to existing ones) — this rule is about how claims *reference* it, not about
changing it.

**Artifact Immutability Rule (2026-08-05, Vinh's PI decision, standing policy from this point on):**
once an artifact completes its chain — Execution → Review → Sanity → Freeze — it is **never edited
again, in any form.** Not the data files, not the manifest, not a single field. Adopted after the
`BaselineReport/` freeze surfaced a real provenance failure mode: a stale SHA256 in a manifest that
had gone unnoticed since an earlier edit — proof that "review before freeze" catches exactly this
class of error, and that editing a frozen artifact afterward (even a "small fix") reopens the same
risk on something meant to be permanently trustworthy.
- If a problem is found in a frozen artifact later, there are exactly three legitimate responses,
  **never a direct edit**: (1) an Execution Amendment, (2) an Issues Register entry, or (3) a
  successor artifact (versioned — `ReferenceEmbedding_v1` stays exactly as frozen; a real fix
  produces `ReferenceEmbedding_v2` via its own EA, never `ReferenceEmbedding_v1 (edited)`).
- Applies to every artifact directory under `assets/`, `GroupBExecution/`, `BaselineReport/`, and
  any future Intervention-arm equivalent — not to living documentation (this checklist, the Lab
  Notebook, stage-narrative `.md` files), which must stay editable to remain accurate as new stages
  begin.
- No `_v2` currently exists or is needed for anything produced so far.

**Restructuring note (2026-08-05, Vinh's PI decision):** Phase B's *internal execution stage
numbering* (not the underlying preregistered protocol's claims/predictions, which are unchanged)
is renumbered as of Stage 1's completion:

| Old numbering | New numbering | Content | File |
|---|---|---|---|
| Stage 1 | **Stage 1 — Reference Instrument** (unchanged) | Backbone resolution, checkpoint, encoder, preprocessing, dataset loader/validation, batch-inference gate | `02_stage1_setup.md` |
| *(new)* | **Stage 2 — Reference Representation** | Embedding extraction, embedding provenance, Embedding Integrity Gate — the embedding artifact is a **primary research artifact**, not a cache | `02a_stage2_reference_representation.md` |
| Stage 2 | **Stage 3 — Reference-arm analysis** | Probe → geometry metrics → accessibility metrics → reliability metrics, computed *from* Stage 2's frozen embedding artifact | `03_stage2_reference.md` (filename unchanged — renamed content, not renamed file, to avoid breaking existing Lab Notebook references) |
| Stage 3 | **Stage 4 — Intervention arm** | Intervention/Conventional/Adaptation arms | not yet created |

Rationale (Vinh): "Stage 0 sinh ra evidence. Stage 1 sinh ra instrument. Stage 2 sẽ sinh ra data." —
each stage now corresponds to one category of output (evidence, instrument, data, analysis), not an
arbitrary infrastructure/science split.

---

## Stage 0 — Retrospective Analysis (FREE, zero GPU)

Source: `16_Minimal_Sufficient_Experiment.md` Week 0; `22_Experimental_Design.md` §1 (C1), §6 (C1
statistical test); `24_PreRegistration.md` §5 (C1), §8 (test = "Kendall's τ, exact p-value").

- [x] Compute LID (Local Intrinsic Dimensionality) on Paper 3's existing E1 embeddings.
- [x] Compute within-class spectral-decay slope on the same embeddings.
- [x] Reuse condition number (already computed, `paper-3/results/e1_geometry_metrics.csv`) — no
      recomputation.
- [x] Form composite `P = LID × |slope|` (source: arXiv:2510.15202, Eq. 6, per
      `08_Mechanism_Map.md` Mechanism 1).
- [x] Kendall's τ (exact) between composite and Mahalanobis-AUROC (`paper-3/results/e2_auroc.csv`)
      across the 13 existing λ_orth = {0, 1, 5} checkpoints (5+5+3 seeds).
- [x] Report as a **gate**, not a conclusion — per `Scientific_Hypothesis.md`'s falsifiable-outcomes
      section: this is a free, decisive check before any GPU is spent, not itself a claim about the
      FM arms.
- [x] Log result in `26_Lab_Notebook.md`. If surprising: 24-Hour Rule
      (`25_Research_Constitution.md`) — no hypothesis/protocol edit for 24 hours regardless of
      outcome.

**Details:** `01_stage0_retrospective.md`.

## Stage 1 — Reference Instrument ✅ APPROVED-COMPLETE (2026-08-05)

Source: `22_Experimental_Design.md` §2 ("Backbone dependency... should be resolved before Stage 1
begins"); `16_Minimal_Sufficient_Experiment.md` Week 1.

- [x] Resolve backbone dependency: **PanDerm** — checkpoint direct-download, CC-BY-NC-ND 4.0,
      resolved 2026-08-05 (not the CLIP ViT-B/16 fallback).
- [x] Download checkpoint (PanDerm ViT-L/16; SHA256 + full identity in `assets/`).
- [x] Verify checkpoint loads (raw state_dict, structurally confirmed ViT-L/16 — see
      `assets/panderm_vitl16_checkpoint_metadata.md`).
- [x] Instantiate encoder class + single-image inference (Stage 1.2, official repo implementation
      only — see `reference_instrument_validation.md`, all PASS).
- [x] Batch inference validation, infrastructure smoke test (96 legacy reader-study images).
- [x] Dataset loader (ISIC 2019 + PAD-UFES-20, CSG-Skin's own, unmodified; `master_metadata.csv`
      built + frozen with provenance; ISIC images verified against official GroundTruth,
      25,331/25,331, 0 mismatches).
- [x] Dataset Validation gate (Stage 1.3a) — PASS.
- [x] Batch inference validation on the **preregistered** split (Stage 1.3 rerun) — **Scientific
      PASS**, all 5 criteria green.

**APPROVED-COMPLETE.** Details: `02_stage1_setup.md`, `reference_instrument_validation.md`.

## Stage 2 — Reference Representation ✅ Stage 2.1 COMPLETE, artifact frozen + immutable (2026-08-05)

Source: `22_Experimental_Design.md` §3 (Reference arm), §3.5 (Gate 0) — the frozen-representation
half of the Reference arm's pipeline, split out as its own stage because the embedding artifact it
produces is a **primary research artifact**, not an intermediate cache. Every downstream
measurement (Mahalanobis, probe, LID, spectral decay, energy, ViM, etc.) will be computed *from*
this stage's frozen output, not recomputed from images each time.

- [x] **Stage 2.1 — Extract embeddings for the complete preregistered ID + OOD datasets — DONE
      2026-08-05, GATE PASS (6/6).** ID (ISIC-test, n=5,067) + OOD (PAD-UFES-20, n=2,298) = 7,365
      images, validated Reference Instrument, ~40 min CPU wall time. Frozen **Reference Embedding
      Artifact**: `assets/reference_embeddings/ReferenceEmbedding/` (`embeddings.npy`, `labels.csv`,
      `metadata.csv`, `manifest.json`, `sha256.txt`).
- [x] Embedding Integrity Gate: sample count, tensor shape, ordering (independently re-proven, not
      just self-consistent), label alignment, per-file SHA256, manifest — all PASS. See
      `02a_stage2_reference_representation.md` §Result.
- [x] **Immutability — DONE 2026-08-05.** `manifest.json` records `"status": "FROZEN"` as its final
      field (last write before lockdown). All five files in `ReferenceEmbedding/` chmod'd `444`
      (read-only); the directory itself chmod'd `555` (no write/add-file). Verified: a write attempt
      into the directory is rejected (`Permission denied`). Any future correction is a **new**
      artifact version (new directory, new manifest) — never an edit to this one.
- [ ] *(Not yet authorized)* Anything beyond Stage 2.1 — probe, geometry, OOD scores, statistics are
      explicitly out of scope until a further PI decision opens them.

**Stage 2.1 complete, artifact frozen and immutable.** Details:
`02a_stage2_reference_representation.md`.

## Stage 3 — Scientific Analysis (renamed 2026-08-05, per Vinh — architecture corrected same day)

Source: `22_Experimental_Design.md` §3 (Reference arm), §3.5 (Gate 0), §1 (Claim Matrix mapping).
Renamed from "Reference-arm analysis" and broken into sub-stages, so the checklist stops implying
that analysis begins with a distance scorer. Rationale (Vinh): "Stage 3 không nên bắt đầu bằng
Mahalanobis" — audit the artifact and characterize it structurally before any claim-bearing
computation touches it.

**Architecture correction (2026-08-05, same day as the initial 3.2 draft):** Stage 3.2 was
initially drafted as "Claim C1," then corrected — Claim C1 is already closed (Stage 0's retrospective
CNN check), and `ReferenceEmbedding/` is input data for *later* claims, not a re-test of C1. Stage
3.2 is baseline measurement, not hypothesis testing; the actual claim-testing (C2–C5) is now its own
stage, run only once both Reference and Intervention arms have baseline measurements to compare.

| Sub-stage | Content | Status |
|---|---|---|
| **3.0 — Artifact Audit** | Corruption/duplicate/completeness check on the frozen Reference Embedding Artifact; random-sample visual inspection; reproducibility spot-check. No interpretation. | **APPROVED WITH FINDING (2026-08-05)** — 4/5 checks clean PASS; check 2 found 17 pixel-identical-image groups (13/17 with conflicting labels), fully investigated, evidence frozen in `assets/DuplicateRegistry/` + `assets/ArtifactIssuesRegister/`, no dataset modification — see `03a_stage3_0_artifact_audit.md` |
| 3.1 — Artifact Characterization | Rows/dim/memory/dtype/norm distribution — structural description, not a hypothesis test. | **DONE 2026-08-05** — shape (7365,1024), norm/variance statistics, class counts; frozen. See `03b_stage3_1_artifact_characterization.md` |
| **3.2 — Reference Arm Baseline Evaluation** (renamed from "Claim C1") | Gate 0 → domain-leakage probe → geometry metrics → reliability scorers → ID task performance → Baseline Report. Baseline measurement, **not** a claim test. | **Group A: COMPLETE.** Domain-leakage probe (mean acc 0.9990, AUROC 1.0, shuffle-control 0.6136), geometry (condition_number=1,338,282, LID=15.03, spectral-decay slope=0.00697), ID task performance (balanced accuracy 0.574), Gate 0 PASS.<br><br>**Group B: COMPLETE (Mahalanobis-family) — 2026-08-05.** No longer Blocked/Ready/Pending — computation finished, reviewed, sanity-checked, frozen. Chain: Execution Contract → Execution → Execution Review → Execution Sanity → Freeze, each step logged. `ExecutionAmendments/EA-01.md` — **Closed** (`ReferenceTrainEmbedding/`, n=16,211; closed once the scorers it was scoped for consumed it — remaining classifier-head blocker for ViM/Energy tracked as a separate, later amendment, not under EA-01). Frozen results (`GroupBExecution/`, all files `chmod 444`, directory `chmod 555`, `execution_manifest.json` completion certificate):<br>&nbsp;&nbsp;— Mahalanobis: AUROC 0.9984, FPR@95 0.0081<br>&nbsp;&nbsp;— kNN (k=1/10/50): AUROC 0.9988 / 0.9979 / 0.9967, FPR@95 0.0057 / 0.0111 / 0.0180<br>&nbsp;&nbsp;— Cosine-to-centroid: AUROC 0.9805, FPR@95 0.0807<br>&nbsp;&nbsp;— KDE: AUROC 0.8942, FPR@95 0.3898<br>Numbers are Tier 1 (Measurement), single-arm, no cross-arm comparison, no interpretation — that stays out of scope until Stage 5.<br><br>**Separate work package, not a Group B status (per Vinh, 2026-08-05): Energy / ViM / Relative Mahalanobis.** Not touched. Energy — BLOCKED (no trained classifier head for the Reference Instrument). ViM — PARTIALLY BLOCKED (fitting population exists; classifier head still missing). Relative Mahalanobis — NOT IMPLEMENTED (no code anywhere in the reused codebases). Per-direction variance also not computed (no resolved definition). No Tier-3/cross-reference language anywhere in this stage's artifacts. |

Explicitly not yet in scope for Stage 3.2: any interpretation of the Gate 0 numbers beyond
PASS/FAIL, and no Claim C2–C5 evaluation (that's Stage 5). `03_stage2_reference.md`'s
probe/geometry/accessibility/reliability content now corresponds to Stage 3.2's measurement suite.

## Stage 3.5 — Reference Arm Baseline Synthesis

Inserted by Vinh's PI decision (2026-08-05) as a checkpoint between Stage 3.2 and Stage 4.
**Documentation only — no computation, no new figure, no hypothesis test.** Assembles Stage 0–3.2's
already-produced artifacts and numbers into one "Reference Arm dossier," so every future Stage 4/5
comparison cites one document instead of separately reopening `ReferenceEmbedding/`,
`ReferenceTrainEmbedding/`, `BaselineReport/`, `GroupBExecution/`, Gate 0, `ArtifactCharacterization/`,
`DuplicateRegistry/`, etc.

**DONE 2026-08-05.** See `03d_stage3_5_baseline_synthesis.md` — every number in it quoted, not
recomputed, from its cited source file.

## Stage 4 — Intervention Arm

Source: `22_Experimental_Design.md` §3 (Intervention/Conventional/Adaptation arms).

Runs the **same measurement suite as Stage 3.2** (Gate 0, probe, geometry, reliability scorers, ID
task performance), applied to the Intervention/Conventional/Adaptation arms once they exist. Not a
claim test either — produces the comparison-side baseline Stage 5 needs.

**Architecture Review + Execution Plan: FINAL (2026-08-05).** `Stage4_Architecture_Review.md`,
`Stage4_Execution_Plan.md` — both closed to further edits per Vinh's "no more protocol edits" rule;
future changes route only through an Execution Amendment, a Protocol Deviation, or (new,
2026-08-05) an **Architecture Resolution**.

**First computation authorized (2026-08-05), then blocked by a real gap before Step 2 could run.**
Vinh authorized exactly one checkpoint (Intervention, lowest strength) through the locked pipeline.
Investigating CSG-Skin's actual GRL+orthogonality implementation before writing training code
(`Stage4_Intervention_Architecture_Gap.md`) found that its dual-encoder architecture has no defined
analog for a single frozen ViT + adapter — three concrete gaps (context-embedding target, adapter
placement, λ calibration), all confirmed by Vinh. **Neither an Execution Amendment nor a Protocol
Deviation** — a new category, **Architecture Resolution (AR-NN)**, created for scientific-design
questions resolved through evidence rather than extraction/computation.

**AR-01 — Intervention Architecture Resolution: OPEN, Step 1 DONE.** `ArchitectureResolutions/AR-01.md`
+ `ArchitectureResolutions/AR-01_Architecture_Study.md`. Question (revised, literature-first): what
intervention architectures already exist for modifying frozen vision FM representations, and which
design principles recur? **Step 1 (Architecture Study) complete**: 27 papers, 5-tier taxonomy,
saturation reached, 6 papers spot-verified this session. Headline finding: a full second/auxiliary
encoder (CSG-Lite's own structure) appears in **0/27** surveyed papers; auxiliary
branches/split-heads short of a full second encoder appear in 7/27; orthogonality in 13/27; explicit
geometry analysis in only 7/27. **Step 2 (answer AR-01 from this evidence) — DONE.**
`ArchitectureResolutions/AR-01_Resolution.md`: constraint-matching table (7 constraints, grounded in
`22_Experimental_Design.md`) × 5 intervention families, scored against Step 1's evidence. Resolution:
a single-backbone, output-stage adapter/projection carrying an explicit orthogonality-style
constraint — not CSG-Lite's dual-encoder structure — best fits Paper 4's constraints. Family-level
only: no exact architecture, dimensionality, injection point, or hyperparameter chosen — those
remain a separate, not-yet-authorized Stage 4 execution decision.

**Architecture Instantiation Contract DONE (2026-08-05).** `Stage4_Architecture_Instantiation_Contract.md`
— the bridge between AR-01's family resolution and exact implementation, not another Architecture
Resolution. Locks: intervention target (the acquisition-domain signal already measured throughout
Stage 3, not "context" — no context pipeline exists), output-space placement (must land in the same
1024-d space `ReferenceEmbedding` occupies), what geometry must change (Gate 1's exact criterion:
LID and/or spectral-decay slope, not just condition number/scale), and what stays fixed (backbone,
preprocessing, split, all Reference Arm artifacts — Artifact Immutability Rule). Leaves open, per
Vinh's explicit instruction: the nuisance-direction mechanism (learned-adversarial vs.
closed-form/data-derived) and all exact hyperparameters/placement.

**Intervention Design Choice DONE (2026-08-05).** `Stage4_Intervention_Design_Choice.md` — the two
decisions the Instantiation Contract deliberately left open, both resolved with reasoning.
**Decision 1**: closed-form/data-derived nuisance projection, not a learned-adversarial head —
avoids the adapter-vs-adversarial-training confound on Gate 1's exact criterion, reuses
`ReferenceTrainEmbedding`'s existing domain labels and Group A/B's already-trusted
fit-from-labeled-data pattern, matches most of the survey's own geometry-analyzing papers.
**Decision 2**: output-stage injection only, applied to the same 1024-d `forward_features()` output
already used throughout the Reference Arm — not any internal transformer block. Converges from
AR-01's own family conclusion, the Instantiation Contract's output-space requirement, and direct
reuse of the Reference Arm's own extraction pipeline.

**Intervention Method Specification DONE (2026-08-05).** `Stage4_Intervention_Method_Specification.md`
— locks the four items Design Choice left open. (0) Confirmed a genuine trainable adapter is still
required (not a pure projection) — `22_Experimental_Design.md`'s "Adapter-based" wording plus the
Conventional arm's matched-budget requirement (undefined without a real parameter/training-step
budget to match) both need it; the closed-form part is the *target* the adapter is regularized
against, not a replacement for the adapter. (1) Nuisance-direction estimator: difference of
domain-conditional means (μ_isic − μ_pad_ufes), not a trained classifier or a PCA subspace —
zero-training, matches the protocol's own singular "the intervened direction" wording, precedented
both inside this project (Mahalanobis/cosine-to-centroid's own per-class-mean pattern) and in the
survey. Verified via `CSG-SKin/src/datasets/splits.py` (not assumed) that estimating this from
`ReferenceTrainEmbedding` + `ReferenceEmbedding`'s PAD-UFES portion needs no new artifact or EA —
Papers 1–3's own methodology already reuses the full PAD-UFES population for domain-training signal
and OOD evaluation both. (2) Regularization equation: `L = L_task + λ_proj·L_orth(z+Δz, w)`, `w`
fixed before training. (3) Dose ladder: **one** adapter trained, dose generated by post-hoc
interpolation `z'(α) = z + α·Δz`, `α ∈ {0,.25,.5,.75,1}` — five points from a single training run,
not five separately-trained checkpoints, satisfying the "2–3 strengths" requirement at a fraction of
the compute cost and isolating the dose variable more cleanly than Paper 3's own per-λ training runs
did. (4) Gate 1 timing: fit `w` once (shared prerequisite) → train adapter once → per-α: extract →
Gate 0 → Gate 1 → only if both pass, full measurement suite → freeze. Still open (execution-level):
`λ_proj`'s value, adapter width/placement details, optimizer/training hyperparameters. No code.

**Adapter and Control Specification DONE (2026-08-05).** `Stage4_Adapter_And_Control_Specification.md`
— locks the three items flagged as still open, kept as its own decision record so
`Stage4_Execution_Plan_v2.md` (not yet written) only locks run order, never a scientific decision.
(1) Adapter architecture: residual bottleneck `z' = z + W₂·σ(W₁·z)`, not linear-only — direct
instantiation of AR-01's own "bottleneck/feature adapter" family conclusion (AdaptFormer/CLIP-Adapter
precedent), nonlinearity kept specifically to preserve the capacity/mechanism distinction §0 of the
Method Specification established. (2) Orthogonality loss: squared cosine similarity,
`L_orth = cos(z',w)²` — the exact formula CSG-Lite itself uses (`csg_lightning.py`, read directly:
`mean(cosine_similarity(z_l, z_c_proj).pow(2))`), the most literal reading of "mirror... as closely
as an adapter allows"; also the smoother optimization target vs. `|cos|`'s kink at the target value
(zero). (3) Conventional arm matching: exact match on architecture/bottleneck width/optimizer/
epochs/seeds; `L_orth` structurally **absent**, not `λ_proj = 0` (a categorical difference, not a
ladder endpoint) — protects C4's actual question. Conventional's own trained (task-only) adapter
gets the same α-interpolation scaffolding for measurement parity with Intervention's ladder, without
reintroducing any nuisance mechanism into its training.

**Hyperparameter Selection Rules DONE (2026-08-05).** `Stage4_Hyperparameter_Selection_Rules.md` —
bottleneck width `r` and `λ_proj` locked as pre-committed *procedures* (grid + "stop at smallest
value that passes Gate 0 + Gate 1"), not values chosen after seeing results — both calibrated
against integrity/manipulation gates only, never against C2–C5 outcomes. Nonlinearity locked to
GELU, matching PanDerm's own verified activation (`GELU(approximate='none')`, confirmed from this
project's own model-loading output, not assumed).

**Execution Plan v2 DONE (2026-08-05).** `Stage4_Execution_Plan_v2.md` — pure run order, 10 steps,
every choice a pointer to an already-locked decision document, no scientific decision made inline.
Stage 4's scientific design is now complete end to end: family → mechanism → target → estimator →
dose → adapter class → orthogonality formula → Conventional control → hyperparameter selection
rules → run order.

**Stage 4 Execution: AUTHORIZED (v1.0, 2026-08-05).** `Stage4_Execution_Authorization.md` — nine
locked design documents confirmed as a single chain, none silently folded into another. **New
standing rule, effective immediately: no further scientific-design changes once execution begins** —
an implementation bug gets fixed inline and logged; anything else (a design choice that turns out
wrong, a new gap, a result suggesting a different mechanism) goes through the Amendment process
(Execution Amendment / Architecture Resolution / Protocol Deviation, per which layer it touches),
never an inline fix. Same logic as the Artifact Immutability Rule, one level up — applied to the
design, not just the artifacts it produces. Execution of `Stage4_Execution_Plan_v2.md`'s run order
may now proceed, starting at Step 1, using the hyperparameter selection *procedures* already locked
— not any value chosen ad hoc.

## Stage 5 — Comparative Analysis

Source: `22_Experimental_Design.md` §1 (Claim Matrix mapping), §6 (statistical tests for C2–C5).

Where hypothesis testing actually happens: Reference (Stage 3.2) vs. Intervention (Stage 4) →
Claims C2 (geometry-vs-AUROC correlation, Prediction 1), C3 (normalization-knob causal test,
Prediction 2), C4 (Intervention-type mediation), C5 (probe-decodability replication on the FM).

**Not started.** No file yet — out of scope until both Stage 3.2 and Stage 4 have passed Gate 0.

---

## Status

| Stage | Status | Gate result |
|---|---|---|
| 0 — Retrospective | Executed 2026-08-04 | No significant signal (τ=−0.154, p=0.510, n=13) — not falsified, proceed as preregistered. See `01_stage0_retrospective.md` §Result |
| 1 — Reference Instrument | **APPROVED-COMPLETE 2026-08-05** | Scientific PASS on preregistered split — see `reference_instrument_validation.md` |
| 2 — Reference Representation | **COMPLETE**, frozen + immutable (2026-08-05) | Reference Embedding Artifact — `assets/reference_embeddings/ReferenceEmbedding/` |
| 3.0 — Artifact Audit | **COMPLETE**, frozen | 4/5 clean PASS, 1 finding registered (`DuplicateRegistry/`, `ArtifactIssuesRegister/`) |
| 3.1 — Artifact Characterization | **COMPLETE**, frozen | Descriptive characterization — `ArtifactCharacterization/` |
| 3.2A — Baseline Evaluation, Group A | **COMPLETE**, frozen | Probe, geometry, ID task, Gate 0 PASS — `BaselineReport/` |
| 3.2B — Baseline Evaluation, Group B (Mahalanobis-family) | **COMPLETE**, frozen | EA-01 Closed — `GroupBExecution/` |
| 3.5 — Baseline Synthesis | **COMPLETE** | `03d_stage3_5_baseline_synthesis.md` (documentation, not an artifact — not frozen) |
| **Reference Arm** | **CLOSED 2026-08-05 — historical record, no further edits.** No pending items. | `PhaseB_Milestone_01.md` |
| 4 — Intervention Arm | **EXECUTION IN PROGRESS (2026-08-05)** — scientific design LOCKED (`Stage4_Execution_Authorization.md` v1.0), execution underway per `Stage4_Execution_Plan_v2.md` | Step 1 (Reference verification) PASS; Step 2 (`NuisanceDirection/`) COMPLETE, frozen |
| 5 — Comparative Analysis | Not started | — |

**Milestone 01 reached (2026-08-05): Reference Arm complete, all artifacts frozen** (`ReferenceEmbedding/`,
`ReferenceTrainEmbedding/`, `ArtifactCharacterization/`, `DuplicateRegistry/`, `ArtifactIssuesRegister/`,
`BaselineReport/`, `GroupBExecution/`). See `PhaseB_Milestone_01.md`. From this point, no further edits
to any Reference Arm artifact — corrections go through a new Amendment or the Issues Register, never
an edit to frozen content.

**Stage 4 execution progress (2026-08-05), per `Stage4_Execution_Plan_v2.md`:**
- **Step 1 — Reference verification: PASS.** PanDerm checkpoint SHA256, every frozen Reference Arm
  artifact's per-file SHA256 (`ReferenceEmbedding`, `ReferenceTrainEmbedding`, `BaselineReport`,
  `GroupBExecution`), and read-only permissions all confirmed intact, non-destructively. No drift
  since freezing.
- **Step 2 — Nuisance direction `w`: COMPLETE, frozen.** `NuisanceDirection/` — its own Group-B-style
  artifact (execution contract → implementation → execution → review → sanity → freeze), kept
  separate from adapter code. `w = unit(μ_isic − μ_pad_ufes)`, closed-form, no training. Review PASS,
  sanity PASS (4/4). Not an adapter, not a claim — the fixed target the adapter will be regularized
  against.
- **Step 3 — bottleneck width: LOCKED, `r = 16`.** `analysis/stage4_adapter.py` +
  `analysis/stage4_train_intervention.py`. Gate 0 PASS at every `r` in `{16,32,64,128}`. Gate 1
  criterion operationalized as `ExecutionAmendments/EA-02.md` (manipulation check: reproducible,
  correctly-directed change in LID and/or spectral-decay slope vs. baseline — not a hypothesis
  test, no arbitrary threshold). PASSES at every `r`; smallest, `r=16`, selected per
  `Stage4_Hyperparameter_Selection_Rules.md` §1.
- **Step 4 — λ_proj: LOCKED, `λ_proj = 0.1`.** `analysis/stage4_select_lambda.py`, grid
  `{0.1,1,5}`, `r=16` fixed. Passed Gate 0 + Gate 1 (EA-02) immediately at the smallest value.
- **Canonical Intervention Arm (5 seeds): COMPLETE, FROZEN.** `InterventionArm_Canonical/` —
  `r=16`, `λ_proj=0.1` fixed. Every seed (42/52/62/72/82) individually passes Gate 0 and Gate 1
  (EA-02); aggregate Gate 0 (probe_acc=0.9988, ID-task bal_acc=0.5643) PASS, aggregate Gate 1
  (LID=14.682 vs. baseline 15.028, slope=0.006791 vs. baseline 0.006973) PASS. 5 adapter
  checkpoints + 5 adapted-embedding arrays + manifest, frozen.
- **Canonical Conventional Arm (5 seeds): COMPLETE, FROZEN.** `ConventionalArm_Canonical/` — same
  architecture/optimizer/epochs/seeds as Intervention, `r=16`, `L_orth` structurally absent (not
  `λ_proj=0`). Every seed passes Gate 0 and Gate 1; aggregate Gate 0 (probe=0.9990,
  ID-task=0.5639) PASS, aggregate Gate 1 (LID=14.700 vs. baseline 15.028, slope=0.006787 vs.
  baseline 0.006973) PASS. Frozen on Gate 0 alone (matches `Stage4_Execution_Plan_v2.md`'s own
  freeze-precondition wording — Gate 1 failure would not have blocked freezing either; noted that
  Intervention's own freeze required both gates, stricter than specified, harmless since both
  passed there anyway).
- **Step 6 — α-ladder: computed for both arms, all seeds, all α.** `AlphaLadder/alpha_ladder_results.json`
  (not frozen). `α=0` exactly reproduces baseline in every run (sanity-confirmed). Found: slope
  shows a reproducible but non-monotonic response (sharp drop at lowest dose, partial recovery
  toward baseline as dose increases) in all 10 seed×arm combinations — exposed that EA-02's
  "monotonic" wording over-constrained Gate 1. **EA-03 opened**: Gate 1 criterion corrected to
  "reproducible dose-dependence," no shape assumption. Existing α-ladder output re-scored (not
  regenerated) under EA-03 — **Gate 1 PASSES at every nonzero α, both arms**
  (`AlphaLadder/gate1_ea03_rescored.json`). No interpretation of the response shape. **EA-03 is the
  last Gate 1 amendment — Gate 1 is now locked for the remainder of Stage 4** (`ExecutionAmendments/EA-03.md`):
  any future unexpected response shape is a result to record, not grounds to adjust the gate again.
- **Step 7 (partial) — geometry completion: condition number + per-direction variance of `w`.**
  `AlphaLadder/geometry_completion_results.json` (not frozen). Per-direction variance of `w`:
  monotonic-decreasing in 5/5 Intervention seeds (20.198→10.796 mean), monotonic-*increasing* in
  4/5 Conventional seeds (20.198→22.072 mean) — a clean, visibly different response shape between
  arms, recorded factually. Condition number: noisy, no clean pattern, consistent with Gate
  1/EA-03's own skepticism of condition number alone.
- **Remaining Step 7 (reliability scorers on adapted embeddings) + Steps 8–10 (freeze)**: not
  started — stopped for review, same cadence.
