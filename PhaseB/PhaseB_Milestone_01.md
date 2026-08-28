# Phase B — Milestone 01: Reference Arm Complete

**Milestone record. No numbers, no conclusions, no hypothesis.** What's done, what exists, what's
deferred, what's next — a timeline marker, not a results document. Scientific content lives in
`03d_stage3_5_baseline_synthesis.md` and the artifacts it indexes; nothing here should ever need to
be cited as a source of numbers.

Declared by Vinh, PI Decision, 2026-08-05: Reference Arm CLOSED, becomes historical record from this
point. No further edits to Reference Arm artifacts or documents beyond what this milestone records.

## Completed

- **Stage 0** — Retrospective Analysis
- **Stage 1** — Reference Instrument (setup + validation)
- **Stage 2** — Reference Representation (embedding extraction)
- **Stage 3** — Reference Arm Baseline Evaluation
  - 3.0 Artifact Audit
  - 3.1 Artifact Characterization
  - 3.2 Baseline Evaluation (Group A + Group B Mahalanobis-family, Gate 0)
  - 3.5 Baseline Synthesis

## Artifacts

- `ReferenceEmbedding` — held-out ID+OOD evaluation population
- `ReferenceTrainEmbedding` — ISIC-train reference-fitting population (EA-01)
- Baseline Package — Group A (`ReferenceArmBaseline/BaselineReport/`), Group B
  (`GroupBExecution/`), Synthesis (`03d_stage3_5_baseline_synthesis.md`)

## Frozen

- `ReferenceEmbedding/` — FROZEN
- `ReferenceTrainEmbedding/` — FROZEN
- `ArtifactCharacterization/` — FROZEN
- `DuplicateRegistry/` — FROZEN
- `ArtifactIssuesRegister/` — FROZEN
- `GroupBExecution/` — FROZEN (`execution_manifest.json` completion certificate)
- `ReferenceArmBaseline/BaselineReport/` — FROZEN (2026-08-05, explicit PI authorization)

All Reference Arm artifacts are now frozen. `03d_stage3_5_baseline_synthesis.md` and this document
remain documentation, not artifacts, and are not frozen.

## Execution Amendments

- **EA-01 — Closed.** `ReferenceTrainEmbedding/` produced, consumed by Mahalanobis/kNN/
  cosine-to-centroid/KDE, reviewed, sanity-checked, frozen.

## Deferred

- **Energy** — logit-based scorers work package; blocked on a trained classifier head
- **ViM** — same work package, same blocker
- **Relative Mahalanobis** — not implemented anywhere in the reused codebases
- **Per-direction variance** — no resolved definition

## Next milestone

**Stage 4 — Intervention Arm.** Not started. Not authorized. No file exists yet.
