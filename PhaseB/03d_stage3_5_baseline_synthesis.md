# Stage 3.5 — Reference Arm Baseline Synthesis

**Documentation only. No computation, no new figure, no hypothesis test.** Inserted by Vinh's PI
decision (2026-08-05) as a checkpoint between Stage 3.2 (Reference Arm Baseline Evaluation) and
Stage 4 (Intervention Arm): a single "Reference Arm dossier" assembling what Stage 0–3.2 already
produced, so every future Stage 4/5 comparison points back to one document instead of separately
reopening `ReferenceEmbedding/`, `ReferenceTrainEmbedding/`, `BaselineReport/`, `GroupBExecution/`,
`Gate 0`, `ArtifactCharacterization/`, `DuplicateRegistry/`, and the rest. Every number below is
**quoted**, not recomputed, from the cited source file.

## 1. Scope

Covers the Reference arm only: Stage 0 (retrospective) through Stage 3.2 (Group A + Group B
Mahalanobis-family). Does **not** cover Stage 4 (Intervention arm — not started) or Stage 5
(Comparative analysis — not started, requires both arms). This document is itself Tier 0/1
(Validation/Measurement indexing) — no Tier 3 (cross-arm comparison / claim) content, because there
is nothing yet to compare against.

## 2. Reference Instrument

PanDerm ViT-L/16 (arXiv:2410.15038), via `models.get_encoder(args, 'PanDerm_Large_LP')`,
`forward_features()` only — no classifier head trained or used for this arm.
Checkpoint metadata: `assets/panderm_vitl16_checkpoint_metadata.{md,json}`.
Preprocessing snapshot: `assets/reference_preprocessing.json`.
Validation: `reference_instrument_validation.md`, `outputs/stage1_3_rerun_preregistered_split.json`.

## 3. Primary Artifacts

| Artifact | Population | Used for | Status | Path |
|---|---|---|---|---|
| `ReferenceEmbedding` | Held-out ID (ISIC-test, n=5,067) + OOD (PAD-UFES, n=2,298) | Evaluation | FROZEN (Stage 2.1) | `assets/reference_embeddings/ReferenceEmbedding/` |
| `ReferenceTrainEmbedding` | ISIC 2019 train (n=16,211) | Reference fitting (Group B) | FROZEN (EA-01, **Closed**) | `assets/reference_embeddings/ReferenceTrainEmbedding/` |

## 4. Artifact Provenance & Integrity

- `ArtifactCharacterization/` — FROZEN. Shape (7365,1024), norm/variance statistics, class counts. (Stage 3.1)
- `DuplicateRegistry/` — FROZEN. 17 pixel-identical-image groups (34 rows), 13/17 with conflicting labels. No dataset modification. (Stage 3.0)
- `ArtifactIssuesRegister/` — FROZEN. Issue #1 = the duplicate finding above — logged, no action taken.
- Standing rule (`00_execution_checklist.md`): every future Claim C1–C5 write-up must explicitly name which `ArtifactIssuesRegister/` entries were consulted.

## 5. Group A — Representation Characterization (COMPLETE)

Source: `ReferenceArmBaseline/BaselineReport/` (not frozen).

- **Gate 0 (Tier 0): PASS.** Both criteria met: domain-leakage probe accuracy (0.9990) exceeds
  majority baseline (0.6880) + shuffle-control (0.6136) by >0.05; ID-task balanced accuracy (0.5740)
  exceeds chance (0.125) by >0.10. → `gate0_verdict.json`
- **Domain-leakage probe (Tier 2):** mean accuracy 0.9990 (±0.0005, 5 seeds), mean AUROC 0.999997,
  shuffle-control accuracy 0.6136. → `probe_results_summary.json`
- **Geometry (Tier 1+2):** condition number 1,338,282; LID (k=50) 15.03; spectral-decay slope
  0.006973; composite (LID × slope) 0.1048. Population: combined ID+OOD, n=7,365. → `geometry_metrics.json`
- **ID task performance (Tier 1+2):** balanced accuracy 0.5740 (±0.0141, 5 seeds), ECE 0.1821
  (±0.0070). → `id_task_performance.json`
- **Not computed:** per-direction variance — no resolved operationalization (no "intervened
  direction" exists on the Reference arm alone). → `not_yet_computed.json`

## 6. Group B — Reliability Characterization, Mahalanobis-family (COMPLETE)

Source: `GroupBExecution/` (FROZEN 2026-08-05; `execution_manifest.json` is the completion certificate).

Chain: Execution Contract (`execution_contract.md`) → Execution (`analysis/groupB_mahalanobis_family.py`)
→ Execution Review (`review.md`) → Execution Sanity (`review_results.md`, 4/4 checks YES) → Freeze.

| Scorer | AUROC | FPR@95 |
|---|---|---|
| Mahalanobis | 0.9984 | 0.0081 |
| kNN (k=1) | 0.9988 | 0.0057 |
| kNN (k=10, primary) | 0.9979 | 0.0111 |
| kNN (k=50) | 0.9967 | 0.0180 |
| Cosine-to-centroid | 0.9805 | 0.0807 |
| KDE | 0.8942 | 0.3898 |

All Tier 1 (Measurement), single-arm. No cross-arm comparison exists yet — there is no Intervention
arm to compare against — and none is implied here.

## 7. Execution Amendments

- **EA-01 — Closed.** Produced `ReferenceTrainEmbedding/` to close one specific gap: Group B lacked
  a reference fitting population. Closed once the four scorers it was scoped for (Mahalanobis, kNN,
  cosine-to-centroid, KDE) consumed it, were reviewed, and were sanity-checked. Full record:
  `ExecutionAmendments/EA-01.md`.

## 8. Explicitly deferred — not part of this baseline

- **Logit-based scorers work package** (Energy, ViM) — blocked on a trained classifier head for the
  Reference Instrument. A completely different dependency from the embedding artifacts above
  (logits, calibration — not embeddings), not in scope for any existing amendment. → `03c_groupB_readiness.md`
- **Relative Mahalanobis** — NOT IMPLEMENTED anywhere in the reused codebases (`paper-3/`,
  `CSG-SKin/`, `DST-Skin/`); a separate open decision, independent of data availability.
- **Per-direction variance** — no resolved definition; the Reference arm alone has no "intervened
  direction" for this to be measured against.

## 9. What this document is not

No new computation. No new figure. No hypothesis test. No cross-arm comparison (Stage 4 doesn't
exist yet). No Tier-3 language. Every number above is quoted, not recomputed, from the cited source
file — if a number here and its source ever disagree, the source file is authoritative, not this one.

## 10. Forward pointer

This is the Reference Arm baseline that every Stage 4 (Intervention) and Stage 5 (Comparative
Analysis) comparison should cite by pointing back to this single document — not to the individual
artifact directories separately.
