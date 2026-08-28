# Stage 3.2 — Group B Readiness Review

**Tier 0 (Validation). Not a computation, not a measurement, not a claim test.** Produces a status
table only — which prerequisites each Group B scorer needs, and whether those prerequisites now
exist. **No scorer is executed here. No score, distance, or statistic is computed.** Requested by
Vinh, 2026-08-05, as the required checkpoint between `ReferenceTrainEmbedding/` (EA-01) landing and
any Group B scorer being authorized to run.

This review re-derives its conclusions from artifact state actually verified on disk (manifests
read directly, not recalled), not from the earlier `GroupB_protocol_reconciliation.md` trace by
assumption — that note remains the source for *why* each requirement exists; this note confirms
*whether it's now satisfied*.

## 1. Artifact-level prerequisites

| Requirement | Status | Evidence |
|---|---|---|
| `ReferenceEmbedding` (held-out ID+OOD) | **PASS** | `manifest.json`: `status: FROZEN`, shape `[7365, 1024]` |
| `ReferenceTrainEmbedding` (ISIC-train, z_train) | **PASS** | `manifest.json`: `status: FROZEN`, shape `[16211, 1024]`, `population: ISIC2019_train` |
| Trained classifier head (Reference Instrument) | **BLOCKED** | No checkpoint exists anywhere under `PhaseB/assets/` (checked: no file matching `*classifier*` or `*head*`). PanDerm has been used exclusively via `forward_features()` since Stage 1.2 — its `head` layer (`num_classes=1000`, ImageNet-pretrained default) has never been fine-tuned or evaluated for this task and is not a valid source of task logits. |

## 2. Per-scorer readiness

Each row traces to the exact requirement established in `ReferenceArmBaseline/GroupB_protocol_reconciliation.md` §2, re-checked against current artifact state.

| Scorer | Needs z_train (`ReferenceTrainEmbedding`)? | Needs trained classifier head? | Status |
|---|---|---|---|
| **Mahalanobis** | Yes | No | **READY** |
| **kNN** | Yes | No | **READY** |
| **Cosine-to-centroid** | Yes (same per-class means as Mahalanobis) | No | **READY** |
| **KDE** | Yes | No | **READY** |
| **Energy** | No | Yes | **BLOCKED** |
| **ViM** | Yes | Yes | **PARTIALLY BLOCKED** — z_train prerequisite now satisfied; classifier-head prerequisite still missing |
| **Relative Mahalanobis** | Would need it, by the standard formulation | Unclear (no implementation to check) | **NOT IMPLEMENTED** — no code found anywhere in `paper-3/`, `CSG-SKin/`, or `DST-Skin/` (confirmed by repo-wide search, `GroupB_protocol_reconciliation.md` §2). Distinct from "blocked": this is an absent-implementation question, not a missing-artifact one — whether to implement it at all is a separate open decision, independent of anything EA-01 addressed. |

## 3. Overall status

**Group B is Partially Ready — not fully ready, not fully blocked.** Three independent states,
correctly kept separate rather than collapsed into one verdict:

- **Mahalanobis-family (Mahalanobis, kNN, cosine-to-centroid, KDE): READY.** Every prerequisite
  these four scorers need now exists and is frozen.
- **Energy: BLOCKED.** Depends entirely on a trained classifier head, which does not exist and is
  untouched by EA-01. Resolving this is a separate decision (whether the Reference arm ever gets a
  trained linear head, and if so, training one) — not an embedding-extraction problem.
- **ViM: PARTIALLY BLOCKED.** One of its two prerequisites (z_train) is now satisfied; the other
  (classifier head, for both fitting and scoring) is not. Blocked on the same open question as
  Energy, not on anything related to `ReferenceTrainEmbedding/`.
- **Relative Mahalanobis: NOT IMPLEMENTED.** Out of scope for any near-term authorization until a
  separate decision is made on whether to implement it at all.

## 4. What this review does not authorize

This table confirms prerequisite readiness only. **It does not itself authorize running any
scorer.** Per Vinh's explicit instruction (2026-08-05), authorization to compute the
Mahalanobis-family scorers is a separate, subsequent PI decision, to be given only after this review
has been read. Energy and ViM's classifier-head dependency is a distinct kind of gap from anything
`ReferenceTrainEmbedding/` resolved, and will be tracked as its own execution amendment if and when
Vinh decides to pursue it — not bundled into whatever authorizes the Mahalanobis-family scorers.
