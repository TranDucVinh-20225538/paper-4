# Reference Arm Baseline Evaluation — Execution Plan

**Status: PARTIALLY COMPLETE (2026-08-05). Stage 3.2 is NOT closed.** Plan approved and executed
same day; PI review found the result genuinely partial, not a pass overall:

- **Group A: COMPLETE.** Domain-leakage probe, geometry metrics (excl. per-direction variance), ID
  task performance, Gate 0 — all executed, PASS. See §8.
- **Group B: BLOCKED.** Reason: reference fitting population unavailable. Traced precisely in
  `GroupB_protocol_reconciliation.md` — not worked around.

Also corrected same day: Tier-2/Tier-0 artifacts must not contain cross-arm/cross-paper comparative
language ("more pronounced than Paper 1") — that is Tier-3 (Inference) territory, out of scope until
Stage 5. `probe_results_summary.json` edited to remove the comparison field; raw measured numbers
only.

## Stage flow at a glance

| Stage | Purpose | Produces | Consumed by |
|---|---|---|---|
| Validation | Verify the instrument/arm is usable at all | Gate report | Measurement |
| Measurement | Generate raw quantities | Measurement artifacts | Description |
| Description | Summarize one arm | Baseline report | Inference |
| Inference | Compare arms, test claims | Claim evidence | Manuscript |

---

## 1. Naming and architecture correction (resolved 2026-08-05)

The first draft of this document called itself "Claim C1 Execution Plan." Wrong label — the fix is
structural, not cosmetic.

**Claim C1 is closed** — it maps specifically to Stage 0's retrospective CNN check
(`01_stage0_retrospective.md`, τ=−0.154, p=0.510, already run). `ReferenceEmbedding/` (PanDerm →
Reference arm → frozen representation) is **input data for later claims, not an answer to C1** — it
cannot re-test a claim that was already tested on different data.

**What this stage actually is: baseline measurement, not hypothesis testing.**

```
Stage 3.2 — Reference Arm Baseline Evaluation      (this document)
    Validation → Measurement → Description → Baseline Report

Stage 4 — Intervention Arm
    (same measurement suite, applied to Intervention/Conventional/Adaptation arms)

Stage 5 — Comparative Analysis
    Reference vs. Intervention → Claims C2, C3, C4, C5 (Predictions 1–2 actually tested here)
```

Renamed from "ClaimC1/" to `ReferenceArmBaseline/` accordingly. No claim in `02_Claim_Matrix.md` is
tested by this stage — it produces the baseline numbers Stage 5 will later need, and runs the
Validation check the protocol requires before the Reference arm's results can be used as evidence
for anything.

## 2. Input artifact

- `assets/reference_embeddings/ReferenceEmbedding/` (frozen, Stage 2.1) — `embeddings.npy` (7365,
  1024), `metadata.csv` (path, domain, label_idx, label_name), `labels.csv`.
  - ID (ISIC-test): n=5,067. OOD (PAD-UFES-20): n=2,298.
- `assets/ArtifactIssuesRegister/issues_register.json` — Issue #1 (34 duplicate rows, 13/17 groups
  with conflicting labels) must be explicitly consulted per the claim-discipline rule in
  `00_execution_checklist.md`.

## 3. Domain-leakage probe data split — RESOLVED, grounded in Paper 1's own code

Resolved by reading (read-only, not executed, not modified) `CSG-SKin/scripts/check_leakage.py` —
the actual script "Paper 1's exact protocol" refers to — rather than guessing a split scheme.

**Protocol-level description (paper-facing):** domain-leakage probing follows the protocol
established in CSG-Skin by fitting a linear probe on randomly partitioned frozen representations,
using stratified train/test splits, repeated across multiple seeds with a shuffle-label control.

**Implementation-level grounding (kept alongside, for reproducibility):**

1. Builds `datamodule.ood_test_dataset` = `pd.concat([isic_test, pad_df])` — the **same** combined
   ID(ISIC-test)+OOD(PAD-UFES) population `ReferenceEmbedding/` already contains. No separate
   training-domain embedding extraction is used or needed.
2. `linear_probe_accuracy()`: `sklearn.model_selection.train_test_split(features, domains,
   test_size=0.3, random_state=seed, stratify=domains)` — an internal 70/30 split of that same
   combined population, stratified by domain, refit per seed.
3. Probe: `make_pipeline(StandardScaler(), LogisticRegression(max_iter=..., random_state=seed))`.
4. Shuffle-label sanity check: domain labels shuffled once with a **fixed** `shuffle_test_seed`
   (script default 123), probe refit on the first probe seed with the same `test_size`.
5. **Seed-count discrepancy, resolved not silently:** script default is 3 seeds (`42,52,62`), but
   `22_Experimental_Design.md` §4 measurement 1 states "5-seed protocol." **Protocol wins over
   implementation default: use 5 seeds (42, 52, 62, 72, 82)**, matching both the protocol text and
   this project's established 5-seed convention (Paper 3's ladder).

## 4. Four-tier structure — every output artifact tagged (per Vinh, 2026-08-05)

Corrected from a three-tier draft: **Gate checks are Validation, not Description** — a gate is a
judgment about whether the arm is usable at all, evaluated against a fixed external reference; it is
not a summary of the arm's own properties the way Tier 2 is.

| Tier | What it is | Example | Exists in Stage 3.2? |
|---|---|---|---|
| **0 — Validation** | Implementation Gate (Gate 0) / Manipulation Gate (Gate 1). Judges whether results from this arm are usable as evidence at all. | Gate 0 PASS/FAIL for the Reference arm | **Yes (Gate 0 only)** — Gate 1 (Manipulation Check) applies to Intervention/Conventional-arm checkpoints only (`22_Experimental_Design.md` §3.5); not applicable to an untouched Reference arm, noted rather than silently skipped |
| **1 — Measurement** | Running a method on the frozen artifact, raw per-run numbers. No aggregation, no interpretation. | Per-seed probe accuracy; per-sample Mahalanobis distance; a single condition-number value | **Yes** |
| **2 — Description** | Aggregating Tier-1 numbers into a summary describing *this arm alone* (mean/std across seeds, AUROC from per-sample scores, histograms). Still no comparison to another arm. | Mean probe accuracy ± std across 5 seeds; Mahalanobis AUROC; summary tables | **Yes** |
| **3 — Inference** | Comparing Reference-arm Tier-2 numbers against Intervention-arm Tier-2 numbers; correlation/causal statistics; Claims C2–C5. | Geometry-vs-AUROC correlation (C2); normalization-knob causal test (C3) | **No — this is Stage 5, not produced here** |

**Execution-order note:** Tier 0 (Gate 0) is epistemically "first" (it gates whether anything else
is usable as evidence) but Gate 0's own *criterion* consumes a Tier-1 quantity (domain-probe
leakage) and a Tier-1/2 quantity (ID task balanced accuracy/ECE) as its input. Chronological
execution is therefore: generate the relevant Tier-1 measurements → evaluate the Tier-0 gate against
them → produce the full Tier-2 description (which includes, but is not limited to, reporting the
Tier-0 verdict). The tier label describes epistemic role, not literal execution order.

Grouped by *what* is measured, independent of tier:

- **Group A — Representation characterization:** domain-leakage probe, geometry metrics (condition
  number, LID, spectral-decay slope, per-direction variance).
- **Group B — Reliability characterization:** distance-based scorers (Mahalanobis primary; kNN,
  energy, ViM, KDE, cosine-to-centroid, relative-Mahalanobis secondary).

**On MSP:** not included — MSP is a *logit-space* score requiring a trained classification head; the
Reference Instrument is used purely via `forward_features` (no head trained on this task), and MSP
isn't in `22_Experimental_Design.md` §4 measurement 2's preregistered scorer list — it appears only
in the literature review of other papers' (Gallo, Paper 2) logit-space comparisons.

## 5. Expected analyses, tagged by tier and group (source: `22_Experimental_Design.md` §4)

| # | Measurement | Group | Tier(s) produced | Method | Reused from |
|---|---|---|---|---|---|
| 1 | Domain-leakage linear probe | A | 1 (per-seed accuracy) → 2 (mean±std, shuffle-control comparison) | 5-seed (42,52,62,72,82), `test_size=0.3` stratified split of `ReferenceEmbedding/` itself, + shuffle-label sanity check (seed 123) | `CSG-SKin/scripts/check_leakage.py` (Paper 1), §3 above |
| 2 | Geometry metrics | A | 1+2 (single condition — no seed/aggregation axis, so raw value *is* the reported value) | Condition number (Paper 3's metric), LID, within-class spectral-decay slope (Mechanism 1), per-direction variance (Mechanism 2) | Paper 3 (condition number) + `PhaseB/analysis/lid_spectral_decay.py` |
| 3 | Distance-based reliability scorers | B | 1 (per-sample scores) → 2 (AUROC/FPR@95TPR per scorer) | Mahalanobis (primary) + kNN/energy/ViM/KDE/cosine-to-centroid/relative-Mahalanobis (secondary), ID-vs-OOD detection | Paper 3's implementation |
| 4 | ID task performance | — (Gate 0 input) | 1 (per-sample correctness/confidence) → 2 (balanced accuracy, ECE) | Feeds Gate 0 only, not a standalone claim | Standard, per §3.5 — **methodology needs the same grounding pass as measurement 1 before execution; not yet confirmed against a precedent script (see §8)** |
| 5 | Probe-weight diffuseness | A (auxiliary) | 1 → 2 | Nearly free — inspects trained probe's weight-vector concentration | `16_Minimal_Sufficient_Experiment.md` |
| — | Gate 0 verdict | — | **0 (Validation)** | PASS/FAIL determination against Paper 1's fixed CNN baseline (0.979) | `22_Experimental_Design.md` §3.5 |

**No row in this table reaches Tier 3.** Tier 3 (Claims C2–C5) is Stage 5, contingent on Stage 4
(Intervention Arm) existing.

## 6. Output artifact (planned, not yet fully created), tier-tagged

`PhaseB/ReferenceArmBaseline/BaselineReport/`, frozen the same way as prior artifacts (`chmod
444`/`555`, SHA256, manifest):

| File | Group | Tier | Contents |
|---|---|---|---|
| `gate0_verdict.json` | — | **0** | PASS/FAIL determination, full provenance |
| `probe_results_raw.json` | A | 1 | Per-seed probe accuracy/AUROC (5 seeds), shuffle-label run |
| `probe_results_summary.json` | A | 2 | Mean±std across seeds; shuffle-control comparison |
| `geometry_metrics.json` | A | 1+2 | Condition number, LID, spectral-decay slope, per-direction variance |
| `reliability_scorers_raw.json` | B | 1 | Per-sample scores (or a pointer to them) for each scorer |
| `reliability_scorers_summary.json` | B | 2 | AUROC/FPR@95TPR per scorer |
| `id_task_performance.json` | — | 1+2 | Balanced accuracy, ECE |
| `manifest.json` | — | — | Birth certificate; references `ReferenceEmbedding/`'s exact file hashes and `ArtifactIssuesRegister`'s Issue #1; states explicitly `"tier_3_claims_tested": []` |

## 7. Dependency graph

```
ReferenceEmbedding/ (Stage 2.1, frozen)
        │
        ├── ArtifactIssuesRegister/ Issue #1 (consulted, not resolved)
        │
        ▼
  ┌─────────────────────────────┬─────────────────────────────┐
  │ Group A: Representation     │ Group B: Reliability         │
  │  - Domain-leakage probe     │  - Mahalanobis (primary)     │
  │  - Geometry metrics         │  - kNN/energy/ViM/KDE/       │
  │  - ID task performance      │    cosine/relative-Maha      │
  └─────────────┬────────────────┴──────────────┬──────────────┘
                │  Tier 1 (raw)                  │  Tier 1 (raw)
                ▼                                ▼
                └──────── Tier 2 (Description / Baseline Report) ────┘
                                        │
                                        ▼
                       Tier 0 — Gate 0 verdict (Reference arm)
                                        │
                        ┌───────────────┴────────────────┐
                        ▼                                 ▼
                      PASS                              FAIL
           (arm eligible as Stage 5's         (re-run with adjusted budget or
            Reference-side baseline)           fall back per Risk Register #3)

                              Stage 4 (Intervention Arm)
                          (same Tier 0 → 1 → 2 pipeline)
                                        │
                                        ▼
                    Stage 5 — Comparative Analysis (Tier 3)
                    Reference Tier-2 vs. Intervention Tier-2
                              → Claims C2, C3, C4, C5
```

## 8. Execution result (2026-08-05) — PARTIAL, two items flagged rather than invented

Executed: domain-leakage probe, geometry metrics (excl. per-direction variance), ID task
performance (resolved by extending measurement 1's methodology to the diagnostic label — a natural
reuse, not a new invention), Gate 0. Script: `PhaseB/analysis/stage3_2_reference_arm_baseline.py`.
Output (not yet frozen, pending PI review of the two open items below):
`ReferenceArmBaseline/BaselineReport/`.

**Two items NOT executed, flagged for a PI decision rather than worked around:**

1. **Per-direction variance** (part of measurement 3, Mechanism 2). No resolved operationalization
   exists for "the intervened direction" on the Reference arm alone — there is no intervention yet
   (Stage 4 doesn't exist), so there's nothing to name as "the" direction. Candidate for later
   resolution once Stage 4 exists, or a proxy definition (e.g. the domain-probe's own weight
   direction) is explicitly approved.
2. **Group B (reliability scorers: Mahalanobis, kNN, energy, ViM, KDE, cosine-to-centroid,
   relative-Mahalanobis).** Discovered during implementation, not anticipated in the original plan:
   Paper 3's own implementation of these scorers fits on a **separate training population**
   (ISIC-TRAIN, ~16k images) distinct from the held-out ID/OOD population used to evaluate them.
   `ReferenceEmbedding/` (Stage 2.1) contains only the held-out ID(ISIC-test)+OOD(PAD-UFES) split —
   Stage 2.1's authorized scope was explicitly "the complete preregistered ID and OOD datasets"
   (the evaluation split), not a training split. Computing Group B requires either a new embedding
   extraction (a Stage 2.2, not yet authorized) or an explicitly-approved methodology deviation
   (e.g. fitting on a held-out subset of the existing ID population). Genuine data-availability
   gap, not a methodology choice — surfaced rather than resolved unilaterally.

## 9. Gate

**Gate 0 — Implementation Integrity Gate** (`22_Experimental_Design.md` §3.5), Reference arm
specifically: representation not collapsed, and — the Reference-arm-specific criterion —
**domain-probe leakage at a magnitude consistent with Paper 1's CNN baseline** (Paper 1: probe
accuracy 0.979 on the CNN).

> Per `16_Minimal_Sufficient_Experiment.md` Risk Register item #3: if there's no domain leakage in
> the untouched FM baseline, there is nothing for the Intervention arm to act on, and the design
> needs a backbone/layer fallback before C2–C5 can be evaluated at all.

## 10. Stop condition

Stop after the Tier-0 Gate 0 verdict and the full Tier-2 Baseline Report are computed and frozen.
**No Tier 3 output is produced in this stage** — Tier 3 is Stage 5, and requires Stage 4
(Intervention Arm) to exist first.
