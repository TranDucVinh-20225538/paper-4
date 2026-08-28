# Stage 3.2 — Reference Arm Baseline Evaluation (formerly "Stage 2 — Baseline")

**Renumbering note (2026-08-05, Vinh's PI decision, corrected twice same day):** this file's
content is now **Stage 3.2 — Reference Arm Baseline Evaluation** within the renamed "Stage 3 —
Scientific Analysis" — the filename is unchanged (`03_stage2_reference.md`) to avoid breaking
existing `26_Lab_Notebook.md` references. Every "Stage 2" below should be read as "Stage 3.2." Two
stages sit before this content: Stage 2 ("Reference Representation" — embedding extraction,
`02a_stage2_reference_representation.md`, complete) and Stage 3.0/3.1 (Artifact Audit / Artifact
Characterization, `03a_`/`03b_stage3_*.md` — audit/structural only, not analysis).

**Important, corrected same day:** this stage is **baseline measurement, not a claim test** — it
was briefly mislabeled "Claim C1," which is wrong (C1 is already closed at Stage 0; see
`ReferenceArmBaseline/reference_arm_baseline_execution_plan.md` §0). This pipeline produces the
Reference-arm side of the comparison Stage 5 (Comparative Analysis) will later run against Stage 4
(Intervention Arm) — it does not, by itself, test C1–C5. Full execution plan, dependency graph, and
the domain-leakage probe's data-split resolution (grounded in `CSG-SKin/scripts/check_leakage.py`):
`ReferenceArmBaseline/reference_arm_baseline_execution_plan.md`. See `00_execution_checklist.md`'s
Stage 3/4/5 table for the full architecture.

**Status: not started.** Blocked on Stage 2 (Reference Representation) producing and passing the
Embedding Integrity Gate on the frozen Reference Embedding Artifact this stage's probe/geometry
work will consume.

## Scope (source: `22_Experimental_Design.md` §3 Reference arm row, §3.5 Gate 0)

Reference arm only — no adaptation, frozen pretrained FM embeddings, linear head only for the
classification task. Pipeline:

```
Frozen representation → Probe → Geometry metrics → Accessibility metrics → Reliability metrics
```

Purpose: establish baseline probe-decodability and distance-scorer behavior *before* any
intervention, and run Gate 0 (Implementation Integrity Gate) specifically for the Reference arm —
per `16_Minimal_Sufficient_Experiment.md` Risk Register item #3: if there's no domain leakage in
the untouched baseline, there's nothing for the Intervention arm to act on, and the design needs a
backbone/layer fallback before any of C2–C5 can be evaluated.

## Checklist

Organized into Group A / Group B, each producing Tier 1 (raw measurement) then Tier 2 (descriptive
report) — never Tier 3 (comparative/claim-testing, that's Stage 5). Full tier/group definitions:
`ReferenceArmBaseline/reference_arm_baseline_execution_plan.md` §3.

**Group A — Representation characterization:**
- [ ] Domain-leakage linear probe: 5 seeds (42,52,62,72,82), internal 70/30 stratified split of
      `ReferenceEmbedding/` itself (`train_test_split(..., test_size=0.3, stratify=domain,
      random_state=seed)`), + shuffle-label sanity check (seed 123) — methodology resolved and
      grounded in `CSG-SKin/scripts/check_leakage.py`, see the execution plan §2.
- [ ] Geometry metrics: condition number, LID, within-class spectral-decay slope, per-direction
      variance of the (not-yet-intervened) candidate direction.

**Group B — Reliability characterization:**
- [ ] Distance-based reliability scorers (Mahalanobis primary; kNN/energy/ViM/KDE/cosine-to-centroid
      /relative-Mahalanobis secondary), reused from Paper 3's implementation. (MSP explicitly not
      included — logit-space score requiring a trained head this pipeline doesn't use; not in the
      preregistered scorer list either. See execution plan §3.)

**Feeds Gate 0 (not a standalone claim):**
- [ ] ID task performance (balanced accuracy, ECE).
- [ ] Gate 0 verdict for the Reference arm: PASS → eligible as Stage 5's Reference-side baseline;
      FAIL → re-run with adjusted budget or fall back per Risk Register item #3, before any
      Intervention-arm work starts.

## Not in scope for this stage

Any intervention arm (Intervention, Conventional, Adaptation) — those are Stage 4, contingent on
this stage's Gate 0 passing. Any Claim C2–C5 evaluation — that's Stage 5 (Comparative Analysis),
contingent on both this stage and Stage 4 passing Gate 0.
