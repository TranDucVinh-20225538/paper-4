# Experimental Design — Paper 4 (Step 6)

**Status: SPECIFICATION ONLY, v2 (2026-08-04, session 8). No code, no implementation, no
execution.** Originally written session 6 per Vinh's Step 6 prompt: "Design the minimal
experimental protocol under the locked RQ... Produce only the experiment specification... Ensure
every experiment directly tests one claim in the updated Claim Matrix." This document is
downstream of, and does not repeat, `16_Minimal_Sufficient_Experiment.md` (the resource-
constrained design sketch) and `Scientific_Hypothesis.md` (the locked hypothesis and predictions).

**v2 changes (merged from `22a_Design_Amendments.md`, following `23_Experiment_Contract.md`'s
audit):** added §3.5 (Implementation Integrity Gate, Manipulation Check Gate), revised §4
measurement 4, and revised §7's C4 success criteria. Full history of what was found, proposed, and
corrected before merge is preserved in `22a_Design_Amendments.md`; the interpretive Evidence Ladder
framework lives separately in `22b_Evidence_Ladder.md` (appendix, not part of this Methods
document).

---

## 0. Locked RQ (restated)

**When a dermatology foundation model's representation is reshaped by an explicit intervention —
domain-adversarial disentanglement as the primary instantiation tested, with a matched
ordinary-fine-tuning arm as a control for intervention type — do the resulting changes in
embedding geometry mediate the accessibility of acquisition-domain shift to post-hoc distance-based
reliability estimators, independent of whether that information remains linearly decodable?**

Chain under test: `Representation intervention → Geometry changes → Accessibility changes →
Reliability changes`. Prediction 3 (architecture-agnosticism) is out of scope per
`21_Prediction3_Removed_Reevaluation.md` — this design tests Predictions 1 and 2 only, on one
architecture family (one dermatology FM) plus the already-existing CNN data from Paper 3.

---

## 1. Claim → experiment mapping

Every experimental component below tests exactly one row from the Claim Matrix's "Unknown"
section (`02_Claim_Matrix.md`, Claim Hierarchy table). No component is included that doesn't map
to one of these five rows.

| # | Claim under test (from Claim Matrix) | Experiment component |
|---|---|---|
| C1 | Which geometric quantities explain the trade-off (LID, spectral-decay slope, condition number, variance allocation)? | Stage 0 — Retrospective geometry computation on Paper 3's existing CNN checkpoints |
| C2 | Does distance-estimator degradation track geometry specifically, not intervention-strength label or probe accuracy? | Stage 3 — Geometry-vs-AUROC correlation analysis (Prediction 1) |
| C3 | Does holding geometry constant across differing intervention strengths hold degradation constant (causal test)? | Stage 3 — Post-hoc normalization-knob manipulation (Prediction 2) |
| C4 | Do an ordinary-fine-tuning intervention and a disentanglement intervention produce the same or different geometry→accessibility relationship? | Stage 2 — *Conventional* arm (matched-budget ordinary FT) vs. *Intervention* arm (GRL+orthogonality) |
| C5 | Does domain information stay linearly decodable on an FM representation after a disentanglement-style intervention, the way it does on the CNN? | Stage 2/3 — Domain-leakage linear probe on every FM checkpoint |

**v2 note:** C1–C5 are all gated by §3.5 (Implementation Integrity Gate, Manipulation Check
Gate). No claim above is evaluated on an arm/checkpoint that hasn't passed its applicable gate.

---

## 2. Datasets

**No new data collection.** Reuses the existing ISIC 2019 ↔ PAD-UFES-20 acquisition-shift testbed
already used in Papers 1–3 (27,629 images: 25,331 ISIC 2019 dermoscopy + 2,298 PAD-UFES-20
smartphone photography). Rationale: this is the only option requiring zero new data collection
(the 6-week budget cannot absorb new dataset curation), and it keeps the acquisition-shortcut
construction identical to Papers 1–3, so results are directly comparable rather than confounded by
a redefinition of what "domain shift" means. Gallo et al.'s dataset choices (PAD-UFES-20 +
Derm7pt-clinic, HAM10000 as near-OOD) are deliberately **not** adopted — their near-OOD framing
(clinical vs. dermoscopic modality as the shift) targets a different question (semantic/modality
OOD) than ours (acquisition-source shortcut within a shared representation). Mixing the two
dataset framings would blur exactly the distinction the novelty analysis established.

- **CNN arm (Stage 0):** no new extraction — reuses Paper 3's existing embeddings and stored
  Mahalanobis/scorer outputs across λ_orth = 0, 1, 5.
- **FM arm (Stages 1–3):** ISIC 2019 (train split) + PAD-UFES-20 (train split) for
  representation/adaptation; held-out splits of both for evaluation, matching Papers 1–3's
  train/eval protocol exactly (same split seeds where feasible, for direct comparability).

**Backbone dependency (flagged, not resolved here):** the target FM is PanDerm (arXiv:2410.15038)
per `02_Claim_Matrix.md`'s "Tier 1 — need architecture/pretraining details" flag. Checkpoint
availability and license terms have not been confirmed in this project. If PanDerm access fails,
the fallback is a CLIP ViT-B/16 backbone (the same one Gallo et al. used and confirmed strong
dermatology near-OOD separability for) — this fallback is scientifically acceptable (it keeps the
FM-vs-CNN contrast) but weakens the "medical-FM-specific" framing and should be flagged explicitly
in the eventual paper if used. **This dependency should be resolved before Stage 1 begins.**

---

## 3. Intervention arms

**Framing note (2026-08-04, session 7, per Vinh):** these four arms are instruments for producing
controlled variation in geometry — they are not, individually or collectively, the object of the
paper. The contribution is the chain `geometry → accessibility → reliability`; none of the arms
should be read as "the paper's method." Named by role rather than by technique, so the design
doesn't read as being organized around any one intervention:

| Arm (role) | What it is | Purpose | Maps to claim |
|---|---|---|---|
| **Reference** | No adaptation; frozen pretrained FM embeddings, linear head only for the classification task | Establishes baseline probe-decodability and distance-scorer behavior before any intervention | Gate for Stage 1 (Risk Register #3) |
| **Intervention** | Adapter-based domain-adversarial training (GRL + orthogonality), 2–3 strengths mirroring Paper 3's λ_orth = 0, 1, 5 as closely as an adapter allows | The primary tested instantiation of "representation intervention" — produces the geometry variation the chain is tested against | C2, C3, C5 |
| **Conventional** | Adapter-based fine-tuning at matched adaptation *budget* (parameter count, training steps) to the Intervention arm, but with the plain task loss only — no adversarial/domain objective | A single matched-budget control point that isolates intervention **type** from intervention **strength/budget** — required because Gallo et al. (Level A) confirms ordinary fine-tuning intensity alone changes near-OOD separability | C4 |
| **Adaptation** | A progressive fine-tuning-intensity ladder (linear-probe / partial-FT / full-adapter-FT rungs), applied to the *same* backbone and dataset as the other three arms | Establishes a within-project fine-tuning-intensity curve to compare the Intervention arm's geometry–accessibility relationship against, across a *range* of conventional-adaptation intensity, not just one matched point | Supports C4 |

**On the Adaptation arm's relationship to Gallo et al.:** its rung structure is *implementationally*
similar to Gallo's R1–R6 (§3.4) — reusing a known, working ladder design is a reasonable
engineering choice — but its *purpose* in this design is not replication. Gallo's own reported
result already establishes that this trade-off exists; re-showing it would be redundant per
`03_Novelty_Threats.md`. The Adaptation arm exists so the Conventional arm's single control point
has a curve to sit on, letting C4's type×geometry interaction test be evaluated across a range of
conventional-adaptation strength, not just one budget-matched pair. If it happens to reproduce
Gallo's ID-accuracy/near-OOD pattern on this project's own backbone and dataset, that is a useful
sanity check that the setup is behaving as expected — not a result to report as a contribution.

---

## 3.5 Gates

Two gates, run in sequence, before any claim (C1–C5) is interpreted. Added in v2 following
`23_Experiment_Contract.md`'s audit (findings 1 and 2) — full rationale in
`22a_Design_Amendments.md`.

**Gate 0 — Implementation Integrity Gate.** Runs per arm (Reference, Intervention, Conventional,
Adaptation), independently.

```
Training converged
        ↓
Representation not collapsed
        ↓
Baseline (Reference arm) reproduces the expected shortcut-leakage pattern
        ↓
PASS → results from this arm are eligible to be interpreted as evidence about the hypothesis
FAIL → results from this arm are inconclusive, not falsifying — re-run with adjusted budget
        before drawing any conclusion
```

Measured via: balanced accuracy and calibration/ECE well above a trivial/majority-class baseline;
training-loss convergence; and, for the Reference arm specifically, domain-probe leakage at a
magnitude consistent with Paper 1's CNN baseline. This replaces the standalone "ID task
performance" item formerly in §4 measurement 4, which had no claim attached — it is now correctly
scoped as an implementation check, not a hypothesis test.

**Gate 1 — Manipulation Check Gate.** Runs on every Intervention-arm and Conventional-arm
checkpoint, and on every normalization-knob manipulation used for C3.

```
Intervention (or normalization knob) applied
        ↓
Did LID and/or spectral-decay slope actually change (not just condition number/scale)?
        ↓
YES → proceed to interpret accessibility/reliability changes as evidence about
      the geometry-mediation chain
NO  → the chain is not testable on this checkpoint — report as
      "claim not testable," not as "hypothesis contradicted"
```

> "Failure at Gate 1 does not constitute evidence against the hypothesis; it indicates that the
> intended manipulation was not achieved and therefore the causal chain cannot be evaluated."

This sentence should be quoted verbatim in any discussion of a Gate 1 failure, in this project or
in the eventual paper.

**On Gate 0/Gate 1 and the shared-assumption risks in `23_Experiment_Contract.md` §3:** these
gates do not eliminate the adapter-capacity or FM-dimensionality dependencies identified there —
under a 1-GPU, single-backbone budget, some coupling across claims is unavoidable and is not a
design defect to engineer away. What the gates do is make the point at which such a dependency
would produce an uninterpretable result *checkable and visible*, rather than allowing it to
silently masquerade as a clean falsification of the hypothesis.

---

## 4. Measurements

Computed identically across all arms/strengths, on held-out evaluation splits:

1. **Domain-leakage linear probe** (Paper 1's exact protocol — 5-seed multi-seed logistic
   regression on frozen post-intervention embeddings, with shuffle-label sanity check). Output:
   probe accuracy/AUROC for recovering acquisition source (ISIC vs. PAD-UFES-20).
2. **Distance-based reliability scorers**, reused from Paper 3's implementation: Mahalanobis
   distance (primary), plus kNN, energy, ViM, KDE, cosine-to-centroid, and relative Mahalanobis
   (secondary, for the multi-scorer convergence check Paper 3 performed). Output: AUROC/FPR@95TPR
   for detecting acquisition-domain shift.
3. **Geometry metrics:** condition number (Paper 3's original metric, for continuity), local
   intrinsic dimensionality (LID), within-class spectral-decay slope (both from Mechanism 1,
   `08_Mechanism_Map.md`, arXiv:2510.15202), and per-direction variance of the intervened
   direction (for Mechanism 2, variance suppression). Computed on the same held-out split as the
   scorers above.
4. **ID task performance** (balanced accuracy, calibration/ECE) — this is the measurement that
   feeds the Implementation Integrity Gate (§3.5), not a standalone claim test. It licenses
   interpretation of every other measurement on a given arm; it does not itself support or
   contradict the hypothesis.
5. **Probe-weight diffuseness** (nearly free, per `16_Minimal_Sufficient_Experiment.md`) — inspects
   the trained probe's weight vector concentration, to help distinguish "geometry mediates" from
   the entangled-encoding alternative (`13_Competing_Hypotheses.md`, Alt 2) if results are
   ambiguous. Not a primary measurement; recorded for interpretation only.

---

## 5. Controls

- **Conventional (ordinary-fine-tuning, matched budget)** controls for intervention *type* vs.
  *strength* — directly required by the Gallo verification (C4).
- **Reference (frozen baseline)** controls for whether any shortcut/geometry-of-interest is present
  before intervention at all (Risk Register #3 gate).
- **β-scaled ℓ2-normalization knob**, applied post hoc at a fixed intervention strength (both on
  the Intervention arm and on the Stage 0 CNN retrospective data), holds geometry approximately
  constant while intervention "strength" as nominally defined varies — the causal control for
  Prediction 2 (C3).
  This is a reused technique from the primary mechanism source (arXiv:2510.15202), not a new
  method to validate.
- **Fixed representation dimensionality** across all arms and strengths — controls for the
  distance-concentration confound (`08_Mechanism_Map.md`, Mechanism 4), which would otherwise
  predict a probe/distance gap from dimensionality alone, independent of any intervention.
- **Matched train/eval splits** across arms — controls for sampling variance being mistaken for an
  intervention effect.
- **Multi-seed training** (5 seeds, matching Paper 1's protocol) for every probe and every adapter
  training run — controls for single-run noise, needed given the small-sample risk already flagged
  in `11_Confound_Map.md` (PAD-UFES-20 ≈ 2.3k images).

---

## 6. Statistical analysis plan

- **C1 (Stage 0, retrospective):** Kendall's τ between LID×spectral-decay composite and
  Mahalanobis-AUROC across Paper 3's 3 existing λ_orth checkpoints — same statistic Paper 3 used
  for condition number, for direct comparability. Report τ and exact p-value.
- **C2 (Prediction 1):** correlation (Kendall's τ, consistent with above) between each geometry
  metric and each scorer's AUROC, across all Intervention-arm strengths and Adaptation-arm rungs
  jointly. Prediction 1 also
  requires showing probe accuracy is comparatively decoupled from the same geometry shifts —
  tested via a second correlation (geometry vs. probe AUROC), expected to be weak/non-significant
  if the hypothesis holds.
- **C3 (Prediction 2):** paired comparison of scorer-AUROC degradation at matched (normalized)
  geometry vs. matched (nominal) intervention strength — if geometry is truly the mediator,
  degradation should track the normalized-geometry grouping, not the nominal-strength grouping.
  Reported as a difference-in-degradation test (e.g., paired t-test or Wilcoxon signed-rank across
  seeds, depending on normality of the seed-level distribution — checked at analysis time, not
  pre-committed here).
- **C4 (control-arm comparison):** interaction test — does intervention *type* (Intervention arm
  vs. Conventional arm, at matched budget) significantly change the geometry–accessibility
  relationship (slope of geometry-vs-AUROC), or does only intervention *strength/budget* matter
  regardless of type? An interaction term (type × geometry) in a mixed-effects or simple
  interaction regression, with seed as a random/grouping factor.
- **C5 (probe-decodability):** replicate Paper 3's own significance test structure — probe AUROC
  vs. chance (0.5) via one-sample test across seeds, and probe AUROC vs. the corresponding
  distance-scorer AUROC via paired comparison, for every arm/strength. This is the direct FM
  analogue of Paper 3's headline 0.72–0.81 (probe) vs. ≈0.40 (Mahalanobis) dissociation.
- **Multiple comparisons:** per `11_Confound_Map.md` (flagged Medium-High risk), correct across
  the multiple scorer × geometry-metric × arm comparisons (Holm-Bonferroni or equivalent) before
  treating any individual correlation as confirmatory; report both corrected and uncorrected
  values for transparency.

---

## 7. Success / failure criteria

Directly inherits `Scientific_Hypothesis.md`'s falsifiable outcomes, operationalized:

**Supports the hypothesis:**
- C2: geometry metrics (not intervention-strength label, not probe accuracy) show significant,
  directionally-consistent correlation with scorer degradation, replicating C1's retrospective
  result on the FM.
- C3: normalization-matched geometry groups show statistically indistinguishable degradation
  despite differing nominal intervention strength (supports geometry as mediator, not label).
- C4 (v2, added following `23_Experiment_Contract.md`): the intervention-type effect (Intervention
  vs. Conventional, at matched budget) **disappears** after conditioning on geometry — supports
  full mediation: geometry, not intervention type, explains the accessibility outcome.
- C5: probe recovers domain information at well-above-chance AUROC while distance scorers remain
  flat/near-chance, on at least one Intervention-arm strength — the FM analogue of Paper 3's core
  result.

**Contradicts the hypothesis (still publishable, per the falsifiability design):**
- Degradation persists even when geometry is held constant (C3 fails) — implicates something
  GRL/orthogonality-specific rather than general geometry.
- C4 (v2): the intervention-type effect **remains significant** after conditioning on geometry —
  evidence against full mediation, consistent with either (a) partial mediation (geometry explains
  some, not all, of the intervention-type effect) or (b) an alternative mechanism specific to
  intervention type, not captured by the geometry metrics measured. Either sub-case means
  "representation intervention" is too coarse a category for the strong version of the claim, and
  the mechanism is at least partly intervention-specific.
- Probe-decodability itself turns out geometry-dependent (C5 fails to replicate the CNN pattern)
  — the "decodable but not accessible" framing needs revision on FMs, not just its explanation.

All C4 outcomes above assume Gate 1 (§3.5) passed on the relevant checkpoints. A Gate 1 failure on
a checkpoint removes it from the C4 analysis rather than counting toward either outcome — per
§3.5's protective sentence, an unmanipulated checkpoint tests nothing.

**Inconclusive / underpowered (a real possible outcome, to be reported honestly):**
- Reference shows no meaningful domain leakage on the frozen FM baseline — this is a Gate 0
  (§3.5) failure for the Reference arm specifically. Per `16_Minimal_Sufficient_Experiment.md`'s
  Risk Register item #3, this would mean there is nothing for the Intervention arm to act on, and
  the design would need to fall back to a different FM or a lower layer for probing before any of
  C2–C5 can be evaluated.
- Any arm failing Gate 0 or any checkpoint failing Gate 1 (§3.5) — reported as inconclusive for
  that arm/checkpoint specifically, not as evidence for or against the hypothesis.

---

## 8. Resource estimate

Extends `16_Minimal_Sufficient_Experiment.md`'s 6-week / 1-GPU budget with the added Conventional
and Adaptation arms (both cheap — same adapter-training infrastructure as the Intervention arm, no
new data pipeline):

| Week | Task | Arms involved | GPU cost |
|---|---|---|---|
| 0 | Retrospective LID/spectral-decay on existing Paper 3 CNN checkpoints (C1) — decisive gate | none (CNN data reused) | None |
| 1 | FM baseline domain-leakage probe, frozen embeddings (Reference) — second gate | Reference | Low (inference only) |
| 2–4 | Adapter-based Intervention ladder (2–3 strengths) **and** matched-budget Conventional control **and** progressive Adaptation ladder — run in parallel where GPU scheduling allows, since Conventional/Adaptation reuse the same adapter infrastructure as Intervention at lower marginal cost | Intervention, Conventional, Adaptation | Main budget (Conventional/Adaptation add ≈30–40% more training runs at the same per-run cost as Intervention, not a new cost category) |
| 5 | Geometry metrics + scorers + normalization-knob manipulation on every checkpoint from all four arms | Reference, Intervention, Conventional, Adaptation | Low (CPU-bound) |
| 6 | Statistical analysis (C1–C5) against Predictions 1 and 2; write-up of confirming or falsifying result | — | None |

**Net change from the prior 6-week sketch:** same wall-clock budget, same GPU-hour order of
magnitude — Conventional/Adaptation are additional *training runs*, not new infrastructure, and
the +30–40% run count is absorbed by the fact that adapter training (not full fine-tuning) is
already cheap per-run. If GPU time does become the binding constraint, the Adaptation arm (the
progressive ladder) is the first candidate to shrink to 3 rungs instead of 6, since its role is a
supporting curve for the Conventional arm's single control point, not a primary claim in itself.

**Open dependency:** PanDerm checkpoint/license confirmation (§2 above) should be resolved before
Week 1 begins; it is on the critical path but is not itself a GPU-cost item.

---

## 9. Explicitly out of scope

- Prediction 3 (architecture-agnosticism) — dropped per `21_Prediction3_Removed_Reevaluation.md`.
- Full fine-tuning of the FM backbone — cost-prohibitive under the stated budget; adapter-based
  intervention is the externally valid, affordable substitute.
- New dataset collection, additional FM backbones beyond the primary + fallback, and the
  alternative-hypothesis distinguishing tests beyond probe-weight-diffuseness (per
  `16_Minimal_Sufficient_Experiment.md`'s cut list — unchanged by this update).
- Any code, implementation, or execution — this document is a specification only, per Vinh's
  explicit instruction.
