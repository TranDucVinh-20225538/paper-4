# Pre-Registration — Paper 4

**Status: PROTOCOL FROZEN (2026-08-04, session 9).** This document is an internal
pre-registration, not a filed one — it has no legal or platform-based standing. Its purpose,
stated by Vinh: Paper 3 drew reviewer questions about multiple comparisons, post-hoc
interpretation, and statistical framing; a dated internal record that the hypotheses, outcomes,
and analysis plan below were fixed *before* running any experiment lets the eventual paper say,
truthfully, "these analyses and primary hypotheses were specified before running the experiments."

**Composition rule for this document: nothing here is new.** Every section below is copied or
directly restructured from `22_Experimental_Design.md` (v2), `22a_Design_Amendments.md`,
`22b_Evidence_Ladder.md`, `23_Experiment_Contract.md`, `Scientific_Hypothesis.md`, and
`05_RQ_Candidates.md`. No hypothesis, prediction, outcome, or success criterion is introduced here
for the first time. Where this document paraphrases rather than quotes, the paraphrase changes
only structure, not content — source is cited per section.

From this point forward, per Vinh's instruction: **no hypothesis or criterion below changes unless
extraordinary evidence forces it, and any such change is logged in §13 (Protocol Deviations), not
silently edited into this document.**

---

## 1. Research Question (locked)

Source: `05_RQ_Candidates.md`.

> When a dermatology foundation model's representation is reshaped by an explicit intervention —
> domain-adversarial disentanglement as the primary instantiation tested, with a matched
> ordinary-fine-tuning arm as a control for intervention type — do the resulting changes in
> embedding geometry mediate the accessibility of acquisition-domain shift to post-hoc
> distance-based reliability estimators, independent of whether that information remains linearly
> decodable?

Chain under test:

```
Representation intervention → Geometry changes → Accessibility changes → Reliability changes
```

---

## 2. Primary hypothesis (locked)

Source: `Scientific_Hypothesis.md`, "Geometry-mediated accessibility after representation
intervention."

> The dissociation between probe-decodability and distance-based reliability-estimator performance
> is not an artifact of the specific architecture (CNN), the specific intervention (GRL +
> orthogonality), or the specific domain (dermoscopy/smartphone) used in Paper 3. It is a general
> property of how a representation-reshaping intervention — of which domain-adversarial
> disentanglement is one instantiation, not the definition — changes local dimensionality,
> spectral structure, and variance allocation in ways that violate the assumptions specific
> distance-based estimators depend on, while leaving the suppressed direction accessible to any
> method that directly optimizes for it. The mediating variable is representation geometry, not
> the intervention family, and not the architecture that produced the representation.

Domain-adversarial disentanglement (GRL + orthogonality) is the primary tested instantiation
(Intervention arm). Ordinary fine-tuning (Conventional and Adaptation arms) is the required
control for intervention type, per the Gallo et al. verification (`03_Novelty_Threats.md`).

---

## 3. Secondary hypotheses

Source: `Scientific_Hypothesis.md` Predictions 1–2, `22_Experimental_Design.md` §7 (v2 C4
addition), and Paper 3's original finding (the object C5 tests for generalization).

- **Prediction 1 (→ C2):** distance-based-estimator degradation co-varies with measurable shifts
  in LID and/or within-class spectral-decay slope — not with a scalar intervention-strength label,
  and not with probe accuracy.
- **Prediction 2 (→ C3):** holding a geometry metric approximately constant across nominally
  different intervention strengths holds distance-estimator degradation approximately constant too
  (the causal, not merely correlational, version of Prediction 1).
- **Mediation hypothesis (→ C4, added in v2):** the intervention-type effect (Intervention vs.
  Conventional, at matched budget) disappears once geometry is conditioned on — i.e., geometry
  fully mediates the type effect, not just the strength effect.
- **Generalization hypothesis (→ C5):** domain/acquisition information remains linearly decodable
  on the FM representation after the Intervention arm's disentanglement objective, while
  distance-based scorers remain flat/uninformative — replicating Paper 3's CNN-era dissociation on
  a foundation model.

**Explicitly out of scope, not a hypothesis under test here:** Prediction 3
(architecture-agnosticism — does geometry explain distance-estimator failure independent of
whether the representation came from a CNN or an FM). Dropped per
`21_Prediction3_Removed_Reevaluation.md`; four independent audits converged on it as the weakest,
least affordable prediction. Reintroducing it would be a protocol deviation (§13), not a secondary
hypothesis.

---

## 4. Primary outcomes

Source: `16_Minimal_Sufficient_Experiment.md` (design explicitly sized for Predictions 1–2 only)
and `22_Experimental_Design.md` §6.

- **C2 / Prediction 1:** Kendall's τ between each geometry metric (LID, within-class spectral-decay
  slope, condition number, per-direction variance) and each reliability scorer's AUROC, across
  Intervention-arm strengths and Adaptation-arm rungs jointly. A second correlation (geometry vs.
  probe AUROC) is reported alongside, expected weak/non-significant if the hypothesis holds.
- **C3 / Prediction 2:** paired comparison (paired t-test or Wilcoxon signed-rank, chosen at
  analysis time by the seed-level distribution's normality) of scorer-AUROC degradation between
  normalization-matched-geometry groupings and nominal-strength groupings.

---

## 5. Secondary outcomes

Source: `22_Experimental_Design.md` §1, §6.

- **C1 (retrospective gate):** Kendall's τ between the LID×spectral-decay composite and
  Mahalanobis-AUROC across Paper 3's existing λ_orth = 0, 1, 5 CNN checkpoints — a decisive gate
  run before any FM budget is spent, not a claim about the FM.
- **C4 (mediation):** interaction term (intervention type × geometry) in an interaction regression
  or mixed-effects model, seed as grouping factor — tests whether the type effect survives
  conditioning on geometry.
- **C5 (decodability replication):** probe AUROC vs. chance (one-sample test across seeds) and
  probe AUROC vs. corresponding scorer AUROC (paired test), for every arm/strength that passes the
  applicable gates.

**Exploratory / auxiliary (not a primary or secondary outcome — explicitly non-primary in
`22_Experimental_Design.md` §4):** probe-weight-diffuseness, used only to help interpret an
ambiguous C2/C4 result, not to test a hypothesis on its own.

---

## 6. Manipulation checks

Source: `22_Experimental_Design.md` §3.5 (Gate 1), `22a_Design_Amendments.md` Amendment 2.

Before any Intervention-arm or normalization-knob checkpoint is used in the C2–C4 analyses, confirm
LID and/or spectral-decay slope actually shifted — not just condition number/scale. See §7 (Gates)
for the full procedure and the required interpretive sentence for a failed check.

---

## 7. Gates

Source: `22_Experimental_Design.md` §3.5.

**Gate 0 — Implementation Integrity Gate** (per arm: Reference, Intervention, Conventional,
Adaptation): training converged, representation not collapsed, baseline reproduces the expected
shortcut-leakage pattern. Measured via balanced accuracy/calibration well above a trivial baseline,
training-loss convergence, and (Reference arm specifically) domain-probe leakage at a magnitude
consistent with Paper 1's CNN baseline.

**Gate 1 — Manipulation Check Gate** (per Intervention/Conventional checkpoint and per
normalization-knob manipulation): did LID and/or spectral-decay slope actually change.

> "Failure at Gate 1 does not constitute evidence against the hypothesis; it indicates that the
> intended manipulation was not achieved and therefore the causal chain cannot be evaluated."

No claim in §4–5 is evaluated on an arm/checkpoint that has not passed its applicable gate(s).

---

## 8. Statistical tests

Source: `22_Experimental_Design.md` §6.

| Outcome | Test |
|---|---|
| C1 | Kendall's τ, exact p-value |
| C2 | Kendall's τ (geometry vs. scorer AUROC; geometry vs. probe AUROC) |
| C3 | Paired t-test or Wilcoxon signed-rank (normality-dependent, decided at analysis time) |
| C4 | Interaction regression / mixed-effects model, type × geometry, seed as grouping factor |
| C5 | One-sample test vs. chance (0.5); paired test, probe AUROC vs. scorer AUROC |

---

## 9. Multiple-comparison policy

Source: `22_Experimental_Design.md` §6, flagged from `11_Confound_Map.md` (Medium-High risk).

Holm-Bonferroni correction (or equivalent) applied across the joint set of scorer × geometry-metric
× arm comparisons before any individual correlation is treated as confirmatory. Both corrected and
uncorrected values are reported for transparency — the corrected value governs the
supports/contradicts determination in §11–12; the uncorrected value is disclosed alongside it, not
substituted for it.

---

## 10. Inclusion / exclusion criteria

Source: `22_Experimental_Design.md` §2, §3.5, §5.

- **Datasets:** ISIC 2019 + PAD-UFES-20 only (Papers 1–3's existing acquisition-shift testbed).
  Gallo et al.'s dataset choices are not adopted (different shift framing — see
  `22_Experimental_Design.md` §2).
- **Splits:** held-out evaluation splits matching Papers 1–3's protocol, same split seeds where
  feasible.
- **Seeds:** 5 seeds per condition (Paper 1's multi-seed protocol), all included unless a
  individual seed fails Gate 0 (training did not converge for that seed), in which case that seed
  is excluded and the exclusion is logged per §13.
- **Arms/checkpoints:** included in C1–C5 analyses only if the applicable gate(s) in §7 are passed.
  A gate failure excludes that specific arm/checkpoint from the relevant analysis; it does not
  exclude the rest of the design.
- **Backbone:** PanDerm as primary target; CLIP ViT-B/16 as the pre-specified fallback (§13).
- **Excluded by design, not by post-hoc decision:** Prediction 3 / architecture comparison, full
  fine-tuning of the FM backbone, any dataset beyond ISIC 2019/PAD-UFES-20, any FM backbone beyond
  the primary + fallback pair — all per `22_Experimental_Design.md` §9.

---

## 11. Success criteria

Source: `22_Experimental_Design.md` §7 ("Supports the hypothesis").

- C2: geometry metrics (not intervention-strength label, not probe accuracy) show significant,
  directionally-consistent correlation with scorer degradation, replicating C1's retrospective
  result on the FM.
- C3: normalization-matched geometry groups show statistically indistinguishable degradation
  despite differing nominal intervention strength.
- C4: the intervention-type effect disappears after conditioning on geometry — supports full
  mediation.
- C5: probe recovers domain information at well-above-chance AUROC while distance scorers remain
  flat/near-chance, on at least one Intervention-arm strength that has passed Gates 0–1.

---

## 12. Failure interpretation

Source: `22_Experimental_Design.md` §7 ("Contradicts" and "Inconclusive").

**Contradicts the hypothesis (publishable as a real result):**
- C3 fails (degradation persists despite matched geometry) — implicates something
  GRL/orthogonality-specific rather than general geometry.
- C4's intervention-type effect remains significant after conditioning on geometry — evidence
  against full mediation; consistent with partial mediation or an intervention-type-specific
  mechanism the measured geometry metrics don't capture.
- C5 fails to replicate the CNN dissociation — the "decodable but not accessible" framing needs
  revision on FMs, not just its explanation.

**Inconclusive (not evidence for or against the hypothesis):**
- Gate 0 failure on any arm — re-run with adjusted budget before drawing any conclusion from that
  arm.
- Gate 1 failure on any checkpoint — per §7's quoted sentence, that checkpoint tests nothing and is
  excluded from C2–C4, not counted as a null result.
- Reference arm shows no baseline domain leakage — per `16_Minimal_Sufficient_Experiment.md` Risk
  Register item #3, nothing exists for the Intervention arm to act on; requires a backbone or
  layer fallback before C2–C5 can be evaluated at all.

---

## 13. Protocol Deviations

New section, requested by Vinh, not present verbatim elsewhere — but every listed trigger is
already established policy in `22_Experimental_Design.md` or `22a_Design_Amendments.md`; this
section only collects them and adds the logging requirement.

**The 24-Hour Rule (`25_Research_Constitution.md`, added 2026-08-04):** if a result is surprising,
no hypothesis, protocol, or claim edit for 24 hours. Only bug/implementation checks and a
`26_Lab_Notebook.md` entry are permitted in that window. The deviation question below is answered
after the 24 hours, not during it.

**Pre-specified deviation triggers and responses:**

| Trigger | Response |
|---|---|
| PanDerm checkpoint/license unavailable | Fall back to CLIP ViT-B/16 (`22_Experimental_Design.md` §2). Log which backbone was actually used; flag explicitly in the eventual paper if the fallback is used, since it weakens the "medical-FM-specific" framing. |
| A checkpoint fails Gate 1 (manipulation not achieved) | Stop interpreting that checkpoint's causal/correlational results; exclude it from C2–C4 per §12. Do not report it as a null result. |
| An arm fails Gate 0 (implementation not sound) | Stop that arm's experiment; re-run with adjusted budget before any result from it is used anywhere in §11–12. |
| Any change to datasets, arms, measurements, or statistical tests not listed in this document | Document the change (what, why, when) in a running Deviations Log appended to this section; analyze it separately from the pre-registered analyses in §4–5, clearly labeled exploratory/post-hoc, never merged into the confirmatory results. |
| Any proposal to reintroduce Prediction 3 or expand scope (new datasets, new backbones, new arms) | Treated as a new protocol amendment, not a deviation — requires the same review path this protocol went through (`22a_Design_Amendments.md`-style amendment, reviewed before merge), not a silent edit here. |

**Deviations Log (empty at freeze time — 2026-08-04):**

*(No deviations recorded yet. Entries added here, dated, as they occur during execution. This log
is append-only — earlier entries are not edited or removed once execution begins.)*

---

## Status

Protocol frozen 2026-08-04 (session 9). `22_Experimental_Design.md` (v2), `22a_Design_Amendments.md`,
`22b_Evidence_Ladder.md`, and `23_Experiment_Contract.md` are treated as closed inputs to this
document, not further amended except through §13's deviation path. Per Vinh's instruction, no
hypothesis, prediction, outcome, or success/failure criterion changes from this point forward
without extraordinary evidence, logged as a deviation, not a silent edit.

**Next step is implementation (code)** — the first time in this project's Step 1–6 discipline
that code is in scope, per Vinh's original workflow (coding forbidden before Step 6, and Step 6's
sub-stages — Design → Contract → Amendments → Pre-Registration — are now all complete).

---

## Deviation log

### D-1 (2026-09-02) — gamma ladder extended to small negative values

**What changed.** The registered gamma ladder is positive-only. The fine grid
run for the Stage 2 conditional item uses gamma in {-.006, -.005, -.004, -.003},
which sharpens the covariance spectrum rather than whitening it.

**Trigger.** The advisor's plan makes a fine gamma grid conditional on the
current configurations lacking common support. That condition is met and was
measured, not assumed: per identity, the fraction of configurations with a
cross-identity neighbour inside the caliper is .00 for Adaptation at caliper
.15 and .20 at the preregistered .25, against .40/.30 and .65/.55 for
Conventional and Intervention.

**Why the registered ladder could not satisfy it.** Adaptation sits at high
standardised spectral slope, and positive gamma lowers slope -- the manuscript's
own dose-response records slope falling by 1.5 to 3.6e-3 across the positive
ladder. So refinement within the registered range moves away from the region
needing coverage. A pilot over gamma in {0, .005, .01, .02, .05, .1, .15, .25}
confirmed this before the extension was chosen: the closest reachable
configuration sits at distance .95 from any Adaptation configuration, against a
caliper of .25, and adding tempering made it worse rather than better
(Conventional alpha=.75 moves from .95 at gamma=0 to 2.05 at gamma=.005).

**Why the extension is inside the theory.** The bound the manuscript uses is
kappa_2(A_gamma) = kappa_2(Sigma)^(|gamma|/2), which is stated on the absolute
value. Negative gamma is a member of the same operator family, not a new one.

**How the extension was chosen.** On geometry alone. The pilot and the grid
script compute LID and spectral slope and nothing else; neither has access to
any scorer or to AUROC, and neither imports a module that does. The claim that
selection was blind to the outcome is therefore checkable from the code rather
than taken on trust. The grid range was fixed from the pilot before any
configuration in it was scored.

**Effect on claims.** The augmented configurations enter the matched analysis
only. Nothing in RQ1, RQ2, or the external-domain results depends on them. The
matched analysis will be reported both with and without them, so a reader can
see what the extension bought.

### D-2 (2026-09-09) — seed set extended from five to ten

**What changed.** The registered seed set is {42, 52, 62, 72, 82}. Five more
were trained: {92, 102, 112, 122, 132}.

**Why.** Not to gain power for a result, but to remove a constraint this study
itself identified. A whole-seed permutation over k seeds attains no p below
1/k!. At five seeds that floor is 1/120 = .0083, above Holm's first threshold
for any family larger than six tests, so two families here could not have
produced a corrected rejection whatever the data showed. At ten seeds the floor
is 1/10! = 2.8e-7 and the constraint is gone. Identifying a design defect and
then remaining inside it is a weaker position than fixing it.

**How the seeds were chosen.** By continuing the registered arithmetic rule,
step 10, rather than by selection. No seed was inspected before being added and
none was discarded. Every analysis is rerun on all ten; no result is reported
on a chosen subset.

**Hyperparameters unchanged.** r=16, lambda_proj=0.1, 100 epochs, lr 1e-3,
batch 512 — the canonical values, not re-selected for the new seeds.

**Verification before writing anything.** The extension script retrains seed 42
and asserts the result is bit-identical to the stored adapter. It is, to
0.000e+00 on all four tensors, so the training path is the registered one and
not a re-implementation of it.

**Artifact separation.** The canonical arm directories are chmod-frozen. They
were left frozen; the added seeds are written to parallel `*Arm_Extended`
directories and the loader resolves canonical first, so which seeds are
registered and which were added stays visible in the layout rather than being
merged away.

**Reporting.** Both the five-seed and ten-seed results are reported, so the
effect of the extension on every conclusion is visible rather than assumed.
