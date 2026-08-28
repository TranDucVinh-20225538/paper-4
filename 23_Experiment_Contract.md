# Experiment Contract — Paper 4

**Method note.** This document only reads `22_Experimental_Design.md` (and, where needed for
grounding, `02_Claim_Matrix.md`, `Scientific_Hypothesis.md`, `11_Confound_Map.md`,
`18_Hidden_Assumptions.md`, `19_Reviewer3_Logic_Attack.md`). It does not modify
`22_Experimental_Design.md`. Its job is to answer six audit questions and apply one rule
throughout:

```
No experiment exists only because it is interesting.
Every experiment must show: Claim → Prediction → Measurement → Inference.
If a link is missing, the experiment is flagged for removal or must be given one.
```

This is a genuine audit, not a formality — several real gaps were found. They are reported
honestly per project discipline, not smoothed over.

---

## 1. Claim coverage

Checked against the five "Unknown" rows in `02_Claim_Matrix.md`'s Claim Hierarchy table.

| Claim | Tested by | Direct or indirect | Sufficient? |
|---|---|---|---|
| C1 — which geometric quantities explain the trade-off | Stage 0, retrospective correlation on Paper 3's existing CNN checkpoints only | Direct, but **CNN-only** — does not by itself answer the question for the FM | Sufficient as a gate/prior, not as a final answer. C1 is only closed jointly with C2. |
| C2 — does degradation track geometry, not label/probe | Stage 3 correlation across Intervention-arm strengths + Adaptation-arm rungs | Direct | **Underpowered as specified** — see §3/§6 below. Only 2–3 Intervention-arm points plus however many Adaptation rungs survive to feed one correlation; this is a very small N for a correlation claim being asked to carry real evidential weight. |
| C3 — does holding geometry constant hold degradation constant (causal) | Stage 3 normalization-knob manipulation | Direct | **Not currently sufficient — see §4.** No manipulation check confirms the knob actually moves the geometry metrics the claim is about (LID/spectral-decay), as opposed to condition number/scale only. |
| C4 — does intervention type change the geometry–accessibility relationship | Interaction regression, Intervention vs. Conventional (+ Adaptation for range) | Direct | Sufficient in structure; **success criteria for this claim are incomplete** — see finding below. |
| C5 — does domain info stay decodable on FM after intervention | Linear probe on every FM checkpoint, Paper 1/3 protocol | Direct | Sufficient — this reuses an already-validated protocol at the same approximate dataset scale. |

**Coverage gap found:** Measurement #4 in `22_Experimental_Design.md` §4 ("ID task performance —
balanced accuracy, calibration/ECE") does not map to any of C1–C5. It is currently justified only
by analogy to Gallo's trade-off and to Paper 1's utility-cost finding — neither is a claim in the
Claim Matrix's Unknown section. Under the no-experiment-for-interest rule, this measurement should
either (a) be tied to an explicit claim, or (b) be reframed as a **manipulation/implementation
check** (did each arm's training actually work, as opposed to collapsing to a degenerate
representation) rather than a hypothesis test. Recommendation: (b) — see §5 and §7, where this
reframing turns out to close a real interpretability gap, not just satisfy a bookkeeping rule.

**No claim in the Claim Matrix is left without an experiment.** All five Unknown rows have at
least one direct test.

---

## 2. Experiment necessity

*"If this experiment is removed, which conclusions become impossible?"*

| Component | If removed | Removable? |
|---|---|---|
| Stage 0 (CNN retrospective) | C1 becomes entirely untested; no other component substitutes for it | **No** |
| Reference arm | Risk Register #3 gate lost; C5 loses its no-intervention comparison point (can't show decodability *changed*, only that it exists) | **No** |
| Intervention arm | C2, C3, C5 all become untestable — this is the arm that produces the geometry variation the whole chain is about | **No** |
| Conventional arm | C4 becomes untestable; the project reverts to the narrower, Reviewer-#3-criticized "disentanglement-specific" framing the relabeling was meant to move past | **No — required by the Gallo verification** |
| Adaptation arm (full ladder) | C4's *interaction* test (slope comparison, not just single-point comparison) becomes impossible; only a weaker one-point comparison would remain | **Reducible, not removable.** Already correctly flagged in §8 of `22_Experimental_Design.md` as the first thing to shrink under budget pressure — that's the right call, but "shrink" is different from "cut," and the document should say so explicitly if it ever gets cut to zero rungs. |
| Normalization-knob manipulation | C3 (the only causal, not correlational, test in the whole design) becomes entirely untestable | **No — single most load-bearing component for the causal claim** |
| Multi-scorer measurement (not just Mahalanobis) | The FM result would no longer rule out "Mahalanobis-specific assumption violation" as an alternative explanation — directly reopens Reviewer #3's Unstated 1 concern and Competing-Hypothesis Alt 1 | **No** |
| ID task performance (measurement #4) | No claim-linked conclusion is lost (it maps to no claim); but see §5 — an *interpretive* function would be lost | **Removable as a claim test. Recommend retaining, reframed as a manipulation check (§7), not removing outright.** |
| Probe-weight diffuseness | No primary conclusion becomes impossible — already explicitly labeled non-primary in `22_Experimental_Design.md` | **Removable without loss to any primary claim.** Cheap enough to keep; should stay explicitly optional. |

**Verdict:** every load-bearing component earns its place. The two components without a claim
attached (ID performance, probe-weight diffuseness) are both cheap, and one of them (ID
performance) turns out to serve a real function once correctly reframed — see §5, §7.

---

## 3. Independence — shared-assumption audit

*"Can two experiments fail together because they share the same assumption? If yes, identify it."*

Three shared assumptions were found, each capable of producing a correlated failure across
multiple claims simultaneously — which would look like "the hypothesis is false" when the real
cause is a single upstream assumption breaking:

**Shared assumption 1 — adapter-based training is a valid proxy for the interventions it stands
in for.** Intervention, Conventional, and Adaptation arms are *all* adapter-based (per
`22_Experimental_Design.md` §3, chosen for cost reasons). If lightweight adapters simply lack the
capacity to reshape geometry the way Paper 3's full dual-encoder training did, **C2, C3, C4, and
C5 could all fail together** for a reason that has nothing to do with whether the hypothesis is
true. This is already named in `18_Hidden_Assumptions.md` (Assumption 3) as the most
load-bearing, least-scrutinized assumption in the whole project — this audit confirms it is also
the single largest correlated-failure risk in the experimental design, since it sits upstream of
three of the five claims and two of the four arms. **Not currently mitigated by any check in
`22_Experimental_Design.md`.**

**Shared assumption 2 — the FM backbone's geometry is measurable in the same sense the CNN's
16-d representation was.** Reference, Intervention, Conventional, and Adaptation all sit on the
same FM backbone. If that backbone's embedding space (likely 512–1024+ dimensions, vs. Paper 3's
16) behaves in a way that makes LID/spectral-decay/condition-number estimation unreliable or
qualitatively different at that scale, **every FM-side claim (C2–C5) could fail or produce noise
together**, independent of the hypothesis. This connects directly to `11_Confound_Map.md`'s
already-flagged High-risk confound "embedding dimensionality mismatch," but that entry was written
as a general confound, not tied to a specific correlated-failure mechanism across C2–C5 — this
audit makes that connection explicit.

**Shared assumption 3 — a single fixed train/eval split is reused across all arms for
comparability.** This is a deliberate, documented control (`22_Experimental_Design.md` §5), but it
cuts both ways: if that particular split has any idiosyncrasy (an unusually easy or hard subset),
the idiosyncrasy is inherited identically by every arm's every measurement. Multi-seed training
controls for model-initialization noise, not for a split-level artifact. **Lower severity than
assumptions 1–2** (it would bias results consistently rather than fail catastrophically), but
worth a documented mitigation (e.g., a second split as a stability check on the primary C2/C3
findings only, not the full protocol).

---

## 4. Causal validity

Only one component makes an explicit causal (not correlational) claim: the normalization-knob
manipulation for C3/Prediction 2. Checking it against intervention/control/measurement/comparison:

| Element | Present? |
|---|---|
| Intervention | Yes — β-scaled ℓ2-normalization applied post hoc |
| Control | Yes — normalized-geometry-matched groups vs. nominal-strength-matched groups |
| Measurement | Yes — scorer AUROC degradation |
| Comparison | Yes — paired test between groupings |

**Structurally complete, but there is a mismatch worth flagging as the single most serious
technical gap in this design.** ℓ2-normalization directly affects vector scale and therefore
covariance-based metrics like condition number. It is not obvious that it moves **local intrinsic
dimensionality or within-class spectral-decay slope** — the specific mediator variables
Prediction 2 and Mechanism 1 (`08_Mechanism_Map.md`, arXiv:2510.15202) are actually about. LID
estimators in particular are often constructed to be scale-invariant. If the knob mostly moves
condition number while leaving LID/spectral-decay roughly fixed, **C3 would be testing "does
holding condition-number-scale constant hold degradation constant," not "does holding
LID/spectral-decay constant hold degradation constant"** — a different, narrower claim than the
one Prediction 2 states. This is not a reason to reject the design; it is a reason to add one
missing step.

**Required fix (see §7):** before interpreting the C3 comparison, verify — as a manipulation
check, not an analysis result — that the normalization knob measurably shifts LID and/or
spectral-decay slope specifically, not just condition number, in the checkpoints being compared.
If it doesn't, the knob needs to be replaced with a manipulation that acts on those quantities
directly (e.g., partial whitening/eigenvalue reshaping rather than uniform rescaling).

No other component in this design makes a causal claim; C2, C4, and C5 are all explicitly stated
as correlational/comparative in `22_Experimental_Design.md`, which is the correct scope for what
they can actually establish.

---

## 5. Negative-result robustness

Checked against `22_Experimental_Design.md` §7's existing three-way outcome structure
(supports / contradicts / inconclusive), plus the specific combinatorial scenario requested.

**Single-axis failures — already covered, still hold up:**
- C3 fails (degradation persists despite matched geometry) → publishable, implicates
  GRL/orthogonality-specifics. Holds.
- C4 shows a significant interaction → publishable, narrows "representation intervention" to
  something more intervention-specific. Holds.
- C5 fails to replicate the CNN dissociation → publishable, revises the "decodable but not
  accessible" framing itself. Holds.

**Gap found: C4's supporting outcome is never stated.** §7 of `22_Experimental_Design.md` lists
a significant type×geometry interaction under "Contradicts," but never states what a **non-significant**
interaction means. A non-significant interaction is actually the outcome that most directly
supports the relabeled hypothesis (representation intervention as a general category, not
disentanglement specifically) — and it is currently missing from the "Supports" list entirely.
This should be added: *"C4 shows no significant type×geometry interaction — Intervention and
Conventional arms follow the same geometry–accessibility relationship, supporting the general
'representation intervention' framing over a disentanglement-specific one."*

**The requested combinatorial case: Reference wins, Conventional wins, Intervention fails,
Adaptation fails.** Under this combination, the *only* arm producing a clean signal is Conventional
— the arm that was designed as a single-point control, not the primary evidence source. This
combination is genuinely ambiguous under the current design: nothing distinguishes "the hypothesis
is false" from "the adapter-based Intervention/Adaptation arms didn't train properly" (Shared
assumption 1, §3). **As currently specified, this combination would not be cleanly publishable
either way — it would be diagnostically stuck.** This is the concrete case that motivates
reframing measurement #4 (ID task performance) as an explicit manipulation/implementation-sanity
check (§7): if Intervention/Adaptation arms show reasonable ID task performance (i.e., they
trained successfully) while still failing to show the predicted geometry/accessibility pattern,
the "fails" result is interpretable as a genuine falsification. If ID performance is degenerate on
those arms, the "fails" result is uninterpretable and the arms need to be re-run with a larger
adapter budget before any conclusion is drawn. **Without this check, several failure combinations
in this design are not distinguishable from harness failure — this is the most actionable finding
in this contract.**

Other combinations checked briefly: Intervention succeeds + Conventional also succeeds identically
→ covered by the C4 fix above (now correctly publishable as support, not left unaddressed).
Intervention succeeds + Conventional fails to show any effect → clean support for both the general
chain and a type-specific boost, no ambiguity. Reference fails (no baseline leakage) → already
correctly handled in §7 as an inconclusive/underpowered case requiring a backbone/layer fallback,
independent of this audit.

---

## 6. Reviewer attack simulation

| Component | Strongest reviewer criticism | Survives as specified? | Single fix |
|---|---|---|---|
| Stage 0 (CNN retrospective) | "This is circular — the mechanism was selected because it fit this exact data." | Yes, already pre-empted (`17_Dependency_Graph.md` treats R0 as a gate, not a conclusion) | None needed; discipline must hold at write-up time (don't cite R0 as independent confirmation) |
| Intervention arm | "A lightweight adapter cannot be assumed equivalent to the full training Paper 3 used — any null result is confounded with adapter capacity." | No | Add a capacity-sanity check: confirm the Intervention arm at its strongest setting reduces domain-probe leakage by a magnitude comparable in scale to Papers 1/3's full-training result, before trusting downstream geometry comparisons |
| Conventional/Adaptation arms | "'Matched budget' (steps, parameter count) is not the same as matched capacity to affect geometry — different objectives can behave very differently at equal step count." | Partially | Optional strengthening: report a budget-equivalence diagnostic (e.g., gradient-norm/loss-curve comparison) alongside step/parameter matching |
| Geometry metrics at FM scale | "Off-the-shelf LID/spectral-decay estimators are known to be unstable in high dimensions and small samples (PAD-UFES-20 ≈ 2.3k images) — where is the estimator-stability check?" | No | Add a bootstrap/subsample stability diagnostic (e.g., confidence interval via resampling) for every LID/spectral-decay estimate, not just a point value |
| Normalization-knob (C3) | "Does this manipulation actually move LID/spectral-decay, or only condition number/scale?" | No — most serious finding in this contract | See §4's required fix: manipulation check before interpretation |
| Geometry-vs-AUROC correlation (C2/C4) | "With 2–3 Intervention strengths and one matched Conventional point, a Kendall's τ has almost no power — one checkpoint could flip the sign." | Partially | Either extend Adaptation-arm rungs specifically for this analysis (cheap, CPU-bound) or explicitly report wide confidence intervals and avoid over-interpreting a single τ point estimate |
| Probe-decodability (C5) | "Is a 5-seed linear probe on ~2.3k images stable enough to trust an AUROC point estimate?" | Yes | None needed — this exact protocol, including the shuffle-label control, is already validated at this dataset scale by Paper 1 |

---

## 7. Claim → Prediction → Measurement → Inference chain check

Applying the rule stated at the top of this document to every primary component:

| Component | Claim | Prediction | Measurement | Inference | Complete? |
|---|---|---|---|---|---|
| Stage 0 | C1 | LID×spectral-decay should correlate with existing Mahalanobis-AUROC | Kendall's τ on existing CNN checkpoints | Gate: proceed or reconsider mechanism before spending FM budget | ✅ |
| Intervention ladder + scorers + probe | C2, C5 | Prediction 1: degradation tracks geometry, not label/probe | Geometry metrics + 7 scorers + probe, per checkpoint | Correlation test; probe-vs-scorer dissociation test | ✅ (power caveat noted, §6) |
| Normalization knob | C3 | Prediction 2: holding geometry constant holds degradation constant | Paired scorer-AUROC comparison at matched geometry vs. matched nominal strength | Causal support/contradiction for geometry-as-mediator | ⚠️ **Incomplete without the manipulation check added in §4** |
| Conventional + Adaptation arms | C4 | (New) type×geometry interaction should be absent if "representation intervention" is a valid general category | Interaction regression | Supports/contradicts the relabeled, general hypothesis | ✅ once the missing "supports" criterion (§5) is added to `22_Experimental_Design.md` |
| ID task performance | *(none currently)* | *(none currently)* | Balanced accuracy / ECE per arm | *(currently unclear what this licenses)* | ❌ **as currently framed.** Recommended repair: reframe as **Claim 0 — "did each arm's training produce a non-degenerate representation?"**, Prediction: ID accuracy should be well above a trivial/majority-class baseline in every arm; Measurement: balanced accuracy/ECE; Inference: **gates interpretation of every other claim** — a failure here means "inconclusive, re-run," not "hypothesis contradicted." This is exactly the missing piece identified in §5's combinatorial analysis. |
| Probe-weight diffuseness | *(none — explicitly auxiliary)* | *(none)* | Weight-vector concentration | Tie-breaker only, for ambiguous C2/C4 results | Acceptable as explicitly non-primary; not held to the full chain requirement |

---

## Net verdict

**Structurally sound and claim-complete: every claim in the Claim Matrix has an experiment, every
experiment has a claim (once §1's one gap is repaired), and no load-bearing component is
removable.** This is not a rubber stamp, though — three findings should be resolved before
`24_PreRegistration.md` is written, in order of severity:

1. **(High)** The normalization-knob manipulation (C3) needs an explicit check that it moves
   LID/spectral-decay, not just condition number/scale, before its comparison can be interpreted
   as a test of Prediction 2 specifically. Without this, the design's only causal test may not be
   testing what it claims to test.
2. **(Medium-High)** Reframe ID task performance (currently claim-less) as an explicit
   implementation-sanity gate ("Claim 0"), checked before interpreting any Intervention/Conventional/
   Adaptation result as evidence about the hypothesis rather than about harness failure. This
   directly resolves the ambiguous combinatorial case in §5.
3. **(Medium)** Add the missing "supports" criterion for C4 (non-significant type×geometry
   interaction) to `22_Experimental_Design.md` §7, and disclose the small-N correlation power
   limitation for C2/C4 explicitly rather than leaving it implicit.

Lower-severity items (adapter-capacity sanity check for the Intervention arm, LID/spectral-decay
estimator stability diagnostic, split-sensitivity check) are worth adopting but do not block moving
forward — they can be folded into `24_PreRegistration.md` as pre-specified robustness checks rather
than requiring a revision to `22_Experimental_Design.md` itself.

This contract does not modify `22_Experimental_Design.md`. Recommended next step: Vinh reviews
this contract; if the three ordered findings above are accepted, the corresponding fixes should be
made to `22_Experimental_Design.md` before `24_PreRegistration.md` locks the statistical plan
around it.
