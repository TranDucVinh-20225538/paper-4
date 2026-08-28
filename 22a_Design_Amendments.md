# Design Amendments — Paper 4 (delta on 22_Experimental_Design.md)

**Status: MERGED into `22_Experimental_Design.md` (2026-08-04, session 8).** This document remains
as the historical record of what was proposed, what Vinh corrected before approving, and why —
kept separate from the merged result so the design history stays legible: what the original Step 6
spec said, what `23_Experiment_Contract.md`'s audit found, what was proposed here, and what
language Vinh corrected before merge. See `22_Experimental_Design.md` §3.5 and §7 for the merged
text, and `22b_Evidence_Ladder.md` for the Evidence Ladder, which was spun out to its own document
rather than merged (see note at the bottom of this file).

Three amendments, all logic fixes — no new experiments, no added workload. **Two language
corrections were made by Vinh before merge; both are reflected below and in the merged text.**

---

## Amendment 1 — Gate 0: Implementation Integrity Gate

**Naming correction (Vinh, before merge):** originally proposed as "Implementation Validity."
Renamed to **Implementation Integrity Gate** — "validity" reads in a paper as internal or
construct validity, which is not what this gate checks. This gate only checks whether the
system is reliable enough to enter hypothesis testing at all (training ran correctly,
representation didn't collapse, baseline reproduces), which "integrity" names more precisely.

Replaces the claim-less "ID task performance" measurement (`22_Experimental_Design.md` §4, item
4) with an explicit gate that runs before any of C1–C5 are interpreted.

```
Gate 0 — Implementation Integrity Gate

Representation not collapsed
        ↓
Training converged
        ↓
Baseline (Reference arm) reproduces the expected shortcut-leakage pattern
        ↓
PASS → results from this arm are eligible to be interpreted as evidence about the hypothesis
FAIL → results from this arm are inconclusive, not falsifying — re-run with adjusted budget
        before drawing any conclusion
```

Applies per arm (Reference, Intervention, Conventional, Adaptation), independently — one arm
failing Gate 0 does not invalidate the others. Measured via: balanced accuracy and calibration/ECE
well above a trivial/majority-class baseline (the original measurement, now correctly scoped as a
sanity check rather than a hypothesis test); training-loss convergence; and, for the Reference arm
specifically, domain-probe leakage at a magnitude consistent with Paper 1's CNN baseline (per
`23_Experiment_Contract.md` §6's capacity-sanity recommendation).

**Why this matters:** without Gate 0, a combination like "Intervention fails, Adaptation fails,
only Conventional shows a pattern" (the exact case `23_Experiment_Contract.md` §5 flagged) is
undiagnosable — there is no way to tell "the hypothesis is false" from "the adapter budget was too
small to train anything meaningful." Gate 0 makes that distinction checkable before any downstream
claim is interpreted.

---

## Amendment 2 — Gate 1: Manipulation Check

New gate, positioned between the Intervention/Conventional/Adaptation arms and the geometry/scorer
analysis in Stage 3 (`22_Experimental_Design.md` §1, §6).

```
Gate 1 — Manipulation Check

Intervention applied
        ↓
Did LID and/or spectral-decay slope actually change (not just condition number/scale)?
        ↓
YES → proceed to interpret accessibility/reliability changes as evidence about
      the geometry-mediation chain
NO  → the chain is not testable on this checkpoint — report as
      "claim not testable," not as "hypothesis contradicted"
```

**Required sentence (Vinh's addition, to be quoted verbatim wherever Gate 1 is discussed in the
eventual paper):**

> "Failure at Gate 1 does not constitute evidence against the hypothesis; it indicates that the
> intended manipulation was not achieved and therefore the causal chain cannot be evaluated."

This is the line to cite if a reviewer asks "the intervention didn't change geometry — doesn't
that mean the hypothesis is wrong?" It preempts that question directly rather than requiring a
new argument to be constructed under review pressure.

This directly resolves `23_Experiment_Contract.md` §4's highest-severity finding: ℓ2-normalization
and adapter training may move condition number/scale without moving the specific geometry
quantities (LID, spectral-decay) the hypothesis is about. Gate 1 requires confirming the mediator
variable actually moved before any claim about its downstream effects is evaluated — this applies
to both the main Intervention ladder (checking the intervention itself induced a geometry shift)
and the normalization-knob manipulation used for C3 (checking the knob moved the right quantities,
not just scale).

**The distinction Vinh added and that this amendment preserves explicitly:** a Gate 1 failure
means *the hypothesis was not tested*, not that it was falsified. These are different outcomes
with different reporting obligations — the first calls for a different manipulation or a larger
budget; the second is a real scientific result. Conflating them would let an untested checkpoint
masquerade as a null result, which is exactly the kind of error this project's evidence-discipline
(Level A/B/C hierarchy, `07_Evidence_Register.md`) exists to prevent in a different form.

---

## Amendment 3 — C4 success criterion (mediation framing)

Adds the missing "supports" outcome to `22_Experimental_Design.md` §7, which currently states only
the "contradicts" side of C4 (a significant type×geometry interaction) and never states what a
null interaction means.

```
If the intervention-type effect (Intervention vs. Conventional, at matched budget)
disappears after conditioning on geometry
        ↓
→ supports the mediation hypothesis: geometry, not intervention type, explains
  the accessibility outcome
```

**Losing-outcome branch added (Vinh's correction):** a mediation hypothesis needs a stated
criterion for losing, not just for winning — otherwise any result short of the ideal one has no
defined interpretation.

```
If the intervention-type effect remains significant after conditioning on geometry
        ↓
→ evidence against full mediation
        ↓
→ consistent with either:
     • partial mediation (geometry explains some, not all, of the intervention-type effect), or
     • an alternative mechanism specific to intervention type, not captured by the
       geometry metrics measured
```

**Scope note, to prevent this from reintroducing Prediction 3:** this is about intervention
*type* (Intervention vs. Conventional/Adaptation, all on the same FM backbone), not architecture
(CNN vs. FM). Prediction 3 (architecture-agnosticism) stays dropped per
`21_Prediction3_Removed_Reevaluation.md`. This amendment is the FM-internal, smaller-scope analogue
of the same mediation logic — testing whether geometry mediates *within* one architecture across
intervention types, which is squarely inside the locked RQ, not a reopening of the claim that was
deliberately cut.

---

## Evidence Ladder — spun out to `22b_Evidence_Ladder.md`

Vinh's addition, not present in either `22_Experimental_Design.md` or `23_Experiment_Contract.md`.
**Before merge, this was moved out of the Methods lineage entirely**, per Vinh's observation that
it describes how results are *interpreted* (an epistemic framework), not how the experiment is
*run* (Methods). It now lives in `22b_Evidence_Ladder.md` as a standalone appendix/supplement, not
merged into `22_Experimental_Design.md`.

Two of its level names were also corrected before finalizing: "Alternative explanations excluded"
→ **"Competing explanations weakened"**, and "Mechanism supported" → **"Mechanistic support"** —
both changes soften an absolute-sounding claim ("excluded," "supported") to the graded, defensible
claim the evidence can actually deliver. See `22b_Evidence_Ladder.md` for the full ladder and the
reasoning behind the rename.

---

## On shared assumptions (§3 of the Contract) — reframed, not removed

Vinh's correction to `23_Experiment_Contract.md`, recorded here since it changes how that section
should be read going forward: the three shared assumptions identified there (adapter-based
training as a proxy for the tested interventions; FM embedding dimensionality/geometry-estimator
reliability; a single fixed train/eval split) are not defects to be engineered away — under a
1-GPU, single-backbone, 6-week budget, some coupling across claims is unavoidable. Trying to make
every claim fully independent would require resources this project doesn't have and isn't the
right fix.

The correct response, already partially built into Amendments 1 and 2, is **acknowledged
dependency**: Gate 0 and Gate 1 don't eliminate the shared assumptions, but they make the point at
which a shared-assumption failure would show up *checkable and visible*, rather than silently
producing an uninterpretable null result across multiple claims at once. That is the appropriate
scope of mitigation for a resource-constrained single-backbone design — full independence was
never the goal.

---

## Status

**MERGED (2026-08-04, session 8).** Three amendments merged into `22_Experimental_Design.md`:
Gate 0 (renamed Implementation Integrity Gate), Gate 1 (with the required protective sentence
added), and the C4 success criterion (with the losing-outcome branch added). The Evidence Ladder
was not merged — spun out to `22b_Evidence_Ladder.md` as a standalone appendix. The shared-
assumptions reframing note stands as project record but required no edit to
`22_Experimental_Design.md` itself (it changes how §3 of `23_Experiment_Contract.md` should be
read, not the design). See `22_Experimental_Design.md` for the merged result ("v2").
