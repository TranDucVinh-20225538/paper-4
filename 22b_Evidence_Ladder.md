# Evidence Ladder — Paper 4 (Appendix / Supplement)

**Status: Standalone epistemic framework, not part of the experimental Methods.** Originally
proposed inside `22a_Design_Amendments.md`; spun out to its own document per Vinh's instruction
(2026-08-04, session 8) because it describes how results are *interpreted*, not how the experiment
is *run*. `22_Experimental_Design.md` (Methods) should not carry this — it belongs with any future
paper's Discussion or Supplementary Material.

---

## Purpose

Reviewers of the eventual paper will ask, implicitly or explicitly, "how far did this evidence
actually get?" A flat supported/not-supported verdict hides that answer. The ladder below orders
this project's evidence requirements so results can be reported at the level they actually reach —
including partial results — rather than collapsed into a binary claim.

```
Level 0 — Implementation Integrity
        ↓
Level 1 — Manipulation
        ↓
Level 2 — Prediction
        ↓
Level 3 — Competing explanations weakened
        ↓
Level 4 — Mechanistic support
```

## Levels

| Level | Requirement | Corresponds to (in `22_Experimental_Design.md`) |
|---|---|---|
| **0 — Implementation Integrity** | The Implementation Integrity Gate passes for the arm(s) in question: representation not collapsed, training converged, baseline reproduces the expected pattern | Gate 0 (§3.5) |
| **1 — Manipulation** | The Manipulation Check Gate passes: the intended intervention actually shifted LID/spectral-decay slope, not just condition number/scale | Gate 1 (§3.5) |
| **2 — Prediction** | C2/C3 patterns are observed: degradation tracks geometry; holding geometry constant holds degradation constant | §1, §6 |
| **3 — Competing explanations weakened** | Multi-scorer convergence weakens a Mahalanobis-specific-artifact explanation (Reviewer #3's Unstated 1 concern, Alt 1 in `13_Competing_Hypotheses.md`); probe-weight-diffuseness weakens simple entangled-encoding (Alt 2) | §4 measurements 2, 5 |
| **4 — Mechanistic support** | C4's mediation criterion is met: the intervention-type effect disappears after conditioning on geometry | §7, C4 |

## Why "weakened," not "excluded" — and "support," not "supported"

Earlier drafting used "alternative explanations excluded" and "mechanism supported." Vinh's
correction: experimental evidence in this design can weaken competing explanations and can support
a mechanism — it cannot exclude alternatives outright, and "supported" reads as a completed,
absolute verdict rather than a graded evidentiary state. The renamed levels ("weakened,"
"mechanistic support") are deliberately more modest claims, matched to what the design can actually
deliver, and less exposed to an easy reviewer correction.

## How to read a partial result

A result that reaches Level 2 but not Level 4 is still a complete, reportable, publishable finding
— for example: "the intervention produces the predicted dissociation (Level 2), and alternative
explanations are weakened (Level 3), but the effect does not survive conditioning on intervention
type (Level 4 not reached) — the mechanism appears intervention-specific rather than general." This
is one of the already-accepted outcomes in `22_Experimental_Design.md` §7. The ladder's role is to
make explicit, in the write-up, exactly which level was reached and why the climb stopped there —
rather than forcing every result into a single supported/contradicted binary.

A result that fails Level 0 or Level 1 has not climbed the ladder at all — per Gate 0 and Gate 1 in
`22_Experimental_Design.md` §3.5, this is reported as inconclusive/not-testable, not as evidence
against the hypothesis at any level.
