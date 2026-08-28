# Stage 4 — Execution Plan

**Documentation only. No code, no commands, no shell.** `Stage4_Architecture_Review.md` answers
*what* Stage 4 produces; this document answers *in what order* — so that once training starts, a
missing gate or missing artifact is caught by reading this plan, not by discovering it mid-run.
Grounded in `22_Experimental_Design.md` §3, §3.5, §5, §8 — no new design content, only sequencing.

## Status-at-a-glance table

Per Vinh's refinement (2026-08-05): while a run is in progress, this table alone should answer
"which step is this." Updated as each artifact type is produced — not a log, just current position.

| Step | Artifact | Gate | Review chain | Freeze |
|---|---|---|---|---|
| 4.1 | Checkpoint (adapter weights) | Gate 0 (§1/§2 Step 5) | — | — (frozen together with 4.2/4.3 at Step 8/7) |
| 4.2 | Embedding (held-out + train-fitting) | Gate 1, Intervention/Conventional only (§1 Step 6) | — | ✓ (§1 Step 8 / §2 Step 7) |
| 4.3 | Measurement suite (geometry, scorers, probe) | — | Execution Contract → Execution → Review → Sanity | ✓ |
| 4.4 | Normalization-knob manipulation (C3 only) | Gate 1 on manipulated version (§3 Step 4) | Execution Contract → Execution → Review → Sanity | ✓ |

## 0. Shared prerequisite (once, before any arm starts)

1. Spot-check a few Reference Arm artifacts' SHA256 against their frozen manifests — confirm the
   fixed anchor (§2 of `Stage4_Architecture_Review.md`) is intact before anything is built on it.
2. Load the Reference Instrument checkpoint (PanDerm ViT-L/16) and the already-frozen preprocessing
   / dataset-split configuration — same instrument, same split, no re-derivation.

## 1. Per-arm execution order — Intervention and Conventional (same pattern; both get Gate 1)

For each strength (Intervention: 2–3 strengths; Conventional: 1 matched-budget point) × each of 5
seeds (§5, matching Paper 1's protocol):

```
Step 1   Load Reference Instrument checkpoint (frozen, unmodified)
    ↓
Step 2   Train adapter checkpoint — this arm, this strength, this seed
    ↓
Step 3   Extract embeddings — held-out eval split; train-split fitting population too, if a
         Group-B-style scorer needs one (EA-01's precedent)
    ↓
Step 4   Compute domain-leakage probe + ID task performance (feeds Gate 0)
    ↓
Step 5   GATE 0 — this arm × strength, aggregated across seeds
              FAIL → inconclusive for this arm/strength; do not proceed to Gate 1 or
                     interpretation; re-run with adjusted budget before drawing any conclusion
              PASS → continue
    ↓
Step 6   GATE 1 — this checkpoint: did LID and/or spectral-decay slope actually change?
              FAIL → checkpoint is still frozen (the failure itself is provenance); flagged
                     not-usable for C3/C4; reported as "claim not testable," never as evidence
                     against the hypothesis
              PASS → continue
    ↓
Step 7   Compute full measurement suite — geometry (incl. per-direction variance, computable
         for the first time here), distance-based reliability scorers, probe-weight diffuseness.
         Tier 1, single-arm, no cross-arm language.
    ↓
Step 8   Execution Contract → Execution → Execution Review → Execution Sanity → Freeze
         (checkpoint + embedding artifact + measurement-suite output, this arm × strength × seed)
    ↓
Step 9   Next seed → Step 1. After all 5 seeds for this strength: next strength → Step 1.
```

## 2. Adaptation arm execution order (same pattern, no Gate 1)

Gate 1 is scoped to Intervention and Conventional checkpoints only (`22_Experimental_Design.md`
§3.5) — not applied here.

For each rung (progressive fine-tuning-intensity ladder) × each of 5 seeds:

```
Step 1   Load Reference Instrument checkpoint
    ↓
Step 2   Train adapter checkpoint — this rung, this seed
    ↓
Step 3   Extract embeddings
    ↓
Step 4   Compute domain-leakage probe + ID task performance
    ↓
Step 5   GATE 0 — this rung, aggregated across seeds
              FAIL → inconclusive for this rung
              PASS → continue
    ↓
Step 6   Compute full measurement suite (no Gate 1 for this arm)
    ↓
Step 7   Execution Contract → Execution → Execution Review → Execution Sanity → Freeze
    ↓
Step 8   Next seed → Step 1. After all 5 seeds for this rung: next rung → Step 1.
```

## 3. C3's normalization-knob manipulation (Intervention arm only, after §1 yields at least one frozen strength)

```
Step 1   Take a frozen Intervention-arm checkpoint's embeddings (already passed Gate 0 + Gate 1)
    ↓
Step 2   Apply the β-scaled ℓ2-normalization knob, post hoc, at a fixed intervention strength
         (reused from arXiv:2510.15202, not a new method)
    ↓
Step 3   Re-compute geometry metrics + distance-based reliability scorers on the normalized version
    ↓
Step 4   GATE 1 on the manipulated version — did the knob actually hold geometry constant as
         intended?
              FAIL → manipulation didn't work as intended; not usable for C3
              PASS → continue
    ↓
Step 5   Execution Contract → Execution → Execution Review → Execution Sanity → Freeze
         (this manipulation's output — a separate artifact from the unmanipulated checkpoint's own)
```

## 4. Cross-arm scheduling

Per `22_Experimental_Design.md` §8: Intervention, Conventional, and Adaptation may run **in
parallel** where GPU scheduling allows — nothing above requires one arm to finish before another
starts. What must stay in order is each arm's own strength/rung × seed sequence (§1/§2) and the
gate-before-freeze chain within it: a checkpoint is never frozen before its applicable gate(s)
resolve, and the §3 manipulation is never attempted on a checkpoint that hasn't itself passed Gate 0
and Gate 1 first.

## 5. Readiness gate before Stage 5

Stage 5 (Comparative Analysis) does not begin until: every arm has at least one strength/rung with
Gate 0 PASS, a Reference-Arm-equivalent synthesis document exists per arm (mirroring
`03d_stage3_5_baseline_synthesis.md`), and Vinh has reviewed and frozen the last Stage 4 artifact —
same Review → Sanity → Freeze discipline the Reference Arm used throughout, not shortened just
because the infrastructure now makes it easy to skip.

## What this document does not do

No code. No command. No shell. No script name, no file path for anything not yet built, no exact
strength/rung count (an execution-level decision, out of scope here per
`Stage4_Architecture_Review.md`'s own closing note). Does not authorize training or GPU execution —
that remains Vinh's separate, explicit decision, given only after this plan itself is reviewed.
