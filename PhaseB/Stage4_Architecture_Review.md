# Stage 4 — Architecture Review

**Planning only. No code, no experiment, no implementation, no training, no GPU computation
authorized by this document.** Written per Vinh's PI decision (2026-08-05): before any Intervention
Arm resource is spent, answer six structural questions so Stage 4's shape is agreed before it costs
anything. Grounded in `22_Experimental_Design.md` (§1, §3, §3.5, §4, §5), `17_Dependency_Graph.md`,
and `03d_stage3_5_baseline_synthesis.md` — nothing below introduces new design content; it restates
and organizes what's already locked in Phase A's protocol, applied specifically to what Stage 4
needs to produce.

## 0. Flow overview

```
Reference Arm  (Stage 3, CLOSED)
      │
      │   Gate 0  →  Freeze
      ▼
Intervention Arm  (+ Conventional, + Adaptation)   (Stage 4 — this document)
      │
      │   Gate 0  →  Gate 1  →  Freeze
      ▼
Comparison Layer   (Stage 5, not yet authorized)
      │
      │   Cross-arm statistical tests (§4 below)
      ▼
Claims C2 – C5
```

One diagram, one read: Reference Arm feeds Intervention Arm as its fixed anchor (§2); Intervention
Arm adds exactly one gate the Reference Arm never needed (Gate 1, §3); nothing compares arms until
the Comparison Layer (§4); no claim is touched before that. Everything below expands one box or one
arrow of this diagram — nothing below adds a new box.

## 1. What artifacts will the Intervention Arm produce?

Per `22_Experimental_Design.md` §3, Stage 4 covers the three arms not yet run (Reference is done —
Stage 3): **Intervention** (GRL+orthogonality adapter, 2–3 strengths mirroring Paper 3's
λ_orth = 0, 1, 5), **Conventional** (matched-budget ordinary fine-tuning, one control point), and
**Adaptation** (progressive fine-tuning-intensity ladder). Mirroring the Reference Arm's own
structure, each arm × strength/rung × seed (5 seeds, §5) is expected to produce:

- **A trained checkpoint** (adapter weights).
- **An embedding artifact** analogous to `ReferenceEmbedding`/`ReferenceTrainEmbedding` — held-out
  eval split (ID+OOD), and, if a scorer needs a fitting population (per EA-01's precedent), a
  train-split embedding artifact.
- **A Gate 0 verdict** (§3.5), reusing the same measurement suite already established for the
  Reference Arm.
- **A Gate 1 verdict** (§3.5) — new to Stage 4, per Intervention/Conventional checkpoint and per
  normalization-knob manipulation.
- **A measurement-suite output** — the same five items Stage 3.2 already produced for the Reference
  Arm (domain-leakage probe, geometry metrics, distance-based reliability scorers, ID task
  performance, probe-weight diffuseness) — **plus per-direction variance of the intervened
  direction**, computable for the first time here, since Stage 4 is where an "intervened direction"
  first exists (it had no resolved definition on the Reference arm alone — `03d_stage3_5_baseline_synthesis.md` §8).

## 2. Dependency on Reference Arm

Direct, not soft:

- **Fixed representation dimensionality** (§5) — every Stage 4 checkpoint must match the Reference
  Arm's embedding dimensionality (1024, PanDerm ViT-L/16) exactly, or the distance-concentration
  confound (`08_Mechanism_Map.md` Mechanism 4) reappears.
- **Matched train/eval splits** (§5) — reuses the exact split already validated in Stage 1.3a and
  frozen into `ReferenceEmbedding`/`ReferenceTrainEmbedding` (`random_state=42`,
  `isic_test_fraction=0.2`, `val_fraction=0.2`). No new split logic.
- **FM-baseline-leakage prerequisite** (`17_Dependency_Graph.md`, node "FM baseline probe shows
  leakage") — Conclusion C (geometry mediates accessibility within the FM setting) requires the
  Reference Arm's own probe to show leakage exists before any intervention. **Already satisfied**:
  Gate 0 criterion 1 PASSED (probe accuracy 0.9990). Stage 4 builds on this; it does not
  re-establish it.
- **Reference Arm as the zero-intervention anchor** — every Stage 5 comparison (§4 below) needs a
  fixed point to compare against; that point is the frozen Reference Arm baseline
  (`03d_stage3_5_baseline_synthesis.md`), cited as one document, not individual artifact files.
- **Reused, not re-derived**: `assets/panderm_vitl16_checkpoint_metadata.json`,
  `assets/reference_preprocessing.json`, CSG-SKin's dataset loader/split logic — same Reference
  Instrument, same preprocessing, already validated. Stage 4 does not re-validate these.

## 3. How do Gate 0 / Gate 1 reapply?

- **Gate 0** (Implementation Integrity Gate) reruns **per arm**, independently — Intervention,
  Conventional, and Adaptation each get their own verdict, same measured quantities as the
  Reference Arm (training convergence, representation-collapse check, ID task balanced
  accuracy/ECE above trivial baseline). A FAIL on one arm makes that arm's results inconclusive,
  not falsifying, and does not block the other arms.
- **Gate 1** (Manipulation Check Gate) is **new to Stage 4** — the Reference Arm has no
  intervention, so Gate 1 never applied to it. Runs on every Intervention-arm and Conventional-arm
  checkpoint, and on every normalization-knob manipulation used for C3: did LID and/or
  spectral-decay slope actually change? A Gate 1 FAIL removes that checkpoint from the C3/C4
  analysis — reported as "claim not testable on this checkpoint," never as evidence against the
  hypothesis (§3.5's protective sentence, quoted verbatim wherever a Gate 1 failure is discussed).
- Same discipline as the Reference Arm: no claim (C2–C5) is evaluated on an arm/checkpoint that
  hasn't passed its applicable gate(s).

## 4. Which artifact compares to which?

No comparison happens inside Stage 4 itself — it produces single-arm measurements only, exactly
like Stage 3.2 did for the Reference Arm (Tier 1/2, no cross-arm language). All cross-arm
comparison is Stage 5's job. Planned now so Stage 4's outputs are shaped to support it:

| Comparison | Arms | Claim | Statistic |
|---|---|---|---|
| Geometry vs. scorer-AUROC correlation | Intervention + Adaptation (jointly) | C2 | Kendall's τ |
| Geometry vs. probe-AUROC correlation (expected weak) | Intervention + Adaptation (jointly) | C2 (companion test) | Kendall's τ |
| Normalized-geometry-matched vs. nominal-strength-matched degradation | Intervention (± normalization knob) | C3 | Paired test (t-test or Wilcoxon, checked at analysis time) |
| Intervention type × geometry interaction | Intervention vs. Conventional, matched budget | C4 | Interaction regression, seed as random/grouping factor |
| Probe AUROC vs. scorer AUROC | Every arm/strength, against its own Reference-Arm-pattern baseline | C5 | One-sample test vs. chance + paired comparison |

None of this runs until Stage 5 is separately authorized.

## 5. Claim C2–C5 → output mapping

Restating `22_Experimental_Design.md` §1, scoped to what Stage 4 must produce so each claim has
something to be tested against later:

| Claim | Requires from Stage 4 | Not from Stage 4 (Stage 5's job) |
|---|---|---|
| **C2** | Geometry metrics + scorer AUROC, per Intervention-arm strength and Adaptation-arm rung | The correlation test itself |
| **C3** | Geometry metrics + scorer AUROC, per Intervention-arm strength, nominal and normalization-knob-adjusted | The paired significance test |
| **C4** | Geometry metrics + scorer AUROC, Intervention **and** Conventional arm, matched budget | The interaction regression |
| **C5** | Probe AUROC + scorer AUROC, every arm/strength | The significance tests |

C1 is already closed (Stage 0, retrospective, CNN data) — not reopened by Stage 4.

## 6. Does the Intervention Arm's freeze chain mirror the Reference Arm's?

**Yes, by default — same pattern, same Artifact Immutability Rule, applied per arm/strength/checkpoint
instead of once:**

```
Execution Contract  →  Execution  →  Execution Review  →  Execution Sanity  →  Freeze
```

Structural differences, not procedural ones:

- **More freeze cycles, not a different process** — one per arm × strength/rung (Intervention ×2–3,
  Conventional ×1, Adaptation ×N rungs), each independently gated and frozen, rather than the
  Reference Arm's single pass.
- **Gate 1 sits inside the chain**, between Execution and Execution Review, for
  Intervention/Conventional checkpoints. A checkpoint failing Gate 1 still gets frozen — the
  failure itself is provenance — but is flagged not-usable-for-C3/C4 in its own manifest, not
  silently dropped.
- **Successor-artifact versioning applies identically** — a Stage 4 artifact found wrong after
  freeze gets an Execution Amendment or a `_v2`, never a direct edit, per the Artifact Immutability
  Rule adopted at Reference Arm close.
- **No new document category needed** — the same five document types the Reference Arm used
  (execution contract, review, sanity, completion manifest, baseline synthesis) repeat, scaled to
  however many arm/strength combinations Stage 4 ends up running.

## What this document does not do

No code. No experiment. No implementation. No training run, no GPU allocation, no checkpoint. Does
not authorize computing anything for Intervention, Conventional, or Adaptation arms. Does not
resolve execution-level decisions this review surfaces but doesn't answer — exact adapter
architecture, exact intervention strengths/rungs, exact compute budget per arm — those are Stage 4
*execution* decisions, out of scope for an architecture-level review, to be resolved and logged
only once Stage 4 execution is separately authorized.
