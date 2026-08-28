# Stage 4 — Execution Plan v2

**Run order only. No scientific decision is made in this document** — every choice below is a
pointer to a document that already decided it. If a step here ever seems to require a new decision,
that's a sign this plan has drifted from its job; the fix is a new specification document, not an
inline choice here. Supersedes `Stage4_Execution_Plan.md` in sequence detail; that document's
cross-arm scheduling note (§4, arms may run in parallel) and Stage 5 readiness gate (§5) still
apply unchanged. No code. No training authorized by this document.

## Decisions this plan draws on, not re-litigates

| # | Decision | Locked in |
|---|---|---|
| 1 | Intervention family (single-backbone, output-stage adapter/projection) | `ArchitectureResolutions/AR-01_Resolution.md` |
| 2 | Intervention target, output space, geometry criterion, fixed scope | `Stage4_Architecture_Instantiation_Contract.md` |
| 3 | Nuisance mechanism (closed-form) + injection location (output-stage) | `Stage4_Intervention_Design_Choice.md` |
| 4 | Adapter requirement, nuisance estimator, projection equation, dose ladder, Gate 1 timing | `Stage4_Intervention_Method_Specification.md` |
| 5 | Adapter architecture, orthogonality loss formula, Conventional matching | `Stage4_Adapter_And_Control_Specification.md` |
| 6 | Bottleneck width / λ_proj selection procedures, nonlinearity | `Stage4_Hyperparameter_Selection_Rules.md` |

## Run order

```
1.  Load frozen PanDerm checkpoint + preprocessing (unchanged since Stage 1; verify against
    assets/panderm_vitl16_checkpoint_metadata.json before use)

2.  Compute nuisance direction w = μ_isic − μ_pad_ufes, unit-normalized
    (ReferenceTrainEmbedding + ReferenceEmbedding's PAD-UFES portion; closed-form, one time,
    shared across every arm below)

3.  Select bottleneck width r
    for r in {16, 32, 64, 128}:
        initialize adapter z' = z + W2·σ(W1·z), σ = GELU
        train Intervention-arm adapter at r, λ_proj = 0.1 (§4 below resolves λ_proj properly;
          this pass exists only to test r, minimum λ_proj from its own grid used as a fixed probe)
        Gate 0, Gate 1
        if both pass: stop, r is fixed for every arm below
        else: next r

4.  Select λ_proj (r now fixed from step 3)
    for λ_proj in {0.1, 1, 5}:
        train Intervention-arm adapter at fixed r, this λ_proj
        Gate 0, Gate 1
        if both pass: stop, λ_proj is fixed
        else: next λ_proj
    → this training run's adapter is the canonical Intervention-arm adapter (all seeds, §5 Controls)

5.  Train Conventional-arm adapter
    same r, same architecture, same optimizer/epochs/seeds as the canonical Intervention adapter
    (step 4) — loss = L_task only, L_orth structurally absent (not λ_proj = 0)

6.  Apply the α ladder to both arms' trained adapters
    z'(α) = z + α·Δz,  α ∈ {0, 0.25, 0.5, 0.75, 1}
    — five embedding extractions per arm per seed, one per α, from each arm's single trained adapter

7.  Per α, per arm: Gate 0
    domain-leakage probe + ID task performance, aggregated across seeds — same criteria as the
    Reference Arm's own Gate 0

8.  Per α, per arm (Intervention and Conventional only — not applicable to Adaptation, which has no
    Gate 1 per `22_Experimental_Design.md` §3.5): Gate 1
    did LID and/or spectral-decay slope actually move relative to this arm's own α = 0?
    FAIL → this α frozen anyway, flagged not-usable for C2–C5, not evidence against the hypothesis
    PASS → continue

9.  Per α that passed both gates: full measurement suite
    geometry (incl. per-direction variance of w), domain-leakage probe, distance-based reliability
    scorers — Tier 1 (Measurement), single-arm, no cross-arm language (same discipline as Stage 3.2)

10. Freeze
    Execution Contract → Execution → Execution Review → Execution Sanity → Freeze, per artifact
    (checkpoint, embeddings, measurement-suite output) — same chain as every prior Stage 3 artifact,
    Artifact Immutability Rule applies from the moment of freeze
```

Adaptation arm (progressive fine-tuning ladder) follows the same shape as steps 1, 5–7, 9–10 —
no adapter-width/λ_proj selection (it isn't the same mechanism), no Gate 1 (not in scope for this
arm) — detailed rung-by-rung sequencing is unchanged from `Stage4_Execution_Plan.md` §2.

## What this plan does not do

Does not choose an `r` or `λ_proj` value — only the procedure that will choose them, already fixed
in `Stage4_Hyperparameter_Selection_Rules.md`. Does not write any code. Does not authorize training —
that is a separate, explicit decision, to be given only after this plan is reviewed.

## Status

```
AR-01:                          COMPLETE (Step 1 + Step 2)
Stage 4 scientific design:      COMPLETE
  Family, mechanism, target, estimator, dose, adapter class,
  orthogonality formula, Conventional control, hyperparameter
  selection rules — all locked, each with its own decision record

Stage 4 execution plan v2:      COMPLETE (this document — run order only)

Training:                       NOT AUTHORIZED
```
