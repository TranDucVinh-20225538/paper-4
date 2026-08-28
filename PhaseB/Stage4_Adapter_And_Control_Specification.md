# Stage 4 — Adapter and Control Specification

**Locks the three items Vinh flagged as still open before `Stage4_Execution_Plan_v2.md` can be
written — its own decision record, deliberately kept separate from the execution-flow document, so
the execution plan only locks *run order*, never hides a scientific decision inside it.** No code.
No training authorized.

## 1. Adapter architecture

**Choice: a standard residual bottleneck adapter — `z' = z + W₂·σ(W₁·z)`** — not a bare linear
residual (`z' = z + W·z`).

**Reasoning.** This is the direct instantiation of what was already concluded, not a fresh choice:
`AR-01_Resolution.md`'s family conclusion named "bottleneck/feature adapter" specifically, and
`AR-01_Architecture_Study.md`'s own Tier 1 entries for that family (AdaptFormer's AdaptMLP,
CLIP-Adapter's feature adapter) are both down→up projection bottlenecks *with* a nonlinearity
between them — not linear-only. A purely linear adapter was considered and rejected: without a
nonlinearity, `A(z)` reduces to a single effective linear map, which risks collapsing the
capacity/mechanism distinction §0 of `Stage4_Intervention_Method_Specification.md` just
established — a linear-only adapter regularized toward orthogonality starts to look less like
"trainable capacity constrained by a target" and more like "another linear projection," blurring
exactly the line that document drew. Keeping `σ` (a standard nonlinearity — exact choice, e.g. GELU
vs. ReLU, is an execution-level detail) preserves genuine trainable capacity distinct from the
closed-form projection mechanism.

**Not locked here**: bottleneck width, exact nonlinearity, initialization scheme — execution-level.

## 2. Orthogonality loss, exact formula

**Choice: squared cosine similarity —**

```
L_orth(z', w) = ( z'·w / (‖z'‖‖w‖) )²  =  cos(z', w)²
```

**not** the absolute-value form `|cos(z', w)|`.

**Reasoning, two independent lines:**

1. **Direct precedent, not a fresh invention.** CSG-Lite's own orthogonality term (`csg_lightning.py`,
   read directly during the investigation that produced `Stage4_Intervention_Architecture_Gap.md`)
   is exactly this: `loss_orth = mean(cosine_similarity(z_l, z_c_proj, dim=1).pow(2))` — squared
   cosine similarity, batch-mean-reduced. Using the identical mathematical form here, applied to a
   different pair of vectors (`z'` vs. the closed-form domain direction `w`, instead of `z_lesion`
   vs. a projected `z_context`), is the most literal reading of `22_Experimental_Design.md`'s
   "mirror [GRL + orthogonality] as closely as an adapter allows" — the mechanism's exact math is
   preserved; only the architecture producing the two vectors being decorrelated differs (§0,
   `Stage4_Intervention_Method_Specification.md`).
2. **Smoother optimization target.** `|cos(z', w)|` has a non-differentiable kink exactly at
   `cos = 0` — the point the training objective is trying to reach and hold. `cos²` is smooth
   everywhere, with zero value *and* zero gradient at orthogonality, making it the better-behaved
   loss to sit at its own equilibrium rather than oscillate around a kink.

## 3. Conventional arm matching

Per `22_Experimental_Design.md` §3: Conventional is "adapter-based fine-tuning at matched adaptation
budget... to the Intervention arm, but with the plain task loss only — no adversarial/domain
objective." Made precise:

**Must match exactly** (same values, not just same order of magnitude): adapter architecture and
bottleneck width (§1), optimizer, learning rate, number of training epochs/steps, number of seeds.

**Must NOT include**: the orthogonality term at all. Not `λ_proj = 0` as a hyperparameter setting —
`L_orth` is **structurally absent** from Conventional's loss. This distinction matters: setting
`λ_proj = 0` would still imply a "λ_proj = 0 point on the Intervention arm's own ladder" reading;
Conventional is a categorically different training objective (`L = L_task`, full stop), not the
zero-strength end of Intervention's own dose axis — keeping this precise protects C4's actual
question (does intervention *type* matter, not just intervention *presence at zero strength*).

**Extension, reasoned not required by protocol text**: Conventional's own trained adapter still
produces some `Δz_conventional` (task-driven, not nuisance-constrained). Applying the *same*
α-interpolation scaffolding — `z'(α) = z + α·Δz_conventional`, same `α ∈ {0,.25,.5,.75,1}` grid — to
Conventional gives Stage 5 a matched measurement structure to compare against Intervention's own
ladder, point for point, without reintroducing any nuisance mechanism into Conventional's training.
This is a measurement-structure choice, not a training-objective one — Conventional's loss stays
exactly `L_task`, unchanged by this.

## Status

```
Stage 4, adapter and control specification:
  Adapter architecture:       LOCKED — residual bottleneck, z' = z + W2·σ(W1·z)
  Orthogonality loss formula: LOCKED — squared cosine similarity, matches CSG-Lite's own formula exactly
  Conventional arm matching:  LOCKED — matches architecture/optimizer/epochs/seeds; L_orth structurally
                               absent, not zeroed; same α-ladder scaffolding for measurement parity

Still open (execution-level): bottleneck width, nonlinearity choice, initialization, λ_proj's value,
  optimizer/learning-rate/epoch values themselves

Training: NOT AUTHORIZED
```
