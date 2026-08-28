# Stage 4 Execution Authorization

**Version: v1.0 — 2026-08-05.** Marks the transition, for the Intervention Arm specifically, from
design to execution: every scientific decision below was locked *before* any Intervention-arm
result existed to be influenced by. From this point, Paper 4's representation intervention is not
being designed — it is being tested.

## Locked documents

| | Document | Decides |
|---|---|---|
| ✓ | `ArchitectureResolutions/AR-01_Architecture_Study.md` | The literature landscape (27 papers, 5-tier taxonomy) |
| ✓ | `ArchitectureResolutions/AR-01_Resolution.md` | Intervention family (single-backbone, output-stage adapter/projection) |
| ✓ | `Stage4_Architecture_Review.md` | What Stage 4 produces, dependency on the Reference Arm, Gate 0/Gate 1 reapplication, planned Stage 5 comparisons |
| ✓ | `Stage4_Architecture_Instantiation_Contract.md` | Intervention target, output space, geometry criterion, fixed scope |
| ✓ | `Stage4_Intervention_Design_Choice.md` | Nuisance mechanism (closed-form, not learned-adversarial), injection location (output-stage) |
| ✓ | `Stage4_Intervention_Method_Specification.md` | Adapter requirement, nuisance-direction estimator, projection/regularization equation, dose ladder, Gate 1 timing |
| ✓ | `Stage4_Adapter_And_Control_Specification.md` | Adapter architecture, orthogonality loss formula, Conventional-arm matching |
| ✓ | `Stage4_Hyperparameter_Selection_Rules.md` | Bottleneck-width and λ_proj selection *procedures*, nonlinearity |
| ✓ | `Stage4_Execution_Plan_v2.md` | Run order — the only thing left to decide, and it decides nothing scientific |

Nine documents, one continuous chain — each with its own decision record, none silently folded into
another. `Stage4_Execution_Plan.md` (v1)'s cross-arm scheduling note and Stage 5 readiness gate
remain in force, carried forward by v2.

## Standing rule, effective immediately

**No further scientific-design changes are permitted once execution begins.** Not to the
intervention target, the nuisance estimator, the projection equation, the adapter architecture, the
orthogonality formula, the Conventional-arm control, or the hyperparameter selection procedures
listed above. If a problem surfaces during execution:

- An **implementation bug** (code doesn't match what a locked document specifies) → fixed directly,
  logged, execution continues.
- Anything else — a design choice that turns out wrong, an unanticipated gap, a result that suggests
  a different mechanism would have been better — is **not** fixed inline. It goes through the
  existing Amendment process (Execution Amendment, Architecture Resolution, or Protocol Deviation,
  per which layer it actually touches — `00_execution_checklist.md`'s standing categorization),
  exactly as EA-01 and AR-01 were handled. The design freezes; only the amendment path can reopen a
  specific, named piece of it.

This is the same Artifact Immutability Rule's logic, applied one level up — to the *design*, not
just the artifacts it produces.

## What this authorizes

Execution of `Stage4_Execution_Plan_v2.md`'s run order, starting at Step 1. Training, embedding
extraction, gate evaluation, and freezing for the Intervention and Conventional arms, per that plan,
using the hyperparameter selection procedures in `Stage4_Hyperparameter_Selection_Rules.md` — not
any value chosen ad hoc.

## What this does not authorize

Any change to the design itself. Any shortcut through the hyperparameter selection procedures (no
skipping to a "likely good" width or λ_proj). Any comparison, interpretation, or claim — Stage 4
produces artifacts; Stage 5 (not yet authorized) is where comparison happens.

## Status

```
Phase A:                COMPLETE
Reference Arm:           CLOSED
AR-01:                    COMPLETE (Step 1 + Step 2)
Stage 4 scientific design: COMPLETE, LOCKED — no further changes without an Amendment

Stage 4 execution:      AUTHORIZED — v1.0, 2026-08-05
```
