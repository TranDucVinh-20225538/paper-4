# Stage 4 — Hyperparameter Selection Rules

**Locks bottleneck width, nonlinearity, and λ_proj — as pre-committed selection *procedures*, not
values chosen after seeing results.** Per Vinh's explicit instruction: these must be "viết trước,
không chọn sau khi thấy kết quả" (written before, not chosen after seeing the result). No code. No
training authorized.

## 1. Bottleneck width `r`

**Rule, fixed before any training**: an ordered grid `r ∈ {16, 32, 64, 128}`, smallest first.
Train the Intervention arm's adapter starting at `r = 16`. Check Gate 0 and Gate 1. If either fails,
move to the next value in the grid and repeat. **Stop at the smallest `r` that passes both Gate 0
and Gate 1.** That value becomes the bottleneck width for every arm in Stage 4 (Intervention,
Conventional, Adaptation) — required for the "matched capacity" comparability §3 of
`Stage4_Adapter_And_Control_Specification.md` already locked.

**Why this is a legitimate pre-registered rule, not a loophole**: the stopping criterion is Gate 0
/ Gate 1 — integrity and manipulation checks, explicitly defined (`22_Experimental_Design.md` §3.5)
as producing "inconclusive, not falsifying" outcomes on failure, never a scientific claim. Selecting
`r` by whether it clears these gates is calibrating implementation capacity against an integrity
bar, not selecting for a favorable C2–C5 result — the grid is walked in one direction only (smallest
to largest, stopping at first pass), so there is no room to compare multiple passing values and pick
the one that "looks best" downstream.

**Why this grid**: `{16, 32, 64, 128}`, powers of two spanning two orders of magnitude, is the
standard range for a single output-stage bottleneck adapter in the surveyed literature
(`AR-01_Architecture_Study.md` — AdaptFormer, CLIP-Adapter-family widths fall in this range relative
to their own backbone dimensionalities). Not derived from CSG-Lite's 16-d `z_lesion` directly —
that number describes a fully-trained CNN branch's own output dimensionality, not a bottleneck width
on top of an already-1024-d frozen representation; a different quantity, not a transferable one.

## 2. Nonlinearity `σ`

**Rule: GELU** (matching `torch.nn.GELU`'s default, `approximate='none'`).

**Reasoning**: not a free choice needing its own search — PanDerm's own MLP blocks use exactly this
activation. Confirmed directly from this project's own tooling, not assumed: when the PanDerm
checkpoint was first loaded (Stage 1.2, and again during the EA-01 extraction run), the printed
module structure showed `(act): GELU(approximate='none')` in every transformer block's MLP. Using
the same activation the backbone itself already uses is the lower-risk default — introducing a
different nonlinearity (ReLU, SwiGLU) would need its own justification for why the adapter's internal
dynamics should differ from the frozen encoder's own, and no such reason has been identified. No
search grid needed here — GELU is the well-grounded default, not one candidate among several.

## 3. `λ_proj`

**Rule, fixed before any training, same principle as §1**: an ordered grid
`λ_proj ∈ {0.1, 1, 5}`, log-ish spaced, loosely inspired by (not copied from) CSG-Lite's own
`λ_orth ∈ {0, 1, 5}` scale — inspired only in the sense of spanning a low/medium/high range; the
exact magnitude is not assumed transferable across architectures with different feature
dimensionalities and loss scales, which is precisely why this is resolved by an empirical rule, not
by reusing CSG's numbers directly. `0` is excluded from this grid (unlike CSG's own) because the
Intervention arm's `λ_proj` is nonzero by definition — a `λ_proj = 0` point would just be
Conventional's own (structurally different) objective, already covered by
`Stage4_Adapter_And_Control_Specification.md` §3.

Train at `λ_proj = 0.1` first (holding `r` at whatever §1's rule selected). Check Gate 0 and Gate 1.
If either fails, move to the next grid value and repeat. **Stop at the smallest `λ_proj` that passes
both.** Same justification as §1: this is the minimum-effective-dose principle, standard in
mechanism studies — the smallest intervention that reliably produces the intended manipulation is
more defensible evidence for the mechanism than the largest one, since it can't be explained away as
"any strong enough perturbation would have moved geometry somehow." Selection is against Gate 0/
Gate 1 only, never against C2–C5 outcomes, for the same reason as §1.

## Status

```
Stage 4, hyperparameter selection rules:
  Bottleneck width r:  RULE LOCKED — grid {16,32,64,128}, stop at smallest passing Gate 0 + Gate 1
  Nonlinearity σ:       LOCKED — GELU, matches PanDerm's own verified activation
  λ_proj:               RULE LOCKED — grid {0.1,1,5}, stop at smallest passing Gate 0 + Gate 1

No value in either grid has been run. No training authorized. Both selection rules are calibrated
against Gate 0/Gate 1 only — never against C2–C5 results.
```
