# Nuisance Direction — Execution Contract

**Tier 0 (Validation/Planning). No code executed here. No score, vector, or output artifact is
produced by this document.** Written before any implementation, per Vinh's explicit instruction
(2026-08-05) to treat this exactly like Group B: its own artifact, its own contract, kept separate
from adapter code — so that if Gate 1 later fails, or an Execution Amendment is needed, it's
immediately clear whether the problem lies in direction estimation, adapter implementation, or
training, rather than all three tangled into one pipeline.

This is Stage 4's first step that produces a new artifact (Step 2 of
`Stage4_Execution_Plan_v2.md`) — not adapter training. It produces the fixed target direction the
adapter will later be regularized against; it does not itself intervene on anything.

## Inputs

| Population | Role | Artifact | Rows |
|---|---|---|---|
| ISIC 2019 train | `μ_isic` | `ReferenceTrainEmbedding/embeddings.npy` (all rows — `domain` is always `isic` in this artifact) | 16,211 |
| PAD-UFES-20 | `μ_pad_ufes` | `ReferenceEmbedding/embeddings.npy`, rows where `metadata.csv`'s `domain == "pad_ufes"` | 2,298 |

Both frozen, both SHA256-verified against their own manifests immediately before use (same pattern
as `groupB_mahalanobis_family.py`) — not assumed intact from a prior check.

**Population choice, justified in advance, not decided after seeing results**: this is the same
split `Stage4_Intervention_Method_Specification.md` §1 already established and verified — Papers
1–3's own precedent (`CSG-SKin/src/datasets/splits.py`, `build_csg_train_dataframe`) reuses the full
PAD-UFES population for domain-training-signal purposes; no new artifact or Execution Amendment
required.

## Method

Closed-form. Deterministic. **No optimization, no gradient, no training** — a fixed-point
computation, not a fitted model in the machine-learning sense.

```
μ_isic     = mean(ReferenceTrainEmbedding embeddings, axis=0)                    ∈ R^1024
μ_pad_ufes = mean(ReferenceEmbedding embeddings where domain == "pad_ufes", axis=0)  ∈ R^1024
w_raw      = μ_isic − μ_pad_ufes
w          = w_raw / ‖w_raw‖₂                                                    (unit-normalized)
```

Formula and sign convention exactly as locked in `Stage4_Intervention_Method_Specification.md` §1 —
not re-decided here.

## Output artifact

| File | Content |
|---|---|
| `nuisance_direction.npz` | `w` (1024,), `w_raw` (1024,, pre-normalization), `mu_isic` (1024,), `mu_pad_ufes` (1024,) |
| `manifest.json` | Provenance: input artifact paths + SHA256, formula, n_isic/n_pad_ufes, `‖w_raw‖₂`, timestamps |
| `sha256.txt` | Per-file checksums |

Reviewed (`review.md`, `review_results.md`), then frozen — same chain as every prior Stage 3/
Group B artifact. Not frozen until sanity-checked.

## What this does not do

Does not train anything. Does not touch the adapter (not yet implemented). Does not compute Gate 0,
Gate 1, or any measurement suite. Does not decide `λ_proj`, bottleneck width, or any other adapter
hyperparameter. Produces exactly one thing: the fixed direction `w` the adapter will later be
regularized against.
