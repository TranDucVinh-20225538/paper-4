# Nuisance Direction — Execution Review

Not Discussion, not Result — a structural record of what fed into what, same role as
`GroupBExecution/review.md`.

## Inputs

**`ReferenceTrainEmbedding`** (for `μ_isic`)
n = 16,211 (ISIC 2019 train split, all rows — `domain` is always `isic` in this artifact)
Status: FROZEN · SHA256-verified before use (`logs/nuisance_direction_run_2026-08-05.log`)

**`ReferenceEmbedding`** (for `μ_pad_ufes`)
n_pad_ufes = 2,298 (filtered from n=7,365 via `metadata.csv`'s `domain == "pad_ufes"`)
Status: FROZEN · SHA256-verified before use

## Implementation

Closed-form, deterministic — no optimizer, no gradient, no training loop.

```
μ_isic     = mean(ReferenceTrainEmbedding.embeddings, axis=0)
μ_pad_ufes = mean(ReferenceEmbedding.embeddings[domain=="pad_ufes"], axis=0)
w_raw      = μ_isic − μ_pad_ufes            (‖w_raw‖ = 9.014975)
w          = w_raw / ‖w_raw‖                 (‖w‖ = 1.000000)
```

Formula locked in `Stage4_Intervention_Method_Specification.md` §1, not decided here. Script:
`analysis/compute_nuisance_direction.py`. Log: `logs/nuisance_direction_run_2026-08-05.log`.
Specification: `NuisanceDirection/execution_contract.md`, written before this script existed.

## Outputs

`NuisanceDirection/`, not frozen at time of writing:

- `nuisance_direction.npz` — `w`, `w_raw`, `mu_isic`, `mu_pad_ufes` (all 1024-d)
- `manifest.json` — full provenance
- `sha256.txt`

## Protocol deviations

**None.** Population, formula, and sign convention match `execution_contract.md` and
`Stage4_Intervention_Method_Specification.md` §1 exactly.

## Known limitations

- **`μ_pad_ufes` is fit on the same PAD-UFES population that also appears in `ReferenceEmbedding`'s
  own OOD evaluation set** — there is no separate PAD-UFES-train population (verified via
  `CSG-SKin/src/datasets/splits.py`, `Stage4_Intervention_Method_Specification.md` §1). This mirrors
  Papers 1–3's own established precedent for domain-training-signal purposes, not a new risk this
  artifact introduces. It does mean the sanity check in `review_results.md` (projecting
  `ReferenceEmbedding`'s own PAD-UFES rows onto `w`) is not a fully independent check on the
  PAD-UFES side — the ISIC side of that same check *is* independent (`μ_isic` is fit on
  `ReferenceTrainEmbedding`, tested on `ReferenceEmbedding`'s disjoint ISIC-test rows).
- Not addressed here: whether this direction, once used as an adapter's regularization target,
  actually produces a Gate-1-passing manipulation — that is Stage 4's next phase, not this artifact's
  job.
