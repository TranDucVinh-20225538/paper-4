# KDE bandwidth under ortho_linear — mechanism confirmed

Operator: `ortho_linear`, T(z) = c·Q·z (μ = 0).

## Framing (read this first)

**Published / primary KDE results use σ̄** (arithmetic mean of per-dimension
stds) — that is the Phase B formula and the Phase C scorer named `kde`. Those
numbers correctly describe the implementation that was run.

**`kde_det` is an addition, not an erratum.** It replaces only the bandwidth
line with det(Σ_c)^(1/(2d)) · n_c^(-1/(d+4)) to show that the invariance group
depends on the hyperparameter, not only on the functional form. Report both
columns side by side everywhere. Never present `kde_det` as a corrected value
replacing a wrong one.

Claim: *KDE as implemented is not rotation-invariant, and here is a one-line
change that makes it so* — not *the earlier numbers were wrong*.

## Mechanism

The Phase-B / Phase-C `kde` scorer uses Scott's rule with
σ̄_c = mean_j std_j(Z_c). A rotation preserves Σ's eigenvalues (hence
total variance) but redistributes variance across coordinate axes;
the arithmetic mean of square roots is **not** invariant to that
redistribution. Bandwidth h_c therefore changes class-by-class in a
non-uniform way, density ranks move, and AUROC moves.

Observed max |log(σ̄_after / (c·σ̄_before))| over classes = 0.1869
(would be 0 if σ̄ scaled exactly by c). Spearman correlation between
per-config mean |log(σ̄_ratio/c)| and max |ΔAUROC|(kde) over OOD rungs:
ρ = 0.8821 (p = 1.36e-05, n = 15).

## Confirmation: det(Σ) bandwidth

Variant `kde_det` uses h_c = det(Σ_c)^(1/(2d)) · n_c^(-1/(d+4)).
det(Σ) is rotation-invariant and scales as c^(2d) under T(z)=cQz, so
h_c scales exactly by c and the scorer is similarity-invariant.

Observed max |log(det_scale_after / (c·det_scale_before))| = 2.878e-05
(numerical noise only).

max |ΔAUROC| under ortho_linear (all seeds, all OOD):

| backbone | kde (σ̄, as implemented) | kde_det (addition) |
|---|---:|---:|
| clip_vitb16 | 0.005264 | 2e-08 |
| dinov2_vitb14 | 0.000134 | 7.89e-06 |
| resnet18_cifar_openood | 0.005939 | 1.137e-08 |
| resnet50_imagenet | 0.009294 | 1e-08 |
| vitb16_imagenet | 0.001272 | 1.05e-06 |

Flat threshold τ = 0.002: kde (σ̄) fails on several backbones;
kde_det is flat everywhere.

## Predictor before any transform: a_c = AM/GM

Rotation leaves σ̄_c unchanged (relative to scale c) exactly when Σ_c is
isotropic. The mechanism-matched statistic is the AM/GM ratio in the
**unrotated** train space:

    a_c = σ̄_c / det(Σ_c)^(1/(2d))

| backbone | mean a_c | max a_c | mean κ₂(Σ_c) at REL_CUTOFF=1e-6 | max\|ΔAUROC\|(kde) |
|---|---:|---:|---:|---:|
| resnet18_cifar_openood | 6.450 | 7.073 | 1.64e5 | 0.005939 |
| resnet50_imagenet | 3.622 | 3.718 | 1.21e5 | 0.009294 |
| clip_vitb16 | 2.312 | 2.470 | 2.61e5 | 0.005264 |
| vitb16_imagenet | 2.049 | 2.163 | 3.45e4 | 0.001272 |
| dinov2_vitb14 | 1.851 | 2.001 | 5.11e4 | 0.000134 |

Across the 15 (backbone × seed) cells:

- Spearman mean a_c ↔ mean|log(σ̄_ratio/c)|: **ρ = 0.982** (p = 8.6e-11)
- Spearman mean a_c ↔ max|ΔAUROC|(kde): **ρ = 0.884** (p = 1.2e-05)
- Spearman mean κ₂ ↔ max|ΔAUROC|(kde): ρ = 0.491 (p = 0.063) — descriptor only;
  κ₂ depends on the rank cutoff, a_c does not.

Prescriptive half: the bandwidth rule decides the invariance group, and the
**size** of the σ̄ breakage is predictable from a_c before any operator runs.

Artifacts: `kde_sigma_bar.csv`, `kde_det_gate.csv`,
`kde_anisotropy_per_class.csv`, `kde_anisotropy_per_backbone.csv`,
`kde_anisotropy_gate_join.csv`, `notes/kde_anisotropy_predictor.json`.

## Gate reading

Code-correctness gate for similarity (`ortho_linear`): **pass** for
maha, knn10, cosine (and rmd). **`kde` (σ̄)** is an explained exception —
the functional form is a similarity-invariant density rank, but the
σ̄ bandwidth hyperparameter is not. `kde_det` confirms the diagnosis as an
addition. Both versions are retained in `SCORER_NAMES`.
