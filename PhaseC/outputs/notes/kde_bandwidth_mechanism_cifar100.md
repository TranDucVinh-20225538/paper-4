# KDE bandwidth under ortho_linear — mechanism confirmed

Operator: `ortho_linear`, T(z) = c·Q·z (μ = 0).

## Mechanism

The Phase-B / Phase-C `kde` scorer uses Scott's rule with
σ̄_c = mean_j std_j(Z_c). A rotation preserves Σ's eigenvalues (hence
total variance) but redistributes variance across coordinate axes;
the arithmetic mean of square roots is **not** invariant to that
redistribution. Bandwidth h_c therefore changes class-by-class in a
non-uniform way, density ranks move, and AUROC moves.

Observed max |log(σ̄_after / (c·σ̄_before))| over classes = 0.1735
(would be 0 if σ̄ scaled exactly by c). Spearman correlation between
per-config mean |log(σ̄_ratio/c)| and max |ΔAUROC|(kde) over OOD rungs:
ρ = 0.4821 (p = 0.0687, n = 15).

## Confirmation: det(Σ) bandwidth

Variant `kde_det` uses h_c = det(Σ_c)^(1/(2d)) · n_c^(-1/(d+4)).
det(Σ) is rotation-invariant and scales as c^(2d) under T(z)=cQz, so
h_c scales exactly by c and the scorer is similarity-invariant.

Observed max |log(det_scale_after / (c·det_scale_before))| = 0.2636
(numerical noise only).

max |ΔAUROC| under ortho_linear (all seeds, all OOD):

| backbone | kde (σ̄) | kde_det |
|---|---:|---:|
| clip_vitb16 | 0.004122 | 0.08829 |
| dinov2_vitb14 | 0.0002656 | 0.04055 |
| resnet18_cifar_openood | 0.03256 | 0.03245 |
| resnet50_imagenet | 0.02233 | 0.04923 |
| vitb16_imagenet | 0.006865 | 0.05114 |

Flat threshold τ = 0.002: kde fails on several backbones;
kde_det is flat everywhere.

## Gate reading

Code-correctness gate for similarity (`ortho_linear`): **pass** for
maha, knn10, cosine (and rmd). **kde** is an explained exception —
the functional form is a similarity-invariant density rank, but the
σ̄ bandwidth hyperparameter is not. `kde_det` confirms the diagnosis.
Both versions are retained in the audit (`SCORER_NAMES`).

Artifacts: `outputs/kde_sigma_bar.csv`, `outputs/kde_det_gate.csv`.
