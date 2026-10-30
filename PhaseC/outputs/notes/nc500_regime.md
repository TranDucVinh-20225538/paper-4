# CIFAR-10 n_c=500 subsample — regime confirmatory

Registered before the run (Follow-up 11). Hold K=10 on CIFAR-10; subsample train to n_c=500 (N=5·10³). This cuts the pooled estimator's sample size; CIFAR-100 did not (N unchanged).

## Predictions

1. λ_min/λ̄ falls on all five by ~(1−√(d/N))².
2. Regime assignment unchanged (ten-order gap vs factor 2–8).

## λ_min/λ̄

| backbone | ratio_full | ratio_n_c500 | fall_obs | MP_fall ≈1/(1−√(d/N))² | regime_full | regime_500 |
|---|---:|---:|---:|---:|---|---|
| clip_vitb16 | 4.132e-04 | 3.531e-04 | 1.17× | 2.16× | computable | computable |
| dinov2_vitb14 | 3.090e-14 | 2.230e-14 | 1.39× | 2.70× | floor | floor |
| resnet18_cifar_openood | 2.556e-03 | 2.323e-03 | 1.10× | 2.16× | computable | computable |
| resnet50_imagenet | 2.964e-03 | 1.733e-03 | 1.71× | 7.72× | computable | computable |
| vitb16_imagenet | 1.193e-13 | 1.031e-13 | 1.16× | 2.70× | floor | floor |

## maha_noridge discordant (clean affine, one seed)

| backbone | general_affine | ortho_linear | ortho_scale | zca |
|---|---|---|---|---|
| clip_vitb16 | 0 | 0 | 0 | 0 |
| dinov2_vitb14 | 42 | 130 | 128 | 1348 |
| resnet18_cifar_openood | 0 | 0 | 0 | 0 |
| resnet50_imagenet | 0 | 0 | 0 | 0 |
| vitb16_imagenet | 60 | 179 | 191 | 1168 |

ε²_float32 = 1.42e-14.

