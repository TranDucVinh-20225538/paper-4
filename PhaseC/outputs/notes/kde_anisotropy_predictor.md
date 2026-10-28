# a_c predictor — class-level test

The earlier report ρ≈0.88 was a_c's own correlation (a_c_mean vs max|ΔAUROC|kde), computed fresh — not the σ̄-deviation figure restated. On the 15-cell panel (5 backbones × 3 seeds, clustered — five independent units, not fifteen) a_c↔|ΔAUROC| ρ=0.88 and σ̄-dev↔|ΔAUROC| ρ=0.88 agree to two digits because a_c predicts the σ̄ move. That panel is descriptive only; the test with power is class-level n=50.

## Score-shift attribution

ID evaluation points only; each point attributed to argmax_c log p_c (the density term that drives the KDE score), not to the ground-truth label.

## Class-level test (the evidence)

n = 50 (= 5 backbones × 10 CIFAR-10 classes).
Spearman a_c ↔ mean|log(σ̄_ratio/c)| (bandwidth mechanism): ρ = 0.9525, p = 2.01e-26. This link is solidly established.
Spearman a_c ↔ mean|Δ kde score| (ID points, argmax_c attribution): ρ = -0.0929, p = 0.521.
If the second is near zero: bandwidth deviation is explained; score displacement is not predicted by anisotropy alone — stop there.

## Backbone aggregate (descriptive only)

n = 5 independent backbones (the 15 cells are clustered by seed). a_c_mean ↔ max|ΔAUROC|kde: ρ = 0.9000, p = 0.0374. Not the test; carries no weight in the write-up.

Artifacts: `kde_anisotropy_per_class.csv`, `kde_anisotropy_class_join.csv`, `kde_anisotropy_per_backbone.csv`, `kde_anisotropy_predictor.json`.
