# a_c predictor — class-level test

The earlier report ρ≈0.88 was a_c's own correlation (a_c_mean vs max|ΔAUROC|kde), computed fresh in this export — not the σ̄-deviation figure restated. Those two predictors agree to two digits on the 15-cell panel (a_c↔|ΔAUROC| ρ=0.88; σ̄-dev↔|ΔAUROC| ρ=0.88) because a_c predicts the σ̄ move that drives the AUROC move. At backbone aggregate n=5 that ρ is descriptive only (p≈.05). The test is class-level.

## Class-level test (the evidence)

n = 50 (= 5 backbones × 10 CIFAR-10 classes).
Spearman a_c ↔ mean|Δ kde score| on that class's ID-test points: ρ = -0.0933, p = 0.519.
Spearman a_c ↔ mean|log(σ̄_ratio/c)| (bandwidth mechanism): ρ = 0.9525, p = 2.01e-26.

## Backbone aggregate (descriptive only)

n = 5. a_c_mean ↔ max|ΔAUROC|kde: ρ = 0.9000, p = 0.0374. Not the test.

Artifacts: `kde_anisotropy_per_class.csv`, `kde_anisotropy_class_join.csv`, `kde_anisotropy_per_backbone.csv`, `kde_anisotropy_predictor.json`.
