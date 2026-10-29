# Manuscript delta — Phase C audit → paper claims

One row per confirmed result. Current claim → replacement → evidence pointer.
Numbers as of `81fbf8b` / Follow-up 09–10 corrections; correct any that drift.

Items 1, 4 and 5 are errors in the paper's own theory-to-practice chain, found by an
extension commissioned as a robustness check. Item 5 is a finding about a published
instrument (Check 1) that currently has **no home** in the manuscript's framing —
flagged here rather than decided.

| # | Current manuscript claim | What the audit says | Evidence |
|---|---|---|---|
| 1 | Mahalanobis is exactly rank-invariant under the affine group (Theorem 1), stated of the audited scorer | The audited scorer carries `eps = 1e-5` and is the Theorem 2 object. The exact Theorem 1 claim is available only where `λ_min(C)/λ̄ ≫ ε²_float32` — three of five backbones (resnet18, clip, resnet50): **268/270** clean-affine readings at exactly 0 discordant pairs of 2×10⁶, max 1−ρ = 4.1e-11; the two exceptions are one pair each (resnet18 / general_affine / texture / seeds 1, 3). Confirmed to one pair in 2×10⁶ across 270 readings. On dinov2 and vitb16, `λ_min(C)` sits at the float32 noise floor and the unregularised estimator is not a computable object. The ridged statement is the only well-posed claim on all five. Do not report `λ_min/ε²` (not scale-invariant); keep `λ_min/λ̄` with ε² quoted beside it. | `FINDINGS.md` §Q1 regime table; `maha_noridge_clean_affine.csv`; `rank_failure_attribution.csv` (ridge vs float32) |
| 2 | The reach ordering across scorers reproduces (PanDerm: maha ≪ knn < kde < cosine) | Mahalanobis least susceptible replicates on **all five** backbones; the full ordering does not — Kendall τ vs PanDerm 0.91, 0.91, 0.33, 0.33, 0.18 (**2 of 5**). RMD is second. Mid-ranks are backbone-dependent. | `FINDINGS.md` §Q2 |
| 3 | Theorem 2 bounds the ridge-induced deviation | Holds **105/105** and is **vacuous**: pred/obs 1.2e+03 to 7.1e+11, dominated by `λ_min(C)` inflating the bound by ~1/λ_min². It carries the right variable and cannot rank configurations (Spearman of predicted_max with 1−ρ = **−0.148**; measured `actual_refit_observed_max` ranks at **0.849**, n=105 matched Theorem 2 rows). | `FINDINGS.md` §Q3; `bounds_evaluation.csv` |
| 4 | Lemma 1 (κ-only) applies to the audit's operators | Lemma 1 assumes the map acts about the origin; the audit's maps act about the fit mean. Lemma 1 fails **28.6%**; Lemma 2 holds **105/105** at tightness 0.12–0.23. First numerical demonstration that the centred/linear distinction has teeth. | `FINDINGS.md` §Q3 |
| 5 | Check 1's `invariant` label identifies invariance | The registered label requires `S ≥ S_high` and is **structurally unreachable** for an exactly (or ridge-near) rank-invariant scorer with wide margins: `S(δ)` bounds arbitrary perturbations of size δ and says nothing about a monotone one. Rank preservation (`ρ = spearman(s, s̃∘T)`) is the correct test. **`invariant_fp` is not a fix** — it drops the reach requirement and is stricter flatness, not invariance. This is a defect in a published instrument that ships as a library. **No manuscript home yet** — framing decision needed (methods erratum vs contribution vs appendix). | `FINDINGS.md` §Q1 (registered / `_fp` / rank-preserving columns); `reach.check1_reading` vs `reach.check1_rank_reading` |
| 6 | PanDerm's reported reach figures (maha 6.5e-05, cosine 0.0478) | Reproduce exactly against the **5-seed mean** of median-quantile S in `PhaseB/outputs/rq1/rq1_4_susceptibility_observed.csv` (6.47e-05 → 6.5e-5; 0.04782 → 0.0478). Not the median, not the 10-seed table. Estimator was unstated; now named. | `FINDINGS.md` §Q2; `write_findings.panderm_reference` |
| 7 | KDE is a comparable density scorer | Scott's rule with `σ̄_c` is rotation-dependent; deviation 1.002–1.205 where the det-based form stays at ~2.9e-05; deviation predicts violation at ρ ≈ 0.882 (n=15); anisotropy predicts deviation at ρ ≈ 0.95 (n=50). **kde_det** is the fix and is an **addition**, not a replacement. Under ortho_linear, prediction written before the rank columns were read; **separation holds**: kde max discordant = 42454, kde_det max = 18 (cosine 1). Q4's ortho_scale AUROC gap is reading the bandwidth mechanism. | `FINDINGS.md` §Q1 (prediction + result), §Q4; `rank_preservation.csv`; `kde_anisotropy_*.csv` |
| 8 | — (no claim) | **Negative result:** anisotropy does not predict KDE score displacement after correctly attributing each ID point to its argmax class — ρ ≈ −0.09, p ≈ 0.52. Report, do not bury. | `FINDINGS.md` §Q4; `kde_anisotropy_class_join.csv` |
| 9 | Difficulty correlates with reach | Holds for maha (Spearman −0.892) and rmd (−0.899). kde's −0.23 is sensitive to **2** named below-chance cells (20 of 300 seed-level rows); pooled was −0.087. Report with that caveat. Below-chance cells named and excluded from the correlation. | `FINDINGS.md` §Q2 (difficulty axis) |

## Standing attribution rule (Phase C)

Every causal claim ships with the test that discriminates it from the nearest rival mechanism, or it ships labelled **unattributed**. If two mechanisms predict the same observable, name both and say which experiment separates them (model: eps sweep vs float64 storage for γ<0 rank failures). "Numerical issue", "arithmetic residual", and "ill-conditioned" are not attributions unless a column shows the alternative was excluded.

Corollaries: never pool a max over a grouping the claim is about (pair within backbone / operator / seed); a claim of exactness takes a count, never a round word.

## Not in this delta

- ImageNet-200 round three: still blocked (ImageNet-1k absent). Scope notes and imglist SHA-256s prepared; do not substitute another ID set.
- CIFAR-100 round two: four signed predictions (reach↑, KDE deviation↑, Theorem 2 looser, λ_min/λ̄↓) — report when audits complete; not yet mapped to manuscript sentences.
