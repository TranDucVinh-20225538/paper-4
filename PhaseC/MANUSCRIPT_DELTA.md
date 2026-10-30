# Manuscript delta v2 — Phase C → the submitted paper

**Supersedes `PhaseC/MANUSCRIPT_DELTA.md` entirely.** That file was built from the Phase C
audit's own framing and from `paper/sections/*.tex`, which is a different (IEEEtran/JBHI)
draft. It is wrong in sign on four rows. This file is built by reading the submitted
manuscript, *"Reading an Unchanged AUROC: Invariance, Insensitivity and Infeasibility in
Distance-Based Out-of-Distribution Detection"* (Springer style, Theorem 1–2, Lemma 1–2,
Proposition 1–3, Corollary 1–3; SI Sections S1–S6). Quotes are verbatim with the
manuscript's own line numbers.

**What the paper is.** A methods paper proposing a three-Check auditing workflow, packaged
as `flatresaudit` v1.0.0 (Zenodo 10.5281/zenodo.23150930), validated on planted ground
truth and applied to one frozen dermatology backbone.

**What the advisor asked for.** Not a robustness check and not error-hunting. Discussion
l.752–756 names the paper's principal limitation, and that is what "more backbones, more
datasets" is aimed at. Every row below is scored against that target.

---

## A. Answers a question the paper explicitly declines to answer

**A1 — Is the reach ordering a property of distance-based detection, or of the
representation?**

> l.752–756: "The evaluation covers one backbone, one layer, one diagnostic task and four
> evaluation domain pairs. **Generalization beyond a single encoder is the principal
> limitation**: every statement is, strictly, about PanDerm ViT-L/16 and these domains, and
> **whether the ordering of scorers by reach is a property of distance-based detection or
> of this representation is not claimed either way**."

Phase C answers it, and the answer **splits**:

| | result | reading |
|---|---|---|
| "Mahalanobis nearly unable to reorder" (l.482, Table 5) | replicates on **5/5** backbones | a property of distance-based detection |
| the full four-scorer ordering | Kendall τ vs PanDerm 0.91, 0.91, 0.33, 0.33, 0.18 — **2/5** | a property of the representation |

This is the single most valuable result of the extension. It converts "not claimed either
way" into a measured answer with a boundary. Compare on the paper's four scorers only;
rmd and kde_det are additions and belong in C/B.

**Text:** rewrite l.752–756 and add the transfer table to Section 6.

## B. Fills a cell the paper marks undetermined

**B1 — KDE.** Table 1 (RQ1 structural classification under T_γ) classifies KDE as
**"Undetermined / Qualitative group only (Sec. 3.4) / Kernel and bandwidth model"**, and
l.249 says "KDE remains quantitatively undetermined."

Phase C determines the bandwidth half. SI S3.3 defines
`h_c = max{σ̄_c · n_c^(−1/(d+4)), 10^−6}` with σ̄_c the **mean per-dimension standard
deviation** — an arithmetic mean, hence rotation-dependent (AM of square roots is concave).
Measured: σ̄ deviates by 1.002–1.205 under rotation where the determinant form
`det(Σ_c)^{1/(2d)}` stays at ~2.9e-05; deviation predicts violation at ρ = 0.882 (n=15);
anisotropy predicts deviation at ρ = 0.95 (n=50). Under `ortho_linear` (pure rotation),
per-backbone max discordant pairs of 2×10⁶: kde 1.50e4 / 4.25e4 / 4.04e4 / 983 / 7111,
kde_det **0 / 0 / 0 / 18 / 10**, with bandwidth relative error ~1e-9–1e-10 on the three
computable backbones against 2.9e-5 and 9.5e-6 on the two at the float32 floor.

So KDE's group dependence is **not** intrinsic to the kernel — it is Scott's rule's
estimator. `kde_det` is an addition, not a replacement.

**Text:** Table 1's KDE row gains a determinate statement for the bandwidth; Section 3.4
gains the AM/GM argument; `kde_det` enters Section 3.6 as a variant.

## C. Adds a precondition to a stated result

**C1 — When is the ε = 0 branch available?** Table 1 gives Mahalanobis admissibility as
"Exact (ε = 0) / bounded (ε > 0)" under Theorem 1 / Theorem 2, and SI S1.5 states
Theorem 1 with "**no fixed ridge is added**". The paper is already careful here; what it
does not state is when ε = 0 is *computable*.

Phase C: with λ_min(C)/λ̄ measured against ε²_float32 = 1.4e-14 —

| backbone | λ_min/λ̄ | ε=0 refit |
|---|---|---|
| resnet50, resnet18, clip | 3.0e-03, 2.6e-03, 4.1e-04 | 268/270 clean-affine readings at exactly **0** discordant of 2×10⁶ (two exceptions, one pair each); max 1−ρ = 4.1e-11 |
| dinov2, vitb16 | 3.0e-14, 1.2e-13 | **not a computable object** — the variance in that direction is float32 rounding noise; ε = 1e-12 is 27–28× *worse* than ε = 1e-5 |

Two independent probes place the same boundary: the unregularised Mahalanobis solve, and
kde_det's determinant (B1). The regime is **stable across datasets** — CIFAR-10 and
CIFAR-100 put the same two backbones at the floor despite a tenfold change in both n_c and
class count — so it is a property of the representation, measured once per backbone.

**Text:** a precondition column or footnote on Table 1's Mahalanobis row.

**C2 — Theorem 2 is sound and uninformative.** Table 1's "Bound requires: per-point y
(Thm 2)". Phase C: holds 105/105; pred/obs 1.2e+03 to 7.1e+11; the pessimism is λ_min(C),
which inflates by ~1/λ_min². It cannot rank configurations (Spearman with 1−ρ = −0.148)
where the measured violation ranks them at 0.849.

**Text:** one sentence in Section 3.4 or Table 1's note.

## D. Confirms stated limitations on real backbones

**D1 — The in-group / low-reach case.** The paper states this four times and quantifies it:

> Algorithm 1, l.7: "INSENSITIVE (LOW OBSERVED REACH); **if δ_max ≈ 0, consult the
> analytical transformation group (Section 3)**"
> §4 l.348–350: "When scores barely change, **only a structural argument can distinguish
> mathematical invariance from insensitivity**."
> §4.1 l.384–386: "*structurally invariant* is reserved for a conclusion that follows from
> the results of Section 3."
> E1 l.426–428: "One case is out of reach of any score-based quantity: in-group flat
> responses whose scores the group transformation leaves unchanged have negligible reach
> and look like insensitivity."
> SI S4.2: "242 of 480 in-group flat responses had negligible reach and were read
> insensitive… **The protocol's stated limitation appeared as stated**."

Phase C reproduces it on five real backbones: Mahalanobis δ_med = 2.54 with S = 7.45e-05
and frac_invariant 0, against knn10 δ_med = 7.29 with S = 0.999. Confirmation, not a defect.

**D2 — Threshold transfer.** l.757–761: thresholds "are conventions fixed before execution
rather than derived", softened by stability "across an order of magnitude in τ and S_high,
with F0 the one threshold that must stay small." Conclusion l.804–805 asks for "validation
of the audit thresholds in each new setting." Phase C's 20-cell τ×S_high grid on five
backbones is exactly that evidence.

**D3 — Checks 2 and 3 transfer.** Check 2: k=10 floor 2.8e-07 against Holm's first
threshold 2.8e-04 over 180 tests, reachable, smallest k = 7. Check 3: w_construction
0.41–0.91, w_learned 0.057–0.25.

## E. Removes the scope limitation

> Abstract l.20: "**Findings concern one backbone**; pretraining overlap remains unverified,
> and no unique geometric mechanism or clinical safety claim is established."
> Conclusion l.804–805: "The empirical findings concern **one frozen backbone**…"

Five backbones (ResNet-18, ResNet-50, ViT-B/16, DINOv2 ViT-B/14, CLIP ViT-B/16), two ID
sets (CIFAR-10, CIFAR-100), six OOD rungs, 95 configurations per backbone. ImageNet-200 is
blocked and is not claimed.

**Text:** abstract l.20, discussion l.752–756, conclusion l.804–805. Keep the pretraining-
overlap and clinical-safety clauses — Phase C does not touch either.

## F. Corrections to Phase C, not to the paper

**F1 — Phase C invented an `invariant` label.** Algorithm 1 has no such branch: its labels
are RESPONSIVE / INSENSITIVE (LOW OBSERVED REACH) / EMPIRICALLY PROTECTED AT TESTED DOSE /
SUSCEPTIBLE, with structural invariance routed to Section 3. `reach.check1_reading`'s
`invariant` label, and the `invariant_fp` and rank-preserving columns built to repair it,
are audit-side artefacts. Keep `check1_rank_reading` as a **tool** — rank preservation is
the right direct test when you want one — but it does not correct the paper, and nothing
in F goes into the manuscript.

**F2 — The audit's "Lemma 1 / Lemma 2" is the audit's own naming.** SI p.3: "Two results
are stated only here: **Lemma 1 (the bound for a purely linear map, which does not apply to
the implemented cosine scorer)** and Proposition 2." The paper already excludes it. Phase
C's 28.6% failure rate of the κ-only form is a *numerical confirmation* of a stated
exclusion — reportable, but as support, not as a correction.

## G. Negatives and caveats to report

**G1** Anisotropy does **not** predict KDE score displacement after attributing each ID
point to its argmax class: ρ = −0.09, p = 0.52.
**G2** kde's difficulty–reach correlation is −0.23, sensitive to 2 below-chance cells
(20 of 300 seed rows); pooled it is −0.087. Maha −0.892 and rmd −0.899 are stable.
**G3** Round-two prediction (iv) — λ_min/λ̄ falls at n_c = 500 — **failed**, by a derivation
error: `scatter/(N−K)` is pooled and N = 50 000 on both datasets, so n_c never enters the
estimator. (iii) failed as a consequence of (iv). Record as one failed registered
prediction with a diagnosed cause.
**G4** PanDerm's published figures reproduce exactly against the **5-seed mean**
(6.47e-05 → 6.5e-5; 0.04782 → 0.0478). The estimator was unstated; name it.

---

## What this changes in the plan

The deliverable is **one transfer section plus three text edits** (abstract l.20,
discussion l.752–756, conclusion l.804–805), not nine rows of claim surgery. No claim of
the paper's is broken. Rows A1, B1 and C1 are new contributions; D is confirmation; E is
the advisor's actual request; F never reaches the manuscript.

Nothing above depends on the two jobs still running. (ii) and the r50 tail land in B1 and
E respectively.
