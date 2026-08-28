# Stage 0 — Retrospective Analysis

**Claim tested:** C1 (`02_Claim_Matrix.md` Claim Hierarchy: "Which geometric quantities explain the
trade-off (LID, spectral-decay slope, condition number, variance allocation)?").

**Role:** decisive, free gate before any GPU is spent (`16_Minimal_Sufficient_Experiment.md` Week
0; `Scientific_Hypothesis.md` falsifiable-outcomes section). Not a claim about the FM arms. Not a
test of Predictions 1–3 (those are C2/C3, Stage 3, on the FM).

**Input data:** Paper 3's existing artifacts only, read-only. Nothing in `paper-3/` is modified.

- `paper-3/results/e1_embeddings/{rung}_s{seed}.npz` — `features` (N×16), `labels` (N,, 8 classes)
  for the 13 primary-ladder checkpoints (`runA_grl`×5 seeds, `runB_orth1`×5 seeds, `runB`×3 seeds —
  λ_orth = 0, 1, 5 respectively). Same features already used to fit `e1_geometry_metrics.csv`'s
  condition number.
- `paper-3/results/e1_geometry_metrics.csv` — condition number, reused as-is, not recomputed.
- `paper-3/results/e2_auroc.csv` — Mahalanobis AUROC (ID=ISIC test vs. OOD=PAD-UFES-20), same 13
  checkpoints, same `reg_eps=1e-5` precision matrix as the geometry metrics (per
  `extract_auroc_e2.py`'s docstring — this is why the two CSVs are comparable at all).

## Method (exact formulas, source: arXiv:2510.15202, per `08_Mechanism_Map.md` Mechanism 1)

**LID (Local Intrinsic Dimensionality).** Levina–Bickel MLE estimator, k=50 nearest neighbors
(paper's Appendix A choice for the main composite), computed per-sample on the raw (untransformed)
16-d features — the same feature space Paper 3's Mahalanobis and condition-number metrics already
use, since Stage 0 does not include the paper's β-normalization sweep (that knob is Stage 3's
Prediction-2 causal control, not part of C1):

```
LID_k(z) = -[ (1/k) * sum_{j=1}^{k} log( r_j(z) / r_k(z) ) ]^{-1}
```

`r_j(z)` = distance from `z` to its j-th nearest neighbor within the same checkpoint's embedding
set. Dataset-average LID = mean of `LID_k(z)` over all points in that checkpoint.

**Within-class spectral-decay slope.** Eigenspectrum of the within-class scatter matrix:

```
S_w = (1/N) * sum_{c=1}^{K} sum_{i: y_i=c} (z_i - mu_c)(z_i - mu_c)^T
```

(K=8 diagnostic classes — the same `labels` array `e1_geometry_metrics.csv` already conditions on;
unregularized, unlike the Mahalanobis precision matrix, since regularization would distort the tail
of the spectrum this metric is specifically about.) Eigenvalues sorted descending, full-range
least-squares fit `log(λ_i) = a + b·i` across all d=16 dimensions; report `|b|` (magnitude).

**Composite:** `P = LID × |slope|` (arXiv:2510.15202 Eq. 6, `m(β)·|s(β)|`, with no β-transform
applied here — see above).

## Statistical test (per `24_PreRegistration.md` §5/§8)

Kendall's τ (exact, `scipy.stats.kendalltau(method="exact")` — valid here because both variables
are continuous floats with no ties, unlike the ordinal-group case `analyze_e1.py` had to handle
with a custom permutation enumeration) between:

1. **Primary C1 test:** composite `P` vs. Mahalanobis-AUROC, n=13 (all checkpoints) and n=9
   (common-seed subset {42, 52, 62} across all three rungs, matching `e1_kendall_tau.csv`'s
   convention for direct comparability).
2. **Secondary/diagnostic** (decomposition of the composite, not a separate claim — Prediction 1's
   own wording is "LID and/or... spectral-decay slope"): LID alone vs. AUROC; slope alone vs.
   AUROC; condition number (already computed) vs. AUROC, reusing Paper 3's own headline metric for
   continuity.

No Holm-Bonferroni correction applied here — the multiple-comparison policy
(`24_PreRegistration.md` §9) governs the C2 scan (multiple scorers × multiple geometry metrics ×
multiple arms on the FM), not this single-composite, single-scorer gate check.

## Interpretation rule (pre-committed, not decided after seeing results)

- τ significant and same-signed as expected (higher composite → lower/worse AUROC) → proceed to
  Stage 1 as preregistered, with the mechanism-consistent signal noted.
- τ not significant / no signal → not a falsification (n=13 is small, and this is a correlational
  gate, not Prediction 2's causal test) and not a stop condition either — Stage 0 is a sanity/gate
  check on pipeline and mechanism-consistency, not itself a test of Predictions 1–2. Proceed to
  Stage 1 as preregistered; log the result and any interpretable pattern in it honestly, and let it
  inform how Stage 3's FM-arm results are read later, without editing the hypothesis or design now.
- Either outcome is logged per the 24-Hour Rule (`25_Research_Constitution.md`) if surprising — no
  hypothesis or protocol edit for 24 hours regardless of which way it comes out.

## Result (executed 2026-08-04)

Full numbers: `outputs/stage0_geometry_metrics.csv`, `outputs/stage0_geometry_auroc_merged.csv`,
`outputs/stage0_c1_kendall_tau.csv`, `outputs/stage0_summary.json`. Run log:
`logs/stage0_run_2026-08-04.log`.

**Primary C1 test:** composite (`LID × |slope|`) vs. Mahalanobis-AUROC —
**τ = −0.154, exact p = 0.510, n = 13** (common-seed subset: τ = −0.111, p = 0.761, n = 9).

**Secondary (decomposition + continuity):**

| Metric vs. AUROC | τ_full | p_exact (n=13) | τ_common | p_exact (n=9) |
|---|---|---|---|---|
| LID alone | −0.026 | 0.952 | 0.000 | 1.000 |
| Spectral-decay slope alone | −0.128 | 0.590 | −0.111 | 0.761 |
| Condition number (Paper 3's own metric) | −0.128 | 0.590 | −0.111 | 0.761 |

**Gate verdict: no significant signal.** Per the pre-committed interpretation rule above, this is
**not a falsification** (n=13, correlational, and C1 is explicitly a gate, not Prediction 1/2
itself). **Proceed as preregistered** — Stage 1 is not blocked. The restricted-range observation
below is recorded as context for how Stage 3's FM-arm results should be read, not as a condition on
proceeding now.

**Observation, not a formal test (no p-value computed for this — logged as a descriptive note, not
a confirmatory result per Research Constitution Rule 4):** the composite, spectral-decay slope, and
condition number all increase monotonically with rung across the three primary rungs (composite:
~3.0 at runA_grl → ~3.4 at runB_orth1 → ~3.8 at runB; visible directly in
`stage0_geometry_auroc_merged.csv`) — i.e., the intervention *does* visibly reshape both the new
metrics and the old one, consistent in direction with Paper 3's own condition-number dose-response
(τ=0.84 vs. rung, `e1_kendall_tau.csv`). What does *not* move is AUROC itself: it stays in a narrow
band (0.365–0.441, entirely below chance) across all 13 checkpoints — Paper 3's own headline
finding. **A plausible, non-hypothesis-editing explanation for the null C1 result: the outcome
variable (AUROC) has very little variance left for any geometry metric to explain, because Paper
3's ladder already drove it to a near-flat floor at every intervention strength tested** — a
restricted-range problem on the dependent variable, not necessarily evidence that geometry and
accessibility are unrelated. This observation is a candidate direction for Stage 3 (where the FM
arms may produce a wider AUROC range to test against), not a reason to revise Predictions 1–2 now.

No bug found on inspection (LID values ~8.6–9.1 for d=16 are in a plausible range; spectral slopes
and the composite move smoothly and monotonically with dose, matching condition number's known
direction; the AUROC join is exact-match, 13/13 rows, values match `e2_auroc.csv` verbatim). Per the
24-Hour Rule (`25_Research_Constitution.md`): no hypothesis, protocol, or claim edit follows from
this result. Logged in `../26_Lab_Notebook.md`, 2026-08-04.

*Wording note (2026-08-05, per Vinh):* this section originally called the outcome the "proceed with
caution" branch. Corrected to "proceed as preregistered" — the protocol's own interpretation rule
(above) never had a caution state, only proceed/does-not-block; the original wording read more
tentative than the decision actually was. `26_Lab_Notebook.md`'s 2026-08-04 entry is unedited (its
own append-only rule) — this note and a new 2026-08-05 entry there record the correction instead.

An exploratory scatter plot (composite geometry metric vs. Mahalanobis-AUROC, labeled by
checkpoint/rung — visualization only, no inferential claim) was added per Vinh's request:
`outputs/figure_stage0_composite_vs_auroc.png` / `.pdf`.
