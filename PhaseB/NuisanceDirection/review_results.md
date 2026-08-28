# Nuisance Direction — Sanity Review

Not a rerun. Not interpretation. Checks whether the computed direction is structurally sane —
non-degenerate, correctly normalized, and plausibly domain-related — before it becomes a fixed
regularization target for anything. Every answer verified by reading the actual output files and
recomputing, not eyeballed.

---

**Is `w` non-degenerate (not near-zero, not NaN/Inf)?**
YES
Reason: `‖w_raw‖ = 9.014975` before normalization — finite, well away from zero. For scale context,
`ArtifactCharacterization/characterization_report.json` records a typical individual embedding norm
of ≈23.08 in this same space; `w_raw`'s norm is a substantial fraction of that, not a numerical
artifact of near-identical class means.

-----

**Is `w` correctly unit-normalized?**
YES
Reason: `‖w‖ = 1.000000`, verified directly from the saved array, not assumed from the formula.

-----

**Does `w` plausibly capture domain-related structure, checked by projection?**
YES
Reason: projected `ReferenceEmbedding`'s own held-out ISIC-test and PAD-UFES rows onto `w`
(`z · w`) — ISIC-test: mean 1.294, std 1.310, n=5,067; PAD-UFES: mean −7.800, std 2.014, n=2,298.
Clear separation (Δmean ≈ 9.09) relative to each group's own spread (std 1.31 / 2.01) — consistent
with `w` pointing from the PAD-UFES centroid toward the ISIC centroid, as its own formula defines.
**Caveat, stated plainly**: the PAD-UFES side of this check is not fully independent — the same
PAD-UFES rows contributed to `μ_pad_ufes` (`review.md`'s Known Limitations). The ISIC-test side *is*
independent (fit on `ReferenceTrainEmbedding`, checked against disjoint `ReferenceEmbedding`
ISIC-test rows), and still shows a clearly non-zero, correctly-signed projection.

-----

**No impossible number?**
YES
Reason: `w`, `w_raw`, `mu_isic`, `mu_pad_ufes` all finite, all shape (1024,), matching the frozen
artifacts' own embedding dimensionality. Population sizes (16,211 / 2,298) match both source
artifacts' own manifests exactly.

---

**Conclusion of this review: no anomaly found.** This is not a claim that `w` is "the right"
direction in any scientific sense — only that the computation is internally consistent and the
resulting direction is not degenerate. Whether it functions as an effective adapter regularization
target is an empirical question for Gate 1, not this review.
