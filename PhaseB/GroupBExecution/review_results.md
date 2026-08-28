# Group B Sanity Review

Not a rerun. Not a reproduction. Not interpretation. Checks whether any number is impossible or
internally inconsistent, before anything moves toward interpretation. Every answer below was
verified by reading the four output JSON files directly, not eyeballed.

---

**Mahalanobis AUROC within expected range?**
YES
Reason: AUROC = 0.9984, FPR@95 = 0.0081. Both lie within the valid range for these metrics ([0,1]),
are finite (no NaN/Inf), and are not exactly at a degenerate boundary (0.5 = no discrimination, or
exactly 0.0/1.0) that would itself suggest a scoring or labeling bug.

-----

**kNN monotonic?**
YES
Reason: AUROC is non-increasing as k grows — k=1: 0.9988, k=10: 0.9979, k=50: 0.9967. FPR@95 is
non-decreasing in the same direction — k=1: 0.0057, k=10: 0.0111, k=50: 0.0180. Both directions
verified programmatically across all three k-values, not just the two endpoints.

-----

**Cosine lower than Mahalanobis?**
YES
Reason: Cosine-to-centroid AUROC = 0.9805 < Mahalanobis AUROC = 0.9984. Cosine-to-centroid uses the
identical per-class means as Mahalanobis with the covariance/precision term dropped — a strictly
simpler scorer with less information available to it. A lower AUROC than the richer scorer it was
derived from is structurally unsurprising; this is not a comparison against any external benchmark
or prior paper.

-----

**No impossible number?**
YES
Reason: All six reported values (Mahalanobis, Cosine, kNN×3, KDE — AUROC and FPR@95 each) verified
in-range [0,1], finite, non-NaN. Sample counts cross-checked and consistent across all four output
files: fit population n=16,211 in every file; evaluation population n_id=5,067 / n_ood=2,298 in
every file that reports it.

---

**Conclusion of this review: no anomaly found that would block moving toward interpretation.** This
is not itself a judgment about whether the results are scientifically interesting — that remains
Tier 3, out of scope until Stage 5 has both arms to compare, per the Research Constitution.
