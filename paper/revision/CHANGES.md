# Revision of the advisor's 2026-08-29 draft

Working copy of `Paper4_Independent_VenueNeutral.tex`. The advisor's original is
untouched in `~/Downloads/Paper4_Independent_3RQ_Ref1Free_Source/`.

## Corrected numbers

**Table 3 (A/B/C AUROC), KDE row.** Was `.7611 / .8713 / .7657`, now
`.9967 / .9986 / .9968`. The old values came from sklearn's `KernelDensity`,
which does not compute reliably at d=1024: its `kd_tree` and `ball_tree` paths
disagree with each other on the same data, and both pin near the theoretical
ceiling for queries far from any training point. The direct Gaussian form used
now agrees with sklearn exactly in one dimension, on synthetic 1024-dimensional
data, and on queries coinciding with a training point. Cosine and k-NN move in
the fourth decimal from the same recomputation; Mahalanobis is unchanged.

**Table 4 (headroom).** Follows from the above. KDE moves from `Material`
headroom to `Near ceiling`, which makes the near-ceiling finding uniform across
all four scorers rather than three.

## Claims that are no longer true

- *"RQ3 is therefore inferentially unresolved"* — answered, see below.
- *"local susceptibility curves cannot be reconstructed from aggregate AUROC
  means"* — they were reconstructed, from the frozen embedding artifact and the
  retained adapters, reproducing every stored aggregate to six decimals.
- *"The highest-priority extension is a locked same-modality external domain"* —
  done: Derm7pt (both arms) and PH2.

## New material

Two tables and roughly two pages of results:

- **Table 5, susceptibility.** The bound-to-observation comparison the draft
  says is unavailable. Two readings: Corollary 1 is valid only at delta_max and
  is uninformative there (S = 1.000 for cosine and KDE); at a common descriptive
  delta the scorers separate by three orders of magnitude, Mahalanobis at
  6.5e-5 against cosine's 4.8e-2. Mahalanobis's flatness is ranking
  insensitivity, not a ceiling.
- **Table 6, external validation.** Three locked pairs under a protocol fixed
  before any external image was embedded.

## The framing

The results split into two questions that a single flat curve conflates.

*Does changing geometry change ranking?* Yes. C-A holds the adapter dose fixed
and varies only tempering; it is positive and excludes zero for cosine on all
three external pairs, with the clinical pair within 2e-4 of the calibration
estimate.

*Does intervention identity matter once geometry is fixed?* No. Zero of twelve
external pair-scorer combinations exclude zero, every negative control above
p=.2.

Together: the intervention acts through the measured geometry, not beyond it.
That is an affirmative answer to the title question, and it needs both halves —
had the first been null too, the finding would have been that nothing moves.

Three controls exclude a pipeline explanation for the second half: recomputed
geometry matches stored values to 5e-5 relative and AUROCs exactly; the same
code reproduces the calibration result to 0.000e0 on the estimate, both bounds
and the pair count; and subsampling PAD-UFES to the external sample size leaves
the effect excluding zero in 100% of forty draws.

## Left for the advisor to decide

**Two numeric issues in the retained analyses, both real, neither patched here
because both change published numbers.**

`estimate` in `hierarchical_auroc_ci.csv` is the mean of the bootstrap
replicates, not the point estimate, so it carries Monte Carlo noise of about
2.4e-4 — 9.9% of the cosine C-A value and larger than the k-NN one. Reporting
the point estimate and using the bootstrap only for the interval would fix it.

The within-class scatter of the raw embeddings is numerically rank 1023: the
class-centred data's smallest singular value is 3.95e-7 against 5.47e-3 for the
next. Since `spectral_decay_slope` fits log(lambda) against index over the full
range, that eigenvalue contributes a point some 21 log units below its
neighbour and inflates the alpha=0 slope by 2.2%, against 0.3% at alpha>0. The
manuscript discusses slope differences of about 0.1%. Phase 3 skips alpha=0,
which costs nothing since every downstream analysis drops it; the retained
alpha=0 rows carry the inflation.

**Three smaller discrepancies noted earlier and not applied**: cosine C-A .0018
is a difference of rounded values and rounds to .0019 from the full ones; the
Delta-LID range "+2.7 to +5.8" omits gamma=.75, which falls to +0.26; and the
configuration-C figures are seed 42's, where the five-seed means are 6.731e-3,
6.744e-3 and 0.26.
