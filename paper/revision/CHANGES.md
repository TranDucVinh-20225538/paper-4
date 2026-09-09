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

## Added after re-reading the advisor's README

`README_VenueNeutral.txt` states that "paired non-inferiority remains future
work". It is not future work any more — the two arms run at matched doses and
matched seeds, so the twenty pairs give a paired answer directly: mean utility
difference $-0.0000$, seed-clustered bootstrap interval $[-.0025,+.0029]$,
one-sided lower bound $-.0022$. RQ2 now reports it, and the Gate 0 paragraph
says plainly why the threshold alone was too weak to carry the word
"preservation": it asks for .225 against a chance rate of .125, so a model
falling from .75 to .30 would clear it.

The README also says the theoretical bounds are "not presented as numerically
validated from aggregate AUROC values" and that such validation "requires
retained image-level score vectors and covariance spectra". Those vectors were
regenerated rather than retained, which is what Table 5 rests on.

The choice of margin is left open on purpose. Which utility loss is clinically
tolerable is a prior decision and the data cannot supply it.

## Second pass — the rest of what could be done without a decision

**Abstract.** Rewritten. It claimed "no paired contrast survives correction" and
that cosine and KDE are "descriptively closer to intervention-history-matched
than slope-matched configurations". Neither is the finding any more.

**Whole-seed permutation.** The draft described it for condition number only and
read it as unresolved. It was run symmetrically on all four geometry metrics:
across the sixteen associations the smallest attainable p is .050 against a
design resolution of 1/120 = .0083, so none reaches it. The permutation null is
the more informative object and is now reported — its 95th percentile of |tau|
runs .14 to .66, and a design where |tau| = .66 arises by chance once in twenty
draws cannot adjudicate the observed coefficients.

**Hierarchical intervals.** The draft said aggregate means are not treated as
image-level observations, correctly, and left it there. The vectors were not
retained but are regenerable, and the regeneration matches every stored
aggregate to six decimals, so the intervals are computed at the image level.

**Data and Code Availability** and **Protocol Registration** sections added,
with a table of the three registered external-phase predictions and their
outcomes. The registration section says plainly what each record is worth: the
2026-08-04 pre-registration is an internal dated record with no platform
standing, while the external-phase protocol was committed before any external
image was encoded and that ordering is verifiable in the repository history
rather than asserted.

## The mixed-effects claim: traced, and replaced

The draft reported three Holm-significant interactions -- LID x Mahalanobis at
p = 2.9e-6, LID x KDE at 2.6e-3, condition number x Mahalanobis at 3.9e-5.

**Where it came from.** Not the advisor. The identical sentence is in
`paper/sections/results.tex`, dated 2026-08-17, and the venue-neutral draft
inherited it. That earlier record also carries coefficients, which matters:
-1.2e-4 for LID x Mahalanobis against the reconstruction's -8.9e-5, and 2.9e-4
for condition number x Mahalanobis against 1.6e-4. Same sign, same order. The
convergence claim matches too. The analysis was run; nobody invented numbers.

**What cannot be recovered** is which population or model specification it was
run on. Four retained sets were tried:

| set | n | converged | Holm-significant |
|---|---|---|---|
| draft's claim | 30 | 16/16 | LID x Maha, LID x KDE, cond x Maha |
| matched_input | 45 | 16/16 | LID x Maha |
| matched_input, C2 only | 25 | 16/16 | cond x Maha |
| raw_run_matrix, C2 only | 30 | 15/16 | slope x cosine, slope x k-NN, slope x KDE |
| raw_run_matrix, all | 55 | 16/16 | slope x k-NN, slope x cosine, LID x Maha |

Two sets match one of the draft's own descriptors each -- n = 30, and 16/16
converged -- and neither returns its triple. No script in the repository
produces those p-values.

**Why it was replaced rather than left.** The conclusion the sentence draws,
that the coefficients are small and mechanism-dependent, survives every
reconstruction, so the science does not turn on this. Two other things do.

The manuscript now carries a Data and Code Availability section. This was the
one claim a reviewer attempting reproduction would fail on, and an
irreproducible number sitting next to a reproducibility statement costs more
than the number is worth.

And the sentence's sub-claim, "spectral slope is not supported in this
stress-test regime", is contradicted twice over: by the reconstructions, where
slope is the interaction that reaches significance on the two sets matching the
draft's descriptors, and independently by the external validation, where C-A
varies only the tempering exponent -- that is, only slope -- and moves ranking
on all four domains. It has been dropped.

**What replaced it.** The reconstruction on the 55-configuration run matrix,
named in the text: slope x k-NN, slope x cosine, LID x Mahalanobis, with
coefficients and Holm p-values. That set was chosen before looking at which
result it gave, on the stated grounds that it is the widest retained set and
the only one where all sixteen models converge.

The instability itself is now reported as the substantive finding at this
level: an association whose identity changes with the analysis population is
not evidence about a mechanism. That is a better reason for the exploratory
label than the draft had.

The provenance gap is stated in Limitations rather than left in the change log.

## Still needs a decision

**Venue.** The README notes the manuscript can be reformatted once a target is
selected.

**Repository path cleanup.** 23,576 rows across two metadata files carry
absolute paths under a local home directory. Harmless in a private repository,
worth removing before the repository is opened. Not done yet because scripts
read the `path` column and a blind rewrite risks breaking them.

## Resolved since the earlier draft of this file

**The two numeric issues are patched.** An earlier version of this document
listed them as outstanding "because both change published numbers", and an
advisor reviewing the repository at that point would have read them as still
open. They are not.

`hierarchical_auroc_ci.py` now reports the point estimate in `estimate` and the
bootstrap mean alongside it in `boot_mean`, so the roughly 2.4e-4 of Monte
Carlo noise is visible rather than silently occupying the estimate column.
`lid_spectral_decay.py` applies a relative numerical rank cutoff (`RANK_RTOL`)
that drops the near-degenerate eigenvalue and reports how many were dropped,
removing the 2.2% inflation of the alpha=0 spectral slope. Effect on reported
numbers: fourth decimal at most, no verdict changes.

**Table 4's configuration-C column is now traceable.** It was computed from a
cached per-example store with no intermediate file recording the per-seed
AUROCs, so the KDE entry in particular could not be checked against anything.
`PhaseB/outputs/rq3/abc_per_seed_auroc.csv` now holds all fifteen
configuration-seed values, and averaging it reproduces Table 4 including the
KDE column at .9968.

**The dropped research line is disclosed in the manuscript.** See below.

## The registered hypothesis is not the one the paper answers

This is the largest gap an outside reader of the repository would find, and it
was not addressed in the manuscript until now.

`24_PreRegistration.md` locks a primary hypothesis about mediation: that the
dissociation between probe-decodability and distance-based estimator
performance is a general property of representation-reshaping interventions,
mediated by representation geometry. That hypothesis was tested. Ten
configurations, probe AUROC .99999 to 1.00000 against Mahalanobis .9983 to
.9984, paired difference significant in nine of ten at p between 2.5e-5 and
8.4e-8, retained in `stage5_c5_raw.json`. An earlier draft, `results.tex` dated
2026-08-17, reported it in full.

The submitted manuscript contains no occurrence of "probe", "GRL", or
"dissociation". The reason is in `19_Reviewer3_Logic_Attack.md`, an internal
adversarial review of the argument's logic: for Mahalanobis, the covariance
structure and the class-mean separation are not separable inputs whose
contributions can be apportioned, since both are parameters of
(z-mu)^T Sigma^-1 (z-mu). Asking which one explains the estimator's behaviour
presupposes a decomposition the estimator does not make.

That is a defensible reason to change direction, and the invariance
formalization is a direct answer to it. But a reader holding only the PDF could
not know that a different hypothesis had been registered, tested, and set
aside, or why. The Protocol Registration section now says so, states that the
dissociation measurement is not withdrawn — only its causal-mediation reading
is — and notes that the supporting columns remain in the released run matrix so
the branch is recoverable rather than deleted.
