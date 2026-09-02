# Paper 4 — geometry and distance-based OOD ranking

Code and analysis artifacts for a study of when representation geometry
predicts distance-based out-of-distribution ranking, on a frozen dermatology
foundation model.

The question is practical. A deployed skin-lesion classifier meets images it
should not be trusted on, and the usual safeguard is a distance-based OOD
scorer operating on the frozen representation. But representations get
modified — for nuisance removal, shortcut suppression, domain adaptation — and
those scorers measure distance *in the space being modified*. Which of them
survive that, and how would you tell?

The awkward part is that a scorer that is genuinely invariant and a scorer that
simply cannot respond produce the same flat AUROC curve. Separating those is
what most of this repository does.

## Layout

    PhaseB/analysis/     analysis scripts, one concern each
    PhaseB/assets/       frozen embedding artifacts and checkpoint manifests
    PhaseB/outputs/      results, one directory per analysis stage
    PhaseB/phase3/       locked protocol for the external-domain phase
    paper/revision/      manuscript and a log of what changed and why
    24_PreRegistration.md    protocol frozen 2026-08-04, before any experiment

## Reproducing

The pretrained backbone is not vendored; it is pinned by SHA-256 in
`PROVENANCE.tsv` against the release published by its authors. Image-level
score vectors were not retained during the original runs but regenerate from
the frozen embedding artifact and the retained adapters:

    python3 PhaseB/analysis/generate_per_example_scores.py

That script cross-checks every regenerated aggregate against the stored one and
stops on a mismatch, so a silent divergence cannot propagate.

## Two things worth knowing before reading the outputs

**Fit population.** Mahalanobis is fit on the 16,211 adapted *training*
embeddings and scores the evaluation set. Fitting and scoring on the same
evaluation set yields AUROC near 0.99997, flat in the dose parameter, and
raises no error. It is the easiest way to get plausible numbers that mean
nothing.

**Correction families.** A whole-seed permutation over five seeds has a
smallest attainable p of 1/120. Holm's first threshold at m tests is 0.05/m.
For m > 6 the floor lies above the threshold and no permutation p-value can
clear it, whatever the data show. Two families in the original analysis were in
that regime; they are now reported descriptively, with their null
distributions.

## Provenance

Analysis history predates version control for this project — `PROVENANCE.tsv`
records file hashes and modification times captured before `git init`. One
claim in an earlier draft could not be reproduced from any retained
configuration set; that is documented in `paper/revision/CHANGES.md` rather
than quietly corrected.

## License

MIT, see `LICENSE`. The datasets are not redistributed here and carry their own
terms.
