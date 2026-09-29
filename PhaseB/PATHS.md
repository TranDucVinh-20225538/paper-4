# Paths that outlive the machine

Nineteen scripts used to open with a literal path under one author's home
directory, so on any other machine each one failed on its first line for a
reason that was never the one being debugged. They now resolve every external
location through `analysis/data_roots.py`.

    python3 PhaseB/analysis/data_roots.py

prints where each root resolves and whether it exists. Override any of them with
the environment variable the module names, or edit `data_roots.json`, which
records the layout the study was run on so that machine keeps working untouched.

Recorded paths in files written from now on use a `{root}` placeholder --
`{archive}/NV/ISIC_0055906.jpg`, `{repo}/PhaseB/outputs` -- which
`data_roots.resolve()` expands. A reader with their own copy of a corpus can find
the file; a reader without one still learns which file a row came from.

## Two roots no longer exist

`archive`, the working copy of the ISIC crops, and the image directories under
`csg` were deleted from the machine this study ran on. The embedding artifacts
under `assets/reference_embeddings/` are what survives of them, hash-pinned. The
consequence is specific: Stage 3.0's artifact audit and the duplicate registry
open images and so cannot be re-run without restoring those corpora from their
distributors, while everything downstream of the embeddings can.

## Why the older records still carry absolute paths

Twenty-four committed files still spell paths absolutely, including both
`metadata.csv` files under `assets/reference_embeddings/`. Every one of them is
hash-pinned in `PROVENANCE.tsv`, and most also in a local `sha256.txt` or
`execution_manifest.json`; five of the directories holding them are mode `r-x`
on purpose. Rewriting those files would mean editing the registry that pins
them, which changes the audit trail rather than the paths in it, and would be
hard for an auditor to tell apart from tampering. They are left as written, and
this file is the note that says why.

The paths in them name one author's home directory. That is a disclosure to
weigh before the repository is made public, not a correctness problem: nothing
reads them, because the corpora they point at are gone.

## On the public branch

`public` carries the same history with every recorded path rewritten to the
`{root}` form and every hash record that covered a rewritten file updated to
match: 25 files, 54 records, propagated until the registry stopped changing.
Artifact entries verify clean afterwards.

The analysis scripts are a separate matter. `PROVENANCE.tsv` is a snapshot taken
on 2026-08-28, and the analysis code has been edited continuously since, so its
entries for `PhaseB/analysis/*.py` were already stale on `main` before this
branch existed -- 18 of 79 -- for the ordinary reason that code changed and a
dated snapshot did not. It pins artifacts, which are frozen, and it does not
pin code, which is not. Read it that way.
